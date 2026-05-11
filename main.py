"""
SDLC entry point.

Orchestrates the AutoGen hub-and-spoke multi-agent workflow:

    UserProxy  →  ProjectManager  ↔  Engineer
                                  ↔  CodeReviewer
                                  ↔  QA

The ProjectManager is the central hub.  It receives the user idea, delegates
tasks to each specialist via explicit handoffs (transfer_to_* tools), and each
specialist hands control back to the PM when done.  The PM decides next steps
at every stage.

The team uses AutoGen's Swarm pattern: agents transfer control to each other
via HandoffMessage rather than taking turns blindly.

Usage
-----
    uv run main.py "Build a REST API for a todo app"
    uv run main.py "Build a REST API for a todo app" --rounds 10
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
from dotenv import load_dotenv

from autogen_agentchat.conditions import FunctionalTermination, MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import Swarm

from agents import ProjectManager, Engineer, CodeReviewer, QA
from agents.config import ensure_workspace_dirs
from core.mcp_client import MCPClientPool
from core.mcp_config import ROLE_SERVERS
from core.swarm_loop_guard import build_swarm_loop_guard_termination
from core.autogen_compat import patch_single_threaded_runtime_shutdown
from core.swebench import (
    assess_patch_relevance,
    build_task_prompt,
    infer_review_blocking,
    load_task_context,
    parse_qa_verdict,
    parse_review_verdict,
)
from core.telemetry import record_handoff, record_message, reset as reset_telemetry, set_final_status, write_if_configured

load_dotenv()
patch_single_threaded_runtime_shutdown()

# ---------------------------------------------------------------------------
# Logging: mirror all logs to a per-session file under data/logs/
# ---------------------------------------------------------------------------

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Console handler (stderr/stdout)
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

# Per-session file handler
logs_dir = Path(__file__).resolve().parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
run_id = os.environ.get("MAS_RUN_ID")
session_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
log_suffix = run_id or session_ts
log_file = logs_dir / f"session_{log_suffix}.log"

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_format)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info("Session log file: %s", log_file)


def _read_patch_text_from_file() -> str:
    patch_path = os.environ.get("MAS_EVAL_PATCH_PATH", "").strip()
    if not patch_path:
        return ""
    try:
        return Path(patch_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.warning("Could not read patch file %s: %s", patch_path, exc)
        return ""


def _git_diff_text(base_commit: str | None = None) -> str:
    workspace = os.environ.get("MAS_WORKSPACE_PATH")
    if not workspace:
        return ""
    diff_cmd = ["git", "diff"]
    if base_commit:
        diff_cmd = ["git", "diff", base_commit]
    result = subprocess.run(
        diff_cmd,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout.strip():
        return result.stdout
    if base_commit:
        result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    return result.stdout


def _current_patch_text(base_commit: str | None = None) -> str:
    patch_text = _read_patch_text_from_file()
    if patch_text.strip():
        return patch_text
    return _git_diff_text(base_commit)


def _latest_qa_verdict(messages: list[dict[str, str]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("source") != "QA":
            continue
        verdict = parse_qa_verdict(message.get("content", ""))
        if verdict is not None:
            return verdict
    return None


def _latest_review_verdict(messages: list[dict[str, str]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("source") != "CodeReviewer":
            continue
        verdict = parse_review_verdict(message.get("content", ""))
        if verdict is not None:
            return verdict
    return None


def _latest_review_blocking(messages: list[dict[str, str]]) -> bool | None:
    for message in reversed(messages):
        if message.get("source") != "CodeReviewer":
            continue
        return infer_review_blocking(message.get("content", ""))
    return None


def _latest_project_complete_summary(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("source") != "ProjectManager":
            continue
        content = message.get("content", "")
        if content.startswith("PROJECT COMPLETE"):
            return content
    return ""


def _messages_for_completion_check(
    messages: list[BaseAgentEvent | BaseChatMessage],
) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, BaseChatMessage):
            continue
        content = getattr(message, "content", None)
        collected.append(
            {
                "source": str(getattr(message, "source", "?")),
                "content": "" if content is None else str(content),
            }
        )
    return collected


def _swebench_success_state(
    messages: list[dict[str, str]],
    *,
    base_commit: str | None,
) -> dict[str, Any]:
    sources = {message.get("source") for message in messages}
    qa_verdict = _latest_qa_verdict(messages)
    review_blocking = _latest_review_blocking(messages)
    patch_present = bool(_current_patch_text(base_commit).strip())
    project_complete_summary = _latest_project_complete_summary(messages)
    return {
        "sources": sources,
        "qa_verdict": qa_verdict,
        "review_blocking": review_blocking,
        "patch_present": patch_present,
        "project_complete_summary": project_complete_summary,
        "qa_verified_ready": (
            "Engineer" in sources
            and "QA" in sources
            and qa_verdict is not None
            and bool(qa_verdict.get("verified"))
            and patch_present
            and review_blocking is not True
        ),
    }


def _build_swebench_project_complete_summary(messages: list[dict[str, str]]) -> str:
    qa_verdict = _latest_qa_verdict(messages)
    review_verdict = _latest_review_verdict(messages)
    summary_lines = [
        "PROJECT COMPLETE",
        "",
        "QA verified all Fail-to-pass tests and Engineer produced a non-empty patch.",
    ]
    if qa_verdict and qa_verdict.get("notes"):
        summary_lines.append(f"QA notes: {qa_verdict['notes']}")
    if review_verdict and review_verdict.get("notes"):
        summary_lines.append(f"Code review: {review_verdict['notes']}")
    return "\n".join(summary_lines)


def _swebench_completion_reached(
    transcript: list[dict[str, str]],
    *,
    base_commit: str | None,
) -> bool:
    if not transcript:
        return False
    state = _swebench_success_state(transcript, base_commit=base_commit)
    if not state["qa_verified_ready"]:
        return False
    last = transcript[-1]
    if last.get("source") == "ProjectManager":
        return last.get("content", "").startswith("PROJECT COMPLETE")
    return last.get("source") == "QA"


def _swebench_completion_reached_from_delta(
    messages: list[BaseAgentEvent | BaseChatMessage],
    *,
    base_commit: str | None,
) -> bool:
    transcript = _messages_for_completion_check(messages)
    return _swebench_completion_reached(transcript, base_commit=base_commit)


def _build_swebench_termination_checker(*, base_commit: str | None):
    transcript: list[dict[str, str]] = []

    def _check(messages: list[BaseAgentEvent | BaseChatMessage]) -> bool:
        # AutoGen termination conditions receive only the new delta, not the
        # full thread, so we accumulate the transcript across callbacks here.
        transcript.extend(_messages_for_completion_check(messages))
        return _swebench_completion_reached(transcript, base_commit=base_commit)

    return _check


async def start_sdlc(idea: str, rounds: int = 20, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Run the SDLC multi-agent workflow for the given idea.

    Parameters
    ----------
    idea:
        The project idea or user requirement to implement.
    rounds:
        Maximum number of agent messages before the workflow stops.

    Returns a structured record with the final message and transcript metadata.
    """
    ensure_workspace_dirs()

    # Collect the union of all server keys needed by any role.
    all_server_keys: list[str] = list(
        {key for keys in ROLE_SERVERS.values() for key in keys}
    )

    # In Docker mode the workspace is the container's /testbed, not a local git
    # repo.  The git MCP server (mcp-server-git) would crash trying to open an
    # empty directory.  Git operations are handled inside the container via bash.
    if os.environ.get("MINI_AGENT_USE_DOCKER"):
        all_server_keys = [k for k in all_server_keys if k != "git"]

    logger.info("Opening MCP connections: %s", all_server_keys)

    async with MCPClientPool(server_keys=all_server_keys) as pool:
        # --- Instantiate role agents ---
        pm = ProjectManager(pool)
        eng = Engineer(pool)
        reviewer = CodeReviewer(pool)
        qa = QA(pool)

        # --- Termination conditions ---
        # Stop when the PM says the project is complete OR we hit the round cap.
        # Optional monopoly guard: Swarm keeps one speaker until a handoff; see
        # core.swarm_loop_guard for per-agent message limits (env configurable).
        if task is not None and os.environ.get("MAS_MODE") == "swebench":
            swebench_termination = _build_swebench_termination_checker(
                base_commit=str(task.get("base_commit") or "") or None,
            )
            termination: FunctionalTermination | MaxMessageTermination | TextMentionTermination = (
                FunctionalTermination(swebench_termination)
                | MaxMessageTermination(max_messages=rounds)
            )
        else:
            termination = (
                TextMentionTermination("PROJECT COMPLETE")
                | MaxMessageTermination(max_messages=rounds)
            )
        swarm_guard = build_swarm_loop_guard_termination()
        if swarm_guard is not None:
            termination |= swarm_guard

        # --- Hub-and-spoke team (Swarm) ---
        # Swarm routes control via HandoffMessage: PM uses transfer_to_* tools
        # to delegate to specialists; each specialist transfers back to PM.
        # The initial task is delivered to the first participant (PM).
        team = Swarm(
            participants=[
                pm.agent,
                eng.agent,
                reviewer.agent,
                qa.agent,
            ],
            termination_condition=termination,
        )

        logger.info("Starting SDLC for: %s", idea)

        final_message = ""
        transcript: list[dict[str, str]] = []
        stream_error = ""
        try:
            async for message in team.run_stream(task=idea):
                if hasattr(message, "content"):
                    content = str(getattr(message, "content", "") or "")
                    source = str(getattr(message, "source", "?"))
                    record_message(
                        source,
                        content,
                        getattr(message, "models_usage", None),
                    )
                    if "handoff" in type(message).__name__.lower() or hasattr(message, "target"):
                        record_handoff(getattr(message, "source", "?"), getattr(message, "target", None))
                    logger.info(
                        "[%s] %s",
                        source,
                        content[:200],
                    )
                    transcript.append({"source": source, "content": content})
                    final_message = content
        except Exception as exc:  # noqa: BLE001
            stream_error = f"{type(exc).__name__}: {exc}"
            logger.info(
                "SDLC workflow stopped due to %s: %s.",
                type(exc).__name__,
                exc,
            )
        finally:
            logger.info("SDLC workflow finished.")

        if task is not None and os.environ.get("MAS_MODE") == "swebench":
            state = _swebench_success_state(transcript, base_commit=str(task.get("base_commit") or "") or None)
            if state["qa_verified_ready"] and not state["project_complete_summary"]:
                final_message = _build_swebench_project_complete_summary(transcript)
                transcript.append({"source": "ProjectManager", "content": final_message})
                logger.info("[ProjectManager] %s", final_message.splitlines()[0])

        return {
            "final_message": final_message,
            "messages": transcript,
            "error": stream_error,
        }


def _write_patch_if_configured(base_commit: str | None = None) -> str | None:
    patch_path = os.environ.get("MAS_EVAL_PATCH_PATH")
    workspace = os.environ.get("MAS_WORKSPACE_PATH")
    if not patch_path or not workspace:
        return None
    # In Docker mode the Engineer already wrote the patch from inside the
    # container.  Skip the host-side git diff so we don't overwrite it with an
    # empty diff from the non-git workspace directory.
    if os.environ.get("MINI_AGENT_USE_DOCKER"):
        return patch_path if Path(patch_path).exists() else None
    diff_cmd = ["git", "diff"]
    if base_commit:
        diff_cmd = ["git", "diff", base_commit]
    diff = subprocess.run(
        diff_cmd,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if not diff:
        diff = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    Path(patch_path).write_text(diff, encoding="utf-8")
    return patch_path


def _write_result(result: dict[str, Any]) -> None:
    result_path = os.environ.get("MAS_EVAL_RESULT_PATH")
    if not result_path:
        return
    Path(result_path).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def _evaluate_swebench_run(
    run_result: dict[str, Any],
    task: dict[str, Any],
    patch_path: str | None,
) -> dict[str, Any]:
    messages = list(run_result.get("messages") or [])
    final_message = str(run_result.get("final_message") or "")
    project_complete_summary = _latest_project_complete_summary(messages)
    if project_complete_summary:
        final_message = project_complete_summary
    qa_verdict = _latest_qa_verdict(messages)
    review_verdict = _latest_review_verdict(messages)
    review_blocking = _latest_review_blocking(messages)
    patch_text = _current_patch_text(task.get("base_commit"))
    patch_present = bool(patch_text.strip())
    patch_relevance = (
        assess_patch_relevance(task, patch_text)
        if patch_present
        else {
            "acceptable": False,
            "changed_files": [],
            "relevant_files": [],
            "suspicious_files": [],
            "notes": ["Patch diff is empty."],
        }
    )

    sources = {message.get("source") for message in messages}
    completion_failures: list[str] = []
    if "Engineer" not in sources:
        completion_failures.append("Engineer never handed control back to the ProjectManager.")
    if "QA" not in sources:
        completion_failures.append("QA never handed control back to the ProjectManager.")
    if not project_complete_summary and not final_message.startswith("PROJECT COMPLETE"):
        completion_failures.append("ProjectManager never emitted a valid PROJECT COMPLETE summary.")
    if qa_verdict is None:
        completion_failures.append("No QA_VERDICT block was produced.")
    elif not qa_verdict.get("verified"):
        completion_failures.append("QA_VERDICT did not verify all Fail-to-pass tests.")
    if not patch_present:
        completion_failures.append("patch.diff is empty or missing content.")
    if review_blocking is True:
        completion_failures.append("CodeReviewer reported blocking findings.")
    if patch_present and not patch_relevance.get("acceptable", False):
        completion_failures.append("Patch relevance sanity check flagged the diff as suspicious.")

    completion_valid = not completion_failures
    telemetry_status = "success" if completion_valid else "failed_verification"
    if not completion_valid:
        if "Engineer" not in sources or "QA" not in sources:
            telemetry_status = "incomplete_handoffs"
        elif not patch_present:
            telemetry_status = "missing_patch_content"
        elif qa_verdict is None or not qa_verdict.get("verified"):
            telemetry_status = "qa_not_verified"
        elif review_blocking is True:
            telemetry_status = "review_blocked"
        elif patch_present and not patch_relevance.get("acceptable", False):
            telemetry_status = "patch_relevance_failed"

    return {
        "status": "ok" if completion_valid else "incomplete",
        "final_message": final_message,
        "patch_path": patch_path,
        "completion_valid": completion_valid,
        "completion_failures": completion_failures,
        "qa_verdict": qa_verdict,
        "review_verdict": review_verdict,
        "review_blocking": review_blocking,
        "patch_present": patch_present,
        "patch_relevance": patch_relevance,
        "handoff_sources": sorted(source for source in sources if source),
        "workflow_error": run_result.get("error", ""),
        "telemetry_status": telemetry_status,
    }


def run_swebench(task_path: str, rounds: int = 100) -> None:
    os.environ["MAS_MODE"] = "swebench"
    os.environ.setdefault("MAS_EVAL_TASK_CONTEXT_PATH", task_path)
    reset_telemetry()
    task = load_task_context(task_path)
    run_result = asyncio.run(start_sdlc(build_task_prompt(task), rounds=rounds, task=task))
    patch_path = _write_patch_if_configured(task.get("base_commit"))
    result = _evaluate_swebench_run(run_result, task, patch_path)
    set_final_status(result["telemetry_status"])
    _write_result(result)
    write_if_configured()
    print("\n=== SWE-bench result ===")
    print(result["final_message"])


def main(idea: str, rounds: int = 100) -> None:
    """
    CLI entry point.

    Parameters`
    ----------
    idea:
        The project idea or user requirement to implement.
    rounds:
        Maximum number of agent messages (default: 100).
    """
    result = asyncio.run(start_sdlc(idea, rounds=rounds))
    print("\n=== Final result ===")
    print(result["final_message"])


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "run_swebench": run_swebench,
        }
    )
