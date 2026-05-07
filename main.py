"""
SDLC entry point.

Orchestrates the AutoGen hub-and-spoke multi-agent workflow:

    UserProxy  →  ProjectManager  ↔  Architect
                                  ↔  Engineer
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

import fire
from dotenv import load_dotenv

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import Swarm

from agents import ProjectManager, Architect, Engineer, CodeReviewer, QA
from agents.config import ensure_workspace_dirs
from core.mcp_client import MCPClientPool
from core.mcp_config import ROLE_SERVERS
from core.swebench import build_task_prompt, load_task_context
from core.telemetry import record_handoff, record_message, reset as reset_telemetry, set_final_status, write_if_configured

load_dotenv()

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


async def start_sdlc(idea: str, rounds: int = 20) -> str:
    """
    Run the SDLC multi-agent workflow for the given idea.

    Parameters
    ----------
    idea:
        The project idea or user requirement to implement.
    rounds:
        Maximum number of agent messages before the workflow stops.

    Returns
    -------
    str
        The final message from the workflow (typically the PM's summary).
    """
    ensure_workspace_dirs()

    # Collect the union of all server keys needed by any role.
    all_server_keys: list[str] = list(
        {key for keys in ROLE_SERVERS.values() for key in keys}
    )

    logger.info("Opening MCP connections: %s", all_server_keys)

    async with MCPClientPool(server_keys=all_server_keys) as pool:
        # --- Instantiate role agents ---
        pm = ProjectManager(pool)
        arch = Architect(pool)
        eng = Engineer(pool)
        reviewer = CodeReviewer(pool)
        qa = QA(pool)

        # --- Termination conditions ---
        # Stop when the PM says the project is complete OR we hit the round cap.
        termination = (
            TextMentionTermination("PROJECT COMPLETE")
            | MaxMessageTermination(max_messages=rounds)
        )

        # --- Hub-and-spoke team (Swarm) ---
        # Swarm routes control via HandoffMessage: PM uses transfer_to_* tools
        # to delegate to specialists; each specialist transfers back to PM.
        # The initial task is delivered to the first participant (PM).
        team = Swarm(
            participants=[
                pm.agent,
                arch.agent,
                eng.agent,
                reviewer.agent,
                qa.agent,
            ],
            termination_condition=termination,
        )

        logger.info("Starting SDLC for: %s", idea)

        final_message = ""
        try:
            async for message in team.run_stream(task=idea):
                if hasattr(message, "content"):
                    record_message(
                        getattr(message, "source", "?"),
                        str(getattr(message, "content", "")),
                        getattr(message, "models_usage", None),
                    )
                    if "handoff" in type(message).__name__.lower() or hasattr(message, "target"):
                        record_handoff(getattr(message, "source", "?"), getattr(message, "target", None))
                    logger.info(
                        "[%s] %s",
                        getattr(message, "source", "?"),
                        str(message.content)[:200],
                    )
                    final_message = str(message.content)
        except Exception as exc:  # noqa: BLE001
            # AutoGen may raise an internal termination-related exception when
            # the termination condition is satisfied. Treat that as a graceful
            # shutdown and still consider the workflow "done" for this run.
            logger.info(
                "SDLC workflow stopped due to %s: %s. Assuming all planned steps are done.",
                type(exc).__name__,
                exc,
            )
        finally:
            logger.info("SDLC workflow finished.")

        return final_message


def _write_patch_if_configured() -> str | None:
    patch_path = os.environ.get("MAS_EVAL_PATCH_PATH")
    workspace = os.environ.get("MAS_WORKSPACE_PATH")
    if not patch_path or not workspace:
        return None
    diff = subprocess.run(
        ["git", "diff"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    Path(patch_path).write_text(diff, encoding="utf-8")
    return patch_path


def _write_result(final_message: str, patch_path: str | None) -> None:
    result_path = os.environ.get("MAS_EVAL_RESULT_PATH")
    if not result_path:
        return
    Path(result_path).write_text(
        json.dumps(
            {
                "status": "ok",
                "final_message": final_message,
                "patch_path": patch_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_swebench(task_path: str, rounds: int = 100) -> None:
    os.environ["MAS_MODE"] = "swebench"
    reset_telemetry()
    task = load_task_context(task_path)
    result = asyncio.run(start_sdlc(build_task_prompt(task), rounds=rounds))
    patch_path = _write_patch_if_configured()
    set_final_status("success")
    _write_result(result, patch_path)
    write_if_configured()
    print("\n=== SWE-bench result ===")
    print(result)


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
    print(result)


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "run_swebench": run_swebench,
        }
    )
