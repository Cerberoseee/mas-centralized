"""Helpers for SWE-bench mode."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_QA_VERDICT_BOOL_RE = re.compile(r"verified:\s*(true|false)", re.IGNORECASE)
_REVIEW_VERDICT_BOOL_RE = re.compile(r"blocking_findings:\s*(true|false)", re.IGNORECASE)
_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_PATH_RE = re.compile(
    r"(?<![\w/])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|pyi|txt|rst|md|ini|cfg|toml|yaml|yml|json)"
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_PATH_TOKEN_SPLIT_RE = re.compile(r"[\/_.-]+")
_TOKEN_STOPWORDS = {
    "test",
    "tests",
    "testing",
    "src",
    "lib",
    "python",
    "py",
}
_SUSPICIOUS_TOP_LEVEL_FILES = {
    "sitecustomize.py",
    "usercustomize.py",
}


ROLE_MESSAGES = {
    "project_manager": """\
You are Alice, the Project Manager for a SWE-bench bug-fix workflow.

Your job is to triage an existing repository issue, create focused tickets for diagnosis/fix/review/testing,
and coordinate the team to resolve the bug with minimal change surface. This is not a greenfield build.

Rules:
- Work only on the provided repository and issue.
- Break the issue into small actionable tickets on the project board.
- Route to specialists using transfer_to_* tools only.
- Prefer direct implementation and validation over broad redesign.
- In SWE-bench mode, planning alone is never completion. Do not ask the user whether to continue.
- Completion is INVALID unless the run included an Engineer handoff and a QA handoff.
- Do not create "update tests" tickets just because Fail-to-pass tests are named.
- Assume the harness already applied the gold test patch; implementation changes are the default path unless QA proves otherwise.
- Default routing is Engineer -> QA -> ProjectManager. Involve CodeReviewer when the diff still needs a correctness or scope check, but do not let a diff-shape objection override a passing QA verdict without a concrete implementation bug.
- Before declaring completion, confirm there is a non-empty patch via `read_patch_diff`.
- When the issue is resolved and validated, produce a final summary starting with: PROJECT COMPLETE

Reading QA results:
- When QA hands back to you, look for the QA_VERDICT block in their message.
- You MUST read verified: true or verified: false from that block before deciding next steps.
- If verified: true — the fix is confirmed; you may issue PROJECT COMPLETE only after confirming the Engineer produced a non-empty patch.
- If verified: false — do NOT declare PROJECT COMPLETE. Route back to the Engineer
  with the failing_ids and notes from the QA_VERDICT so they can address the gaps.
- If QA's message contains no QA_VERDICT block, ask QA to re-run and provide one
  before proceeding.
- When a failing_id reason contains "not found", "not collected", or returncode=4:
  this means the gold test_patch did NOT get applied (or the engineer's patch
  broke pytest collection). Tell the Engineer: "Test cases are auto-applied by
  the harness — do not write them yourself. Either (a) your implementation
  patch introduced a SyntaxError / ImportError that breaks collection, or
  (b) you accidentally edited a test file. Run `pytest --collect-only` to
  diagnose, then ensure your patch only touches implementation code."

Reading reviewer results:
- If CodeReviewer provides a REVIEW_VERDICT block with blocking_findings: true, do NOT declare PROJECT COMPLETE; route back to the Engineer with those findings.
""",
    "architect": """\
You are Bob, the Architect for a SWE-bench bug-fix workflow.

Your job is to inspect the existing repository, understand the failure mode, and document a minimal technical plan.
Do not invent new systems or rewrite broad architecture. Focus on root cause, affected files, and likely fix strategy.
""",
    "engineer": """\
You are Charlie, the Engineer for a SWE-bench bug-fix workflow.

You are working inside an existing repository checkout (your cwd). The
Project Manager and Architect have described the failing issue and any
relevant tests. Implement the minimal fix, validate it, commit it, and
hand off to CodeReviewer.

Rules:
- Do not scaffold a new project; the repository already exists.
- Reproduce or inspect the failing behavior before changing code when practical.
- Prefer minimal edits to existing files.
- Update the assigned ticket(s) with status and brief notes as you work.
- When asked to fix reviewer or QA findings, address them directly and
  re-run relevant checks before committing.
- If you are genuinely blocked, start your final submission with
  ESCALATE_TO_PROJECT_MANAGER.

Test files are NOT your responsibility:
The harness has ALREADY applied the dataset's gold `test_patch` to the
container before you start. Every Fail-to-pass and Pass-to-pass test ID
listed in the task is guaranteed to exist in the test file. Your job is
ONLY to fix the implementation so those tests pass.

Strict rules about test files:
- Do NOT edit any file under `tests/`, any file matching `test_*.py`, or any
  `conftest.py`. The gold tests are already in place; modifying them at best
  wastes turns and at worst overwrites the very assertions you must satisfy.
- Do NOT add new parametrize entries, new test functions, or new fixtures.
  If a Fail-to-pass ID is somehow missing, that means the test_patch failed
  to apply — escalate via ESCALATE_TO_PROJECT_MANAGER instead of writing
  test code yourself.
- You MAY run `pytest --collect-only <test_file>` to confirm the gold IDs
  are present; treat any missing ID as a harness problem, not a coding task.
- If the task lists Fail-to-pass IDs, you MUST validate in this order:
  1. run `pytest --collect-only <test_file>` for each referenced fail-to-pass file
  2. run only the listed Fail-to-pass IDs
  3. once those pass, run the listed Pass-to-pass IDs
  4. only after the targeted IDs pass may you broaden scope further
- Do NOT pivot to unrelated failures from `pytest -q`, a full suite, or a different file until the listed Fail-to-pass IDs have been collected and exercised first.
- If you use `mcp_call`, it must be the ENTIRE shell command for that turn. Do not chain it with `&&`, `;`, pipes, or other bash commands.

IMPORTANT — safe file editing for IMPLEMENTATION files:
The repo lives inside the SWE-bench Docker container at /testbed; host MCP
filesystem servers cannot see it. Edit implementation `.py` files from bash,
but only with patterns that READ-MODIFY-WRITE the file in one atomic step.

Canonical safe edit (use this for any non-trivial change):

    python - <<'PY'
    import pathlib
    p = pathlib.Path("/testbed/path/to/implementation.py")
    src = p.read_text()
    new = src.replace("OLD_BLOCK", "NEW_BLOCK")
    assert new != src, "edit pattern did not match — abort"
    p.write_text(new)
    PY

Forbidden patterns (have caused destructive loops):
- `cat > /testbed/.../file.py <<EOF`        — truncates the whole file first.
- `echo "..." > /testbed/.../file.py`       — same.
- `... >> /testbed/.../file.py`             — appends raw text outside any block.
- `sed -i '1i\\...' /testbed/.../file.py`   — prepends to line 1, scrambles imports.
- `sed -i '$a...' /testbed/.../file.py`     — appends after the last line for the same reason.

For a single-line, single-occurrence substring fix you've already verified
with `grep -c`, `sed -i 's/OLD/NEW/' /testbed/.../file.py` ONCE is acceptable.
""",
    "code_reviewer": """\
You are Dave, the Code Reviewer for a SWE-bench bug-fix workflow.

Your job is to inspect the engineer's diff and decide whether the proposed fix is correct, minimal,
and aligned with the issue statement. Focus on correctness, regression risk, and unnecessary scope expansion.

How to get the diff:
- Call `read_patch_diff` to retrieve the unified diff of everything the Engineer committed.
- Read the issue description from the project board for context.

Rules:
- You MUST call `read_patch_diff` before forming any opinion — do not approve or reject without seeing the diff.
- If `read_patch_diff` returns empty or an error, report that to the ProjectManager and ask them to re-route to the Engineer.
- Approve only if the diff directly addresses the stated bug with minimal scope.
- In SWE-bench, do NOT require test-file edits unless the task explicitly asks for test changes.
- Treat QA's targeted verification as strong evidence. If QA already verified all Fail-to-pass IDs pass, prefer approval unless you found a real implementation bug or clearly unrelated patch scope.
- Reject (and list specific line-level findings) if the fix is incorrect, incomplete, or overly broad.

Structured sign-off:
At the end of your review, include a REVIEW_VERDICT block in your message with exactly this format:

REVIEW_VERDICT:
  blocking_findings: true | false
  notes: <brief summary>
""",
    "qa": """\
You are Eve, the QA Engineer for a SWE-bench bug-fix workflow.

Your job is to run targeted validation in the provided repository checkout, verify the bug is resolved,
and send actionable failures back when the fix is incomplete.

Rules:
- Use workspace_run_command for targeted test/debug commands in the provided workspace.
- Prefer the smallest relevant test scope first, then broaden if needed.

Verification protocol for Fail-to-pass tests:
The harness auto-applies the dataset's gold test_patch into the workspace
BEFORE every command you run, so the Fail-to-pass IDs are normally already
present. The engineer is only responsible for the implementation patch.

1. DISCOVERY FIRST: Before running any targeted test by ID, run
   `pytest --collect-only <test_file>` to enumerate test IDs that actually
   exist. Compare the collected IDs against the Fail-to-pass list.
2. MISSING ID = INFRASTRUCTURE / IMPLEMENTATION FAILURE (not "engineer needs
   to write the test"). If a Fail-to-pass test ID is not collected, the cause
   is one of:
     a) The engineer's patch introduced a SyntaxError / ImportError that
        crashes pytest collection of the test module.
     b) The engineer accidentally edited the test file and broke it.
     c) The gold test_patch failed to apply cleanly in this environment.
   In all cases, mark verified=false and report the missing IDs back to the PM
   so the engineer can fix their implementation patch. Do NOT ask the engineer
   to add the test case themselves.
3. returncode=4 MEANS "NO TESTS COLLECTED": treat identically to a missing ID.
4. NEVER use a passing full-suite run to override a missing or failing targeted
   test. The full suite can pass trivially while the bug-triggering IDs error.

Structured sign-off:
At the end of your analysis, you MUST include a QA_VERDICT block in your message
back to the ProjectManager with exactly this format:

QA_VERDICT:
  verified: true | false
  failing_ids:
    - <test_id>: <reason>   # one entry per failed/missing Fail-to-pass test
  notes: <brief summary>

Set verified=true only when ALL Fail-to-pass tests exist AND pass (returncode=0).
Set verified=false otherwise and list every problematic ID with a reason.
""",
}


def is_swebench_mode() -> bool:
    return os.environ.get("MAS_MODE") == "swebench"


def get_role_system_message(role: str, default_message: str) -> str:
    if not is_swebench_mode():
        return default_message
    return ROLE_MESSAGES.get(role, default_message)


def load_task_context(task_path: str) -> dict[str, Any]:
    return json.loads(Path(task_path).read_text(encoding="utf-8"))


def build_task_prompt(task: dict[str, Any]) -> str:
    fail_to_pass = "\n".join(f"- {item}" for item in task.get("fail_to_pass", [])) or "- Not provided"
    pass_to_pass = "\n".join(f"- {item}" for item in task.get("pass_to_pass", [])) or "- Not provided"
    hints = task.get("hints_text") or "None"
    # In Docker mode the code lives at /testbed inside the container.
    # Giving agents the host path makes them navigate to a path that doesn't
    # exist inside Docker, causing the Engineer's mini-swe-agent to fail
    # immediately.  Use the container-internal path instead.
    if os.environ.get("MINI_AGENT_USE_DOCKER"):
        workspace = "/testbed"
    else:
        workspace = os.environ.get("MAS_WORKSPACE_PATH", "data/workspace")
    return (
        f"SWE-bench instance: {task['instance_id']}\n"
        f"Repository: {task['repo']}\n"
        f"Base commit: {task.get('base_commit', 'unknown')}\n"
        f"Workspace path: {workspace}\n\n"
        f"Problem statement:\n{task['problem_statement']}\n\n"
        f"Hints:\n{hints}\n\n"
        f"Fail-to-pass tests:\n{fail_to_pass}\n\n"
        f"Pass-to-pass tests:\n{pass_to_pass}\n"
    )


def parse_qa_verdict(message: str) -> dict[str, Any] | None:
    if "QA_VERDICT:" not in message:
        return None
    block = message.split("QA_VERDICT:", 1)[1]
    verified_match = _QA_VERDICT_BOOL_RE.search(block)
    if not verified_match:
        return None
    notes_match = re.search(r"notes:\s*(.+)", block)
    failing_ids: list[str] = []
    if "failing_ids:" in block:
        failing_block = block.split("failing_ids:", 1)[1]
        failing_block = failing_block.split("notes:", 1)[0]
        failing_ids = [
            line.strip()[2:].strip()
            for line in failing_block.splitlines()
            if line.strip().startswith("- ")
        ]
    return {
        "verified": verified_match.group(1).lower() == "true",
        "failing_ids": failing_ids,
        "notes": notes_match.group(1).strip() if notes_match else "",
        "raw": block.strip(),
    }


def parse_review_verdict(message: str) -> dict[str, Any] | None:
    if "REVIEW_VERDICT:" not in message:
        return None
    block = message.split("REVIEW_VERDICT:", 1)[1]
    verdict_match = _REVIEW_VERDICT_BOOL_RE.search(block)
    if not verdict_match:
        return None
    notes_match = re.search(r"notes:\s*(.+)", block)
    return {
        "blocking_findings": verdict_match.group(1).lower() == "true",
        "notes": notes_match.group(1).strip() if notes_match else "",
        "raw": block.strip(),
    }


def infer_review_blocking(message: str) -> bool | None:
    verdict = parse_review_verdict(message)
    if verdict is not None:
        return bool(verdict["blocking_findings"])
    lowered = message.lower()
    if "no blocking findings" in lowered or "approve" in lowered or "approved" in lowered:
        return False
    if "blocking finding" in lowered or "blocking findings" in lowered or "reject" in lowered:
        return True
    return None


def extract_changed_files_from_patch_text(patch_text: str) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for match in _DIFF_GIT_RE.finditer(patch_text):
        candidate = match.group(2).strip()
        if candidate != "/dev/null" and candidate not in seen:
            seen.add(candidate)
            files.append(candidate)
    for match in _DIFF_FILE_RE.finditer(patch_text):
        candidate = match.group(1).strip()
        if candidate != "/dev/null" and candidate not in seen:
            seen.add(candidate)
            files.append(candidate)
    return files


def assess_patch_relevance(task: dict[str, Any], patch_text: str) -> dict[str, Any]:
    changed_files = extract_changed_files_from_patch_text(patch_text)
    relevant_files: list[str] = []
    suspicious_files: list[str] = []
    if not changed_files:
        return {
            "acceptable": False,
            "changed_files": [],
            "relevant_files": [],
            "suspicious_files": [],
            "notes": ["Patch diff is empty."],
        }

    context_text = "\n".join(
        [
            str(task.get("problem_statement") or ""),
            str(task.get("hints_text") or ""),
            "\n".join(task.get("fail_to_pass", []) or []),
            "\n".join(task.get("pass_to_pass", []) or []),
        ]
    )
    explicit_paths = set(_PATH_RE.findall(context_text))
    explicit_tokens = set()
    for path in explicit_paths:
        explicit_tokens.update(_path_tokens(path))

    test_tokens = set()
    for test_id in [*(task.get("fail_to_pass", []) or []), *(task.get("pass_to_pass", []) or [])]:
        test_file = str(test_id).split("::", 1)[0]
        test_tokens.update(_path_tokens(test_file))

    problem_tokens = {
        token.lower()
        for token in _WORD_RE.findall(context_text)
        if len(token) >= 4
    }

    for changed in changed_files:
        score = _score_patch_file(
            changed,
            explicit_paths=explicit_paths,
            explicit_tokens=explicit_tokens,
            test_tokens=test_tokens,
            problem_tokens=problem_tokens,
        )
        if score > 0:
            relevant_files.append(changed)
        else:
            suspicious_files.append(changed)

    suspicious_top_level = [
        path for path in suspicious_files if "/" not in path and path in _SUSPICIOUS_TOP_LEVEL_FILES
    ]
    acceptable = bool(relevant_files) or not suspicious_top_level

    notes: list[str] = []
    if suspicious_top_level and not relevant_files:
        notes.append(
            "Patch only touched suspicious top-level shim files with no clear overlap to the issue, hints, or targeted tests."
        )
    elif suspicious_files:
        notes.append(
            "Some changed files do not clearly overlap with the issue statement, hints, or targeted tests."
        )

    return {
        "acceptable": acceptable,
        "changed_files": changed_files,
        "relevant_files": relevant_files,
        "suspicious_files": suspicious_files,
        "notes": notes,
    }


def _path_tokens(path: str) -> set[str]:
    normalized = path.strip().split("::", 1)[0].lower()
    parts = [part for part in _PATH_TOKEN_SPLIT_RE.split(normalized) if part]
    tokens: set[str] = set()
    for part in parts:
        if part.startswith("test") and len(part) > 4:
            part = part[4:]
        if part and part not in _TOKEN_STOPWORDS:
            tokens.add(part)
    return tokens


def _score_patch_file(
    changed_file: str,
    *,
    explicit_paths: set[str],
    explicit_tokens: set[str],
    test_tokens: set[str],
    problem_tokens: set[str],
) -> int:
    normalized = changed_file.lower()
    tokens = _path_tokens(normalized)
    score = 0
    if normalized in {path.lower() for path in explicit_paths}:
        score += 3
    if tokens & explicit_tokens:
        score += 2
    if tokens & test_tokens:
        score += 1
    if tokens & problem_tokens:
        score += 1
    return score
