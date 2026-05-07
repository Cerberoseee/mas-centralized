"""Helpers for SWE-bench mode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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
- When the issue is resolved and validated, produce a final summary starting with: PROJECT COMPLETE
""",
    "architect": """\
You are Bob, the Architect for a SWE-bench bug-fix workflow.

Your job is to inspect the existing repository, understand the failure mode, and document a minimal technical plan.
Do not invent new systems or rewrite broad architecture. Focus on root cause, affected files, and likely fix strategy.
""",
    "engineer": """\
You are Charlie, the Engineer for a SWE-bench bug-fix workflow.

Your job is to work inside an existing repository checkout, reproduce or inspect the failing behavior, implement the fix,
run targeted validation, and commit the minimal required code changes.

Rules:
- Do not scaffold a new project.
- Use workspace_run_command for targeted inspection/test commands in the provided workspace.
- Prefer minimal edits to existing files.
- When asked to fix reviewer or QA findings, address them directly and re-run relevant checks.
""",
    "code_reviewer": """\
You are Dave, the Code Reviewer for a SWE-bench bug-fix workflow.

Your job is to inspect the diff in the existing repository and decide whether the proposed fix is correct, minimal,
and aligned with the issue statement. Focus on correctness, regression risk, and unnecessary scope expansion.
""",
    "qa": """\
You are Eve, the QA Engineer for a SWE-bench bug-fix workflow.

Your job is to run targeted validation in the provided repository checkout, verify the bug is resolved,
and send actionable failures back when the fix is incomplete.

Rules:
- Use workspace_run_command for targeted test/debug commands in the provided workspace.
- Prefer the smallest relevant test scope first, then broaden if needed.
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
