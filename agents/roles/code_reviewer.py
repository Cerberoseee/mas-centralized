"""
Code Reviewer agent.

Receives a review request from the ProjectManager, inspects the code and
diffs, provides actionable feedback, and reports back.
"""
from __future__ import annotations

import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff

from core.autogen_config import get_model_client
from core.mcp_client import MCPClientPool
from core.mcp_tools import BOARD_TOOLS, DOCS_TOOLS, CODE_READ_TOOLS, GIT_READ_TOOLS, PATCH_TOOLS, bind_tools
from core.swebench import get_role_system_message


_SYSTEM_MESSAGE = """\
You are Dave, a Code Reviewer.

Your responsibilities:
- Inspect the implementation code and git diffs to evaluate correctness,
  readability, maintainability, performance, and adherence to standards.
- Use the knowledge base docs as reference for architecture and conventions.
- Provide clear, actionable feedback with specific file/line references.
- If critical issues are found, list them explicitly so the Engineer can
  address them.
- You will be given the ticket file path(s) on the project board (typically
  under data/project_board/tickets/). Update the corresponding ticket file:
  - If changes are acceptable, move Status to DONE.
  - If significant issues remain, move Status back to IN PROGRESS and clearly
    state what must be fixed.
- When your review is complete, hand control back to the ProjectManager
  using the transfer_to_ProjectManager tool.

Handoff tools available to you:
- transfer_to_ProjectManager : return control to the ProjectManager when done.

Other tools available to you:
- board_*       : read and write the project board (data/project_board/).
- docs_*        : read from the knowledge base (data/knowledge_base/).
- code_read_*   : read the implementation code (data/workspace/).
- git_*         : inspect diffs, commits, and branch state.

Rules:
- Chain of Thought: Before executing any tool call or handoff, you MUST output your internal reasoning explicitly (e.g., "Thought: First I need to review the diffs..."). Think step-by-step.
- Never attempt to read or write paths outside these data/ directories.
- Do NOT modify any code files.
- Always call transfer_to_ProjectManager when your review is complete.
"""


def _is_docker_mode() -> bool:
    return os.environ.get("MINI_AGENT_USE_DOCKER", "").strip().lower() in ("1", "true", "yes")


class CodeReviewer:
    """Constructs an AutoGen AssistantAgent configured for the Code Reviewer role."""

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        # In Docker mode the git MCP server is not in the pool (the repo lives
        # inside the container, not on the host), and CODE_READ_TOOLS point at
        # an empty host workspace.  Use read_patch_diff instead so the reviewer
        # can still inspect what the Engineer actually changed.
        if _is_docker_mode():
            review_tools = [*BOARD_TOOLS, *DOCS_TOOLS, *PATCH_TOOLS]
        else:
            review_tools = [*BOARD_TOOLS, *DOCS_TOOLS, *CODE_READ_TOOLS, *GIT_READ_TOOLS]
        self.agent = AssistantAgent(
            name="CodeReviewer",
            model_client=get_model_client(),
            tools=bind_tools(pool, *review_tools),
            handoffs=[
                Handoff(target="ProjectManager", description="Return control to the ProjectManager when review is complete."),
            ],
            system_message=get_role_system_message("code_reviewer", _SYSTEM_MESSAGE),
        )
