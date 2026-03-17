"""
QA agent.

Receives a validation request from the ProjectManager, designs and executes
tests, and reports results.
"""
from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff

from core.autogen_config import get_model_client
from core.mcp_client import MCPClientPool
from core.mcp_tools import BOARD_TOOLS, bind_tools, CODE_WRITE_TOOLS


_SYSTEM_MESSAGE = """\
You are Eve, a QA Engineer.

Your responsibilities:
- Inspect the project board and the implemented code to understand what
  should be tested.
- Design test cases covering happy paths, edge cases, and regressions.
- Document your test results clearly, listing any failures or quality
  concerns.
- You will be given the ticket file path(s) on the project board (typically
  under data/project_board/tickets/). When you finish validating a ticket,
  add a brief summary of test results to that ticket file and make sure the
  Status accurately reflects whether the work is DONE or needs further changes.
- When all testing is complete, hand control back to the ProjectManager
  using the transfer_to_ProjectManager tool.

Handoff tools available to you:
- transfer_to_ProjectManager : return control to the ProjectManager when done.

Other tools available to you:
- board_*       : read and write the project board (data/project_board/).
- code_read_*   : read the implementation code (data/workspace/).

Rules:
- Never attempt to read or write paths outside these data/ directories.
- Do NOT modify implementation code.
- Always call transfer_to_ProjectManager when all testing is complete.
"""


class QA:
    """Constructs an AutoGen AssistantAgent configured for the QA role."""

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        self.agent = AssistantAgent(
            name="QA",
            model_client=get_model_client(),
            tools=bind_tools(pool, *BOARD_TOOLS, *CODE_WRITE_TOOLS),
            handoffs=[
                Handoff(target="ProjectManager", description="Return control to the ProjectManager when testing is complete."),
            ],
            system_message=_SYSTEM_MESSAGE,
        )
