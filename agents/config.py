"""
Path constants shared across agents and tools.

The actual workspace directories are also used by the MCP server
configurations in core/mcp_config.py.  Keep them in sync.
"""
from __future__ import annotations

import os

WORKSPACE_SUFFIX = os.environ.get("WORKSPACE_SUFFIX", "").strip()
DATA_DIRNAME = f"data_{WORKSPACE_SUFFIX}" if WORKSPACE_SUFFIX else "data"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{DATA_DIRNAME}"))

BOARD_PATH = os.environ.get("MAS_BOARD_PATH") or os.path.join(BASE_DIR, "project_board")
DOCS_PATH = os.environ.get("MAS_DOCS_PATH") or os.path.join(BASE_DIR, "knowledge_base")
CODE_PATH = os.environ.get("MAS_WORKSPACE_PATH") or os.path.join(BASE_DIR, "workspace")


def ensure_workspace_dirs() -> None:
    """Create the workspace directories if they do not exist."""
    for path in [BOARD_PATH, DOCS_PATH, CODE_PATH]:
        os.makedirs(path, exist_ok=True)
