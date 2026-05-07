"""
Thin MCP-backed tool wrappers for AutoGen agents.

These functions are simple adapters:
- AutoGen sees them as Python tools.
- Internally they call MCP servers via `MCPClientPool.call_tool`.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import os
import shlex
import subprocess
from typing import Any, Callable

from core.mcp_client import MCPClientPool
from core.mcp_config import BOARD_PATH, CODE_PATH, DOCS_PATH
from core.telemetry import record_tool_event


BOARD_PREFIXES = ("data/project_board", "project_board")
DOCS_PREFIXES = ("data/knowledge_base", "knowledge_base")
CODE_PREFIXES = ("data/workspace", "workspace")


def _normalize_scoped_path(path: str, prefixes: tuple[str, ...], root: str) -> str:
    """Normalize a tool path into an absolute path under the configured root."""
    if not path:
        return root

    normalized = path.replace("\\", "/").strip()
    if normalized in (".", "/"):
        return root

    normalized = normalized.lstrip("/")
    for prefix in prefixes:
        if normalized == prefix:
            return root
        if normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix) + 1 :]
            break

    cleaned = os.path.normpath(normalized).replace("\\", "/")
    if cleaned in ("", "."):
        return root
    if cleaned.startswith("../") or cleaned == "..":
        raise ValueError(f"Path '{path}' escapes the scoped root.")

    candidate = os.path.abspath(os.path.join(root, cleaned))
    if os.path.commonpath([candidate, root]) != root:
        raise ValueError(f"Path '{path}' escapes the scoped root.")
    return candidate


async def _fs_call(pool: MCPClientPool, server_key: str, tool: str, args: dict[str, Any]) -> Any:
    try:
        result = await pool.call_tool(server_key, tool, args)
        record_tool_event(tool, True, server_key=server_key)
    except Exception as exc:
        record_tool_event(tool, False, server_key=server_key, error=str(exc))
        raise
    # MCP results carry content blocks; unwrap text for simplicity.
    if hasattr(result, "content"):
        blocks = result.content
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
    return result


# ---------------------------------------------------------------------------
# Filesystem tools (server-filesystem MCP)
# ---------------------------------------------------------------------------

async def board_read_file(pool: MCPClientPool, path: str) -> str:
    """Read a ticket or document from the project board (data/project_board/)."""
    norm = _normalize_scoped_path(path, BOARD_PREFIXES, BOARD_PATH)
    return await _fs_call(pool, "fs_board", "read_file", {"path": norm})


async def board_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    """Create or update a ticket/document on the project board."""
    norm = _normalize_scoped_path(path, BOARD_PREFIXES, BOARD_PATH)
    return await _fs_call(pool, "fs_board", "write_file", {"path": norm, "content": content})


async def board_create_directory(pool: MCPClientPool, path: str) -> str:
    """Create a directory on the project board."""
    norm = _normalize_scoped_path(path, BOARD_PREFIXES, BOARD_PATH)
    return await _fs_call(pool, "fs_board", "create_directory", {"path": norm})


async def board_list_directory(pool: MCPClientPool, path: str = "") -> str:
    """List entries on the project board."""
    norm = _normalize_scoped_path(path, BOARD_PREFIXES, BOARD_PATH)
    return await _fs_call(pool, "fs_board", "list_directory", {"path": norm})


async def board_get_file_info(pool: MCPClientPool, path: str) -> str:
    """Get metadata for a board path."""
    norm = _normalize_scoped_path(path, BOARD_PREFIXES, BOARD_PATH)
    return await _fs_call(pool, "fs_board", "get_file_info", {"path": norm})


async def board_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    """Read multiple board files at once."""
    norm_paths = [_normalize_scoped_path(p, BOARD_PREFIXES, BOARD_PATH) for p in paths]
    return await _fs_call(pool, "fs_board", "read_multiple_files", {"paths": norm_paths})


async def docs_read_file(pool: MCPClientPool, path: str) -> str:
    """Read a knowledge-base document (data/knowledge_base/)."""
    norm = _normalize_scoped_path(path, DOCS_PREFIXES, DOCS_PATH)
    return await _fs_call(pool, "fs_docs", "read_file", {"path": norm})


async def docs_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    """Create or update a knowledge-base document."""
    norm = _normalize_scoped_path(path, DOCS_PREFIXES, DOCS_PATH)
    return await _fs_call(pool, "fs_docs", "write_file", {"path": norm, "content": content})


async def docs_create_directory(pool: MCPClientPool, path: str) -> str:
    """Create a directory in the knowledge base."""
    norm = _normalize_scoped_path(path, DOCS_PREFIXES, DOCS_PATH)
    return await _fs_call(pool, "fs_docs", "create_directory", {"path": norm})


async def docs_list_directory(pool: MCPClientPool, path: str = "") -> str:
    """List entries in the knowledge base."""
    norm = _normalize_scoped_path(path, DOCS_PREFIXES, DOCS_PATH)
    return await _fs_call(pool, "fs_docs", "list_directory", {"path": norm})


async def docs_get_file_info(pool: MCPClientPool, path: str) -> str:
    """Get metadata for a knowledge-base path."""
    norm = _normalize_scoped_path(path, DOCS_PREFIXES, DOCS_PATH)
    return await _fs_call(pool, "fs_docs", "get_file_info", {"path": norm})


async def docs_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    """Read multiple knowledge-base files at once."""
    norm_paths = [_normalize_scoped_path(p, DOCS_PREFIXES, DOCS_PATH) for p in paths]
    return await _fs_call(pool, "fs_docs", "read_multiple_files", {"paths": norm_paths})


async def code_read_file(pool: MCPClientPool, path: str) -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "read_file", {"path": norm})


async def code_list_directory(pool: MCPClientPool, path: str = "") -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "list_directory", {"path": norm})


async def code_get_file_info(pool: MCPClientPool, path: str) -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "get_file_info", {"path": norm})


async def code_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    norm_paths = [_normalize_scoped_path(p, CODE_PREFIXES, CODE_PATH) for p in paths]
    return await _fs_call(pool, "fs_code", "read_multiple_files", {"paths": norm_paths})


async def code_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "write_file", {"path": norm, "content": content})


async def code_create_directory(pool: MCPClientPool, path: str) -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "create_directory", {"path": norm})


async def code_move_file(pool: MCPClientPool, source: str, destination: str) -> str:
    norm_src = _normalize_scoped_path(source, CODE_PREFIXES, CODE_PATH)
    norm_dst = _normalize_scoped_path(destination, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "move_file", {"source": norm_src, "destination": norm_dst})


async def code_search_files(pool: MCPClientPool, path: str, pattern: str) -> str:
    norm = _normalize_scoped_path(path, CODE_PREFIXES, CODE_PATH)
    return await _fs_call(pool, "fs_code", "search_files", {"path": norm, "pattern": pattern})


def _allowed_commands() -> set[str]:
    raw = os.environ.get(
        "MAS_ALLOWED_COMMANDS",
        "python,python3,pytest,py.test,uv,pip,pip3,tox,git,ls,cat,sed,grep,rg",
    )
    return {item.strip() for item in raw.split(",") if item.strip()}


async def workspace_run_command(pool: MCPClientPool, command: str, timeout_seconds: int = 300) -> str:
    del pool
    parts = shlex.split(command)
    if not parts:
        record_tool_event("workspace_run_command", False, error="empty_command")
        return "ERROR: empty command"
    if parts[0] not in _allowed_commands():
        record_tool_event("workspace_run_command", False, error="not_allowlisted", command=parts[0])
        return f"ERROR: command '{parts[0]}' is not allowlisted"
    timeout_limit = min(timeout_seconds, int(os.environ.get("MAS_COMMAND_TIMEOUT", "300")))
    completed = subprocess.run(
        parts,
        cwd=os.environ.get("MAS_WORKSPACE_PATH", CODE_PATH),
        capture_output=True,
        text=True,
        timeout=timeout_limit,
        check=False,
    )
    success = completed.returncode == 0
    record_tool_event(
        "workspace_run_command",
        success,
        command=command,
        returncode=completed.returncode,
    )
    return (
        f"returncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


BOARD_TOOLS = [
    board_read_file,
    board_write_file,
    board_create_directory,
    board_list_directory,
    board_get_file_info,
    board_read_multiple_files,
]

DOCS_TOOLS = [
    docs_read_file,
    docs_write_file,
    docs_create_directory,
    docs_list_directory,
    docs_get_file_info,
    docs_read_multiple_files,
]

CODE_READ_TOOLS = [
    code_read_file,
    code_list_directory,
    code_get_file_info,
    code_read_multiple_files,
    code_search_files,
]

CODE_WRITE_TOOLS = [
    *CODE_READ_TOOLS,
    code_write_file,
    code_create_directory,
    code_move_file,
]

SHELL_TOOLS = [
    workspace_run_command,
]


# ---------------------------------------------------------------------------
# Git tools (mcp-server-git)
# ---------------------------------------------------------------------------

REPO_ROOT = os.environ.get("WORKSPACE_GIT_ROOT", CODE_PATH)


async def _git_call(pool: MCPClientPool, tool: str, args: dict[str, Any]) -> str:
    # mcp-server-git (recent versions) expects a repo_path argument for most tools.
    # Default to this repository root so agents don't need to thread it through.
    args.setdefault("repo_path", REPO_ROOT)
    try:
        result = await pool.call_tool("git", tool, args)
        record_tool_event(tool, True, server_key="git")
    except Exception as exc:
        record_tool_event(tool, False, server_key="git", error=str(exc))
        raise
    if hasattr(result, "content"):
        blocks = result.content
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
    return str(result)


async def git_status(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_status", {})


async def git_diff_unstaged(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_diff_unstaged", {})


async def git_diff_staged(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_diff_staged", {})


async def git_diff(pool: MCPClientPool, target: str) -> str:
    return await _git_call(pool, "git_diff", {"target": target})


async def git_log(pool: MCPClientPool, max_count: int = 10) -> str:
    return await _git_call(pool, "git_log", {"max_count": max_count})


async def git_show(pool: MCPClientPool, revision: str) -> str:
    return await _git_call(pool, "git_show", {"revision": revision})


async def git_add(pool: MCPClientPool, files: list[str]) -> str:
    return await _git_call(pool, "git_add", {"files": files})


async def git_commit(pool: MCPClientPool, message: str) -> str:
    return await _git_call(pool, "git_commit", {"message": message})


async def git_create_branch(pool: MCPClientPool, branch_name: str, start_point: str | None = None) -> str:
    args: dict[str, Any] = {"branch_name": branch_name}
    if start_point:
        args["start_point"] = start_point
    return await _git_call(pool, "git_create_branch", args)


async def git_checkout(pool: MCPClientPool, branch_name: str) -> str:
    return await _git_call(pool, "git_checkout", {"branch_name": branch_name})


GIT_READ_TOOLS = [
    git_status,
    git_diff_unstaged,
    git_diff_staged,
    git_diff,
    git_log,
    git_show,
]

GIT_WRITE_TOOLS = [
    *GIT_READ_TOOLS,
    git_add,
    git_commit,
    git_create_branch,
    git_checkout,
]


# ---------------------------------------------------------------------------
# bind_tools helper (captures MCPClientPool for AutoGen)
# ---------------------------------------------------------------------------

def bind_tools(pool: MCPClientPool, *tool_fns: Callable) -> list[Callable]:
    """
    Bind `pool` as the first argument of each tool function and return a list
    of AutoGen-compatible async callables.

    The returned callables:
    - keep the original ``__name__`` and ``__doc__``
    - have a signature that matches everything *after* the ``pool`` parameter
    - are always async (sync functions are wrapped with asyncio.to_thread)
    """
    bound: list[Callable] = []
    for fn in tool_fns:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        inner_params = params[1:]  # drop the pool param
        new_sig = sig.replace(parameters=inner_params)

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, _fn=fn, _pool=pool, **kwargs: Any) -> Any:
                return await _fn(_pool, *args, **kwargs)

            _async_wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
            bound.append(_async_wrapper)
        else:

            @functools.wraps(fn)
            async def _sync_wrapper(*args: Any, _fn=fn, _pool=pool, **kwargs: Any) -> Any:
                return await asyncio.to_thread(_fn, _pool, *args, **kwargs)

            _sync_wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
            bound.append(_sync_wrapper)

    return bound


__all__ = [
    "BOARD_TOOLS",
    "DOCS_TOOLS",
    "CODE_READ_TOOLS",
    "CODE_WRITE_TOOLS",
    "SHELL_TOOLS",
    "GIT_READ_TOOLS",
    "GIT_WRITE_TOOLS",
    "bind_tools",
]
