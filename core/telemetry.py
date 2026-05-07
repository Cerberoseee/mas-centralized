"""Structured telemetry helpers for evaluator-facing runs."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any


def _initial_state() -> dict[str, Any]:
    return {
        "messages": 0,
        "handoffs": 0,
        "tool_calls": 0,
        "tool_failures": 0,
        "retries": 0,
        "escalations": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "per_agent": {},
        "tool_events": [],
        "message_events": [],
        "final_status": "",
        "patch_path": os.environ.get("MAS_EVAL_PATCH_PATH"),
    }


_STATE: dict[str, Any] = _initial_state()


def reset() -> None:
    global _STATE
    _STATE = _initial_state()


def record_message(source: str, content: str | None = None, models_usage: Any | None = None) -> None:
    _STATE["messages"] += 1
    agent = _STATE["per_agent"].setdefault(
        source,
        {
            "messages": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    agent["messages"] += 1
    if models_usage is not None:
        prompt_tokens = int(getattr(models_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(models_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(models_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        _STATE["prompt_tokens"] += prompt_tokens
        _STATE["completion_tokens"] += completion_tokens
        _STATE["total_tokens"] += total_tokens
        agent["prompt_tokens"] += prompt_tokens
        agent["completion_tokens"] += completion_tokens
        agent["total_tokens"] += total_tokens
    _STATE["message_events"].append({"source": source, "content": (content or "")[:200]})


def record_handoff(source: str, target: str | None = None) -> None:
    _STATE["handoffs"] += 1
    if target == "ProjectManager":
        _STATE["escalations"] += 1
    _STATE["message_events"].append({"source": source, "target": target, "type": "handoff"})


def record_retry() -> None:
    _STATE["retries"] += 1


def record_tool_event(name: str, success: bool, **details: Any) -> None:
    _STATE["tool_calls"] += 1
    if not success:
        _STATE["tool_failures"] += 1
    event = {"tool": name, "success": success}
    event.update(details)
    _STATE["tool_events"].append(event)


def set_final_status(status: str) -> None:
    _STATE["final_status"] = status


def snapshot() -> dict[str, Any]:
    return deepcopy(_STATE)


def write_if_configured(path: str | None = None) -> None:
    target = path or os.environ.get("MAS_EVAL_TELEMETRY_PATH")
    if not target:
        return
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(snapshot(), handle, indent=2)
