"""
Swarm-level guard against any single agent monopolizing the conversation.

In AutoGen Swarm, the current speaker stays active until it emits a
HandoffMessage. Without a cap, any role can spin on tool calls indefinitely.
This module adds a FunctionalTermination that limits (1) total chat messages
from any one agent and (2) a trailing run of messages from the same agent.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Sequence

from autogen_agentchat.conditions import FunctionalTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGENT_MESSAGES = 18
_DEFAULT_MAX_CONSECUTIVE_AGENT_MESSAGES = 9


def _effective_limit(
    keys: tuple[str, ...],
    default: int,
) -> int | None:
    """
    First non-empty parsable env among `keys` wins (legacy QA-specific keys last).
    If none set, use ``default``. A value <= 0 turns that limit off entirely.
    """
    for key in keys:
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            continue
        try:
            v = int(raw.strip(), 10)
        except ValueError:
            continue
        if v <= 0:
            return None
        return v
    return default


def swarm_monopoly_exceeded(
    messages: Sequence[BaseAgentEvent | BaseChatMessage],
    *,
    max_messages_per_agent: int | None,
    max_consecutive_same_agent: int | None,
) -> bool:
    chat = [m for m in messages if isinstance(m, BaseChatMessage)]

    if max_messages_per_agent is not None:
        counts = Counter(m.source for m in chat)
        for source, n in counts.items():
            if n >= max_messages_per_agent:
                logger.warning(
                    "Swarm loop guard: agent %r total chat messages %s >= limit %s",
                    source,
                    n,
                    max_messages_per_agent,
                )
                return True

    if max_consecutive_same_agent is not None and chat:
        last_source = chat[-1].source
        consecutive = 0
        for m in reversed(chat):
            if m.source == last_source:
                consecutive += 1
            else:
                break
        if consecutive >= max_consecutive_same_agent:
            logger.warning(
                "Swarm loop guard: agent %r trailing chat messages %s >= limit %s",
                last_source,
                consecutive,
                max_consecutive_same_agent,
            )
            return True

    return False


def swarm_loop_guard_enabled() -> bool:
    for key in ("MAS_SWARM_LOOP_GUARD", "MAS_QA_LOOP_GUARD"):
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            continue
        return raw.strip().lower() not in ("0", "false", "no", "off")
    return True


def build_swarm_loop_guard_termination() -> FunctionalTermination | None:
    if not swarm_loop_guard_enabled():
        return None

    max_per = _effective_limit(
        ("MAS_MAX_AGENT_MESSAGES", "MAS_MAX_QA_MESSAGES"),
        _DEFAULT_MAX_AGENT_MESSAGES,
    )
    max_consec = _effective_limit(
        (
            "MAS_MAX_CONSECUTIVE_AGENT_MESSAGES",
            "MAS_MAX_CONSECUTIVE_QA_MESSAGES",
        ),
        _DEFAULT_MAX_CONSECUTIVE_AGENT_MESSAGES,
    )
    if max_per is None and max_consec is None:
        return None

    def _check(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        return swarm_monopoly_exceeded(
            messages,
            max_messages_per_agent=max_per,
            max_consecutive_same_agent=max_consec,
        )

    return FunctionalTermination(_check)
