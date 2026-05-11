"""Compatibility patches for AutoGen runtime behavior."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_RUNTIME_SHUTDOWN_PATCHED = False


def patch_single_threaded_runtime_shutdown() -> None:
    """Patch AutoGen runtime shutdown for Python 3.13 asyncio.Queue semantics.

    Python 3.13's ``asyncio.Queue.shutdown(immediate=True)`` decrements the
    unfinished-task counter for queued items. AutoGen's embedded runtime can
    still have in-flight handlers that call ``task_done()`` during graceful
    team shutdown, which leads to ``ValueError: task_done() called too many
    times`` noise after a successful run.

    For the embedded team shutdown path we want a graceful stop, so using
    ``immediate=False`` preserves the queue bookkeeping while still waking the
    runtime loop once the queue is idle.
    """

    global _RUNTIME_SHUTDOWN_PATCHED
    if _RUNTIME_SHUTDOWN_PATCHED:
        return

    try:
        from autogen_core._single_threaded_agent_runtime import RunContext
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not patch AutoGen runtime shutdown: %s", exc)
        return

    async def _stop(self) -> None:
        self._stopped.set()
        self._runtime._message_queue.shutdown(immediate=False)  # type: ignore[attr-defined]
        await self._run_task

    async def _stop_when_idle(self) -> None:
        await self._runtime._message_queue.join()  # type: ignore[attr-defined]
        self._stopped.set()
        self._runtime._message_queue.shutdown(immediate=False)  # type: ignore[attr-defined]
        await self._run_task

    RunContext.stop = _stop
    RunContext.stop_when_idle = _stop_when_idle
    _RUNTIME_SHUTDOWN_PATCHED = True
    logger.info("Patched AutoGen SingleThreadedAgentRuntime shutdown for Python 3.13 queue semantics.")
