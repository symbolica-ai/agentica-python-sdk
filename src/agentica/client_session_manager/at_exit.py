"""
Async cleanup before event loop shutdown.

Catches SIGINT to run async cleanup while the loop is still healthy.
Falls back to sync cleanup for normal exits when executor is dead.
"""

import asyncio
import atexit
import logging
import signal
import weakref
from functools import partial
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

__all__ = ["register", "unregister"]

type AsyncCleanupCallback = Callable[[], Coroutine[Any, Any, None]]
type SyncCleanupCallback = Callable[[], None]

_registry: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, "_LoopEntry"] = (
    weakref.WeakKeyDictionary()
)
_sync_callbacks: list[SyncCleanupCallback] = []
_cleanup_done = False
_atexit_registered = False


class _LoopEntry:
    """Tracks cleanup callbacks and original close() for a single event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        if not hasattr(loop, "_orig_close"):
            loop._orig_close = loop.close  # type: ignore
        try:
            self._close_ref = weakref.WeakMethod(loop._orig_close)  # type: ignore
        except TypeError:
            self._close_ref = lambda: loop._orig_close  # type: ignore
        self.callbacks: list[AsyncCleanupCallback] = []

    def orig_close(self) -> None:
        close_fn = self._close_ref()
        if close_fn:
            close_fn()


def register(
    async_callback: AsyncCleanupCallback,
    sync_fallback: SyncCleanupCallback | None = None,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Register cleanup callbacks. sync_fallback runs if async fails (executor dead)."""
    entry = _get_or_create_entry(loop)
    entry.callbacks.append(async_callback)

    if sync_fallback:
        _sync_callbacks.append(sync_fallback)
        _ensure_atexit()


def unregister(
    async_callback: AsyncCleanupCallback,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Remove a previously registered callback."""
    entry = _get_or_create_entry(loop)
    while async_callback in entry.callbacks:
        entry.callbacks.remove(async_callback)


def _ensure_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_run_sync_fallbacks)
        _atexit_registered = True


def _get_or_create_entry(loop: asyncio.AbstractEventLoop | None = None) -> _LoopEntry:
    if loop is None:
        loop = asyncio.get_running_loop()

    if loop not in _registry:
        _registry[loop] = _LoopEntry(loop)
        loop.close = partial(_patched_close, loop)  # type: ignore
        loop.add_signal_handler(signal.SIGINT, partial(_on_sigint, loop))

    return _registry[loop]


def _on_sigint(loop: asyncio.AbstractEventLoop) -> None:
    """Handle Ctrl+C by running cleanup while the loop is still healthy."""
    global _cleanup_done

    if _cleanup_done:
        raise KeyboardInterrupt

    async def cleanup_then_exit():
        global _cleanup_done
        await _run_all_async(loop)
        _sync_callbacks.clear()  # Async worked, don't need sync fallback
        _cleanup_done = True
        raise KeyboardInterrupt

    loop.create_task(cleanup_then_exit())


async def _run_all_async(loop: asyncio.AbstractEventLoop) -> None:
    entry = _registry.get(loop)
    if not entry or not entry.callbacks:
        return

    for callback in entry.callbacks:
        try:
            await callback()
        except Exception as e:
            logger.error(f"Cleanup callback failed: {e}", exc_info=True)

    entry.callbacks.clear()


def _patched_close(loop: asyncio.AbstractEventLoop) -> None:
    """Attempt async cleanup when loop.close() is called."""
    entry = _registry.get(loop)
    if entry and entry.callbacks and not _cleanup_done:
        try:
            loop.run_until_complete(_run_all_async(loop))
            # Don't set _cleanup_done here; async may have silently failed
            # Let atexit sync fallback run as backup
        except Exception:
            pass

    if entry:
        entry.orig_close()


def _run_sync_fallbacks() -> None:
    """Run sync fallback callbacks via atexit."""
    global _cleanup_done

    if _cleanup_done:
        return

    _cleanup_done = True

    for callback in _sync_callbacks:
        try:
            callback()
        except Exception as e:
            logger.error(f"Sync cleanup callback failed: {e}", exc_info=True)

    _sync_callbacks.clear()
