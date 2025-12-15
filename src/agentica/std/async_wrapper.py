import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, Callable
from weakref import WeakKeyDictionary

__all__ = [
    'call_async_safe',
]


class _AsyncExecutor:
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._ready: threading.Event = threading.Event()

    def _run_event_loop(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()
        except Exception as e:
            print(f"Error in async executor event loop: {e}")
        finally:
            self._loop = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._shutdown = False
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Failed to start async executor within timeout")

    def submit[T, Q, P](self, coro: Coroutine[Q, P, T]) -> T:
        if self._loop is None:
            raise RuntimeError("Async executor not started")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = future.result()
            return result
        except BaseException as e:
            raise e

    def shutdown(self):
        if self._loop is not None and not self._shutdown:
            self._shutdown = True
            _ = self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


_executors: WeakKeyDictionary[threading.Thread, _AsyncExecutor] = WeakKeyDictionary()
_executor_lock = threading.Lock()


def _cleanup_executors():
    with _executor_lock:
        for executor in list(_executors.values()):
            try:
                executor.shutdown()
            except Exception:
                pass
        _executors.clear()


_ = atexit.register(_cleanup_executors)


def _get_or_create_executor() -> _AsyncExecutor:
    current_thread = threading.current_thread()

    with _executor_lock:
        if current_thread in _executors:
            executor = _executors[current_thread]
            if (
                hasattr(executor, '_thread')
                and executor._thread is not None
                and executor._thread.is_alive()
            ):
                return executor

        executor = _AsyncExecutor()
        executor.start()
        _executors[current_thread] = executor
        return executor


def call_async_safe(func: Callable, *args, **kwargs) -> Any:
    executor = _get_or_create_executor()
    coro = func(*args, **kwargs)
    return executor.submit(coro)
