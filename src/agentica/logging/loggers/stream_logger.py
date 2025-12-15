import asyncio
import itertools
from collections.abc import AsyncIterator, Iterator
from contextvars import Token
from typing import Awaitable, Callable, override

from agentica.common import Chunk

from ..agent_logger import AgentLogger, DefaultLogId, LogId, contextual_enable_streaming

__all__ = ['StreamLogger']


class _Sentinel: ...


_id_counter = itertools.count()


class StreamLogger(AgentLogger):
    """
    A logger that exposes an iterator over the agent's text-generation stream.

    NOTE: This is really intended to be used *per invocation*!
    The iterator this exposes finishes when an invocation is complete.

    However, If you wish to use this across multiple invocations,
    you may *reconsume* the iterator after it has finished, that way
    each time you consume the iterator, you advance one invocation.

    Example:
    --------

    ```python
    stream = StreamLogger()
    with stream:
        result1 = await agent.call(task1)
        result2 = await agent.call(task2)

    # Consume stream for the first time.
    print("Task 1 invocation:")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()

    # Consume stream for the second time.
    print("Task 2 invocation:")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    ```
    """

    local_id: LogId | None
    _queue: asyncio.Queue[Chunk | _Sentinel]
    _sentinel: _Sentinel
    _streaming_token: Token[bool] | None
    _on_chunk: Callable[[Chunk], Awaitable[None]] | None

    def __init__(self, on_chunk: Callable[[Chunk], Awaitable[None]] | None = None) -> None:
        self.local_id = None
        self._sentinel = _Sentinel()
        self._queue = asyncio.Queue()
        self._streaming_token = None
        self._on_chunk = on_chunk

    @override
    def should_stream(self) -> bool:
        return True

    @override
    def on_spawn(self) -> None:
        if self.local_id is None:
            self.local_id = DefaultLogId()

    @override
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        pass

    @override
    def on_call_exit(self, result: object) -> None:
        self._queue.put_nowait(self._sentinel)

    @override
    async def on_chunk(self, chunk: Chunk) -> None:
        await self._queue.put(chunk)
        if self._on_chunk is not None:
            await self._on_chunk(chunk)

    @override
    def __enter__(self) -> None:
        self._streaming_token = contextual_enable_streaming.set(True)
        return super().__enter__()

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._streaming_token is not None
        contextual_enable_streaming.reset(self._streaming_token)
        return super().__exit__(exc_type, exc_value, traceback)

    async def __anext__(self) -> Chunk:
        chunk = await self._queue.get()
        if chunk is self._sentinel:
            raise StopAsyncIteration
        assert isinstance(chunk, Chunk)
        return chunk

    def __aiter__(self) -> AsyncIterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        chunk = self._queue.get_nowait()
        if chunk is self._sentinel:
            raise StopIteration
        assert isinstance(chunk, Chunk)
        return chunk

    def __iter__(self) -> Iterator[Chunk]:
        return self
