"""
Base class for logging agent chat histories and invocations.
"""

import asyncio
import itertools
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from typing import Protocol, Self, override

from agentica.common import Chunk

__all__ = [
    'AgentLogger',
    'CompositeLogger',
    'DefaultLogId',
    'NamedLogId',
    'NoopLogger',
    'NoLogging',
    'contextual_logger',
    'no_logging',
    'Streaming',
    'enable_streaming',
]


class IntCmp(Protocol):
    """Comparable to an integer."""

    def __int__(self) -> int: ...
    def __lt__(self, value: Self | int, /) -> bool: ...
    def __gt__(self, value: Self | int, /) -> bool: ...


class DefaultLogId:
    lid: int
    _id_counter = itertools.count()

    def __init__(self) -> None:
        cls = type(self)
        self.id = next(cls._id_counter)

    def __str__(self) -> str:
        return f"logger:{self.id}"

    def __int__(self) -> int:
        return self.lid

    def __lt__(self, other: IntCmp) -> bool:
        return self.lid < int(other)

    def __gt__(self, other: IntCmp) -> bool:
        return self.lid > int(other)


type LogId = IntCmp


class NamedLogId(DefaultLogId):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return self.name


class AgentLogger(ABC):
    local_id: LogId | None
    parent_local_id: LogId | None
    _token: Token['list[AgentLogger] | None']

    @abstractmethod
    def on_spawn(self) -> None:
        pass

    @abstractmethod
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        pass

    @abstractmethod
    def on_call_exit(self, result: object) -> None:
        pass

    @abstractmethod
    async def on_chunk(self, chunk: Chunk) -> None:
        pass

    def should_stream(self) -> bool:
        """Whether this logger should trigger streaming to the client."""
        return False

    def clone(self) -> Self:
        import copy

        return copy.copy(self)

    # Make this a context manager which sets a "global" logger

    def __enter__(self) -> None:
        loggers = contextual_logger.get()
        if loggers is None:
            self._token = contextual_logger.set([self])
        else:
            self._token = contextual_logger.set(loggers + [self])

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        contextual_logger.reset(self._token)


contextual_logger: ContextVar[list[AgentLogger] | None] = ContextVar(
    "contextual_logger", default=None
)


class CompositeLogger(AgentLogger):
    local_id: LogId | None
    loggers: list[AgentLogger]
    owned_by_listener: bool

    def __init__(self, loggers: list[AgentLogger], *, owned_by_listener: bool = False) -> None:
        self.loggers = loggers
        self.owned_by_listener = owned_by_listener
        self.local_id = None

    @override
    def on_spawn(self) -> None:
        first_logger = None
        for logger in self.loggers:
            if first_logger is None:
                if not hasattr(logger, 'local_id') or logger.local_id is None:
                    logger.on_spawn()
                first_logger = logger
                logger.local_id = first_logger.local_id
            else:
                if not hasattr(logger, 'local_id') or logger.local_id is None:
                    logger.local_id = first_logger.local_id
                    logger.on_spawn()
                else:
                    logger.local_id = first_logger.local_id

    @override
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        for logger in self.loggers:
            logger.on_call_enter(user_prompt, parent_local_id)

    @override
    def on_call_exit(self, result: object) -> None:
        for logger in self.loggers:
            logger.on_call_exit(result)

    @override
    async def on_chunk(self, chunk: Chunk) -> None:
        async with asyncio.TaskGroup() as tg:
            for logger in self.loggers:
                _ = tg.create_task(logger.on_chunk(chunk))

    def should_stream(self) -> bool:
        return any(logger.should_stream() for logger in self.loggers)

    def __repr__(self) -> str:
        loggers = self.loggers
        return f"<CompositeLogger({loggers=}) at 0x{id(self):x}>"


class NoopLogger(AgentLogger):
    local_id: LogId | None

    def __init__(self) -> None:
        self.local_id = None

    def on_spawn(self) -> None:
        if self.local_id is None:
            self.local_id = DefaultLogId()

    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        pass

    def on_call_exit(self, result: object) -> None:
        pass

    async def on_chunk(self, chunk: Chunk) -> None:
        pass


class NoLogging:
    def __enter__(self) -> None:
        self._token = contextual_logger.set([])

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        contextual_logger.reset(self._token)


no_logging = NoLogging()

contextual_enable_streaming: ContextVar[bool] = ContextVar(
    "contextual_enable_streaming", default=False
)


class Streaming:
    def __enter__(self) -> None:
        self._token = contextual_enable_streaming.set(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        contextual_enable_streaming.reset(self._token)


enable_streaming = Streaming()
