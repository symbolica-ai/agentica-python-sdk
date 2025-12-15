"""
Library which logs agent chat histories and invocations.

Logging enabled by default, but can be disabled by calling `stdout_logging(False)`,
and re-enabled by calling `stdout_logging(True)`.

When logging is enabled, roughly the following will be printed to stdout:

```shell
Spawned Agent 25 (./logs/agent-25.log)
► Agent 25: Get one subagent to work out the 32nd power of 3, then another subagent to work out the 34th power, then return both results.
Spawned Agent 26 (./logs/agent-26.log)
► Agent 26: Work out the 32nd power of 3
◄ Agent 26: 1853020188851841
Spawned Agent 27 (./logs/agent-27.log)
► Agent 27: Work out the 34th power of 3
◄ Agent 27: 16677181699666569
◄ Agent 25: (1853020188851841, 16677181699666569)
```

This gives the user a high-level overview of what's happening in the system,
then the full chat histories for individual agents are written to files, which look roughly like this:

```file: ./logs/agent-26.log
<message role="user">
        Work out the 32nd power of 3

        When you have completed your task or query, return an instance of AgentResult[int].
</message>
<message role="agent">
        <ipython>AgentResult(result=3**32)</ipython>
</message>
```

"""

import asyncio
from contextvars import ContextVar
from typing import Callable
from uuid import uuid4

from .. import flags
from ..client_session_manager import ClientSessionManager
from .agent_logger import AgentLogger, CompositeLogger, NoopLogger
from .loggers.file_logger import FileLogger
from .loggers.print_logger import PrintLogger
from .loggers.standard_logger import StandardLogger

__all__ = [
    'AgentListener',
    'StandardListener',
    'FileOnlyListener',
    'PrintOnlyListener',
    'set_default_agent_listener',
    'get_default_agent_listener',
]


class LoggerToken(str):
    def __new__(cls):
        return super().__new__(cls, uuid4())


class AgentListener:
    logger: AgentLogger
    _previous_loggers: dict[LoggerToken, AgentLogger]
    _listen_task: asyncio.Task[None] | None

    # We need .listen() to block until the connection is actually established
    connected: asyncio.Event

    def __init__(self, logger: AgentLogger) -> None:
        self.logger = logger
        self._listen_task = None
        self._previous_loggers = dict()
        self.connected = asyncio.Event()

    async def listen(self, csm: ClientSessionManager, uid: str, iid: str | None = None) -> None:
        if self._listen_task is not None:
            await self.connected.wait()
            raise AlreadyListening("listen should be called only once")

        # Create listening task on /echo/{uid}/{iid} endpoint
        async def log_chat_history(connected=self.connected) -> None:
            while True:
                async for chunk in csm.echo(uid, iid, connected=connected):
                    # These must be sent sequentially.
                    logger = self.logger
                    await logger.on_chunk(chunk)
                self.connected.clear()

        if self._listen_task is None:
            self._listen_task = asyncio.create_task(log_chat_history())
            await self.connected.wait()
        else:
            raise RuntimeError(
                "previous listen task is still active; listen should be called only once"
            )

    def close(self) -> None:
        if self._listen_task is not None:
            if not self._listen_task.done():
                _ = self._listen_task.cancel()
        self._listen_task = None

    def __del__(self) -> None:
        self.close()

    def add_loggers(self, loggers: list[AgentLogger]) -> LoggerToken:
        if isinstance(self.logger, CompositeLogger):
            logger = CompositeLogger([*self.logger.loggers, *loggers], owned_by_listener=True)
        else:
            logger = CompositeLogger([self.logger, *loggers], owned_by_listener=True)

        token = LoggerToken()
        self._previous_loggers[token] = self.logger

        self.logger = logger
        self.logger.on_spawn()

        return token

    def reset(self, token: LoggerToken) -> None:
        self.logger = self._previous_loggers[token]
        del self._previous_loggers[token]


class AlreadyListening(Exception):
    pass


StandardListener = lambda: AgentListener(StandardLogger())
FileOnlyListener = lambda: AgentListener(FileLogger())
PrintOnlyListener = lambda: AgentListener(PrintLogger())
NoopListener = lambda: AgentListener(NoopLogger())


_default_agent_listener: ContextVar[Callable[[], AgentListener] | None] = ContextVar(
    '_default_agent_listener'
)

GLOBAL_DEFAULT_AGENT_LISTENER = None if flags.in_testing else StandardListener


def set_default_agent_listener(
    listener: Callable[[], AgentListener] | None, local_context: bool = True
) -> None:
    """
    Set the default agent listener for all agents and agentic functions in the current scope.
    This is the StandardListener by default.

    Parameters
    ----------
    listener : Callable[[], AgentListener] | None
        The listener constructor for logging an agent or agentic function's activity and chat history.
        If None, no listener will be used.
    local_context : bool
        If True, is only for this Context.

    Returns
    -------
    None

    Note
    ----
    The use of context-specific loggers will override the default agent listener. See `spawn` and `agentic` for more details.
    """
    if local_context:
        _ = _default_agent_listener.set(listener)
    else:
        global GLOBAL_DEFAULT_AGENT_LISTENER
        GLOBAL_DEFAULT_AGENT_LISTENER = listener


def get_default_agent_listener() -> Callable[[], AgentListener] | None:
    return _default_agent_listener.get(GLOBAL_DEFAULT_AGENT_LISTENER)
