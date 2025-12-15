"""
The agent module provides the `Agent` class for interacting with Agentica's agents.

Agents are AI agents that can execute Python code and use tools to accomplish tasks. They can:
- Execute arbitrary Python code in a sandboxed environment
- Return strongly typed results
- Stream their responses in real-time
- Use provided tools and functions
- Maintain conversation context

The main class is `Agent` which provides methods to:
- Create and initialize agents
- Execute tasks with or without return types
- Stream agent responses
- Provide tools and MCP configurations
- Control agent behaviour and execution

Example:
    ```python
    agent = await spawn("Help me analyze data")
    result = await agent(list[str], "Extract keywords from the text", text="...")
    ```

See the `Agent` class documentation for detailed usage information.
"""

import asyncio
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from types import NoneType
from typing import Any, Callable, Literal, Self, overload

from agentica_internal.core.anno import is_anno
from agentica_internal.core.log import LogBase
from agentica_internal.core.result import ERRORED
from agentica_internal.core.utils import copy_doc, unreachable
from agentica_internal.repl import REPL_VAR
from agentica_internal.session_manager_messages import CreateAgentRequest
from agentica_internal.warpc import forbidden
from agentica_internal.warpc.worlds.sdk_world import SDKWorld
from typing_extensions import deprecated

from agentica.agentica_client.global_csm import current_global_csm, get_global_csm
from agentica.client_session_manager import INVALID_UID
from agentica.client_session_manager.client_session_manager import AgentInvocationHandle
from agentica.coming_soon import (
    check_return_type_supported,
    check_value_supported,
)
from agentica.errors import InvocationError
from agentica.logging.agent_listener import (
    AgentListener,
    AlreadyListening,
    LoggerToken,
    get_default_agent_listener,
)
from agentica.logging.agent_logger import (
    CompositeLogger,
    contextual_enable_streaming,
    contextual_logger,
)
from agentica.std.utils import spoof_signature
from agentica.template import maybe_prompt_template
from agentica.unmcp.mcp_function import MCPFunction

from .common import (
    DEFAULT_AGENT_MODEL,
    MaxTokens,
    ModelStrings,
    Usage,
    _find_parent_invocation,
    _find_parent_local_id,
    _reset_current_invocation,
    _reset_current_local_id,
    _set_current_invocation,
    _set_current_local_id,
)

__all__ = ['Agent', 'DEFAULT_AGENT_LISTENER']


class DefaultAgentListener: ...


DEFAULT_AGENT_LISTENER = DefaultAgentListener()


class Agent(LogBase):
    """
    An instance of this class is an individual long lived agent which can be
    invoked multiple times via `await agent.call(return_type, 'instructions')`.
    """

    # Agentica
    __uid: str
    __last_iid: str | None
    __last_total: Usage | None
    __usages: dict[str, Usage]  # iid -> usage
    __init_lock: asyncio.Lock
    __global_scope: dict[str, Any]
    __mcp: str | None
    __json: bool
    __model: ModelStrings
    __max_tokens: MaxTokens
    __logging: bool
    _listener: AgentListener | None
    _prepare_fn: Callable[[], Any]
    _after_fn: Callable[[], Any]
    __listener_constructor: Callable[[], AgentListener] | None
    _world: SDKWorld
    _globals_payload: bytes | None
    __name: str
    __for_function: bool = False

    # Python
    __doc__: str | None
    __system__: str | None
    __module__: str

    # fmt: off
    def __init__(
        self,
        premise: str | None = None,
        scope: dict[str, Any] | bytes | None = None,
        *,
        system: str | None = None,
        mcp: str | None = None,
        model: ModelStrings = DEFAULT_AGENT_MODEL,
        # fmt off because pyright needs this to be on one line
        listener: Callable[[], AgentListener] | DefaultAgentListener | None = DEFAULT_AGENT_LISTENER,
        max_tokens: int | MaxTokens = MaxTokens.default(),
        _world: SDKWorld | None = None,
        _logging: bool = False,
        _call_depth: int = 0,
        _role: Literal['function', 'agent'] = 'agent',
    ):
        """
        Spawn a new agent.

        Parameters
        ----------
        premise : str | None
            An optional premise for the agent.
            This will be attached to the system prompt of all invocations of this agent.
            This argument cannot be provided along with the `system` argument.
        scope : dict[str, Any]
            An optional default set of resources which the agent will have access to indefinitely.
            Resources in scope may be arbitrary Python functions, methods, objects, iterators, types or any other Python value.
            These resources may additionally be specified per invocation later on.
        system : str | None
            An optional system prompt for the agent.
            This will be the system prompt of all invocations of this agent.
            This argument cannot be provided along with the `premise` argument.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used
            during the execution of the agent.
        model : str
            The model which backs your agent.
            One of:
            - 'openai:gpt-3.5-turbo'
            - 'openai:gpt-4o'
            - 'openai:gpt-4.1'
            - 'openai:gpt-5'
            - 'anthropic:claude-sonnet-4'
            - 'anthropic:claude-opus-4.1'
            - 'anthropic:claude-sonnet-4.5'
            - 'anthropic:claude-opus-4.5'
            or any OpenRouter model slug.
        listener : Callable[[], AgentListener] | None
            Optional listener constructor for logging the agent's activity and chat history.
            If None, no listener will be used.
        max_tokens : int | MaxTokens
            When an integer is supplied, this is the maximum number of tokens for an invocation.
            For more fine-grained control, a `MaxTokens` object may be passed.

        Note
        ----
        The default agent listener is the StandardListener, but can be changed for all agents and agentic functions in the current scope with `set_default_agent_listener`.
        If a context-specific logger is used in the current scope, the logger will be added to the listener: if the listener is None, then the listener will be set to
        - the default agent listener, if it is not None, or
        - the StandardListener, if the default agent listener is None
        """
        # Agentica - Initialize LogBase
        super().__init__(logging=_logging)
        self.__listener_constructor = (
            get_default_agent_listener() if isinstance(listener, DefaultAgentListener) else listener
        )
        if self.__listener_constructor is not None:
            self._listener = self.__listener_constructor()
            self._listener.logger.on_spawn()
        else:
            self._listener = None
        self.__uid = INVALID_UID
        self.__last_iid = None
        self.__last_total = None
        self.__usages = {}
        self.__init_lock = asyncio.Lock()
        # this is needed because functions *are* just agents, but
        # agents don't take locals (and in fact send their globals AS locals),
        # whereas functions do,so AgentRepl needs
        # to decide to swap globals and locals *only when it's an agent*,
        # and we must tell it which case it is in. why are they mixed up like this?
        if isinstance(scope, bytes):
            # this COULD imply we're from a function, but a function can also
            # pass just a dict, too
            self._globals_payload = scope
            self.__global_scope = dict()
        elif isinstance(scope, dict):
            # yuck, but this is already so complicated...
            # we absolutely should rethink this, and probably just serialize
            # a special MagicFrame(...) that says: what are your locals, globals,
            # constants, and desired return type. that's actually what InitMsg
            # is already almost equivalent to!
            self.__global_scope = scope | {REPL_VAR.ROLE: _role}
            self._globals_payload = None
        else:
            self.__global_scope = {REPL_VAR.ROLE: _role}
            # todo: issue a message here!
            self._globals_payload = None
        self.__json = False
        self.__model = model
        self.__max_tokens = MaxTokens.from_max_tokens(max_tokens)
        self.__mcp = mcp
        self._world = _world or SDKWorld(logging=_logging)
        self.__name = premise[:10] + '...' if premise else 'anonymous'
        # Python
        self.__doc__ = premise
        if system and premise:
            raise ValueError(
                "Providing a system prompt and premise is not supported. Please provide only one."
            )
        self.__system__ = system
        self.__module__ = sys._getframe(_call_depth + 1).f_globals["__name__"]
        for k, v in self.__global_scope.items():
            if k.startswith('_') and not k.startswith('__'):
                raise ValueError(f"Global scope key {k} cannot start with an underscore")
            try:
                check_value_supported(v, mode='json' if self.__json else 'code')
            except Exception as e:
                e.add_note(f"{k} is not supported")
                raise e

    def __repr__(self) -> str:
        return object.__repr__(self)

    @copy_doc(__init__)
    @classmethod
    async def spawn(
        cls: type[Self],
        premise: str | None = None,
        scope: dict[str, Any] | None = None,
        *,
        system: str | None = None,
        mcp: str | None = None,
        model: ModelStrings = DEFAULT_AGENT_MODEL,
        max_tokens: int | MaxTokens = MaxTokens.default(),
        listener: Callable[[], AgentListener]
        | DefaultAgentListener
        | None = DEFAULT_AGENT_LISTENER,
        _logging: bool = False,
        _call_depth: int = 0,
    ) -> Self:
        """
        Spawn an agent.

        Parameters
        ----------
        premise : str or None
            An optional premise for the agent.
            This will be attached to the system prompt of all invocations of this agent.
            This argument cannot be provided along with the `system` argument.
        scope : dict[str, Any]
            An optional default set of resources which the agent will have access to indefinitely.
            Resources in scope may be arbitrary Python functions, methods, objects, iterators, types or any other Python value.
            These resources may additionally be specified per invocation later on.
        system : str or None
            An optional system prompt for the agent.
            This will be the system prompt of all invocations of this agent.
            This argument cannot be provided along with the `premise` argument.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used
            during the execution of the agent.
        model : str
            The model which backs your agent.
            One of:
            - 'openai:gpt-3.5-turbo'
            - 'openai:gpt-4o'
            - 'openai:gpt-4.1'
            - 'openai:gpt-5'
            - 'anthropic:claude-sonnet-4'
            - 'anthropic:claude-opus-4.1'
            - 'anthropic:claude-sonnet-4.5'
            - 'anthropic:claude-opus-4.5'
            or any OpenRouter model slug.
        listener : Callable[[], AgentListener] | None
            Optional listener constructor for logging the agent's activity and chat history.
            If None, no listener will be used.
        max_tokens : int | MaxTokens
            When an integer is supplied, this is the maximum number of tokens for an invocation.
            For more fine-grained control, a `MaxTokens` object may be passed.

        Note
        ----
        The default agent listener is the StandardListener, but can be changed for all agents and agentic functions in the current scope with `set_default_agent_listener`.
        If a context-specific logger is used in the current scope, the logger will be added to the listener: if the listener is None, then the listener will be set to
        - the default agent listener, if it is not None, or
        - the StandardListener, if the default agent listener is None
        """
        self = cls(
            premise,
            scope,
            system=system,
            model=model,
            max_tokens=max_tokens,
            mcp=mcp,
            listener=listener,
            _logging=_logging,
            _call_depth=_call_depth + 1,
        )
        await self._ensure_init()
        return self

    def __short_str__(self) -> str:
        return f"ClientAgent[{self.__name!r}]"

    async def _ensure_init(self) -> None:
        csm = await get_global_csm()
        if csm.uid_resource_exists(self.__uid):
            return

        async with self.__init_lock:
            # if a globals payload already exists, skip creating a new one
            if self._globals_payload is None:
                if self.__mcp:
                    mcp_functions = await MCPFunction.from_json(self.__mcp)
                    self.__global_scope.update({f.__name__: f.__wrapped__ for f in mcp_functions})
                self._globals_payload = self._world.to_payload(self.__global_scope)

            if self.__uid == INVALID_UID:
                csm = await get_global_csm()
                cmar = CreateAgentRequest(
                    doc=self.__doc__,
                    system=maybe_prompt_template(self.__system__) if self.__system__ else None,
                    warp_globals_payload=self._globals_payload,
                    json=self.__json,
                    model=self.__model,
                    streaming=False,
                    max_tokens_per_invocation=self.__max_tokens.per_invocation,
                    max_tokens_per_round=self.__max_tokens.per_round,
                    max_rounds=self.__max_tokens.rounds,
                )
                self.log(f"Creating agent {cmar=}")
                self.__uid = await csm.new_agent(cmar)

    @property
    def last_iid(self) -> str | None:
        return self.__last_iid

    def set_listener(self, listener: AgentListener) -> None:
        if self._listener is not None:
            self._listener.close()
        self._listener = listener
        self._listener.logger.on_spawn()


    @staticmethod
    def _before_after[O](f: Callable[..., Awaitable[O]]) -> Callable[..., Awaitable[O]]:
        async def wrapper_before_after(self: 'Agent', *args: Any, **kwargs: Any) -> O:
            token = await self._before()
            try:
                return await f(self, *args, **kwargs)
            finally:
                await self._after(token)

        spoof_signature(wrapper_before_after, f)
        return wrapper_before_after

    async def _before(self) -> LoggerToken | None:
        await self._ensure_init()

        # If there are contextual loggers, add them to the listener
        loggers = contextual_logger.get()
        token = None

        loggers = [logger.clone() for logger in loggers] if loggers else None

        if self._listener is None:
            # Need to create the listener for the first time
            default_listener = get_default_agent_listener()
            # Check the default agent listener
            if default_listener is not None:
                self._listener = default_listener()
                if loggers:
                    self._listener.add_loggers(loggers)
            else:
                if loggers:
                    self._listener = AgentListener(CompositeLogger(loggers, owned_by_listener=True))

            if self._listener is not None:
                self._listener.logger.on_spawn()
        else:
            # Otherwise, just add them
            token = self._listener.add_loggers(loggers or [])

        try:
            if self._listener is not None:
                # Wait for the listener to be connected and listening to the right UID.
                await self._listener.listen(await get_global_csm(), self.__uid)
        except AlreadyListening:
            pass

        return token

    async def _after(self, token: LoggerToken | None) -> None:
        # Reset the listener state
        match self._listener, token:
            case None, None:
                # No listener and no token, nothing to do
                pass
            case _, None:
                # Listener was set, but no token!
                # This means the listener was created for the first time
                # To reset to the previous state, we must destroy the listener
                self._listener.close()
                del self._listener
                self._listener = None
            case None, _:
                # Listener was None, but somehow got a token...
                unreachable()
            case _, _:
                # Listener was set, and has a token
                self._listener.reset(token)

        # Now register the usage after the invocation
        if self.__last_iid is not None:
            try:
                _ = await self._fetch_usage(self.__last_iid)
            except:
                pass

    # __call__

    @overload
    async def call(self, task: str, /, mcp: str | None = None, **scope: Any) -> str:
        """
        Invokes and agent with string return type. This is the final response from the agent.

        Parameters
        ----------
        task : str
            The agent's task (or objective) for this invocation of the agent.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used during the agent's run.
        scope : dict[str, Any]
            Any additional resources added to the agent's scope for this invocation.

        Returns
        -------
        Awaitable[str]
            An awaitable result of type `str` which the agent returns.
        """

    @overload
    async def call(
        self, return_type: None, task: str, /, mcp: str | None = None, **scope: Any
    ) -> None:
        """
        Invokes and agent.

        This is with a `None` return type, useful when you do not care about the result, only the side effects.

        Parameters
        ----------
        return_type : None
            The return type is `None`, indicating that the agent will not return a result.
        task : str
            The agent's task (or objective) for this invocation of the agent.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used during the agent's run.
        scope : dict[str, Any]
            Any additional resources added to the agent's scope for this invocation.

        Returns
        -------
        Awaitable[None]
            An awaitable None which finishes when the agent has completed.
        """
        ...

    @overload
    async def call[T](
        self, return_type: type[T], task: str, /, mcp: str | None = None, **scope: Any
    ) -> T:
        """
        Invokes and agent with arbitrary return type.

        Parameters
        ----------
        return_type : type[T]
            Provide a return type for the agent to have it return an instance of a specific type `T`.
        task : str
            The agent's task (or objective) for this invocation of the agent.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used during the agent's run.
        scope : dict[str, Any]
            Any additional resources added to the agent's scope for this invocation.

        Returns
        -------
        Awaitable[T]
            An awaitable result of type `T` which the agent returns.
        """
        ...

    @_before_after
    async def call[T](
        self,
        fst: str | type[T] | None,
        snd: str | None = None,
        /,
        mcp: str | None = None,
        **scope: Any,
    ) -> T | str | None:
        """
        Invokes and agent with arbitrary return type.

        Parameters
        ----------
        return_type : type[T]
            Provide a return type for the agent to have it return an instance of a specific type `T`.
        task : str
            The agent's task (or objective) for this invocation of the agent.
        mcp : str or None
            The string of a path to a .json file representing an MCP configuration.
            Any servers and/or tools of servers outlined in the config can be used during the agent's run.
        scope : dict[str, Any]
            Any additional resources added to the agent's scope for this invocation.

        Returns
        -------
        Awaitable[T]
            An awaitable result of type `T` which the agent returns.
        """
        return_type: type[T] | type[None] | type[str]
        task_desc: str
        local_scope = scope

        if is_anno(fst) or fst is None:
            if not isinstance(snd, str):
                raise ValueError(
                    "Second argument must be string task description if first argument is type"
                )
            assert not isinstance(fst, str)
            if fst is None:
                # `None` really means `NoneType`
                return_type = NoneType
            else:
                return_type = fst

            task_desc = snd
        elif isinstance(fst, str):
            task_desc = fst
            return_type = str
        else:
            raise ValueError(f"Invalid first argument: {fst}")

        if mcp:
            mcp_functions = await MCPFunction.from_json(mcp)
            local_scope.update({f.__name__: f.__wrapped__ for f in mcp_functions})

        # used for testing
        if prepare_fn := getattr(self, '_prepare_fn', None):
            prepare_fn()

        should_stream = False
        if self._listener is not None:
            should_stream = self._listener.logger.should_stream()
        invocation = await self._get_invocation(
            task_desc, return_type, local_scope, streaming=should_stream
        )

        local_id_token = _set_current_local_id(
            self._listener.logger.local_id if self._listener is not None else None
        )
        try:
            return await invocation.invoke()
        finally:
            _reset_current_local_id(local_id_token)
            if after_fn := getattr(self, '_after_fn', None):
                after_fn()

    # backwards compatibility
    __call__ = deprecated("use explicit .call() instead")(call)

    # getting usage

    async def _fetch_usage(self, iid: str) -> Usage:
        csm = current_global_csm()
        if csm is None:
            raise ValueError("Called before client session manager was created")
        # Fetch logs for this invocation and and sum usages
        total = Usage(0, 0, 0)
        logs = await csm.logs(self.__uid, iid, {'type': 'sm_inference_usage'})
        for log in logs:
            # Additional filter to ensure we only process usage logs
            if 'usage' in log:
                total += Usage.from_completions(log['usage'], last_usage=total)

        usage = total
        if self.__last_total is not None:
            usage -= self.__last_total.replace(output_tokens=0)
        self.__usages[iid] = usage
        self.__last_total = total
        return usage

    def last_usage(self) -> Usage:
        """Get the usage for the last invocation."""
        if self.__last_iid is None:
            raise ValueError("No invocation has been made yet")
        if self.__last_iid not in self.__usages:
            raise ValueError(f"Usage not found for invocation {self.__last_iid}")
        return self.__usages[self.__last_iid]

    def total_usage(self) -> Usage:
        """Get the total usage across all invocations."""
        return sum(self.__usages.values(), Usage(0, 0, 0))

    async def _get_invocation(
        self,
        task_desc: str,
        return_type: type,
        tools: dict[str, Any],
        *,
        streaming: bool,
    ) -> "_AgentInvocation":
        self.log("_get_invocation()")

        mode = 'json' if self.__json else 'code'
        for name, tool in tools.items():
            if name.startswith('_'):
                raise ValueError(f"Globals name {name} cannot start with an underscore")
            check_value_supported(tool, mode)

        tools[REPL_VAR.RETURN_TYPE] = return_type
        tools[REPL_VAR.TASK_DESCRIPTION] = str(task_desc)

        csm = await get_global_csm()
        check_return_type_supported(return_type, mode)

        # Insert return type into scope if not already present
        is_plain_return_type = isinstance(return_type, type)
        if is_plain_return_type:
            name = getattr(return_type, '__name__', None)
            module = getattr(return_type, '__module__', '')
            have_return_type_already = (
                any(t is return_type for t in tools.values()) or name in tools
            )
            is_builtin = module.startswith('builtins')
            if name and not have_return_type_already and not is_builtin:
                tools[name] = return_type

        tools["__return_type"] = return_type

        self.log("Setting up world")
        locals_payload = self._world.to_payload(tools)

        self.log(f"csm.invoke_agent({self.__uid!r}, task_desc={task_desc!r}, ...)")
        parent_uid, parent_iid = _find_parent_invocation()
        if self._listener is not None:
            parent_id = _find_parent_local_id()
            self._listener.logger.on_call_enter(task_desc, parent_id)
        mai_handle = await csm.invoke_agent(
            uid=self.__uid,
            parent_uid=parent_uid,
            parent_iid=parent_iid,
            warp_locals_payload=locals_payload,
            task_desc=task_desc,
            streaming=streaming or contextual_enable_streaming.get(),
        )
        self.__last_iid = mai_handle.iid
        self.log(f"iid={mai_handle.iid!r}")
        return _AgentInvocation(self.__uid, mai_handle.iid, self._world, mai_handle, self.log, self._listener)

    async def close(self) -> None:
        # global listener cleanup
        if hasattr(self, '_listener'):
            if self._listener is not None:
                self._listener.close()

        # Use close_agent which handles cleanup + destroy
        if csm := current_global_csm():
            await csm.close_agent(self.__uid)

        try:
            if hasattr(self, '_world'):
                self._world.close()
        except:
            pass

    def sync_close(self) -> None:
        # try to send destroy to session manager
        try:
            loop = asyncio.get_event_loop()
            coro = self.close()
            if loop.is_running():
                loop.create_task(coro, name=f"Agent<{self.__uid}>.close")
            else:
                loop.run_until_complete(coro)
        except:
            pass


    def __del__(self) -> None:
        self.sync_close()

    def _set_prepare_fn(self, fn: Callable) -> None:
        # used for testing
        assert not hasattr(self, '_prepare_fn')
        self._prepare_fn = fn

    def _set_after_fn(self, fn: Callable) -> None:
        # used for testing
        assert not hasattr(self, '_after_fn')
        self._after_fn = fn

    def ___warp_as___(self):
        # create minimal agent class that wraps the real agent
        fake_self = object.__new__(_Agent)
        fake_self.___real_self___ = self
        return fake_self

    @classmethod
    def ___class_warp_as___(cls):
        return _Agent


@dataclass
class _AgentInvocation:
    uid: str
    iid: str
    world: SDKWorld
    handle: AgentInvocationHandle
    log: Callable[..., None]
    listener: AgentListener | None

    async def invoke(self) -> Any:
        self.log(f"Agent.invoke({self.uid!r}, {self.iid!r})")

        try:
            self.log("getting event loop")
            loop = asyncio.get_running_loop()
            self.log(f"event_loop: {loop} @ {id(loop)}")
        except RuntimeError:
            self.log("no event loop")
            raise InvocationError(
                "No event loop is running; use asyncio.run() to call async agents"
            )

        invocation_token = _set_current_invocation(self.uid, self.iid)

        self.log("world.start_msg_loop")
        future = self.world.create_future(self.iid, self.handle.future_result)
        task = self.world.start_msg_loop(self.handle.send_message, self.handle.recv_message, future)
        result = ERRORED

        try:
            await future
            result = future.result()
            return result

        finally:
            _reset_current_invocation(invocation_token)

            if self.listener is not None:
                self.log('calling listener on_call_exit')
                self.listener.logger.on_call_exit(result)

            if not future.done():
                self.log('cancelling result future')
                try:
                    future.cancel()
                except:
                    pass

            if not task.done():
                self.log('cancelling msg_loop task')
                try:
                    task.cancel()
                except:
                    pass


# create minimal agent class that wraps the real agent
class _Agent:
    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError("You cannot create your own Agents.")

    async def call[T](self, return_type: type[T], task: str, /, **scope: Any) -> T:
        return await self.___real_self___.call(return_type, task, **scope)


def _fake_agent_names():
    # so that it can be warped
    mod_name = 'builtins'

    _Agent.__doc__ = Agent.__doc__
    _Agent.__name__ = Agent.__name__
    _Agent.__qualname__ = Agent.__qualname__
    _Agent.__init__.__module__ = mod_name
    _Agent.__static_attributes__ = ()
    _Agent.__module__ = mod_name

    _Agent.__init__.__doc__ = Agent.__init__.__doc__
    _Agent.__init__.__qualname__ = Agent.__init__.__qualname__
    _Agent.__init__.__module__ = mod_name

    _Agent.call.__doc__ = Agent.call.__doc__
    _Agent.call.__qualname__ = Agent.call.__qualname__
    _Agent.call.__module__ = mod_name


_fake_agent_names()
del _fake_agent_names

forbidden.whitelist_modules('agentica.agent')
forbidden.whitelist_objects(
    Agent.call,
    Agent.spawn,
    _Agent,
    _Agent.__init__,
    _Agent.call,
)


@copy_doc(Agent.__init__)
async def spawn(
    premise: str | None = None,
    scope: dict[str, Any] | None = None,
    *,
    system: str | None = None,
    mcp: str | None = None,
    model: ModelStrings = DEFAULT_AGENT_MODEL,
    listener: Callable[[], AgentListener] | DefaultAgentListener | None = DEFAULT_AGENT_LISTENER,
    max_tokens: int | MaxTokens = MaxTokens.default(),
    _logging: bool = False,
    _call_depth: int = 0,
) -> Agent:
    """
    Spawn a new agent.

    Parameters
    ----------
    premise : str or None
        An optional premise for the agent.
        This will be attached to the system prompt of all invocations of this agent.
        This argument cannot be provided along with the `system` argument.
    scope : dict[str, Any]
        An optional default set of resources which the agent will have access to indefinitely.
        Resources in scope may be arbitrary Python functions, methods, objects, iterators, types or any other Python value.
        These resources may additionally be specified per invocation later on.
    system : str or None
        An optional system prompt for the agent.
        This will be the system prompt of all invocations of this agent.
        This argument cannot be provided along with the `premise` argument.
    mcp : str or None
        The string of a path to a .json file representing an MCP configuration.
        Any servers and/or tools of servers outlined in the config can be used
        during the execution of the agent.
    model : str
        The model which backs your agent.
        One of:
        - 'openai:gpt-3.5-turbo'
        - 'openai:gpt-4o'
        - 'openai:gpt-4.1'
        - 'openai:gpt-5'
        - 'anthropic:claude-sonnet-4'
        - 'anthropic:claude-opus-4.1'
        - 'anthropic:claude-sonnet-4.5'
        - 'anthropic:claude-opus-4.5'
        or any OpenRouter model slug.
    listener : Callable[[], AgentListener] | None
        Optional listener constructor for logging the agent's activity and chat history.
        If None, no listener will be used.
    max_tokens : int | MaxTokens
        When an integer is supplied, this is the maximum number of tokens for an invocation.
        For more fine-grained control, a `MaxTokens` object may be passed.

    Returns
    -------
    Agent
        An agent object.

    Note
    ----
    The default agent listener is the StandardListener, but can be changed for all agents and agentic functions in the current scope with `set_default_agent_listener`.
    If a context-specific logger is used in the current scope, the logger will be added to the listener: if the listener is None, then the listener will be set to
    - the default agent listener, if it is not None, or
    - the StandardListener, if the default agent listener is None
    """
    agent = await Agent.spawn(
        premise,
        scope,
        system=system,
        mcp=mcp,
        model=model,
        max_tokens=max_tokens,
        listener=listener,
        _logging=_logging,
        _call_depth=_call_depth + 1,
    )
    return agent
