"""
The function module provides the wrapper class for creating agentic functions.
"""

import asyncio
import inspect
from types import CodeType, FunctionType
from typing import Any, Callable

from agentica_internal.core.log import LogBase
from agentica_internal.cpython.function import func_arg_names
from agentica_internal.repl import REPL_VAR
from agentica_internal.warpc import forbidden
from agentica_internal.warpc.worlds.sdk_world import SDKWorld

from agentica.agent import DEFAULT_AGENT_LISTENER, DefaultAgentListener
from agentica.client_session_manager import INVALID_UID
from agentica.coming_soon import check_return_type_supported
from agentica.logging.agent_listener import AgentListener

from .agent import Agent
from .common import DEFAULT_AGENT_MODEL, MaxTokens, ModelStrings, Usage

__all__ = ['AgenticFunction']


class AgenticFunction[**I, O](LogBase):
    """
    Wraps a stub Python function to be instead implemented by an LLM.

    The intention is that objects of this class look exactly like the original function,
    and `__call__` dispatches to use the server-side agentic function.
    """

    __USE_CACHED_GLOBALS: bool = True

    # Agentica
    __uid: str
    __init_lock: asyncio.Lock
    __global_scope: dict[str, Any]
    __system: str | None
    __premise: str | None
    __persist: bool
    __persistent_agent: Agent | None
    __model: ModelStrings
    __arg_names: tuple[str, ...]
    __json: bool
    __max_tokens: MaxTokens
    __listener_constructor: Callable[[], AgentListener] | DefaultAgentListener | None
    __logging: bool
    __return_type: type[O]
    __cached_globals_world: SDKWorld | None
    __cached_globals_payload: bytes | None
    _prepare_fn: Callable[[], Any]
    _after_fn: Callable[[], Any]
    __warp_proxy: FunctionType
    # Python
    __wrapped__: Callable[I, O]
    __name__: str
    __qualname__: str
    __doc__: str | None
    __module__: str
    __annotations__: dict[str, Any]
    __defaults__: tuple[Any, ...] | None
    __kwdefaults__: dict[str, Any] | None
    __code__: CodeType
    __signature__: inspect.Signature

    # Usage tracking for a single agentic function
    __total_usage: Usage
    __last_usage: Usage | None
    __usages: dict[str, Usage]

    def __init__(
        self,
        /,
        wrapped: Callable[I, O],
        globals: dict[str, Any],
        json: bool = False,
        system: str | None = None,
        premise: str | None = None,
        persist: bool = False,
        model: ModelStrings = DEFAULT_AGENT_MODEL,
        max_tokens: int | MaxTokens = MaxTokens.default(),
        mcp: str | None = None,
        listener: Callable[[], AgentListener]
        | DefaultAgentListener
        | None = DEFAULT_AGENT_LISTENER,
        logging: bool = False,
    ):
        # Agentica - Initialize LogBase
        super().__init__(logging=logging)
        self.__uid = INVALID_UID
        self.__init_lock = asyncio.Lock()
        self.__global_scope = globals
        self.__system = system
        self.__premise = premise
        self.__persist = persist
        self.__persistent_agent = None
        self.__model = model
        self.__max_tokens = MaxTokens.from_max_tokens(max_tokens)
        self.__arg_names = func_arg_names(wrapped)
        self.__json = json
        self.__mcp = mcp
        self.__listener_constructor = listener
        self.__cached_globals_world = None
        self.__cached_globals_payload = None
        # Python
        self.__wrapped__ = wrapped
        self.__name__ = wrapped.__name__
        self.__qualname__ = wrapped.__qualname__
        self.__doc__ = wrapped.__doc__
        self.__module__ = wrapped.__module__
        self.__annotations__ = wrapped.__annotations__
        if hasattr(wrapped, '__defaults__'):
            self.__defaults__ = wrapped.__defaults__
        if hasattr(wrapped, '__kwdefaults__'):
            self.__kwdefaults__ = wrapped.__kwdefaults__
        if hasattr(wrapped, '__code__'):
            self.__code__ = wrapped.__code__
        if hasattr(wrapped, '__signature__'):
            self.__signature__ = getattr(wrapped, '__signature__')
        else:
            self.__signature__ = wrapped.__signature__ = inspect.signature(wrapped)

        if 'return' not in self.__wrapped__.__annotations__:
            raise ValueError('Your agentic function must be annotated with an explicit return type')
        # Yes, async function have unadulterated `return` type annotations (i.e. not wrapped in `Future` or whatnot).
        self.__return_type = self.__wrapped__.__annotations__['return']
        check_return_type_supported(self.__return_type, mode='json' if self.__json else 'code')

        self.__global_scope[REPL_VAR.SELF_FN] = self.__wrapped__
        self.__global_scope[REPL_VAR.RETURN_TYPE] = self.__return_type
        self.__global_scope[REPL_VAR.ROLE] = 'function'
        self.__global_scope[REPL_VAR.TASK_DESCRIPTION] = _fmt_docstring(self.__doc__)

        if self.__system and self.__premise:
            raise TypeError("`system` and `premise` are mutually exclusive arguments")

        self.__total_usage = Usage(0, 0, 0)
        self.__last_usage = None
        self.__usages = {}

    async def _get_agent(self) -> Agent:
        if self.__persist and self.__persistent_agent is not None:
            return self.__persistent_agent

        world = None
        payload = self.__global_scope
        if self.__USE_CACHED_GLOBALS and self.__cached_globals_world is not None:
            world = self.__cached_globals_world.clone()
            payload = self.__cached_globals_payload

        agent = Agent(
            premise=self.__premise,
            system=self.__system,
            scope=payload,
            mcp=self.__mcp,
            model=self.__model,
            listener=self.__listener_constructor,
            max_tokens=self.__max_tokens,
            _logging=self.logging,
            _world=world,
            _role='function',  # needed to tell ReplAgent which case it is in
        )
        agent.__module__ = self.__module__
        await agent._ensure_init()

        if self.__USE_CACHED_GLOBALS and self.__cached_globals_world is None:
            self.__cached_globals_world = agent._world.clone()
            self.__cached_globals_payload = agent._globals_payload

        if self.__persist:
            self.__persistent_agent = agent

        return agent

    async def __call__(self, *args: I.args, **kwargs: I.kwargs) -> O:
        try:
            bound = self.__signature__.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            raise TypeError(f"Error calling agentic function '{self.__name__}': {e}") from e

        local_scope: dict[str, Any] = dict(bound.arguments)

        agent = await self._get_agent()

        # used for testing
        if prepare_fn := getattr(self, '_prepare_fn', None):
            prepare_fn()

        try:
            return await agent.call(
                self.__return_type,
                _fmt_docstring(self.__doc__),
                mcp=None,
                **local_scope,
            )
        finally:
            # used for testing
            if after_fn := getattr(self, '_after_fn', None):
                after_fn()

            # fetch usage
            try:
                self.__last_usage = agent.last_usage()
                self.__total_usage += self.__last_usage
                assert (iid := agent.last_iid) is not None
                self.__usages[iid] = self.__last_usage
            except:
                pass

            # close non-persistent agent
            if not self.__persist:
                # TODO: decide if we should only enable this during testing
                # i think not, unless we have logic to reset the corresponding sandbox on
                # the server
                await agent.close()

    def last_usage(self) -> Usage:
        if self.__last_usage is None:
            raise ValueError("No call has been made yet")
        return self.__last_usage

    def total_usage(self) -> Usage:
        return self.__total_usage

    def _set_prepare_fn(self, fn: Callable) -> None:
        # used for testing
        assert not hasattr(self, '_prepare_fn')
        self._prepare_fn = fn

    def _set_after_fn(self, fn: Callable) -> None:
        # used for testing
        assert not hasattr(self, '_after_fn')
        self._after_fn = fn

    def ___warp_as___(self):
        if not hasattr(self, '__warp_proxy'):
            from agentica_internal.warpc.magic_utils import create_magic_proxy_function

            self.__warp_proxy = create_magic_proxy_function(self.__wrapped__, self.__call__)
        return self.__warp_proxy


def _fmt_docstring(docstring: str | None) -> str:
    from textwrap import dedent

    if docstring is None:
        return ""
    return dedent(docstring).strip()


forbidden.whitelist_modules('agentica.function')
forbidden.whitelist_objects(
    AgenticFunction,
    AgenticFunction.__call__,
)
