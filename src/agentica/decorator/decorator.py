"""
The decorator module provides the `agentic` decorator for creating LLM-implemented functions.
"""

import ast
import inspect
import textwrap
import types
from collections.abc import Callable
from inspect import iscoroutinefunction
from logging import getLogger
from types import FunctionType, MethodType
from typing import Any

from agentica_internal.warpc import forbidden

from agentica.agent import DEFAULT_AGENT_LISTENER, DefaultAgentListener
from agentica.coming_soon import ComingSoon, Feature, check_value_supported
from agentica.common import DEFAULT_AGENT_MODEL, MaxTokens
from agentica.function import AgenticFunction, ModelStrings
from agentica.logging.agent_listener import AgentListener

logger = getLogger(__name__)


__all__ = ["agentic"]


def agentic[**I, O](
    *scope_defined: Any,
    scope: dict[str, Any] | None = None,
    system: str | None = None,
    premise: str | None = None,
    mcp: str | None = None,
    persist: bool = False,
    model: ModelStrings = DEFAULT_AGENT_MODEL,
    listener: Callable[[], AgentListener] | DefaultAgentListener | None = DEFAULT_AGENT_LISTENER,
    max_tokens: int | MaxTokens = MaxTokens.default(),
    _logging: bool = False,
) -> Callable[[Callable[I, O]], Callable[I, O]]:
    """
    Decorator for creating LLM-implemented functions.

    This decorator is used on functions that will be implemented by an LLM.
    The decorated function should have a descriptive docstring but an empty
    body (containing only ``...``).

    Parameters
    ----------
    *scope_defined: Any
        A list of runtime resources as in `scope`. The names of the resources are not specified explicitly and are instead derived automatically from the resources themselves.
        `scope` and `scope_defined` can be used together to specify resources with both explicit and implicit names. The names can't be repeated between the two.
        Example:
        ```
        @agentic(my_func, db_connection, cache) # the same as @agentic(scope={"my_func": my_func, "db_connection": db_connection, "cache": cache})
        async def my_function():
            ...
        ```
    scope : dict[str, Any]
        A dictionary of names mapped to runtime resources that are in scope and which may be used during the execution of the agentic function.
        Resources in scope may be arbitrary Python functions, methods, objects, iterators, types or any other Python value.
    system : str | None
        An optional system prompt for the agentic function.
        This will be the system prompt of all invocations of this agentic function.
        This argument cannot be provided along with the `premise` argument.
    premise : str | None
        An optional premise for the function.
        This will be attached to the system prompt of all invocations of this agentic function.
        This argument cannot be provided along with the `system` argument.
    mcp : str | None
        The string of a path to a .json file representing an MCP configuration.
        Any servers and/or tools of servers outlined in the config can be used during the execution of the agentic function.
    persist : bool
        Whether to persist the function state/history between calls.
    model : str
        The model used to execute the agentic function.
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
            Optional listener constructor for logging the agentic function's activity and chat history.
            If None, no listener will be used.
    max_tokens : int | MaxTokens
        When an integer is supplied, this is the maximum number of tokens for an invocation.
        For more fine-grained control, a `MaxTokens` object may be passed.

    Returns
    -------
    Callable[[Callable[I, O]], Callable[I, O]]
        The decorated function that will be implemented by the LLM.

    Note
    ----
    The default agent listener is the StandardListener, but can be changed for all agents and agentic functions in the current scope with `set_default_agent_listener`.
    If a context-specific logger is used in the current scope, the logger will be added to the listener: if the listener is None, then the listener will be set to
    - the default agent listener, if it is not None, or
    - the StandardListener, if the default agent listener is None

    """

    if system and premise:
        raise TypeError("`system` and `premise` are mutually exclusive arguments")

    def decorator(func: Callable[I, O]) -> Callable[I, O]:
        assert isinstance(func, FunctionType)

        if not iscoroutinefunction(func):
            raise ComingSoon(Feature.SYNC_COMPATIBLE, 'code')

        actual_scope = gather_scope(func, scope_defined, scope)
        for obj in actual_scope.values():
            check_value_supported(obj)

        mf = AgenticFunction(
            wrapped=func,
            globals=actual_scope,
            system=system,
            premise=premise,
            persist=persist,
            model=model,
            listener=listener,
            max_tokens=max_tokens,
            logging=_logging,
            mcp=mcp,
        )

        return mf

    return decorator


def gather_scope(
    func: FunctionType, scope_defined: tuple[Any, ...], scope: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Extract scope resources from positional arguments passed to @agentic(name1, name2, ...).

    For each resource, we first try to get its name from `__qualname__` or `__name__`
    (works for functions, classes, etc.). If that fails, we fall back to AST parsing
    to extract the *syntactic* variable name from the decorator call.

    Note: AST parsing only works when using decorator syntax. When @agentic is called
    explicitly as a function (e.g. `agentic(x)(fn)`), AST parsing won't work and the
    resource must have `__name__`/`__qualname__` or be specified via the `scope` kwarg.
    """
    dct = dict[str, Any]()
    func_name = func.__name__

    for i, resource in enumerate(scope_defined):
        key = get_resource_key(resource, i, func_name, func)
        if key is None:
            raise TypeError(
                f"@agentic function {func_name!r} was provided a resource that has no __qualname__ and no __name__: {resource!r}. Specify it in the `scope` parameter instead."
            )

        validate_scope_key(key, func_name)

        # Not the most robust way to check
        if key in dct or (scope is not None and key in scope):
            raise TypeError(
                f"@agentic function {func_name!r} was provided a resource with the same name twice: '{key}'."
            )

        dct[key] = resource

    if scope is not None:
        if not isinstance(scope, dict):
            raise TypeError(
                f"@agentic function {func_name!r} was provided `scope` of type {type(scope).__name__}. Expected dict[str, Any]."
            )

        bad_keys = [k for k in scope if not isinstance(k, str)]
        if bad_keys:
            raise TypeError(
                f"@agentic function {func_name!r} was provided `scope` that contains non-string keys: {bad_keys!r}. Only strings are expected as keys."
            )
        dct.update(scope)

    return dct


def validate_scope_key(key: str, func_name: str) -> None:
    if key.startswith('_'):
        raise TypeError(
            f"@agentic function {func_name!r} scope resource {key!r} cannot start with an underscore"
        )


def try_parse_variable_name(func: FunctionType, index: int) -> str | None:
    """
    Parse the variable name of a resource at the given index in the decorator arguments.
    Uses AST parsing to extract the variable name from the decorator call. Note: only works when used as a decorator, not when called directly as a function.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except Exception:
        return None
    if len(tree.body) != 1:
        return None
    func_def = tree.body[0]
    if not isinstance(func_def, ast.AsyncFunctionDef):
        return None
    for dec_node in func_def.decorator_list:
        match dec_node:
            case ast.Call(func=ast.Name(id="agentic"), args=agentic_args):
                agentic_arg = agentic_args[index]
                # ast.Name is defined to be a variable name, so this matches exactly what we need and nothing else.
                match agentic_arg:
                    case ast.Name(id=variable_name):
                        return variable_name
    return None


def try_get_resource_key_for_method(
    value: Any,
    index: int,
    func_name: str,
    func: FunctionType,
) -> str | None:
    cls = type(value)
    if cls is MethodType:
        # devise a reasonable name for bound instance and class methods
        self = value.__self__
        method_func = value.__func__
        method_name = method_func.__name__
        if method_name == "__new__":
            raise TypeError(
                f"@agentic function {func_name!r} was provided a resource that is user-defined __new__ of {cls.__name__}. Specify it in the `scope` parameter instead."
            )

        if isinstance(self, type):
            # classmethod: self is the class
            class_name = self.__qualname__
            # MyClass.foo -> MyClass_foo
            return f'{class_name}_{method_name}'
        else:
            # Bound instance method: self is an instance
            class_name = type(self).__qualname__
            # MyClass().foo -> 'bound_MyClass_foo'
            return f'bound_{class_name}_{method_name}'
    return None


def get_resource_key_impl(
    value: Any,
    index: int,
    func_name: str,
    func: FunctionType,
) -> str | None:
    # It's just too complex to handle it properly (for example __new__ causes a lot problems), so we just raise an error
    if type(value) is types.BuiltinFunctionType:
        raise TypeError(
            f"@agentic function {func_name!r} was provided a resource that is builtin: {value}. Specify it in the `scope` parameter instead."
        )

    method_key = try_get_resource_key_for_method(value, index, func_name, func)
    if method_key is not None:
        return method_key

    name = getattr(value, '__qualname__', None)
    if name is not None:
        # lambdas are a special case of an object with __qualname__ defined
        # if `value` syntactically is a variable name referencing a lambda, we will parse it correctly
        # if `value` syntactically is a lambda, we will return None here and raise, user should use `scope` instead
        if name.endswith('<lambda>'):
            return try_parse_variable_name(func, index)
        if type(name) is str:
            return name

    name = getattr(value, '__name__', None)
    if type(name) is str:
        return name

    # Fallback to AST parsing to parse variable names correctly
    variable_name = try_parse_variable_name(func, index)
    if variable_name is not None:
        return variable_name

    if name is None:
        return None

    # This basically means that the user provided a value with __name__ (or __qualname__) defined as a non-string
    cls_name = type(value).__name__
    msg = f"@agentic function {func_name!r} was provided invalid global resource of type {cls_name!r}."
    raise TypeError(msg)


def niceify_resource_key(key: str) -> str:
    # If <locals> is present in the name, it means the value specified has a function scope, just drop the <locals> part
    key = key.replace('.<locals>.', '_').replace('.', '_')
    return key


def get_resource_key(
    value: Any,
    index: int,
    func_name: str,
    func: FunctionType,
) -> str | None:
    key = get_resource_key_impl(value, index, func_name, func)
    if key is not None:
        key = niceify_resource_key(key)

    return key


forbidden.whitelist_modules('agentica.decorator')
forbidden.whitelist_objects(
    agentic,
)
