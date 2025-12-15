"""
Tools and definitions common to agentic functions and agents.
"""

from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, Union, overload

from agentica_internal.core.unset import UNSET, Unset

logger = getLogger(__name__)

__all__ = [
    'ModelStrings',
    'ToolModeStrings',
    'DEFAULT_AGENT_MODEL',
    'Role',
    'Chunk',
    'MaxTokens',
    'Usage',
    'AgentRole',
    'SystemRole',
    'UserRole',
    'make_role',
    'last_usage',
    'total_usage',
    '_set_current_local_id',
    '_reset_current_local_id',
    '_set_current_invocation',
    '_reset_current_invocation',
    '_find_parent_invocation',
    '_find_parent_local_id',
]

type PrimarySupportedModels = Literal[
    'openai:gpt-3.5-turbo',
    'openai:gpt-4o',
    'openai:gpt-4.1',
    'openai:gpt-5',
    'anthropic:claude-sonnet-4',
    'anthropic:claude-opus-4.1',
    'anthropic:claude-sonnet-4.5',
    'anthropic:claude-opus-4.5',
]

type ModelStrings = PrimarySupportedModels | str

type ToolModeStrings = Literal['code', 'json']

DEFAULT_AGENT_MODEL = 'openai:gpt-4.1'

DEFAULT_AGENT_MAX_TOKENS_PER_INVOCATION: int | None = None  # unlimited
DEFAULT_AGENT_MAX_TOKENS_PER_ROUND: int | None = None  # unlimited
DEFAULT_AGENT_MAX_ROUNDS: int | None = None  # unlimited

_current_local_id: ContextVar[int | None] = ContextVar('_current_local_id', default=None)

_current_invocation: ContextVar[tuple[str, str] | None] = ContextVar(
    '_current_invocation', default=None
)


def _set_current_local_id(local_id: int | None) -> Any:
    return _current_local_id.set(local_id)


def _reset_current_local_id(token: Any) -> None:
    _current_local_id.reset(token)


def _set_current_invocation(uid: str, iid: str) -> Any:
    """Set the current invocation context (uid, iid) for nested agent spawning."""
    return _current_invocation.set((uid, iid))


def _reset_current_invocation(token: Any) -> None:
    """Reset the current invocation context."""
    _current_invocation.reset(token)


def _find_parent_invocation() -> tuple[str | None, str | None]:
    r = _current_invocation.get()
    return r if r is not None else (None, None)


def _find_parent_local_id() -> int | None:
    return _current_local_id.get()


class _Role[Field, Value](ABC):
    @overload
    def __getitem__(self, key: Field) -> Value: ...

    @overload
    def __getitem__(self, key: Literal['name']) -> str | None: ...

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __eq__[F, V](self, other: 'str | _Role[F, V]') -> bool:
        if isinstance(other, str):
            return getattr(self, 'role') == other
        if hasattr(other, 'role'):
            return getattr(self, 'role') == getattr(other, 'role')
        return False

    def __str__(self) -> str:
        return f"{getattr(self, 'role')}"

    __repr__ = __str__


class UserRole(_Role[Literal['role'], Literal['user']]):
    role: Literal['user'] = 'user'
    name: str | None

    def __init__(self, name: str | None):
        self.name = name

    def __eq__[F, V](self, other: 'str | _Role[F, V]') -> bool:
        if isinstance(other, UserRole):
            return self.name == other.name
        return super().__eq__(other)

    def __str__(self) -> str:
        return f"{self.role}<{self.name}>"

    __repr__ = __str__


class AgentRole(_Role[Literal['role'], Literal['agent']]):
    role: Literal['agent'] = 'agent'


class SystemRole(_Role[Literal['role'], Literal['system']]):
    role: Literal['system'] = 'system'


type Role = UserRole | AgentRole | SystemRole


def make_role(role: str, name: str | None = None) -> Role:
    match role:
        case 'user':
            return UserRole(name)
        case 'agent':
            return AgentRole()
        case 'system':
            return SystemRole()
        case _:
            raise ValueError(f"Invalid role: {role}")


@dataclass
class Chunk:
    role: Role
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass(slots=True, frozen=True)
class MaxTokens:
    """
    Control the maximum number of tokens an agent or agentic function can generate:

    per_invocation: int | None
        The maximum number of tokens for an invocation (unlimited if None).
    per_round: int | None
        The maximum number of tokens for a round of inference (unlimited if None).
    rounds: int | None
        The maximum number of rounds of inference (unlimited if None).
    """

    per_invocation: int | None = DEFAULT_AGENT_MAX_TOKENS_PER_INVOCATION
    per_round: int | None = DEFAULT_AGENT_MAX_TOKENS_PER_ROUND
    rounds: int | None = DEFAULT_AGENT_MAX_ROUNDS

    @classmethod
    def default(cls) -> Self:
        return cls()

    @classmethod
    def from_max_tokens(
        cls,
        per_invocation: 'int | MaxTokens | None | Unset' = UNSET,
        per_round: int | None | Unset = UNSET,
        rounds: int | None | Unset = UNSET,
    ) -> Self:
        """If the first argument is a `MaxTokens` object, use its values for the other arguments."""
        if isinstance(per_invocation, MaxTokens):
            assert per_round is UNSET
            assert rounds is UNSET
            o: MaxTokens = per_invocation
            return cls(
                per_invocation=o.per_invocation,
                per_round=o.per_round,
                rounds=o.rounds,
            )
        new = cls.default()
        if per_invocation is not UNSET:
            object.__setattr__(new, 'per_invocation', per_invocation)
        if per_round is not UNSET:
            object.__setattr__(new, 'per_round', per_round)
        if rounds is not UNSET:
            object.__setattr__(new, 'rounds', rounds)
        return new


@dataclass(slots=True, frozen=True)
class Usage:
    """
    Token usage statistics:
    - input_tokens:  the number of tokens consumed by the model
    - output_tokens: the number of tokens consequently generated by the model
    - total_tokens:  the total number of tokens *processed*, not double-counting generated then re-consumed tokens.

    NOTE: total_tokens is not the same as input_tokens + output_tokens,
    it is just the total number of tokens involved in the belonging timeframe.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int

    @classmethod
    def from_completions(cls, usage: dict[str, int], last_usage: Self | None = None) -> Self:
        input_tokens = usage.get('prompt_tokens', -1)
        output_tokens = usage.get('completion_tokens', -1)
        total_tokens = usage.get('total_tokens', -1)
        if -1 in {input_tokens, output_tokens, total_tokens}:
            logger.warning(f"Usage is missing fields: {usage!r}")

        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)
        total_tokens = max(total_tokens, 0)

        if last_usage is not None:
            # input tokens and total tokens are cumulative,
            # so we need to subtract the last usage
            input_tokens -= last_usage.input_tokens
            total_tokens -= last_usage.total_tokens

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def replace(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> Self:
        return type(self)(
            input_tokens=or_if_none(input_tokens, self.input_tokens),
            output_tokens=or_if_none(output_tokens, self.output_tokens),
            total_tokens=or_if_none(total_tokens, self.total_tokens),
        )

    def __add__(self, other: Self) -> Self:
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __sub__(self, other: Self) -> Self:
        return type(self)(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
        )


def or_if_none(value: Any, default: Any) -> Any:
    return value if value is not None else default


if TYPE_CHECKING:
    from .agent import Agent
    from .function import AgenticFunction


def last_usage(ag: Union['AgenticFunction', 'Agent', Callable[..., Any]]) -> Usage:
    """Get the usage stats of the last invocation of an agentic function or agent."""
    from .agent import Agent
    from .function import AgenticFunction

    if isinstance(ag, AgenticFunction):
        return ag.last_usage()
    if isinstance(ag, Agent):
        return ag.last_usage()
    raise TypeError(f"Expected an agentic function or agent, got {type(ag)}")


def total_usage(ag: Union['AgenticFunction', 'Agent', Callable[..., Any]]) -> Usage:
    """Get the total usage stats of an agentic function or agent."""
    from .agent import Agent
    from .function import AgenticFunction

    if isinstance(ag, AgenticFunction):
        return ag.total_usage()
    if isinstance(ag, Agent):
        return ag.total_usage()
    raise TypeError(f"Expected an agentic function or agent, got {type(ag)}")
