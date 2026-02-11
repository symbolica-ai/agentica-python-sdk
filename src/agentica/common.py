"""
Tools and definitions common to agentic functions and agents.
"""

from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Self, Union, overload

from agentica_internal.core.unset import UNSET, Unset
from agentica_internal.session_manager_messages import CacheTTL, ReasoningEffort
from openai.types.responses import ResponseUsage

logger = getLogger(__name__)

__all__ = [
    'CacheTTL',
    'ModelStrings',
    'ReasoningEffort',
    'DEFAULT_AGENT_MODEL',
    'Role',
    'Chunk',
    'MaxTokens',
    'AgentRole',
    'SystemRole',
    'UserRole',
    'make_role',
    'last_usage',
    'total_usage',
    'sum_usages',
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
    type: str | None = None  # 'reasoning', 'output_text', 'code', etc.

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


def _collect_into_list(result: dict[str, Any], key: str, value: Any) -> None:
    """Add value(s) into a deduplicated list at result[key]."""
    existing = result.get(key)
    if not isinstance(existing, list):
        existing = []
        result[key] = existing
    # Flatten lists (from already-aggregated usages) into the result
    items = value if isinstance(value, list) else [value]
    for item in items:
        if item not in existing:
            existing.append(item)


def _sum_dicts(dicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recursively sum numeric fields across dicts.

    - bool values are collected into a deduplicated list (checked before int since bool is a subclass of int).
    - int/float values are summed.
    - dict values are recursively merged.
    - All other values (including lists from prior aggregation) are collected into a deduplicated list.
    """
    result: dict[str, Any] = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, bool):
                _collect_into_list(result, key, value)
            elif isinstance(value, (int, float)):
                result[key] = result.get(key, 0) + value
            elif isinstance(value, dict):
                existing = result.get(key)
                if isinstance(existing, dict):
                    result[key] = _sum_dicts([existing, value])
                else:
                    result[key] = _sum_dicts([value])
            else:
                _collect_into_list(result, key, value)
    return result


def sum_usages(usages: 'Iterable[ResponseUsage]') -> ResponseUsage:
    """Sum multiple ResponseUsage objects into a single total, preserving extra fields.

    Numeric fields (input_tokens, cost, etc.) are summed. Non-numeric fields
    (is_byok, etc.) are collected into a deduplicated list. Nested dicts
    (input_tokens_details, cost_details, etc.) are recursively merged.

    A ``inference_count`` extra field is injected (1 per input usage) so the aggregated
    result records how many raw inferences it represents. When re-aggregating
    already-aggregated usages (e.g. total_usage() summing per-invocation
    last_usage() results), the inner counts sum naturally.

    Examples::

        # agent.last_usage() sums the per-round usages within one invocation:
        #   round 1: {input_tokens: 100, cost: 0.001, is_byok: True,  inference_count: 1}
        #   round 2: {input_tokens: 200, cost: 0.002, is_byok: False, inference_count: 1}
        #   result:  {input_tokens: 300, cost: 0.003, is_byok: [True, False], inference_count: 2}

        # agent.total_usage() then sums the per-invocation usages:
        #   invocation 1: {input_tokens: 300, cost: 0.003, is_byok: [True], inference_count: 1}
        #   invocation 2: {input_tokens: 400, cost: 0.004, is_byok: [True], inference_count: 2}
        #   result:       {input_tokens: 700, cost: 0.007, is_byok: [True],  inference_count: 3}

    See ``tests/integration/test_usage_stats.py::test_extra_usage_fields_*``
    for end-to-end demonstrations.
    """
    return ResponseUsage.model_validate(
        _sum_dicts({'inference_count': 1, **u.model_dump(exclude_none=True)} for u in usages)
    )


if TYPE_CHECKING:
    from .agent import Agent
    from .function import AgenticFunction


def last_usage(ag: Union['AgenticFunction', 'Agent', Callable[..., Any]]) -> 'ResponseUsage':
    """Get the usage stats of the last invocation of an agentic function or agent."""
    from .agent import Agent
    from .function import AgenticFunction

    if isinstance(ag, AgenticFunction):
        return ag.last_usage()
    if isinstance(ag, Agent):
        return ag.last_usage()
    raise TypeError(f"Expected an agentic function or agent, got {type(ag)}")


def total_usage(ag: Union['AgenticFunction', 'Agent', Callable[..., Any]]) -> 'ResponseUsage':
    """Get the total usage stats of an agentic function or agent."""
    from .agent import Agent
    from .function import AgenticFunction

    if isinstance(ag, AgenticFunction):
        return ag.total_usage()
    if isinstance(ag, Agent):
        return ag.total_usage()
    raise TypeError(f"Expected an agentic function or agent, got {type(ag)}")
