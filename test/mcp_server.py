from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator Server", log_level="WARNING")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


add.__example__ = {"a": 1, "b": 2}


@mcp.tool()
def is_even(a: int) -> bool:
    """Check if a number is even."""
    return a % 2 == 0


is_even.__example__ = {"a": 2}


@mcp.tool()
def optional_arg(a: int, b: int = 0) -> int:
    """Add two numbers with an optional second argument."""
    return a + b


optional_arg.__example__ = {"a": 1}


@mcp.tool()
def takes_tuples(a: tuple[int, bool]) -> tuple[int, bool] | None:
    return a if a[1] else None


takes_tuples.__example__ = {"a": (1, True)}


@mcp.tool()
def union_type(a: int | str) -> int | bool:
    """Accepts int or str; returns int | bool."""
    return a if isinstance(a, int) else False


union_type.__example__ = {"a": 1}


@mcp.tool()
def colour_enum(color: Literal["red", "green", "blue"]) -> Literal["ok"]:
    """Accepts one of a string enum and returns a literal."""
    _ = color
    return "ok"


colour_enum.__example__ = {"color": "red"}


@mcp.tool()
def nullable_optional(a: int | None) -> int | None:
    """Optional/nullable integer passthrough."""
    return a


nullable_optional.__example__ = {"a": 1}


@mcp.tool()
def sum_list(values: list[int]) -> int:
    """Sum a list of integers."""
    return sum(values)


sum_list.__example__ = {"values": [1, 2, 3]}


@mcp.tool()
def stringify_mixed(values: list[int | str]) -> list[str]:
    """Stringify a list containing ints or strs."""
    return [str(v) for v in values]


stringify_mixed.__example__ = {"values": [1, "2", 3]}


@mcp.tool()
def map_counts(counts: dict[str, int]) -> dict[str, int]:
    """Echo a mapping of string keys to integer values."""
    return counts


map_counts.__example__ = {"counts": {"a": 1, "b": 2}}


@mcp.tool()
def dict_union_values(values: dict[str, int | bool]) -> dict[str, int | bool]:
    return values


dict_union_values.__example__ = {"values": {"a": 1, "b": True}}


@mcp.tool()
def nested_map(flags: dict[str, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    """Echo a nested mapping of booleans."""
    return flags


nested_map.__example__ = {"flags": {"a": {"b": True}, "c": {"d": False}}}


@mcp.tool()
def accepts_literal(num: Literal[0, 1, 2]) -> Literal[0, 1, 2]:
    """Echo a small integer literal set."""
    return num


accepts_literal.__example__ = {"num": 1}


@mcp.tool()
def meaning_of_life() -> dict[str, int]:
    """The meaning of life, the universe, and everything."""
    return {"meaning_of_life": 42}


meaning_of_life.__example__ = {}


@mcp.tool()
def passthrough_any(obj: dict[str, Any]) -> dict[str, Any]:
    """Pass through any JSON-like mapping."""
    return obj


@mcp.tool()
def fail() -> None:
    raise ValueError("boom")


fail.__example__ = {}


passthrough_any.__example__ = {"obj": {"a": 1, "b": "2"}}


if __name__ == "__main__":
    mcp.run(transport="stdio")
