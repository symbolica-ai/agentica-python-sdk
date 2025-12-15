import asyncio
import inspect
import json
import types
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

from fastmcp.client import Client
from fastmcp.mcp_config import MCPConfig
from mcp.types import CallToolResult, Tool, ToolAnnotations

from agentica.std.utils import spoof_signature

from .sigs import build_signature_and_annotations


class MCPFunction:
    # For Python
    __name__: str
    __doc__: str
    __qualname__: str
    __module__: str
    __annotations__: dict[str, Any]
    __signature__: inspect.Signature
    __wrapped__: Callable[..., Any]
    _is_coroutine: bool
    # For MCP
    __tool_inputSchema: dict[str, Any]
    __tool_outputSchema: dict[str, Any]
    __mcp_config: MCPConfig
    __tool_annotations__: list[ToolAnnotations]
    __tool_meta__: dict[str, Any]

    def __init__(
        self,
        *,
        tool: Tool,
        mcp_config: MCPConfig,
    ):
        # For MCP
        self.__mcp_config = mcp_config
        self.__tool_inputSchema = tool.inputSchema or dict()
        self.__tool_outputSchema = tool.outputSchema or dict()
        self.__tool_annotations__ = tool.annotations or list()
        self.__tool_meta__ = tool.meta or dict()

        # For Python
        self.__name__ = tool.name
        self.__doc__ = tool.description or None
        self.__qualname__ = tool.name
        self.__module__ = tool.name
        # let the imposition begin
        self._is_coroutine = asyncio.coroutines._is_coroutine
        signature, annotations = build_signature_and_annotations(
            self.__tool_inputSchema, self.__tool_outputSchema
        )
        self.__signature__ = signature
        self.__annotations__ = annotations

        async def _wrapped(*args, **kwargs):
            return await self.__call__(*args, **kwargs)

        self.__wrapped__ = _wrapped

        spoof_signature(self.__wrapped__, self)

    def __repr__(self) -> str:
        return f"<function {self.__qualname__} at 0x{id(self):x}>"

    # Forward important things to wrapped function
    def __getattr__(self, name: str) -> Any:
        if name in {"__code__", "__defaults__", "__kwdefaults__", "__dict__"}:
            return getattr(self.__wrapped__, name)
        raise AttributeError(name)

    @classmethod
    async def from_mcp_config(cls, mcp_config: MCPConfig) -> list["MCPFunction"]:
        client = Client(mcp_config)
        async with client:
            tools = await client.session.list_tools()
            return [cls(tool=tool, mcp_config=mcp_config) for tool in tools.tools]

    @classmethod
    async def from_json(cls, path_to_config: str) -> list["MCPFunction"]:
        fp = Path(path_to_config)
        if not fp.exists():
            raise FileNotFoundError(f"Config file {path_to_config} does not exist")
        if not fp.suffix == ".json":
            raise ValueError("Config file must be a .json file")
        with fp.open("r") as fh:
            mcp_dict = json.load(fh)
        return await cls.from_mcp_config(MCPConfig.from_dict(mcp_dict))

    async def __call__(self, *args, **kwargs):
        # Build argument dict from kwargs and remaining positional args in schema order
        args_dict = dict(kwargs)
        if self.__tool_inputSchema is not None and ("properties" in self.__tool_inputSchema):
            missing_keys = [
                k for k in self.__tool_inputSchema["properties"].keys() if k not in args_dict
            ]
            for i, key in enumerate(missing_keys):
                if i >= len(args):
                    break
                args_dict[key] = args[i]

        async with Client(self.__mcp_config) as client:
            result: CallToolResult = await client.session.call_tool(self.__name__, args_dict)
            if result.isError:
                raise RuntimeError(";".join(tc.text for tc in result.content))

            # prefer structured content
            structured = getattr(result, "structuredContent", None)
            value = None
            if isinstance(structured, dict):
                if "result" in structured:
                    value = structured["result"]
                elif structured not in ({}, None):
                    value = structured

            # coerce list -> tuple if expected return is tuple
            expected_ret = self.__annotations__.get("return")

            def _expects_tuple(t: Any) -> bool:
                if t is None:
                    return False
                origin = get_origin(t)
                if origin is tuple:
                    return True
                if origin in (types.UnionType, getattr(__import__('typing'), 'Union', object)):
                    return any(get_origin(a) is tuple for a in get_args(t))
                return False

            if isinstance(value, list) and _expects_tuple(expected_ret):
                return tuple(value)

            return value
