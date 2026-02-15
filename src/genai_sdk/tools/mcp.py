"""MCP tool bridge interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .base import ToolContext


class MCPClient(Protocol):
    """Protocol for a minimal MCP client implementation."""

    async def list_tools(self) -> list[dict[str, Any]]:
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        ...


@dataclass(slots=True)
class MCPBoundTool:
    """SDK tool wrapper around a remote MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    client: MCPClient

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return await self.client.call_tool(self.name, args)

    def to_provider_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class MCPToolset:
    """Loader that converts MCP-discovered tools into SDK-compatible tools."""

    def __init__(self, client: MCPClient):
        self.client = client

    async def load(self) -> list[MCPBoundTool]:
        """Discover tools from MCP and return bound wrappers."""
        tools = await self.client.list_tools()
        return [
            MCPBoundTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("input_schema", {"type": "object", "properties": {}}),
                client=self.client,
            )
            for t in tools
        ]
