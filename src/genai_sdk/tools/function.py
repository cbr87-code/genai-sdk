"""Adapter for registering plain Python callables as tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..errors import ToolExecutionError
from .base import ToolContext

ToolCallable = Callable[[dict[str, Any], ToolContext], str | dict[str, Any] | Awaitable[str | dict[str, Any]]]


@dataclass(slots=True)
class FunctionTool:
    """Tool implementation backed by a Python function/coroutine."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolCallable

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Invoke the user function and coerce output to a string payload."""
        try:
            result = self.fn(args, ctx)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            raise ToolExecutionError(f"Tool {self.name} failed: {exc}") from exc

        if isinstance(result, dict):
            import json

            return json.dumps(result)
        return str(result)

    def to_provider_schema(self) -> dict[str, Any]:
        """Return JSON-schema tool definition for provider requests."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
