"""Tool interfaces used by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolContext:
    """Execution context passed to tools."""

    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Protocol for callable tools available to an agent."""

    name: str
    description: str
    input_schema: dict[str, Any]

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Execute the tool and return a string payload for the model."""
        ...

    def to_provider_schema(self) -> dict[str, Any]:
        """Return tool schema in provider-compatible format."""
        ...
