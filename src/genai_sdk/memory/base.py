"""Memory backend protocol for session history management."""

from __future__ import annotations

from typing import Protocol

from ..types import Message


class MemoryBackend(Protocol):
    """Protocol for pluggable memory persistence layers."""

    async def append(self, session_id: str, messages: list[Message]) -> None:
        """Append messages to a session."""
        ...

    async def load(self, session_id: str, limit: int = 20) -> list[Message]:
        """Load the most recent messages for a session."""
        ...

    async def summarize_if_needed(self, session_id: str, budget: int) -> None:
        """Compact session history when it exceeds a message budget."""
        ...
