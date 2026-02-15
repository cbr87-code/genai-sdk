"""In-memory message store for lightweight development use."""

from __future__ import annotations

from collections import defaultdict

from ..types import Message


class InMemoryMemory:
    """Simple dictionary-backed session memory backend."""

    def __init__(self):
        self._messages: dict[str, list[Message]] = defaultdict(list)

    async def append(self, session_id: str, messages: list[Message]) -> None:
        self._messages[session_id].extend(messages)

    async def load(self, session_id: str, limit: int = 20) -> list[Message]:
        return self._messages[session_id][-limit:]

    async def summarize_if_needed(self, session_id: str, budget: int) -> None:
        msgs = self._messages[session_id]
        if len(msgs) <= budget:
            return
        summary = "\n".join(f"[{m.role}] {m.content}" for m in msgs[:-budget])
        self._messages[session_id] = [Message(role="system", content=f"Summary:\n{summary}")] + msgs[-budget:]
