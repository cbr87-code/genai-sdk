"""Retriever interfaces and shared RAG data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Document:
    """Document input for indexing."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Retrieved chunk with similarity score and metadata."""

    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Protocol for pluggable retrieval backends."""

    async def add_documents(self, docs: list[Document]) -> None:
        """Index documents for future retrieval."""
        ...

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return top-k chunks relevant to the query."""
        ...
