"""Minimal local vector retriever based on deterministic hash embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256

from .base import Document, RetrievedChunk


@dataclass(slots=True)
class _IndexedChunk:
    document_id: str
    text: str
    vector: list[float]
    metadata: dict


class SimpleVectorRetriever:
    """Lightweight retriever with no external vector database dependency."""

    def __init__(self, dimensions: int = 64, chunk_size: int = 500, overlap: int = 50):
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: list[_IndexedChunk] = []

    async def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            for chunk in _chunk_text(doc.text, self.chunk_size, self.overlap):
                self._chunks.append(
                    _IndexedChunk(
                        document_id=doc.id,
                        text=chunk,
                        vector=_hash_embed(chunk, self.dimensions),
                        metadata=doc.metadata,
                    )
                )

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q = _hash_embed(query, self.dimensions)
        scored = [
            (
                _cosine_similarity(q, item.vector),
                item,
            )
            for item in self._chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(document_id=item.document_id, text=item.text, score=score, metadata=item.metadata)
            for score, item in scored[:k]
        ]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


def _hash_embed(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for token in text.lower().split():
        digest = sha256(token.encode("utf-8")).digest()
        idx = digest[0] % dimensions
        values[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
