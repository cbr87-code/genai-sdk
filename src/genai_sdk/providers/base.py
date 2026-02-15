"""Provider abstraction used by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..config import GenerationConfig
from ..types import Message, ToolCall, Usage


@dataclass(slots=True)
class ProviderRequest:
    """Request envelope passed from SDK runtime to a provider adapter."""

    model: str
    messages: list[Message]
    generation: GenerationConfig
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ProviderResponse:
    """Provider output containing text, optional tool calls, and usage data."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class ProviderEvent:
    """Streaming event emitted by :meth:`Provider.stream`."""

    type: str
    data: Any


@dataclass(slots=True)
class EmbeddingRequest:
    """Batch embedding request."""

    model: str
    texts: list[str]


@dataclass(slots=True)
class EmbeddingResponse:
    """Batch embedding response."""

    vectors: list[list[float]]


class Provider:
    """Interface provider adapters must implement."""

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Run a single non-streaming generation request."""
        raise NotImplementedError

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        """Stream generation events."""
        raise NotImplementedError

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for one or more input strings."""
        raise NotImplementedError
