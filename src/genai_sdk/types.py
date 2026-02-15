"""Core SDK data types shared across providers, tools, and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    """One chat turn exchanged between user, assistant, system, or tool."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """A model request to execute a named tool with arguments."""

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(slots=True)
class ToolResult:
    """A normalized tool execution output."""

    name: str
    call_id: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    """Token usage metadata reported by a provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class AgentResult:
    """Normalized result returned from :meth:`genai_sdk.agent.Agent.run`."""

    output_text: str
    messages: list[Message]
    tool_calls: list[ToolCall]
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    session_id: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
