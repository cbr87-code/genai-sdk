"""Configuration objects for model and agent runtime behavior."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationConfig:
    """LLM sampling and response-shaping parameters."""

    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelConfig:
    """Model identity and provider selector."""

    model: str
    provider: str = "openai_compatible"


@dataclass(slots=True)
class AgentConfig:
    """Top-level runtime settings for an :class:`Agent`."""

    model: ModelConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    max_tool_iterations: int = 4
    tool_timeout_seconds: float = 30.0
    memory_window_messages: int = 20
    summary_trigger_messages: int = 40
    retrieval_top_k: int = 5
