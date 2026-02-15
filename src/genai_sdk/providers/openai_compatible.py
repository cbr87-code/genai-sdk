"""OpenAI-compatible HTTP provider adapter."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from ..errors import ProviderError
from ..types import Message, ToolCall, Usage
from .base import EmbeddingRequest, EmbeddingResponse, Provider, ProviderEvent, ProviderRequest, ProviderResponse


class OpenAICompatibleProvider(Provider):
    """Provider that talks to OpenAI-compatible `/chat/completions` APIs."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        """Create a provider client.

        Args:
            base_url: Base API URL ending in `/v1` for OpenAI-compatible servers.
            api_key: Bearer token.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _messages_payload(messages: list[Message]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content, "name": m.name} for m in messages]

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Execute a chat completion request and normalize the response."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._messages_payload(request.messages),
            "temperature": request.generation.temperature,
            "top_p": request.generation.top_p,
        }
        if request.generation.max_tokens is not None:
            payload["max_tokens"] = request.generation.max_tokens
        if request.generation.stop:
            payload["stop"] = request.generation.stop
        if request.generation.seed is not None:
            payload["seed"] = request.generation.seed
        if request.generation.presence_penalty is not None:
            payload["presence_penalty"] = request.generation.presence_penalty
        if request.generation.frequency_penalty is not None:
            payload["frequency_penalty"] = request.generation.frequency_penalty
        if request.generation.response_format is not None:
            payload["response_format"] = request.generation.response_format
        if request.tools:
            payload["tools"] = request.tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise ProviderError(f"Provider returned {response.status_code}: {response.text}")

        data = response.json()
        choice = data["choices"][0]["message"]
        tool_calls = []
        for tc in choice.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    name=tc["function"]["name"],
                    arguments=_parse_tool_arguments(tc["function"].get("arguments", "{}")),
                    call_id=tc.get("id", ""),
                )
            )

        usage = data.get("usage", {})
        return ProviderResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        """Yield a minimal stream from a non-streaming provider call.

        This keeps the SDK streaming shape stable while remaining simple in v0.
        """
        response = await self.generate(request)
        yield ProviderEvent(type="content", data=response.content)
        if response.tool_calls:
            yield ProviderEvent(type="tool_calls", data=response.tool_calls)
        yield ProviderEvent(type="done", data=response.usage)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Call the embeddings endpoint and return vectors."""
        payload = {"model": request.model, "input": request.texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise ProviderError(f"Embedding endpoint returned {response.status_code}: {response.text}")
        data = response.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return EmbeddingResponse(vectors=vectors)


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict):
        return data
    return {}
