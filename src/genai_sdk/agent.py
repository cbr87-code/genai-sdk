"""Primary Agent runtime for inference, tools, memory, and retrieval."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from typing import Any, Sequence

try:
    from pydantic import BaseModel, ValidationError
except ImportError:  # pragma: no cover - fallback for minimal environments
    class ValidationError(Exception):
        """Fallback validation error when pydantic is unavailable."""

    class BaseModel:  # type: ignore[no-redef]
        """Fallback BaseModel marker when pydantic is unavailable."""

from .config import AgentConfig
from .errors import StructuredOutputError, ToolExecutionError
from .memory.in_memory import InMemoryMemory
from .providers.base import Provider, ProviderRequest
from .rag.base import RetrievedChunk
from .tools.base import Tool, ToolContext
from .types import AgentResult, Message, ToolCall, Usage


class Agent:
    """High-level runtime object for executing agent turns."""

    def __init__(
        self,
        config: AgentConfig,
        provider: Provider,
        tools: Sequence[Tool] | None = None,
        memory: Any | None = None,
        retriever: Any | None = None,
    ):
        """Create an agent instance.

        Args:
            config: Runtime and model configuration.
            provider: Provider adapter used for model calls.
            tools: Optional set of registered tools.
            memory: Session memory backend. Defaults to in-memory storage.
            retriever: Optional retriever used for RAG context injection.
        """
        self.config = config
        self.provider = provider
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory or InMemoryMemory()
        self.retriever = retriever

    async def run(
        self,
        input: str | list[Message],
        session_id: str | None = None,
        user_id: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> AgentResult:
        """Execute one agent turn and return the normalized result."""
        started = time.perf_counter()
        sid = session_id or str(uuid.uuid4())

        incoming = [Message(role="user", content=input)] if isinstance(input, str) else input
        history = await self.memory.load(sid, limit=self.config.memory_window_messages)

        rag_chunks: list[RetrievedChunk] = []
        rag_context = ""
        if self.retriever and incoming:
            query = incoming[-1].content
            rag_chunks = await self.retriever.retrieve(query, k=self.config.retrieval_top_k)
            if rag_chunks:
                rag_context = "\n\n".join(
                    [f"[doc:{c.document_id} score={c.score:.3f}] {c.text}" for c in rag_chunks]
                )

        messages = list(history)
        if rag_context:
            messages.append(
                Message(
                    role="system",
                    content=(
                        "Use only the provided context when relevant. If uncertain, say so.\n"
                        f"Context:\n{rag_context}"
                    ),
                )
            )
        messages.extend(incoming)

        tool_calls_accum: list[ToolCall] = []
        usage = Usage()

        for _ in range(self.config.max_tool_iterations + 1):
            provider_request = ProviderRequest(
                model=self.config.model.model,
                messages=messages,
                generation=self.config.generation,
                tools=[t.to_provider_schema() for t in self.tools.values()],
            )
            response = await self.provider.generate(provider_request)
            usage = response.usage

            assistant_msg = Message(role="assistant", content=response.content)
            messages.append(assistant_msg)

            if not response.tool_calls:
                break

            tool_calls_accum.extend(response.tool_calls)
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                if not tool:
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Tool {call.name} not found",
                            name=call.name,
                            tool_call_id=call.call_id,
                        )
                    )
                    continue

                try:
                    result = await asyncio.wait_for(
                        tool.call(call.arguments, ToolContext(session_id=sid, user_id=user_id)),
                        timeout=self.config.tool_timeout_seconds,
                    )
                except Exception as exc:
                    raise ToolExecutionError(f"Tool {call.name} failed: {exc}") from exc

                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        name=call.name,
                        tool_call_id=call.call_id,
                    )
                )

        output_text = messages[-1].content if messages else ""
        if response_model is not None:
            output_text = self._validate_structured_output(output_text, response_model)

        await self.memory.append(sid, incoming + [Message(role="assistant", content=output_text)])
        await self.memory.summarize_if_needed(sid, budget=self.config.summary_trigger_messages)

        latency_ms = int((time.perf_counter() - started) * 1000)
        citations = [asdict(c) for c in rag_chunks]
        return AgentResult(
            output_text=output_text,
            messages=messages,
            tool_calls=tool_calls_accum,
            usage=usage,
            latency_ms=latency_ms,
            session_id=sid,
            citations=citations,
        )

    def run_sync(
        self,
        input: str | list[Message],
        session_id: str | None = None,
        user_id: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> AgentResult:
        """Synchronous wrapper around :meth:`run`."""
        return asyncio.run(self.run(input, session_id=session_id, user_id=user_id, response_model=response_model))

    @staticmethod
    def _validate_structured_output(text: str, model: type[BaseModel]) -> str:
        """Validate JSON output against a Pydantic model."""
        try:
            parsed = json.loads(text)
            obj = model.model_validate(parsed)
            return obj.model_dump_json()
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise StructuredOutputError(f"Failed to validate structured output: {exc}") from exc
