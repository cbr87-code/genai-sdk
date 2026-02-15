# genai-sdk

Minimal Python SDK for building simple AI agents.

## Features (v0)
- Async-first `Agent` API with sync wrapper
- OpenAI-compatible provider adapter
- Prompt templates with variable validation
- Tool/function calling, including MCP tool bridge
- Session memory (`InMemoryMemory`, `SQLiteMemory`)
- Basic local RAG (`SimpleVectorRetriever`)
- Structured outputs via Pydantic models

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Core API
- `Agent`: orchestrates prompting, tool calls, session memory, and optional RAG.
- `OpenAICompatibleProvider`: provider adapter for OpenAI-compatible APIs.
- `FunctionTool`: wraps Python callables as JSON-schema-described tools.
- `MCPToolset`: loads MCP-discovered tools and exposes them to `Agent`.
- `InMemoryMemory` / `SQLiteMemory`: session memory backends.
- `SimpleVectorRetriever`: local retrieval backend for small RAG workloads.

## Quickstart

```python
import asyncio
import os

from genai_sdk import Agent, AgentConfig, GenerationConfig, ModelConfig
from genai_sdk.providers import OpenAICompatibleProvider


async def main():
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    agent = Agent(
        config=AgentConfig(
            model=ModelConfig(model="gpt-5-nano"),
            generation=GenerationConfig(temperature=1.0),
        ),
        provider=provider,
    )
    result = await agent.run("Hello")
    print(result.output_text)


asyncio.run(main())
```

## Run examples

```bash
python examples/simple_chat.py
python examples/agent_with_tools.py
python examples/agent_with_rag.py
python examples/mcp_tools.py
```

## Project layout
- `src/genai_sdk/agent.py`: agent loop and orchestration
- `src/genai_sdk/providers/`: provider interfaces and adapters
- `src/genai_sdk/tools/`: function and MCP tools
- `src/genai_sdk/memory/`: session memory backends
- `src/genai_sdk/rag/`: retrieval interfaces and implementation
- `examples/`: runnable examples
- `tests/`: unit/integration-style tests with fakes
