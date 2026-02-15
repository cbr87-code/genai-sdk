import asyncio
import json
import unittest

from genai_sdk.agent import Agent
from genai_sdk.config import AgentConfig, ModelConfig
from genai_sdk.providers.base import Provider, ProviderRequest, ProviderResponse
from genai_sdk.tools.function import FunctionTool
from genai_sdk.types import ToolCall, Usage


class FakeProvider(Provider):
    def __init__(self):
        self.calls = 0

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1 and request.tools:
            return ProviderResponse(
                content="",
                tool_calls=[ToolCall(name="echo", arguments={"text": "hello"}, call_id="c1")],
                usage=Usage(total_tokens=10),
            )
        return ProviderResponse(content=json.dumps({"answer": "done"}), usage=Usage(total_tokens=20))

    async def stream(self, request: ProviderRequest):
        yield None

    async def embed(self, request):
        return None


class Out:
    @staticmethod
    def model_validate(data):
        if not isinstance(data, dict) or "answer" not in data or not isinstance(data["answer"], str):
            raise ValueError("answer must be a string")
        return Out(data["answer"])

    def __init__(self, answer: str):
        self.answer = answer

    def model_dump_json(self):
        return json.dumps({"answer": self.answer})


class TestAgent(unittest.TestCase):
    def test_agent_runs_tools_and_structured_output(self) -> None:
        async def _run() -> None:
            provider = FakeProvider()

            async def echo(args, ctx):
                return args["text"]

            tool = FunctionTool(
                name="echo",
                description="Echo input text",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                fn=echo,
            )

            agent = Agent(
                config=AgentConfig(model=ModelConfig(model="gpt-test")),
                provider=provider,
                tools=[tool],
            )

            result = await agent.run("hi", session_id="s1", response_model=Out)
            self.assertEqual(json.loads(result.output_text)["answer"], "done")
            self.assertEqual(provider.calls, 2)
            self.assertEqual(result.session_id, "s1")

        asyncio.run(_run())
