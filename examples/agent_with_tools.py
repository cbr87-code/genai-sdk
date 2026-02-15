import os

from genai_sdk import Agent, AgentConfig, ModelConfig
from genai_sdk.providers import OpenAICompatibleProvider
from genai_sdk.tools.function import FunctionTool


def weather_tool(args, _ctx):
    city = args.get("city", "unknown")
    return {"city": city, "forecast": "sunny", "temp_c": 24}


async def main():
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    tool = FunctionTool(
        name="get_weather",
        description="Get weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        fn=weather_tool,
    )
    agent = Agent(config=AgentConfig(model=ModelConfig(model="gpt-4o-mini")), provider=provider, tools=[tool])
    result = await agent.run("What's the weather in San Francisco?")
    print(result.output_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
