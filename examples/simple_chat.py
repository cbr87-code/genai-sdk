import os

from genai_sdk import Agent, AgentConfig, ModelConfig
from genai_sdk.providers import OpenAICompatibleProvider


async def main():
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    agent = Agent(
        config=AgentConfig(model=ModelConfig(model=os.getenv("MODEL", "gpt-4o-mini"))),
        provider=provider,
    )
    result = await agent.run("Write a haiku about clean APIs.")
    print(result.output_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
