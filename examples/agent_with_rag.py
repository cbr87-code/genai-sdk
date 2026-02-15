import os

from genai_sdk import Agent, AgentConfig, ModelConfig
from genai_sdk.providers import OpenAICompatibleProvider
from genai_sdk.rag.base import Document
from genai_sdk.rag.simple_vector import SimpleVectorRetriever


async def main():
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    retriever = SimpleVectorRetriever()
    await retriever.add_documents(
        [
            Document(id="doc1", text="The internal policy says refunds are valid for 30 days with receipt."),
            Document(id="doc2", text="Premium support is available Monday to Friday from 9am to 5pm."),
        ]
    )

    agent = Agent(
        config=AgentConfig(model=ModelConfig(model="gpt-4o-mini")),
        provider=provider,
        retriever=retriever,
    )
    result = await agent.run("What is the refund policy?")
    print(result.output_text)
    print(result.citations)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
