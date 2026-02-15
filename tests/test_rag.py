import asyncio
import unittest

from genai_sdk.rag.base import Document
from genai_sdk.rag.simple_vector import SimpleVectorRetriever


class TestSimpleVectorRetriever(unittest.TestCase):
    def test_simple_vector_retriever_returns_results(self) -> None:
        async def _run() -> None:
            retriever = SimpleVectorRetriever()
            await retriever.add_documents(
                [
                    Document(id="d1", text="Python SDK for agents and tools."),
                    Document(id="d2", text="Cooking recipes and kitchen planning."),
                ]
            )

            results = await retriever.retrieve("agent tools", k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].document_id, "d1")

        asyncio.run(_run())
