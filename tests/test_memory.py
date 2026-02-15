import asyncio
import unittest

from genai_sdk.memory.in_memory import InMemoryMemory
from genai_sdk.types import Message


class TestInMemoryMemory(unittest.TestCase):
    def test_in_memory_append_and_load(self) -> None:
        async def _run() -> None:
            mem = InMemoryMemory()
            await mem.append("s1", [Message(role="user", content="hi"), Message(role="assistant", content="hey")])
            out = await mem.load("s1")
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0].content, "hi")

        asyncio.run(_run())

    def test_in_memory_summarize(self) -> None:
        async def _run() -> None:
            mem = InMemoryMemory()
            msgs = [Message(role="user", content=f"m{i}") for i in range(6)]
            await mem.append("s1", msgs)
            await mem.summarize_if_needed("s1", budget=3)
            out = await mem.load("s1", limit=10)
            self.assertEqual(out[0].role, "system")
            self.assertIn("Summary", out[0].content)

        asyncio.run(_run())
