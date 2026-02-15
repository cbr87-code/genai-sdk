import unittest

from genai_sdk.providers.openai_compatible import OpenAICompatibleProvider
from genai_sdk.types import Message


class TestOpenAICompatibleProvider(unittest.TestCase):
    def test_messages_payload_preserves_tool_shapes(self) -> None:
        messages = [
            Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{\"text\":\"hello\"}"},
                        }
                    ]
                },
            ),
            Message(
                role="tool",
                content="hello",
                name="echo",
                tool_call_id="call_1",
            ),
        ]

        payload = OpenAICompatibleProvider._messages_payload(messages)
        self.assertEqual(payload[0]["role"], "assistant")
        self.assertIsNone(payload[0]["content"])
        self.assertEqual(payload[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(payload[1]["role"], "tool")
        self.assertEqual(payload[1]["tool_call_id"], "call_1")
        self.assertEqual(payload[1]["name"], "echo")

