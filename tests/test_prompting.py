import unittest

from genai_sdk.errors import ConfigurationError
from genai_sdk.prompting import PromptTemplate


class TestPromptTemplate(unittest.TestCase):
    def test_prompt_template_renders(self) -> None:
        t = PromptTemplate("Hello {name}")
        self.assertEqual(t.render(name="Ada"), "Hello Ada")

    def test_prompt_template_missing_variable(self) -> None:
        t = PromptTemplate("Hello {name}")
        with self.assertRaises(ConfigurationError):
            t.render()
