"""Provider adapters.

This module keeps optional dependencies lazy so importing `genai_sdk`
does not require all provider extras to be installed.
"""

try:
    from .openai_compatible import OpenAICompatibleProvider
except ImportError:  # pragma: no cover - optional dependency guard
    OpenAICompatibleProvider = None  # type: ignore[assignment]

__all__ = ["OpenAICompatibleProvider"]
