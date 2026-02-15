class GenAISDKError(Exception):
    """Base SDK exception."""


class ConfigurationError(GenAISDKError):
    """Raised for invalid configuration."""


class ProviderError(GenAISDKError):
    """Raised when provider request/response fails."""


class ToolExecutionError(GenAISDKError):
    """Raised when tool execution fails."""


class StructuredOutputError(GenAISDKError):
    """Raised when structured output validation fails."""
