import importlib.metadata
from caller.caller import (
    AutoCaller,
    OpenRouterCaller,
    OpenAICaller,
    AnthropicCaller,
    LocalCaller,
    RetryConfig,
)
from caller.cache import CacheConfig
from caller.types import (
    FunctionDescription,
    Tool,
    ToolChoiceFunction,
    ToolChoice,
    ResponseFormat,
    InferenceConfig,
    ChatMessage,
    ToolMessage,
    Message,
    ChatHistory,
    Request,
    Response,
)

try:
    __version__ = importlib.metadata.version("caller")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "AutoCaller",
    "OpenRouterCaller",
    "OpenAICaller",
    "AnthropicCaller",
    "LocalCaller",
    "RetryConfig",
    "CacheConfig",
    "FunctionDescription",
    "Tool",
    "ToolChoiceFunction",
    "ToolChoice",
    "ResponseFormat",
    "InferenceConfig",
    "ChatMessage",
    "ToolMessage",
    "Message",
    "ChatHistory",
    "Request",
    "Response",
]
