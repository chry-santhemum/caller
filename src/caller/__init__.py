import importlib.metadata
from caller.caller import OpenRouterCaller, RetryConfig
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
    "OpenRouterCaller",
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
