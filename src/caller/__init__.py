import importlib.metadata
from caller.caller import Caller, RetryConfig
from caller.cache import CacheConfig
from caller.rate_limiter import RateLimitConfig
from caller.types import (
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
    OpenaiResponse
)

try:
    __version__ = importlib.metadata.version("caller")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Caller",
    "CacheConfig",
    "RateLimitConfig",
    "RetryConfig",
    "ChatMessage",
    "ChatHistory",
    "InferenceConfig",
    "ToolArgs",
    "OpenaiResponse",
]
