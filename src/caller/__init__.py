"""
Caller: A Python library for LLM API calls with caching and rate limiting.

Provides unified interface for multiple LLM providers with:
- SQLite-based caching with chunk management
- Per-model rate limiting
- Support for OpenRouter and Anthropic APIs
"""

from caller.caller import (
    Caller,
    OpenrouterCaller,
    AnthropicCaller,
    MultiClientCaller,
    PooledCaller,
    CallerConfig,
    CacheByModel,
    OpenaiResponse,
    get_universal_caller,
    sample_from_model,
    sample_across_models,
    sample_from_model_parallel,
)

from caller.llm_types import (
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
    APIRequestCache,
)

from caller.cache import (
    SQLiteCacheBackend,
    CacheChunk,
    ChunkedCacheManager,
)

from caller.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    ModelRateLimitManager,
    estimate_tokens,
)

__version__ = "0.1.0"

__all__ = [
    # Caller classes
    "Caller",
    "OpenrouterCaller",
    "AnthropicCaller",
    "MultiClientCaller",
    "PooledCaller",
    "CallerConfig",
    "CacheByModel",
    "OpenaiResponse",
    "get_universal_caller",
    # Sampling functions
    "sample_from_model",
    "sample_across_models",
    "sample_from_model_parallel",
    # Types
    "ChatMessage",
    "ChatHistory",
    "InferenceConfig",
    "ToolArgs",
    "APIRequestCache",
    # Cache
    "SQLiteCacheBackend",
    "CacheChunk",
    "ChunkedCacheManager",
    # Rate limiting
    "RateLimitConfig",
    "RateLimiter",
    "ModelRateLimitManager",
    "estimate_tokens",
]
