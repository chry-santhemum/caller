"""
Caller: A Python library for LLM API calls with caching and rate limiting.

Main entry point: Caller class

Example:
    from caller import Caller

    caller = Caller()

    # Single call
    response = await caller.call_one("Hello!", model="anthropic/claude-3.5-sonnet")
    print(response.first_response)

    # Batch calls with shared parameters
    responses = await caller.call(["Hi", "Hello", "Hey"], model="gpt-4", max_tokens=10)
"""

from caller.caller import (
    Caller,
    OpenaiResponse,
)

from caller.llm_types import (
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
)

__version__ = "0.2.0"

__all__ = [
    # Main class
    "Caller",
    "OpenaiResponse",
    # Types for advanced usage
    "ChatMessage",
    "ChatHistory",
    "InferenceConfig",
    "ToolArgs",
]
