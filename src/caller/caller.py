"""
Unified Caller class for LLM API calls with caching, rate limiting, and retry logic.
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, Literal
from json import JSONDecodeError

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist

import openai
import anthropic
from openai import OpenAI, AsyncOpenAI
from openai._types import omit as OPENAI_OMIT
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message import Message
from anthropic._types import omit as ANTHROPIC_OMIT

from caller.llm_types import (
    APIRequestCache,
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
)
from caller.cache import SQLiteCacheBackend, ChunkedCacheManager, CacheConfig


logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting behavior."""

    # Thresholds for proactive waiting (only for providers with rate limit headers)
    min_requests_remaining: int = 5  # Wait if fewer than this many requests remaining
    min_tokens_remaining: int = 1000  # Wait if fewer than this many tokens remaining


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = 8  # Maximum number of retry attempts
    min_wait_seconds: float = 1.0  # Minimum wait time between retries
    max_wait_seconds: float = 60.0  # Maximum wait time between retries
    exponential_multiplier: float = 2.0  # Exponential backoff multiplier


def is_thinking_model(model_name: str) -> bool:
    """Whether or not there is an explicit thinking mode for this model."""
    THINKING_MODELS = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "openai/gpt-5",
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "openai/o3",
        "deepseek/deepseek-r1",
    ]
    return model_name in THINKING_MODELS


class OpenaiResponse(BaseModel):
    """Unified response format for all providers."""
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        try:
            content = self.choices[0]["message"]["content"]
            if content is None:
                raise ValueError(f"No content found in OpenaiResponse: {self}")
            if isinstance(content, dict):
                content = content.get("text", "")
            return content
        except (TypeError, KeyError, IndexError) as e:
            raise ValueError(f"No content found in OpenaiResponse: {self}") from e

    @property
    def reasoning_content(self) -> str | None:
        """Returns the reasoning content if it exists, otherwise None."""
        try:
            possible_keys = ["reasoning_content", "reasoning"]
            for key in possible_keys:
                if key in self.choices[0]["message"]:
                    return self.choices[0]["message"][key]

                content = self.choices[0]["message"].get("content")
                if isinstance(content, dict) and key in content:
                    return content[key]
        except (KeyError, IndexError):
            pass
        return None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_content is not None

    def has_response(self) -> bool:
        if len(self.choices) == 0:
            return False
        first_choice = self.choices[0]
        if first_choice.get("message") is None:
            return False
        if first_choice["message"].get("content") is None:
            return False
        return True

    @property
    def hit_content_filter(self) -> bool:
        """Check if response was blocked by content filter."""
        try:
            first_choice = self.choices[0]
            finish_reason = first_choice.get("finishReason") or first_choice.get("finish_reason")
            return finish_reason == "content_filter"
        except (KeyError, IndexError):
            return False

    @property
    def abnormal_finish(self) -> bool:
        """Check if response finished abnormally."""
        try:
            first_choice = self.choices[0]
            finish_reason = first_choice.get("finishReason") or first_choice.get("finish_reason")
            return finish_reason not in ["stop", "length", None]
        except (KeyError, IndexError):
            return False


class RateLimitState(BaseModel):
    """Rate limit state parsed from response headers."""
    requests_remaining: int | None = None
    requests_reset: datetime | None = None
    tokens_remaining: int | None = None
    tokens_reset: datetime | None = None
    input_tokens_remaining: int | None = None
    output_tokens_remaining: int | None = None


class HeaderRateLimiter:
    """
    Rate limiter that uses response headers from APIs.
    Only applies to Anthropic and OpenAI direct APIs.
    OpenRouter has no rate limit headers, so we rely on retries.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self.states: dict[str, RateLimitState] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, model: str) -> asyncio.Lock:
        """Get or create lock for a model."""
        if model not in self.locks:
            self.locks[model] = asyncio.Lock()
        return self.locks[model]

    async def wait_if_needed(self, model: str, provider: str) -> None:
        """
        Check rate limits and wait if necessary.
        Only applies to providers that return rate limit headers.
        """
        if provider == "openrouter":
            return  # OpenRouter: no proactive limiting, rely on retries

        lock = self._get_lock(model)
        async with lock:
            state = self.states.get(model)
            if not state:
                return

            from datetime import timezone
            now = datetime.now(timezone.utc)
            if state.requests_reset and now >= state.requests_reset:
                return

            if state.requests_remaining is not None and state.requests_remaining < self.config.min_requests_remaining:
                if state.requests_reset:
                    wait_time = (state.requests_reset - now).total_seconds()
                    if wait_time > 0:
                        logger.warning(
                            f"Rate limit low for {model} "
                            f"({state.requests_remaining} requests remaining), "
                            f"waiting {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                        return
            if state.input_tokens_remaining is not None and state.input_tokens_remaining < self.config.min_tokens_remaining:
                if state.tokens_reset:
                    wait_time = (state.tokens_reset - now).total_seconds()
                    if wait_time > 0:
                        logger.warning(
                            f"Token limit low for {model} "
                            f"({state.input_tokens_remaining} input tokens remaining), "
                            f"waiting {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)

    def update_from_headers(self, model: str, provider: str, headers: dict | None) -> None:
        """Update rate limit state from response headers."""
        if not headers or provider == "openrouter":
            return

        if provider == "anthropic":
            self._parse_anthropic_headers(model, headers)
        elif provider == "openai":
            self._parse_openai_headers(model, headers)

    def _parse_anthropic_headers(self, model: str, headers: dict) -> None:
        """Parse Anthropic rate limit headers."""
        try:
            state = RateLimitState()

            if "anthropic-ratelimit-requests-remaining" in headers:
                state.requests_remaining = int(headers["anthropic-ratelimit-requests-remaining"])
            if "anthropic-ratelimit-requests-reset" in headers:
                state.requests_reset = datetime.fromisoformat(
                    headers["anthropic-ratelimit-requests-reset"].replace("Z", "+00:00")
                )

            if "anthropic-ratelimit-input-tokens-remaining" in headers:
                state.input_tokens_remaining = int(headers["anthropic-ratelimit-input-tokens-remaining"])

            if "anthropic-ratelimit-output-tokens-remaining" in headers:
                state.output_tokens_remaining = int(headers["anthropic-ratelimit-output-tokens-remaining"])

            if "anthropic-ratelimit-tokens-reset" in headers:
                state.tokens_reset = datetime.fromisoformat(
                    headers["anthropic-ratelimit-tokens-reset"].replace("Z", "+00:00")
                )

            self.states[model] = state
            logger.debug(f"Updated rate limits for {model}: {state}")
        except Exception as e:
            logger.warning(f"Failed to parse Anthropic headers: {e}")

    def _parse_openai_headers(self, model: str, headers: dict) -> None:
        """Parse OpenAI rate limit headers."""
        try:
            state = RateLimitState()

            if "x-ratelimit-remaining-requests" in headers:
                state.requests_remaining = int(headers["x-ratelimit-remaining-requests"])
            if "x-ratelimit-reset-requests" in headers:
                reset_str = headers["x-ratelimit-reset-requests"]
                state.requests_reset = self._parse_reset_time(reset_str)

            if "x-ratelimit-remaining-tokens" in headers:
                state.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
            if "x-ratelimit-reset-tokens" in headers:
                reset_str = headers["x-ratelimit-reset-tokens"]
                state.tokens_reset = self._parse_reset_time(reset_str)

            self.states[model] = state
            logger.debug(f"Updated rate limits for {model}: {state}")
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI headers: {e}")

    def _parse_reset_time(self, reset_str: str) -> datetime:
        """Parse reset time string like '7m12s' into datetime."""
        import re
        from datetime import timezone

        total_seconds = 0

        m_match = re.search(r'(\d+)m', reset_str)
        if m_match:
            total_seconds += int(m_match.group(1)) * 60

        # Don't match milliseconds here
        s_match = re.search(r'(\d+)s(?!$)', reset_str)
        if s_match:
            total_seconds += int(s_match.group(1))

        ms_match = re.search(r'(\d+)ms', reset_str)
        if ms_match:
            total_seconds += int(ms_match.group(1)) / 1000

        return datetime.now(timezone.utc) + timedelta(seconds=total_seconds)


RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic._exceptions.OverloadedError,
)

CHANCE_EXCEPTIONS = (
    ValidationError,
    JSONDecodeError,
    ValueError,
)


class Caller:
    """
    Unified LLM API caller with caching, rate limiting, and retry logic.

    Usage:
        # Basic usage
        caller = Caller()

        # Single call
        response = await caller.call_one(messages, model="anthropic/claude-3.5-sonnet")

        # Batch calls with shared parameters
        responses = await caller.call(["Hi", "Hello", "Hey"], model="gpt-4", max_tokens=10)

        # Configure caching behavior
        from caller import CacheConfig
        cache_config = CacheConfig(
            no_cache_models={"o1", "gpt-4o-realtime"},
            max_chunks_in_memory=20,      # Keep 20 chunks in RAM
            entries_per_chunk=100,         # 100 entries per chunk
            max_age_days=60
        )
        caller = Caller(cache_config=cache_config)

        # Configure rate limiting and retry behavior
        from caller import RateLimitConfig, RetryConfig
        rate_limit_config = RateLimitConfig(min_requests_remaining=10)
        retry_config = RetryConfig(max_attempts=5, max_wait_seconds=30)
        caller = Caller(
            rate_limit_config=rate_limit_config,
            retry_config=retry_config
        )

        # Or synchronously
        response = caller.call_one_sync(messages, model="gpt-4")
    """

    def __init__(
        self,
        cache_dir: str = ".cache/caller",
        default_provider: str = "openrouter",
        openrouter_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        dotenv_path: str | Path | None = None,
        cache_config: CacheConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Caller with API clients and caching.

        Args:
            cache_dir: Directory for SQLite cache
            default_provider: Default provider to use (always "openrouter" by default)
            openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            dotenv_path: Path to .env file (optional)
            cache_config: Cache configuration (CacheConfig object)
            rate_limit_config: Rate limiting configuration (RateLimitConfig object)
            retry_config: Retry behavior configuration (RetryConfig object)
        """
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()

        self.openrouter_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.openrouter_client = AsyncOpenAI(
            api_key=self.openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        ) if self.openrouter_key else None

        self.anthropic_client = AsyncAnthropic(
            api_key=self.anthropic_key
        ) if self.anthropic_key else None

        self.openai_client = AsyncOpenAI(
            api_key=self.openai_key
        ) if self.openai_key else None

        self.default_provider = default_provider

        self.cache_config = cache_config or CacheConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Each model will have its own database and cache manager
        self.model_caches: dict[str, APIRequestCache] = {}

        self.rate_limiter = HeaderRateLimiter(self.rate_limit_config)

    def _get_cache(self, model: str) -> APIRequestCache:
        """Get or create cache for a model. Each model gets its own database file."""
        if model not in self.model_caches:
            # Create per-model database file
            # Sanitize model name for filename
            safe_model_name = model.replace("/", "_").replace(":", "_")
            db_path = self.cache_dir / f"{safe_model_name}.db"

            # Create per-model backend and manager
            backend = SQLiteCacheBackend(str(db_path))
            manager = ChunkedCacheManager(
                backend,
                max_chunks=self.cache_config.max_chunks_in_memory,
                entries_per_chunk=self.cache_config.entries_per_chunk,
            )

            self.model_caches[model] = APIRequestCache(
                model_name=model,
                backend=backend,
                manager=manager,
                response_type=OpenaiResponse
            )
        return self.model_caches[model]

    def _get_provider(self, model: str, provider_override: str | None = None) -> str:
        """
        Determine which provider to use.
        Default: always openrouter (as specified by user).
        """
        if provider_override:
            return provider_override
        return self.default_provider

    async def call_one(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: dict | None = None,
        reasoning: dict | None = None,
        extra_body: dict | None = None,
        tool_args: ToolArgs | None = None,
        disable_cache: bool = False,
    ) -> OpenaiResponse:
        """
        Make a single async API call.

        Args:
            messages: Chat messages (ChatHistory, list of ChatMessage, or single string)
            model: Model name (e.g., "anthropic/claude-3.5-sonnet", "gpt-4")
            provider: Override provider ("openrouter", "anthropic", "openai")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            response_format: Response format specification
            reasoning: Reasoning configuration for thinking models
            extra_body: Extra parameters for the API
            tool_args: Tool/function calling arguments
            disable_cache: Skip cache lookup and storage

        Returns:
            OpenaiResponse with the model's response
        """
        if isinstance(messages, str):
            messages = ChatHistory.from_user(messages)
        elif not isinstance(messages, ChatHistory):
            messages = ChatHistory(messages=messages)

        config = InferenceConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            reasoning=reasoning,
            extra_body=extra_body,
        )

        provider_to_use = self._get_provider(model, provider)

        should_cache = not disable_cache and model not in self.cache_config.no_cache_models

        if should_cache:
            cache = self._get_cache(model)
            cached_response = await cache.get_model_call(messages, config, tool_args)
            if cached_response:
                logger.debug(f"Cache hit for model {model}")
                return cached_response

        await self.rate_limiter.wait_if_needed(model, provider_to_use)

        response = await self._call_with_retry(
            messages, config, provider_to_use, tool_args
        )

        if should_cache and response.has_response() and not response.abnormal_finish:
            cache = self._get_cache(model)
            await cache.add_model_call(messages, config, response, tool_args)

        return response

    async def _call_with_retry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        provider: str,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """
        Make API call with automatic retry on transient errors.
        Uses exponential backoff configured via retry_config.
        """
        last_exception = None
        wait_time = self.retry_config.min_wait_seconds

        for attempt in range(self.retry_config.max_attempts):
            try:
                if provider == "openrouter":
                    return await self._call_openrouter(messages, config, tool_args)
                elif provider == "anthropic":
                    return await self._call_anthropic(messages, config, tool_args)
                elif provider == "openai":
                    return await self._call_openai(messages, config, tool_args)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.retry_config.max_attempts - 1:
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.retry_config.max_attempts}: "
                        f"{type(e).__name__}: {str(e)[:100]}. Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time)
                    wait_time = min(
                        wait_time * self.retry_config.exponential_multiplier,
                        self.retry_config.max_wait_seconds
                    )
                else:
                    logger.error(f"All {self.retry_config.max_attempts} retry attempts exhausted")
                    raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: retry loop completed without success or exception")

    async def _call_openrouter(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call OpenRouter API."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter API key not provided")

        # Handle thinking models
        if is_thinking_model(config.model):
            if config.reasoning is None:
                to_pass_reasoning = {"reasoning": OPENAI_OMIT}
            else:
                config.reasoning.pop("max_tokens", None)
                to_pass_reasoning = {"reasoning_effort": config.reasoning.get("effort")}
        else:
            to_pass_reasoning = {}

        # Provider-specific routing
        to_pass_extra_body = config.extra_body or {}
        if config.model == "meta-llama/llama-3.1-8b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["cerebras/fp16", "novita/fp8", "deepinfra/fp8"]}
            }
        elif config.model == "meta-llama/llama-3.1-70b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["deepinfra/turbo", "fireworks"]}
            }

        logger.debug(f"Calling OpenRouter with model: {config.model}")

        create_kwargs = {
            "model": config.model,
            "messages": [msg.to_openai_content() for msg in messages.messages],
        }

        if config.max_tokens is not None:
            create_kwargs["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            create_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            create_kwargs["top_p"] = config.top_p
        if config.frequency_penalty:
            create_kwargs["frequency_penalty"] = config.frequency_penalty
        if config.response_format is not None:
            create_kwargs["response_format"] = config.response_format
        if tool_args is not None:
            create_kwargs["tools"] = tool_args.tools
        if to_pass_extra_body:
            create_kwargs["extra_body"] = to_pass_extra_body
        if to_pass_reasoning:
            create_kwargs.update(to_pass_reasoning)

        try:
            chat_completion = await self.openrouter_client.chat.completions.create(**create_kwargs)
        except Exception as e:
            note = f"Model: {config.model}. Provider: openrouter"
            e.add_note(note)
            raise

        response = OpenaiResponse.model_validate(chat_completion.model_dump())

        # OpenRouter doesn't have rate limit headers, so no update needed

        return response

    async def _call_anthropic(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call Anthropic API directly."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not provided")

        # Separate system messages
        non_system = [msg for msg in messages.messages if msg.role != "system"]
        system_msgs = [msg for msg in messages.messages if msg.role == "system"]

        if len(system_msgs) > 1:
            raise ValueError("Anthropic does not support multiple system messages")

        system_content = system_msgs[0].content if system_msgs else ANTHROPIC_OMIT

        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in non_system
        ]

        # Handle thinking models
        if config.reasoning is not None and is_thinking_model(config.model):
            to_pass_thinking = {
                "type": "enabled",
                "budget_tokens": config.reasoning["max_tokens"],
            }
            to_pass_temperature = 1.0
        else:
            to_pass_thinking = ANTHROPIC_OMIT
            to_pass_temperature = config.temperature if config.temperature is not None else ANTHROPIC_OMIT

        if config.max_tokens is None:
            raise ValueError("Anthropic requires max_tokens")

        logger.debug(f"Calling Anthropic with model: {config.model}")

        create_kwargs = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
        }

        if system_content != ANTHROPIC_OMIT:
            create_kwargs["system"] = system_content
        if to_pass_temperature != ANTHROPIC_OMIT:
            create_kwargs["temperature"] = to_pass_temperature
        if config.top_p is not None:
            create_kwargs["top_p"] = config.top_p
        if to_pass_thinking != ANTHROPIC_OMIT:
            create_kwargs["thinking"] = to_pass_thinking
        if config.extra_body:
            create_kwargs["extra_body"] = config.extra_body

        raw_response: Message = await self.anthropic_client.messages.create(**create_kwargs)

        # Note: The anthropic SDK doesn't expose headers directly on the response object
        # We'd need to use httpx directly to get headers, which we'll skip for now
        # self.rate_limiter.update_from_headers(config.model, "anthropic", headers)

        if raw_response.content[0].type == "thinking":
            if len(raw_response.content) >= 2:
                response_content = {
                    "reasoning": raw_response.content[0].thinking,
                    "text": raw_response.content[1].text,
                }
            else:
                response_content = {
                    "reasoning": raw_response.content[0].thinking,
                    "text": "",
                }
        else:
            response_content = {
                "text": raw_response.content[0].text,
            }

        response = OpenaiResponse(
            id=raw_response.id,
            choices=[{"message": {"content": response_content, "role": "assistant"}, "finish_reason": "stop"}],
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=raw_response.usage.model_dump(),
        )

        return response

    async def _call_openai(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call OpenAI API directly."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not provided")

        # Handle thinking models
        if is_thinking_model(config.model):
            if config.reasoning is None:
                to_pass_reasoning = {"reasoning": OPENAI_OMIT}
            else:
                config.reasoning.pop("max_tokens", None)
                to_pass_reasoning = {"reasoning_effort": config.reasoning.get("effort")}
        else:
            to_pass_reasoning = {}

        logger.debug(f"Calling OpenAI with model: {config.model}")

        create_kwargs = {
            "model": config.model,
            "messages": [msg.to_openai_content() for msg in messages.messages],
        }

        if config.max_tokens is not None:
            create_kwargs["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            create_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            create_kwargs["top_p"] = config.top_p
        if config.frequency_penalty:
            create_kwargs["frequency_penalty"] = config.frequency_penalty
        if config.response_format is not None:
            create_kwargs["response_format"] = config.response_format
        if tool_args is not None:
            create_kwargs["tools"] = tool_args.tools
        if to_pass_reasoning:
            create_kwargs.update(to_pass_reasoning)

        chat_completion = await self.openai_client.chat.completions.create(**create_kwargs)

        response = OpenaiResponse.model_validate(chat_completion.model_dump())

        # Note: Similar to Anthropic, we'd need httpx to get headers
        # self.rate_limiter.update_from_headers(config.model, "openai", headers)

        return response

    def call_one_sync(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        **kwargs
    ) -> OpenaiResponse:
        """
        Synchronous wrapper around async call_one().

        Args:
            Same as call_one()

        Returns:
            OpenaiResponse
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "call_one_sync() cannot be called from an async context. "
                "Use await caller.call_one() instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                return asyncio.run(self.call_one(messages, model, **kwargs))
            else:
                raise

    async def call(
        self,
        messages: list[str | ChatHistory | Sequence[ChatMessage]] | list[dict],
        model: str | None = None,
        max_parallel: int = 10,
        **kwargs
    ) -> list[OpenaiResponse]:
        """
        Make multiple async API calls in parallel.

        Two usage modes:
        1. List of messages with shared parameters:
           call(["Hi", "Hello", "Hey"], model="gpt-4", max_tokens=10)

        2. List of request dicts with individual parameters:
           call([
               {"messages": "Hi", "model": "gpt-4", "max_tokens": 10},
               {"messages": "Hello", "model": "claude", "max_tokens": 20}
           ])

        Args:
            messages: Either a list of message inputs OR a list of request dicts
            model: Model name (required if using mode 1)
            max_parallel: Maximum number of parallel requests
            **kwargs: Additional parameters to apply to all requests (mode 1 only)

        Returns:
            List of OpenaiResponse objects
        """
        if not messages:
            return []

        # Determine which mode we're in
        if isinstance(messages[0], dict):
            # Mode 2: List of request dicts
            requests = messages
        else:
            # Mode 1: List of messages with shared parameters
            if model is None:
                raise ValueError("model parameter is required when passing a list of messages")

            requests = [
                {"messages": msg, "model": model, **kwargs}
                for msg in messages
            ]

        responses = await Slist(requests).par_map_async(
            func=lambda req: self.call_one(**req),
            max_par=max_parallel,
            tqdm=True,
        )
        return list(responses)

    async def flush(self):
        """Flush cache to disk. SQLite commits immediately, so this is mostly a no-op."""
        pass

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        await self.flush()
