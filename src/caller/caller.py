"""
Unified Caller class for LLM API calls with caching, rate limiting, and retry logic.
"""
import caller.patches
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence, Literal
from json import JSONDecodeError
import httpx

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist

import openai
import anthropic
from openai import AsyncOpenAI
from openai._types import omit as OPENAI_OMIT
from anthropic import AsyncAnthropic

from caller.types import (
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
    OpenaiResponse,
)
from caller.cache import CacheConfig, Cache
from caller.rate_limiter import RateLimitConfig, HeaderRateLimiter


logger = logging.getLogger(__name__)


# TODO: Find a better way to do this
def is_thinking_model(model_name: str) -> bool:
    model_name_without_provider = model_name.split("/")[-1]
    THINKING_MODELS = [
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
        "gemini-2.5",
        "gpt-5",
        "o3",
        "deepseek-r1",
    ]
    for model in THINKING_MODELS:
        if model_name_without_provider.startswith(model):
            return True
    return False


# TODO: Some of these probably shouldn't be retried
RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic._exceptions.OverloadedError,
    ValidationError,
    JSONDecodeError,
    ValueError,
)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = 8  # Maximum number of retry attempts
    min_wait_seconds: float = 1.0  # Minimum wait time between retries
    max_wait_seconds: float = 30.0  # Maximum wait time between retries
    exponential_multiplier: float = 2.0  # Exponential backoff multiplier


class Caller:
    """
    Main LLM caller class.

    Configure caching behavior:
        ```python
        from caller import CacheConfig
        cache_config = CacheConfig(
            no_cache_models={"o1", "gpt-4o-realtime"},
            max_chunks_in_memory=20,      # Keep 20 chunks in RAM
            entries_per_chunk=100,        # 100 entries per chunk
        )
        caller = Caller(cache_config=cache_config)
        ```

    Configure rate limiting and retry behavior:
        ```python
        from caller import RateLimitConfig, RetryConfig
        rate_limit_config = RateLimitConfig(min_requests_remaining=10)
        retry_config = RetryConfig(max_attempts=5, max_wait_seconds=30)
        caller = Caller(
            rate_limit_config=rate_limit_config,
            retry_config=retry_config
        )
        ```
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai", "openrouter"] = "openrouter",
        api_key: str | None = None,
        dotenv_path: str | Path | None = None,
        cache_config: CacheConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Caller with API clients and caching.

        Args:
            cache_config: Cache configuration (CacheConfig object)
            rate_limit_config: Rate limiting configuration (RateLimitConfig object)
            retry_config: Retry behavior configuration (RetryConfig object)
        """
        load_dotenv(dotenv_path)
        self._provider = provider

        if provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = AsyncAnthropic(
                api_key=self.api_key
            )
        elif provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = AsyncOpenAI(
                api_key=self.api_key
            )

        assert self.api_key is not None

        self.cache_config = cache_config or CacheConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()

        self.cache_dir = Path(self.cache_config.base_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Each model will have its own database and chunk manager
        self.model_caches: dict[str, Cache] = {}
        self._cache_lock = asyncio.Lock()  # Lock for cache creation

        self.rate_limiter = HeaderRateLimiter(self.rate_limit_config)


    async def call(
        self,
        messages: Sequence[str | ChatHistory | Sequence[ChatMessage]] | Sequence[dict],
        max_parallel: int,
        desc: str = "",
        model: str | None = None,
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
            max_parallel: Maximum number of parallel requests
            desc: Description for tqdm
            model: Model name (required only if using mode 1)
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
            tqdm=desc != "",
            desc=desc,
        )
        return list(responses)


    async def call_one(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: dict | None = None,
        reasoning: str | int | None = None,
        extra_body: dict | None = None,
        tool_args: ToolArgs | None = None,
        disable_cache: bool = False,
    ) -> OpenaiResponse:
        """
        Make a single async API call.
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

        should_cache = (not disable_cache) and (model not in self.cache_config.no_cache_models)

        if should_cache:
            cache = await self._get_cache(model)
            cached_response = await cache.get_entry(messages, config, tool_args)
            if cached_response:
                logger.debug(f"Cache hit for model {model}")
                return cached_response

        await self.rate_limiter.wait_if_needed(model, self._provider)

        response = await self._call_with_retry(
            messages, config, tool_args
        )

        if should_cache and response.has_response() and not response.abnormal_finish:
            cache = await self._get_cache(model)
            await cache.put_entry(
                response=response,
                messages=messages,
                config=config,
                tools=tool_args,
            )

        return response

    async def _call_with_retry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """
        Make API call with automatic retry on transient errors.
        Uses exponential backoff configured via retry_config.
        """
        wait_time = self.retry_config.min_wait_seconds

        for attempt in range(self.retry_config.max_attempts):
            logger.debug(f"Attempt {attempt + 1}/{self.retry_config.max_attempts} to call {config.model}")
            try:
                if self._provider == "openrouter":
                    return await self._call_openrouter(messages, config, tool_args)
                elif self._provider == "anthropic":
                    return await self._call_anthropic(messages, config, tool_args)
                elif self._provider == "openai":
                    return await self._call_openai(messages, config, tool_args)

            except RETRYABLE_EXCEPTIONS as e:
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

    async def _call_openrouter(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call OpenRouter API."""
        assert self._provider == "openrouter"

        # Handle thinking models
        if is_thinking_model(config.model):
            if config.reasoning is None:
                to_pass_reasoning = {"reasoning": OPENAI_OMIT}
            elif isinstance(config.reasoning, int):
                to_pass_reasoning = {"reasoning": {"max_tokens": config.reasoning}}
            elif isinstance(config.reasoning, str):
                to_pass_reasoning = {"reasoning": {"effort": config.reasoning}}
            else:
                raise ValueError(f"Invalid reasoning parameter: {type(config.reasoning)}")
        else:
            to_pass_reasoning = {}

        # Provider-specific routing (to avoid unreliable providers)
        # You can add more here
        to_pass_extra_body = config.extra_body or {}
        to_pass_extra_body.update(to_pass_reasoning)
        if config.model == "meta-llama/llama-3.1-8b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["cerebras/fp16", "novita/fp8", "deepinfra/fp8"], 'allow_fallbacks': False}
            }
        elif config.model == "meta-llama/llama-3.1-70b-instruct":
            to_pass_extra_body = {
                "provider": {"order": ["deepinfra/turbo", "fireworks"], 'allow_fallbacks': False}
            }
        elif config.model.startswith("anthropic/"):
            to_pass_extra_body = {
                "provider": {"order": ["google-vertex", "anthropic"], 'allow_fallbacks': False}
            }

        logger.debug(f"Calling OpenRouter with model: {config.model}")

        create_kwargs = {
            "model": config.model,
            "messages": messages.to_openai_messages(),
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

        try:
            logger.debug(f"Calling OpenRouter with model: {config.model}")
            chat_completion = await self.client.chat.completions.create(**create_kwargs)
            logger.debug(f"Got response from OpenRouter for model: {config.model}")
        except Exception as e:
            note = f"Model: {config.model}. Provider: openrouter"
            e.add_note(note)
            raise

        response = OpenaiResponse.model_validate(chat_completion.model_dump())
        return response

    async def _call_anthropic(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call Anthropic API directly using httpx to get rate limit headers."""
        assert self._provider == "anthropic"

        # Separate system messages
        non_system = [msg for msg in messages.messages if msg.role != "system"]
        system_msgs = [msg for msg in messages.messages if msg.role == "system"]

        if len(system_msgs) > 1:
            raise ValueError("Anthropic does not support multiple system messages")

        system_content = system_msgs[0].content if system_msgs else None

        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in non_system
        ]

        # Handle thinking models
        if config.reasoning is not None and is_thinking_model(config.model):
            assert isinstance(config.reasoning, int)
            to_pass_thinking = {
                "type": "enabled",
                "budget_tokens": config.reasoning,
            }
            to_pass_temperature = 1.0
        else:
            to_pass_thinking = None
            to_pass_temperature = config.temperature

        if config.max_tokens is None:
            raise ValueError("Anthropic requires max_tokens")

        logger.debug(f"Calling Anthropic with model: {config.model}")

        # Build request body
        request_body = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
        }

        if system_content:
            request_body["system"] = system_content
        if to_pass_temperature is not None:
            request_body["temperature"] = to_pass_temperature
        if config.top_p is not None:
            request_body["top_p"] = config.top_p
        if to_pass_thinking:
            request_body["thinking"] = to_pass_thinking
        if config.extra_body:
            request_body.update(config.extra_body)

        # Make direct HTTP request
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=request_body, timeout=120.0)

                # Parse rate limit headers
                self.rate_limiter.update_from_headers(config.model, "anthropic", dict(response.headers))

                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_message = error_data.get("error", {}).get("message", response.text)

                    # Raise as ValueError which is in RETRYABLE_EXCEPTIONS
                    # We can't use Anthropic SDK exceptions as they require the SDK response object
                    raise ValueError(f"Anthropic API error ({response.status_code}): {error_message}")

                raw_response = response.json()

            except httpx.TimeoutException:
                raise ValueError("Request timed out")
            except httpx.ConnectError as e:
                raise ValueError(f"Connection error: {e}")

        # Process response based on content type
        if raw_response["content"][0]["type"] == "thinking":
            if len(raw_response["content"]) >= 2:
                response_content = {
                    "reasoning": raw_response["content"][0]["thinking"],
                    "text": raw_response["content"][1]["text"],
                }
            else:
                response_content = {
                    "reasoning": raw_response["content"][0]["thinking"],
                    "text": "",
                }
        else:
            response_content = {
                "text": raw_response["content"][0]["text"],
            }

        response = OpenaiResponse(
            id=raw_response["id"],
            choices=[{"message": {"content": response_content, "role": "assistant"}, "finish_reason": "stop"}],
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=raw_response["usage"],
        )

        return response

    async def _call_openai(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tool_args: ToolArgs | None,
    ) -> OpenaiResponse:
        """Call OpenAI API directly using httpx to get rate limit headers."""
        assert self._provider == "openai"

        # Handle thinking models
        if is_thinking_model(config.model):
            if config.reasoning is None:
                to_pass_reasoning = None
            else:
                assert isinstance(config.reasoning, str)
                to_pass_reasoning = {"reasoning_effort": config.reasoning}
        else:
            to_pass_reasoning = None

        logger.debug(f"Calling OpenAI with model: {config.model}")

        # Build request body
        request_body = {
            "model": config.model,
            "messages": [msg.to_openai_content() for msg in messages.messages],
        }

        if config.max_tokens is not None:
            request_body["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            request_body["temperature"] = config.temperature
        if config.top_p is not None:
            request_body["top_p"] = config.top_p
        if config.frequency_penalty:
            request_body["frequency_penalty"] = config.frequency_penalty
        if config.response_format is not None:
            request_body["response_format"] = config.response_format
        if tool_args is not None:
            request_body["tools"] = tool_args.tools
        if to_pass_reasoning:
            request_body.update(to_pass_reasoning)

        # Make direct HTTP request
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=request_body, timeout=120.0)

                # Parse rate limit headers
                self.rate_limiter.update_from_headers(config.model, "openai", dict(response.headers))

                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_message = error_data.get("error", {}).get("message", response.text)

                    # Raise as ValueError which is in RETRYABLE_EXCEPTIONS
                    # We can't use OpenAI SDK exceptions as they require the SDK response object
                    raise ValueError(f"OpenAI API error ({response.status_code}): {error_message}")

                raw_response = response.json()

            except httpx.TimeoutException:
                raise ValueError("Request timed out")
            except httpx.ConnectError as e:
                raise ValueError(f"Connection error: {e}")

        response = OpenaiResponse.model_validate(raw_response)

        return response

    async def _get_cache(self, model: str) -> Cache:
        """Get or create cache for a model. Each model gets its own database file."""
        async with self._cache_lock:  # Ensure only one cache is created per model
            if model not in self.model_caches:
                self.model_caches[model] = Cache(
                    model_name=model,
                    response_type=OpenaiResponse,
                    cache_config=self.cache_config,
                )
        return self.model_caches[model]

    async def close(self):
        """Close all cache connections."""
        for cache in self.model_caches.values():
            await cache.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup: close all cache connections."""
        await self.close()
        return False
