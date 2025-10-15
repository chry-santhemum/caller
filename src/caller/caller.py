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

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from slist import Slist

import openai
import anthropic
from openai import AsyncOpenAI
from openai._types import omit as OPENAI_OMIT
from anthropic import AsyncAnthropic
from anthropic.types.message import Message
from anthropic._types import omit as ANTHROPIC_OMIT

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
            print(f"{model_name} is a thinking model")
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
        self.provider = provider

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

        # Each model will have its own database and cache manager
        self.model_caches: dict[str, Cache] = {}

        self.rate_limiter = HeaderRateLimiter(self.rate_limit_config)


    def _get_cache(self, model: str) -> Cache:
        """Get or create cache for a model. Each model gets its own database file."""
        if model not in self.model_caches:
            self.model_caches[model] = Cache(
                model_name=model,
                response_type=OpenaiResponse,
                cache_config=self.cache_config,
            )
        return self.model_caches[model]


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
        reasoning: dict | None = None,
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
            cache = self._get_cache(model)
            cached_response = await cache.get_entry(messages, config, tool_args)
            if cached_response:
                logger.debug(f"Cache hit for model {model}")
                return cached_response

        await self.rate_limiter.wait_if_needed(model, self.provider)

        response = await self._call_with_retry(
            messages, config, tool_args
        )

        if should_cache and response.has_response() and not response.abnormal_finish:
            cache = self._get_cache(model)
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
        last_exception = None
        wait_time = self.retry_config.min_wait_seconds

        for attempt in range(self.retry_config.max_attempts):
            try:
                if self.provider == "openrouter":
                    return await self._call_openrouter(messages, config, tool_args)
                elif self.provider == "anthropic":
                    return await self._call_anthropic(messages, config, tool_args)
                elif self.provider == "openai":
                    return await self._call_openai(messages, config, tool_args)

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
        assert self.provider == "openrouter"

        # Handle thinking models
        if is_thinking_model(config.model):
            if config.reasoning is None:
                to_pass_reasoning = {"reasoning": OPENAI_OMIT}
            else:
                to_pass_reasoning = {"reasoning": config.reasoning}
                assert ("max_tokens" in config.reasoning) ^ ("effort" in config.reasoning)
        else:
            to_pass_reasoning = {}

        # Provider-specific routing (to avoid unreliable providers)
        # You can add more here
        to_pass_extra_body = config.extra_body or {}
        to_pass_extra_body.update(to_pass_reasoning)
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
            chat_completion = await self.client.chat.completions.create(**create_kwargs)
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
        """Call Anthropic API directly."""
        assert self.provider == "anthropic"

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

        raw_response: Message = await self.client.messages.create(**create_kwargs)

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
        assert self.provider == "openai"

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

        chat_completion = await self.client.chat.completions.create(**create_kwargs)

        response = OpenaiResponse.model_validate(chat_completion.model_dump())

        # Note: Similar to Anthropic, we'd need httpx to get headers
        # self.rate_limiter.update_from_headers(config.model, "openai", headers)

        return response


    async def call(
        self,
        messages: list[str | ChatHistory | Sequence[ChatMessage]] | list[dict],
        max_parallel: int,
        model: str | None = None,
        desc: str = "",  # Description for tqdm
        disable_cache: bool = False,
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
                {"messages": msg, "model": model, "disable_cache": disable_cache, **kwargs}
                for msg in messages
            ]

        responses = await Slist(requests).par_map_async(
            func=lambda req: self.call_one(**req),
            max_par=max_parallel,
            tqdm=desc != "",
            desc=desc,
        )
        return list(responses)

