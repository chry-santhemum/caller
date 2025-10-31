"""
Unified Caller class for LLM API calls with caching, rate limiting, and retry logic.
"""
import caller.patches
import os
import httpx
import random
import asyncio
import logging
import requests
from pathlib import Path
from typing import Sequence, Literal, Optional
from json import JSONDecodeError
from slist import Slist
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

import openai
import anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from caller.types import (
    Tool,
    ToolChoice,
    ResponseFormat,
    ChatMessage,
    ChatHistory,
    InferenceConfig,
    Request,
    Response,
)
from caller.cache import CacheConfig, Cache


logger = logging.getLogger(__name__)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = 8  # Maximum number of retry attempts
    min_wait_seconds: float = 1.0  # Minimum wait time between retries
    max_wait_seconds: float = 30.0  # Maximum wait time between retries
    multiplier: float = 2.0  # Exponential backoff multiplier
    retryable_exceptions: tuple = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.PermissionDeniedError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic._exceptions.OverloadedError,
        ValidationError,
        JSONDecodeError,
        ValueError,
    )


class CallerBaseClass(ABC):

    def __init__(self, cache_config: Optional[CacheConfig] = None, retry_config: Optional[RetryConfig] = None) -> None:
        self.cache_config = cache_config or CacheConfig()
        self.retry_config = retry_config or RetryConfig()

        if self.cache_config.base_path is not None:
            self.cache_dir = Path(self.cache_config.base_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Each model will have its own database and chunk manager
            self.model_caches: dict[str, Cache] = {}
            self._cache_lock = asyncio.Lock()  # Lock for cache creation
        else:
            self.cache_dir = None  # caching disabled

    @abstractmethod
    async def _call(self, request: Request) -> Response:
        pass

    async def _call_with_retry(self, request: Request) -> Response:
        """
        Wraps _call() with automatic retry on transient errors.
        Uses jittered exponential backoff configured via retry_config.
        """
        wait_time = self.retry_config.min_wait_seconds

        for attempt in range(self.retry_config.max_attempts):
            logger.debug(f"Attempt {attempt + 1}/{self.retry_config.max_attempts} to call {request.model}")
            try:
                return await self._call(request)

            except self.retry_config.retryable_exceptions as e:
                if attempt < self.retry_config.max_attempts - 1:
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.retry_config.max_attempts}: "
                        f"{type(e).__name__}: {str(e)[:100]}. Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time + random.uniform(0, 1))
                    wait_time = min(
                        wait_time * self.retry_config.multiplier,
                        self.retry_config.max_wait_seconds
                    )
                else:
                    logger.error(f"All {self.retry_config.max_attempts} retry attempts exhausted")
                    raise

    async def call_one(
        self,
        messages: ChatHistory | Sequence[ChatMessage] | str,
        model: str,
        disable_cache: bool = False,
        response_format: Optional[dict] = None,  # pass in the desired json schema
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[str | int] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[ToolChoice] = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        top_a: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        top_logprobs: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> Response:
        """
        Make a single async API call.
        """
        if isinstance(messages, str):
            messages = ChatHistory.from_user(messages)
        elif not isinstance(messages, ChatHistory):
            messages = ChatHistory(messages=messages)

        config = InferenceConfig(
            response_format=ResponseFormat(type="json_schema", json_schema=response_format) if response_format is not None else None,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            seed=seed,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_a=top_a,
            logit_bias=logit_bias,
            top_logprobs=top_logprobs,
            reasoning=reasoning,
            extra_body=extra_body,
        )

        should_cache = (not disable_cache) and (model not in self.cache_config.no_cache_models) and (self.cache_dir is not None)

        if should_cache:
            cache = await self._get_cache(model)
            cached_response = await cache.get_entry(messages, config)
            if cached_response:
                logger.debug(f"Cache hit for model {model}")
                return cached_response

        response = await self._call_with_retry(
            Request(
                model=model,
                messages=messages,
                config=config,
            )
        )

        if should_cache and response.has_response and response.finish_reason == "stop":
            assert self.cache_dir is not None
            cache = await self._get_cache(model)
            await cache.put_entry(
                response=response,
                messages=messages,
                config=config,
            )

        return response


    async def call(
        self,
        messages: Sequence[ChatHistory | Sequence[ChatMessage] | str],
        model: str,
        max_parallel: int,
        desc: str = "",
        **kwargs
    ) -> list[Response]:
        """
        Make multiple async API calls in parallel.
        """
        if not messages:
            return []

        tasks = [
            {"messages": msg, "model": model, **kwargs}
            for msg in messages
        ]

        responses = await Slist(tasks).par_map_async(
            func=lambda task: self.call_one(**task),
            max_par=max_parallel,
            tqdm=desc != "",
            desc=desc,
        )
        return list(responses)

    async def _get_cache(self, model: str) -> Cache:
        """Get or create cache for a model. Each model gets its own database file."""
        async with self._cache_lock:  # Ensure only one cache is created per model
            if model not in self.model_caches:
                self.model_caches[model] = Cache(
                    model_name=model,
                    response_type=Response,
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
        await self.close()
        return False


class OpenRouterCaller(CallerBaseClass):

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        dotenv_path: Optional[str|Path] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        super().__init__(cache_config=cache_config, retry_config=retry_config)
        load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        assert self.api_key is not None, "API key not provided and not found in .env"


    def check_model_support(self, model_name: str, property: str) -> bool:
        """
        Check if a model supports various parameters.
        e.g. reasoning, tools, structured responses.
        """
        
        split_model_name = model_name.split("/")
        assert len(split_model_name) == 2, "Model name must be in the format of author/slug"
        author, slug = split_model_name

        url = f"https://openrouter.ai/api/v1/parameters/{author}/{slug}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        
        assert response.json()["data"]["model"] == model_name, "Model name does not match"
        return property in response.json()["data"]["supported_parameters"]


    async def _call(self, request: Request) -> Response:
        request_body = request.to_request()
        if request_body["extra_body"] is None:
            request_body["extra_body"] = {}
        request_body["extra_body"]["provider"] = {"require_parameters": True}

        # Provider-specific routing (to avoid unreliable providers)
        if request.model == "meta-llama/llama-3.1-8b-instruct":
            request_body["extra_body"]["provider"].update({
                "order": ["cerebras/fp16", "novita/fp8", "deepinfra/fp8"],
                "allow_fallbacks": False,
            })
        elif request.model == "meta-llama/llama-3.1-70b-instruct":
            request_body["extra_body"]["provider"].update({
                "order": ["deepinfra/turbo", "fireworks"],
                "allow_fallbacks": False,
            })
        elif request.model.startswith("anthropic/"):
            request_body["extra_body"]["provider"].update({
                "order": ["anthropic", "google-vertex"],
                "allow_fallbacks": False,
            })
        elif request.model.startswith("gpt-5"):
            request_body["extra_body"]["provider"].update({
                "order": ["openai"],
                "allow_fallbacks": False,
            })

        try:
            chat_completion = await self.client.chat.completions.create(**request_body)
        except Exception as e:
            note = f"Model: {request.model}. OpenRouter API error."
            e.add_note(note)
            raise

        return Response.model_validate(chat_completion.model_dump())
