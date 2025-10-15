"""Rate limiting module."""

import asyncio
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting behavior."""

    # Thresholds for proactive waiting (only for providers with rate limit headers)
    min_requests_remaining: int = 5  # Wait if fewer than this many requests remaining
    min_tokens_remaining: int = 1000  # Wait if fewer than this many tokens remaining


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