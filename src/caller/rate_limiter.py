"""
Rate limiting system with sliding window algorithm for LLM API calls.

Implements per-model rate limiting for:
- Requests per minute (RPM)
- Input tokens per minute (TPM)
- Output tokens per minute (TPM)
"""

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Configuration for rate limits on a single model."""

    rpm: Optional[int] = None           # Requests per minute
    input_tpm: Optional[int] = None     # Input tokens per minute
    output_tpm: Optional[int] = None    # Output tokens per minute


class RateLimiter:
    """
    Implements sliding window rate limiting for a single model.

    Tracks requests and token usage over a 60-second sliding window,
    blocking new requests when limits would be exceeded.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter with configuration.

        Args:
            config: Rate limit configuration specifying rpm, input_tpm, output_tpm
        """
        self.config = config
        self.request_times: deque[float] = deque()
        self.input_tokens: deque[tuple[float, int]] = deque()
        self.output_tokens: deque[tuple[float, int]] = deque()

    async def wait_if_needed(self, estimated_input_tokens: int = 0) -> None:
        """
        Block until request can proceed without exceeding rate limits.

        Uses sliding window algorithm to check if adding a new request
        would exceed any configured limits. If so, waits until it's safe.

        Args:
            estimated_input_tokens: Estimated tokens for the upcoming request
        """
        current_time = time.time()

        # Clean up old entries outside the 60-second window
        self._clean_old_entries(current_time)

        # Calculate how long we need to wait
        wait_time = self._calculate_wait_time(estimated_input_tokens, current_time)

        if wait_time > 0:
            logger.info(
                f"Rate limit approaching - waiting {wait_time:.2f}s "
                f"(requests: {len(self.request_times)}, "
                f"input_tokens: {sum(t[1] for t in self.input_tokens)}, "
                f"output_tokens: {sum(t[1] for t in self.output_tokens)})"
            )
            await asyncio.sleep(wait_time)

            # Clean again after waiting
            current_time = time.time()
            self._clean_old_entries(current_time)

    def record_request(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record a completed request with actual token usage.

        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens generated
        """
        current_time = time.time()
        self.request_times.append(current_time)
        self.input_tokens.append((current_time, input_tokens))
        self.output_tokens.append((current_time, output_tokens))

        logger.debug(
            f"Recorded request: {input_tokens} input tokens, {output_tokens} output tokens"
        )

    def _clean_old_entries(self, current_time: float) -> None:
        """
        Remove entries older than 60 seconds from all tracking queues.

        Args:
            current_time: Current timestamp to compare against
        """
        cutoff_time = current_time - 60.0

        # Remove old request times
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()

        # Remove old input token entries
        while self.input_tokens and self.input_tokens[0][0] < cutoff_time:
            self.input_tokens.popleft()

        # Remove old output token entries
        while self.output_tokens and self.output_tokens[0][0] < cutoff_time:
            self.output_tokens.popleft()

    def _calculate_wait_time(
        self,
        estimated_input_tokens: int,
        current_time: float
    ) -> float:
        """
        Calculate required wait time to avoid exceeding any rate limit.

        Checks each configured limit (rpm, input_tpm, output_tpm) and
        determines how long to wait if adding the new request would exceed any limit.

        Args:
            estimated_input_tokens: Estimated tokens for upcoming request
            current_time: Current timestamp

        Returns:
            Wait time in seconds (0 if no waiting needed)
        """
        wait_times = []

        # Check RPM limit
        if self.config.rpm is not None:
            current_requests = len(self.request_times)
            if current_requests >= self.config.rpm:
                # Need to wait until oldest request falls outside window
                oldest_time = self.request_times[0]
                wait_time = oldest_time + 60.0 - current_time
                wait_times.append(max(0, wait_time))

        # Check input TPM limit
        if self.config.input_tpm is not None and estimated_input_tokens > 0:
            current_input = sum(tokens for _, tokens in self.input_tokens)
            if current_input + estimated_input_tokens > self.config.input_tpm:
                # Need to wait until we have enough budget
                # Find when oldest tokens will expire
                if self.input_tokens:
                    oldest_time = self.input_tokens[0][0]
                    wait_time = oldest_time + 60.0 - current_time
                    wait_times.append(max(0, wait_time))

        # Check output TPM limit (conservative - assume similar to input)
        if self.config.output_tpm is not None:
            current_output = sum(tokens for _, tokens in self.output_tokens)
            # Assume output will be similar to input (conservative estimate)
            estimated_output_tokens = estimated_input_tokens
            if current_output + estimated_output_tokens > self.config.output_tpm:
                if self.output_tokens:
                    oldest_time = self.output_tokens[0][0]
                    wait_time = oldest_time + 60.0 - current_time
                    wait_times.append(max(0, wait_time))

        # Return the maximum wait time needed
        return max(wait_times) if wait_times else 0.0


class ModelRateLimitManager:
    """
    Manages rate limiters for all models in the system.

    Loads rate limit configurations from config.json and provides
    a unified interface for rate limiting across multiple models.
    """

    def __init__(self, config_path: Path | str = Path("config.json")):
        """
        Initialize manager and load rate limits from config file.

        Args:
            config_path: Path to config.json file
        """
        self.config_path = Path(config_path)
        self.limiters: dict[str, RateLimiter] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load rate limit configurations from config.json."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            rate_limits = config_data.get('rate_limits', {})

            for model_name, limits in rate_limits.items():
                config = RateLimitConfig(**limits)
                self.limiters[model_name] = RateLimiter(config)
                logger.info(
                    f"Loaded rate limits for {model_name}: "
                    f"rpm={config.rpm}, input_tpm={config.input_tpm}, "
                    f"output_tpm={config.output_tpm}"
                )

            if not self.limiters:
                logger.warning("No rate limits configured in config.json")

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    async def wait_for_model(
        self,
        model: str,
        estimated_input_tokens: int = 0
    ) -> None:
        """
        Wait if rate limit would be exceeded for the given model.

        If the model has no rate limits configured, returns immediately.

        Args:
            model: Model identifier
            estimated_input_tokens: Estimated tokens for the upcoming request
        """
        if model in self.limiters:
            await self.limiters[model].wait_if_needed(estimated_input_tokens)
        else:
            logger.debug(f"No rate limits configured for model: {model}")

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """
        Record actual token usage from API response.

        Args:
            model: Model identifier
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens generated
        """
        if model in self.limiters:
            self.limiters[model].record_request(input_tokens, output_tokens)
        else:
            logger.debug(f"No rate limits to record for model: {model}")


def estimate_tokens(text: str) -> int:
    """
    Simple character-based token estimation.

    Uses a conservative estimate of ~4 characters per token,
    which works reasonably well for English text and code.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated number of tokens (minimum 1)
    """
    return max(1, len(text) // 4)
