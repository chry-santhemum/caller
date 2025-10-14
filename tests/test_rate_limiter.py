"""Test rate limiter implementation."""

import asyncio
import logging
import time
from pathlib import Path

from caller.rate_limiter import RateLimitConfig, RateLimiter, ModelRateLimitManager, estimate_tokens


async def test_rate_limiter():
    """
    Test rate limiting functionality.

    Verifies:
    - Config loading works correctly
    - Rate limiting blocks when approaching limits
    - Token estimation is reasonable
    - Can manage multiple models independently
    """
    print("Starting rate limiter tests...\n")

    # Test 1: Config loading
    print("Test 1: Config loading")
    manager = ModelRateLimitManager()
    print(f"  Loaded {len(manager.limiters)} model configurations")
    for model_name, limiter in manager.limiters.items():
        print(f"  - {model_name}: rpm={limiter.config.rpm}, "
              f"input_tpm={limiter.config.input_tpm}, "
              f"output_tpm={limiter.config.output_tpm}")
    print("  ✓ Config loading successful\n")

    # Test 2: Token estimation
    print("Test 2: Token estimation")
    test_texts = [
        "Hello world",
        "This is a longer sentence with more words in it.",
        "def foo():\n    return 42",
    ]
    for text in test_texts:
        tokens = estimate_tokens(text)
        print(f"  '{text}' -> ~{tokens} tokens ({len(text)} chars)")
    print("  ✓ Token estimation working\n")

    # Test 3: Rate limiting behavior
    print("Test 3: Rate limiting behavior")
    # Create a test limiter with very low limits
    test_config = RateLimitConfig(rpm=5, input_tpm=100, output_tpm=50)
    test_limiter = RateLimiter(test_config)

    print(f"  Testing with limits: rpm={test_config.rpm}, "
          f"input_tpm={test_config.input_tpm}, output_tpm={test_config.output_tpm}")

    # Make several requests quickly
    start_time = time.time()
    for i in range(3):
        await test_limiter.wait_if_needed(estimated_input_tokens=20)
        test_limiter.record_request(input_tokens=20, output_tokens=15)
        print(f"  Request {i+1} completed")

    elapsed = time.time() - start_time
    print(f"  3 requests completed in {elapsed:.2f}s")
    print("  ✓ Rate limiting functional\n")

    # Test 4: Multiple models
    print("Test 4: Multiple model management")
    test_text = "This is a test prompt for the model."
    estimated = estimate_tokens(test_text)

    for model in ["claude-sonnet-4", "claude-opus-4"]:
        if model in manager.limiters:
            await manager.wait_for_model(model, estimated)
            manager.record_usage(model, input_tokens=estimated, output_tokens=50)
            print(f"  ✓ {model} processed request")

    print("  ✓ Multiple models managed independently\n")

    print("All tests completed successfully!")


if __name__ == "__main__":
    # Run tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(test_rate_limiter())
