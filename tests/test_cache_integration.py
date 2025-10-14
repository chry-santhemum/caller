"""Test script for cache integration."""
import asyncio
import os
import time
from pathlib import Path
from llm_types import (
    APIRequestCache,
    ChatHistory,
    InferenceConfig,
    InferenceResponse,
)


async def test_cache_integration():
    """Test the new SQLite-based cache implementation."""
    print("Testing SQLite Cache Integration...")

    # Setup test environment
    test_cache_dir = Path(".test_cache")
    test_cache_dir.mkdir(exist_ok=True)

    # Create cache instances for different models
    cache1 = APIRequestCache(
        cache_path=test_cache_dir / "gpt-4.jsonl",
        response_type=InferenceResponse
    )

    cache2 = APIRequestCache(
        cache_path=test_cache_dir / "claude-sonnet.jsonl",
        response_type=InferenceResponse
    )

    # Create test data
    messages = ChatHistory.from_user("What is the capital of France?")
    config = InferenceConfig(model="gpt-4", temperature=0.7)
    response = InferenceResponse(raw_responses=["Paris is the capital of France."])

    print("✓ Test data created")

    # Test 1: Add to cache
    print("\nTest 1: Adding entry to cache...")
    await cache1.add_model_call(
        messages=messages,
        config=config,
        response=response,
        tools=None
    )
    print("✓ Entry added to cache")

    # Test 2: Retrieve from cache
    print("\nTest 2: Retrieving entry from cache...")
    cached_response = await cache1.get_model_call(
        messages=messages,
        config=config,
        tools=None
    )

    if cached_response is None:
        print("✗ Failed to retrieve from cache")
        return False

    if cached_response.raw_responses[0] != response.raw_responses[0]:
        print("✗ Retrieved response doesn't match original")
        return False

    print("✓ Entry retrieved successfully")
    print(f"  Retrieved: {cached_response.raw_responses[0]}")

    # Test 3: Cache miss for different model
    print("\nTest 3: Testing cache isolation between models...")
    cached_response2 = await cache2.get_model_call(
        messages=messages,
        config=config,
        tools=None
    )

    if cached_response2 is not None:
        print("✗ Found entry in wrong model's cache")
        return False

    print("✓ Cache correctly isolated between models")

    # Test 4: Add to second model's cache
    print("\nTest 4: Adding entry to second model's cache...")
    response2 = InferenceResponse(raw_responses=["Paris is the capital and largest city of France."])
    await cache2.add_model_call(
        messages=messages,
        config=InferenceConfig(model="claude-sonnet", temperature=0.7),
        response=response2,
        tools=None
    )
    print("✓ Entry added to second model's cache")

    # Test 5: Both caches work independently
    print("\nTest 5: Verifying both caches work independently...")
    cached_response1 = await cache1.get_model_call(messages, config, None)
    cached_response2 = await cache2.get_model_call(
        messages,
        InferenceConfig(model="claude-sonnet", temperature=0.7),
        None
    )

    if cached_response1 is None or cached_response2 is None:
        print("✗ Failed to retrieve from one or both caches")
        return False

    print("✓ Both caches work independently")
    print(f"  Model 1: {cached_response1.raw_responses[0]}")
    print(f"  Model 2: {cached_response2.raw_responses[0]}")

    # Test 6: Verify shared database file
    print("\nTest 6: Verifying shared database file...")
    db_path = test_cache_dir / "cache.db"
    if not db_path.exists():
        print("✗ Database file not found")
        return False

    print(f"✓ Database file created at: {db_path}")
    print(f"  Database size: {db_path.stat().st_size} bytes")

    # Test 7: Test flush (should be no-op)
    print("\nTest 7: Testing flush...")
    await cache1.flush()
    await cache2.flush()
    print("✓ Flush completed (no-op)")

    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    shutil.rmtree(test_cache_dir)
    print("✓ Test files cleaned up")

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_cache_integration())
    exit(0 if success else 1)
