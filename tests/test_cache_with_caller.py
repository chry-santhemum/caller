"""Test cache integration with caller patterns."""
import asyncio
import time
from pathlib import Path
from caller.llm_types import (
    APIRequestCache,
    ChatHistory,
    InferenceConfig,
    ToolArgs,
)
from caller.caller import OpenaiResponse
from caller.cache import SQLiteCacheBackend, ChunkedCacheManager


class CacheByModel:
    """Helper class to manage caches for multiple models (mimics internal behavior)."""
    def __init__(self, cache_path: Path, cache_type=OpenaiResponse):
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, APIRequestCache] = {}
        self.cache_type = cache_type

    def get_cache(self, model: str) -> APIRequestCache:
        if model not in self.cache:
            path = self.cache_path / f"{model}.jsonl"
            self.cache[model] = APIRequestCache(
                cache_path=path, response_type=self.cache_type
            )
        return self.cache[model]

    async def flush(self) -> None:
        for cache in self.cache.values():
            await cache.flush()


async def test_cache_by_model():
    """Test the CacheByModel wrapper with new SQLite backend."""
    print("Testing CacheByModel with SQLite backend...")

    # Setup
    test_cache_dir = Path(".test_cache_by_model")
    test_cache_dir.mkdir(exist_ok=True)

    cache_by_model = CacheByModel(test_cache_dir, cache_type=OpenaiResponse)

    # Create test data for multiple models
    messages = ChatHistory.from_user("Hello, world!")

    configs = [
        InferenceConfig(model="gpt-4", temperature=0.7),
        InferenceConfig(model="claude-sonnet", temperature=0.5),
        InferenceConfig(model="gpt-3.5-turbo", temperature=0.9),
    ]

    responses = [
        OpenaiResponse(
            choices=[{"message": {"content": "Response from GPT-4", "role": "assistant"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model="gpt-4"
        ),
        OpenaiResponse(
            choices=[{"message": {"content": "Response from Claude", "role": "assistant"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model="claude-sonnet"
        ),
        OpenaiResponse(
            choices=[{"message": {"content": "Response from GPT-3.5", "role": "assistant"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model="gpt-3.5-turbo"
        ),
    ]

    print("✓ Test data created for 3 models")

    # Test: Add entries for all models
    print("\nAdding entries to cache...")
    for config, response in zip(configs, responses):
        cache = cache_by_model.get_cache(config.model)
        await cache.add_model_call(
            messages=messages,
            config=config,
            response=response,
            tools=None
        )
        print(f"  ✓ Added entry for {config.model}")

    # Test: Retrieve entries for all models
    print("\nRetrieving entries from cache...")
    for config, expected_response in zip(configs, responses):
        cache = cache_by_model.get_cache(config.model)
        cached_response = await cache.get_model_call(
            messages=messages,
            config=config,
            tools=None
        )

        if cached_response is None:
            print(f"  ✗ Failed to retrieve for {config.model}")
            return False

        if cached_response.first_response != expected_response.first_response:
            print(f"  ✗ Response mismatch for {config.model}")
            return False

        print(f"  ✓ Retrieved for {config.model}: {cached_response.first_response}")

    # Test: Verify all models share same DB
    print("\nVerifying shared database...")
    db_path = test_cache_dir / "cache.db"
    if not db_path.exists():
        print("✗ Database file not found")
        return False

    print(f"✓ All models share database at: {db_path}")
    print(f"  Database size: {db_path.stat().st_size} bytes")

    # Test: Flush all caches
    print("\nTesting flush...")
    await cache_by_model.flush()
    print("✓ Flush completed for all caches")

    # Test: Cache miss for different messages
    print("\nTesting cache miss...")
    different_messages = ChatHistory.from_user("Different message")
    cache = cache_by_model.get_cache("gpt-4")
    result = await cache.get_model_call(
        messages=different_messages,
        config=configs[0],
        tools=None
    )

    if result is not None:
        print("✗ Found entry that shouldn't exist")
        return False

    print("✓ Cache miss works correctly")

    # Test: Cache hit after adding entry
    print("\nTesting cache hit after new entry...")
    new_response = OpenaiResponse(
        choices=[{"message": {"content": "New response", "role": "assistant"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        created=int(time.time()),
        model="gpt-4"
    )
    await cache.add_model_call(
        messages=different_messages,
        config=configs[0],
        response=new_response,
        tools=None
    )

    result = await cache.get_model_call(
        messages=different_messages,
        config=configs[0],
        tools=None
    )

    if result is None or result.first_response != new_response.first_response:
        print("✗ Failed to retrieve newly added entry")
        return False

    print("✓ Cache hit works for newly added entry")

    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    shutil.rmtree(test_cache_dir)
    print("✓ Test files cleaned up")

    print("\n✅ All CacheByModel tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_cache_by_model())
    exit(0 if success else 1)
