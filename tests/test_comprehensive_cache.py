"""Comprehensive cache test simulating real-world usage."""
import asyncio
import time
from pathlib import Path
from caller.llm_types import (
    ChatHistory,
    InferenceConfig,
)
from caller.caller import OpenaiResponse, CacheByModel


async def test_comprehensive_cache_behavior():
    """Test comprehensive cache behavior including performance."""
    print("="*60)
    print("Comprehensive Cache Test")
    print("="*60)

    # Setup
    test_cache_dir = Path(".test_comprehensive")
    test_cache_dir.mkdir(exist_ok=True)
    cache_by_model = CacheByModel(test_cache_dir, cache_type=OpenaiResponse)

    # Test 1: Multiple cache writes for same model
    print("\n[Test 1] Multiple writes for same model...")
    model = "gpt-4"
    cache = cache_by_model.get_cache(model)

    start_time = time.time()
    num_entries = 100
    for i in range(num_entries):
        messages = ChatHistory.from_user(f"Question {i}")
        config = InferenceConfig(model=model, temperature=0.7)
        response = OpenaiResponse(
            choices=[{"message": {"content": f"Answer {i}", "role": "assistant"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model=model
        )
        await cache.add_model_call(messages, config, response, None)

    write_time = time.time() - start_time
    print(f"  ✓ Wrote {num_entries} entries in {write_time:.2f}s ({num_entries/write_time:.1f} writes/sec)")

    # Test 2: Cache hits
    print("\n[Test 2] Testing cache hits...")
    start_time = time.time()
    hit_count = 0
    for i in range(num_entries):
        messages = ChatHistory.from_user(f"Question {i}")
        config = InferenceConfig(model=model, temperature=0.7)
        cached = await cache.get_model_call(messages, config, None)
        if cached and cached.first_response == f"Answer {i}":
            hit_count += 1

    read_time = time.time() - start_time
    print(f"  ✓ Retrieved {hit_count}/{num_entries} entries in {read_time:.2f}s ({hit_count/read_time:.1f} reads/sec)")

    if hit_count != num_entries:
        print(f"  ✗ Expected {num_entries} hits, got {hit_count}")
        return False

    # Test 3: Cache misses
    print("\n[Test 3] Testing cache misses...")
    start_time = time.time()
    miss_count = 0
    for i in range(10):
        messages = ChatHistory.from_user(f"Unknown question {i}")
        config = InferenceConfig(model=model, temperature=0.7)
        cached = await cache.get_model_call(messages, config, None)
        if cached is None:
            miss_count += 1

    miss_time = time.time() - start_time
    print(f"  ✓ Verified {miss_count}/10 cache misses in {miss_time:.2f}s")

    # Test 4: Multiple models
    print("\n[Test 4] Testing multiple models...")
    models = ["gpt-4", "gpt-3.5-turbo", "claude-sonnet", "claude-haiku"]
    for model_name in models:
        cache = cache_by_model.get_cache(model_name)
        messages = ChatHistory.from_user("Test message")
        config = InferenceConfig(model=model_name, temperature=0.5)
        response = OpenaiResponse(
            choices=[{"message": {"content": f"Response from {model_name}", "role": "assistant"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model=model_name
        )
        await cache.add_model_call(messages, config, response, None)

    # Verify all models have separate cache entries
    for model_name in models:
        cache = cache_by_model.get_cache(model_name)
        messages = ChatHistory.from_user("Test message")
        config = InferenceConfig(model=model_name, temperature=0.5)
        cached = await cache.get_model_call(messages, config, None)
        if cached is None or cached.first_response != f"Response from {model_name}":
            print(f"  ✗ Failed to retrieve for {model_name}")
            return False

    print(f"  ✓ Successfully cached and retrieved for {len(models)} different models")

    # Test 5: Different temperatures create different cache entries
    print("\n[Test 5] Testing parameter sensitivity...")
    messages = ChatHistory.from_user("Temperature test")
    temps = [0.0, 0.5, 0.7, 1.0]
    cache = cache_by_model.get_cache("gpt-4")

    for temp in temps:
        config = InferenceConfig(model="gpt-4", temperature=temp)
        response = OpenaiResponse(
            choices=[{"message": {"content": f"Response at temp {temp}", "role": "assistant"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            created=int(time.time()),
            model="gpt-4"
        )
        await cache.add_model_call(messages, config, response, None)

    # Verify all temperatures have separate entries
    for temp in temps:
        config = InferenceConfig(model="gpt-4", temperature=temp)
        cached = await cache.get_model_call(messages, config, None)
        if cached is None or cached.first_response != f"Response at temp {temp}":
            print(f"  ✗ Failed to retrieve for temperature {temp}")
            return False

    print(f"  ✓ Successfully cached {len(temps)} different temperature configs separately")

    # Test 6: Verify shared database
    print("\n[Test 6] Verifying database structure...")
    db_path = test_cache_dir / "cache.db"
    if not db_path.exists():
        print("  ✗ Database file not found")
        return False

    db_size = db_path.stat().st_size
    print(f"  ✓ Database file: {db_path}")
    print(f"  ✓ Database size: {db_size:,} bytes ({db_size/1024:.1f} KB)")

    # Test 7: Flush
    print("\n[Test 7] Testing flush...")
    await cache_by_model.flush()
    print("  ✓ Flush completed successfully")

    # Test 8: Persistence - create new cache instances and verify data persists
    print("\n[Test 8] Testing persistence...")
    new_cache_by_model = CacheByModel(test_cache_dir, cache_type=OpenaiResponse)
    new_cache = new_cache_by_model.get_cache("gpt-4")

    messages = ChatHistory.from_user("Question 0")
    config = InferenceConfig(model="gpt-4", temperature=0.7)
    cached = await new_cache.get_model_call(messages, config, None)

    if cached is None or cached.first_response != "Answer 0":
        print("  ✗ Data not persisted correctly")
        return False

    print("  ✓ Data persists across cache instances")

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  - Total entries added: {num_entries + len(models) + len(temps)}")
    print(f"  - Models tested: {len(models)}")
    print(f"  - Write speed: {num_entries/write_time:.1f} entries/sec")
    print(f"  - Read speed: {hit_count/read_time:.1f} entries/sec")
    print(f"  - Database size: {db_size/1024:.1f} KB")
    print("="*60)

    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    shutil.rmtree(test_cache_dir)
    print("✓ Test files cleaned up")

    print("\n✅ All comprehensive tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_cache_behavior())
    exit(0 if success else 1)
