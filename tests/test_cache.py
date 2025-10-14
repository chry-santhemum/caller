"""Test cache implementation."""

import time
import json
import os
import asyncio
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache import SQLiteCacheBackend, ChunkedCacheManager


async def test_cache():
    """Test the cache implementation."""
    # Use a test database
    test_db_path = ".api_cache/test_cache.db"

    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    print("Testing SQLite Cache Backend...")

    # Initialize backend
    backend = SQLiteCacheBackend(db_path=test_db_path)
    await backend.initialize()
    print("✓ Database initialized")

    # Test writing entries
    test_entry = {
        "cache_key": "test_key_1",
        "model": "gpt-4",
        "timestamp": int(time.time()),
        "messages_json": json.dumps([{"role": "user", "content": "Hello"}]),
        "config_json": json.dumps({"temperature": 0.7}),
        "response_json": json.dumps({"content": "Hi there!"}),
        "tools_json": None,
        "created_at": int(time.time())
    }

    await backend.put_entry(test_entry)
    print("✓ Entry written to database")

    # Test reading entry
    retrieved = await backend.get_entry("test_key_1")
    assert retrieved is not None
    assert retrieved["cache_key"] == "test_key_1"
    print("✓ Entry retrieved from database")

    # Test chunk manager
    print("\nTesting Chunked Cache Manager...")
    manager = ChunkedCacheManager(backend=backend, max_chunks=3)

    current_time = int(time.time())

    # Add entries
    await manager.put_entry(
        cache_key="chunk_test_1",
        model="gpt-4",
        timestamp=current_time,
        response=json.dumps({"response": "test 1"}),
        messages_json=json.dumps([]),
        config_json=json.dumps({}),
    )
    print("✓ Entry added via manager")

    # Retrieve entry
    response = await manager.get_entry("chunk_test_1", "gpt-4", current_time)
    assert response is not None
    print("✓ Entry retrieved via manager")

    # Test LRU eviction by adding more chunks than max_chunks
    print("\nTesting LRU eviction...")
    for i in range(5):
        # Use different timestamps to create different chunks
        ts = current_time + (i * 3600)  # 1 hour apart
        await manager.put_entry(
            cache_key=f"lru_test_{i}",
            model="gpt-4",
            timestamp=ts,
            response=json.dumps({"response": f"test {i}"}),
            messages_json=json.dumps([]),
            config_json=json.dumps({}),
        )

    # Should have max_chunks in memory
    assert len(manager.chunks) <= manager.max_chunks
    print(f"✓ LRU eviction working (chunks in memory: {len(manager.chunks)})")

    # Test stats
    stats = await backend.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Model counts: {stats['model_counts']}")
    print(f"  DB size: {stats['db_size_bytes']} bytes")

    # Test cleanup
    deleted = await backend.cleanup_old_entries(max_age_days=0)
    print(f"\n✓ Cleanup test completed (would delete {deleted} entries if they were old)")

    print("\n✅ All tests passed!")

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


if __name__ == "__main__":
    asyncio.run(test_cache())
