"""SQLite-based chunked cache system for LLM API responses."""

import time
import json
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import aiosqlite
from pydantic import BaseModel


class CacheChunk(BaseModel):
    """Represents a time-windowed chunk of cache entries in memory."""

    model: str
    time_window_start: int  # Unix timestamp at hour boundary
    time_window_end: int
    entries: dict[str, str]  # cache_key -> response_json
    last_accessed: float  # For LRU eviction


class SQLiteCacheBackend:
    """Async SQLite backend for persistent cache storage."""

    def __init__(self, db_path: str = ".cache/caller/cache.db"):
        self.db_path = db_path
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    messages_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    tools_json TEXT,
                    created_at INTEGER NOT NULL
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_timestamp
                ON cache_entries(model, timestamp)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache_entries(created_at)
            """)

            await db.commit()

    async def get_entry(self, cache_key: str) -> dict | None:
        """Get a single cache entry by key."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM cache_entries WHERE cache_key = ?",
                (cache_key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None

    async def put_entry(self, entry: dict) -> None:
        """Insert or replace a cache entry."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO cache_entries
                (cache_key, model, timestamp, messages_json, config_json,
                 response_json, tools_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["cache_key"],
                entry["model"],
                entry["timestamp"],
                entry["messages_json"],
                entry["config_json"],
                entry["response_json"],
                entry.get("tools_json"),
                entry["created_at"]
            ))
            await db.commit()

    async def get_chunk(
        self,
        model: str,
        chunk_start: int,
        chunk_end: int
    ) -> list[dict]:
        """Get all entries in a time window for a specific model."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM cache_entries
                WHERE model = ? AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
            """, (model, chunk_start, chunk_end)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def cleanup_old_entries(self, max_age_days: int) -> int:
        """Delete entries older than max_age_days and return count deleted."""
        cutoff_timestamp = int(time.time()) - (max_age_days * 24 * 3600)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM cache_entries WHERE created_at < ?",
                (cutoff_timestamp,)
            )
            deleted_count = cursor.rowcount
            await db.commit()
            return deleted_count

    async def get_stats(self) -> dict:
        """Return cache statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            # Total entries
            async with db.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ) as cursor:
                total_entries = (await cursor.fetchone())[0]

            # Entries per model
            async with db.execute("""
                SELECT model, COUNT(*) as count
                FROM cache_entries
                GROUP BY model
            """) as cursor:
                model_counts = {row[0]: row[1] for row in await cursor.fetchall()}

            # Oldest entry timestamp
            async with db.execute(
                "SELECT MIN(created_at) FROM cache_entries"
            ) as cursor:
                oldest_entry = (await cursor.fetchone())[0]

            # Database file size
            db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

            return {
                "total_entries": total_entries,
                "model_counts": model_counts,
                "oldest_entry_timestamp": oldest_entry,
                "db_size_bytes": db_size,
            }


class ChunkedCacheManager:
    """Manages in-memory chunks with LRU eviction and SQLite backend."""

    def __init__(
        self,
        backend: SQLiteCacheBackend,
        max_chunks: int = 10,
        window_size: int = 3600  # 1 hour default
    ):
        self.backend = backend
        self.max_chunks = max_chunks
        self.window_size = window_size
        self.chunks: dict[tuple[str, int], CacheChunk] = {}
        self.access_order: OrderedDict[tuple[str, int], None] = OrderedDict()

    def get_chunk_id(self, timestamp: int) -> int:
        """Calculate chunk ID (timestamp aligned to hour boundary)."""
        return (timestamp // self.window_size) * self.window_size

    async def get_entry(
        self,
        cache_key: str,
        model: str,
        timestamp: int
    ) -> str | None:
        """Get response from chunk or load from database."""
        chunk_id = self.get_chunk_id(timestamp)
        chunk_key = (model, chunk_id)

        # Check if chunk is in memory
        if chunk_key not in self.chunks:
            await self.load_chunk(model, chunk_id)

        # Update access order for LRU
        if chunk_key in self.chunks:
            self.access_order.move_to_end(chunk_key)
            self.chunks[chunk_key].last_accessed = time.time()

            # Return entry if it exists in chunk
            return self.chunks[chunk_key].entries.get(cache_key)

        return None

    async def put_entry(
        self,
        cache_key: str,
        model: str,
        timestamp: int,
        response: str,
        messages_json: str,
        config_json: str,
        tools_json: str | None = None
    ) -> None:
        """Add entry to chunk and persist to database."""
        # Persist to database first
        entry = {
            "cache_key": cache_key,
            "model": model,
            "timestamp": timestamp,
            "messages_json": messages_json,
            "config_json": config_json,
            "response_json": response,
            "tools_json": tools_json,
            "created_at": int(time.time())
        }
        await self.backend.put_entry(entry)

        # Add to in-memory chunk
        chunk_id = self.get_chunk_id(timestamp)
        chunk_key = (model, chunk_id)

        # Load or create chunk
        if chunk_key not in self.chunks:
            await self.load_chunk(model, chunk_id)

        # If chunk still doesn't exist (no entries in DB for this window), create it
        if chunk_key not in self.chunks:
            chunk_end = chunk_id + self.window_size
            self.chunks[chunk_key] = CacheChunk(
                model=model,
                time_window_start=chunk_id,
                time_window_end=chunk_end,
                entries={},
                last_accessed=time.time()
            )

        # Add entry to chunk
        self.chunks[chunk_key].entries[cache_key] = response
        self.chunks[chunk_key].last_accessed = time.time()

        # Update access order
        self.access_order.move_to_end(chunk_key)

    async def load_chunk(self, model: str, chunk_id: int) -> None:
        """Load chunk from database into memory."""
        chunk_key = (model, chunk_id)

        # Don't reload if already in memory
        if chunk_key in self.chunks:
            return

        # Evict LRU chunk if at max capacity
        if len(self.chunks) >= self.max_chunks:
            self.evict_lru_chunk()

        # Load from database
        chunk_end = chunk_id + self.window_size
        entries_data = await self.backend.get_chunk(model, chunk_id, chunk_end)

        # Only create chunk if there are entries
        if entries_data:
            entries = {
                entry["cache_key"]: entry["response_json"]
                for entry in entries_data
            }

            chunk = CacheChunk(
                model=model,
                time_window_start=chunk_id,
                time_window_end=chunk_end,
                entries=entries,
                last_accessed=time.time()
            )

            self.chunks[chunk_key] = chunk
            self.access_order[chunk_key] = None

    def evict_lru_chunk(self) -> None:
        """Remove least recently used chunk from memory."""
        if not self.access_order:
            return

        # Get least recently used chunk (first item in OrderedDict)
        lru_chunk_key = next(iter(self.access_order))

        # Remove from both dictionaries
        del self.chunks[lru_chunk_key]
        del self.access_order[lru_chunk_key]
