"""SQLite-based response caching."""

import time
from pathlib import Path
from collections import OrderedDict
import json
import hashlib
from pathlib import Path
from typing import Optional, Type, Sequence, Generic, TypeVar
from pydantic import BaseModel, ValidationError
import aiosqlite


from caller.types import (
    ChatHistory,
    InferenceConfig,
    ToolArgs,
)


APIResponse = TypeVar("APIResponse", bound=BaseModel)

def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
    other_hash: str,
    tools: ToolArgs | None,
) -> str:
    """Generate a deterministic cache key from API call inputs."""
    config_dump = config.model_dump_json(exclude_none=True)
    tools_json = tools.model_dump_json() if tools is not None else ""

    str_messages = (
        ",".join([str(msg) for msg in messages.messages])
        + deterministic_hash(config_dump)
        + tools_json
    )

    hash_of_history_not_messages = messages.model_dump(exclude_none=True)
    del hash_of_history_not_messages["messages"]
    str_history = (
        json.dumps(hash_of_history_not_messages) if hash_of_history_not_messages else ""
    )

    return deterministic_hash(str_messages + str_history + other_hash)



class CacheConfig(BaseModel):
    """Configuration for cache behavior."""

    # Models to never cache
    no_cache_models: set[str] = set()

    # In-memory cache settings
    max_chunks_in_memory: int = 64  # Max number of chunks to keep in memory
    entries_per_chunk: int = 128  # Number of entries per chunk

    # Disk cache settings
    max_entries_per_model: int | None = None  # Max entries per model (None = unlimited)

    # Cache loading behavior
    prefetch_adjacent_chunks: int = 0  # Load N adjacent chunks on cache miss (0 = disabled)


class CacheChunk(BaseModel):
    """Represents a fixed-size chunk of cache entries in memory."""

    model: str
    chunk_index: int  # Chunk number (0, 1, 2, ...)
    entries: dict[str, str]  # cache_key -> response_json
    last_accessed: float  # For LRU eviction


class CacheBackend:
    """Async SQLite backend for persistent cache storage."""

    def __init__(self, db_path: str | Path):
        """
        db_path: Path to the database file.
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    messages_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    tools_json TEXT,
                    created_at INTEGER NOT NULL
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_chunk
                ON cache_entries(model, chunk_index)
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

    async def get_model_entry_count(self, model: str) -> int:
        """Get the total number of entries for a model."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE model = ?",
                (model,)
            ) as cursor:
                count = (await cursor.fetchone())[0]
                return count

    async def put_entry(self, entry: dict) -> None:
        """Insert or replace a cache entry."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO cache_entries
                (cache_key, model, chunk_index, timestamp, messages_json, config_json,
                 response_json, tools_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["cache_key"],
                entry["model"],
                entry["chunk_index"],
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
        chunk_index: int
    ) -> list[dict]:
        """Get all entries in a specific chunk for a model."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM cache_entries
                WHERE model = ? AND chunk_index = ?
                ORDER BY timestamp ASC
            """, (model, chunk_index)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

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


class ChunkManager:
    """Manages in-memory chunks with LRU eviction and SQLite backend."""

    def __init__(
        self,
        backend: CacheBackend,
        max_chunks: int,
        entries_per_chunk: int
    ):
        self.backend = backend
        self.max_chunks = max_chunks
        self.entries_per_chunk = entries_per_chunk
        self.chunks: dict[tuple[str, int], CacheChunk] = {}
        self.access_order: OrderedDict[tuple[str, int], None] = OrderedDict()

    async def get_entry(
        self,
        cache_key: str,
        model: str,
        timestamp: int
    ) -> str | None:
        """Get response from cache. Queries DB directly then loads chunk if needed."""
        # First check if we have the key in any loaded chunk for this model
        for (m, chunk_idx), chunk in self.chunks.items():
            if m == model and cache_key in chunk.entries:
                self.access_order.move_to_end((m, chunk_idx))
                chunk.last_accessed = time.time()
                return chunk.entries[cache_key]

        # Not in memory, query database directly
        entry = await self.backend.get_entry(cache_key)
        if not entry:
            return None

        # Load the chunk that contains this entry
        chunk_index = entry["chunk_index"]
        await self.load_chunk(model, chunk_index)

        # Return from the loaded chunk
        chunk_key = (model, chunk_index)
        if chunk_key in self.chunks:
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
        # Calculate chunk index based on current entry count for this model
        entry_count = await self.backend.get_model_entry_count(model)
        chunk_index = entry_count // self.entries_per_chunk

        # Persist to database first
        entry = {
            "cache_key": cache_key,
            "model": model,
            "chunk_index": chunk_index,
            "timestamp": timestamp,
            "messages_json": messages_json,
            "config_json": config_json,
            "response_json": response,
            "tools_json": tools_json,
            "created_at": int(time.time())
        }
        await self.backend.put_entry(entry)

        # Add to in-memory chunk
        chunk_key = (model, chunk_index)

        # Load or create chunk
        if chunk_key not in self.chunks:
            await self.load_chunk(model, chunk_index)

        # If chunk still doesn't exist (no entries in DB for this chunk), create it
        if chunk_key not in self.chunks:
            self.chunks[chunk_key] = CacheChunk(
                model=model,
                chunk_index=chunk_index,
                entries={},
                last_accessed=time.time()
            )

        # Add entry to chunk
        self.chunks[chunk_key].entries[cache_key] = response
        self.chunks[chunk_key].last_accessed = time.time()

        # Update access order
        self.access_order.move_to_end(chunk_key)

    async def load_chunk(self, model: str, chunk_index: int) -> None:
        """Load chunk from database into memory."""
        chunk_key = (model, chunk_index)

        # Don't reload if already in memory
        if chunk_key in self.chunks:
            return

        # Evict LRU chunk if at max capacity
        if len(self.chunks) >= self.max_chunks:
            self.evict_lru_chunk()

        # Load from database
        entries_data = await self.backend.get_chunk(model, chunk_index)

        # Only create chunk if there are entries
        if entries_data:
            entries = {
                entry["cache_key"]: entry["response_json"]
                for entry in entries_data
            }

            chunk = CacheChunk(
                model=model,
                chunk_index=chunk_index,
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



class Cache(Generic[APIResponse]):
    def __init__(
        self,
        model_name: str,
        backend: CacheBackend,
        manager: ChunkManager,
        response_type: Type[APIResponse]
    ):
        """
        Cache for a specific model's API responses.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-3-5-sonnet")
            backend: Shared SQLite backend
            manager: Shared chunked cache manager
            response_type: Type to deserialize responses into
        """
        self.model_name = model_name
        self.backend = backend
        self.manager = manager
        self.response_type = response_type
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.backend.initialize()
            self._initialized = True

    async def add_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        response: APIResponse,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        """Store an API call and its response in the cache."""
        await self._ensure_initialized()

        key = file_cache_key(messages, config, other_hash, tools=tools)

        timestamp = int(time.time())
        response_json = response.model_dump_json()
        messages_json = messages.model_dump_json()
        config_json = config.model_dump_json()
        tools_json = tools.model_dump_json() if tools else None

        await self.manager.put_entry(
            cache_key=key,
            model=self.model_name,
            timestamp=timestamp,
            response=response_json,
            messages_json=messages_json,
            config_json=config_json,
            tools_json=tools_json
        )

    async def get_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[APIResponse]:
        """Retrieve a cached response for the given inputs, or None if not found."""
        await self._ensure_initialized()

        key = file_cache_key(messages, config, other_hash, tools=tools)
        timestamp = int(time.time())

        response_str = await self.manager.get_entry(key, self.model_name, timestamp)

        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                return response
            except ValidationError as e:
                print(f"Warning: Failed to validate cache entry for key {key}")
                return None
        return None

