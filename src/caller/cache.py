"""SQLite-based response caching."""

import time
import hashlib
import asyncio
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Type, Generic, TypeVar
from pydantic import BaseModel, ValidationError
import aiosqlite

from caller.types import (
    ChatHistory,
    InferenceConfig,
)


APIResponse = TypeVar("APIResponse", bound=BaseModel)

def deterministic_hash(input: str) -> str:
    return hashlib.sha1(input.encode()).hexdigest()

def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
) -> tuple[str, str]:
    """Returns: str_key, cache_key"""
    messages_json = messages.model_dump_json(exclude_none=True)
    config_dump = config.model_dump_json(exclude_none=True)

    str_key = messages_json + config_dump
    cache_key = deterministic_hash(str_key)

    return str_key, cache_key



class CacheConfig(BaseModel):
    """Configuration for cache behavior."""

    base_path: str | None = ".cache/caller"  # None = disable caching

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


class Backend:
    """Async SQLite backend for persistent cache storage."""

    def __init__(self, model: str, db_path: str | Path = ".cache/caller"):
        """
        model: Model name
        db_path: Path to the cache base directory.
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.safe_model_name = model.replace("/", "_").replace(":", "_")
        self.db_path = db_path / f"{self.safe_model_name}.db"
        self._connection: aiosqlite.Connection | None = None


    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create a persistent connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
        return self._connection

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        db = await self._get_connection()

        await db.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                str_key TEXT NOT NULL,
                response_json TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_index
            ON cache_entries(chunk_index)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON cache_entries(created_at)
        """)

        await db.commit()


    async def get_entry(self, cache_key: str) -> dict | None:
        """Get a single cache entry by key."""
        db = await self._get_connection()
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM cache_entries WHERE cache_key = ?",
            (cache_key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

    async def get_entry_count(self) -> int:
        """Get the total number of entries."""
        db = await self._get_connection()
        async with db.execute(
            "SELECT COUNT(*) FROM cache_entries"
        ) as cursor:
            count = (await cursor.fetchone())[0]
            return count

    async def put_entry(self, entry: dict) -> None:
        """Insert or replace a cache entry."""
        db = await self._get_connection()

        await db.execute("""
            INSERT OR REPLACE INTO cache_entries
            (cache_key, str_key, response_json, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry["cache_key"],
            entry["str_key"],
            entry["response_json"],
            entry["chunk_index"],
            entry["created_at"]
        ))
        await db.commit()

    async def get_chunk(
        self,
        chunk_index: int
    ) -> list[dict]:
        """Get all entries in a specific chunk for a model."""
        db = await self._get_connection()
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM cache_entries
            WHERE chunk_index = ?
            ORDER BY created_at ASC
        """, (chunk_index,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None


class ChunkManager:
    """Manages in-memory chunks with LRU eviction and SQLite backend."""

    def __init__(
        self,
        backend: Backend,
        max_chunks: int,
        entries_per_chunk: int
    ):
        self.backend = backend
        self.model = backend.model
        self.max_chunks = max_chunks
        self.entries_per_chunk = entries_per_chunk

        self.chunks: dict[int, CacheChunk] = {}
        self.access_order: OrderedDict[int, None] = OrderedDict()

    async def get_entry(
        self,
        cache_key: str,
    ) -> str | None:
        """Get response from cache. Queries DB directly then loads chunk if needed."""
        # Check if we have the key in memory
        for chunk_idx, chunk in self.chunks.items():
            if cache_key in chunk.entries:
                self.access_order.move_to_end(chunk_idx)
                chunk.last_accessed = time.time()
                return chunk.entries[cache_key]

        # Not in memory, query database
        entry = await self.backend.get_entry(cache_key)
        if not entry:
            return None

        # Load the chunk that contains this entry
        chunk_index = entry["chunk_index"]
        await self.load_chunk(chunk_index)

        # Return from the loaded chunk
        if chunk_index in self.chunks:
            return self.chunks[chunk_index].entries.get(cache_key)

        return None

    async def put_entry(
        self,
        cache_key: str,
        str_key: str,
        response: str,
    ) -> None:
        """Add entry to chunk and persist to database."""
        # Calculate chunk index based on current entry count for this model
        entry_count = await self.backend.get_entry_count()
        chunk_index = entry_count // self.entries_per_chunk

        # Persist to database first
        entry = {
            "cache_key": cache_key,
            "str_key": str_key,
            "chunk_index": chunk_index,
            "response_json": response,
            "created_at": int(time.time())
        }
        await self.backend.put_entry(entry)

        # Load or create chunk
        if chunk_index not in self.chunks:
            await self.load_chunk(chunk_index)

        # If chunk still doesn't exist (no entries in DB for this chunk), create it
        if chunk_index not in self.chunks:
            self.chunks[chunk_index] = CacheChunk(
                model=self.model,
                chunk_index=chunk_index,
                entries={},
                last_accessed=time.time()
            )

        # Add entry to chunk
        self.chunks[chunk_index].entries[cache_key] = response
        self.chunks[chunk_index].last_accessed = time.time()

        # Update access order
        self.access_order.move_to_end(chunk_index)

    async def load_chunk(self, chunk_index: int) -> None:
        """Load chunk from database into memory."""
        if chunk_index in self.chunks:
            return

        # Evict LRU chunk if at max capacity
        if len(self.chunks) >= self.max_chunks:
            self.evict_lru_chunk()

        # Load from database
        entries_data = await self.backend.get_chunk(chunk_index)

        # Only create chunk if there are entries
        if entries_data:
            entries = {
                entry["cache_key"]: entry["response_json"]
                for entry in entries_data
            }

            chunk = CacheChunk(
                model=self.model,
                chunk_index=chunk_index,
                entries=entries,
                last_accessed=time.time()
            )
            self.chunks[chunk_index] = chunk
            self.access_order[chunk_index] = None

    def evict_lru_chunk(self) -> None:
        """Remove least recently used chunk from memory."""
        if not self.access_order:
            return

        # Get least recently used chunk (first item in OrderedDict)
        lru_chunk_index = next(iter(self.access_order))

        del self.chunks[lru_chunk_index]
        del self.access_order[lru_chunk_index]



class Cache(Generic[APIResponse]):
    def __init__(
        self,
        model_name: str,
        response_type: Type[APIResponse],
        cache_config: CacheConfig,
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
        self.backend = Backend(model_name, cache_config.base_path)
        self.manager = ChunkManager(self.backend, cache_config.max_chunks_in_memory, cache_config.entries_per_chunk)
        self.response_type = response_type
        self.cache_config = cache_config
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        if not self._initialized:
            async with self._init_lock:  # Prevent concurrent initialization
                if not self._initialized:  # Double-check after acquiring lock
                    await self.backend.initialize()
                    self._initialized = True

    async def put_entry(
        self,
        response: APIResponse,
        messages: ChatHistory,
        config: InferenceConfig,
    ) -> None:
        """Store an API call and its response in the cache."""
        await self._ensure_initialized()

        str_key, cache_key = file_cache_key(messages=messages, config=config)

        await self.manager.put_entry(
            cache_key=cache_key,
            str_key=str_key,
            response=response.model_dump_json(),
        )

    async def get_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
    ) -> Optional[APIResponse]:
        """Retrieve a cached response for the given inputs, or None if not found."""
        await self._ensure_initialized()

        _, cache_key = file_cache_key(messages=messages, config=config)
        response_str = await self.manager.get_entry(cache_key)

        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                return response
            except ValidationError:
                print(f"Warning: Failed to validate cache entry for key {cache_key}")
                return None
        return None

    async def close(self) -> None:
        """Close the backend connection."""
        await self.backend.close()

