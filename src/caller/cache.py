"""SQLite-based response caching."""

import json
import time
import hashlib
import asyncio
from loguru import logger
from pathlib import Path
from cachetools import LRUCache
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
    to_dump = {
        "messages": messages.model_dump(exclude_none=True),
        "config": config.model_dump(exclude_none=True),
    }
    str_key = json.dumps(to_dump, sort_keys=True)
    cache_key = deterministic_hash(str_key)

    return str_key, cache_key


class CacheConfig(BaseModel):
    """
    Configuration for cache behavior.
    """

    base_path: str | None = ".cache/caller"  # None = disable caching
    no_cache_models: set[str] = set()

    cache_chunk_size: int = 128
    max_entries_in_memory: int = 8192
    max_entries_in_disk: int | None = 131072


class Backend:
    """SQLite cache backend."""

    def __init__(self, safe_model_name: str, cache_config: CacheConfig):
        if cache_config.base_path is None:
            raise ValueError("base_path cannot be None when trying to enable caching")

        self.db_path = Path(cache_config.base_path) / f"{safe_model_name}.db"
        self.cache_config = cache_config
        self.in_memory_cache = LRUCache(maxsize=cache_config.max_entries_in_memory)

        self._initialized = False

        # Eviction
        self._evict_counter = 0
        self._eviction_check_interval = 1000  # check every 1000 inserts

        # Writes for last_used
        self._pending_updates: dict[str, int] = {}  # cache_key -> last_used timestamp
        self._update_batch_size = 1000
        self._update_interval = 60
        self._last_flush_time = time.time()

        # Locks
        self._write_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()


    async def initialize(self):
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode = WAL")  # enable WAL mode
                await db.execute(
                    "PRAGMA synchronous = NORMAL"
                )  # sync less frequently for faster writes

                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        str_key TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        last_used INTEGER NOT NULL
                    )
                """
                )

                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_last_used
                    ON cache_entries(last_used)
                """
                )

                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_created_at
                    ON cache_entries(created_at)
                """
                )

                await db.commit()

            self._initialized = True

    async def _flush_pending(self):
        """Flush all pending last_used updates to database."""
        should_flush = (
            len(self._pending_updates) >= self._update_batch_size
            or (time.time() - self._last_flush_time) >= self._update_interval
        )
        
        if should_flush:
            async with self._write_lock:
                # Grab current pending updates and clear the dict
                updates_to_flush = dict(self._pending_updates)
                self._pending_updates.clear()
                self._last_flush_time = time.time()
                
                if not updates_to_flush:
                    return

                async with aiosqlite.connect(self.db_path) as db:
                    # Batch update using executemany
                    await db.executemany(
                        "UPDATE cache_entries SET last_used = ? WHERE cache_key = ?",
                        [(timestamp, key) for key, timestamp in updates_to_flush.items()]
                    )
                    await db.commit()

    async def get_entry(self, cache_key: str) -> dict | None:
        await self.initialize()
        current_time = int(time.time())

        if cache_key in self.in_memory_cache:
            entry = self.in_memory_cache[cache_key]
            entry["last_used"] = current_time
            self._pending_updates[cache_key] = current_time
            await self._flush_pending()
            return entry

        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return None

                    target_entry = dict(row)
                    target_created_at = target_entry["created_at"]

                await db.execute(
                    """
                    UPDATE cache_entries SET last_used = ? WHERE cache_key = ?
                    """,
                    (current_time, cache_key),
                )
                await db.commit()

                target_entry["last_used"] = current_time
                self.in_memory_cache[cache_key] = target_entry
            
        await self._prefetch_chunk(target_created_at, cache_key)
        return target_entry

    async def _prefetch_chunk(self, created_at: int, cache_key: str) -> None:
        """Prefetch a chunk of entries with similar creation time."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    """
                    SELECT * FROM cache_entries
                    WHERE created_at >= ? AND cache_key != ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """,
                    (created_at, cache_key, self.cache_config.cache_chunk_size),
                ) as cursor:
                    rows = await cursor.fetchall()

                    for row in rows:
                        entry = dict(row)
                        entry_key = entry["cache_key"]

                        if entry_key not in self.in_memory_cache:
                            self.in_memory_cache[entry_key] = entry

        except Exception:
            logger.warning("Failed to prefetch chunk", exc_info=True)

    async def put_entry(self, entry: dict):
        await self.initialize()
        cache_key = entry["cache_key"]

        # Add to in-memory cache immediately
        self.in_memory_cache[cache_key] = entry
        self._pending_updates.pop(cache_key, None)

        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key, str_key, response, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        entry["cache_key"],
                        entry["str_key"],
                        entry["response"],
                        entry["created_at"],
                        entry["last_used"],
                    ),
                )

                # Only check once every self._eviction_check_interval inserts
                self._evict_counter += 1
                if (
                    self.cache_config.max_entries_in_disk is not None
                    and self._evict_counter % self._eviction_check_interval == 0
                ):
                    async with db.execute("SELECT COUNT(*) FROM cache_entries") as cursor:
                        count = (await cursor.fetchone())[0]

                    logger.debug(
                        f"Evicting old cache entries, current count: {count}/{self.cache_config.max_entries_in_disk}"
                    )

                    if count > self.cache_config.max_entries_in_disk:
                        to_delete = count - int(self.cache_config.max_entries_in_disk * 0.9)
                        await db.execute(
                            """
                            DELETE FROM cache_entries
                            WHERE cache_key IN (
                                SELECT cache_key FROM cache_entries
                                ORDER BY last_used ASC
                                LIMIT ?
                            )
                        """,
                            (to_delete,),
                        )

                await db.commit()


class Cache(Generic[APIResponse]):
    """
    Interface for cache operations.
    """
    def __init__(
        self,
        safe_model_name: str,
        response_type: Type[APIResponse],
        cache_config: CacheConfig,
    ):
        self.safe_model_name = safe_model_name
        self.response_type = response_type
        self.cache_config = cache_config
        self.backend = Backend(safe_model_name, cache_config)

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        if self._initialized:
            return

        async with self._init_lock:
            await self.backend.initialize()
            self._initialized = True

    async def put_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        response: APIResponse,
    ) -> None:
        await self.initialize()

        str_key, cache_key = file_cache_key(messages=messages, config=config)

        await self.backend.put_entry(
            entry={
                "cache_key": cache_key,
                "str_key": str_key,
                "response": response.model_dump_json(exclude_none=True),
                "created_at": int(time.time()),
                "last_used": int(time.time()),
            }
        )

    async def get_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
    ) -> Optional[APIResponse]:
        await self.initialize()

        _, cache_key = file_cache_key(messages=messages, config=config)
        entry = await self.backend.get_entry(cache_key)

        if entry:
            try:
                response = self.response_type.model_validate_json(entry["response"])
                return response
            except ValidationError:
                logger.warning(f"Failed to validate cache entry for key {cache_key}.\nResponse: {entry['response']}", exc_info=True)
                return None
        return None
