"""LLM types module."""

import time
import json
import hashlib
from slist import Slist
from pathlib import Path
from typing import Optional, Type, Sequence, Generic, TypeVar, Mapping, Any, Literal
from pydantic import BaseModel, ValidationError
from caller.cache import SQLiteCacheBackend, ChunkedCacheManager

# Generic to say what we are caching
APIResponse = TypeVar("APIResponse", bound=BaseModel)


class ToolArgs(BaseModel):
    tools: Sequence[Mapping[Any, Any]]
    tool_choice: str


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": self.content,
            }
        else:
            assert self.image_type, "Please provide an image type"
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self.image_type};base64,{self.image_content}"
                        },
                    },
                ],
            }

    def to_anthropic_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                ],
            }
        else:
            """
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image1_media_type,
                    "data": image1_data,
                },
            },
            """
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.image_type or "image/jpeg",
                            "data": self.image_content,
                        },
                    },
                ],
            }


class ChatHistory(BaseModel):
    messages: Sequence[ChatMessage] = []

    def as_text(self) -> str:
        return "\n".join([msg.as_text() for msg in self.messages])

    @staticmethod
    def from_system(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="system", content=content)])

    @staticmethod
    def from_user(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="user", content=content)])

    def remove_system(self) -> "ChatHistory":
        """Remove all system prompts and creates a new copy."""
        new_messages = []
        for msg in self.messages:
            if msg.role != "system":
                # Create a copy of the ChatMessage
                new_messages.append(msg.model_copy())
        assert not any(msg.role == "system" for msg in new_messages)

        return ChatHistory(messages=new_messages)

    def add_user(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="user", content=content)]
        return ChatHistory(messages=new_messages)

    def add_assistant(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [
            ChatMessage(role="assistant", content=content)
        ]
        return ChatHistory(messages=new_messages)

    def add_messages(self, messages: Sequence[ChatMessage]) -> "ChatHistory":
        new_messages = list(self.messages) + list(messages)
        return ChatHistory(messages=new_messages)

    def to_openai_messages(self) -> list[dict]:
        return [msg.to_openai_content() for msg in self.messages]

    def get_first(self, role: Literal["system", "user", "assistant"]) -> str | None:
        """
        Get the first message with the given role, if exists.
        Returns None otherwise.
        """
        for msg in self.messages:
            if msg.role == role:
                return msg.content
        return None


class InferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    response_format: dict | None = None
    continue_final_message: bool | None = None
    reasoning: dict | None = None
    extra_body: dict | None = None


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(
                f"This response has multiple responses {self.raw_responses}"
            )
        else:
            return self.raw_responses[0]


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
    other_hash: str,
    tools: ToolArgs | None,
) -> str:
    config_dump = config.model_dump_json(
        exclude_none=True
    )  # for backwards compatibility
    tools_json = (
        tools.model_dump_json() if tools is not None else ""
    )  # for backwards compatibility
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


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class APIRequestCache(Generic[APIResponse]):
    def __init__(self, cache_path: Path | str, response_type: Type[APIResponse]):
        # Convert model-specific .jsonl path to shared .db path
        # e.g., ".api_cache/gpt-4.jsonl" -> ".api_cache/cache.db"
        cache_dir = Path(cache_path).parent
        db_path = cache_dir / "cache.db"

        # Extract model name from path (e.g., "gpt-4.jsonl" -> "gpt-4")
        self.model_name = Path(cache_path).stem
        self.response_type = response_type

        # Initialize SQLite backend
        self.backend = SQLiteCacheBackend(str(db_path))
        self.manager = ChunkedCacheManager(self.backend)

        # Initialize DB (async, called in get_model_call if needed)
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.backend.initialize()
            self._initialized = True

    async def flush(self) -> None:
        # SQLite commits immediately, so flush is a no-op
        pass

    async def add_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        response: APIResponse,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        await self._ensure_initialized()

        # Calculate cache key
        key = file_cache_key(messages, config, other_hash, tools=tools)

        # Serialize all data
        timestamp = int(time.time())
        response_json = response.model_dump_json()
        messages_json = messages.model_dump_json()
        config_json = config.model_dump_json()
        tools_json = tools.model_dump_json() if tools else None

        # Store in cache
        await self.manager.put_entry(
            cache_key=key,
            model=self.model_name,
            timestamp=timestamp,
            response=response_json,
            messages_json=messages_json,
            config_json=config_json,
            tools_json=tools_json
        )

    async def get_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[APIResponse]:
        await self._ensure_initialized()

        # Calculate cache key (reuse existing file_cache_key function)
        key = file_cache_key(messages, config, other_hash, tools=tools)

        # Get current timestamp
        timestamp = int(time.time())

        # Try to get from cache
        response_str = await self.manager.get_entry(key, self.model_name, timestamp)

        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                return response
            except ValidationError as e:
                print(f"Warning: Failed to validate cache entry for key {key}")
                return None
        return None


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


class HashableBaseModel(BaseModel):
    def model_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)

    class Config:
        # this is needed for the hashable base model
        frozen = True
