"""LLM types module."""

from typing import Sequence, Mapping, Any, Literal
from pydantic import BaseModel



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
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    response_format: dict | None = None
    reasoning: str | int | None = None
    extra_body: dict | None = None


class OpenaiResponse(BaseModel):
    """Unified response format for all providers."""
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        try:
            content = self.choices[0]["message"]["content"]
            if content is None:
                raise ValueError(f"No content found in OpenaiResponse: {self}")
            if isinstance(content, dict):
                content = content.get("text", "")
            return content
        except (TypeError, KeyError, IndexError) as e:
            raise ValueError(f"No content found in OpenaiResponse: {self}") from e

    @property
    def reasoning_content(self) -> str | None:
        """Returns the reasoning content if it exists, otherwise None."""
        try:
            possible_keys = ["reasoning_content", "reasoning"]
            for key in possible_keys:
                if key in self.choices[0]["message"]:
                    return self.choices[0]["message"][key]

                content = self.choices[0]["message"].get("content")
                if isinstance(content, dict) and key in content:
                    return content[key]
        except (KeyError, IndexError):
            pass
        return None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_content is not None

    def has_response(self) -> bool:
        if len(self.choices) == 0:
            return False
        first_choice = self.choices[0]
        if first_choice.get("message") is None:
            return False
        if first_choice["message"].get("content") is None:
            return False
        return True

    @property
    def hit_content_filter(self) -> bool:
        """Check if response was blocked by content filter."""
        try:
            first_choice = self.choices[0]
            finish_reason = first_choice.get("finishReason") or first_choice.get("finish_reason")
            return finish_reason == "content_filter"
        except (KeyError, IndexError):
            return False

    @property
    def abnormal_finish(self) -> bool:
        """Check if response finished abnormally."""
        try:
            first_choice = self.choices[0]
            finish_reason = first_choice.get("finishReason") or first_choice.get("finish_reason")
            return finish_reason not in ["stop", "length", None]
        except (KeyError, IndexError):
            return False
