"""
Types for LLM API calls.

Reference:
https://openrouter.ai/docs/api-reference/overview

TODO:
* Add image support
* Add streaming support
"""

import logging
from typing import Sequence, Any, Literal, Optional, Union
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class FunctionDescription(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Any


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDescription

class FunctionName(BaseModel):
    name: str

class ToolChoiceFunction(BaseModel):
    type: Literal["function"]
    function: FunctionName

ToolChoice = Union[Literal["none"], Literal["auto"], ToolChoiceFunction]


class ResponseFormat(BaseModel):
    type: Literal["json_schema"]
    json_schema: dict


class InferenceConfig(BaseModel):
    """
    All the optional parameters.
    """

    response_format: Optional[ResponseFormat] = None
    stop: Optional[list] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    # Sampling parameters
    seed: Optional[int] = None
    top_p: Optional[float] = None  # (0, 1]
    frequency_penalty: Optional[float] = None  # [-2, 2]
    presence_penalty: Optional[float] = None  # [-2, 2]
    repetition_penalty: Optional[float] = None  # (0, 2]
    min_p: Optional[float] = None  # [0, 1]
    top_a: Optional[float] = None  # [0, 1]
    logit_bias: Optional[dict[int, float]] = None
    top_logprobs: Optional[int] = None

    # Extra body
    reasoning: Optional[str | int] = None
    extra_body: Optional[dict] = None



class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
        }


class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: Optional[str]=None

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }


Message = Union[ChatMessage, ToolMessage]


class ChatHistory(BaseModel):
    messages: Sequence[Message] = []

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


class Request(BaseModel):
    """
    Main request format for OpenRouter.
    """
    model: str
    messages: list[Message] | ChatHistory
    config: InferenceConfig

    def to_request(self) -> dict:
        request_body = {"model": self.model}
        if isinstance(self.messages, ChatHistory):
            request_body["messages"] = self.messages.to_openai_messages()
        else:
            request_body["messages"] = [msg.to_openai_content() for msg in self.messages]

        config_dict = self.config.model_dump()

        if config_dict["reasoning"] is None:
            pass
        elif isinstance(config_dict["reasoning"], int):
            if config_dict["extra_body"] is None:
                config_dict["extra_body"] = {}
            config_dict["extra_body"]["reasoning"] = {"max_tokens": config_dict["reasoning"]}
        elif isinstance(config_dict["reasoning"], str):
            if config_dict["extra_body"] is None:
                config_dict["extra_body"] = {}
            config_dict["extra_body"]["reasoning"] = {"effort": config_dict["reasoning"]}

        config_dict.pop("reasoning")

        request_body.update(config_dict)
        return request_body


class NonStreamingChoice(BaseModel):
    finish_reason: Optional[str]
    native_finish_reason: Optional[str]
    message: dict
    error: Optional[dict]


class Response(BaseModel):
    """Unified response format for all providers."""
    id: str
    choices: list[NonStreamingChoice]
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    usage: dict

    @property
    def first_choice(self) -> NonStreamingChoice|None:
        """Returns the first choice of the response."""
        if len(self.choices) == 0:
            logger.warning(f"No choices found in Response: {self}")
            return None
        return self.choices[0]

    @property
    def first_response(self) -> str|None:
        """Returns the first response's content if it exists, otherwise None."""
        first_choice = self.first_choice
        if first_choice is None:
            return None
        content = first_choice.message["content"]
        if content is None:
            logger.warning(f"No content found in first choice of Response: {self}")
        return content

    @property
    def has_response(self) -> bool:
        return self.first_response is not None

    @property
    def reasoning_content(self) -> str|None:
        """Returns the first response's reasoning content if it exists, otherwise None."""
        first_choice = self.first_choice
        if first_choice is None:
            return None
        if "reasoning_details" not in first_choice.message:
            logger.warning(f"No reasoning details found in first choice of Response: {self}")
            return None

        reasoning_details = first_choice.message["reasoning_details"]
        if reasoning_details["type"] == "reasoning.summary":
            return reasoning_details["summary"]
        elif reasoning_details["type"] == "reasoning.text":
            return reasoning_details["text"]
        elif reasoning_details["type"] == "reasoning.encrypted":
            logger.info(f"Reasoning details are encrypted in Response: {self}")
            return reasoning_details["data"]

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_content is not None

    @property
    def finish_reason(self) -> str|None:
        """
        Returns the finish reason of the response.
        
        Possible values: "stop", "length", "content_filter", "error", "tool_calls".
        """
        first_choice = self.first_choice
        if first_choice is None:
            return None
        finish_reason = first_choice.finish_reason or first_choice.native_finish_reason
        return finish_reason
