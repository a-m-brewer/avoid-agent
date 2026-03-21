"""Module for agent providers."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from avoid_agent.agent.tools import ToolDefinition


@dataclass
class ProviderToolCall:
    """Represents a call to a tool as specified by the provider."""

    id: str
    name: str
    arguments: dict


@dataclass
class Message(metaclass=ABCMeta):
    """Base class for messages exchanged with the provider."""


@dataclass
class UserMessage(Message):
    """Message from the user."""

    text: str


@dataclass
class AssistantMessage(Message):
    """Message from the assistant."""

    tool_calls: list[ProviderToolCall]
    text: str | None


@dataclass
class ToolResultMessage(Message):
    """Message containing the result of a tool call."""

    tool_call_id: str
    content: str


@dataclass
class ProviderResponse:
    """Structured response from the provider after processing a message."""

    message: AssistantMessage
    stop: bool
    input_tokens: int


class ProviderStream(metaclass=ABCMeta):
    """Context manager for streaming responses from the provider."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def text_stream(self) -> Iterator[str]:
        """Yields text chunks as they are received from the provider."""

    @abstractmethod
    def get_final_message(self) -> ProviderResponse:
        """Returns the final structured response after the stream is complete."""


class Provider(metaclass=ABCMeta):
    """Base class for providers."""

    def __init__(self, system: str, model: str, max_tokens: int):
        self.system = system
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def stream(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> ProviderStream:
        """Sends messages to the provider and returns a stream of responses."""

    @abstractmethod
    def compact(self, messages: list[Message], keep_last: int = 6) -> list[Message]:
        """Compacts a list of messages to reduce token count, keeping the last N messages."""
