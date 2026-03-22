"""Module for agent providers."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import os
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


def get_provider(model: str | None, system: str, max_tokens: int | None) -> Provider:
    selected_model = model or os.getenv("DEFAULT_MODEL")
    if selected_model is None:
        raise ValueError(
            "No model specified and DEFAULT_MODEL not set in environment variables."
        )

    internal_max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "8192"))

    model_parts = selected_model.split("/")
    if len(model_parts) < 2:
        raise ValueError(
            f"Invalid model format: {selected_model}. Expected format 'provider/modelname'."
        )

    provider_name = model_parts[0].lower()
    model_name = "/".join(model_parts[1:])

    if provider_name == "openai" or provider_name == "openrouter":
        # pylint: disable=import-outside-toplevel
        from avoid_agent.providers.openai import OpenAIProvider

        base_url = None
        api_key = None
        if provider_name == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")

        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(
                f"API key for {provider_name} not found in environment variables."
            )

        return OpenAIProvider(
            system=system,
            model=model_name,
            max_tokens=internal_max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    if provider_name == "anthropic":
        # pylint: disable=import-outside-toplevel
        from avoid_agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            system=system, model=model_name, max_tokens=internal_max_tokens
        )

    raise ValueError(f"Unsupported provider: {provider_name}")
