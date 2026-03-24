"""Module for agent providers."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Iterator, Literal, TypeAlias
import urllib.request

from avoid_agent.providers.openai_codex_oauth import get_valid_credentials, load_credentials

from avoid_agent.agent.tools import ToolDefinition


AVAILABLE_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-1",
        "claude-haiku-4-5",
    ],
    "openai": [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
    "openai-codex": [
        "codex-mini-latest",
    ],
    "openrouter": [
        "openai/gpt-5",
        "anthropic/claude-sonnet-4-6",
    ],
    "ollama": [
        "devstral-small:latest",
    ],
}

_MODEL_CACHE: dict[str, object] = {"expires_at": 0.0, "models": []}
_MODEL_CACHE_TTL_SECONDS = 300
_CONFIG_PATH = Path.home() / ".avoid-agent" / "config.json"


def load_user_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_config(config: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def get_saved_model() -> str | None:
    model = load_user_config().get("default_model")
    return model if isinstance(model, str) and model.strip() else None


def save_selected_model(model: str) -> None:
    config = load_user_config()
    config["default_model"] = model
    save_user_config(config)


def _fallback_models() -> list[str]:
    out: list[str] = []
    for provider_name, models in AVAILABLE_MODELS.items():
        out.extend(f"{provider_name}/{model}" for model in models)
    return sorted(out)


def _list_dynamic_models() -> list[str]:
    providers_to_models: dict[str, list[str]] = {}

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from avoid_agent.providers.anthropic import list_models as list_anthropic_models

        try:
            models = list_anthropic_models(api_key=anthropic_key)
            if models:
                providers_to_models["anthropic"] = models
        except Exception:  # pylint: disable=broad-except
            pass

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from avoid_agent.providers.openai import list_models as list_openai_models

        try:
            models = list_openai_models(api_key=openai_key)
            if models:
                providers_to_models["openai"] = models
        except Exception:  # pylint: disable=broad-except
            pass

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from avoid_agent.providers.openai import list_models as list_openai_models

        try:
            models = list_openai_models(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )
            if models:
                providers_to_models["openrouter"] = models
        except Exception:  # pylint: disable=broad-except
            pass

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models_payload = payload.get("models") if isinstance(payload, dict) else []
        models = [
            name
            for item in models_payload
            if isinstance(item, dict)
            for name in [item.get("name")]
            if isinstance(name, str) and name.strip()
        ]
        if models:
            providers_to_models["ollama"] = models
    except Exception:  # pylint: disable=broad-except
        pass

    creds = load_credentials()
    if creds:
        try:
            from avoid_agent.providers.openai_codex import list_models as list_codex_models

            models = list_codex_models(credentials=creds)
            if models:
                providers_to_models["openai-codex"] = models
        except Exception:  # pylint: disable=broad-except
            pass

    out: list[str] = []
    for provider_name, models in providers_to_models.items():
        out.extend(f"{provider_name}/{model}" for model in models)

    if not out:
        return []

    return sorted(set(out))


def list_available_models() -> list[str]:
    """Return available models in provider/model format for model picker UX."""
    now = time.time()
    expires_at = float(_MODEL_CACHE.get("expires_at", 0.0) or 0.0)
    cached_models = _MODEL_CACHE.get("models")
    if isinstance(cached_models, list) and cached_models and now < expires_at:
        return list(cached_models)

    dynamic = _list_dynamic_models()
    models = dynamic or _fallback_models()

    _MODEL_CACHE["models"] = models
    _MODEL_CACHE["expires_at"] = now + _MODEL_CACHE_TTL_SECONDS
    return models

# tool_choice controls whether the model must call a tool.
# "auto" = model decides, "required" = must call a tool, "none" = no tools
ToolChoice: TypeAlias = Literal["auto", "required", "none"]

StopReason: TypeAlias = Literal["stop", "tool_use", "length", "error", "aborted"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_reasoning_text(item: dict) -> str:
    summary = item.get("summary")
    if isinstance(summary, list):
        return "\n\n".join(
            str(part.get("text", "")).strip()
            for part in summary
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        )
    if isinstance(summary, str):
        return summary
    return ""


@dataclass
class Usage:
    """Token usage metadata captured for an assistant message."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AssistantTextBlock:
    """Assistant text content."""

    text: str
    item_id: str | None = None
    type: Literal["text"] = "text"


@dataclass
class AssistantThinkingBlock:
    """Assistant reasoning content kept for provider replay."""

    text: str
    raw_item: dict | None = None
    type: Literal["thinking"] = "thinking"


@dataclass
class ProviderToolCall:
    """Represents a call to a tool as specified by the provider."""

    id: str
    name: str
    arguments: dict
    item_id: str | None = None  # Responses API item ID (fc_xxx), separate from call_id
    type: Literal["tool_call"] = "tool_call"


AssistantContentBlock: TypeAlias = (
    AssistantTextBlock | AssistantThinkingBlock | ProviderToolCall
)


@dataclass
class Message(metaclass=ABCMeta):
    """Base class for messages exchanged with the provider."""


@dataclass
class UserMessage(Message):
    """Message from the user."""

    text: str
    timestamp: int = field(default_factory=_now_ms)


@dataclass
class AssistantMessage(Message):
    """Message from the assistant."""

    text: str | None = None
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    text_id: str | None = None  # Responses API message item ID for history replay
    reasoning_items: list[dict] = field(default_factory=list)  # reasoning items for replay
    content: list[AssistantContentBlock] = field(default_factory=list)
    stop_reason: StopReason = "stop"
    timestamp: int = field(default_factory=_now_ms)
    usage: Usage = field(default_factory=Usage)
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.content:
            self.content = []
            for reasoning_item in self.reasoning_items:
                self.content.append(
                    AssistantThinkingBlock(
                        text=_extract_reasoning_text(reasoning_item),
                        raw_item=reasoning_item,
                    )
                )
            if self.text:
                self.content.append(
                    AssistantTextBlock(text=self.text, item_id=self.text_id)
                )
            self.content.extend(self.tool_calls)
            return

        if self.text is None:
            text_blocks = [
                block for block in self.content if isinstance(block, AssistantTextBlock)
            ]
            if text_blocks:
                self.text = "".join(block.text for block in text_blocks)
                if len(text_blocks) == 1 and self.text_id is None:
                    self.text_id = text_blocks[0].item_id

        if not self.tool_calls:
            self.tool_calls = [
                block for block in self.content if isinstance(block, ProviderToolCall)
            ]

        if not self.reasoning_items:
            self.reasoning_items = [
                block.raw_item
                for block in self.content
                if isinstance(block, AssistantThinkingBlock) and block.raw_item
            ]


@dataclass
class ToolResultMessage(Message):
    """Message containing the result of a tool call."""

    tool_call_id: str
    content: str
    tool_name: str | None = None
    is_error: bool = False
    timestamp: int = field(default_factory=_now_ms)
    details: dict = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Structured response from the provider after processing a message."""

    message: AssistantMessage
    stop_reason: StopReason
    input_tokens: int = 0

    @property
    def stop(self) -> bool:
        """Backward-compatible boolean stop flag."""
        return self.stop_reason == "stop"


@dataclass
class ProviderEvent:
    """Structured event emitted while streaming a provider response."""

    type: Literal[
        "text_delta",
        "tool_call_detected",
        "reasoning_item",
        "status",
        "error",
        "raw",
    ]
    text: str | None = None
    tool_call: ProviderToolCall | None = None
    reasoning_item: dict | None = None
    status: str | None = None
    error: str | None = None
    raw_event: dict | None = None


class ProviderStream(metaclass=ABCMeta):
    """Context manager for streaming responses from the provider."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def event_stream(self) -> Iterator[ProviderEvent]:
        """Yields structured events as they are received from the provider."""

    @abstractmethod
    def get_final_message(self) -> ProviderResponse:
        """Returns the final structured response after the stream is complete."""


class Provider(metaclass=ABCMeta):
    """Base class for providers."""

    def __init__(
        self,
        system: str,
        model: str,
        max_tokens: int,
        *,
        thinking_enabled: bool | None = None,
        effort: Literal["low", "medium", "high"] | None = None,
    ):
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        # Optional reasoning controls. Defaults: thinking off, effort high.
        self.thinking_enabled: bool = bool(thinking_enabled) if thinking_enabled is not None else False
        self.effort: Literal["low", "medium", "high"] = (
            effort if effort in ("low", "medium", "high") else "high"
        )
        self.supports_thinking: bool = True

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
    ) -> ProviderStream:
        """Sends messages to the provider and returns a stream of responses."""


def normalize_messages(messages: list[Message]) -> list[Message]:

    """Normalize message history before sending it back to a provider.

    - Drops aborted/error assistant turns that should not be replayed.
    - Inserts synthetic tool results for orphaned tool calls if the conversation
      continued without a matching tool result.
    """
    normalized: list[Message] = []
    pending_tool_calls: list[ProviderToolCall] = []
    seen_tool_results: set[str] = set()

    def flush_orphaned_tool_calls() -> None:
        nonlocal pending_tool_calls, seen_tool_results
        if not pending_tool_calls:
            return

        for tool_call in pending_tool_calls:
            if tool_call.id in seen_tool_results:
                continue
            normalized.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content="No result provided",
                    is_error=True,
                )
            )

        pending_tool_calls = []
        seen_tool_results = set()

    for message in messages:
        if isinstance(message, AssistantMessage):
            flush_orphaned_tool_calls()

            if message.stop_reason in ("error", "aborted"):
                continue

            normalized.append(message)
            if message.tool_calls:
                pending_tool_calls = list(message.tool_calls)
                seen_tool_results = set()
            continue

        if isinstance(message, ToolResultMessage):
            seen_tool_results.add(message.tool_call_id)
            normalized.append(message)
            continue

        if isinstance(message, UserMessage):
            flush_orphaned_tool_calls()
            normalized.append(message)
            continue

        normalized.append(message)

    return normalized


def get_provider(
    model: str | None,
    system: str,
    max_tokens: int | None,
    *,
    thinking_enabled: bool | None = None,
    effort: Literal["low", "medium", "high"] | None = None,
) -> Provider:
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
            thinking_enabled=thinking_enabled,
            effort=effort,
        )

    if provider_name == "ollama":
        # pylint: disable=import-outside-toplevel
        from avoid_agent.providers.openai import OpenAIProvider

        base_url = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')}/v1"
        provider = OpenAIProvider(
            system=system,
            model=model_name,
            max_tokens=internal_max_tokens,
            base_url=base_url,
            api_key="ollama",
            thinking_enabled=thinking_enabled,
            effort=effort,
        )
        provider.supports_thinking = False
        return provider

    if provider_name == "anthropic":
        # pylint: disable=import-outside-toplevel
        from avoid_agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            system=system,
            model=model_name,
            max_tokens=internal_max_tokens,
            thinking_enabled=thinking_enabled,
            effort=effort,
        )

    if provider_name == "openai-codex":
        # pylint: disable=import-outside-toplevel
        from avoid_agent.providers.openai_codex import OpenAICodexProvider
        from avoid_agent.providers.openai_codex_oauth import get_valid_credentials
        creds = get_valid_credentials()
        return OpenAICodexProvider(
            system=system,
            model=model_name,
            max_tokens=internal_max_tokens,
            credentials=creds,
            thinking_enabled=thinking_enabled,
            effort=effort,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")
