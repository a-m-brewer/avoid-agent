"""Module for Anthropic provider."""

from collections.abc import Iterator
from functools import lru_cache
from itertools import groupby
import re
import subprocess

from anthropic import Anthropic
import anthropic


def list_models(api_key: str) -> list[str]:
    """List available Anthropic model IDs for the authenticated account."""
    client = Anthropic(api_key=api_key)
    response = client.models.list()
    ids = [getattr(model, "id", "") for model in response.data]
    return sorted(model_id for model_id in ids if model_id)

from avoid_agent.agent.tools import ToolDefinition
from avoid_agent.providers import (
    AssistantMessage,
    AssistantTextBlock,
    Message,
    Provider,
    ProviderEvent,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolChoice,
    ToolResultMessage,
    UserMessage,
    Usage,
    normalize_messages,
)


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Return True for models that use adaptive thinking (Sonnet/Opus 4.6+)."""
    return any(s in model_id for s in ("sonnet-4-6", "sonnet-4.6", "opus-4-6", "opus-4.6"))


def _map_stop_reason(stop_reason: str | None) -> str:
    if stop_reason == "tool_use":
        return "tool_use"
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason in ("end_turn", "stop_sequence", None):
        return "stop"
    return "error"


class AnthropicStream(ProviderStream):
    """Context manager for streaming responses from the Anthropic provider."""

    def __init__(self, ctx: anthropic.MessageStreamManager[None]):
        self._ctx = ctx
        self._stream = None

    def __enter__(self):
        self._stream = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)

    def event_stream(self) -> Iterator[ProviderEvent]:
        for delta in self._stream.text_stream:
            yield ProviderEvent(type="text_delta", text=delta)

    def get_final_message(self) -> ProviderResponse:
        final_message = self._stream.get_final_message()

        content = []
        tool_calls = []
        for block in final_message.content:
            if block.type == "text":
                content.append(AssistantTextBlock(text=block.text))
            elif block.type == "tool_use":
                tool_call = ProviderToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
                tool_calls.append(tool_call)
                content.append(tool_call)

        text = "".join(
            block.text for block in content if isinstance(block, AssistantTextBlock)
        ) or None
        stop_reason = _map_stop_reason(final_message.stop_reason)
        if tool_calls and stop_reason == "stop":
            stop_reason = "tool_use"

        return ProviderResponse(
            message=AssistantMessage(
                text=text,
                tool_calls=tool_calls,
                content=content,
                stop_reason=stop_reason,
                usage=Usage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=getattr(final_message.usage, "output_tokens", 0),
                    total_tokens=(
                        final_message.usage.input_tokens
                        + getattr(final_message.usage, "output_tokens", 0)
                    ),
                ),
            ),
            stop_reason=stop_reason,
            input_tokens=final_message.usage.input_tokens,
        )


# Fallback version used when the `claude` binary is not installed.  Keep this
# reasonably current so OAuth requests stay within the expected range.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.83"

# Claude Code identity claim required as the FIRST system-prompt block for OAuth.
_CC_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


@lru_cache(maxsize=1)
def _get_claude_code_version() -> str:
    """Return the installed Claude Code version, or the fallback constant.

    Runs ``claude --version`` once and caches the result for the process
    lifetime.  The output format is ``2.1.83 (Claude Code)``; we extract
    the leading semver component.  Falls back to *_CLAUDE_CODE_VERSION_FALLBACK*
    if the binary is absent, the call times out, or the output is unparseable.
    """
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            match = re.search(r"\d+\.\d+\.\d+", result.stdout)
            if match:
                return match.group(0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return _CLAUDE_CODE_VERSION_FALLBACK


class AnthropicProvider(Provider):
    """Provider for Anthropic models."""

    def __init__(
        self,
        system: str,
        model: str,
        max_tokens: int,
        *,
        api_key: str | None = None,
        auth_token: str | None = None,
        thinking_enabled: bool | None = None,
        effort: str | None = None,
    ):
        # Store whether we're in OAuth mode so stream() can format the system
        # prompt as a structured array (required by the Claude Code OAuth path).
        self._oauth = bool(auth_token)
        super().__init__(
            system,
            model,
            max_tokens,
            thinking_enabled=thinking_enabled,
            effort=effort,  # Anthropic currently uses 'thinking' only; keep for API symmetry
        )
        if auth_token:
            self._client = Anthropic(
                auth_token=auth_token,
                default_headers={
                    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
                    "x-app": "cli",
                    # Match the installed Claude Code version for accurate stealth;
                    # falls back to the bundled constant if `claude` is not found.
                    "user-agent": f"claude-cli/{_get_claude_code_version()}",
                },
            )
        else:
            self._client = Anthropic(api_key=api_key)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
    ) -> ProviderStream:
        provider_messages = self.__get_provider_messages(normalize_messages(messages))
        provider_tools = self.__get_provider_tools(tools)

        # For OAuth tokens, the system prompt MUST be a structured array with the
        # Claude Code identity as the first block.  This matches what the real
        # Claude Code client sends and is required for higher-tier model access.
        if self._oauth:
            system_param: str | list[dict] = [
                {"type": "text", "text": _CC_IDENTITY, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}},
            ]
        else:
            system_param = self.system

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_param,
            "tools": provider_tools,
            "messages": provider_messages,
        }
        if tool_choice != "auto" and provider_tools:
            kwargs["tool_choice"] = {"type": "any" if tool_choice == "required" else tool_choice}

        # Thinking configuration
        if self.thinking_enabled:
            if _supports_adaptive_thinking(self.model):
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                budget = min(1024, max(128, self.max_tokens // 4))
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif _supports_adaptive_thinking(self.model):
            # Adaptive-thinking models require explicit disable; the API rejects
            # requests that omit the thinking field for these model classes.
            kwargs["thinking"] = {"type": "disabled"}

        anthropic_stream = self._client.messages.stream(**kwargs)
        return AnthropicStream(ctx=anthropic_stream)

    def __get_provider_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for item in self._batch_messages(messages):
            if isinstance(item, list):
                result.append(self.__convert_tool_results(item))
            else:
                result.append(self.__convert_to_provider_message(item))
        return self._merge_consecutive_user_messages(result)

    @staticmethod
    def _merge_consecutive_user_messages(messages: list[dict]) -> list[dict]:
        """Merge consecutive user-role messages into one.

        Anthropic requires strictly alternating user/assistant roles.
        The runtime appends a controller state UserMessage after the real
        user message (and after tool-result batches), so consecutive user
        messages appear on every request.  Merging them into a single
        multi-block user turn satisfies the API constraint without
        changing the internal message model.
        """
        if not messages:
            return messages
        result = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            if prev["role"] == "user" and msg["role"] == "user":
                prev_content = prev["content"]
                curr_content = msg["content"]
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(curr_content, str):
                    curr_content = [{"type": "text", "text": curr_content}]
                result[-1] = {"role": "user", "content": prev_content + curr_content}
            else:
                result.append(msg)
        return result

    @staticmethod
    def _batch_messages(messages: list[Message]):
        for is_tool_result, group in groupby(
            messages, key=lambda m: isinstance(m, ToolResultMessage)
        ):
            if is_tool_result:
                yield list(group)  # list of ToolResultMessage
            else:
                for message in group:
                    yield message  # individual non-tool messages

    def __convert_to_provider_message(self, message: Message) -> dict:
        if isinstance(message, UserMessage):
            # If there are attached images, build a multipart content array.
            if message.images:
                content: list[dict] = []
                for img in message.images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img.data,
                        },
                    })
                if message.text:
                    content.append({"type": "text", "text": message.text})
                return {"role": "user", "content": content}
            return {"role": "user", "content": message.text}

        if isinstance(message, AssistantMessage):
            content = []
            for block in message.content:
                if isinstance(block, AssistantTextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ProviderToolCall):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )

            if not content and message.text is not None:
                return {"role": "assistant", "content": message.text}

            return {"role": "assistant", "content": content}

        raise ValueError(f"Unknown message type: {type(message)}")

    def __convert_tool_results(self, messages: list[ToolResultMessage]) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                for m in messages
            ],
        }

    def __get_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [self.__convert_to_provider_tool(tool) for tool in tools]

    def __convert_to_provider_tool(self, tool: ToolDefinition) -> dict:
        properties = {}
        for param in tool.parameters:
            properties[param.name] = {
                "type": param.type.value,
                "description": param.description,
            }

        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": [
                    required_param.name
                    for required_param in tool.parameters
                    if required_param.required
                ],
            },
        }
