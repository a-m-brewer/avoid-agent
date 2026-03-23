"""Module for Anthropic provider."""

from collections.abc import Iterator
from itertools import groupby

from anthropic import Anthropic
import anthropic

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
    ToolResultMessage,
    UserMessage,
    Usage,
    normalize_messages,
)


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


class AnthropicProvider(Provider):
    """Provider for Anthropic models."""

    def __init__(self, system: str, model: str, max_tokens: int):
        super().__init__(system, model, max_tokens)
        self._client = Anthropic()

    def stream(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> ProviderStream:
        provider_messages = self.__get_provider_messages(normalize_messages(messages))
        provider_tools = self.__get_provider_tools(tools)

        anthropic_stream = self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=provider_tools,
            messages=provider_messages,
        )

        return AnthropicStream(ctx=anthropic_stream)

    def compact(self, messages: list[Message], keep_last: int = 6) -> list[Message]:
        """Compacts messages by keeping the last N and summarizing the rest."""

        to_summarize = self.__get_provider_messages(
            normalize_messages(messages[:-keep_last])
        )
        recent = messages[-keep_last:]

        summary_response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                *to_summarize,
                {
                    "role": "user",
                    "content": "Summarize this conversation so far. Include: what the user asked for, what was explored, what changes were made, and any important findings. Be concise but complete.",
                },
            ],
        )

        summary_text = summary_response.content[0].text

        return [
            UserMessage(text=f"[Conversation summary]\n{summary_text}"),
            AssistantMessage(text="Understood.", tool_calls=[]),
            *recent,
        ]

    def __get_provider_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for item in self._batch_messages(messages):
            if isinstance(item, list):
                result.append(self.__convert_tool_results(item))
            else:
                result.append(self.__convert_to_provider_message(item))
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
