"""Module for Anthropic provider."""

from collections.abc import Iterator
from itertools import groupby

from anthropic import Anthropic
import anthropic

from avoid_agent.agent.tools import ToolDefinition
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
)


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

    def text_stream(self) -> Iterator[str]:
        return self._stream.text_stream

    def get_final_message(self) -> ProviderResponse:
        final_message = self._stream.get_final_message()

        tool_calls = (
            []
            if final_message.stop_reason != "tool_use"
            else [
                ProviderToolCall(id=block.id, name=block.name, arguments=block.input)
                for block in final_message.content
                if block.type == "tool_use"
            ]
        )

        text = next(
            (block.text for block in final_message.content if block.type == "text"),
            None,
        )
        return ProviderResponse(
            message=AssistantMessage(text=text, tool_calls=tool_calls),
            stop=final_message.stop_reason == "end_turn",
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
        provider_messages = self.__get_provider_messages(messages)
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

        to_summarize = self.__get_provider_messages(messages[:-keep_last])
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
            has_tool_calls = bool(message.tool_calls)
            if not has_tool_calls and message.text is not None:
                return {"role": "assistant", "content": message.text}

            tm = {"role": "assistant"}
            tc = []

            if message.text is not None:
                tc = [{"type": "text", "text": message.text}]

            for tool_call in message.tool_calls:
                tc.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }
                )

            tm["content"] = tc
            return tm

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
