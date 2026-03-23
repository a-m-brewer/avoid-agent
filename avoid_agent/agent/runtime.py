"""Core agent runtime loop and turn validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from avoid_agent.agent.tools import ToolDefinition, run_tool
from avoid_agent.permissions import command_prefix
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderEvent,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
)

_COMPLETION_CLAIM_RE = re.compile(
    r"("
    r"I\s+(?:changed|edited|modified|updated|applied|wired"
    r"|patched|rewrote|implemented|refactored)"
    r"|✅\s*(?:I\s|What\s|Changed)"
    r")",
    re.MULTILINE,
)

_MUTATION_HINT_RE = re.compile(
    r"\b("
    r"add|apply|build|change|create|edit|fix|implement|modify|patch|refactor|remove|rename|rewrite|update|wire|write"
    r")\b"
)
_NON_MUTATION_HINT_RE = re.compile(
    r"\b("
    r"analy[sz]e|analysis|compare|explain|gap analysis|investigate|look into|plan|review|summari[sz]e|why|what"
    r")\b"
)
_ALLOWS_NO_TOOL_RE = re.compile(
    r"("
    r"\?|"
    r"\b(?:I\s+can(?:not|'t)|cannot|can't)\b|"
    r"\b(?:need|needs)\b.+\b(?:input|confirmation|details|path|file|filename)\b|"
    r"\b(?:do you want|which|what file|please provide)\b|"
    r"\b(?:no changes? (?:are|were) needed|nothing to change|already up to date|already implemented)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

MUTATION_CAPABLE_TOOLS = {"write_file", "edit_file", "run_bash"}
MAX_COMPLETION_RETRIES = 2
HALLUCINATION_CORRECTION = (
    "[SYSTEM] The user asked for implementation work, but your previous turn ended "
    "without using a mutation-capable tool. Text responses do not edit files. "
    "If changes are needed, you MUST call write_file, edit_file, or run_bash. "
    "If you cannot proceed, explain the blocker or ask a specific question."
)


@dataclass
class RuntimeEvent:
    """Runtime event emitted while processing a user turn."""

    type: str
    provider_event: ProviderEvent | None = None
    tool_call: ProviderToolCall | None = None
    tool_result: ToolResultMessage | None = None
    message: str | None = None


@dataclass
class RunTurnResult:
    """Result of executing one user turn."""

    messages: list[Message]
    input_tokens: int = 0


def _looks_like_hallucinated_completion(message: AssistantMessage) -> bool:
    """Detect explicit completion claims without relying on them for control flow."""
    if message.tool_calls:
        return False
    if not message.text:
        return False
    return bool(_COMPLETION_CLAIM_RE.search(message.text))


def _request_expects_mutation(text: str) -> bool:
    lowered = text.lower()
    if _MUTATION_HINT_RE.search(lowered):
        return True
    if "?" in lowered and not _MUTATION_HINT_RE.search(lowered):
        return False
    if _NON_MUTATION_HINT_RE.search(lowered):
        return False
    return False


def _allows_no_tool_completion(message: AssistantMessage) -> bool:
    if message.stop_reason in ("error", "aborted"):
        return True
    if not message.text:
        return False
    return bool(_ALLOWS_NO_TOOL_RE.search(message.text))


def _tool_result_is_error(content: str) -> bool:
    return content.startswith("Error:") or content == "User denied this command."


class AgentRuntime:
    """Runs a structured assistant/tool loop for one user turn."""

    def __init__(
        self,
        provider: Provider,
        tool_definitions: list[ToolDefinition],
        allowed_prefixes: set[str],
        request_permission: Callable[[str], str] | None = None,
        save_allowed_prefixes: Callable[[set[str]], None] | None = None,
        on_event: Callable[[RuntimeEvent], None] | None = None,
    ):
        self._provider = provider
        self._tool_definitions = tool_definitions
        self._allowed_prefixes = allowed_prefixes
        self._request_permission = request_permission
        self._save_allowed_prefixes = save_allowed_prefixes
        self._on_event = on_event

    def run_user_turn(self, messages: list[Message], text: str) -> RunTurnResult:
        run_messages = [*messages, UserMessage(text=text)]
        expects_mutation = _request_expects_mutation(text)
        mutation_outcomes = 0
        retries = 0
        input_tokens = 0

        while True:
            with self._provider.stream(
                messages=run_messages,
                tools=self._tool_definitions,
            ) as stream:
                for provider_event in stream.event_stream():
                    self._emit(RuntimeEvent(type="provider_event", provider_event=provider_event))
                response = stream.get_final_message()

            input_tokens = response.input_tokens
            run_messages.append(response.message)

            if response.message.stop_reason in ("error", "aborted"):
                break

            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    self._emit(RuntimeEvent(type="tool_execution_start", tool_call=tool_call))
                    tool_result = self._run_tool_call(tool_call)
                    run_messages.append(tool_result)
                    if tool_call.name in MUTATION_CAPABLE_TOOLS:
                        mutation_outcomes += 1
                    self._emit(RuntimeEvent(type="tool_result", tool_result=tool_result))
                retries = 0
                continue

            validation_error = self._validate_turn(
                expects_mutation=expects_mutation,
                mutation_outcomes=mutation_outcomes,
                message=response.message,
            )
            if validation_error is None:
                break

            if retries < MAX_COMPLETION_RETRIES:
                retries += 1
                run_messages.append(UserMessage(text=HALLUCINATION_CORRECTION))
                self._emit(RuntimeEvent(type="validation_error", message=validation_error))
                continue

            response.message.stop_reason = "error"
            response.message.error_message = validation_error
            self._emit(RuntimeEvent(type="validation_error", message=validation_error))
            break

        return RunTurnResult(messages=run_messages, input_tokens=input_tokens)

    def _run_tool_call(self, tool_call: ProviderToolCall) -> ToolResultMessage:
        if tool_call.name == "run_bash":
            command = tool_call.arguments.get("command", "")
            prefix = command_prefix(command)
            if prefix not in self._allowed_prefixes and self._request_permission is not None:
                decision = self._request_permission(command)
                if decision == "deny":
                    return ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content="User denied this command.",
                        is_error=True,
                    )
                if decision == "save":
                    self._allowed_prefixes.add(prefix)
                    if self._save_allowed_prefixes is not None:
                        self._save_allowed_prefixes(self._allowed_prefixes)

        content = run_tool(tool_call.name, tool_call.arguments)
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
            is_error=_tool_result_is_error(content),
        )

    def _validate_turn(
        self,
        *,
        expects_mutation: bool,
        mutation_outcomes: int,
        message: AssistantMessage,
    ) -> str | None:
        if not expects_mutation:
            return None
        if mutation_outcomes > 0:
            return None
        if _allows_no_tool_completion(message):
            return None

        if _looks_like_hallucinated_completion(message):
            return (
                "Hallucination detected: the assistant claimed edits were complete "
                "without using a mutation-capable tool."
            )

        return (
            "Implementation request ended without any mutation-capable tool execution. "
            "The assistant must either make the change with tools or explain a concrete blocker."
        )

    def _emit(self, event: RuntimeEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)
