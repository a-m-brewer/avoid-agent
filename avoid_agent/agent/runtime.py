"""Core agent runtime with explicit execution control and context management."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import threading
from typing import Callable

from avoid_agent.agent.context import ContextResult, ContextStrategy, prepare_context
from avoid_agent.agent.tools import ToolDefinition, ToolRunResult, run_tool
from avoid_agent.permissions import command_prefix
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderEvent,
    ProviderToolCall,
    ToolChoice,
    ToolResultMessage,
    UserMessage,
)

class TurnCancelledError(Exception):
    """Raised when a running user turn is cancelled via a cancel token."""


MAX_INVALID_STEP_RETRIES = 2
_SUCCESS_WITHOUT_EVIDENCE_RE = re.compile(
    r"\b(done|implemented|fixed|completed)\b",
    re.IGNORECASE,
)
_TOOL_ACCESS_BLOCKER_RE = re.compile(
    r"(did not include tool access|no tool access|cannot run .*tool|cannot run .*verification|"
    r"please allow me to run|allow me to run tests|required verification tools|"
    r"\bneed to (read|write|edit|run|modify|implement|inspect|check|call|execute|"
    r"look at|examine|update|add|create|search|find|make changes)\b|"
    r"\bneed (additional|more|further) .{0,50}(tool|file|command|call)\b|"
    r"\btool calls? (are |to )?(needed|required|necessary|to complete)\b)",
    re.IGNORECASE,
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


@dataclass(slots=True)
class StructuredAction:
    """Structured non-tool action proposed by the model."""

    plan: str
    tool: str
    args: dict


def _tool_result_is_error(content: str) -> bool:
    return content.startswith("Error:") or content == "User denied this command."


def _validate_tool_call(
    tool_call: ProviderToolCall, tool_definitions: list[ToolDefinition]
) -> str | None:
    """Validate a tool call against known definitions."""
    tool_def = next((t for t in tool_definitions if t.name == tool_call.name), None)
    if tool_def is None:
        return f"Unknown tool: {tool_call.name}"

    required_params = {p.name for p in tool_def.parameters if p.required}
    missing = required_params - set(tool_call.arguments.keys())
    if missing:
        return (
            f"Missing required arguments for {tool_call.name}: "
            f"{', '.join(sorted(missing))}"
        )

    return None


def _parse_structured_action(text: str | None) -> StructuredAction | None:
    """Parse a controller-compatible JSON action from assistant text."""
    if text is None:
        return None

    payload = text.strip()
    if not payload:
        return None

    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            payload = "\n".join(lines[1:-1]).strip()

    data = _decode_structured_json(payload)
    if data is None:
        return None

    if not isinstance(data, dict):
        return None

    plan = data.get("plan")
    action = data.get("action")
    if not isinstance(plan, str) or not plan.strip():
        return None
    if not isinstance(action, dict):
        return None

    tool = action.get("tool")
    args = action.get("args")
    if not isinstance(tool, str) or not tool.strip():
        return None
    if not isinstance(args, dict):
        return None

    return StructuredAction(plan=plan.strip(), tool=tool.strip(), args=args)


def _decode_structured_json(payload: str) -> dict | None:
    """Best-effort JSON extraction from assistant text.

    Providers do not always return a bare JSON object even when instructed to.
    Accept the first decodable JSON object so structured completion survives
    harmless wrappers like prose prefixes or fenced code blocks.
    """
    decoder = json.JSONDecoder()

    try:
        decoded = decoder.decode(payload)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        return decoded
    for suffix in ("}", "}}", "}}}"):
        try:
            decoded = decoder.decode(payload + suffix)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            return decoded

    # Common fenced format:
    # ```json
    # {...}
    # ```
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            fenced_payload = "\n".join(lines[1:-1]).strip()
            if fenced_payload.lower().startswith("json\n"):
                fenced_payload = fenced_payload[5:].strip()
            try:
                decoded = decoder.decode(fenced_payload)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                return decoded
            for suffix in ("}", "}}", "}}}"):
                try:
                    decoded = decoder.decode(fenced_payload + suffix)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, dict):
                    return decoded

    for index, char in enumerate(payload):
        if char != "{":
            continue
        try:
            decoded, _end = decoder.raw_decode(payload[index:])
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            return decoded
        for suffix in ("}", "}}", "}}}"):
            try:
                decoded, _end = decoder.raw_decode(payload[index:] + suffix)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                return decoded

    return None


def _looks_like_structured_terminal_response(text: str | None) -> bool:
    """Heuristic for malformed controller JSON emitted in place of a tool call."""
    if text is None:
        return False

    payload = text.strip()
    if not payload:
        return False

    if payload.startswith("{") or payload.startswith("```"):
        return True

    return '"plan"' in payload and '"action"' in payload


def _preview(text: str, limit: int = 280) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _display_text_for_structured_action(structured: StructuredAction) -> str:
    if structured.tool == "blocker":
        reason = structured.args.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    if structured.tool == "complete":
        summary = structured.args.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
    return structured.plan


class ExecutionController:
    """Validates tool proposals, executes them, and emits verifiable results."""

    def __init__(
        self,
        tool_definitions: list[ToolDefinition],
        allowed_prefixes: set[str],
        request_permission: Callable[[str], str] | None = None,
        save_allowed_prefixes: Callable[[set[str]], None] | None = None,
    ):
        self._tool_definitions = tool_definitions
        self._allowed_prefixes = allowed_prefixes
        self._request_permission = request_permission
        self._save_allowed_prefixes = save_allowed_prefixes

    def execute_tool_call(
        self,
        tool_call: ProviderToolCall,
        *,
        plan: str | None = None,
    ) -> ToolResultMessage:
        validation_error = _validate_tool_call(tool_call, self._tool_definitions)
        if validation_error:
            return ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=f"Error: {validation_error}",
                is_error=True,
                details={
                    "plan": plan or f"Execute {tool_call.name}",
                    "action": {"tool": tool_call.name, "args": tool_call.arguments},
                    "verification": {
                        "status": "failed",
                        "message": validation_error,
                    },
                },
            )

        permission_denied = self._maybe_request_permission(tool_call)
        if permission_denied is not None:
            return permission_denied

        tool_run = run_tool(tool_call.name, tool_call.arguments)
        proof = self._extract_proof(tool_call, tool_run)
        verification = self._verify(tool_call, tool_run, proof)
        is_error = _tool_result_is_error(tool_run.content) or verification["status"] != "verified"
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=tool_run.content,
            is_error=is_error,
            details={
                "plan": plan or f"Execute {tool_call.name}",
                "action": {"tool": tool_call.name, "args": tool_call.arguments},
                "proof": proof,
                "verification": verification,
            },
        )

    def extract_plan(self, message: AssistantMessage) -> str:
        structured = _parse_structured_action(message.text)
        if structured is not None:
            return structured.plan
        if message.text and message.text.strip():
            return message.text.strip()
        if message.tool_calls:
            tool_names = ", ".join(tool_call.name for tool_call in message.tool_calls)
            return f"Execute tool call(s): {tool_names}"
        return "No plan provided."

    def collect_verified_tool_ids(
        self,
        messages: list[Message],
        turn_start: int,
    ) -> list[str]:
        verified: list[str] = []
        for message in messages[turn_start:]:
            if not isinstance(message, ToolResultMessage):
                continue
            verification = message.details.get("verification", {})
            if verification.get("status") == "verified":
                verified.append(message.tool_call_id)
        return verified

    def validate_terminal_message(
        self,
        message: AssistantMessage,
        *,
        verified_tool_ids: list[str],
    ) -> str | None:
        if message.tool_calls:
            return None

        structured = _parse_structured_action(message.text)
        if structured is None:
            if _looks_like_structured_terminal_response(message.text):
                return (
                    "Invalid step: if no tool is called, the response must be a single "
                    "valid JSON object with `plan` and an `action` of `blocker` or "
                    "`complete`. The JSON was malformed or incomplete."
                )
            if message.text and _SUCCESS_WITHOUT_EVIDENCE_RE.search(message.text):
                return (
                    "Invalid step: success was claimed without a real tool execution and "
                    "without evidence."
                )
            return (
                "Invalid step: if no tool is called, the response must be JSON with "
                "`plan` and an `action` of `blocker` or `complete`."
            )

        if structured.tool == "blocker":
            reason = structured.args.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return "Invalid blocker action: args.reason must be a non-empty string."
            if self._tool_definitions and _TOOL_ACCESS_BLOCKER_RE.search(reason):
                return (
                    "Invalid blocker action: tool access is available in this turn. "
                    "If verification or inspection is needed, call the real tool instead "
                    "of asking for permission in text."
                )
            return None

        if structured.tool == "complete":
            evidence = structured.args.get("evidence")
            if not isinstance(evidence, list):
                return "Invalid complete action: args.evidence must be a list."

            # Allow empty evidence only when no tools were executed this turn
            # (e.g. a conversational reply that genuinely needed no tool calls).
            if not evidence and verified_tool_ids:
                return (
                    "Invalid complete action: args.evidence must be a non-empty list "
                    "when tool calls were made this turn."
                )

            missing = [
                evidence_id
                for evidence_id in evidence
                if not isinstance(evidence_id, str) or evidence_id not in verified_tool_ids
            ]
            if missing:
                return (
                    "Invalid complete action: evidence must reference verified tool_call_id "
                    f"values from this turn. Unknown or unverified: {', '.join(map(str, missing))}"
                )
            return None

        return (
            f"Invalid step: action.tool `{structured.tool}` requires a real tool call. "
            "Only `blocker` and `complete` may omit tool execution."
        )

    def build_state_message(
        self,
        messages: list[Message],
        turn_start: int,
        user_request: str,
    ) -> UserMessage:
        turn_messages = messages[turn_start:]
        tool_results = [
            message for message in turn_messages if isinstance(message, ToolResultMessage)
        ]
        verified_ids = self.collect_verified_tool_ids(messages, turn_start)
        last_plan = self._last_plan(tool_results)

        lines = [
            "[EXECUTION STATE - CONTROLLER GENERATED]",
            "This is grounded controller state, not a new user request.",
            f"Current user request: {user_request}",
            f"Current plan: {last_plan}",
            "Verified tool_call_id values this turn: "
            + (", ".join(verified_ids) if verified_ids else "none"),
            "Completed steps:",
        ]

        if tool_results:
            for tool_result in tool_results[-4:]:
                verification = tool_result.details.get("verification", {})
                action = tool_result.details.get("action", {})
                proof = tool_result.details.get("proof", {})
                tool_name = action.get("tool") or tool_result.tool_name or "tool"
                outcome = verification.get("status", "unknown")
                proof_label = proof.get("kind", "proof")
                lines.append(
                    f"- {tool_result.tool_call_id}: {tool_name} -> {outcome} ({proof_label})"
                )
        else:
            lines.append("- none")

        lines.append("Recent tool outputs:")
        if tool_results:
            for tool_result in tool_results[-3:]:
                lines.append(
                    f"- {tool_result.tool_call_id}: {_preview(tool_result.content)}"
                )
        else:
            lines.append("- none")

        lines.append("Not yet executed:")
        if tool_results:
            lines.append("- Only the tool calls listed above are real. Nothing else has happened.")
        else:
            lines.append("- No tool has been executed yet in this turn.")

        lines.extend(
            [
                "If you need another action, call a real tool.",
                "If you are blocked, respond with JSON only:",
                '{"plan":"...","action":{"tool":"blocker","args":{"reason":"..."}}}',
                "If the task is complete, respond with JSON only and cite verified evidence:",
                '{"plan":"...","action":{"tool":"complete","args":{"summary":"...","evidence":["tool_call_id"]}}}',
            ]
        )
        return UserMessage(text="\n".join(lines))

    def build_retry_message(
        self,
        validation_error: str,
        verified_tool_ids: list[str],
    ) -> UserMessage:
        evidence = ", ".join(verified_tool_ids) if verified_tool_ids else "none"
        return UserMessage(
            text=(
                "[EXECUTION CONTROLLER]\n"
                f"{validation_error}\n"
                f"Verified evidence available this turn: {evidence}\n"
                "Next response must be one of:\n"
                "1. A real tool call via the tool API.\n"
                "2. JSON blocker only: "
                '{"plan":"...","action":{"tool":"blocker","args":{"reason":"..."}}}\n'
                "3. JSON complete only: "
                '{"plan":"...","action":{"tool":"complete","args":{"summary":"...","evidence":["tool_call_id"]}}}\n'
                "Do not claim work was done unless the evidence is real."
            )
        )

    def _maybe_request_permission(
        self,
        tool_call: ProviderToolCall,
    ) -> ToolResultMessage | None:
        if tool_call.name != "run_bash":
            return None

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
                    details={
                        "action": {"tool": tool_call.name, "args": tool_call.arguments},
                        "verification": {
                            "status": "failed",
                            "message": "User denied this command.",
                        },
                    },
                )
            if decision == "save":
                self._allowed_prefixes.add(prefix)
                if self._save_allowed_prefixes is not None:
                    self._save_allowed_prefixes(self._allowed_prefixes)
        return None

    def _extract_proof(
        self,
        tool_call: ProviderToolCall,
        tool_run: ToolRunResult,
    ) -> dict:
        proof = tool_run.details.get("proof")
        if isinstance(proof, dict) and proof:
            return proof
        return {
            "kind": "tool_output",
            "tool": tool_call.name,
            "preview": _preview(tool_run.content),
        }

    def _verify(
        self,
        tool_call: ProviderToolCall,
        tool_run: ToolRunResult,
        proof: dict,
    ) -> dict:
        if _tool_result_is_error(tool_run.content):
            return {"status": "failed", "message": tool_run.content}

        if tool_call.name == "read_file":
            if proof.get("kind") != "file_read" or not proof.get("path"):
                return {
                    "status": "failed",
                    "message": "Missing file-read proof for read_file.",
                }
            return {
                "status": "verified",
                "message": f"Read {proof['path']} and captured file contents.",
            }

        if tool_call.name == "write_file":
            if proof.get("kind") != "file_change":
                return {
                    "status": "failed",
                    "message": "Missing file-change proof for write_file.",
                }
            if not proof.get("changed"):
                return {
                    "status": "failed",
                    "message": "write_file reported success but did not change the file.",
                }
            if proof.get("after_sha256") != proof.get("expected_sha256"):
                return {
                    "status": "failed",
                    "message": "write_file verification failed: file contents do not match the requested write.",
                }
            return {
                "status": "verified",
                "message": f"Wrote and verified {proof['path']}.",
            }

        if tool_call.name == "edit_file":
            if proof.get("kind") != "file_change":
                return {
                    "status": "failed",
                    "message": "Missing file-change proof for edit_file.",
                }
            if not proof.get("changed"):
                return {
                    "status": "failed",
                    "message": "edit_file reported success but did not change the file.",
                }
            return {
                "status": "verified",
                "message": f"Edited and verified {proof['path']}.",
            }

        if tool_call.name == "run_bash":
            if proof.get("kind") != "command":
                return {
                    "status": "failed",
                    "message": "Missing command proof for run_bash.",
                }
            exit_code = proof.get("exit_code")
            if exit_code != 0:
                return {
                    "status": "failed",
                    "message": f"run_bash exited with status {exit_code}.",
                }
            return {
                "status": "verified",
                "message": f"run_bash completed with exit code {exit_code}.",
            }

        if not tool_run.content.strip():
            return {
                "status": "failed",
                "message": "Tool returned empty output, so the step has no proof.",
            }
        return {
            "status": "verified",
            "message": f"{tool_call.name} returned non-empty output.",
        }

    @staticmethod
    def _last_plan(tool_results: list[ToolResultMessage]) -> str:
        for tool_result in reversed(tool_results):
            plan = tool_result.details.get("plan")
            if isinstance(plan, str) and plan.strip():
                return plan
        return "No verified plan yet."


class AgentRuntime:
    """Runs the assistant/tool loop with a strict execution controller."""

    def __init__(
        self,
        provider: Provider,
        tool_definitions: list[ToolDefinition],
        allowed_prefixes: set[str],
        request_permission: Callable[[str], str] | None = None,
        save_allowed_prefixes: Callable[[set[str]], None] | None = None,
        on_event: Callable[[RuntimeEvent], None] | None = None,
        context_strategy: ContextStrategy = "compact+window",
        token_budget: int | None = None,
        tool_choice: ToolChoice = "auto",
        compaction_cooldown_turns: int = 3,
    ):
        self._provider = provider
        self._on_event = on_event
        self._context_strategy = context_strategy
        # Use provided token_budget or compute dynamically from model
        if token_budget is not None:
            self._token_budget = token_budget
        else:
            from avoid_agent.providers import compute_token_budget
            self._token_budget = compute_token_budget(provider.model, provider.max_tokens)
        self._tool_choice = tool_choice
        self._tool_definitions = tool_definitions
        self._compaction_cooldown_turns = compaction_cooldown_turns
        self._turn_count = 0
        self._last_compaction_turn = -compaction_cooldown_turns  # Allow first compaction
        self._controller = ExecutionController(
            tool_definitions=tool_definitions,
            allowed_prefixes=allowed_prefixes,
            request_permission=request_permission,
            save_allowed_prefixes=save_allowed_prefixes,
        )

    def run_user_turn(
        self,
        messages: list[Message],
        text: str,
        cancel_token: threading.Event | None = None,
    ) -> RunTurnResult:
        """Execute one user turn, optionally supporting mid-turn cancellation.

        Args:
            messages: Conversation history so far.
            text: The new user message text.
            cancel_token: Optional threading.Event.  When the event is set the
                loop exits at the next safe checkpoint (between provider calls
                or between tool executions) and raises TurnCancelledError.
        """
        run_messages = [*messages, UserMessage(text=text)]
        turn_start = len(messages)
        input_tokens = 0
        invalid_step_retries = 0
        synthetic_tool_call_count = 0

        def _check_cancel() -> None:
            if cancel_token is not None and cancel_token.is_set():
                raise TurnCancelledError("Turn cancelled by user")

        while True:
            _check_cancel()
            context_messages = self._prepare_messages(run_messages, turn_start, text)
            with self._provider.stream(
                messages=context_messages,
                tools=self._tool_definitions,
                tool_choice=self._tool_choice,
            ) as stream:
                for provider_event in stream.event_stream():
                    self._emit(RuntimeEvent(type="provider_event", provider_event=provider_event))
                    _check_cancel()
                response = stream.get_final_message()

            input_tokens = response.input_tokens
            run_messages.append(response.message)

            if response.message.stop_reason in ("error", "aborted"):
                break

            structured = None
            if not response.message.tool_calls:
                structured = _parse_structured_action(response.message.text)
                if structured is not None:
                    self._emit(
                        RuntimeEvent(
                            type="structured_action",
                            message=_display_text_for_structured_action(structured),
                        )
                    )

            if response.message.tool_calls:
                plan = self._controller.extract_plan(response.message)
                for tool_call in response.message.tool_calls:
                    _check_cancel()
                    self._emit(RuntimeEvent(type="tool_execution_start", tool_call=tool_call))
                    tool_result = self._controller.execute_tool_call(tool_call, plan=plan)
                    run_messages.append(tool_result)
                    self._emit(RuntimeEvent(type="tool_result", tool_result=tool_result))
                self._turn_count += 1
                invalid_step_retries = 0
                continue

            if structured is not None and structured.tool not in ("blocker", "complete"):
                synthetic_tool_call_count += 1
                tool_call = ProviderToolCall(
                    id=f"json_action_{synthetic_tool_call_count}",
                    name=structured.tool,
                    arguments=structured.args,
                )
                _check_cancel()
                self._emit(RuntimeEvent(type="tool_execution_start", tool_call=tool_call))
                tool_result = self._controller.execute_tool_call(tool_call, plan=structured.plan)
                run_messages.append(tool_result)
                self._emit(RuntimeEvent(type="tool_result", tool_result=tool_result))
                self._turn_count += 1
                invalid_step_retries = 0
                continue

            verified_tool_ids = self._controller.collect_verified_tool_ids(
                run_messages, turn_start
            )
            validation_error = self._controller.validate_terminal_message(
                response.message,
                verified_tool_ids=verified_tool_ids,
            )
            if validation_error is None:
                break

            self._emit(RuntimeEvent(type="validation_error", message=validation_error))
            if invalid_step_retries >= MAX_INVALID_STEP_RETRIES:
                response.message.stop_reason = "error"
                response.message.error_message = validation_error
                break

            invalid_step_retries += 1
            run_messages.append(
                self._controller.build_retry_message(validation_error, verified_tool_ids)
            )

        return RunTurnResult(messages=run_messages, input_tokens=input_tokens)

    def _summarize(self, prompt: str) -> str:
        """Use the provider to summarize text. Used as the callback for compaction."""
        summary_messages = [UserMessage(text=f"[INTERNAL SUMMARIZER]\n{prompt}")]
        with self._provider.stream(
            messages=summary_messages, tools=[], tool_choice="none"
        ) as stream:
            for _ in stream.event_stream():
                pass
            response = stream.get_final_message()
        return response.message.text or ""

    def _prepare_messages(
        self,
        messages: list[Message],
        turn_start: int,
        user_request: str,
    ) -> list[Message]:
        """Apply context management and append grounded controller state."""
        # Check hysteresis: don't compact if we compacted recently unless way over budget
        turns_since_compaction = self._turn_count - self._last_compaction_turn
        over_budget_threshold = 1.10  # Only force compaction if 10% over budget

        if self._context_strategy in ("compact", "compact+window"):
            # Check if we're over budget enough to warrant immediate compaction
            estimated = self._estimate_tokens(messages)
            if estimated <= self._token_budget * over_budget_threshold:
                # Within hysteresis window - skip compaction
                if turns_since_compaction < self._compaction_cooldown_turns:
                    grounded_messages = list(messages)
                    grounded_messages.append(
                        self._controller.build_state_message(messages, turn_start, user_request)
                    )
                    return grounded_messages

        result: ContextResult = prepare_context(
            messages=messages,
            token_budget=self._token_budget,
            strategy=self._context_strategy,
            summarize=self._summarize,
        )
        if result.action != "none":
            self._last_compaction_turn = self._turn_count
            self._emit(
                RuntimeEvent(
                    type="context_trimmed",
                    message=(
                        f"Context {result.action}: "
                        f"{result.original_tokens} -> {result.trimmed_tokens} tokens"
                    ),
                )
            )
        grounded_messages = list(result.messages)
        grounded_messages.append(
            self._controller.build_state_message(messages, turn_start, user_request)
        )
        return grounded_messages

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate tokens using the same heuristic as context.py."""
        from avoid_agent.agent.context import estimate_tokens
        return estimate_tokens(messages)

    def _emit(self, event: RuntimeEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)
