"""Headless mode: non-interactive programmatic use of avoid-agent.

This module provides structured JSON I/O for headless operation.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from avoid_agent.agent.context import ContextStrategy
from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.cli.shared import gather_initial_context
from avoid_agent.infra.config import DEFAULT_MODEL, MAX_TOKENS
from avoid_agent.learnings import capture_session
from avoid_agent.permissions import load_allowed, save_allowed
from avoid_agent.providers import AssistantMessage
from avoid_agent import providers
from avoid_agent.prompts import build_system_prompt
from avoid_agent.session import load_session, save_session

if TYPE_CHECKING:
    argparse.Namespace


def _headless_fatal(message: str) -> None:
    """Print a fatal error as JSON to stderr and exit."""
    sys.stderr.write(json.dumps({"type": "fatal", "message": message}) + "\n")
    sys.exit(1)


def _display_text_for_assistant_message(message: AssistantMessage) -> str:
    """Extract display text from an assistant message."""
    from avoid_agent.agent.runtime import _parse_structured_action
    structured = _parse_structured_action(message.text)
    if structured is None:
        return message.text or ""

    if structured.tool == "blocker":
        reason = structured.args.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason

    if structured.tool == "complete":
        summary = structured.args.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary

    return structured.plan


def run(args) -> int:
    """Run the agent in headless mode.

    This is the main entry point for the `avoid-agent headless` command.

    Args:
        args: argparse.Namespace with the following attributes:
            - prompt: Single-turn prompt text
            - model: Model override
            - max_turns: Max turns in multi-turn stdin mode
            - auto_approve: Auto-approve all bash commands
            - session: Session name for persistence
            - no_session: Don't persist session
            - context_strategy: Context management strategy
            - context_budget: Max input tokens for context
            - compaction_cooldown: Min turns between compactions

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    load_dotenv()

    model = args.model or DEFAULT_MODEL
    max_tokens = MAX_TOKENS
    max_turns = args.max_turns
    auto_approve = args.auto_approve
    session_name = args.session or f"headless-{__import__('os').getpid()}"
    no_session = args.no_session
    context_strategy: ContextStrategy = args.context_strategy

    valid_strategies: set[ContextStrategy] = {"window", "compact", "compact+window"}
    if context_strategy not in valid_strategies:
        _headless_fatal(f"Invalid context strategy: {context_strategy}")

    tool_definitions = find_available_tools()
    cwd, git_status, top_level_structure = gather_initial_context()
    system = build_system_prompt(
        working_directory=cwd,
        git_status=git_status,
        top_level_file_structure=top_level_structure,
    )
    provider = providers.get_provider(model=model, system=system, max_tokens=max_tokens)

    allowed_prefixes = load_allowed()

    def emit_event(event_dict: dict) -> None:
        sys.stderr.write(json.dumps(event_dict) + "\n")
        sys.stderr.flush()

    def headless_request_permission(command: str) -> str:
        if auto_approve:
            emit_event({"type": "permission_auto_approved", "command": command})
            return "allow"
        emit_event({"type": "permission_denied", "command": command})
        return "deny"

    if no_session:
        messages = []
    else:
        saved = load_session(cwd, session_name)
        messages = saved if saved is not None else []

    def handle_event(event: RuntimeEvent) -> None:
        if event.type == "provider_event" and event.provider_event:
            pe = event.provider_event
            if pe.type == "text_delta" and pe.text:
                emit_event({"type": "text_delta", "text": pe.text})
            elif pe.type == "tool_call_detected" and pe.tool_call:
                tc = pe.tool_call
                emit_event({"type": "tool_call_detected", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
            elif pe.type == "reasoning_item" and pe.reasoning_item:
                emit_event({"type": "reasoning", "item": pe.reasoning_item})
            elif pe.type == "status" and pe.status:
                emit_event({"type": "status", "message": pe.status})
            elif pe.type == "error" and pe.error:
                emit_event({"type": "error", "message": pe.error})
        elif event.type == "tool_execution_start" and event.tool_call:
            tc = event.tool_call
            emit_event({"type": "tool_execution_start", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
        elif event.type == "tool_result" and event.tool_result:
            tr = event.tool_result
            emit_event({
                "type": "tool_result",
                "id": tr.tool_call_id,
                "name": tr.tool_name,
                "content": tr.content[:2000],
                "is_error": tr.is_error,
            })
        elif event.type == "validation_error" and event.message:
            emit_event({"type": "validation_error", "message": event.message})
        elif event.type == "structured_action" and event.message:
            emit_event({"type": "structured_action", "message": event.message})
        elif event.type == "context_trimmed" and event.message:
            emit_event({"type": "context_trimmed", "message": event.message})

    def run_one_turn(prompt: str, turn_number: int) -> dict:
        nonlocal messages
        emit_event({"type": "turn_start", "turn": turn_number, "prompt": prompt[:500]})

        tool_calls_log: list[dict] = []

        def capturing_handler(event: RuntimeEvent) -> None:
            handle_event(event)
            if event.type == "tool_execution_start" and event.tool_call:
                tool_calls_log.append({
                    "id": event.tool_call.id,
                    "name": event.tool_call.name,
                    "arguments": event.tool_call.arguments,
                })
            if event.type == "tool_result" and event.tool_result:
                for tc in reversed(tool_calls_log):
                    if tc["id"] == event.tool_result.tool_call_id:
                        tc["result_preview"] = event.tool_result.content[:500]
                        tc["is_error"] = event.tool_result.is_error
                        break

        try:
            runtime = AgentRuntime(
                provider=provider,
                tool_definitions=tool_definitions,
                allowed_prefixes=allowed_prefixes,
                request_permission=headless_request_permission,
                save_allowed_prefixes=save_allowed if not no_session else None,
                on_event=capturing_handler,
                context_strategy=context_strategy,
                token_budget=args.context_budget,
                compaction_cooldown_turns=args.compaction_cooldown,
            )
            result = runtime.run_user_turn(messages, prompt)
            messages = result.messages

            assistant_text = None
            if messages and isinstance(messages[-1], AssistantMessage):
                assistant_text = _display_text_for_assistant_message(messages[-1])

            if not no_session:
                save_session(cwd, messages, session_name)

            emit_event({"type": "turn_complete", "turn": turn_number, "input_tokens": result.input_tokens})

            return {
                "success": True,
                "turn": turn_number,
                "input_tokens": result.input_tokens,
                "assistant_text": assistant_text,
                "session": session_name if not no_session else None,
                "messages_count": len(messages),
                "tool_calls": tool_calls_log,
                "error": None,
            }
        except Exception as e:  # pylint: disable=broad-except
            emit_event({"type": "error", "message": str(e)})
            return {
                "success": False,
                "turn": turn_number,
                "input_tokens": 0,
                "assistant_text": None,
                "session": session_name if not no_session else None,
                "messages_count": len(messages),
                "tool_calls": tool_calls_log,
                "error": str(e),
            }

    # Single-turn mode
    if args.prompt is not None:
        result = run_one_turn(args.prompt, 1)
        failed_tool_calls = [tool_call for tool_call in result["tool_calls"] if tool_call.get("is_error") is True]
        errors = [result["error"]] if isinstance(result["error"], str) and result["error"].strip() else []
        session_id = session_name
        capture_session(session_id, failed_tool_calls, errors)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()
        return 0 if result["success"] else 1

    # Multi-turn stdin mode
    if sys.stdin.isatty():
        sys.stderr.write(
            "Error: headless mode requires --prompt or piped stdin.\n"
            "Usage: python -m avoid_agent headless --prompt '...'\n"
            "   or: echo '{\"prompt\": \"...\"}' | python -m avoid_agent headless\n"
        )
        return 1

    turn = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            emit_event({"type": "error", "message": f"Invalid JSON input: {e}"})
            continue

        if request.get("command") == "quit":
            emit_event({"type": "session_end", "turns": turn})
            break

        prompt = request.get("prompt")
        if not prompt or not isinstance(prompt, str):
            emit_event({"type": "error", "message": "Input must have a 'prompt' string field"})
            continue

        turn += 1
        if turn > max_turns:
            emit_event({"type": "error", "message": f"Max turns ({max_turns}) exceeded"})
            error_result = {"success": False, "error": f"Max turns ({max_turns}) exceeded", "turn": turn}
            sys.stdout.write(json.dumps(error_result) + "\n")
            sys.stdout.flush()
            break

        result = run_one_turn(prompt, turn)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()

    return 0
