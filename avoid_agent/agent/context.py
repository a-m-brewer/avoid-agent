from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from avoid_agent.providers import (
    AssistantMessage,
    AssistantTextBlock,
    AssistantThinkingBlock,
    Message,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
)

# Type alias for strategy selection
ContextStrategy = Literal["window", "compact", "compact+window"]


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate token count for a message list using a chars/4 heuristic."""
    char_count = 0
    for message in messages:
        if isinstance(message, UserMessage):
            char_count += len(message.text)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, AssistantTextBlock):
                    char_count += len(block.text)
                elif isinstance(block, AssistantThinkingBlock):
                    char_count += len(block.text)
                elif isinstance(block, ProviderToolCall):
                    char_count += len(block.name)
                    char_count += len(str(block.arguments))
                    char_count += len(block.id)
        elif isinstance(message, ToolResultMessage):
            char_count += len(message.content)
            char_count += len(message.tool_call_id)
            char_count += len(message.tool_name or "")

    return int(char_count / 4)


# ---------------------------------------------------------------------------
# Turn grouping
# ---------------------------------------------------------------------------

TurnGroup = list[Message]


def group_turns(messages: list[Message]) -> list[TurnGroup]:
    """Split a flat message list into indivisible turn groups.

    Grouping rules:
    - A UserMessage is its own group.
    - An AssistantMessage starts a new group. If it has tool_calls,
      all immediately following ToolResultMessages belong to the same group.
    - An AssistantMessage without tool_calls is its own group.
    """
    groups: list[TurnGroup] = []
    current: TurnGroup = []

    for message in messages:
        if isinstance(message, UserMessage):
            # User messages are always their own group
            if current:
                groups.append(current)
            groups.append([message])
            current = []

        elif isinstance(message, AssistantMessage):
            # Start a new group for each assistant message
            if current:
                groups.append(current)
            current = [message]

        elif isinstance(message, ToolResultMessage):
            # Tool results attach to the current assistant group
            current.append(message)

    # Don't forget the last group
    if current:
        groups.append(current)

    return groups


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def sliding_window(
    messages: list[Message],
    token_budget: int,
) -> list[Message]:
    """Keep the first turn group and as many recent groups as fit the budget.

    Returns a trimmed message list that fits within token_budget.
    The first group (usually the initial user message) is always kept
    because it establishes the task context.
    """
    groups = group_turns(messages)

    if len(groups) <= 1:
        return messages

    # The first group is always pinned
    first_group = groups[0]
    first_cost = estimate_tokens(first_group)

    # Walk backwards from the end, accumulating groups that fit
    remaining_budget = token_budget - first_cost
    kept_from_end: list[TurnGroup] = []

    for group in reversed(groups[1:]):
        group_cost = estimate_tokens(group)
        if group_cost > remaining_budget:
            break  # Can't fit this group or anything before it
        kept_from_end.append(group)
        remaining_budget -= group_cost

    # Reverse back to chronological order
    kept_from_end.reverse()

    # Flatten: first group + kept recent groups
    result: list[Message] = []
    result.extend(first_group)
    for group in kept_from_end:
        result.extend(group)

    return result


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

COMPACT_PROMPT = """\
Summarize the following conversation between a user and a coding assistant.
Preserve these details — the assistant will continue working with only your summary as context:

- What the user originally asked for
- Which files were read, created, or modified, and what changes were made
- Key decisions, trade-offs, or design choices discussed
- Errors encountered and how they were resolved
- The current state of the task (what's done, what's still pending)

Be concise but complete. Use bullet points. Do not editorialize or add suggestions.\
"""


def _format_messages_for_summary(messages: list[Message]) -> str:
    """Render messages as plain text for the compaction prompt."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            parts.append(f"USER: {message.text}")
        elif isinstance(message, AssistantMessage):
            if message.text:
                parts.append(f"ASSISTANT: {message.text}")
            for tc in message.tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
                parts.append(f"ASSISTANT [tool_call]: {tc.name}({args_str})")
        elif isinstance(message, ToolResultMessage):
            label = f"TOOL RESULT ({message.tool_name or 'unknown'})"
            # Truncate large tool results — the summary doesn't need full file contents
            content = message.content
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"
            parts.append(f"{label}: {content}")
    return "\n\n".join(parts)


# Type for the summarize callback: takes a prompt string, returns summary text
SummarizeFn = Callable[[str], str]


def compact_messages(
    messages: list[Message],
    summarize: SummarizeFn,
    keep_last: int = 4,
) -> list[Message]:
    """Compact old messages into a summary, keeping the last N turn groups.

    Args:
        messages: Full conversation history.
        summarize: Callable that takes a prompt and returns a summary string.
                   This is how we stay provider-agnostic — the caller passes
                   in a function that calls whatever LLM they want.
        keep_last: Number of recent turn groups to preserve verbatim.

    Returns:
        A shorter message list: [first_group, summary_message, ...recent_groups]
    """
    groups = group_turns(messages)

    # Nothing to compact if the conversation is short
    if len(groups) <= keep_last + 1:
        return messages

    first_group = groups[0]
    groups_to_compact = groups[1:-keep_last]
    recent_groups = groups[-keep_last:]

    # Flatten the groups we're compacting into a message list for summarization
    messages_to_summarize: list[Message] = []
    for group in groups_to_compact:
        messages_to_summarize.extend(group)

    # Build the prompt and call the summarizer
    conversation_text = _format_messages_for_summary(messages_to_summarize)
    full_prompt = f"{COMPACT_PROMPT}\n\n---\n\n{conversation_text}"
    summary_text = summarize(full_prompt)

    # The summary becomes a UserMessage so it slots into any provider's format
    summary_message = UserMessage(
        text=f"[CONVERSATION SUMMARY]\n{summary_text}"
    )

    # Reassemble: first group + summary + recent groups
    result: list[Message] = []
    result.extend(first_group)
    result.append(summary_message)
    for group in recent_groups:
        result.extend(group)

    return result


# ---------------------------------------------------------------------------
# Combined context manager
# ---------------------------------------------------------------------------


@dataclass
class ContextResult:
    """Result of context preparation, including what action was taken."""

    messages: list[Message]
    action: Literal["none", "window", "compact", "compact+window"]
    original_tokens: int
    trimmed_tokens: int


def prepare_context(
    messages: list[Message],
    token_budget: int,
    strategy: ContextStrategy = "compact+window",
    summarize: SummarizeFn | None = None,
    keep_last: int = 4,
) -> ContextResult:
    """Prepare messages to fit within a token budget using the chosen strategy.

    Strategies:
        "window"          — sliding window only, drops old groups
        "compact"         — summarize old groups, keep recent ones
        "compact+window"  — compact first, then apply window if still over budget

    The compact strategies require a summarize callback.
    """
    original_tokens = estimate_tokens(messages)

    # If we're within budget, no trimming needed
    if original_tokens <= token_budget:
        return ContextResult(
            messages=messages,
            action="none",
            original_tokens=original_tokens,
            trimmed_tokens=original_tokens,
        )

    if strategy == "window":
        result = sliding_window(messages, token_budget)
        return ContextResult(
            messages=result,
            action="window",
            original_tokens=original_tokens,
            trimmed_tokens=estimate_tokens(result),
        )

    if strategy == "compact":
        if summarize is None:
            raise ValueError("compact strategy requires a summarize callback")
        result = compact_messages(messages, summarize, keep_last)
        return ContextResult(
            messages=result,
            action="compact",
            original_tokens=original_tokens,
            trimmed_tokens=estimate_tokens(result),
        )

    if strategy == "compact+window":
        if summarize is None:
            raise ValueError("compact+window strategy requires a summarize callback")
        compacted = compact_messages(messages, summarize, keep_last)
        compacted_tokens = estimate_tokens(compacted)
        if compacted_tokens > token_budget:
            result = sliding_window(compacted, token_budget)
            return ContextResult(
                messages=result,
                action="compact+window",
                original_tokens=original_tokens,
                trimmed_tokens=estimate_tokens(result),
            )
        return ContextResult(
            messages=compacted,
            action="compact",
            original_tokens=original_tokens,
            trimmed_tokens=compacted_tokens,
        )

    raise ValueError(f"Unknown strategy: {strategy}")
