"""Tests for hallucination detection in the agent loop."""

from avoid_agent.__main__ import _looks_like_hallucinated_completion
from avoid_agent.providers import AssistantMessage, ProviderToolCall


def _msg(text, tool_calls=None):
    return AssistantMessage(text=text, tool_calls=tool_calls or [])


def _msg_with_tool(text):
    return AssistantMessage(
        text=text,
        tool_calls=[
            ProviderToolCall(id="tc_1", name="edit_file", arguments={})
        ],
    )


# --- Should detect hallucinated completions ---

def test_detects_implemented_claim():
    assert _looks_like_hallucinated_completion(
        _msg("Implemented the changes in __main__.py.")
    )


def test_detects_applied_claim():
    assert _looks_like_hallucinated_completion(
        _msg("Applied the patch to all three providers.")
    )


def test_detects_edited_claim():
    assert _looks_like_hallucinated_completion(
        _msg("I edited the file to fix the bug.")
    )


def test_detects_changed_claim():
    assert _looks_like_hallucinated_completion(
        _msg("I changed the streaming logic in the TUI.")
    )


def test_detects_modified_claim():
    assert _looks_like_hallucinated_completion(
        _msg("Modified the provider to support events.")
    )


def test_detects_updated_claim():
    assert _looks_like_hallucinated_completion(
        _msg("Updated the conversation component.")
    )


def test_detects_wired_claim():
    assert _looks_like_hallucinated_completion(
        _msg("Wired the new event model into __main__.py.")
    )


def test_detects_checkmark_pattern():
    assert _looks_like_hallucinated_completion(
        _msg("✅ I changed the streaming to use events.")
    )


def test_detects_checkmark_what_changed():
    assert _looks_like_hallucinated_completion(
        _msg("✅ What changed\n\n- New event model")
    )


# --- Should NOT flag these ---

def test_ignores_plan_text():
    assert not _looks_like_hallucinated_completion(
        _msg("I'll implement the changes next.")
    )


def test_ignores_question():
    assert not _looks_like_hallucinated_completion(
        _msg("Do you want me to edit the file?")
    )


def test_ignores_none_text():
    assert not _looks_like_hallucinated_completion(
        _msg(None)
    )


def test_ignores_empty_text():
    assert not _looks_like_hallucinated_completion(
        _msg("")
    )


def test_ignores_when_tool_calls_present():
    assert not _looks_like_hallucinated_completion(
        _msg_with_tool("Implemented the changes in __main__.py.")
    )


def test_ignores_simple_acknowledgment():
    assert not _looks_like_hallucinated_completion(
        _msg("Understood. Ready.")
    )


def test_ignores_explanation():
    assert not _looks_like_hallucinated_completion(
        _msg("The function needs to be refactored for clarity.")
    )
