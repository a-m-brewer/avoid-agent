"""Tests for system prompt assembly."""

from avoid_agent.prompts.system_prompt import build_system_prompt


def test_includes_runtime_context_when_provided():
    prompt = build_system_prompt(
        working_directory="/tmp/project",
        git_status="M avoid_agent/__main__.py",
        top_level_file_structure=".\n./avoid_agent\n./tests",
    )

    assert "## Runtime Context" in prompt
    assert "Working directory:\n/tmp/project" in prompt
    assert "Git status:\nM avoid_agent/__main__.py" in prompt
    assert "Top-level file structure:\n.\n./avoid_agent\n./tests" in prompt


def test_omits_runtime_context_when_not_provided():
    prompt = build_system_prompt()
    assert "## Runtime Context" not in prompt


def test_runtime_context_is_truncated_when_too_large():
    prompt = build_system_prompt(
        git_status="x" * 5000,
        top_level_file_structure="y" * 9000,
    )

    assert "... [truncated]" in prompt


def test_section_order_places_runtime_before_response_style():
    prompt = build_system_prompt(working_directory="/tmp/project")

    runtime_index = prompt.index("## Runtime Context")
    style_index = prompt.index("## Response Style")

    assert runtime_index < style_index
