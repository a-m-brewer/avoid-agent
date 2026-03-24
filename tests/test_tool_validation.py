"""Tests for tool call argument validation."""

from avoid_agent.agent.runtime import _validate_tool_call
from avoid_agent.agent.tools import ToolDefinition, ParamDefinition, ParameterType
from avoid_agent.providers import ProviderToolCall


def _tool_call(name: str, arguments: dict) -> ProviderToolCall:
    return ProviderToolCall(id="tc_1", name=name, arguments=arguments)


TOOLS = [
    ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters=[
            ParamDefinition(name="path", type=ParameterType.STR, description="File path", required=True),
        ],
    ),
    ToolDefinition(
        name="edit_file",
        description="Edit a file",
        parameters=[
            ParamDefinition(name="path", type=ParameterType.STR, description="File path", required=True),
            ParamDefinition(name="old_string", type=ParameterType.STR, description="Text to replace", required=True),
            ParamDefinition(name="new_string", type=ParameterType.STR, description="Replacement", required=True),
        ],
    ),
    ToolDefinition(
        name="run_bash",
        description="Run a command",
        parameters=[
            ParamDefinition(name="command", type=ParameterType.STR, description="Command", required=True),
        ],
    ),
]


def test_valid_tool_call():
    result = _validate_tool_call(_tool_call("read_file", {"path": "/tmp/x"}), TOOLS)
    assert result is None


def test_unknown_tool():
    result = _validate_tool_call(_tool_call("delete_file", {"path": "/tmp/x"}), TOOLS)
    assert result is not None
    assert "Unknown tool" in result


def test_missing_required_arg():
    result = _validate_tool_call(_tool_call("edit_file", {"path": "/tmp/x"}), TOOLS)
    assert result is not None
    assert "Missing required" in result
    assert "new_string" in result
    assert "old_string" in result


def test_all_required_args_present():
    result = _validate_tool_call(
        _tool_call("edit_file", {"path": "/tmp/x", "old_string": "a", "new_string": "b"}),
        TOOLS,
    )
    assert result is None
