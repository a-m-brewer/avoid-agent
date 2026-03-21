"""Tests for the inspector module."""

# pylint: disable=unused-argument,missing-class-docstring

from typing import Annotated

import pytest

from avoid_agent.agent.tools import ParamDefinition, ParameterType
from avoid_agent.agent.tools.inspector import (
    MissingDescriptionException,
    MissingTypeAnnotationException,
    generate_tool_schema,
)


# --- Test helper functions to inspect ---
# These stubs exist purely for their signatures; the bodies are intentionally empty.


def sample_tool(path: Annotated[str, "The file path"], count: Annotated[int, "How many times"]) -> str:
    """A sample tool for testing."""


def optional_param_tool(
    name: Annotated[str, "The name"],
    verbose: Annotated[bool, "Enable verbose mode"] = False,
) -> None:
    """Tool with an optional parameter."""


def no_docstring_tool(path: Annotated[str, "A path"]) -> str:
    pass  # intentionally no docstring


def missing_annotation_tool(path: str) -> str:
    """Tool with a plain type hint (no Annotated)."""


def float_tool(ratio: Annotated[float, "A ratio"]) -> str:
    """Tool with a float param."""


def bool_tool(verbose: Annotated[bool, "Enable verbose"]) -> str:
    """Tool with a bool param."""


def unknown_type_tool(items: Annotated[list, "A list of items"]) -> str:
    """Tool with a type not in TYPE_MAP."""


def no_param_tool() -> str:
    """Tool with no parameters."""


# --- Tests ---

class TestGenerateToolSchema:

    def test_name_and_description(self):
        result = generate_tool_schema(sample_tool)
        assert result.name == "sample_tool"
        assert result.description == "A sample tool for testing."

    def test_parameter_types(self):
        result = generate_tool_schema(sample_tool)
        assert len(result.parameters) == 2

        path_param = result.parameters[0]
        assert path_param == ParamDefinition(
            name="path",
            type=ParameterType.STR,
            description="The file path",
            required=True,
        )

        count_param = result.parameters[1]
        assert count_param == ParamDefinition(
            name="count",
            type=ParameterType.INT,
            description="How many times",
            required=True,
        )

    def test_optional_parameter(self):
        result = generate_tool_schema(optional_param_tool)
        verbose_param = result.parameters[1]
        assert verbose_param.required is False

    def test_required_parameter(self):
        result = generate_tool_schema(optional_param_tool)
        name_param = result.parameters[0]
        assert name_param.required is True

    def test_missing_docstring_raises(self):
        with pytest.raises(MissingDescriptionException):
            generate_tool_schema(no_docstring_tool)

    def test_missing_annotated_hint_raises(self):
        with pytest.raises(MissingTypeAnnotationException):
            generate_tool_schema(missing_annotation_tool)

    def test_return_hint_is_skipped(self):
        result = generate_tool_schema(sample_tool)
        param_names = [p.name for p in result.parameters]
        assert "return" not in param_names

    def test_float_type_mapping(self):
        result = generate_tool_schema(float_tool)
        assert result.parameters[0].type == ParameterType.FLOAT

    def test_bool_type_mapping(self):
        result = generate_tool_schema(bool_tool)
        assert result.parameters[0].type == ParameterType.BOOL

    def test_unknown_type_defaults_to_str(self):
        result = generate_tool_schema(unknown_type_tool)
        assert result.parameters[0].type == ParameterType.STR

    def test_no_parameters(self):
        result = generate_tool_schema(no_param_tool)
        assert not result.parameters
