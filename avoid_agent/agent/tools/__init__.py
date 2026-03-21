"""Defines tools that the agent can use."""

from dataclasses import dataclass
from enum import StrEnum


class ParameterType(StrEnum):
    """
    Represents parameter types as a string enumeration, mapping str, int,
    float, and bool to their corresponding string identifiers ('string',
    'integer', 'number', 'boolean').
    """
    STR = 'string'
    INT = 'integer'
    FLOAT = 'number'
    BOOL = 'boolean'


TYPE_MAP = {
    str: ParameterType.STR,
    int: ParameterType.INT,
    float: ParameterType.FLOAT,
    bool: ParameterType.BOOL,
}

@dataclass(slots=True)
class ParamDefinition:
    """Defines a parameter for a tool."""
    name: str
    type: ParameterType
    description: str
    required: bool


@dataclass(slots=True)
class ToolDefinition:
    """Defines a tool that the agent can use."""
    name: str
    description: str
    parameters: list[ParamDefinition]
