"""Inspector tool to generate JSON schema for agent tools based on their function signatures and type hints."""

import inspect
import typing
import types

from avoid_agent.agent.tools import TYPE_MAP, ParamDefinition, ParameterType, ToolDefinition


class InspectorException(Exception):
    """Custom exception for inspector errors."""



class MissingDescriptionException(InspectorException):
    """Raised when a tool function is missing a docstring description."""


class MissingTypeAnnotationException(InspectorException):
    """Raised when a parameter is missing a type annotation."""


def generate_tool_schema(tool_func) -> ToolDefinition:
    hints = typing.get_type_hints(tool_func, include_extras=True)
    sig = inspect.signature(tool_func)
    func_name = tool_func.__name__
    func_description = tool_func.__doc__

    if not func_description:
        raise MissingDescriptionException(
            f"Tool function '{func_name}' is missing a docstring description.")

    parameters: list[ParamDefinition] = []
    for name, hint in hints.items():
        if name == 'return':
            continue

        actual_type = None
        description = None

        if typing.get_origin(hint) is typing.Annotated:
            actual_type, description = typing.get_args(hint)
        else:
            origin = typing.get_origin(hint)
            if origin in (typing.Union, types.UnionType):
                non_none_args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
                if len(non_none_args) == 1 and typing.get_origin(non_none_args[0]) is typing.Annotated:
                    actual_type, description = typing.get_args(non_none_args[0])

        if actual_type is None or description is None:
            raise MissingTypeAnnotationException(
                f"Parameter '{name}' is missing an Annotated type hint with a description.")

        parameters.append(ParamDefinition(
            name=name,
            type=TYPE_MAP.get(actual_type, ParameterType.STR),
            description=description,
            required=sig.parameters[name].default is inspect.Parameter.empty
        ))

    return ToolDefinition(
        name=func_name,
        description=func_description,
        parameters=parameters
    )


if __name__ == "__main__":
    from avoid_agent.agent.tools.core import read_file
    definition = generate_tool_schema(read_file)
    print(definition)
