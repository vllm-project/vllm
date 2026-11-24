# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

from openai.types.responses import (
    FunctionTool,
    NamespaceTool,
    ToolChoiceCustom,
    ToolChoiceFunction,
)
from openai.types.responses.tool import Tool
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    _resolve_named_tool_choice,
    _tool_field,
    _validate_custom_tool_formats,
)
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.utils import flat_namespace_tool_name, get_function_tools

_RESPONSE_TOOLS = TypeAdapter(list[Tool])


def _merge_search_tools(initial: list[Tool], additional: list[Tool]) -> list[Tool]:
    """Keep declared and earlier definitions while adding new namespace members."""
    tools = list(initial)
    indices = {
        (tool.type, tool.name): index
        for index, tool in enumerate(tools)
        if tool.type in ("function", "custom", "namespace")
    }
    for tool in additional:
        if tool.type not in ("function", "custom", "namespace"):
            continue
        key = (tool.type, tool.name)
        index = indices.get(key)
        if index is None:
            indices[key] = len(tools)
            tools.append(tool)
        elif isinstance(tool, NamespaceTool):
            existing = tools[index]
            assert isinstance(existing, NamespaceTool)
            members = list(existing.tools)
            seen = {(member.type, member.name) for member in members}
            for member in tool.tools:
                member_key = (member.type, member.name)
                if member_key not in seen:
                    members.append(member)
                    seen.add(member_key)
            tools[index] = existing.model_copy(update={"tools": members})
    return tools


def _check_tool_names(tools: list[Tool]) -> dict[str, str]:
    namespaces: set[str] = set()
    names: dict[str, str] = {}
    for tool in tools:
        if tool.type == "namespace":
            if tool.name in namespaces:
                raise VLLMValidationError(
                    f"Duplicate tool namespace {tool.name!r}.", parameter="tools"
                )
            namespaces.add(tool.name)
            candidates = [
                (flat_namespace_tool_name(tool.name, member.name), member.type)
                for member in tool.tools
            ]
        elif tool.type in ("function", "custom"):
            candidates = [(tool.name, tool.type)]
        else:
            continue
        for name, kind in candidates:
            if name in names:
                raise VLLMValidationError(
                    f"Duplicate or colliding tool name {name!r}.", parameter="tools"
                )
            names[name] = kind
    return names


def resolve_search_tool_choice(request: ResponsesRequest) -> ResponsesRequest:
    """Resolve a tool choice deferred until the response history was loaded."""
    choice = request.tool_choice
    if isinstance(choice, (ToolChoiceFunction, ToolChoiceCustom)):
        name = _resolve_named_tool_choice(choice.name, request.tools)
        if name != choice.name:
            return request.model_copy(
                update={"tool_choice": choice.model_copy(update={"name": name})}
            )
    elif not request.tools:
        if choice == "auto":
            return request.model_copy(update={"tool_choice": "none"})
        if choice == "required":
            raise VLLMValidationError(
                "Tool choice 'required' must be specified with 'tools' parameter.",
                parameter="tool_choice",
            )
    return request


def replay_minicpm5_tool_search(
    request: ResponsesRequest, previous_tools: list[Tool]
) -> ResponsesRequest:
    """Use the tools from search results and omit search records from the prompt."""
    input_items = request.input
    try:
        loaded: list[object] = []
        if isinstance(input_items, list):
            filtered = []
            for item in input_items:
                item_type = _tool_field(item, "type")
                if item_type == "tool_search_output":
                    item_tools = _tool_field(item, "tools")
                    if isinstance(item_tools, Iterable) and not isinstance(
                        item_tools, (dict, str)
                    ):
                        loaded.extend(
                            tool
                            for tool in item_tools
                            if _tool_field(tool, "type")
                            in ("function", "custom", "namespace")
                        )
                elif item_type != "tool_search_call":
                    filtered.append(item)
            if len(filtered) != len(input_items):
                input_items = filtered
        discovered = _RESPONSE_TOOLS.validate_python(loaded)
    except ValidationError as exc:
        raise VLLMValidationError(
            "Invalid tool definitions in tool_search_output.", parameter="input"
        ) from exc

    if input_items is request.input and not previous_tools:
        return resolve_search_tool_choice(request)

    replayed = _merge_search_tools(previous_tools, discovered)
    tools = _merge_search_tools(request.tools, replayed)
    names = _check_tool_names(tools)
    _validate_custom_tool_formats(tools)
    declared_tools = request.tools
    request = request.model_copy(update={"input": input_items, "tools": tools})
    request._declared_tools = declared_tools
    request._replayed_search_tools = replayed
    request = resolve_search_tool_choice(request)
    choice = request.tool_choice
    if isinstance(choice, (ToolChoiceFunction, ToolChoiceCustom)) and (
        names.get(choice.name) != choice.type
        or not any(
            isinstance(tool, FunctionTool) and tool.name == choice.name
            for tool in get_function_tools(tools)
        )
    ):
        raise VLLMValidationError(
            f"Tool choice {choice.name!r} is not a supported {choice.type} tool.",
            parameter="tool_choice",
        )
    return request
