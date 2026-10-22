# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-executed ``tool_search`` for the Responses API.

The model sees a client ``tool_search`` tool as a plain function. Its calls are
returned as ``tool_search_call`` items for the client to execute, and tools
loaded by ``tool_search_output`` input items are exposed on later turns.
"""

import json
from collections.abc import Iterable
from typing import Any, Literal

from openai.types.responses import (
    FunctionTool,
    NamespaceTool,
    ResponseToolSearchCall,
    ToolSearchTool,
)
from openai.types.responses.tool import Tool
from pydantic import TypeAdapter

from vllm.utils import random_uuid

TOOL_SEARCH_NAME = "tool_search"

_TOOL_ADAPTER: TypeAdapter[Tool] = TypeAdapter(Tool)


def has_client_tool_search(tools: Iterable[Tool]) -> bool:
    """Whether ``tools`` holds a client ``tool_search`` the model can call."""
    tools = list(tools)
    if any(t.type == "function" and t.name == TOOL_SEARCH_NAME for t in tools):
        return False
    return any(isinstance(t, ToolSearchTool) and t.execution == "client" for t in tools)


def _item_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return None


def _loaded_tools(request_input: str | list[Any]) -> list[Tool]:
    if isinstance(request_input, str):
        return []
    loaded: list[Tool] = []
    for item in request_input:
        data = _item_dict(item)
        if data is None or data.get("type") != "tool_search_output":
            continue
        loaded.extend(_TOOL_ADAPTER.validate_python(t) for t in data["tools"])
    return loaded


def _merge(tools: list[Tool]) -> list[Tool]:
    """Drop repeated functions; merge namespaces loaded by separate searches."""
    merged: dict[tuple[str, str] | int, Tool] = {}
    for idx, tool in enumerate(tools):
        name = getattr(tool, "name", None)
        if name is None or tool.type not in ("function", "namespace"):
            merged[idx] = tool
            continue
        key = (tool.type, name)
        prev = merged.get(key)
        if prev is None:
            merged[key] = tool
        elif isinstance(prev, NamespaceTool) and isinstance(tool, NamespaceTool):
            known = {t.name for t in prev.tools}
            merged[key] = prev.model_copy(
                update={
                    "tools": [
                        *prev.tools,
                        *(t for t in tool.tools if t.name not in known),
                    ]
                }
            )
    return list(merged.values())


def build_model_tools(tools: list[Tool], request_input: str | list[Any]) -> list[Tool]:
    """Tools as rendered for the model and used to parse its tool calls."""
    if not has_client_tool_search(tools):
        return tools
    model_tools: list[Tool] = []
    for tool in tools:
        if isinstance(tool, ToolSearchTool) and tool.execution == "client":
            model_tools.append(
                FunctionTool(
                    type="function",
                    name=TOOL_SEARCH_NAME,
                    description=tool.description,
                    parameters=tool.parameters or {"type": "object", "properties": {}},
                    strict=False,
                )
            )
        else:
            model_tools.append(tool)
    return _merge(model_tools + _loaded_tools(request_input))


def make_tool_search_call(
    call_id: str,
    arguments: str,
    status: Literal["in_progress", "completed", "incomplete"] = "completed",
    item_id: str | None = None,
) -> ResponseToolSearchCall:
    try:
        parsed: object = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        parsed = arguments
    return ResponseToolSearchCall(
        id=item_id or f"tsc_{random_uuid()}",
        call_id=call_id,
        type="tool_search_call",
        execution="client",
        status=status,
        arguments=parsed,
    )


def tool_search_call_arguments(data: dict[str, Any]) -> str:
    arguments = data.get("arguments")
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {})


def tool_search_output_content(data: dict[str, Any]) -> str:
    """Tool-message text listing what a search loaded, by callable name."""
    # vllm.tool_parsers imports the Responses protocol, which imports this module.
    from vllm.tool_parsers.utils import flat_namespace_tool_name

    loaded: list[dict[str, Any]] = []
    for raw in data.get("tools", []):
        tool = _TOOL_ADAPTER.validate_python(raw)
        if isinstance(tool, NamespaceTool):
            loaded.extend(
                {
                    "name": flat_namespace_tool_name(tool.name, t.name),
                    "description": getattr(t, "description", None),
                }
                for t in tool.tools
            )
        elif isinstance(tool, FunctionTool):
            loaded.append({"name": tool.name, "description": tool.description})
    return json.dumps({"loaded_tools": loaded})
