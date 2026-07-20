# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Responses API namespace-tool flattening helpers.

Namespace tools group several function tools under one namespace. The engine
only understands flat function tools, so these helpers flatten a NamespaceTool
into ``namespace__toolname`` function tools on the way in, and reconstruct the
(name, namespace) split on the way out.
"""

from openai.types.responses import FunctionTool, NamespaceTool

from vllm.tool_parsers.utils import (
    ResponsesToolCallName,
    build_responses_tool_call_name_map,
    flat_namespace_tool_name,
    iter_response_function_tool_dicts,
    iter_response_function_tool_info,
    resolve_responses_tool_call_name,
)


def _function_tool(name: str, properties: dict | None = None) -> FunctionTool:
    return FunctionTool(
        type="function",
        name=name,
        description=f"{name} description",
        strict=False,
        parameters={
            "type": "object",
            "properties": properties or {},
        },
    )


def _namespace_tool(namespace: str, *child_names: str) -> NamespaceTool:
    # NamespaceTool children are namespace-scoped ToolFunction dicts, not the
    # top-level FunctionTool; pass dicts and let pydantic coerce them.
    return NamespaceTool(
        type="namespace",
        name=namespace,
        description=f"{namespace} bundle",
        tools=[
            {
                "type": "function",
                "name": child,
                "description": f"{child} description",
                "parameters": {"type": "object", "properties": {}},
            }
            for child in child_names
        ],
    )


def test_flat_namespace_tool_name():
    assert flat_namespace_tool_name("ns", "tool") == "ns__tool"


def test_iter_response_function_tool_info_function_tool():
    ft = _function_tool("shell", {"cmd": {"type": "string"}})
    info = iter_response_function_tool_info(ft)
    assert info == [
        ("shell", {"type": "object", "properties": {"cmd": {"type": "string"}}})
    ]


def test_iter_response_function_tool_info_namespace_tool():
    ns = _namespace_tool("agents", "spawn", "wait")
    names = [name for name, _ in iter_response_function_tool_info(ns)]
    assert names == ["agents__spawn", "agents__wait"]


def test_iter_response_function_tool_dicts_flattens_children():
    ns = _namespace_tool("multi_agent_v1", "spawn_agent", "wait_agent")
    ft = _function_tool("plain_tool")
    dicts = iter_response_function_tool_dicts([ns, ft])
    names = [d["name"] for d in dicts]
    assert names == [
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__wait_agent",
        "plain_tool",
    ]


def test_build_and_resolve_name_map_roundtrip():
    ns = _namespace_tool("multi_agent_v1", "spawn_agent")
    name_map = build_responses_tool_call_name_map([ns])
    assert name_map == {
        "multi_agent_v1__spawn_agent": ResponsesToolCallName(
            name="spawn_agent", namespace="multi_agent_v1"
        )
    }
    resolved = resolve_responses_tool_call_name(
        "multi_agent_v1__spawn_agent", tool_call_name_map=name_map
    )
    assert resolved.name == "spawn_agent"
    assert resolved.namespace == "multi_agent_v1"


def test_resolve_passthrough_for_unknown_name():
    """A non-namespaced (or unknown) name resolves to itself with no namespace."""
    resolved = resolve_responses_tool_call_name("plain_tool", tool_call_name_map={})
    assert resolved.name == "plain_tool"
    assert resolved.namespace is None


def test_resolve_builds_map_from_tools_when_map_absent():
    ns = _namespace_tool("agents", "spawn")
    resolved = resolve_responses_tool_call_name("agents__spawn", tools=[ns])
    assert resolved == ResponsesToolCallName(name="spawn", namespace="agents")


def test_build_name_map_empty_or_none():
    assert build_responses_tool_call_name_map(None) == {}
    assert build_responses_tool_call_name_map([]) == {}
    # Plain function tools are not in the map (only namespace children are).
    assert build_responses_tool_call_name_map([_function_tool("plain")]) == {}
