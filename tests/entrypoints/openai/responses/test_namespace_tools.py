# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses API namespace-tool support at the request/serving boundary.

Codex-style clients send a ``type: "namespace"`` grouped tool (e.g.
``multi_agent_v1`` bundling ``spawn_agent`` / ``wait_agent``). vLLM must accept
it, flatten it into ``namespace__name`` function tools for the engine, and split
the name back on the way out.
"""

from openai.types.responses import NamespaceTool

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts

_NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Tools for spawning and managing sub-agents.",
    "tools": [
        {
            "type": "function",
            "name": "spawn_agent",
            "description": "Spawn a sub-agent.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        },
        {
            "type": "function",
            "name": "wait_agent",
            "description": "Wait for a sub-agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {"type": "array", "items": {"type": "string"}}
                },
            },
        },
    ],
}


def test_namespace_tool_accepted_in_request():
    """A namespace tool must validate (previously 400'd: unknown tool type)."""
    request = ResponsesRequest(model="m", input="hi", tools=[_NAMESPACE_TOOL])

    assert len(request.tools) == 1
    tool = request.tools[0]
    assert isinstance(tool, NamespaceTool)
    assert tool.type == "namespace"
    assert [child.name for child in tool.tools] == ["spawn_agent", "wait_agent"]


def test_construct_tool_dicts_flattens_namespace():
    """construct_tool_dicts flattens namespace children into `namespace__name`
    function tools the engine understands."""
    request = ResponsesRequest(model="m", input="hi", tools=[_NAMESPACE_TOOL])

    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)

    assert tool_dicts is not None
    names = [d["function"]["name"] for d in tool_dicts]
    assert names == [
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__wait_agent",
    ]
    assert all(d["type"] == "function" for d in tool_dicts)


def test_construct_tool_dicts_none_for_tool_choice_none():
    request = ResponsesRequest(model="m", input="hi", tools=[_NAMESPACE_TOOL])
    assert construct_tool_dicts(request.tools, "none") is None


def test_plain_function_tool_not_flattened():
    """Plain (non-namespace) function tools pass through unchanged."""
    plain = {
        "type": "function",
        "name": "shell_command",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    }
    request = ResponsesRequest(model="m", input="hi", tools=[plain])

    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
    assert [d["function"]["name"] for d in tool_dicts] == ["shell_command"]
