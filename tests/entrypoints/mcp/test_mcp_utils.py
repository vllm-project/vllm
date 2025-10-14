# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MCP utils."""

from openai.types.responses.tool import (
    CodeInterpreter,
    CodeInterpreterContainerCodeInterpreterToolAuto,
    FunctionTool,
    Mcp,
    WebSearchPreviewTool,
)

from vllm.entrypoints.mcp.mcp_utils import normalize_tool_to_mcp


def test_normalize_mcp_tool_passthrough():
    """MCP tools should pass through unchanged."""
    mcp_tool = Mcp(
        type="mcp", server_label="weather", server_url="http://localhost:8765/sse"
    )
    result = normalize_tool_to_mcp(mcp_tool)
    assert result == mcp_tool
    assert result.server_label == "weather"
    assert result.server_url == "http://localhost:8765/sse"


def test_normalize_code_interpreter():
    """CodeInterpreter should convert to MCP with server_label='code_interpreter'."""
    # For test purposes we provide a minimal container (required by Pydantic)
    # Just testing that type is correctly converted
    code_tool = CodeInterpreter(
        type="code_interpreter",
        container=CodeInterpreterContainerCodeInterpreterToolAuto(type="auto"),
    )
    result = normalize_tool_to_mcp(code_tool)

    assert isinstance(result, Mcp)
    assert result.type == "mcp"
    assert result.server_label == "code_interpreter"
    # Container field is intentionally discarded


def test_normalize_web_search_preview():
    """WebSearchPreviewTool should convert to MCP with server_label='browser'."""
    search_tool = WebSearchPreviewTool(
        type="web_search_preview",
        search_context_size="medium",
    )
    result = normalize_tool_to_mcp(search_tool)

    assert isinstance(result, Mcp)
    assert result.type == "mcp"
    assert result.server_label == "browser"
    # search_context_size is intentionally discarded


def test_normalize_other_tools_passthrough():
    """Other tool types should pass through unchanged."""
    # Using a FunctionTool as an example of a non-converted tool type
    function_tool = FunctionTool(
        type="function",
        name="test_func",
        function={"name": "test_func", "description": "Test function"},
    )

    result = normalize_tool_to_mcp(function_tool)

    # Should be unchanged
    assert result == function_tool
    assert result.type == "function"
