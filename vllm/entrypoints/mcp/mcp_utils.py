# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MCP tool utilities for backward compatibility and tool normalization."""

import json
from typing import TYPE_CHECKING

from openai.types.responses.tool import (
    CodeInterpreter,
    Mcp,
    Tool,
    WebSearchPreviewTool,
)

if TYPE_CHECKING:
    from mcp import ClientSession


def normalize_tool_to_mcp(tool: Tool) -> Tool:
    """
    Convert legacy tool types to MCP format for unified handling.

    This provides backward compatibility by converting legacy tool types
    (CodeInterpreter, WebSearchPreviewTool) to the unified MCP format.
    All downstream code can then handle tools uniformly via the MCP protocol.

    Args:
        tool: Any Tool type from OpenAI protocol

    Returns:
        - If already MCP: returns as-is
        - If CodeInterpreter: converts to MCP with server_label="code_interpreter"
          Note: container field is discarded (not needed for MCP protocol)
        - If WebSearchPreviewTool: converts to MCP with server_label="browser"
          Note: search_context_size and user_location fields are discarded
        - Otherwise: returns as-is (function tools, etc. pass through unchanged)
    """
    # Already MCP - return as-is
    if isinstance(tool, Mcp):
        return tool

    # CodeInterpreter → MCP with server_label="code_interpreter"
    # Note: Discarding container field as it's not needed for MCP protocol
    if isinstance(tool, CodeInterpreter):
        return Mcp(
            type="mcp",
            server_label="code_interpreter",
        )

    # WebSearchPreviewTool → MCP with server_label="browser"
    # Note: Discarding search_context_size and user_location fields
    # These could be passed as headers in the future if needed
    if isinstance(tool, WebSearchPreviewTool):
        return Mcp(
            type="mcp",
            server_label="browser",
        )

    # All other tool types (FunctionTool, FileSearchTool, etc.) pass through unchanged
    return tool


async def call_mcp_tool(
    tool_session: "ClientSession",
    tool_name: str,
    tool_args_str: str,
) -> str:
    """Generic MCP tool call handler

    Args:
        tool_session: The MCP client session or Tool instance
        tool_name: The tool name to call
        tool_args: The args for the tool call

    Returns:
        A string representation of the MCP call output
    """
    # TODO: Env variable for returning json parsing error to model
    # instead of erroring the request
    tool_args = json.loads(tool_args_str)
    result = await tool_session.call_tool(tool_name, tool_args)
    # TODO: Support handling structured MCP call output
    result_str = result.content[0].text
    return result_str
