# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tool-parser utility functions (find_tool_parameters, etc.)."""

from __future__ import annotations

import pytest
from openai.types.responses import FunctionTool, WebSearchTool

from vllm.entrypoints.generate.base.protocol import FunctionDefinition
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.utils import (
    find_tool_parameters,
    find_tool_properties,
    get_json_schema_from_tools,
)

pytestmark = pytest.mark.cpu_test

FUNCTION_TOOL = FunctionTool(
    type="function",
    name="get_weather",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
WEB_SEARCH_TOOL = WebSearchTool(type="web_search")


class TestNonFunctionToolsSkipped:
    """Non-function tools (web_search, etc.) must be silently skipped
    by the tool-schema utilities instead of raising TypeError."""

    def test_find_tool_properties_skips_web_search(self):
        tools = [WEB_SEARCH_TOOL, FUNCTION_TOOL]
        props = find_tool_properties(tools, "get_weather")
        assert props == {"city": {"type": "string"}}

    def test_find_tool_properties_only_non_function_tools(self):
        props = find_tool_properties([WEB_SEARCH_TOOL], "get_weather")
        assert props == {}

    def test_get_json_schema_with_mixed_tools(self):
        tools = [WEB_SEARCH_TOOL, FUNCTION_TOOL]
        schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
        assert isinstance(schema, dict)
        any_of = schema["items"]["anyOf"]
        assert len(any_of) == 1
        assert any_of[0]["properties"]["name"]["enum"] == ["get_weather"]

    def test_find_tool_parameters_skips_web_search(self):
        tools = [WEB_SEARCH_TOOL, FUNCTION_TOOL]
        params = find_tool_parameters(tools, "get_weather")
        assert params == FUNCTION_TOOL.parameters

    def test_find_tool_parameters_only_non_function_tools(self):
        params = find_tool_parameters([WEB_SEARCH_TOOL], "get_weather")
        assert params is None


class TestFindToolParameters:
    """find_tool_parameters returns the full parameters object, or None."""

    def test_returns_full_parameters_for_function_tool(self):
        params = find_tool_parameters([FUNCTION_TOOL], "get_weather")
        assert params == FUNCTION_TOOL.parameters
        assert "required" in params
        assert "properties" in params

    def test_returns_full_parameters_for_chat_tool(self):
        chat_tool = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )
        params = find_tool_parameters([chat_tool], "get_weather")
        assert params == chat_tool.function.parameters

    def test_returns_none_when_tools_empty_or_missing(self):
        assert find_tool_parameters(None, "get_weather") is None
        assert find_tool_parameters([], "get_weather") is None

    def test_returns_none_when_name_absent(self):
        assert find_tool_parameters([FUNCTION_TOOL], "unknown") is None

    def test_returns_none_when_parameters_is_none(self):
        chat_tool = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(name="no_params", parameters=None),
        )
        assert find_tool_parameters([chat_tool], "no_params") is None
