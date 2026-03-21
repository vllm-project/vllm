# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that tool_choice=required works correctly when the tools list
contains non-function tools (e.g. WebSearchTool) that lack a JSON schema.

Regression test: previously, passing a WebSearchTool alongside FunctionTools
with tool_choice="required" would crash _get_json_schema_from_tools because
WebSearchTool has no .name / .parameters attributes.
"""

import pytest
from openai.types.responses import FunctionTool

from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

WEATHER_TOOL = FunctionTool(
    type="function",
    name="get_weather",
    description="Get the weather",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
        },
        "required": ["city"],
    },
    strict=False,
)


class _FakeWebSearchTool:
    """Minimal stand-in for openai.types.responses.WebSearchTool.

    We avoid importing the real class because it may not exist in all
    openai SDK versions.  The key property is that it is NOT a FunctionTool
    and NOT a ChatCompletionToolsParam.
    """

    type = "web_search_preview"


def test_required_with_mixed_function_and_non_function_tools():
    """tool_choice=required with FunctionTool + non-function tool should
    return a valid schema containing only the FunctionTool."""
    tools = [WEATHER_TOOL, _FakeWebSearchTool()]
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")

    assert isinstance(schema, dict)
    # Schema should reference get_weather only
    any_of = schema["items"]["anyOf"]
    assert len(any_of) == 1
    assert any_of[0]["properties"]["name"]["enum"] == ["get_weather"]


def test_required_with_only_non_function_tools():
    """tool_choice=required with only non-function tools should return None
    rather than crashing."""
    tools = [_FakeWebSearchTool()]
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
    assert schema is None


def test_required_with_only_function_tools_unchanged():
    """tool_choice=required with only FunctionTools should work as before."""
    tools = [WEATHER_TOOL]
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")

    assert isinstance(schema, dict)
    any_of = schema["items"]["anyOf"]
    assert len(any_of) == 1
    assert any_of[0]["properties"]["name"]["enum"] == ["get_weather"]


def test_auto_ignores_non_function_tools():
    """tool_choice=auto should return None regardless of tool types."""
    tools = [WEATHER_TOOL, _FakeWebSearchTool()]
    schema = get_json_schema_from_tools(tools=tools, tool_choice="auto")
    assert schema is None


def test_none_returns_none():
    """tool_choice=none should return None."""
    tools = [WEATHER_TOOL]
    schema = get_json_schema_from_tools(tools=tools, tool_choice="none")
    assert schema is None
