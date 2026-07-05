# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: _get_tool_schema_defs must not mutate the caller's tool.

Before the fix, `_get_tool_schema_defs` called `params.pop("$defs", {})`,
which removed `$defs` from the underlying parameters dict on the first call.
A second call with the same tool would then see no `$defs`, causing $ref
cross-references to silently fail to resolve.

This test asserts:
  (a) The original tool's parameters still contain `$defs` after the first
      call to `_get_json_schema_from_tools`.
  (b) A second call returns a schema whose `$defs` block is identical to
      the first call's result.
"""

import pytest
from pydantic import TypeAdapter

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.utils import _get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

# A tool whose parameters use a $defs block with a $ref cross-reference.
_TOOL_WITH_DEFS = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"$ref": "#/$defs/Location"},
            },
            "required": ["location"],
            "$defs": {
                "Location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
        },
    },
}


@pytest.fixture
def tool_with_defs() -> ChatCompletionToolsParam:
    return TypeAdapter(ChatCompletionToolsParam).validate_python(_TOOL_WITH_DEFS)


def test_defs_not_mutated_after_first_call(tool_with_defs):
    """After the first call, $defs must still be present on the tool."""
    tools = [tool_with_defs]
    _get_json_schema_from_tools(tools)

    params = tool_with_defs.function.parameters
    assert params is not None, "parameters should not be None"
    assert "$defs" in params, (
        "$defs was removed from the tool's parameters dict — "
        "_get_tool_schema_defs is mutating the caller's tool"
    )


def test_second_call_returns_same_defs(tool_with_defs):
    """Both calls must return a schema with identical $defs."""
    tools = [tool_with_defs]
    schema_first = _get_json_schema_from_tools(tools)
    schema_second = _get_json_schema_from_tools(tools)

    assert "$defs" in schema_first, "first call: $defs missing from returned schema"
    assert "$defs" in schema_second, (
        "second call: $defs missing — _get_tool_schema_defs mutated the tool "
        "on the first call, so $defs was empty on the second call"
    )
    assert schema_first["$defs"] == schema_second["$defs"], (
        "$defs differ between first and second call — mutation occurred"
    )


def test_defs_not_duplicated_in_embedded_params(tool_with_defs):
    """$defs must live only at the top level, not inside each tool's params."""
    schema = _get_json_schema_from_tools([tool_with_defs])

    assert "$defs" in schema, "top-level $defs missing from returned schema"
    for variant in schema["items"]["anyOf"]:
        params = variant["properties"]["parameters"]
        assert "$defs" not in params, (
            "$defs duplicated inside the embedded tool parameters — it should "
            "only appear once at the top level of the schema"
        )
