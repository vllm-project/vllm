# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``response_format`` handling in ``ToolParser.adjust_request`` (#39929).

A tools request carries one structured-outputs slot. When the tool choice
derives no schema of its own, a ``response_format`` would take that slot and
mask out every tool-call token, so it has to be dropped -- but only where tool
calling is actually possible.
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser

pytestmark = pytest.mark.cpu_test

JSON_OBJECT = {"type": "json_object"}
JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "name": "answer",
    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
}


@pytest.fixture
def weather_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def adjusted(**kwargs) -> ChatCompletionRequest:
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "What is the weather in Dallas?"}],
        **kwargs,
    )
    return ToolParser(MagicMock()).adjust_request(request)


def test_auto_tool_choice_drops_response_format(weather_tool):
    request = adjusted(
        tools=[weather_tool], tool_choice="auto", response_format=JSON_OBJECT
    )

    assert request.response_format is None


def test_omitted_tool_choice_drops_response_format(weather_tool):
    # Omitting tool_choice alongside tools defaults it to "auto", so the same
    # suppression applies to requests that never mention tool_choice at all.
    request = adjusted(tools=[weather_tool], response_format=JSON_OBJECT)

    assert request.tool_choice == "auto"
    assert request.response_format is None


def test_tool_choice_none_keeps_response_format(weather_tool):
    # The caller opted out of tool calls, so there is nothing to make room for.
    request = adjusted(
        tools=[weather_tool], tool_choice="none", response_format=JSON_OBJECT
    )

    assert request.response_format is not None


def test_response_format_without_tools_is_untouched():
    request = adjusted(response_format=JSON_OBJECT)

    assert request.response_format is not None


@pytest.fixture
def responses_weather_tool() -> dict:
    return {
        "type": "function",
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def adjusted_responses(**kwargs) -> ResponsesRequest:
    request = ResponsesRequest(input="What is the weather in Dallas?", **kwargs)
    return ToolParser(MagicMock()).adjust_request(request)


def test_auto_tool_choice_drops_responses_text_format(responses_weather_tool):
    # The Responses API carries the same constraint under text.format.
    request = adjusted_responses(
        tools=[responses_weather_tool],
        tool_choice="auto",
        text={"format": JSON_SCHEMA_FORMAT, "verbosity": "low"},
    )

    assert request.text.format is None
    # Only the format is dropped; verbosity is unrelated to structured outputs.
    assert request.text.verbosity == "low"


def test_tool_choice_none_keeps_responses_text_format(responses_weather_tool):
    request = adjusted_responses(
        tools=[responses_weather_tool],
        tool_choice="none",
        text={"format": JSON_SCHEMA_FORMAT},
    )

    assert request.text.format is not None


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
    ids=["required", "named"],
)
def test_forced_tool_choice_still_installs_tool_schema(weather_tool, tool_choice):
    # Unchanged path: a forced choice derives its own schema and drops
    # response_format on the way.
    request = adjusted(
        tools=[weather_tool], tool_choice=tool_choice, response_format=JSON_OBJECT
    )

    assert request.response_format is None
    assert request.structured_outputs is not None
    assert request.structured_outputs.json is not None
