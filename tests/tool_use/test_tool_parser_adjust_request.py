# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ResponseFormat
from vllm.tool_parsers.abstract_tool_parser import ToolParser

MODEL = "facebook/opt-125m"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
}


def _json_object_format() -> ResponseFormat:
    return ResponseFormat(type="json_object")


def _parser() -> ToolParser:
    return ToolParser(MagicMock())


@pytest.mark.parametrize("tool_choice", ["auto", None])
def test_adjust_request_clears_response_format_for_auto(tool_choice):
    kwargs = {
        "model": MODEL,
        "messages": [],
        "tools": [WEATHER_TOOL],
        "response_format": _json_object_format(),
    }
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    request = ChatCompletionRequest(**kwargs)  # type: ignore[arg-type]

    adjusted = _parser().adjust_request(request)

    assert adjusted.response_format is None


def test_adjust_request_preserves_response_format_for_none():
    response_format = _json_object_format()
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        response_format=response_format,
    )  # type: ignore[arg-type]

    adjusted = _parser().adjust_request(request)

    assert adjusted.response_format is response_format


def test_adjust_request_preserves_response_format_without_tools():
    response_format = _json_object_format()
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[],
        response_format=response_format,
    )  # type: ignore[arg-type]

    adjusted = _parser().adjust_request(request)

    assert adjusted.response_format is response_format
