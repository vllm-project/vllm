# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for response_format handling in ToolParser.adjust_request.

With tool_choice="auto" no schema is derived from the tools, so a
user-supplied response_format used to stay on the request and constrain
decoding to plain JSON, which prevented the model from ever emitting
tool-call tokens (https://github.com/vllm-project/vllm/issues/39929).
"""

from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "strict": True,
}


def _build_request(**overrides: Any) -> ChatCompletionRequest:
    data: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the weather in Dallas?"}],
        "tools": [WEATHER_TOOL],
        "response_format": {"type": "json_object"},
        **overrides,
    }
    return ChatCompletionRequest.model_validate(data)


def _build_responses_request(**overrides: Any) -> ResponsesRequest:
    data: dict[str, Any] = {
        "model": "test-model",
        "input": [{"role": "user", "content": "What is the weather in Dallas?"}],
        "tools": [WEATHER_TOOL_RESPONSES],
        "text": {"format": {"type": "json_object"}},
        **overrides,
    }
    return ResponsesRequest.model_validate(data)


def _adjust(
    request: ChatCompletionRequest | ResponsesRequest,
) -> ChatCompletionRequest | ResponsesRequest:
    parser = ToolParser.__new__(ToolParser)
    return ToolParser.adjust_request(parser, request)


def test_auto_clears_response_format() -> None:
    """auto: response_format must be cleared so tool-call tokens stay
    reachable, and no structured output constraint is added."""
    request = _build_request(tool_choice="auto")

    _adjust(request)

    assert request.response_format is None
    assert request.structured_outputs is None


def test_unset_tool_choice_clears_response_format() -> None:
    """Unset tool_choice with tools present defaults to auto during request
    validation and must be treated the same."""
    request = _build_request()
    assert request.tool_choice == "auto", (
        "Precondition: unset tool_choice defaults to auto when tools are set"
    )

    _adjust(request)

    assert request.response_format is None


def test_null_tool_choice_clears_response_format() -> None:
    """An explicit null tool_choice passes validation untouched and must be
    treated like auto."""
    request = _build_request(tool_choice=None)
    assert request.tool_choice is None

    _adjust(request)

    assert request.response_format is None


def test_none_tool_choice_preserves_response_format() -> None:
    """none: the caller asked for a formatted reply with tool calling
    disabled, so response_format must survive."""
    request = _build_request(tool_choice="none")

    _adjust(request)

    assert request.response_format is not None
    assert request.response_format.type == "json_object"


def test_required_overrides_response_format_with_tool_schema() -> None:
    """required: the schema derived from the tools replaces response_format,
    same as before this fix (#32006)."""
    request = _build_request(tool_choice="required")

    _adjust(request)

    assert request.response_format is None
    assert request.structured_outputs is not None
    assert request.structured_outputs.json is not None


def test_no_tools_preserves_response_format() -> None:
    """Without tools adjust_request returns early and response_format is
    untouched."""
    request = _build_request(tools=None, tool_choice=None)

    _adjust(request)

    assert request.response_format is not None
    assert request.response_format.type == "json_object"


def test_responses_auto_clears_text_format() -> None:
    """Responses API: text.format is the response_format analog and must be
    dropped for auto so tool-call tokens stay reachable."""
    request = _build_responses_request()
    assert request.tool_choice == "auto", (
        "Precondition: ResponsesRequest.tool_choice defaults to auto"
    )
    assert request.text is not None and request.text.format is not None

    _adjust(request)

    assert request.text is None or request.text.format is None


def test_responses_none_tool_choice_preserves_text_format() -> None:
    """Responses API: tool_choice none keeps the caller's text format."""
    request = _build_responses_request(tool_choice="none")

    _adjust(request)

    assert request.text is not None
    assert request.text.format is not None
