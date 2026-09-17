# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ``PoolsideV1ToolParser``.

Covers two bugs:

1. ``adjust_request`` did not skip the forced ``structured_outputs`` JSON
   for ``required``/named tool choice. These models emit XML tool calls
   (``<tool_call>...<arg_value>...</arg_value></tool_call>``) per the chat
   template, so guided JSON decoding conflicts with the format: the call
   leaks as content with empty ``tool_calls``. ``adjust_request`` now skips
   the constraint for both ChatCompletion (``ChatCompletionNamedToolChoice``)
   and Responses (``ToolChoiceFunction``) named choices.

2. ``extract_tool_calls`` stripped string-typed argument values, corrupting
   content whose whitespace is significant (e.g. code/file bodies losing
   leading indent and trailing newline). String values are now kept verbatim;
   only non-string types are stripped/deserialized.
"""

from __future__ import annotations

import json
from typing import Any

from openai.types.responses.tool_param import FunctionToolParam

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.poolside_v1_tool_parser import PoolsideV1ToolParser


def _write_file_tool() -> dict[str, Any]:
    """Tool with a string arg (``content``) and a non-string arg (``mode``)."""
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "mode": {"type": "integer"},
                },
                "required": ["content"],
            },
        },
    }


def _responses_write_file_tool() -> FunctionToolParam:
    return FunctionToolParam(
        type="function",
        name="write_file",
        description="Write content to a file",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "mode": {"type": "integer"},
            },
            "required": ["content"],
        },
        strict=True,
    )


def _build_chat_request(*, tool_choice: str | dict[str, Any]) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "poolside-test",
            "messages": [{"role": "user", "content": "write the file"}],
            "tools": [_write_file_tool()],
            "tool_choice": tool_choice,
        }
    )


def _build_responses_request(
    *, tool_choice: str | dict[str, Any], include: list[str] | None = None
) -> ResponsesRequest:
    return ResponsesRequest(
        model="poolside-test",
        input=[{"role": "user", "content": "write the file"}],
        tools=[_responses_write_file_tool()],
        tool_choice=tool_choice,
        stream=True,
        max_output_tokens=200,
        include=include,
    )


class _StubTokenizer:
    """Minimal tokenizer stub to satisfy ``PoolsideV1ToolParser.__init__``."""

    def get_vocab(self) -> dict[str, int]:
        return {"<tool_call>": 151_657, "</tool_call>": 151_658}


def _make_parser(request: ChatCompletionRequest) -> PoolsideV1ToolParser:
    return PoolsideV1ToolParser(_StubTokenizer(), tools=request.tools)


# ---------------------------------------------------------------------------
# Bug 1: required/named must skip forced structured_outputs (#39870 pattern)
# ---------------------------------------------------------------------------


def test_required_skips_structured_outputs_chatcompletion() -> None:
    request = _build_chat_request(tool_choice="required")
    _make_parser(request).adjust_request(request)

    assert request.structured_outputs is None
    assert request.skip_special_tokens is False


def test_named_skips_structured_outputs_chatcompletion() -> None:
    request = _build_chat_request(
        tool_choice={"type": "function", "function": {"name": "write_file"}}
    )
    _make_parser(request).adjust_request(request)

    assert request.structured_outputs is None
    assert request.skip_special_tokens is False


def test_required_skips_structured_outputs_responses() -> None:
    request = _build_responses_request(tool_choice="required")
    PoolsideV1ToolParser(_StubTokenizer()).adjust_request(request)

    assert request.text is None
    assert request.skip_special_tokens is False


def test_named_skips_structured_outputs_responses() -> None:
    # Responses-API named choice parses to ToolChoiceFunction, a different
    # type than the ChatCompletion named choice; both must be handled.
    request = _build_responses_request(
        tool_choice={"type": "function", "name": "write_file"}
    )
    PoolsideV1ToolParser(_StubTokenizer()).adjust_request(request)

    assert request.text is None
    assert request.skip_special_tokens is False


def test_auto_still_keeps_special_tokens() -> None:
    request = _build_chat_request(tool_choice="auto")
    _make_parser(request).adjust_request(request)

    assert request.skip_special_tokens is False


# ---------------------------------------------------------------------------
# Bug 2: string arg whitespace must be preserved (#42026 pattern)
# ---------------------------------------------------------------------------


def test_string_arg_preserves_whitespace() -> None:
    request = _build_chat_request(tool_choice="auto")
    parser = _make_parser(request)

    content = "    def f():\n        return 1\n"
    model_output = (
        "<tool_call>write_file\n"
        "<arg_key>content</arg_key>\n"
        f"<arg_value>{content}</arg_value>\n"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    # Leading indent and trailing newline must survive verbatim.
    assert args["content"] == content


def test_non_string_arg_still_deserialized() -> None:
    request = _build_chat_request(tool_choice="auto")
    parser = _make_parser(request)

    model_output = (
        "<tool_call>write_file\n"
        "<arg_key>content</arg_key>\n"
        "<arg_value>hi</arg_value>\n"
        "<arg_key>mode</arg_key>\n"
        "<arg_value> 420 </arg_value>\n"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["content"] == "hi"
    # Non-string value is stripped and parsed to its native type.
    assert args["mode"] == 420


def test_responses_extract_tool_calls_with_flat_tools() -> None:
    # required/named Responses calls route into extract_tool_calls with flat
    # FunctionTool (.name); _is_string_type must not raise.
    request = _build_responses_request(tool_choice="required")
    parser = PoolsideV1ToolParser(_StubTokenizer(), tools=request.tools)

    content = "  x = 1\n"
    model_output = (
        "<tool_call>write_file\n"
        "<arg_key>content</arg_key>\n"
        f"<arg_value>{content}</arg_value>\n"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["content"] == content


def _weather_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _build_weather_request(*, tool_choice: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "poolside-test",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [_weather_tool()],
            "tool_choice": tool_choice,
        }
    )


def test_no_newline_after_name_non_streaming() -> None:
    request = _build_weather_request(tool_choice="auto")
    parser = _make_parser(request)

    model_output = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"city": "Paris"}


def test_newline_after_name_still_parses_non_streaming() -> None:
    request = _build_weather_request(tool_choice="auto")
    parser = _make_parser(request)

    model_output = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert result.tool_calls[0].function.name == "get_weather"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"city": "Paris"}


def test_no_newline_after_name_streaming() -> None:
    request = _build_weather_request(tool_choice="auto")
    parser = _make_parser(request)

    model_output = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "</tool_call>"
    )

    name = ""
    args = ""
    prev = ""
    for ch in model_output:
        cur = prev + ch
        delta = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=cur,
            delta_text=ch,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        prev = cur
        if delta is None:
            continue
        for tc in delta.tool_calls or []:
            if tc.function is None:
                continue
            if tc.function.name:
                name += tc.function.name
            if tc.function.arguments:
                args += tc.function.arguments

    assert name == "get_weather"
    assert json.loads(args) == {"city": "Paris"}


def _stream_partial_start_token(request: ResponsesRequest):
    parser = _make_parser(request)
    delta = parser.tool_call_start_token[0]
    return parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta,
        delta_text=delta,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )


def test_streaming_responses_request_without_logprobs() -> None:
    request = _build_responses_request(tool_choice="auto")
    assert _stream_partial_start_token(request) is None


def test_streaming_responses_request_with_logprobs_emits_empty_delta() -> None:
    request = _build_responses_request(
        tool_choice="auto", include=["message.output_text.logprobs"]
    )
    result = _stream_partial_start_token(request)
    assert result is not None
    assert result.content == ""
