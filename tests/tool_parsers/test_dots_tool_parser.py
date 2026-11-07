# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import get_function_delta
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.dots_tool_parser import DotsToolParser


def _tool(name: str, properties: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": properties},
        },
    )


def _request(tools: list[ChatCompletionToolsParam]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="auto",
    )


def _stream(
    parser: DotsToolParser,
    chunks: Iterable[str],
    request: ChatCompletionRequest,
) -> list[DeltaMessage]:
    previous_text = ""
    messages: list[DeltaMessage] = []
    for chunk in chunks:
        current_text = previous_text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
    return messages


@pytest.fixture
def parser() -> DotsToolParser:
    return DotsToolParser(MagicMock())


def test_registered_in_tool_parser_manager() -> None:
    assert ToolParserManager.get_tool_parser("dots") is DotsToolParser


def test_non_stream_xml_converts_schema_types_and_resolves_ref(
    parser: DotsToolParser,
) -> None:
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_location",
            "description": "Set location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"$ref": "#/$defs/Location"},
                    "days": {"type": "integer"},
                    "include_weather": {"type": "boolean"},
                },
                "$defs": {
                    "Location": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                },
            },
        },
    )
    request = _request([tool])
    text = (
        "ok<dots_function_call>"
        '<invoke name="set_location">'
        '<parameter name="location">{"city": "Shanghai"}</parameter>'
        '<parameter name="days">3</parameter>'
        '<parameter name="include_weather">true</parameter>'
        "</invoke>"
        "</dots_function_call>"
    )

    result = parser.extract_tool_calls(text, request)

    assert result.content == "ok"
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "set_location"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "location": {"city": "Shanghai"},
        "days": 3,
        "include_weather": True,
    }


def test_non_stream_supports_multiple_invokes_and_json_fallback(
    parser: DotsToolParser,
) -> None:
    tools = [
        _tool("search", {"query": {"type": "string"}}),
        _tool("open", {"id": {"type": "integer"}}),
    ]
    request = _request(tools)
    text = (
        "<dots_function_call>"
        '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
        '<invoke name="open"><parameter name="id">7</parameter></invoke>'
        "</dots_function_call>"
        '<dots_function_call>{"name":"search","arguments":{"query":"tables"}}'
        "</dots_function_call>"
    )

    result = parser.extract_tool_calls(text, request)

    assert [call.function.name for call in result.tool_calls] == [
        "search",
        "open",
        "search",
    ]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        {"query": "chairs"},
        {"id": 7},
        {"query": "tables"},
    ]


def test_non_stream_unknown_tool_is_left_as_content(parser: DotsToolParser) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])
    text = (
        "<dots_function_call>"
        '<invoke name="ghost"><parameter name="query">chairs</parameter></invoke>'
        "</dots_function_call>"
    )

    result = parser.extract_tool_calls(text, request)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text


def test_streaming_buffers_partial_marker_and_emits_all_complete_calls(
    parser: DotsToolParser,
) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])
    chunks = [
        "visible<dots_func",
        (
            "tion_call>"
            '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
            "</dots_function_call>"
            "<dots_function_call>"
            '<invoke name="search"><parameter name="query">tables</parameter></invoke>'
            "</dots_function_call>"
        ),
    ]

    messages = _stream(parser, chunks, request)

    assert "".join(message.content or "" for message in messages) == "visible"
    calls = [call for message in messages for call in message.tool_calls]
    assert [call.index for call in calls] == [0, 1]
    assert [json.loads(get_function_delta(call).arguments or "") for call in calls] == [
        {"query": "chairs"},
        {"query": "tables"},
    ]


def test_streaming_filters_unknown_tools_and_surfaces_content(
    parser: DotsToolParser,
) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])
    text = (
        "<dots_function_call>"
        '<invoke name="ghost"><parameter name="query">chairs</parameter></invoke>'
        "</dots_function_call>"
    )

    messages = _stream(parser, [text], request)

    assert len(messages) == 1
    assert messages[0].tool_calls == []
    assert "ghost" in (messages[0].content or "")
    assert parser._buffer == ""


def test_streaming_malformed_block_does_not_block_later_valid_call(
    parser: DotsToolParser,
) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])
    messages = _stream(
        parser,
        [
            "<dots_function_call>garbage</dots_function_call>",
            (
                "<dots_function_call>"
                '<invoke name="search">'
                '<parameter name="query">chairs</parameter>'
                "</invoke></dots_function_call>"
            ),
        ],
        request,
    )

    assert messages[0].content == "garbage"
    assert messages[0].tool_calls == []
    assert messages[1].tool_calls[0].function is not None
    assert messages[1].tool_calls[0].function.name == "search"


def test_streaming_strips_stray_end_marker(parser: DotsToolParser) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])
    messages = _stream(parser, ["some text </dots_function_call>"], request)

    assert len(messages) == 1
    assert messages[0].content == "some text "
    assert messages[0].tool_calls == []


def test_streaming_flushes_partial_opening_marker_at_eof(
    parser: DotsToolParser,
) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])

    messages = _stream(parser, ["answer <dots_func"], request)

    assert messages[0].content == "answer "
    assert parser.flush_pending_normal_text() == "<dots_func"
    assert parser.flush_pending_normal_text() == ""


def test_streaming_emits_complete_json_before_end_marker_without_duplication(
    parser: DotsToolParser,
) -> None:
    request = _request([_tool("search", {"query": {"type": "string"}})])

    messages = _stream(
        parser,
        [
            "<dots_function_call>",
            '{"name":"search","arguments":{"query":"chairs"}}',
            "</dots_function_call>",
        ],
        request,
    )

    calls = [call for message in messages for call in message.tool_calls]
    assert [get_function_delta(call).name for call in calls] == ["search", None]
    assert "".join(get_function_delta(call).arguments or "" for call in calls) == (
        '{"query": "chairs"}'
    )
    assert messages[-1].tool_calls == calls


def test_convert_param_value_null_literal_kept_for_strings(
    parser: DotsToolParser,
) -> None:
    # The dots bug: a string param the model wrote as the literal text "null"
    # must not become JSON null.
    assert parser._convert_param_value("null", "string") == "null"
    assert parser._convert_param_value("Null", "string") == "Null"
    # Nullable schemas still get the real null.
    assert parser._convert_param_value("null", ["string", "null"]) is None
    assert parser._convert_param_value("null", "null") is None
    # On other non-nullable scalars the shared coercer maps the text to JSON
    # null (its trailing json.loads fallback); schema validation rejects it
    # downstream either way.
    assert parser._convert_param_value("null", "integer") is None
    assert parser._convert_param_value("null", "boolean") is None


def test_convert_param_value_number_never_emits_non_finite(
    parser: DotsToolParser,
) -> None:
    # inf/nan are not valid JSON numbers; keep the raw text rather than emit
    # `Infinity`/`NaN` into the arguments JSON.
    assert parser._convert_param_value("inf", "number") == "inf"
    assert parser._convert_param_value("Infinity", "number") == "Infinity"
    assert parser._convert_param_value("-inf", "number") == "-inf"
    assert parser._convert_param_value("nan", "number") == "nan"
    assert parser._convert_param_value("5", "number") == 5
    assert parser._convert_param_value("5.5", "number") == 5.5


def test_convert_param_value_boolean_rejects_invalid_text(
    parser: DotsToolParser,
) -> None:
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("1", "boolean") is True
    assert parser._convert_param_value("false", "boolean") is False
    assert parser._convert_param_value("0", "boolean") is False
    # Invalid boolean text must survive for schema validation, not silently
    # read as False.
    assert parser._convert_param_value("maybe", "boolean") == "maybe"
    assert parser._convert_param_value("yes", "boolean") == "yes"


def test_non_stream_xml_null_literal_and_anyof_nullable(
    parser: DotsToolParser,
) -> None:
    tool = _tool(
        "search",
        {
            "query": {"type": "string"},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "flag": {"type": "boolean"},
        },
    )
    request = _request([tool])
    text = (
        "<dots_function_call>"
        '<invoke name="search">'
        '<parameter name="query">null</parameter>'
        '<parameter name="note">null</parameter>'
        '<parameter name="flag">maybe</parameter>'
        "</invoke>"
        "</dots_function_call>"
    )

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "query": "null",
        "note": None,
        "flag": "maybe",
    }
