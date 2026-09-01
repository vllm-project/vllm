# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest

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
    assert [json.loads(call.function.arguments or "") for call in calls] == [
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
    assert [call.function.name for call in calls] == ["search", None]
    assert "".join(call.function.arguments or "" for call in calls) == (
        '{"query": "chairs"}'
    )
    assert messages[-1].tool_calls == calls


def test_unterminated_block_with_whitespace_is_not_cubic(
    parser: DotsToolParser,
) -> None:
    """A short model output must not take seconds to parse.

    ``_block_regex`` matched ``\\s*(.*?)\\s*`` between the markers. Under
    ``DOTALL`` a dot matches a space too, so those three quantifiers competed
    for the same run of spaces and a block with no closing marker after it
    backtracked through every way of splitting that run: a ~3 KB model output
    cost more than a minute of CPU on the event loop.
    """
    request = _request([_tool("get_weather", {"city": {"type": "string"}})])
    model_output = DotsToolParser.tool_call_start_token + " " * 4096

    start = time.perf_counter()
    result = parser.extract_tool_calls(model_output, request)
    elapsed = time.perf_counter() - start

    assert not result.tools_called
    assert result.content == model_output
    # Before the fix this took minutes; a generous bound keeps the test stable
    # on slow CI machines while still failing on the old behaviour.
    assert elapsed < 5.0, f"parsing took {elapsed:.1f}s"


def test_adjacent_blocks_still_parsed_separately(parser: DotsToolParser) -> None:
    """The capture must stay lazy: a block ends at the *first* closing marker.

    Guards the fix against a rewrite that lets one block swallow the next.
    """
    request = _request([_tool("get_weather", {"city": {"type": "string"}})])
    block = "{}<invoke name=get_weather><parameter name=city>{}</parameter></invoke>{}"
    model_output = block.format(
        DotsToolParser.tool_call_start_token,
        "Paris",
        DotsToolParser.tool_call_end_token,
    ) + block.format(
        DotsToolParser.tool_call_start_token,
        "Berlin",
        DotsToolParser.tool_call_end_token,
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert len(result.tool_calls) == 2
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Berlin"}


def test_whitespace_padded_block_content_is_still_stripped(
    parser: DotsToolParser,
) -> None:
    """Making the whitespace runs possessive must not change what is captured."""
    request = _request([_tool("get_weather", {"city": {"type": "string"}})])
    model_output = (
        DotsToolParser.tool_call_start_token + "\n   <invoke name=get_weather>"
        "<parameter name=city>Paris</parameter></invoke>   \n"
        + DotsToolParser.tool_call_end_token
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
