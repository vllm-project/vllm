# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.granite_20b_fc_tool_parser import Granite20bFCToolParser


class TestGranite20bFcToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="granite-20b-fc",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                '<function_call> {"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}'
            ),
            parallel_tool_calls_output=(
                '<function_call> {"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}\n'
                '<function_call> {"name": "get_time", '
                '"arguments": {"timezone": "Asia/Tokyo"}}'
            ),
            various_data_types_output="""<function_call> {
  "name": "test_function",
  "arguments": {
    "string_field": "hello",
    "int_field": 42,
    "float_field": 3.14,
    "bool_field": true,
    "null_field": null,
    "array_field": ["a", "b", "c"],
    "object_field": {"nested": "value"},
    "empty_array": [],
    "empty_object": {}
  }
}""",
            empty_arguments_output=(
                '<function_call> {"name": "refresh", "arguments": {}}'
            ),
            surrounding_text_output="""Let me check the weather for you.
<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}""",
            escaped_strings_output="""<function_call> {
  "name": "test_function",
  "arguments": {
    "quoted": "He said \\"hello\\"",
    "path": "C:\\\\Users\\\\file.txt",
    "newline": "line1\\nline2",
    "unicode": "emoji: 🎉"
  }
}""",
            malformed_input_outputs=[
                '<function_call> {"name": "func", "arguments": {',
                '<function_call> [{"name": "func", "arguments": {}}]',
                '{"name": "func", "arguments": {}}',
                '<function_call> {"name": 123}',
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers
            xfail_streaming={
                "test_surrounding_text": (
                    "Granite 20B FC streaming requires <function_call> at start"
                ),
            },
            xfail_nonstreaming={},
        )


class _StubTokenizer:
    """Minimal tokenizer: granite-20b-fc resolves no special-token ids, so
    plain string deltas are enough to drive streaming tests."""

    def get_vocab(self) -> dict[str, int]:
        return {}

    def tokenize(self, text: str) -> list[str]:
        return []


def test_streaming_parallel_calls_single_delta():
    """Parallel calls arriving whole in one delta must not be lost.

    This happens e.g. with async scheduling or stream_interval > 1, where
    a single streamed delta can contain the entire tool call message.
    Previously the "starting a new tool" branch recorded the new state
    and returned without emitting anything.
    """
    parser = Granite20bFCToolParser(_StubTokenizer())
    model_output = (
        '<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
        '<function_call> {"name": "get_time", "arguments": {"tz": "UTC"}}'
    )

    reconstructor = run_tool_extraction_streaming(
        parser, [model_output], assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"tz": "UTC"}


def test_streaming_parallel_calls_single_delta_with_empty_tail():
    """A whole message followed by one empty delta must emit every tool."""
    parser = Granite20bFCToolParser(_StubTokenizer())
    model_output = (
        '<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
        '<function_call> {"name": "get_time", "arguments": {"tz": "UTC"}}'
    )

    reconstructor = run_tool_extraction_streaming(
        parser, [model_output, ""], assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"tz": "UTC"}


def test_streaming_parallel_calls_two_coarse_deltas():
    """Calls completed inside a coarse delta must all be emitted in full."""
    parser = Granite20bFCToolParser(_StubTokenizer())
    deltas = [
        # the start marker plus an incomplete first tool call
        '<function_call> {"name": "get_weather", "arg',
        # completes the first tool call and carries the whole second one
        'uments": {"city": "Tokyo"}}\n'
        '<function_call> {"name": "get_time", "arguments": {"tz": "UTC"}}',
    ]

    reconstructor = run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"tz": "UTC"}


def test_streaming_parallel_calls_three_coarse_deltas():
    """A tool completed before the last one must be emitted in full."""
    parser = Granite20bFCToolParser(_StubTokenizer())
    deltas = [
        '<function_call> {"name": "get_weather", "arg',
        'uments": {"city": "Tokyo"}}\n<function_call> {"na',
        'me": "get_time", "arguments": {"tz": "UTC"}}',
    ]

    reconstructor = run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"tz": "UTC"}


def _collect_streaming_deltas(tool_parser, deltas):
    """Drive the parser manually and collect every DeltaMessage."""
    request = ChatCompletionRequest(messages=[], model="test-model")
    messages = []
    previous_text, previous_token_ids = "", []
    for delta in deltas:
        tokens = tool_parser.model_tokenizer.tokenize(delta)
        token_delta = [
            tool_parser.vocab.get(token)
            for token in tokens
            if token in tool_parser.vocab
        ]
        current_text = previous_text + delta
        current_token_ids = previous_token_ids + token_delta
        message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_token_ids,
            current_token_ids,
            token_delta,
            request,
        )
        if message is not None:
            messages.append(message)
        previous_text, previous_token_ids = current_text, current_token_ids
    return messages


def test_streaming_no_empty_tool_call_deltas():
    """A delta with nothing new to send must yield None, not a chunk.

    A new call arriving after the previous one was fully streamed used
    to emit a DeltaToolCall whose only content was an empty arguments
    string -- a meaningless chunk for the client. The same applies to a
    trailing empty delta after the whole message was streamed.
    """
    parser = Granite20bFCToolParser(_StubTokenizer())
    deltas = [
        '<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}',
        '\n<function_call> {"name": "get_time", "arg',
        'uments": {"tz": "UTC"}}',
        "",
    ]

    messages = _collect_streaming_deltas(parser, deltas)

    assert messages
    for message in messages:
        assert message.content is not None or message.tool_calls, (
            "streaming must not emit empty DeltaMessages"
        )
        for tool_call in message.tool_calls or []:
            function = tool_call.function
            assert function is not None, (
                "tool call deltas must include a function delta"
            )
            assert function.name or function.arguments, (
                "tool call deltas must carry a name or an argument diff"
            )


def test_streaming_partial_call_then_coarse_delta():
    """A partially streamed call completed by a coarse delta must be
    flushed together with the calls the coarse delta completes.

    The deltas below stream nothing from the first call yet, then send
    its name with the arguments still incomplete; the last delta closes
    the first call and carries two complete calls, so it must flush the
    first call's arguments and batch-emit the two new calls in one
    update.
    """
    parser = Granite20bFCToolParser(_StubTokenizer())
    deltas = [
        '<function_call> {"name": "get_weather", "arg',
        'uments": {"city": "Tokyo"',
        '}}\n<function_call> {"name": "get_time", '
        '"arguments": {"tz": "UTC"}}\n<function_call> '
        '{"name": "send_email", "arguments": {"to": "a@example.com"}}',
    ]

    reconstructor = run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
        "send_email",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"tz": "UTC"}
    assert json.loads(reconstructor.tool_calls[2].function.arguments) == {
        "to": "a@example.com"
    }
