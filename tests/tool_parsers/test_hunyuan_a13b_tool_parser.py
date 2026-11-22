# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall
from vllm.tool_parsers import ToolParser, ToolParserManager


def make_tool_call(name, arguments):
    return ToolCall(
        type="function",
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


# TODO: add reason prefix and suffix.


@pytest.mark.parametrize(
    "model_output,expected_tool_calls,expected_content",
    [
        # No tool call
        ("How can I help you today?", [], "How can I help you today?"),
        # Single tool call, no content
        (
            '<tool_calls>[{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}]</tool_calls>',  # noqa: E501
            [
                make_tool_call(
                    "get_weather", {"city": "San Francisco", "metric": "celsius"}
                )
            ],
            None,
        ),
        # Multiple tool calls
        (
            '<tool_calls>[{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}, {"name": "register_user", "arguments": {"name": "John Doe", "age": 37, "address": {"city": "San Francisco", "state": "CA"}, "role": null, "passed_test": true, "aliases": ["John", "Johnny"]}}]</tool_calls>',  # noqa: E501
            [
                make_tool_call(
                    "get_weather", {"city": "San Francisco", "metric": "celsius"}
                ),
                make_tool_call(
                    "register_user",
                    {
                        "name": "John Doe",
                        "age": 37,
                        "address": {"city": "San Francisco", "state": "CA"},
                        "role": None,
                        "passed_test": True,
                        "aliases": ["John", "Johnny"],
                    },
                ),
            ],
            None,
        ),
        # Content before tool call
        (
            'I will call the tool now. <tool_calls>[{"name": "get_weather", "arguments": {"city": "Boston"}}]</tool_calls>',  # noqa: E501
            [make_tool_call("get_weather", {"city": "Boston"})],
            "I will call the tool now. ",
        ),
        # Content after tool call (should be stripped)
        (
            '<tool_calls>[{"name": "get_weather", "arguments": {"city": "Seattle"}}]</tool_calls>\nThank you!',  # noqa: E501
            [make_tool_call("get_weather", {"city": "Seattle"})],
            None,
        ),
        (
            '<tool_calls>[{"name": "complex_tool", "arguments": {"level1": {"level2": {"level3": {"value": 123}}}}}]</tool_calls>',
            [
                make_tool_call(
                    "complex_tool", {"level1": {"level2": {"level3": {"value": 123}}}}
                )
            ],
            None,
        ),
    ],
)
def test_hunyuan_a13b_tool_parser_extract(
    model_output, expected_tool_calls, expected_content
):
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hunyuan_a13b")(
        mock_tokenizer
    )
    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=False
    )

    # align the random id.
    for idx in range(len(tool_calls)):
        tool_calls[idx].id = expected_tool_calls[idx].id
    assert tool_calls == expected_tool_calls
    assert content == expected_content


# Streaming test: simulate incremental output
@pytest.mark.parametrize(
    "model_deltas,expected_tool_calls",
    [
        (
            [
                '<tool_calls>[{"name": "get_weather", ',
                '"arguments": {"city": "San Francisco", ',
                '"metric": "celsius"}}]',
                "</tool_calls>",
            ],
            [
                make_tool_call(
                    "get_weather", {"city": "San Francisco", "metric": "celsius"}
                )
            ],
        ),
        (
            [
                '<tool_calls>[{"name":',
                ' "get_weather",',
                ' "arguments":',
                ' {"city": "Boston"}',
                "}]",
                "</tool_calls>",
            ],
            [make_tool_call("get_weather", {"city": "Boston"})],
        ),
        (
            [
                "",
                '<tool_calls>[{"name":',
                ' "get_weather",',
                ' "arguments":',
                ' {"city": "Boston"}',
                "}]",
                "</tool_calls>",
                "\n</answer>",
            ],
            [make_tool_call("get_weather", {"city": "Boston"})],
        ),
        pytest.param(
            [
                '<tool_calls>[{"name": "complex_tool",',
                ' "arguments": ',
                ' {"level1": {"level2": ',
                '{"level3": {"value": 123}}}}}',
                "]</tool_calls>",
            ],
            [
                make_tool_call(
                    "complex_tool", {"level1": {"level2": {"level3": {"value": 123}}}}
                )
            ],
            marks=pytest.mark.xfail(
                reason="stream parsing not support nested json yet."
            ),
        ),
    ],
)
def test_hunyuan_a13b_tool_parser_streaming(model_deltas, expected_tool_calls):
    mock_tokenizer = MagicMock()

    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hunyuan_a13b")(
        mock_tokenizer
    )
    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_deltas, assert_one_tool_per_delta=False
    )

    # align the random id.
    for idx in range(len(reconstructor.tool_calls)):
        reconstructor.tool_calls[idx].id = expected_tool_calls[idx].id

    assert reconstructor.tool_calls == expected_tool_calls


@pytest.mark.parametrize(
    "model_output",
    [
        "[1] Paris is the capital of France.",
        "[See the docs](https://docs.vllm.ai) for details.",
        "[ 1] spaced citation",
        "[\n\nSee the list below.",
        "[1, 2, 3]",
        "[]",
        "<tool_calls>[1] text",
        '[\u00a0{"name": "f"}] uses a no-break space.',
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_hunyuan_a13b_tool_parser_plain_text_starting_with_bracket(
    default_tokenizer, model_output, streaming
):
    """A reply that starts with "[" but not with an array of objects is content."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hunyuan_a13b")(
        default_tokenizer
    )
    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )
    assert tool_calls == []
    assert content == model_output


@pytest.mark.parametrize(
    "model_output",
    [
        '<tool_calls>[ {"name": "f", "arguments": {"a": 1}}]</tool_calls>',
        '<tool_calls>[\n  {"name": "f", "arguments": {"a": 1}}\n]</tool_calls>',
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_hunyuan_a13b_tool_parser_whitespace_after_bracket(
    default_tokenizer, model_output, streaming
):
    """Whitespace between "[" and "{" still opens a tool call."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hunyuan_a13b")(
        default_tokenizer
    )
    _, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming, assert_one_tool_per_delta=False
    )
    assert [
        (t.function.name, json.loads(t.function.arguments)) for t in tool_calls
    ] == [("f", {"a": 1})]


def test_hunyuan_a13b_tool_parser_non_ascii():
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("hunyuan_a13b")(
        mock_tokenizer
    )
    model_output = '<tool_calls>[{"name": "get_weather", "arguments": {"city": "北京"}}]</tool_calls>'
    _, tool_calls = run_tool_extraction(tool_parser, model_output, streaming=False)
    args = tool_calls[0].function.arguments
    assert "北京" in args
    assert "\\u" not in args
