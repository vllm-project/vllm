# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager


class TestPhi4MiniToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, default_tokenizer: TokenizerLike) -> TokenizerLike:
        """Add some phi4mini specific tokens to the default vocab."""

        tokenizer = default_tokenizer
        tokenizer_vocab = tokenizer.get_vocab()
        tokenizer.get_vocab = MagicMock()
        tokenizer_vocab.update(
            {
                "functools": 32000,
            }
        )
        tokenizer.get_vocab.return_value = tokenizer_vocab
        return tokenizer

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="phi4_mini_json",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                'functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
            ),
            parallel_tool_calls_output="""functools[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
]""",
            various_data_types_output="""functools[{
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
}]""",
            empty_arguments_output='functools[{"name": "refresh", "arguments": {}}]',
            surrounding_text_output="""Let me check the weather for you.
functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
Would you like to know more?""",
            escaped_strings_output="""functools[{
  "name": "test_function",
  "arguments": {
    "quoted": "He said \\"hello\\"",
    "path": "C:\\\\Users\\\\file.txt",
    "newline": "line1\\nline2",
    "unicode": "emoji: 🎉"
  }
}]""",
            malformed_input_outputs=[
                'functools[{"name": "func", "arguments": {',
                'functools[{"name": "func", "arguments": "not a dict"}]',
                'functools{"name": "func"}',  # Missing brackets
                'functools[{"name": "func"}]',  # Missing arguments/parameters
                "functools[] This is just text",  # Empty functools
                "functools[ This is just text ]",  # functools with invalid JSON
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            # Phi-4 Mini strips content when tool calls present
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            parallel_tool_calls_expected_content=None,
            # xfail markers
            xfail_streaming={},
            xfail_nonstreaming={
                "test_various_data_types": (
                    "Phi4MiniJsonToolParser regex has nesting limitations "
                    "with nested objects"
                ),
                "test_malformed_input": (
                    "Phi4MiniJsonToolParser incorrectly sets "
                    "tools_called=True on empty array"
                ),
            },
        )


@pytest.fixture
def phi4mini_parser(default_tokenizer: TokenizerLike) -> ToolParser:
    return ToolParserManager.get_tool_parser("phi4_mini_json")(default_tokenizer)


def test_streaming_split_marker_is_not_emitted_as_content(
    phi4mini_parser: ToolParser,
) -> None:
    """The `functools` marker spans several tokens and must not leak."""
    deltas = [
        "func",
        "tools",
        "[",
        '{"name": "get_weather", ',
        '"arguments": {"city": ',
        '"Tokyo"}}]',
    ]

    reconstructor = run_tool_extraction_streaming(phi4mini_parser, deltas)

    assert reconstructor.other_content == "", (
        f"Marker leaked into content: {reconstructor.other_content!r}"
    )
    assert len(reconstructor.tool_calls) == 1, (
        f"Expected 1 tool call, got {len(reconstructor.tool_calls)}"
    )
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }


def test_streaming_parameters_key_is_streamed_as_arguments(
    phi4mini_parser: ToolParser,
) -> None:
    """phi-4-mini templates emit either `arguments` or `parameters`."""
    deltas = [
        'functools[{"name": "get_time", ',
        '"parameters": {"timezone": "Asia/Tokyo"}}]',
    ]

    reconstructor = run_tool_extraction_streaming(phi4mini_parser, deltas)

    assert len(reconstructor.tool_calls) == 1, (
        f"Expected 1 tool call, got {len(reconstructor.tool_calls)}"
    )
    assert reconstructor.tool_calls[0].function.name == "get_time"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "timezone": "Asia/Tokyo"
    }


def test_streaming_text_before_marker_is_preserved(
    phi4mini_parser: ToolParser,
) -> None:
    """Content preceding the marker must still reach the client."""
    deltas = [
        "Let me check that.\n",
        'functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]',
    ]

    reconstructor = run_tool_extraction_streaming(phi4mini_parser, deltas)

    assert reconstructor.other_content == "Let me check that.\n", (
        f"Expected leading content to be preserved, got {reconstructor.other_content!r}"
    )
    assert len(reconstructor.tool_calls) == 1, (
        f"Expected 1 tool call, got {len(reconstructor.tool_calls)}"
    )
