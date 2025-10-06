# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer


class TestPhi4MiniToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, default_tokenizer: AnyTokenizer) -> AnyTokenizer:
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
            xfail_streaming={
                "test_no_tool_calls": "Phi4 Mini streaming not implemented",
                "test_single_tool_call_simple_args": (
                    "Phi4 Mini streaming not implemented"
                ),
                "test_parallel_tool_calls": "Phi4 Mini streaming not implemented",
                "test_various_data_types": "Phi4 Mini streaming not implemented",
                "test_empty_arguments": "Phi4 Mini streaming not implemented",
                "test_surrounding_text": "Phi4 Mini streaming not implemented",
                "test_escaped_strings": "Phi4 Mini streaming not implemented",
                "test_streaming_reconstruction": "Phi4 Mini streaming not implemented",
            },
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
