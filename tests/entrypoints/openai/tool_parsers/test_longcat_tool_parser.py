# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer


class TestLongCatToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, default_tokenizer: AnyTokenizer) -> AnyTokenizer:
        """Add some longcat specific tokens to the default vocab."""
        tokenizer = default_tokenizer
        tokenizer_vocab = tokenizer.get_vocab()
        tokenizer.get_vocab = MagicMock()
        tokenizer_vocab.update(
            {
                "<longcat_tool_call>": 32000,
                "</longcat_tool_call>": 32001,
            }
        )
        tokenizer.get_vocab.return_value = tokenizer_vocab
        return tokenizer

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="longcat",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                '<longcat_tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</longcat_tool_call>'
            ),
            parallel_tool_calls_output=(
                '<longcat_tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</longcat_tool_call>\n'
                '<longcat_tool_call>{"name": "get_time", '
                '"arguments": {"timezone": "Asia/Tokyo"}}</longcat_tool_call>'
            ),
            various_data_types_output="""<longcat_tool_call>{
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
}</longcat_tool_call>""",
            empty_arguments_output=(
                '<longcat_tool_call>{"name": "refresh", "arguments": {}}'
                "</longcat_tool_call>"
            ),
            surrounding_text_output=(
                "Let me check the weather for you.\n"
                '<longcat_tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</longcat_tool_call>\n'
                "Here is the result."
            ),
            escaped_strings_output="""<longcat_tool_call>{
  "name": "test_function",
  "arguments": {
    "quoted": "He said \\"hello\\"",
    "path": "C:\\\\Users\\\\file.txt",
    "newline": "line1\\nline2",
    "unicode": "emoji: 🎉"
  }
}</longcat_tool_call>""",
            malformed_input_outputs=[
                '<longcat_tool_call>{"name": "func", "arguments": {',
                (
                    '<longcat_tool_call>{"name": "func", '
                    '"arguments": "not a dict"}</longcat_tool_call>'
                ),
                "Some text with <longcat_tool_call>invalid json",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers
            xfail_streaming={
                "test_malformed_input": "Streaming has complex buffering behavior",
            },
            xfail_nonstreaming={},
            # Configuration
            allow_empty_or_json_empty_args=True,
        )
