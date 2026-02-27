# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.entrypoints.openai.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction


class TestGraniteToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="granite",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                '<|tool_call|> [{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}]'
            ),
            parallel_tool_calls_output="""<|tool_call|> [
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
]""",
            various_data_types_output="""<tool_call> [{
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
            empty_arguments_output=(
                '<|tool_call|> [{"name": "refresh", "arguments": {}}]'
            ),
            surrounding_text_output="""Let me check the weather for you.
<|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
I'll get that information.""",
            escaped_strings_output="""<tool_call> [{
  "name": "test_function",
  "arguments": {
    "quoted": "He said \\"hello\\"",
    "path": "C:\\\\Users\\\\file.txt",
    "newline": "line1\\nline2",
    "unicode": "emoji: 🎉"
  }
}]""",
            malformed_input_outputs=[
                '<|tool_call|> [{"name": "func", "arguments": {',
                '<|tool_call|> {"name": "func", "arguments": {}}',  # Not an array
                '[{"name": "func", "arguments": "not a dict"}]',
                'Some text [{"name": "func"}]',  # JSON but not tool call format
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            # Granite strips content when tool calls present
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers
            xfail_streaming={
                "test_malformed_input": (
                    "Streaming mode incorrectly creates tool call from malformed JSON"
                ),
                "test_surrounding_text": (
                    "Parser doesn't handle surrounding text correctly in streaming"
                ),
                "test_streaming_reconstruction": (
                    "Streaming mode doesn't strip <|tool_call|> marker from content"
                ),
            },
            xfail_nonstreaming={
                "test_surrounding_text": (
                    "Parser doesn't handle surrounding text correctly in non-streaming"
                ),
            },
        )

    # Granite-Specific Tests

    @pytest.mark.parametrize("streaming", [True, False])
    def test_granite_token_prefix_format(self, tool_parser, streaming):
        """Verify parser handles Granite 3.0 <|tool_call|> token format."""
        single_tool_call_token = (
            '<|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
        )
        content, tool_calls = run_tool_extraction(
            tool_parser, single_tool_call_token, streaming=streaming
        )
        assert len(tool_calls) == 1, (
            f"Expected 1 tool call from token format, got {len(tool_calls)}"
        )
        assert tool_calls[0].function.name == "get_weather"

    @pytest.mark.parametrize("streaming", [True, False])
    def test_granite_string_prefix_format(self, tool_parser, streaming):
        """Verify parser handles Granite 3.1 <tool_call> string format."""
        single_tool_call_string = (
            '<tool_call> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
        )
        content, tool_calls = run_tool_extraction(
            tool_parser, single_tool_call_string, streaming=streaming
        )
        assert len(tool_calls) == 1, (
            f"Expected 1 tool call from string format, got {len(tool_calls)}"
        )
        assert tool_calls[0].function.name == "get_weather"
