# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import DummyTokenizer, run_tool_extraction_streaming
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


def test_streaming_parallel_calls_single_delta():
    parser = Granite20bFCToolParser(DummyTokenizer())
    model_output = (
        '<function_call> {"name": "get_weather", "arguments": {"x": 42}}'
        '<function_call> {"name": "b", "arguments": {"y": 7}}'
    )

    reconstructor = run_tool_extraction_streaming(
        parser, [model_output], assert_one_tool_per_delta=False
    )

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "b",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"x": 42}
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {"y": 7}
