# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.entrypoints.openai.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer


class TestStep3ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> AnyTokenizer:
        return get_tokenizer("stepfun-ai/step3")

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="step3",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            parallel_tool_calls_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_sep｜>"
                '<｜tool_call_begin｜><steptml:invoke name="get_time">'
                '<steptml:parameter name="timezone">Asia/Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            various_data_types_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="test_function">'
                '<steptml:parameter name="string_field">hello</steptml:parameter>'
                '<steptml:parameter name="int_field">42</steptml:parameter>'
                '<steptml:parameter name="float_field">3.14</steptml:parameter>'
                '<steptml:parameter name="bool_field">true</steptml:parameter>'
                '<steptml:parameter name="null_field">null</steptml:parameter>'
                '<steptml:parameter name="array_field">'
                '["a", "b", "c"]</steptml:parameter>'
                '<steptml:parameter name="object_field">'
                '{"nested": "value"}</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            empty_arguments_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="refresh"></steptml:invoke>'
                "<｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>\n\n"
                "I'll get that information."
            ),
            escaped_strings_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="test_function">'
                '<steptml:parameter name="quoted">He said "hello"</steptml:parameter>'
                '<steptml:parameter name="path">C:\\Users\\file.txt</steptml:parameter>'
                '<steptml:parameter name="newline">line1\nline2</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            malformed_input_outputs=[
                (
                    "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                    '<steptml:invoke name="func">'
                ),
                (
                    '<｜tool_call_begin｜><steptml:invoke name="func">'
                    "</steptml:invoke><｜tool_call_end｜>"
                ),
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers
            xfail_nonstreaming={
                "test_single_tool_call_simple_args": (
                    "Step3 parser non-streaming has bugs"
                ),
                "test_parallel_tool_calls": ("Step3 parser non-streaming has bugs"),
                "test_various_data_types": "Step3 parser non-streaming has bugs",
                "test_empty_arguments": "Step3 parser non-streaming has bugs",
                "test_surrounding_text": "Step3 parser non-streaming has bugs",
                "test_escaped_strings": "Step3 parser non-streaming has bugs",
            },
            xfail_streaming={
                "test_parallel_tool_calls": (
                    "Step3 parser has significant bugs in both streaming "
                    "and non-streaming"
                ),
                "test_streaming_reconstruction": (
                    "Step3 parser non-streaming has bugs, so streaming "
                    "doesn't match non-streaming"
                ),
            },
            supports_typed_arguments=False,
        )
