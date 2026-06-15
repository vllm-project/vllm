# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PoolsideV1ToolParser."""

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)


def _tool_call(name: str, args: list[tuple[str, str]]) -> str:
    arg_block = "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args
    )
    return f"<tool_call>{name}\n{arg_block}</tool_call>"


class TestPoolsideV1ToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="poolside_v1",
            no_tool_calls_output=(
                "Sure, I can help with that. The capital of France is Paris."
            ),
            single_tool_call_output=_tool_call("get_weather", [("city", "Tokyo")]),
            parallel_tool_calls_output=(
                _tool_call("get_weather", [("city", "Tokyo")])
                + _tool_call("get_time", [("timezone", "UTC")])
            ),
            various_data_types_output=_tool_call(
                "complex_call",
                [
                    ("string_field", "hello"),
                    ("int_field", "42"),
                    ("float_field", "3.14"),
                    ("bool_field", "true"),
                    ("null_field", "null"),
                    ("array_field", '["a", "b", "c"]'),
                    ("object_field", '{"nested": "value"}'),
                ],
            ),
            empty_arguments_output=_tool_call("ping", []),
            surrounding_text_output=(
                "Let me check the weather for you. "
                + _tool_call("get_weather", [("city", "Tokyo")])
            ),
            escaped_strings_output=_tool_call(
                "echo", [("message", 'He said "hello" and left')]
            ),
            malformed_input_outputs=[
                # Unterminated tool_call
                "<tool_call>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
                # Empty function name: parser still matches; emits ToolCall(name="")
                "<tool_call>\n"
                "<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
                # Stray end tag with no start
                "</tool_call>some text",
                # arg_value without arg_key
                "<tool_call>get_weather\n<arg_value>Tokyo</arg_value></tool_call>",
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            supports_typed_arguments=True,
            # Streaming uses _tools_enabled(); with no tools in the request it
            # treats all output as plain content, so streaming and non-streaming
            # results diverge for every tool-call test.
            xfail_streaming={
                "test_single_tool_call_simple_args": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_parallel_tool_calls": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_various_data_types": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_empty_arguments": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_surrounding_text": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_escaped_strings": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
                "test_streaming_reconstruction": (
                    "Streaming parser requires request.tools to emit tool deltas"
                ),
            },
        )
