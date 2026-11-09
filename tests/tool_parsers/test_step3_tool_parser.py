# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers.step3_tool_parser import Step3ToolParser


class TestStep3ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
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

    def test_parameter_value_with_angle_bracket(
        self, tool_parser: Any, streaming: bool
    ):
        model_output = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="run_query">'
            '<steptml:parameter name="sql">SELECT * FROM t WHERE a < 5'
            "</steptml:parameter>"
            '<steptml:parameter name="html"><b>hi</b></steptml:parameter>'
            "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
        )
        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "run_query"
        assert json.loads(tool_calls[0].function.arguments) == {
            "sql": "SELECT * FROM t WHERE a < 5",
            "html": "<b>hi</b>",
        }


_SCHEMA_PROPERTIES = {
    "count": {"type": "integer"},
    "opt_count_list": {"type": ["integer", "null"]},
    "opt_count_any_of": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    "opt_ratio": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    "opt_flag": {"oneOf": [{"type": "boolean"}, {"type": "null"}]},
    "opt_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "zip_code": {"type": "string"},
    "filters": {"type": "object"},
    "tags": {"type": "array"},
    "untyped": {},
}


def _cast(params: dict[str, str]) -> dict[str, Any]:
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "search",
            "parameters": {"type": "object", "properties": _SCHEMA_PROPERTIES},
        },
    )
    parser = Step3ToolParser(MagicMock(), tools=[tool])
    return parser._cast_arguments("search", dict(params))


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("count", "5", 5),
        ("opt_count_list", "5", 5),
        ("opt_count_any_of", "5", 5),
        ("opt_count_any_of", "null", None),
        ("opt_ratio", "0.5", 0.5),
        ("opt_flag", "true", True),
        ("opt_name", "Tokyo", "Tokyo"),
        ("opt_name", "null", None),
        ("zip_code", "10001", "10001"),
        ("filters", '{"a": 1}', {"a": 1}),
        ("tags", '["x", "y"]', ["x", "y"]),
        ("untyped", "10001", "10001"),
    ],
)
def test_cast_arguments_follows_schema_types(key: str, raw: str, expected: Any):
    assert _cast({key: raw})[key] == expected


def test_cast_arguments_keeps_uncoercible_values():
    out = _cast({"opt_count_any_of": "many", "opt_flag": "maybe"})
    assert out == {"opt_count_any_of": "many", "opt_flag": "maybe"}
