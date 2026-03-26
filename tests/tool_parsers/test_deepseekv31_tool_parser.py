# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers.deepseekv31_tool_parser import (
    DeepSeekV31ToolParser,
)

MODEL = "deepseek-ai/DeepSeek-V3.1"


@pytest.fixture(scope="module")
def deepseekv31_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(deepseekv31_tokenizer):
    return DeepSeekV31ToolParser(deepseekv31_tokenizer)


def test_extract_tool_calls_with_tool(parser):
    model_output = (
        "normal text"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "foo"
    assert result.tool_calls[0].function.arguments == '{"x":1}'
    assert result.content == "normal text"


def test_extract_tool_calls_with_multiple_tools(parser):
    model_output = (
        "some prefix text"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
        " some suffix text"
    )

    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called
    assert len(result.tool_calls) == 2

    assert result.tool_calls[0].function.name == "foo"
    assert result.tool_calls[0].function.arguments == '{"x":1}'

    assert result.tool_calls[1].function.name == "bar"
    assert result.tool_calls[1].function.arguments == '{"y":2}'

    # prefix is content
    assert result.content == "some prefix text"


class TestDeepSeekV31ToolParserStreaming(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return get_tokenizer(MODEL)

    @pytest.fixture
    def streaming(self) -> bool:
        """Only test streaming mode."""
        return True

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="deepseek_v31",
            no_tool_calls_output="normal text",
            single_tool_call_output=(
                "normal text"
                "<｜tool▁calls▁begin｜>"
                '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
                "<｜tool▁calls▁end｜>"
            ),
            parallel_tool_calls_output=(
                "some prefix text"
                "<｜tool▁calls▁begin｜>"
                '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
                '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
                "<｜tool▁calls▁end｜>"
                " some suffix text"
            ),
            various_data_types_output=(
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>foo<｜tool▁sep｜>"
                '{"string_field": "hello", "int_field": 42, '
                '"float_field": 3.14, "bool_field": true, '
                '"null_field": null, '
                '"array_field": [1, 2], '
                '"object_field": {"k": "v"}}'
                "<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜>"
            ),
            empty_arguments_output=(
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{}"
                "<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜>"
            ),
            surrounding_text_output=(
                "some prefix text"
                "<｜tool▁calls▁begin｜>"
                '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}'
                "<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜>"
                " some suffix text"
            ),
            escaped_strings_output=(
                "<｜tool▁calls▁begin｜>"
                '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"text": "a\\"b"}'
                "<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜>"
            ),
            malformed_input_outputs=[
                (
                    "<｜tool▁calls▁begin｜>"
                    '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1'
                    "<｜tool▁call▁end｜>"
                    "<｜tool▁calls▁end｜>"
                ),
            ],
            single_tool_call_expected_name="foo",
            single_tool_call_expected_args={"x": 1},
            single_tool_call_expected_content="normal text",
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["foo", "bar"],
            xfail_streaming={},
            xfail_nonstreaming={},
        )
