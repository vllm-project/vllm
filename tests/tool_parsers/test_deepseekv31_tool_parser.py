# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.deepseekv31_tool_parser import (
    DeepSeekV31ToolParser,
)

MODEL = "deepseek-ai/DeepSeek-V3.1"

TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
TOOL_CALL_START = "<｜tool▁call▁begin｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_SEP = "<｜tool▁sep｜>"


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


def test_streaming_close_brace_with_end_token_but_quote_in_prior_delta(parser):
    deltas = [
        TOOL_CALLS_START,
        TOOL_CALL_START,
        "get_weather",
        TOOL_SEP,
        '{"city": "NYC',
        '"',  # quote arrives alone
        "}" + TOOL_CALL_END,  # brace + end token in same delta
        TOOL_CALLS_END,
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=True
    )
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    args = reconstructor.tool_calls[0].function.arguments
    # The closing "}" must not be dropped
    assert args.endswith("}")
    assert "city" in args
    assert "NYC" in args


def test_streaming_close_brace_alone_with_end_token(parser):
    deltas = [
        TOOL_CALLS_START,
        TOOL_CALL_START,
        "get_weather",
        TOOL_SEP,
        '{"x": 1',
        "}" + TOOL_CALL_END,  # closing brace + end token
        TOOL_CALLS_END,
    ]
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=True
    )
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    args = reconstructor.tool_calls[0].function.arguments
    assert args.endswith("}")
    assert "x" in args
