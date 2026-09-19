# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.tokenizers import get_tokenizer
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


def test_streaming_whole_tool_call_in_one_delta(parser):
    # A complete tool call (begin tag, name, arguments, end tag) arriving in a
    # single streaming delta must be emitted, matching the non-streaming result.
    # Before the fix, start_count == end_count on the first pass sent the parser
    # into the "closing" branch with nothing opened, dropping the call.
    model_output = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    nonstream = parser.extract_tool_calls(model_output, None)
    assert len(nonstream.tool_calls) == 1

    reconstructor = run_tool_extraction_streaming(parser, [model_output])

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "foo"
    assert reconstructor.tool_calls[0].function.name == (
        nonstream.tool_calls[0].function.name
    )
    assert reconstructor.tool_calls[0].function.arguments == '{"x":1}'
    assert reconstructor.tool_calls[0].function.arguments == (
        nonstream.tool_calls[0].function.arguments
    )


def test_streaming_parallel_tool_calls_in_one_delta(parser):
    # Multiple complete tool calls in a single delta must all be emitted.
    model_output = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    reconstructor = run_tool_extraction_streaming(
        parser, [model_output], assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 2
    assert reconstructor.tool_calls[0].function.name == "foo"
    assert reconstructor.tool_calls[0].function.arguments == '{"x":1}'
    assert reconstructor.tool_calls[1].function.name == "bar"
    assert reconstructor.tool_calls[1].function.arguments == '{"y":2}'
