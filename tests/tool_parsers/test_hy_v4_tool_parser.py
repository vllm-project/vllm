# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 tool call parser."""

from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

STRUCTURAL_TOKENS = [
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]


@pytest.fixture(scope="module")
def hy_v4_tokenizer():
    # The extractor requires the structural tokens in the vocab and detects
    # markers on token ids, so they must tokenize atomically.
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_tokens(STRUCTURAL_TOKENS)
    return tokenizer


@pytest.fixture
def hy_v4_tool_parser(hy_v4_tokenizer):
    return HYV4ToolParser(hy_v4_tokenizer)


@pytest.fixture
def mock_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_time",
                parameters={
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                },
            ),
        ),
    ]
    request.tool_choice = "auto"
    return request


def test_parallel_tool_calls_in_one_delta(hy_v4_tool_parser, mock_request):
    """Batched decode can deliver several newline-separated calls in one delta;
    streaming must emit all of them, matching non-streaming."""
    delta = (
        "<tool_calls>"
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Tokyo</arg_value></tool_call>"
        "\n"
        "<tool_call>get_time<arg_key>timezone</arg_key>"
        "<arg_value>UTC</arg_value></tool_call>"
        "</tool_calls>"
    )
    reconstructor = run_tool_extraction_streaming(
        hy_v4_tool_parser,
        [delta],
        request=mock_request,
        assert_one_tool_per_delta=False,
    )
    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
