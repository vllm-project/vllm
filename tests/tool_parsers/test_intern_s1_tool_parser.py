# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.intern_s1_tool_parser import _build_initial_arguments_delta

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def tokenizer() -> TokenizerLike:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<|action_start|>": 92540,
        "<|plugin|>": 92541,
        "<|action_end|>": 92542,
        "<think>": 92543,
        "</think>": 92544,
    }
    return tokenizer


@pytest.fixture
def tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={"type": "object", "properties": {}},
            )
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_time",
                parameters={"type": "object", "properties": {}},
            )
        ),
    ]


@pytest.fixture
def chat_request(tools: list[ChatCompletionToolsParam]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=tools,
        tool_choice="auto",
    )


@pytest.fixture
def tool_parser(
    tokenizer: TokenizerLike,
    tools: list[ChatCompletionToolsParam],
):
    return ToolParserManager.get_tool_parser("intern-s1")(tokenizer, tools)


def test_intern_s1_tool_parser_registered():
    parser_cls = ToolParserManager.get_tool_parser("intern-s1")
    assert parser_cls.__name__ == "InternS1ToolParser"


def test_nonstreaming_supports_spaced_special_tokens(tool_parser, chat_request):
    content, tool_calls = run_tool_extraction(
        tool_parser,
        '<|action_start|> <|plugin|>\n{"name": "get_weather", '
        '"parameters": {"city": "Tokyo"}}\n<|action_end|>',
        request=chat_request,
        streaming=False,
    )

    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_nonstreaming_supports_multiple_action_blocks(tool_parser, chat_request):
    content, tool_calls = run_tool_extraction(
        tool_parser,
        '<|action_start|><|plugin|>{"name": "get_weather", '
        '"parameters": {"city": "Tokyo"}}<|action_end|>\n'
        '<|action_start|><|plugin|>{"name": "get_time", '
        '"parameters": {"timezone": "Asia/Tokyo"}}<|action_end|>\n'
        "Visible answer.",
        request=chat_request,
        streaming=False,
    )

    assert content is not None
    assert content.strip() == "Visible answer."
    assert "<|action_start|>" not in content
    assert [tool.function.name for tool in tool_calls] == [
        "get_weather",
        "get_time",
    ]


def test_nonstreaming_gracefully_handles_malformed_json(
    tool_parser,
    chat_request,
):
    content, tool_calls = run_tool_extraction(
        tool_parser,
        '<|action_start|><|plugin|>{"name": "get_weather", "parameters": {'
        "<|action_end|>",
        request=chat_request,
        streaming=False,
    )

    assert content is not None
    assert "<|action_start|>" in content
    assert tool_calls == []


def test_build_initial_arguments_delta_falls_back_to_full_arguments():
    arguments_json = '{"city": "Tokyo"}'

    assert (
        _build_initial_arguments_delta(arguments_json, "<|plugin|>") == arguments_json
    )
