# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="mcp__gmail-cleanup__list_filters",
                parameters={},
            ),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


class TestGlm47ExtractToolCalls:
    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


def _stream_chunks(parser, request, chunks):
    current_text = ""
    deltas = []
    for chunk in chunks:
        current_text += chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta and delta.tool_calls:
            deltas.extend(delta.tool_calls)
    return deltas


def _tool_call_args_from_deltas(deltas, index):
    return "".join(
        tool_call.function.arguments or ""
        for tool_call in deltas
        if tool_call.index == index and tool_call.function
    )


class TestGlm47Streaming:
    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        deltas = _stream_chunks(glm47_tool_parser, mock_request, chunks)

        assert glm47_tool_parser.prev_tool_call_arr[0]["name"] == "get_current_date"
        assert json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"]) == {}
        assert any(
            tool_call.function
            and tool_call.function.name == "get_current_date"
            and tool_call.index == 0
            for tool_call in deltas
        )
        assert json.loads(_tool_call_args_from_deltas(deltas, 0)) == {}

    def test_no_args_mcp_tool_name(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "mcp__gmail-cleanup__list_filters",
            "</tool_call>",
        ]
        deltas = _stream_chunks(glm47_tool_parser, mock_request, chunks)

        assert (
            glm47_tool_parser.prev_tool_call_arr[0]["name"]
            == "mcp__gmail-cleanup__list_filters"
        )
        assert json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"]) == {}
        assert any(
            tool_call.function
            and tool_call.function.name == "mcp__gmail-cleanup__list_filters"
            and tool_call.index == 0
            for tool_call in deltas
        )
        assert json.loads(_tool_call_args_from_deltas(deltas, 0)) == {}

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        _stream_chunks(glm47_tool_parser, mock_request, chunks)
        args = json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"])
        assert args["city"] == "Beijing"
