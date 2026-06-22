# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.minimax_m2 import (
    THINK_END,
    TOOL_CALL_END,
    TOOL_CALL_START,
    MinimaxM2Parser,
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            THINK_END: 99,
            TOOL_CALL_START: 100,
            TOOL_CALL_END: 101,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return MinimaxM2Parser(mock_tokenizer)


def make_tools(*names: str):
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name=name,
                parameters={
                    "type": "object",
                    "properties": {},
                },
            ),
        )
        for name in names
    ]


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "</think>This is a regular response without tool calls.",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "This is a regular response without tool calls."

    def test_single_tool_call(self, parser, mock_request):
        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Seattle</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "city": "Seattle",
        }

    def test_multiple_invokes(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "<minimax:tool_call>"
            '<invoke name="search"><parameter name="q">OpenAI</parameter></invoke>'
            '<invoke name="search"><parameter name="q">vLLM</parameter></invoke>'
            "</minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == ["search", "search"]
        assert json.loads(result.tool_calls[0].function.arguments) == {"q": "OpenAI"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"q": "vLLM"}

    def test_schema_type_coercion(self, mock_tokenizer, mock_request):
        tools = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="forecast",
                    parameters={
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer"},
                            "include_hourly": {"type": "boolean"},
                        },
                    },
                ),
            )
        ]
        parser = MinimaxM2Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="forecast">'
            '<parameter name="days">5</parameter>'
            '<parameter name="include_hourly">true</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert json.loads(result.tool_calls[0].function.arguments) == {
            "days": 5,
            "include_hourly": True,
        }

    def test_invalid_tool_name_is_rejected(self, mock_tokenizer, mock_request):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="img_gen">'
            '<parameter name="prompt">a cat</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is False
        assert result.tool_calls == []

    def test_mixed_tool_names_only_return_valid(self, mock_tokenizer, mock_request):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            "<minimax:tool_call>"
            '<invoke name="img_gen"><parameter name="prompt">cat</parameter></invoke>'
            '<invoke name="search"><parameter name="query">news</parameter></invoke>'
            "</minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == ["search"]
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": "news",
        }


class TestStreaming:
    def test_streaming_single_tool_call(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="get_weather">',
                '<parameter name="city">Seattle</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Seattle",
        }

    def test_streaming_multiple_invokes(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="a"><parameter name="x">1</parameter></invoke>',
                '<invoke name="b"><parameter name="y">2</parameter></invoke>',
                "</minimax:tool_call>",
            ],
        )

        tool_names = [
            tc.function.name
            for delta, _ in results
            if delta and delta.tool_calls
            for tc in delta.tool_calls
            if tc.function and tc.function.name
        ]
        assert tool_names == ["a", "b"]

    def test_streaming_invoke_prefix_split_before_quote(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                "<invoke name=",
                '"get_weather">',
                '<parameter name="city">Seattle</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Seattle",
        }

    def test_streaming_invalid_tool_name_is_rejected(
        self, mock_tokenizer, mock_request
    ):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="img_gen">',
                '<parameter name="prompt">cat</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) is None
        assert collect_tool_arguments(results) == ""


class TestReasoning:
    def test_extract_reasoning_without_start_token(self, parser, mock_request):
        reasoning, content = parser.extract_reasoning(
            "This is reasoning</think>This is content",
            mock_request,
        )

        assert reasoning == "This is reasoning"
        assert content == "This is content"

    def test_extract_reasoning_without_end_token(self, parser, mock_request):
        reasoning, content = parser.extract_reasoning(
            "This is still reasoning",
            mock_request,
        )

        assert reasoning == "This is still reasoning"
        assert content is None

    def test_extract_content_ids_without_end_token(self, parser):
        assert parser.extract_content_ids([1, 2, 3]) == []

    def test_extract_content_ids_after_end_token(self, parser):
        assert parser.extract_content_ids([1, 99, 2, 3]) == [2, 3]
