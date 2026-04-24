# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the HYV3 tool call parser."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.hy_v3_tool_parser import HYV3ToolParser

parser_name = "hy_v3"
MODEL = "tencent/Hy3-preview"


@pytest.fixture(scope="module")
def hy_v3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def hy_v3_tool_parser(hy_v3_tokenizer):
    return HYV3ToolParser(hy_v3_tokenizer)


@pytest.fixture
def mock_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
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
    request.tool_choice = "auto"
    return request


class TestHYV3ExtractToolCalls:
    def test_no_tool_call(self, hy_v3_tool_parser, mock_request):
        out = "This is a plain response."
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, hy_v3_tool_parser, mock_request):
        out = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, hy_v3_tool_parser, mock_request):
        out = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, hy_v3_tool_parser, mock_request):
        out = (
            "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing"
            "</arg_value><arg_key>date</arg_key><arg_value>2026-03-30</arg_value></tool_call></tool_calls>"
        )
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "Beijing",
            "date": "2026-03-30",
        }

    def test_args_with_newlines(self, hy_v3_tool_parser, mock_request):
        out = (
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing"
            "</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>"
        )
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "Beijing",
            "date": "2026-03-30",
        }

    def test_content_before(self, hy_v3_tool_parser, mock_request):
        out = "Checking.<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, hy_v3_tool_parser, mock_request):
        out = (
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing"
            "</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n"
            "<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>"
        )
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, hy_v3_tool_parser, mock_request):
        out = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = hy_v3_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _simulate_streaming(
    parser: HYV3ToolParser,
    deltas: list[str],
    request: ChatCompletionRequest,
) -> list[DeltaMessage | None]:
    results: list[DeltaMessage | None] = []
    previous_text = ""
    previous_token_ids: list[int] = []
    vocab = parser.vocab
    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_token_ids = [tid for tok, tid in vocab.items() if tok in delta_text]
        current_token_ids = previous_token_ids + delta_token_ids
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )
        results.append(result)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return results


def _collect_streaming_tool_calls(results: list[DeltaMessage | None]) -> list[dict]:
    tool_calls: dict[int, dict] = {}
    for result in results:
        if result is None or not result.tool_calls:
            continue
        for tc in result.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "name": tc.function.name or "",
                    "arguments": tc.function.arguments or "",
                }
            else:
                if tc.function.name:
                    tool_calls[idx]["name"] += tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments
    return [tool_calls[i] for i in sorted(tool_calls.keys())]


def _collect_streaming_content(results: list[DeltaMessage | None]) -> str:
    parts = []
    for result in results:
        if result is not None and result.content:
            parts.append(result.content)
    return "".join(parts)


class TestHYV3ExtractToolCallsStreaming:
    def test_no_tool_call_streaming(self, hy_v3_tool_parser, mock_request):
        deltas = ["This is ", "a plain ", "response."]
        results = _simulate_streaming(hy_v3_tool_parser, deltas, mock_request)
        content = _collect_streaming_content(results)
        assert content == "This is a plain response."
        assert len(_collect_streaming_tool_calls(results)) == 0

    def test_zero_arg_streaming(self, hy_v3_tool_parser, mock_request):
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(hy_v3_tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_current_date"
        assert json.loads(tc[0]["arguments"]) == {}

    def test_args_streaming(self, hy_v3_tool_parser, mock_request):
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(hy_v3_tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1 and tc[0]["name"] == "get_weather"
        assert json.loads(tc[0]["arguments"]) == {
            "city": "Beijing",
            "date": "2026-03-30",
        }

    def test_content_before_streaming(self, hy_v3_tool_parser, mock_request):
        deltas = [
            "Checking.",
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(hy_v3_tool_parser, deltas, mock_request)
        assert "Checking." in _collect_streaming_content(results)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1 and tc[0]["name"] == "get_current_date"

    def test_multiple_streaming(self, hy_v3_tool_parser, mock_request):
        deltas = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Hangzhou</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        results = _simulate_streaming(hy_v3_tool_parser, deltas, mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 2
        assert json.loads(tc[0]["arguments"])["city"] == "Beijing"
        assert json.loads(tc[1]["arguments"])["city"] == "Hangzhou"

    def test_all_in_one_delta_streaming(self, hy_v3_tool_parser, mock_request):
        out = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        results = _simulate_streaming(hy_v3_tool_parser, [out], mock_request)
        tc = _collect_streaming_tool_calls(results)
        assert len(tc) == 1 and tc[0]["name"] == "get_current_date"
        assert json.loads(tc[0]["arguments"]) == {}
