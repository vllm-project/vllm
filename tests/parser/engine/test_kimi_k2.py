# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the KimiK2Parser engine implementation.

Covers non-streaming tool extraction, reasoning extraction,
streaming tool calls, thinking-disabled mode, and is_reasoning_end.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.kimi_k2 import (
    THINK_END,
    THINK_START,
    TOOL_ARG_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_SECTION_END,
    TOOL_SECTION_START,
    KimiK2Parser,
)

_THINK_START_ID = 100
_THINK_END_ID = 101
_TOOL_SECTION_START_ID = 200
_TOOL_CALL_START_ID = 201
_TOOL_CALL_END_ID = 202


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            THINK_START: _THINK_START_ID,
            THINK_END: _THINK_END_ID,
            TOOL_SECTION_START: _TOOL_SECTION_START_ID,
            TOOL_CALL_START: _TOOL_CALL_START_ID,
            TOOL_CALL_END: _TOOL_CALL_END_ID,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return KimiK2Parser(mock_tokenizer)


@pytest.fixture
def parser_no_think(mock_tokenizer):
    return KimiK2Parser(mock_tokenizer, chat_template_kwargs={"thinking": False})


def _tool(tool_id: str, args: str) -> str:
    return f"{TOOL_CALL_START}{tool_id} {TOOL_ARG_START}{args}{TOOL_CALL_END}"


def _wrap(*tool_strs: str) -> str:
    return TOOL_SECTION_START + "".join(tool_strs) + TOOL_SECTION_END


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        output = f"{THINK_START}Let me think.{THINK_END}Here is the answer."
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content is not None
        assert "Here is the answer." in result.content

    def test_single_tool_call(self, parser, mock_request):
        output = f"{THINK_START}I need to look this up.{THINK_END}" + _wrap(
            _tool("functions.get_weather:0", '{"city": "Tokyo"}')
        )
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.id == "functions.get_weather:0"
        assert json.loads(tc.function.arguments) == {"city": "Tokyo"}

    def test_parallel_tool_calls(self, parser, mock_request):
        output = f"{THINK_START}Comparing cities.{THINK_END}" + _wrap(
            _tool("functions.get_weather:0", '{"city": "Tokyo"}'),
            _tool("functions.get_weather:1", '{"city": "Paris"}'),
        )
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Tokyo"}
        assert result.tool_calls[1].function.name == "get_weather"
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "Paris"}

    def test_three_different_tools(self, parser, mock_request):
        output = f"{THINK_START}Multiple tasks.{THINK_END}" + _wrap(
            _tool("functions.get_weather:0", '{"city": "NYC"}'),
            _tool("functions.get_news:1", '{"topic": "tech"}'),
            _tool("functions.send_email:2", '{"to": "a@b.com"}'),
        )
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 3
        names = [tc.function.name for tc in result.tool_calls]
        assert names == ["get_weather", "get_news", "send_email"]

    def test_implicit_reasoning_end_via_tool_section(self, parser, mock_request):
        """<|tool_calls_section_begin|> ends reasoning even without </think>."""
        output = f"{THINK_START}No closing think tag." + _wrap(
            _tool("functions.search:0", '{"q": "vllm"}')
        )
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_no_functions_prefix_in_id(self, parser, mock_request):
        output = f"{THINK_START}Sure.{THINK_END}" + _wrap(
            _tool("get_weather:0", '{"city": "NYC"}')
        )
        result = parser.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].id == "get_weather:0"

    def test_special_tokens_do_not_leak_into_content(self, parser, mock_request):
        output = f"{THINK_START}Think.{THINK_END}" + _wrap(
            _tool("functions.fn:0", '{"k": "v"}')
        )
        result = parser.extract_tool_calls(output, mock_request)
        for marker in [
            THINK_START,
            THINK_END,
            TOOL_SECTION_START,
            TOOL_CALL_START,
            TOOL_ARG_START,
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]:
            if result.content:
                assert marker not in result.content


class TestReasoning:
    def test_extract_reasoning_basic(self, parser, mock_request):
        output = f"{THINK_START}My reasoning.{THINK_END}The answer."
        reasoning, content = parser.extract_reasoning(output, mock_request)
        assert reasoning is not None
        assert "My reasoning." in reasoning
        assert content is not None
        assert "The answer." in content

    def test_extract_reasoning_implicit_end(self, parser, mock_request):
        """<|tool_calls_section_begin|> ends reasoning implicitly."""
        output = f"{THINK_START}Thinking." + _wrap(
            _tool("functions.search:0", '{"q": "hi"}')
        )
        reasoning, content = parser.extract_reasoning(output, mock_request)
        assert reasoning is not None
        assert "Thinking." in reasoning

    def test_reasoning_streaming_basic(self, parser):
        chunks = [
            THINK_START,
            "First part. ",
            "Second part.",
            THINK_END,
            "Content after.",
        ]
        r, c = simulate_reasoning_streaming(parser, chunks)
        assert "First part." in r
        assert "Second part." in r
        assert "Content after." in c

    def test_reasoning_streaming_implicit_end(self, parser):
        """Reasoning ends via tool section token in streaming."""
        chunks = [THINK_START, "Some reasoning.", TOOL_SECTION_START]
        r, c = simulate_reasoning_streaming(parser, chunks)
        assert "Some reasoning." in r

    def test_think_start_absorbed(self, parser, mock_request):
        """<think> tag should not appear in reasoning output."""
        output = f"{THINK_START}Content.{THINK_END}Rest."
        reasoning, _ = parser.extract_reasoning(output, mock_request)
        assert reasoning is not None
        assert THINK_START not in reasoning
        assert THINK_END not in reasoning


class TestStreaming:
    def test_single_tool_call(self, parser, mock_request):
        chunks = [
            THINK_START,
            "Let me check.",
            THINK_END,
            TOOL_SECTION_START,
            TOOL_CALL_START,
            "functions.get_weather:0 ",
            TOOL_ARG_START,
            '{"city": "Tokyo"}',
            " ",
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        assert collect_function_name(results) == "get_weather"
        args = collect_tool_arguments(results)
        assert json.loads(args) == {"city": "Tokyo"}

    def test_incremental_args(self, parser, mock_request):
        """Arguments arriving in multiple chunks concatenate correctly."""
        chunks = [
            THINK_START,
            "Think.",
            THINK_END,
            TOOL_SECTION_START,
            TOOL_CALL_START,
            "functions.search:0 ",
            TOOL_ARG_START,
            '{"q":',
            ' "hello world"',
            "}",
            " ",
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        args = collect_tool_arguments(results)
        assert json.loads(args) == {"q": "hello world"}

    def test_parallel_tool_calls(self, parser, mock_request):
        chunks = [
            THINK_START,
            "Multi.",
            THINK_END,
            TOOL_SECTION_START,
            TOOL_CALL_START,
            "functions.get_weather:0 ",
            TOOL_ARG_START,
            '{"city": "A"} ',
            TOOL_CALL_END,
            TOOL_CALL_START,
            "functions.get_news:1 ",
            TOOL_ARG_START,
            '{"topic": "B"} ',
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        assert "get_weather" in names
        assert "get_news" in names

    def test_no_tools_plain_content(self, parser, mock_request):
        """Streaming without tool section yields content only."""
        chunks = [THINK_START, "Reasoning.", THINK_END, "Plain response."]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        content = collect_content(results)
        assert "Plain response." in content

    def test_markers_do_not_leak(self, parser, mock_request):
        """Special-token markers must not appear in content output."""
        chunks = [
            THINK_START,
            "Think.",
            THINK_END,
            TOOL_SECTION_START,
            TOOL_CALL_START,
            "functions.fn:0 ",
            TOOL_ARG_START,
            '{"k": "v"} ',
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        content = collect_content(results)
        for marker in [
            TOOL_SECTION_START,
            TOOL_CALL_START,
            TOOL_ARG_START,
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]:
            assert marker not in content


class TestThinkingDisabled:
    def test_plain_content(self, parser_no_think, mock_request):
        result = parser_no_think.extract_tool_calls(
            "Hello, how can I help?", mock_request
        )
        assert result.tools_called is False
        assert result.content == "Hello, how can I help?"

    def test_tool_call(self, parser_no_think, mock_request):
        output = _wrap(_tool("functions.get_weather:0", '{"city": "Paris"}'))
        result = parser_no_think.extract_tool_calls(output, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}

    def test_streaming_tool_call(self, parser_no_think, mock_request):
        chunks = [
            TOOL_SECTION_START,
            TOOL_CALL_START,
            "functions.search:0 ",
            TOOL_ARG_START,
            '{"q": "cats"} ',
            TOOL_CALL_END,
            TOOL_SECTION_END,
        ]
        results = simulate_tool_streaming(parser_no_think, mock_request, chunks)
        assert collect_function_name(results) == "search"
        args = collect_tool_arguments(results)
        assert json.loads(args) == {"q": "cats"}

    def test_is_reasoning_end_always_true(self, parser_no_think):
        assert parser_no_think.is_reasoning_end([]) is True
        assert parser_no_think.is_reasoning_end([1, 2, 3]) is True


class TestIsReasoningEnd:
    def test_think_end_returns_true(self, parser):
        assert parser.is_reasoning_end([_THINK_END_ID]) is True

    def test_tool_section_start_returns_true(self, parser):
        assert parser.is_reasoning_end([_TOOL_SECTION_START_ID]) is True

    def test_think_start_returns_false(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID]) is False

    def test_empty_ids_with_thinking_returns_false(self, parser):
        assert parser.is_reasoning_end([]) is False

    def test_think_end_after_start(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID, _THINK_END_ID]) is True

    def test_scan_is_reversed_start_after_end(self, parser):
        """Scanning in reverse: THINK_START found first → not ended."""
        assert parser.is_reasoning_end([_THINK_END_ID, _THINK_START_ID]) is False

    def test_tool_section_after_think_end(self, parser):
        ids = [_THINK_START_ID, _THINK_END_ID, _TOOL_SECTION_START_ID]
        assert parser.is_reasoning_end(ids) is True
