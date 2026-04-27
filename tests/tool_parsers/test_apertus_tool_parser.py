# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.apertus_tool_parser import (
    TOOL_CALLS_PREFIX,
    TOOL_CALLS_SUFFIX,
    ApertusToolParser,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    # Include the tool call tokens in the vocab for the parser
    tokenizer.get_vocab.return_value = {
        TOOL_CALLS_PREFIX: 100,
        TOOL_CALLS_SUFFIX: 101
    }
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return ApertusToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


# ---------------------------------------------------------------------------
# Non-streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:

    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"get_weather": {"location": "London"}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"get_weather": {"location": "San Francisco", "unit": "celsius"}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            '<|tools_prefix|>[{"get_weather": {"location": "Paris"}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"get_weather": {"location": "London"}}, {"get_time": {"location": "London"}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"complex_function": {"nested": {"inner": "value"}, "list": ["a", "b"]}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "complex_function"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"inner": "value"}, "list": ["a", "b"]}

    def test_incomplete_tool_call(self, parser, mock_request):
        model_output = '<|tools_prefix|>[{"get_weather": {"location": "London"}'
        result = parser.extract_tool_calls(model_output, mock_request)

        # Incomplete — no <|tools_suffix|> end marker, regex won't match
        assert result.tools_called is False
        assert result.content == model_output


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestStreamingExtraction:

    def _simulate_streaming(self, parser: ApertusToolParser, mock_request: Any,
                            chunks: list[str]) -> list[tuple[Any, str]]:
        results: list[tuple[Any, str]] = []
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            delta_token_ids: list[int] = [0]
            current_token_ids = previous_token_ids + delta_token_ids

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_token_ids),
                request=mock_request,
            )
            results.append((delta, current_text))
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        return results

    def _collect_arguments(self, results):
        args_text = ""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(
                        tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        arg = func.get("arguments", "")
                    else:
                        arg = getattr(func, "arguments", "") or ""
                    if arg:
                        args_text += arg
        return args_text

    def _collect_function_name(self, results):
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(
                        tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        name = func.get("name")
                    else:
                        name = getattr(func, "name", None)
                    if name:
                        return name
        return None

    def test_basic_streaming_single_tool(self, parser, mock_request):
        chunks = [
            "<|tools_prefix|>",
            '[{"get_weather": ',
            '{"location": "Paris, ',
            'France"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Paris, France"}

    def test_streaming_multi_tool(self, parser, mock_request):
        chunks = [
            "<|tools_prefix|>",
            '[{"get_weather": {"location": "Tokyo"}}',
            ', {"get_time": {"location": "Tokyo"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # Check for first tool
        assert any(
            delta and delta.tool_calls
            and any(tc.function.name == "get_weather" for tc in delta.tool_calls)
            for delta, _ in results)

        # Check for second tool
        assert any(
            delta and delta.tool_calls
            and any(tc.function.name == "get_time" for tc in delta.tool_calls)
            for delta, _ in results)

        # Check arguments
        args_text = self._collect_arguments(results)
        # Note: Since we have multiple tool calls, we'd need to collect per index if we want to be precise,
        # but _collect_arguments joins them all.
        # Index 0: {"location": "Tokyo"}
        # Index 1: {"location": "Tokyo"}
        # The joint string should contain them.
        assert '{"location": "Tokyo"}' in args_text

    def test_streaming_text_before_tool_call(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tools_prefix|>",
            '[{"get_weather": {"location": "London"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

    def test_streaming_partial_tag_buffering(self, parser, mock_request):
        chunks = [
            "Content",
            "<|tools_",
            "prefix|>",
            '[{"f": {"a": 1}}]',
            "<|tools_suf",
            "fix|>",
            "More content"
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = [d.content for d, _ in results if d and d.content]
        full_content = "".join(content_parts)
        assert "Content" in full_content
        assert "More content" in full_content
        assert "<|tools_prefix|>" not in full_content
        assert "<|tools_suffix|>" not in full_content
