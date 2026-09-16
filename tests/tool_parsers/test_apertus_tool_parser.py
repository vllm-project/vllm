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
    tokenizer.get_vocab.return_value = {TOOL_CALLS_PREFIX: 100, TOOL_CALLS_SUFFIX: 101}
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
            '<|tools_prefix|>[{"get_weather": '
            '{"location": "San Francisco", '
            '"unit": "celsius"}}]<|tools_suffix|>'
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
            '<|tools_prefix|>[{"get_weather": '
            '{"location": "London"}}, '
            '{"get_time": {"location": "London"}}]<|tools_suffix|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"complex_function": '
            '{"nested": {"inner": "value"}, '
            '"list": ["a", "b"]}}]<|tools_suffix|>'
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

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_missing_tool_suffix(self, parser, mock_request):
        model_output = (
            '<|tools_prefix|>[{"get_weather": '
            '{"location": "San Francisco", "unit": "celsius"}}]'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestStreamingExtraction:
    def _simulate_streaming(
        self, parser: ApertusToolParser, mock_request: Any, chunks: list[str]
    ) -> list[tuple[Any, str]]:
        results: list[tuple[Any, str]] = []
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            # Simulate a token ID sequence matching the chunk progression
            delta_token_ids: list[int] = [0] * max(1, len(chunk) // 4)
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

    def _collect_tool_calls(self, results) -> dict[int, dict[str, Any]]:
        """Properly tracks and concatenates streamed tool arguments by their Index."""
        tool_calls = {}
        for delta, _ in results:
            if not delta or not getattr(delta, "tool_calls", None):
                continue

            for tc in delta.tool_calls:
                idx = (
                    tc.get("index", 0)
                    if isinstance(tc, dict)
                    else getattr(tc, "index", 0)
                )
                func = (
                    tc.get("function", {})
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                )
                if not func:
                    continue

                name = (
                    func.get("name")
                    if isinstance(func, dict)
                    else getattr(func, "name", None)
                )
                args = (
                    func.get("arguments")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", None)
                )

                if idx not in tool_calls:
                    tool_calls[idx] = {"name": "", "arguments": ""}

                if name:
                    tool_calls[idx]["name"] += name
                if args:
                    tool_calls[idx]["arguments"] += args

        return tool_calls

    def _collect_content(self, results) -> str:
        """Collects generated normal text outside of the tool calls."""
        return "".join(
            delta.content
            for delta, _ in results
            if delta and getattr(delta, "content", None)
        )

    def test_basic_streaming_single_tool(self, parser, mock_request):
        chunks = [
            "<|tools_prefix|>",
            '[{"get_weather": ',
            '{"location": "Paris, ',
            'France"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        tcs = self._collect_tool_calls(results)

        assert len(tcs) == 1
        assert tcs[0]["name"] == "get_weather"
        assert json.loads(tcs[0]["arguments"]) == {"location": "Paris, France"}

    def test_streaming_missing_tool_suffix(self, parser, mock_request):
        chunks = [
            "<|tools_prefix|>",
            '[{"get_weather": ',
            '{"location": "Paris, ',
            'France"}}]',
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        tcs = self._collect_tool_calls(results)

        assert len(tcs) == 1
        assert tcs[0]["name"] == "get_weather"
        assert json.loads(tcs[0]["arguments"]) == {"location": "Paris, France"}

    def test_streaming_partial_tag_buffering_missing_tool_suffix(
        self, parser, mock_request
    ):
        chunks = ["Content", "<|tools_", "prefix|>", '[{"f": ', '{"a": 1}}]']

        results = self._simulate_streaming(parser, mock_request, chunks)
        content = self._collect_content(results)

        assert "Content" in content
        assert "<|tools_prefix|>" not in content
        assert "<|tools_suffix|>" not in content

        tcs = self._collect_tool_calls(results)

        assert len(tcs) == 1
        assert tcs[0]["name"] == "f"
        assert json.loads(tcs[0]["arguments"]) == {"a": 1}

    def test_streaming_multi_tool(self, parser, mock_request):
        chunks = [
            "<|tools_prefix|>",
            '[{"get_weather": {"location": "Tokyo"}}',
            ', {"get_time": {"location": "Tokyo"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        tcs = self._collect_tool_calls(results)

        assert len(tcs) == 2
        assert tcs[0]["name"] == "get_weather"
        assert json.loads(tcs[0]["arguments"]) == {"location": "Tokyo"}
        assert tcs[1]["name"] == "get_time"
        assert json.loads(tcs[1]["arguments"]) == {"location": "Tokyo"}

    def test_streaming_text_before_tool_call(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tools_prefix|>",
            '[{"get_weather": {"location": "London"}}]',
            "<|tools_suffix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        content = self._collect_content(results)

        assert content.strip() == "Let me check the weather."
        tcs = self._collect_tool_calls(results)

        assert len(tcs) == 1
        assert tcs[0]["name"] == "get_weather"
        assert json.loads(tcs[0]["arguments"]) == {"location": "London"}

    def test_streaming_partial_tag_buffering(self, parser, mock_request):
        chunks = [
            "Content",
            "<|tools_",
            "prefix|>",
            '[{"f": {"a": 1}}]',
            "<|tools_suf",
            "fix|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        content = self._collect_content(results)

        assert "Content" in content
        assert "<|tools_prefix|>" not in content
        assert "<|tools_suffix|>" not in content

        tc = self._collect_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "f"
        assert json.loads(tc[0]["arguments"]) == {"a": 1}

    # ---------------------------------------------------------------------------
    # Edge Cases: Multi-Token Prediction (MTP) & vLLM Chunking Anomalies
    # ---------------------------------------------------------------------------

    def test_mtp_streaming_massive_chunk(self, parser, mock_request):
        """Simulates MTP predicting text, tool calls,
        and trailing text all in a single chunk."""
        chunks = [
            "Sure! "
            '<|tools_prefix|>[{"get_weather": {"location": "London"}}]<|tools_suffix|>'
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        content = self._collect_content(results)
        assert "Sure! " in content

        tc = self._collect_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "London"}

    def test_mtp_streaming_multiple_tools_burst(self, parser, mock_request):
        """Simulates MTP predicting an array of multiple tools in one single chunk."""
        chunks = [
            '<|tools_prefix|>[{"get_weather": '
            '{"location": "London"}}, '
            '{"get_time": {"location": "Paris"}}]<|tools_suffix|>'
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        tc = self._collect_tool_calls(results)
        assert len(tc) == 2
        assert tc[0]["name"] == "get_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "London"}
        assert tc[1]["name"] == "get_time"
        assert json.loads(tc[1]["arguments"]) == {"location": "Paris"}

    def test_mtp_streaming_skip_and_catch_up(self, parser, mock_request):
        """Simulates MTP chunks that jump over entire tools
        (e.g., from middle of tool 1 to middle of tool 3)."""
        chunks = [
            '<|tools_prefix|>[{"t1": {"a": 1}',
            '}, {"t2": {"b": 2}}, {"t3": {"c": 3',
            "}}]<|tools_suffix|>",
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        tc = self._collect_tool_calls(results)
        assert len(tc) == 3
        assert tc[0]["name"] == "t1"
        assert json.loads(tc[0]["arguments"]) == {"a": 1}
        assert tc[1]["name"] == "t2"
        assert json.loads(tc[1]["arguments"]) == {"b": 2}
        assert tc[2]["name"] == "t3"
        assert json.loads(tc[2]["arguments"]) == {"c": 3}

    def test_vllm_streaming_character_by_character(self, parser, mock_request):
        """Simulates worst-case vLLM fragmentation where
        chunks arrive character-by-character."""
        text = (
            'Hi <|tools_prefix|>[{"get_weather": '
            '{"location": "London"}}]<|tools_suffix|> '
        )
        chunks = list(text)
        results = self._simulate_streaming(parser, mock_request, chunks)

        content = self._collect_content(results)
        assert "Hi" in content

        tc = self._collect_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "London"}

    def test_vllm_streaming_empty_deltas(self, parser, mock_request):
        """Simulates vLLM stream producing empty string chunks
        (e.g., hidden tokens or artifacts)."""
        chunks = [
            "Wait",
            "",
            "<|tools_prefix|>",
            "",
            '[{"get_weather": ',
            "",
            '{"location": "London"}}]',
            "<|tools_suffix|>",
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        content = self._collect_content(results)
        assert content == "Wait"

        tc = self._collect_tool_calls(results)
        assert len(tc) == 1
        assert tc[0]["name"] == "get_weather"
        assert json.loads(tc[0]["arguments"]) == {"location": "London"}
