# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.gemma4_tool_parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4ToolParser,
    _parse_gemma4_args,
    _parse_gemma4_array,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    # Include the tool call start token in the vocab for the parser
    tokenizer.get_vocab.return_value = {TOOL_CALL_START: 48, TOOL_CALL_END: 49}
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return Gemma4ToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


# ---------------------------------------------------------------------------
# Unit tests for _parse_gemma4_args (shared parser logic)
# ---------------------------------------------------------------------------


class TestParseGemma4Args:
    def test_empty_string(self):
        assert _parse_gemma4_args("") == {}

    def test_whitespace_only(self):
        assert _parse_gemma4_args("   ") == {}

    def test_single_string_value(self):
        result = _parse_gemma4_args('location:<|"|>Paris<|"|>')
        assert result == {"location": "Paris"}

    def test_string_value_with_comma(self):
        result = _parse_gemma4_args('location:<|"|>Paris, France<|"|>')
        assert result == {"location": "Paris, France"}

    def test_multiple_string_values(self):
        result = _parse_gemma4_args(
            'location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>'
        )
        assert result == {"location": "San Francisco", "unit": "celsius"}

    def test_integer_value(self):
        result = _parse_gemma4_args("count:42")
        assert result == {"count": 42}

    def test_float_value(self):
        result = _parse_gemma4_args("score:3.14")
        assert result == {"score": 3.14}

    def test_boolean_true(self):
        result = _parse_gemma4_args("flag:true")
        assert result == {"flag": True}

    def test_boolean_false(self):
        result = _parse_gemma4_args("flag:false")
        assert result == {"flag": False}

    def test_mixed_types(self):
        result = _parse_gemma4_args(
            'name:<|"|>test<|"|>,count:42,active:true,score:3.14'
        )
        assert result == {
            "name": "test",
            "count": 42,
            "active": True,
            "score": 3.14,
        }

    def test_nested_object(self):
        result = _parse_gemma4_args('nested:{inner:<|"|>value<|"|>}')
        assert result == {"nested": {"inner": "value"}}

    def test_array_of_strings(self):
        result = _parse_gemma4_args('items:[<|"|>a<|"|>,<|"|>b<|"|>]')
        assert result == {"items": ["a", "b"]}

    def test_unterminated_string(self):
        """Unterminated strings should take everything after the delimiter."""
        result = _parse_gemma4_args('key:<|"|>unterminated')
        assert result == {"key": "unterminated"}

    def test_empty_value(self):
        """Key with no value after colon."""
        result = _parse_gemma4_args("key:")
        assert result == {"key": ""}


class TestParseGemma4Array:
    def test_string_array(self):
        result = _parse_gemma4_array('<|"|>a<|"|>,<|"|>b<|"|>')
        assert result == ["a", "b"]

    def test_empty_array(self):
        result = _parse_gemma4_array("")
        assert result == []

    def test_bare_values(self):
        result = _parse_gemma4_array("42,true,3.14")
        assert result == [42, True, 3.14]


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
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>San Francisco<|"|>,'
            'unit:<|"|>celsius<|"|>}'
            "<tool_call|>"
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
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
            "<tool_call|>"
            '<|tool_call>call:get_time{location:<|"|>London<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:complex_function{"
            'nested:{inner:<|"|>value<|"|>},'
            'list:[<|"|>a<|"|>,<|"|>b<|"|>]}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "complex_function"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"inner": "value"}, "list": ["a", "b"]}

    def test_tool_call_with_number_and_boolean(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:set_status{"
            "is_active:true,"
            "count:42,"
            "score:3.14}"
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "set_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"is_active": True, "count": 42, "score": 3.14}

    def test_incomplete_tool_call(self, parser, mock_request):
        model_output = '<|tool_call>call:get_weather{location:<|"|>London'
        result = parser.extract_tool_calls(model_output, mock_request)

        # Incomplete — no <tool_call|> end marker, regex won't match
        assert result.tools_called is False
        assert result.content == model_output

    def test_hyphenated_function_name(self, parser, mock_request):
        """Ensure function names with hyphens are parsed correctly."""
        model_output = (
            '<|tool_call>call:get-weather{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get-weather"

    def test_dotted_function_name(self, parser, mock_request):
        """Ensure function names with dots are parsed correctly."""
        model_output = (
            '<|tool_call>call:weather.get{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "weather.get"

    def test_no_arguments(self, parser, mock_request):
        """Tool calls with empty arguments."""
        model_output = "<|tool_call>call:get_status{}<tool_call|>"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestStreamingExtraction:
    """Tests for the streaming tool call extraction.

    These simulate the token-by-token streaming that vLLM performs,
    feeding incremental text to extract_tool_calls_streaming() and
    verifying that the accumulated argument deltas form valid JSON.
    """

    def _simulate_streaming(
        self, parser: Gemma4ToolParser, mock_request: Any, chunks: list[str]
    ) -> list[tuple[Any, str]]:
        """Feed chunks through the streaming parser and collect results.

        Returns a list of (delta_message, accumulated_text) tuples.
        """
        results: list[tuple[Any, str]] = []
        previous_text = ""
        previous_token_ids = []

        for chunk in chunks:
            current_text = previous_text + chunk
            # Use token ID 48 for tool_call start, 49 for end, 0 otherwise
            delta_token_ids = []
            if TOOL_CALL_START in chunk:
                delta_token_ids.append(48)
            elif TOOL_CALL_END in chunk:
                delta_token_ids.append(49)
            else:
                delta_token_ids.append(0)

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
        """Collect all argument deltas from streaming results into one string."""
        args_text = ""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        arg = func.get("arguments", "")
                    else:
                        arg = getattr(func, "arguments", "") or ""
                    if arg:
                        args_text += arg
        return args_text

    def _collect_function_name(self, results):
        """Extract the function name from streaming results."""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        name = func.get("name")
                    else:
                        name = getattr(func, "name", None)
                    if name:
                        return name
        return None

    def test_basic_streaming_single_tool(self, parser, mock_request):
        """Simulate the exact streaming scenario from the bug report.

        Model generates:
        <|tool_call>call:get_weather{location:<|"|>Paris, France<|"|>}<tool_call|>

        Expected: arguments should be valid JSON {"location": "Paris, France"}
        """
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            ", France",
            '<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # Verify function name
        name = self._collect_function_name(results)
        assert name == "get_weather", f"Expected 'get_weather', got '{name}'"

        # Verify arguments form valid JSON
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Paris, France"}

    def test_streaming_multi_arg(self, parser, mock_request):
        """Streaming with multiple arguments."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Tokyo<|"|>,',
            'unit:<|"|>celsius<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Tokyo", "unit": "celsius"}

    def test_streaming_no_extra_brace(self, parser, mock_request):
        """Verify the closing } is NOT leaked into arguments (Bug #2)."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text

        # The args text must be valid JSON (no extra })
        parsed = json.loads(args_text)
        assert parsed == {"location": "London"}

        # Specifically assert no double-brace
        assert args_text.count("}") <= 1, (
            f"Arguments contain extra closing brace: {args_text!r}"
        )

    def test_streaming_no_unquoted_keys(self, parser, mock_request):
        """Verify keys are properly quoted in JSON (Bug #1)."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Must start with { and contain quoted key
        assert args_text.lstrip().startswith("{"), (
            f"Arguments don't start with '{{': {args_text!r}"
        )
        assert '"location"' in args_text, (
            f"Key 'location' not properly quoted: {args_text!r}"
        )

    def test_streaming_name_no_call_prefix(self, parser, mock_request):
        """Verify function name has no 'call:' prefix."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        name = self._collect_function_name(results)
        assert name == "get_weather"
        assert not name.startswith("call:"), f"Name has 'call:' prefix: {name!r}"

    def test_streaming_text_before_tool_call(self, parser, mock_request):
        """Text before tool call should be emitted as content."""
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # First chunks should be content
        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

    def test_streaming_numeric_args(self, parser, mock_request):
        """Streaming with numeric and boolean argument values."""
        chunks = [
            "<|tool_call>",
            "call:set_config{",
            "count:42,",
            "active:true}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        if args_text:
            parsed_args = json.loads(args_text)
            assert parsed_args["count"] == 42
            assert parsed_args["active"] is True

    def test_streaming_empty_args(self, parser, mock_request):
        """Tool call with no arguments."""
        chunks = [
            "<|tool_call>",
            "call:get_status{}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        name = self._collect_function_name(results)
        assert name == "get_status"
