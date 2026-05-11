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
    TOOL_RESPONSE,
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
    # Include Gemma4 tool boundary tokens in the vocab for the parser.
    tokenizer.get_vocab.return_value = {
        TOOL_CALL_START: 48,
        TOOL_CALL_END: 49,
        TOOL_RESPONSE: 50,
    }

    def decode(token_ids, skip_special_tokens=False):
        token_map = {
            48: TOOL_CALL_START,
            49: TOOL_CALL_END,
            50: TOOL_RESPONSE,
            100: "call:first_tool{alpha:15",
            101: ",beta:1,gamma:1990}",
            102: "call:second_tool{delta:9,epsilon:30}",
        }
        return "".join(token_map.get(token_id, "") for token_id in token_ids)

    tokenizer.decode.side_effect = decode
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return Gemma4ToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.parallel_tool_calls = True
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

    def test_exponent_number_value(self):
        result = _parse_gemma4_args("score:1e3")
        assert result == {"score": 1000.0}

    def test_trailing_dot_float_partial_withheld(self):
        assert _parse_gemma4_args("score:108.", partial=True) == {}
        assert _parse_gemma4_args("score:108.,next:1", partial=True) == {}
        assert _parse_gemma4_args("score:108.") == {"score": 108.0}

    def test_boolean_true(self):
        result = _parse_gemma4_args("flag:true")
        assert result == {"flag": True}

    def test_boolean_false(self):
        result = _parse_gemma4_args("flag:false")
        assert result == {"flag": False}

    def test_null_value(self):
        # Bare `null` must parse as None (Python), not the string "null".
        # Without this, tool_choice=auto would emit `{"param": "null"}`
        # instead of `{"param": null}` for nullable tool parameters.
        result = _parse_gemma4_args("param:null")
        assert result == {"param": None}
        assert json.dumps(result) == '{"param": null}'

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

    def test_empty_value_partial_withheld(self):
        """Key with no value is withheld in partial mode to avoid premature emission."""
        result = _parse_gemma4_args("key:", partial=True)
        assert result == {}
        # also with a space after the colon
        result = _parse_gemma4_args("key: ", partial=True)
        assert result == {}

    def test_empty_value_after_other_keys_partial_withheld(self):
        """Trailing key with no value is withheld; earlier keys are kept."""
        result = _parse_gemma4_args('name:<|"|>test<|"|>,flag:', partial=True)
        assert result == {"name": "test"}

    @pytest.mark.timeout(5)
    def test_malformed_partial_array(self):
        result = _parse_gemma4_args(":[t:[]")
        assert isinstance(result, dict)


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

    def test_trailing_dot_float_partial_withheld(self):
        assert _parse_gemma4_array("108.", partial=True) == []
        assert _parse_gemma4_array("108.,109", partial=True) == []
        assert _parse_gemma4_array("108.") == [108.0]

    @pytest.mark.timeout(5)
    def test_string_element_with_closing_bracket(self):
        result = _parse_gemma4_array('[<|"|>a]b<|"|>,<|"|>c<|"|>],<|"|>tail<|"|>')
        assert result == [["a]b", "c"], "tail"]

    @pytest.mark.timeout(5)
    def test_stray_closing_bracket(self):
        result = _parse_gemma4_array("42,]trailing")
        assert result == [42]


# ---------------------------------------------------------------------------
# Non-streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you toalpha?"
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
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            # Use token ID 48 for tool_call start, 49 for end, 0 otherwise
            delta_token_ids: list[int] = []
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

    def _collect_tool_calls_by_index(self, results):
        calls: dict[int, dict[str, str | None]] = {}
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = calls.setdefault(tc.index, {"name": None, "arguments": ""})
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        name = func.get("name")
                        arg = func.get("arguments", "")
                    else:
                        name = getattr(func, "name", None)
                        arg = getattr(func, "arguments", "") or ""
                    if name:
                        entry["name"] = name
                    if arg:
                        entry["arguments"] += arg
        return calls

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

    def test_streaming_boolean_split_across_chunks(self, parser, mock_request):
        """Boolean value split across token boundaries must not corrupt JSON."""
        chunks = [
            "<|tool_call>",
            "call:search{input:{all:" + "true"[:3],
            "e}}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["input"]["all"] is True

    def test_streaming_false_split_across_chunks(self, parser, mock_request):
        """Boolean false split across chunks."""
        chunks = [
            "<|tool_call>",
            "call:set{flag:" + "false"[:4],
            "e}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["flag"] is False

    def test_streaming_number_split_across_chunks(self, parser, mock_request):
        """Number split across chunks must not change type."""
        chunks = [
            "<|tool_call>",
            "call:set{count:4",
            "2}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["count"] == 42

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

    def test_streaming_split_delimiter_no_invalid_json(self, parser, mock_request):
        """Partial <|"|> delimiter chars must not leak into streamed JSON.

        Reproduces the bug from https://github.com/vllm-project/vllm/issues/38946
        where a token boundary splits the string delimiter, leaving fragments
        like '<|' at the end of a parsed value which then corrupt the JSON.
        """
        chunks = [
            "<|tool_call>",
            "call:todowrite{",
            'content:<|"|>Buy milk<|',
            '"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        # Must be valid JSON — the original bug caused a JSON parse error
        parsed_args = json.loads(args_text)
        assert parsed_args["content"] == "Buy milk"

        # Ensure no raw delimiter fragments leaked into the JSON
        assert "<|" not in args_text, (
            f"Partial delimiter leaked into JSON: {args_text!r}"
        )

    def test_streaming_does_not_duplicate_plain_text_after_tool_call(
        self, parser, mock_request, monkeypatch
    ):
        """Buffered plain text after a tool call must not corrupt current_text."""
        captured_current_texts: list[str] = []
        original_extract_streaming = parser._extract_streaming

        def wrapped_extract_streaming(current_text, request):
            captured_current_texts.append(current_text)
            return original_extract_streaming(current_text, request)

        monkeypatch.setattr(parser, "_extract_streaming", wrapped_extract_streaming)

        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|><",
            "div>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        content_parts = [
            delta.content for delta, _ in results if delta is not None and delta.content
        ]
        assert "".join(content_parts) == "<div>"
        assert captured_current_texts[-1].endswith("<tool_call|><div>")
        assert not captured_current_texts[-1].endswith("<tool_call|><<div>")

    def test_streaming_html_argument_does_not_duplicate_tag_prefixes(
        self, parser, mock_request
    ):
        """HTML content inside tool arguments must not be duplicated."""
        chunks = [
            "<|tool_call>",
            "call:write_file{",
            'path:<|"|>index.html<|"|>,',
            'content:<|"|><!DOCTYPE html>\n<',
            'html lang="zh-CN">\n<',
            "head>\n    <",
            'meta charset="UTF-8">\n    <',
            'meta name="viewport" content="width=device-width">\n',
            '<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text

        parsed_args = json.loads(args_text)
        assert parsed_args["path"] == "index.html"
        assert (
            parsed_args["content"] == "<!DOCTYPE html>\n"
            '<html lang="zh-CN">\n'
            "<head>\n"
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width">\n'
        )

    def test_streaming_trailing_bare_bool_not_duplicated(self, parser, mock_request):
        """Trailing bare boolean must not be streamed twice."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'file_path:<|"|>src/env.py<|"|>,',
            'old_string:<|"|>old_val<|"|>,',
            'new_string:<|"|>new_val<|"|>,',
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        parsed_args = json.loads(args_text)
        assert parsed_args == {
            "file_path": "src/env.py",
            "old_string": "old_val",
            "new_string": "new_val",
            "replace_all": False,
        }

        assert args_text.count("replace_all") == 1

    def test_streaming_complete_tool_call_in_single_delta(self, parser, mock_request):
        """A complete tool call in one delta must be emitted."""
        chunks = [
            '<|tool_call>call:get_station{location:<|"|>Milano<|"|>}<tool_call|>',
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        calls = self._collect_tool_calls_by_index(results)

        assert set(calls) == {0}
        assert calls[0]["name"] == "get_station"
        assert json.loads(calls[0]["arguments"] or "{}") == {"location": "Milano"}

    def test_streaming_multiple_complete_tool_calls_in_single_delta(
        self, parser, mock_request
    ):
        """Multiple complete calls in one accepted delta must all be emitted."""
        chunks = [
            "<|tool_call>call:first_tool{alpha:15,beta:1,gamma:1990}"
            "<tool_call|><|tool_call>call:second_tool{delta:9,epsilon:30}"
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        calls = self._collect_tool_calls_by_index(results)

        assert set(calls) == {0, 1}
        assert calls[0]["name"] == "first_tool"
        assert json.loads(calls[0]["arguments"] or "{}") == {
            "alpha": 15,
            "beta": 1,
            "gamma": 1990,
        }
        assert calls[1]["name"] == "second_tool"
        assert json.loads(calls[1]["arguments"] or "{}") == {
            "delta": 9,
            "epsilon": 30,
        }

    def test_streaming_parallel_false_keeps_only_first_call_in_single_delta(
        self, parser, mock_request
    ):
        """parallel_tool_calls=false exposes only the first streamed call."""
        mock_request.parallel_tool_calls = False
        chunks = [
            "<|tool_call>call:first_tool{alpha:15,beta:1,gamma:1990}"
            "<tool_call|><|tool_call>call:second_tool{delta:9,epsilon:30}"
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        calls = self._collect_tool_calls_by_index(results)

        assert set(calls) == {0}
        assert calls[0]["name"] == "first_tool"
        assert json.loads(calls[0]["arguments"] or "{}") == {
            "alpha": 15,
            "beta": 1,
            "gamma": 1990,
        }

    def test_streaming_close_then_open_in_same_delta(self, parser, mock_request):
        """A delta can close one call and open the next without dropping either."""
        chunks = [
            "<|tool_call>call:first_tool{alpha:15,beta:1,gamma:1990}",
            "<tool_call|><|tool_call>call:second_tool{delta:9,epsilon:30}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        calls = self._collect_tool_calls_by_index(results)

        assert set(calls) == {0, 1}
        assert json.loads(calls[0]["arguments"] or "{}") == {
            "alpha": 15,
            "beta": 1,
            "gamma": 1990,
        }
        assert json.loads(calls[1]["arguments"] or "{}") == {
            "delta": 9,
            "epsilon": 30,
        }

    def test_streaming_incomplete_tool_call_emits_no_partial_args(
        self, parser, mock_request
    ):
        """Clients cannot retract partial args if generation ends early."""
        chunks = [
            "<|tool_call>",
            "call:first_tool{",
            "alpha:15,beta:1",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        assert self._collect_arguments(results) == ""
        assert self._collect_function_name(results) is None

    def test_streaming_repairs_final_end_marker_from_token_ids(
        self, parser, mock_request
    ):
        """Speculative streaming may omit the terminal end marker from text."""
        chunks = [
            (TOOL_CALL_START, [48]),
            ("call:first_tool{alpha:15,beta:1,gamma:1990}", [0]),
            ("", [49, 50]),
        ]
        results = []
        previous_text = ""
        previous_token_ids = []

        for delta_text, delta_token_ids in chunks:
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + delta_token_ids
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_token_ids),
                request=mock_request,
            )
            results.append((delta, current_text))
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        calls = self._collect_tool_calls_by_index(results)
        assert calls[0]["name"] == "first_tool"
        assert json.loads(calls[0]["arguments"] or "{}") == {
            "alpha": 15,
            "beta": 1,
            "gamma": 1990,
        }
        assert all(
            not (delta and delta.content and TOOL_RESPONSE in delta.content)
            for delta, _ in results
        )

    def test_streaming_repairs_empty_mtp_tool_text_from_token_ids(
        self, parser, mock_request
    ):
        """MTP can emit tool token IDs while text deltas stay empty."""
        chunks = [
            ("", [48]),
            ("", [100]),
            ("", [101, 49, 48, 102, 49, 50]),
        ]
        results = []
        previous_text = ""
        previous_token_ids = []

        for delta_text, delta_token_ids in chunks:
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + delta_token_ids
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_token_ids),
                request=mock_request,
            )
            results.append((delta, current_text))
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        calls = self._collect_tool_calls_by_index(results)
        assert set(calls) == {0, 1}
        assert calls[0]["name"] == "first_tool"
        assert json.loads(calls[0]["arguments"] or "{}") == {
            "alpha": 15,
            "beta": 1,
            "gamma": 1990,
        }
        assert calls[1]["name"] == "second_tool"
        assert json.loads(calls[1]["arguments"] or "{}") == {
            "delta": 9,
            "epsilon": 30,
        }
        assert all(
            not (delta and delta.content and TOOL_RESPONSE in delta.content)
            for delta, _ in results
        )

    def test_streaming_trailing_dot_float_split_across_chunks(
        self, parser, mock_request
    ):
        """Split decimal floats must not corrupt the final streamed JSON."""
        chunks = [
            "<|tool_call>",
            "call:set_coordinates{latitude:108.",
            "2,longitude:22.",
            "8}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        assert args_text
        assert json.loads(args_text) == {"latitude": 108.2, "longitude": 22.8}


    def test_streaming_split_start_delimiter_after_completed_call(
        self, parser, mock_request
    ):
        """A split start delimiter after a close delimiter should still replay."""
        chunks = [
            "<|tool_call>",
            "call:get_station_info{",
            'location:<|"|>Milano<|"|>}<tool_call|><',
            '|tool_call>call:get_station_info{location:<|"|>Piacenza<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        calls = self._collect_tool_calls_by_index(results)

        assert set(calls) == {0, 1}
        assert calls[0]["name"] == "get_station_info"
        assert json.loads(calls[0]["arguments"] or "{}") == {"location": "Milano"}
        assert calls[1]["name"] == "get_station_info"
        assert json.loads(calls[1]["arguments"] or "{}") == {"location": "Piacenza"}

    def test_streaming_filename_suffix_preserved_across_chunks(
        self, parser, mock_request
    ):
        """File extensions split across chunks must not be dropped."""
        chunks = [
            "<|tool_call>",
            "call:read_file{",
            'path:<|"|>src/main.',
            'rs<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        assert json.loads(args_text) == {"path": "src/main.rs"}

    def test_streaming_string_prefix_preserved_across_chunks(
        self, parser, mock_request
    ):
        """String values split after the first character must be preserved."""
        chunks = [
            "<|tool_call>",
            "call:open_item{",
            'mode:<|"|>e',
            'dit<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        assert json.loads(args_text) == {"mode": "edit"}
