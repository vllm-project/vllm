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

        def wrapped_extract_streaming(previous_text, current_text, delta_text):
            captured_current_texts.append(current_text)
            return original_extract_streaming(previous_text, current_text, delta_text)

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

    def test_streaming_incomplete_string_then_next_key(self, parser, mock_request):
        """Test incomplete string with delimiter + next key in same chunk."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>code with }] chars',  # No closing delimiter yet
            '<|"|>,path:<|"|>file.txt<|"|>}',  # Delimiter + next key together
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify both keys exist as separate keys
        assert "oldText" in parsed_args, "oldText key should exist"
        assert "path" in parsed_args, "path key should exist as separate field"

        # Verify no key bleeding into string values
        assert ",path:" not in parsed_args["oldText"], (
            f"'path' key incorrectly merged into oldText. Full args: {args_text!r}"
        )
        assert "<|" not in parsed_args["oldText"], (
            f"Delimiter fragment in oldText. Full args: {args_text!r}"
        )

        # Verify expected values
        assert parsed_args["oldText"] == "code with }] chars"
        assert parsed_args["path"] == "file.txt"

    def test_streaming_complete_string_comma_in_next_chunk(self, parser, mock_request):
        """Test complete string value when comma separator arrives in separate chunk."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>function() { return []; }<|"|>',  # Complete string
            ',path:<|"|>code.js<|"|>}',  # Comma + next key in separate chunk
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify both keys exist
        assert "oldText" in parsed_args
        assert "path" in parsed_args

        # Verify no key bleeding
        assert ",path:" not in parsed_args["oldText"], (
            f"'path' key incorrectly merged into oldText. Full args: {args_text!r}"
        )

        # Verify expected values
        assert parsed_args["oldText"] == "function() { return []; }"
        assert parsed_args["path"] == "code.js"

    def test_streaming_multiple_incomplete_strings(self, parser, mock_request):
        """Test multiple string values incomplete at different chunk boundaries."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>some long text value',  # Incomplete string
            '<|"|>,newText:<|"|>replacement',  # First closes, second incomplete
            '<|"|>,path:<|"|>src/app.py<|"|>}',  # Second closes, third complete
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify all three keys exist
        assert "oldText" in parsed_args
        assert "newText" in parsed_args
        assert "path" in parsed_args

        # Verify no key bleeding
        assert ",newText:" not in parsed_args["oldText"], (
            f"'newText' key incorrectly merged into oldText. Full args: {args_text!r}"
        )
        assert ",path:" not in parsed_args["newText"], (
            f"'path' key incorrectly merged into newText. Full args: {args_text!r}"
        )

        # Verify expected values
        assert parsed_args["oldText"] == "some long text value"
        assert parsed_args["newText"] == "replacement"
        assert parsed_args["path"] == "src/app.py"

    def test_streaming_string_ending_with_withholding_chars(self, parser, mock_request):
        """Test incomplete string ending with structural characters like }."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>function() { return []; }',  # Incomplete, ends with }
            '<|"|>,path:<|"|>code.js<|"|>}',  # Closes oldText, adds path
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify both keys exist
        assert "oldText" in parsed_args
        assert "path" in parsed_args

        # Verify no key bleeding
        assert ",path:" not in parsed_args["oldText"], (
            f"'path' key incorrectly merged into oldText. Full args: {args_text!r}"
        )

        # Verify expected values
        assert parsed_args["oldText"] == "function() { return []; }"
        assert parsed_args["path"] == "code.js"

    def test_streaming_realistic_code_with_multiple_keys(self, parser, mock_request):
        """Test code containing }] with multiple incomplete strings."""
        chunks = [
            "<|tool_call>",
            "call:ReplaceText{",
            # oldText incomplete - Go code ending with }
            'oldText:<|"|>// qgaSplit function\n'
            "func qgaSplit(data []byte) {\n"
            "  return 0, nil, nil\n"
            "}",
            # oldText closes, newText incomplete
            '<|"|>,newText:<|"|>// QGAReader implementation\ntype QGAReader struct {}',
            # newText closes, path complete
            '<|"|>,path:<|"|>qga.go<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify all three keys exist
        assert "oldText" in parsed_args
        assert "newText" in parsed_args
        assert "path" in parsed_args

        # Verify no key bleeding
        assert ",newText:" not in parsed_args["oldText"], (
            f"'newText' key incorrectly merged into oldText. Full args: {args_text!r}"
        )
        assert ",path:" not in parsed_args["oldText"], (
            f"'path' key incorrectly merged into oldText. Full args: {args_text!r}"
        )
        assert ",path:" not in parsed_args["newText"], (
            f"'path' key incorrectly merged into newText. Full args: {args_text!r}"
        )

        # Verify oldText contains expected Go code
        assert "func qgaSplit" in parsed_args["oldText"]
        assert parsed_args["oldText"].endswith("}")

        # Verify newText contains expected Go code
        assert "QGAReader" in parsed_args["newText"]
        assert "struct" in parsed_args["newText"]

        # Verify path is correct
        assert parsed_args["path"] == "qga.go"

    def test_streaming_string_value_ending_with_closing_brace(
        self, parser, mock_request
    ):
        """Test string value ending with } is preserved (e.g., 'return {}')."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # Value ends with }, complete in this chunk
            'oldText:<|"|>return {}<|"|>',
            # Next key arrives in separate chunk
            ',path:<|"|>test.js<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify closing brace is preserved in string value
        assert parsed_args["oldText"] == "return {}", (
            f"Closing brace incorrectly stripped from string value. "
            f"Expected 'return {{}}', got {parsed_args['oldText']!r}. "
            f"Full args: {args_text!r}"
        )

        # Verify path exists as separate key
        assert "path" in parsed_args
        assert parsed_args["path"] == "test.js"

    def test_streaming_string_value_ending_with_closing_bracket(
        self, parser, mock_request
    ):
        """Test string value ending with ] is preserved (e.g., 'data = []')."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>data = []<|"|>',  # Complete string ending with ]
            ',path:<|"|>test.py<|"|>}',  # Next key in separate chunk
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify closing bracket is preserved in string value
        assert parsed_args["oldText"] == "data = []", (
            f"Closing bracket incorrectly stripped from string value. "
            f"Expected 'data = []', got {parsed_args['oldText']!r}. "
            f"Full args: {args_text!r}"
        )

        assert "path" in parsed_args
        assert parsed_args["path"] == "test.py"

    def test_streaming_string_ending_with_multiple_structural_chars(
        self, parser, mock_request
    ):
        """Test string ending with multiple structural characters like }]."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # Value ends with }] - both in withholding list
            'oldText:<|"|>func() { return [1, 2, 3]; }<|"|>',
            ',path:<|"|>code.go<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Verify valid JSON
        parsed_args = json.loads(args_text)

        # Verify structural characters preserved in string value
        expected = "func() { return [1, 2, 3]; }"
        assert parsed_args["oldText"] == expected, (
            f"Structural characters incorrectly stripped from string value. "
            f"Expected {expected!r}, got {parsed_args['oldText']!r}. "
            f"Full args: {args_text!r}"
        )

        assert "path" in parsed_args
        assert parsed_args["path"] == "code.go"

    def test_streaming_value_with_comma_then_key_no_space(self, parser, mock_request):
        """Test string value ending with comma followed immediately by next key."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>item1,item2,item3',  # Incomplete, ends with comma
            '<|"|>,path:<|"|>file.txt<|"|>}',  # Closes, then new key starts with comma
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # The comma in "item3," is PART of oldText, but ",path:" is a new key
        assert parsed_args["oldText"] == "item1,item2,item3"
        assert ",path:" not in parsed_args["oldText"]
        assert "path" in parsed_args
        assert parsed_args["path"] == "file.txt"

    def test_streaming_chunk_ends_mid_key_name(self, parser, mock_request):
        """Test chunk boundary splitting a key name."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>code<|"|>,pa',  # Chunk ends mid-key "pa"
            'th:<|"|>file.txt<|"|>}',  # Key name completes
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        assert parsed_args["oldText"] == "code"
        assert ",pa" not in parsed_args["oldText"]
        assert "path" in parsed_args
        assert parsed_args["path"] == "file.txt"

    def test_streaming_value_contains_key_like_pattern(self, parser, mock_request):
        """Test string value containing text that looks like a key (e.g., ',path:')."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # String value itself contains ",path:" pattern
            'oldText:<|"|>Set ,path: option in config',
            '<|"|>,actualPath:<|"|>config.yaml<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # The ",path:" inside the string value should be preserved
        assert parsed_args["oldText"] == "Set ,path: option in config"
        assert "actualPath" in parsed_args
        assert parsed_args["actualPath"] == "config.yaml"

    def test_streaming_very_long_value_with_structural_chars(
        self, parser, mock_request
    ):
        """Test very long string value with many structural characters."""
        # Create a long code snippet with many }, ], etc.
        code = """function test() {
    const data = [1, 2, 3];
    const obj = {key: "value"};
    return {data, obj};
}"""

        chunks = [
            "<|tool_call>",
            "call:Edit{",
            f'oldText:<|"|>{code}',  # Long string, incomplete
            '<|"|>,path:<|"|>test.js<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        assert parsed_args["oldText"] == code
        assert ",path:" not in parsed_args["oldText"]
        assert "path" in parsed_args
        assert parsed_args["path"] == "test.js"

    def test_streaming_empty_string_then_next_key(self, parser, mock_request):
        """Test empty string value followed by another key."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|><|"|>,',  # Empty string + comma
            'path:<|"|>file.txt<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        assert parsed_args["oldText"] == ""
        assert "path" in parsed_args
        assert parsed_args["path"] == "file.txt"

    def test_streaming_number_value_then_string_key(self, parser, mock_request):
        """Test numeric value followed by string key."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'lineNumber:42,oldText:<|"|>code',  # Number then string
            '<|"|>,path:<|"|>file.txt<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        assert parsed_args["lineNumber"] == 42
        assert parsed_args["oldText"] == "code"
        assert "path" in parsed_args
        assert parsed_args["path"] == "file.txt"

    def test_streaming_string_value_ending_with_exact_bug_pattern(
        self, parser, mock_request
    ):
        """Test string value ending with pattern that looks like next key.

        Adversarial: The string value literally ends with text that looks
        exactly like '},path:"' which might confuse the parser.
        """
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # String value ends with literal text "},path:"
            'oldText:<|"|>The config uses },path:',
            '<|"|>,actualPath:<|"|>test.js<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # The literal "},path:" should be in the string value
        assert parsed_args["oldText"] == "The config uses },path:"
        assert "actualPath" in parsed_args
        assert parsed_args["actualPath"] == "test.js"

    def test_streaming_withholding_with_identical_suffix_pattern(
        self, parser, mock_request
    ):
        """Test when withheld suffix characters match start of next chunk."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>code }]',  # Ends with }] (will be withheld)
            '<|"|>,}],path:<|"|>file.txt<|"|>}',  # Next chunk also starts with }]
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # Should NOT duplicate the }]
        assert parsed_args["oldText"] == "code }]"
        assert "}],path" in parsed_args  # This is a separate key
        assert parsed_args["}],path"] == "file.txt"

    def test_streaming_json_chars_at_every_boundary(self, parser, mock_request):
        """Test JSON structural characters at every possible chunk boundary."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'a:<|"|>}',  # Value is just '}'
            '<|"|>,b:<|"|>]',  # Value is just ']'
            '<|"|>,c:<|"|>",',  # Value is just '",'
            '<|"|>,d:<|"|>ok<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        assert parsed_args["a"] == "}"
        assert parsed_args["b"] == "]"
        assert parsed_args["c"] == '",'
        assert parsed_args["d"] == "ok"

    def test_streaming_missing_string_delimiter_causes_overconsumption(
        self, parser, mock_request
    ):
        """Test parser handles malformed input missing closing delimiter."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # MALFORMED: Missing closing delimiter after "code with }]"
            'oldText:<|"|>code with }],path:<|"|>file.txt<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # Verify parser correctly handles malformed input
        assert "oldText" in parsed_args, "oldText key should exist"

        # Verify no overconsumption
        assert ",path:" not in parsed_args["oldText"], (
            f"Parser incorrectly consumed ',path:' into oldText. "
            f"Got: {parsed_args['oldText']!r}, full JSON: {args_text!r}"
        )

        # Verify correct parsing despite missing delimiter
        assert parsed_args["oldText"] == "code with }]"
        assert "path" in parsed_args
        assert parsed_args["path"] == "file.txt"

    def test_streaming_value_with_comma_no_colon(self, parser, mock_request):
        """Test well-formed string value containing comma without colon."""
        chunks = [
            "<|tool_call>",
            "call:GetData{",
            'location:<|"|>Paris, France<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        parsed_args = json.loads(args_text)

        assert parsed_args["location"] == "Paris, France", (
            "Should preserve comma in well-formed value"
        )

    def test_streaming_value_ending_with_comma_colon(self, parser, mock_request):
        """Test well-formed value ending with ',identifier:' pattern is preserved."""
        chunks = [
            "<|tool_call>",
            "call:ProcessData{",
            'pattern:<|"|>foo,bar:<|"|>,next:<|"|>value<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        parsed_args = json.loads(args_text)

        assert parsed_args["pattern"] == "foo,bar:", (
            "Should preserve ',bar:' pattern in well-formed value"
        )
        assert parsed_args["next"] == "value", (
            "Should correctly parse next key after value with ',identifier:' pattern"
        )

    def test_streaming_malformed_with_comma_colon_pattern(self, parser, mock_request):
        """Test malformed input with ',identifier:' pattern and missing delimiter."""
        chunks = [
            "<|tool_call>",
            "call:Replace{",
            'old:<|"|>foo,bar:<|"|>replacement<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        parsed_args = json.loads(args_text)

        assert parsed_args["old"] == "foo", (
            "Should stop at comma when delimiter not followed by structural char"
        )
        assert parsed_args["bar"] == "replacement", (
            "Should correctly parse 'bar' as the next key"
        )

    def test_streaming_search_replace_realistic(self, parser, mock_request):
        """Test realistic search/replace args with commas and colons."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>const config = {a: 1, b: 2}<|"|>,',
            'newText:<|"|>const config = {a: 1, b: 2, c: 3}<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        parsed_args = json.loads(args_text)

        assert parsed_args["oldText"] == "const config = {a: 1, b: 2}", (
            "Should preserve code with commas and colons in well-formed value"
        )
        assert parsed_args["newText"] == "const config = {a: 1, b: 2, c: 3}", (
            "Should correctly parse second argument"
        )

    def test_streaming_extra_delimiter_in_value(self, parser, mock_request):
        """Test string value containing unescaped delimiter."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            # String contains literal <|"|> that should be escaped but isn't
            'oldText:<|"|>use <|"|> as delimiter<|"|>,path:<|"|>file.txt<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        parsed_args = json.loads(args_text)

        # Verify unescaped delimiter doesn't cause key bleeding
        if "oldText" in parsed_args and ",path:" in parsed_args["oldText"]:
            pytest.fail(
                f"Unescaped delimiter caused ',path:' to bleed into oldText. "
                f"oldText value: {parsed_args['oldText']!r}"
            )

    def test_streaming_partial_mode_string_never_closes(self, parser, mock_request):
        """Test incomplete string that grows across many chunks without closing."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'oldText:<|"|>line1',  # Incomplete
            " line2",  # Still incomplete
            " line3",  # Still incomplete
            ",path",  # STILL incomplete - "," is part of the string!
            ':<|"|>',  # Wait, now we have :<|"|> but no closing for oldText
            'file.txt<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        try:
            parsed_args = json.loads(args_text)

            # The ,path: sequence appeared while oldText was incomplete
            # Did the parser incorrectly treat it as a new key?
            if "oldText" in parsed_args:
                # oldText should be "line1 line2 line3,path"
                # But if the parser got confused, it might be "line1 line2 line3"
                expected_value = "line1 line2 line3,path"
                if (
                    parsed_args["oldText"] != expected_value
                    and ",path" not in parsed_args["oldText"]
                ):
                    pytest.fail(
                        f"Parser lost ',path' from oldText value. "
                        f"Expected: {expected_value!r}, "
                        f"Got: {parsed_args['oldText']!r}"
                    )

        except json.JSONDecodeError as e:
            pytest.fail(f"JSON parse failed: {e}. Raw output: {args_text!r}")
