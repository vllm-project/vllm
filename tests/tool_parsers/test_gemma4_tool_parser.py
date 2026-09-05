# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.gemma4 import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    _parse_gemma4_args,
    _parse_gemma4_array,
)
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TOOL_CALL_START_ID = 48
TOOL_CALL_END_ID = 49
CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
CHANNEL_START_ID = 50
CHANNEL_END_ID = 51


def _make_tool(name, properties):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
    )


_TOOLS = [
    _make_tool(
        "set_status",
        {
            "is_active": {"type": "boolean"},
            "count": {"type": "integer"},
            "score": {"type": "number"},
        },
    ),
    _make_tool(
        "set_config",
        {
            "count": {"type": "integer"},
            "active": {"type": "boolean"},
        },
    ),
    _make_tool(
        "search",
        {
            "input": {
                "type": "object",
                "properties": {"all": {"type": "boolean"}},
            },
        },
    ),
    _make_tool(
        "set",
        {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
        },
    ),
    _make_tool(
        "Edit",
        {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
    ),
]


@pytest.fixture
def mock_tokenizer():
    vocab = {
        TOOL_CALL_START: TOOL_CALL_START_ID,
        TOOL_CALL_END: TOOL_CALL_END_ID,
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }
    decode_map = {v: k for k, v in vocab.items()}

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return Gemma4EngineToolParser(mock_tokenizer, tools=_TOOLS)


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
    """Bare values are converted to their JSON types (int, float, bool, None)."""

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

    def test_null_value(self):
        result = _parse_gemma4_args("param:null")
        assert result == {"param": None}

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

    def test_delimited_keys_stripped(self):
        """Keys wrapped in <|"|> delimiters are stripped."""
        result = _parse_gemma4_args('<|"|>location<|"|>:<|"|>Paris<|"|>')
        assert result == {"location": "Paris"}

        result = _parse_gemma4_args('outer:{<|"|>inner<|"|>:<|"|>val<|"|>}')
        assert result == {"outer": {"inner": "val"}}

        result = _parse_gemma4_args('<|"|>name<|"|>:<|"|>Alice<|"|>,count:42')
        assert result == {"name": "Alice", "count": 42}

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

    def test_trailing_dot_float_partial_withheld(self):
        """Bare float ending with '.' is withheld in partial mode.

        Regression test for #42047: float("108.") → 108.0 causes
        streaming diff corruption (108.0 → 108.2 becomes 108.02).
        """
        # Single key with trailing dot — withheld entirely
        result = _parse_gemma4_args("left:108.,right:22.8", partial=True)
        assert result == {}

        # Stable key before trailing-dot key — stable key is kept
        result = _parse_gemma4_args(
            'name:<|"|>test<|"|>,score:3.,count:1', partial=True
        )
        assert result == {"name": "test"}

        # Non-partial mode parses trailing dot normally
        result = _parse_gemma4_args("left:108.,right:22.8", partial=False)
        assert result == {"left": 108.0, "right": 22.8}

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

    @pytest.mark.timeout(5)
    def test_string_element_with_closing_bracket(self):
        result = _parse_gemma4_array('[<|"|>a]b<|"|>,<|"|>c<|"|>],<|"|>tail<|"|>')
        assert result == [["a]b", "c"], "tail"]

    @pytest.mark.timeout(5)
    def test_stray_closing_bracket(self):
        result = _parse_gemma4_array("42,]trailing")
        assert result == [42]

    def test_trailing_dot_float_partial_withheld(self):
        """Array elements with trailing dot withheld in partial mode."""
        result = _parse_gemma4_array("108.,22.8", partial=True)
        assert result == []

        # Stable elements before trailing-dot element are kept
        result = _parse_gemma4_array("42,108.,3", partial=True)
        assert result == [42]


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

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

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

    _SPECIAL_TOKEN_IDS = {
        TOOL_CALL_START: TOOL_CALL_START_ID,
        TOOL_CALL_END: TOOL_CALL_END_ID,
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }

    def _simulate_streaming(
        self, parser: Any, mock_request: Any, chunks: list[str]
    ) -> list[tuple[Any, str]]:
        """Feed chunks through the streaming parser and collect results.

        Returns a list of (delta_message, accumulated_text) tuples.
        """
        results: list[tuple[Any, str]] = []
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            found: list[tuple[int, int]] = []
            for token, tid in self._SPECIAL_TOKEN_IDS.items():
                pos = 0
                while True:
                    idx = chunk.find(token, pos)
                    if idx < 0:
                        break
                    found.append((idx, tid))
                    pos = idx + len(token)
            found.sort()
            delta_token_ids: list[int] = [tid for _, tid in found] if found else [0]

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
        assert args_text is not None
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
        self, parser, mock_request
    ):
        """Buffered plain text after a tool call must not corrupt content."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
            "<",
            "div>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        content_parts = [
            delta.content for delta, _ in results if delta is not None and delta.content
        ]
        assert "".join(content_parts) == "<div>"
        assert "<<div>" not in "".join(content_parts)

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

    def _collect_tool_calls_by_index(self, results):
        """Group streamed tool-call fragments by their ``index``.

        Returns ``{index: {"name": str | None, "arguments": str}}`` where
        ``arguments`` is the concatenation of every streamed argument
        fragment for that index (which should form valid JSON once complete).
        """
        by_index: dict[int, dict[str, Any]] = {}
        for delta, _ in results:
            if not (delta and delta.tool_calls):
                continue
            for tc in delta.tool_calls:
                entry = by_index.setdefault(tc.index, {"name": None, "arguments": ""})
                func = tc.function
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
        return by_index

    def test_streaming_single_chunk_complete_tool_call(self, parser, mock_request):
        """A backend may deliver a whole tool call in one streaming delta.

        The start token, ``call:name{...}`` payload and the end token all
        arrive in a single chunk. The parser must still emit one
        ``DeltaToolCall`` with the correct name + complete arguments JSON
        (rather than swallowing it and finishing with finish_reason="stop").
        """
        chunks = [
            '<|tool_call>call:name_a_color{color_hex:<|"|>00ff11<|"|>}<tool_call|>',
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # Exactly one delta should carry tool_calls, and it must not be
        # emitted as plain content (which would yield finish_reason="stop").
        tool_call_deltas = [
            delta for delta, _ in results if delta is not None and delta.tool_calls
        ]
        assert len(tool_call_deltas) == 1, (
            "Expected exactly one delta carrying the batched tool call"
        )
        assert all(
            delta.content is None for delta, _ in results if delta is not None
        ), "Complete tool call must not leak as content"

        by_index = self._collect_tool_calls_by_index(results)
        assert set(by_index) == {0}
        assert by_index[0]["name"] == "name_a_color"
        assert json.loads(by_index[0]["arguments"]) == {"color_hex": "00ff11"}

    def test_streaming_multi_chunk_batched_tool_calls(self, parser, mock_request):
        """A single delta may batch MULTIPLE complete tool calls.

        ``<|tool_call>...<tool_call|><|tool_call>...<tool_call|>`` arriving in
        one chunk must emit BOTH calls (one DeltaToolCall each, with distinct
        indices), not just the first.
        """
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
            '<|tool_call>call:get_time{timezone:<|"|>GMT<|"|>}<tool_call|>',
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        by_index = self._collect_tool_calls_by_index(results)
        assert set(by_index) == {0, 1}, (
            f"Expected two tool calls (indices 0 and 1), got {sorted(by_index)}"
        )

        assert by_index[0]["name"] == "get_weather"
        assert json.loads(by_index[0]["arguments"]) == {"location": "London"}

        assert by_index[1]["name"] == "get_time"
        assert json.loads(by_index[1]["arguments"]) == {"timezone": "GMT"}

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
