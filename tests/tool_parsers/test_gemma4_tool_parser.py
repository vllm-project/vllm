# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Gemma4ToolParser — covers both native and fallback formats,
argument parsing, streaming, and edge cases.

Run with:
    pytest tests/tool_parsers/test_gemma4_tool_parser.py -v
"""

import json

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

# ---------------------------------------------------------------------------
# Import helpers from the parser module under test
# ---------------------------------------------------------------------------
from gemma4_tool_parser import (
    _parse_gemma4_args,
    _parse_gemma4_array,
    _parse_gemma4_value,
    _detect_format,
    TOOL_CALL_START,
    TOOL_CALL_END,
    FALLBACK_TOOL_CALL_START,
    FALLBACK_TOOL_CALL_END,
    STRING_DELIM,
)

from vllm_mock import _tokenize_for_streaming


# ---------------------------------------------------------------------------
# Tokenizer fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gemma4_tokenizer() -> TokenizerLike:
    """
    GPT-2 tokenizer augmented with Gemma4 special tokens.
    Using a real Gemma4 tokenizer would be ideal in CI, but GPT-2 is
    sufficient for unit-level parser tests.
    """
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_tokens([
        TOOL_CALL_START,   # <|tool_call>
        TOOL_CALL_END,     # <tool_call|>
        STRING_DELIM,      # <|"|>
    ])
    return tok


# ---------------------------------------------------------------------------
# Helpers: build raw model output strings
# ---------------------------------------------------------------------------

def _native(func_name: str, args_str: str) -> str:
    """Build a native-format tool call string."""
    return f"{TOOL_CALL_START}call:{func_name}{{{args_str}}}{TOOL_CALL_END}"


def _fallback(func_name: str, args_str: str) -> str:
    """Build a fallback XML-style tool call string."""
    return f"{FALLBACK_TOOL_CALL_START}{func_name}{{{args_str}}}{FALLBACK_TOOL_CALL_END}"


def _gemma4_str(value: str) -> str:
    """Wrap a string in Gemma4 string delimiters."""
    return f"{STRING_DELIM}{value}{STRING_DELIM}"



# ---------------------------------------------------------------------------
# Reusable argument strings and their expected parsed dicts
# ---------------------------------------------------------------------------

EMPTY_ARGS_STR = ""
EMPTY_ARGS_DICT = {}

SIMPLE_ARGS_STR = f"city:{_gemma4_str('Tokyo')}"
SIMPLE_ARGS_DICT = {"city": "Tokyo"}

MULTI_ARGS_STR = (
    f"location:{_gemma4_str('San Francisco')},unit:{_gemma4_str('celsius')}"
)
MULTI_ARGS_DICT = {"location": "San Francisco", "unit": "celsius"}

NUMERIC_ARGS_STR = "count:42,ratio:3.14,flag:true,disabled:false"
NUMERIC_ARGS_DICT = {"count": 42, "ratio": 3.14, "flag": True, "disabled": False}

NESTED_ARGS_STR = f"outer:{{inner:{_gemma4_str('deep')}}}"
NESTED_ARGS_DICT = {"outer": {"inner": "deep"}}

ARRAY_ARGS_STR = f"items:[{_gemma4_str('a')},{_gemma4_str('b')},{_gemma4_str('c')}]"
ARRAY_ARGS_DICT = {"items": ["a", "b", "c"]}

COMPLEX_ARGS_STR = (
    f"action:{_gemma4_str('create')},"
    f"id:{_gemma4_str('preferences')},"
    f"content:{{short_answers:true,count:5,label:{_gemma4_str('main')}}}"
)
COMPLEX_ARGS_DICT = {
    "action": "create",
    "id": "preferences",
    "content": {"short_answers": True, "count": 5, "label": "main"},
}

CONTENT_TEXT = "Sure, I'll do that for you."


# ===========================================================================
# Section 1: Unit tests for _parse_gemma4_value
# ===========================================================================

class TestParseGemma4Value:
    def test_empty_string(self):
        assert _parse_gemma4_value("") == ""

    def test_whitespace_only(self):
        assert _parse_gemma4_value("  ") == ""

    def test_true(self):
        assert _parse_gemma4_value("true") is True

    def test_false(self):
        assert _parse_gemma4_value("false") is False

    def test_integer(self):
        assert _parse_gemma4_value("42") == 42

    def test_negative_integer(self):
        assert _parse_gemma4_value("-7") == -7

    def test_float(self):
        result = _parse_gemma4_value("3.14")
        assert abs(result - 3.14) < 1e-9

    def test_bare_string(self):
        # Bare strings (no delimiters) should be returned as-is
        assert _parse_gemma4_value("hello") == "hello"


# ===========================================================================
# Section 2: Unit tests for _parse_gemma4_args
# ===========================================================================

class TestParseGemma4Args:
    def test_empty(self):
        assert _parse_gemma4_args("") == {}

    def test_whitespace_only(self):
        assert _parse_gemma4_args("   ") == {}

    def test_single_string(self):
        assert _parse_gemma4_args(SIMPLE_ARGS_STR) == SIMPLE_ARGS_DICT

    def test_multiple_strings(self):
        assert _parse_gemma4_args(MULTI_ARGS_STR) == MULTI_ARGS_DICT

    def test_numeric_and_boolean(self):
        assert _parse_gemma4_args(NUMERIC_ARGS_STR) == NUMERIC_ARGS_DICT

    def test_nested_object(self):
        assert _parse_gemma4_args(NESTED_ARGS_STR) == NESTED_ARGS_DICT

    def test_array_value(self):
        assert _parse_gemma4_args(ARRAY_ARGS_STR) == ARRAY_ARGS_DICT

    def test_complex_mixed(self):
        assert _parse_gemma4_args(COMPLEX_ARGS_STR) == COMPLEX_ARGS_DICT

    def test_unterminated_string_value(self):
        # Unterminated string — should capture rest of input
        raw = f"key:{STRING_DELIM}unclosed"
        result = _parse_gemma4_args(raw)
        assert result["key"] == "unclosed"

    def test_deeply_nested(self):
        raw = f"a:{{b:{{c:{_gemma4_str('deep')}}}}}"
        result = _parse_gemma4_args(raw)
        assert result == {"a": {"b": {"c": "deep"}}}

    def test_array_of_objects(self):
        raw = (
            f"items:[{{name:{_gemma4_str('Alice')}}},"
            f"{{name:{_gemma4_str('Bob')}}}]"
        )
        result = _parse_gemma4_args(raw)
        assert result == {"items": [{"name": "Alice"}, {"name": "Bob"}]}

    def test_string_with_spaces(self):
        raw = f"msg:{_gemma4_str('hello world')}"
        assert _parse_gemma4_args(raw) == {"msg": "hello world"}

    def test_string_with_comma_inside(self):
        raw = f"addr:{_gemma4_str('123 Main St, Springfield')}"
        assert _parse_gemma4_args(raw) == {"addr": "123 Main St, Springfield"}


# ===========================================================================
# Section 3: Unit tests for _parse_gemma4_array
# ===========================================================================

class TestParseGemma4Array:
    def test_empty(self):
        assert _parse_gemma4_array("") == []

    def test_strings(self):
        raw = f"{_gemma4_str('x')},{_gemma4_str('y')}"
        assert _parse_gemma4_array(raw) == ["x", "y"]

    def test_numbers(self):
        assert _parse_gemma4_array("1,2,3") == [1, 2, 3]

    def test_mixed_types(self):
        raw = f"{_gemma4_str('a')},42,true"
        assert _parse_gemma4_array(raw) == ["a", 42, True]

    def test_nested_arrays(self):
        raw = "[1,2],[3,4]"
        result = _parse_gemma4_array(raw)
        assert result == [[1, 2], [3, 4]]


# ===========================================================================
# Section 4: Unit tests for _detect_format
# ===========================================================================

class TestDetectFormat:
    def test_no_tool_call(self):
        assert _detect_format("Hello, how can I help?") == "none"

    def test_native_only(self):
        text = _native("my_func", SIMPLE_ARGS_STR)
        assert _detect_format(text) == "native"

    def test_fallback_only(self):
        text = _fallback("my_func", SIMPLE_ARGS_STR)
        assert _detect_format(text) == "fallback"

    def test_both_formats(self):
        text = _native("f1", "") + _fallback("f2", "")
        assert _detect_format(text) == "both"

    def test_fallback_start_only_no_end(self):
        # Only opening tag present — should not count as fallback
        assert _detect_format(FALLBACK_TOOL_CALL_START + "hangup{}") == "none"


# ===========================================================================
# Section 5: Non-streaming extraction tests
# ===========================================================================

class TestExtractToolCallsNonStreaming:

    @pytest.fixture
    def parser(self, gemma4_tokenizer):
        return ToolParserManager.get_tool_parser("gemma4")(gemma4_tokenizer)

    # --- No tool call ---

    def test_plain_text(self, parser):
        out = parser.extract_tool_calls("Hello there!", request=None)
        assert not out.tools_called
        assert out.content == "Hello there!"
        assert out.tool_calls == []

    # --- Native format ---

    @pytest.mark.parametrize("args_str,expected_dict", [
        pytest.param(EMPTY_ARGS_STR, EMPTY_ARGS_DICT, id="native_empty_args"),
        pytest.param(SIMPLE_ARGS_STR, SIMPLE_ARGS_DICT, id="native_simple_args"),
        pytest.param(MULTI_ARGS_STR, MULTI_ARGS_DICT, id="native_multi_args"),
        pytest.param(NUMERIC_ARGS_STR, NUMERIC_ARGS_DICT, id="native_numeric_bool"),
        pytest.param(NESTED_ARGS_STR, NESTED_ARGS_DICT, id="native_nested"),
        pytest.param(ARRAY_ARGS_STR, ARRAY_ARGS_DICT, id="native_array"),
        pytest.param(COMPLEX_ARGS_STR, COMPLEX_ARGS_DICT, id="native_complex"),
    ])
    def test_native_format(self, parser, args_str, expected_dict):
        model_output = _native("my_tool", args_str)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert len(out.tool_calls) == 1
        tc = out.tool_calls[0]
        assert tc.function.name == "my_tool"
        assert json.loads(tc.function.arguments) == expected_dict

    def test_native_content_before_tool_call(self, parser):
        model_output = CONTENT_TEXT + _native("hangup_call", EMPTY_ARGS_STR)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert out.content == CONTENT_TEXT
        assert out.tool_calls[0].function.name == "hangup_call"

    def test_native_multiple_tool_calls(self, parser):
        model_output = (
            _native("tool_a", SIMPLE_ARGS_STR)
            + _native("tool_b", NUMERIC_ARGS_STR)
        )
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0].function.name == "tool_a"
        assert out.tool_calls[1].function.name == "tool_b"

    # --- Fallback format ---

    @pytest.mark.parametrize("args_str,expected_dict", [
        pytest.param(EMPTY_ARGS_STR, EMPTY_ARGS_DICT, id="fallback_empty_args"),
        pytest.param(SIMPLE_ARGS_STR, SIMPLE_ARGS_DICT, id="fallback_simple_args"),
        pytest.param(MULTI_ARGS_STR, MULTI_ARGS_DICT, id="fallback_multi_args"),
        pytest.param(NUMERIC_ARGS_STR, NUMERIC_ARGS_DICT, id="fallback_numeric_bool"),
        pytest.param(NESTED_ARGS_STR, NESTED_ARGS_DICT, id="fallback_nested"),
        pytest.param(ARRAY_ARGS_STR, ARRAY_ARGS_DICT, id="fallback_array"),
        pytest.param(COMPLEX_ARGS_STR, COMPLEX_ARGS_DICT, id="fallback_complex"),
    ])
    def test_fallback_format(self, parser, args_str, expected_dict):
        model_output = _fallback("my_tool", args_str)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert len(out.tool_calls) == 1
        tc = out.tool_calls[0]
        assert tc.function.name == "my_tool"
        assert json.loads(tc.function.arguments) == expected_dict

    def test_fallback_hangup_empty_args(self, parser):
        """Exact reproduction of the reported bug case."""
        model_output = "<tool_call>hangup_call{}</tool_call>"
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].function.name == "hangup_call"
        assert json.loads(out.tool_calls[0].function.arguments) == {}

    def test_fallback_content_before_tool_call(self, parser):
        model_output = CONTENT_TEXT + _fallback("hangup_call", EMPTY_ARGS_STR)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert out.content == CONTENT_TEXT
        assert out.tool_calls[0].function.name == "hangup_call"

    def test_fallback_multiple_tool_calls(self, parser):
        model_output = (
            _fallback("tool_a", SIMPLE_ARGS_STR)
            + _fallback("tool_b", EMPTY_ARGS_STR)
        )
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert len(out.tool_calls) == 2
        assert out.tool_calls[0].function.name == "tool_a"
        assert json.loads(out.tool_calls[0].function.arguments) == SIMPLE_ARGS_DICT
        assert out.tool_calls[1].function.name == "tool_b"
        assert json.loads(out.tool_calls[1].function.arguments) == {}

    # --- Function names ---

    @pytest.mark.parametrize("func_name", [
        "simple",
        "with_underscore",
        "with-hyphen",
        "module.method",
        "a1b2c3",
    ])
    def test_fallback_various_function_names(self, parser, func_name):
        model_output = _fallback(func_name, EMPTY_ARGS_STR)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.tools_called
        assert out.tool_calls[0].function.name == func_name

    # --- No content leak ---

    def test_no_content_when_only_tool_call(self, parser):
        model_output = _fallback("hangup_call", EMPTY_ARGS_STR)
        out = parser.extract_tool_calls(model_output, request=None)
        assert out.content is None


# ===========================================================================
# Section 6: Streaming extraction tests
# ===========================================================================

class TestExtractToolCallsStreaming:

    @pytest.fixture
    def parser(self, gemma4_tokenizer):
        return ToolParserManager.get_tool_parser("gemma4")(gemma4_tokenizer)

    def _stream(self, parser, deltas, assert_one_per_delta=False):
        return run_tool_extraction_streaming(
            parser,
            deltas,
            assert_one_tool_per_delta=assert_one_per_delta,
        )

    # --- Native format streaming ---

    def test_native_streaming_simple(self, parser):
        full = _native("get_weather", SIMPLE_ARGS_STR)
        # AFTER (always token-aligned):
        deltas = _tokenize_for_streaming(full)
        result = self._stream(parser, deltas)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == SIMPLE_ARGS_DICT

    def test_native_streaming_empty_args(self, parser):
        full = _native("hangup_call", EMPTY_ARGS_STR)
        deltas = _tokenize_for_streaming(full)
        result = self._stream(parser, deltas)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "hangup_call"
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_native_streaming_complex_args(self, parser):
        full = _native("manage_prefs", COMPLEX_ARGS_STR)
        deltas = _tokenize_for_streaming(full)
        result = self._stream(parser, deltas, assert_one_per_delta=False)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "manage_prefs"
        assert json.loads(result.tool_calls[0].function.arguments) == COMPLEX_ARGS_DICT

    def test_native_streaming_with_content_prefix(self, parser):
        full = CONTENT_TEXT + _native("do_thing", SIMPLE_ARGS_STR)
        deltas = _tokenize_for_streaming(full)
        result = self._stream(parser, deltas, assert_one_per_delta=False)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "do_thing"
        assert result.content is not None
        assert CONTENT_TEXT in result.content

    # --- Fallback format streaming ---

    def test_fallback_streaming_hangup_empty_args(self, parser):
        """Exact reported bug: <tool_call>call:hangup_call{}</tool_call> streamed."""
        deltas = [
            "<tool_call>",
            "call",
            ":",
            "hangup_call",
            "{}",
            "</tool_call>",
        ]
        result = self._stream(parser, deltas)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "hangup_call"
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_fallback_streaming_simple_args(self, parser):
        full = _fallback("get_weather", SIMPLE_ARGS_STR)
        deltas = _tokenize_for_streaming(full)
        result = self._stream(parser, deltas, assert_one_per_delta=False)
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

    def test_skip_special_tokens_unchanged_when_tool_choice_none(self, parser):
        """tool_choice=none means tools are disabled; don't override."""
        request = ChatCompletionRequest(
            model="gemma4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "hangup_call",
                    "description": "Hang up",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            tool_choice="none",
        )
        original = request.skip_special_tokens
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens == original


# ===========================================================================
# Section 8: Parametrized combined streaming + non-streaming
# ===========================================================================

@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("model_output,func_name,expected_args", [
    pytest.param(
        "<tool_call>hangup_call{}</tool_call>",
        "hangup_call",
        {},
        id="fallback_empty",
    ),
    pytest.param(
        _fallback("get_weather", SIMPLE_ARGS_STR),
        "get_weather",
        SIMPLE_ARGS_DICT,
        id="fallback_simple",
    ),
    pytest.param(
        _native("get_weather", SIMPLE_ARGS_STR),
        "get_weather",
        SIMPLE_ARGS_DICT,
        id="native_simple",
    ),
    pytest.param(
        _fallback("complex_func", COMPLEX_ARGS_STR),
        "complex_func",
        COMPLEX_ARGS_DICT,
        id="fallback_complex",
    ),
    pytest.param(
        _native("complex_func", COMPLEX_ARGS_STR),
        "complex_func",
        COMPLEX_ARGS_DICT,
        id="native_complex",
    ),
])
def test_tool_extraction_both_modes(
    streaming,
    model_output,
    func_name,
    expected_args,
    gemma4_tokenizer,
):
    parser = ToolParserManager.get_tool_parser("gemma4")(gemma4_tokenizer)
    content, tool_calls = run_tool_extraction(
        parser, model_output, streaming=streaming
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == func_name
    assert json.loads(tool_calls[0].function.arguments) == expected_args
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
