# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV32ToolParser.

These tests use a minimal mock tokenizer so no real model weights are required.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Token IDs are not used by the V32 parser logic, so we only need the
# tokenizer object to be truthy (the parser checks `if not self.model_tokenizer`).
MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}
MOCK_TOKENIZER.tokenize.return_value = []


def make_parser(tools=None) -> DeepSeekV32ToolParser:
    return DeepSeekV32ToolParser(MOCK_TOKENIZER, tools=tools)


def make_tool_param(name: str, params: dict) -> MagicMock:
    """Build a mock tool matching the ChatCompletionToolsParam shape."""
    tool = MagicMock()
    tool.function.name = name
    tool.function.parameters = params
    return tool


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


# Shorthand for the DSML tokens used throughout
FC_START = "<｜DSML｜function_calls>"
FC_END = "</｜DSML｜function_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    """Build a complete model-output tool call string."""
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}' for k, v in params.items()
    )
    return f'{FC_START}\n{INV_START}{func_name}">\n{param_strs}\n{INV_END}\n{FC_END}'


# ---------------------------------------------------------------------------
# Tests: DeepSeekV32ToolParser._convert_param_value
# ---------------------------------------------------------------------------


class TestConvertParamValue:
    @pytest.fixture
    def parser(self):
        return make_parser()

    def test_null(self, parser):
        assert parser._convert_param_value("null", "string") is None
        assert parser._convert_param_value("NULL", "integer") is None

    def test_string(self, parser):
        assert parser._convert_param_value("hello", "string") == "hello"

    def test_integer_valid(self, parser):
        assert parser._convert_param_value("42", "integer") == 42

    def test_integer_invalid_falls_back_to_str(self, parser):
        assert parser._convert_param_value("abc", "int") == "abc"

    def test_number_float(self, parser):
        assert parser._convert_param_value("3.14", "number") == pytest.approx(3.14)

    def test_number_whole_returns_int(self, parser):
        assert parser._convert_param_value("5.0", "number") == 5
        assert isinstance(parser._convert_param_value("5.0", "number"), int)

    def test_boolean_true(self, parser):
        assert parser._convert_param_value("true", "boolean") is True
        assert parser._convert_param_value("1", "bool") is True

    def test_boolean_false(self, parser):
        assert parser._convert_param_value("false", "boolean") is False
        assert parser._convert_param_value("False", "bool") is False

    def test_object_valid_json(self, parser):
        assert parser._convert_param_value('{"k": 1}', "object") == {"k": 1}

    def test_object_invalid_json_falls_back(self, parser):
        assert parser._convert_param_value("not-json", "object") == "not-json"

    def test_array_valid_json(self, parser):
        assert parser._convert_param_value("[1, 2]", "array") == [1, 2]

    def test_unknown_type_tries_json_then_string(self, parser):
        assert parser._convert_param_value("123", "unknown") == 123
        assert parser._convert_param_value("hello", "unknown") == "hello"


# ---------------------------------------------------------------------------
# Tests: extract_tool_calls (non-streaming)
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    @pytest.fixture
    def parser(self):
        return make_parser()

    def test_no_tool_call(self, parser):
        result = parser.extract_tool_calls("just some text", None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "just some text"

    def test_single_tool_no_params(self, parser):
        model_output = f'{FC_START}\n{INV_START}get_time">\n{INV_END}\n{FC_END}'
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_time"
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_single_tool_with_params(self, parser):
        model_output = build_tool_call(
            "get_weather", {"location": "SF", "date": "2024-01-16"}
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {
            "location": "SF",
            "date": "2024-01-16",
        }

    def test_content_before_tool_call(self, parser):
        model_output = "Sure, let me check! " + build_tool_call(
            "get_weather", {"location": "NYC"}
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert result.content == "Sure, let me check! "

    def test_no_content_prefix_returns_none(self, parser):
        model_output = build_tool_call("get_weather", {"location": "NYC"})
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert result.content is None

    def test_multiple_tools(self, parser):
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}get_weather">\n'
            f'{PARAM_START}location" string="true">SF{PARAM_END}\n'
            f"{INV_END}\n"
            f'{INV_START}get_weather">\n'
            f'{PARAM_START}location" string="true">NYC{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "SF"}
        assert json.loads(result.tool_calls[1].function.arguments) == {
            "location": "NYC"
        }


# ---------------------------------------------------------------------------
# Tests: extract_tool_calls_streaming
# ---------------------------------------------------------------------------


class TestExtractToolCallsStreaming:
    """Simulate character-by-character streaming and verify reconstructed args."""

    @pytest.fixture
    def parser(self):
        return make_parser()

    def _stream(self, parser, full_text: str, request=None):
        """Drive the parser line-by-line and collect non-None deltas.

        Real tokenizers emit multi-character chunks, not individual characters.
        Streaming character-by-character would never deliver the full sentinel
        token (e.g. '｜DSML｜') in a single delta, so we split on newlines to
        ensure each sentinel always lands in one chunk.
        """
        if request is None:
            request = make_request()
        # Split into lines, preserving the trailing newline in each chunk.
        chunks: list[str] = []
        remaining = full_text
        while remaining:
            nl = remaining.find("\n")
            if nl == -1:
                chunks.append(remaining)
                break
            chunks.append(remaining[: nl + 1])
            remaining = remaining[nl + 1 :]

        deltas = []
        prev = ""
        for chunk in chunks:
            curr = prev + chunk
            result = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[1],
                request=request,
            )
            prev = curr
            if result is not None:
                deltas.append(result)
        return deltas

    def _reconstruct_args(self, deltas, tool_index=0) -> str:
        """Concatenate all argument fragments for a given tool index."""
        fragments = []
        for d in deltas:
            if d.tool_calls:
                for tc in d.tool_calls:
                    if tc.index == tool_index and tc.function and tc.function.arguments:
                        fragments.append(tc.function.arguments)
        return "".join(fragments)

    def test_plain_content_no_tool(self, parser):
        full_text = "Hello, world!"
        deltas = self._stream(parser, full_text)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert "Hello, world!" in content
        assert all(not d.tool_calls for d in deltas)

    def test_single_tool_streaming(self, parser):
        full_text = build_tool_call("get_weather", {"location": "SF"})
        deltas = self._stream(parser, full_text)
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"location": "SF"}

    def test_tool_name_emitted(self, parser):
        full_text = build_tool_call("my_func", {"x": "1"})
        deltas = self._stream(parser, full_text)
        func_names = [
            tc.function.name
            for d in deltas
            if d.tool_calls
            for tc in d.tool_calls
            if tc.function and tc.function.name
        ]
        assert any("my_func" in n for n in func_names)

    def test_content_before_tool_call_streaming(self, parser):
        full_text = "Thinking... " + build_tool_call("fn", {"a": "b"})
        deltas = self._stream(parser, full_text)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert "Thinking" in content

    def test_type_conversion_in_streaming(self):
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="add",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = build_tool_call("add", {"x": "3", "y": "4"})
        deltas = self._stream(parser, full_text)
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"x": 3, "y": 4}

    def test_multiple_tools_streaming(self, parser):
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}func_a">\n'
            f'{PARAM_START}p" string="true">v1{PARAM_END}\n'
            f"{INV_END}\n"
            f'{INV_START}func_b">\n'
            f'{PARAM_START}q" string="true">v2{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)

        # Collect function names by index
        names_by_index: dict[int, str] = {}
        for d in deltas:
            if d.tool_calls:
                for tc in d.tool_calls:
                    if tc.function and tc.function.name:
                        names_by_index[tc.index] = tc.function.name

        assert names_by_index.get(0) == "func_a"
        assert names_by_index.get(1) == "func_b"

        assert json.loads(self._reconstruct_args(deltas, tool_index=0)) == {"p": "v1"}
        assert json.loads(self._reconstruct_args(deltas, tool_index=1)) == {"q": "v2"}

    def test_state_reset_on_new_stream(self, parser):
        """A second stream (previous_text == '') must reset state cleanly."""
        full_text = build_tool_call("fn", {"k": "v"})
        # First stream
        self._stream(parser, full_text)
        # Second stream - should produce identical results
        deltas2 = self._stream(parser, full_text)
        assert json.loads(self._reconstruct_args(deltas2)) == {"k": "v"}

    def test_empty_arguments_streaming(self, parser):
        """Invoke block with zero parameters should produce empty JSON."""
        full_text = f'{FC_START}\n{INV_START}get_time">\n{INV_END}\n{FC_END}'
        deltas = self._stream(parser, full_text)
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {}

    def test_unique_tool_call_ids(self, parser):
        """Each tool call in a parallel stream must get a distinct id."""
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}fn_a">\n'
            f'{PARAM_START}x" string="true">1{PARAM_END}\n'
            f"{INV_END}\n"
            f'{INV_START}fn_b">\n'
            f'{PARAM_START}y" string="true">2{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        ids = [
            tc.id
            for d in deltas
            if d.tool_calls
            for tc in d.tool_calls
            if tc.id is not None
        ]
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_eos_after_tool_calls(self, parser):
        """EOS token (empty delta_text, non-empty delta_token_ids) returns
        a non-None DeltaMessage so the serving framework can finalize."""
        full_text = build_tool_call("fn", {"k": "v"})
        # Drive through the full text first
        deltas = self._stream(parser, full_text)
        assert any(d.tool_calls for d in deltas)
        # Now simulate EOS: empty delta_text, but token ids present
        prev = full_text
        result = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev,
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[2],  # EOS token id
            request=make_request(),
        )
        assert result is not None

    def test_streaming_matches_non_streaming(self, parser):
        """Streaming and non-streaming must produce the same result."""
        full_text = build_tool_call(
            "get_weather", {"location": "SF", "date": "2024-01-16"}
        )
        # Non-streaming
        non_stream = parser.extract_tool_calls(full_text, None)
        assert non_stream.tools_called
        ns_name = non_stream.tool_calls[0].function.name
        ns_args = json.loads(non_stream.tool_calls[0].function.arguments)
        # Streaming
        deltas = self._stream(parser, full_text)
        s_names = [
            tc.function.name
            for d in deltas
            if d.tool_calls
            for tc in d.tool_calls
            if tc.function and tc.function.name
        ]
        s_args = json.loads(self._reconstruct_args(deltas))
        assert s_names[0] == ns_name
        assert s_args == ns_args

    def _stream_chunked(self, parser, full_text: str, chunk_size: int, request=None):
        """Drive the parser with fixed-size chunks (simulates stream interval).

        Unlike ``_stream`` which splits on newlines, this splits the text
        into ``chunk_size``-character pieces so the start token can be
        split across chunks — exactly what happens with stream interval > 1.
        """
        if request is None:
            request = make_request()
        chunks = [
            full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)
        ]
        deltas = []
        prev = ""
        for chunk in chunks:
            curr = prev + chunk
            result = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[1],
                request=request,
            )
            prev = curr
            if result is not None:
                deltas.append(result)
        return deltas

    def test_single_tool_chunked_stream_interval(self, parser):
        """Start token split across chunks (stream interval > 1)."""
        full_text = build_tool_call("get_weather", {"location": "SF"})
        # Use a chunk size that splits the start token
        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"location": "SF"}

    def test_content_before_tool_chunked(self, parser):
        """Content before tool call with chunked streaming."""
        full_text = "Thinking... " + build_tool_call("fn", {"a": "b"})
        deltas = self._stream_chunked(parser, full_text, chunk_size=7)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert "Thinking" in content
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"a": "b"}

    def test_multiple_tools_chunked(self, parser):
        """Multiple tools with chunked streaming."""
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}func_a">\n'
            f'{PARAM_START}p" string="true">v1{PARAM_END}\n'
            f"{INV_END}\n"
            f'{INV_START}func_b">\n'
            f'{PARAM_START}q" string="true">v2{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream_chunked(parser, full_text, chunk_size=10)
        assert json.loads(self._reconstruct_args(deltas, tool_index=0)) == {"p": "v1"}
        assert json.loads(self._reconstruct_args(deltas, tool_index=1)) == {"q": "v2"}

    def test_no_emission_while_incomplete(self, parser):
        """No tool calls should be emitted until an invoke block completes."""
        # Stream only a partial invoke (no closing tag)
        partial_text = (
            f"{FC_START}\n"
            f'{INV_START}fn">\n'
            f'{PARAM_START}k" string="true">val{PARAM_END}\n'
        )
        deltas = self._stream(parser, partial_text)
        # Should have no tool call deltas yet
        assert all(not d.tool_calls for d in deltas)

    def test_no_marker_leak_chunked(self, parser):
        """Chunked streaming must NOT leak DSML start-marker fragments
        as content (GitHub #40801)."""
        full_text = build_tool_call("fn", {"k": "v"})
        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert content == ""
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"k": "v"}

    def test_no_marker_leak_with_prefix_chunked(self, parser):
        """Content before a tool call must not include start-marker
        fragments when chunked (GitHub #40801)."""
        full_text = "Hello!" + build_tool_call("fn", {"a": "b"})
        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert content == "Hello!"
        assert "DSML" not in content
        assert "<｜" not in content
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"a": "b"}

    def test_no_marker_leak_char_by_char(self, parser):
        """Character-by-character streaming must not leak marker
        fragments (GitHub #40801)."""
        full_text = build_tool_call("fn", {"k": "v"})
        deltas = self._stream_chunked(parser, full_text, chunk_size=1)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert content == ""
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"k": "v"}

    def test_no_marker_leak_all_split_points(self, parser):
        """Start token split at every possible boundary must not
        leak (GitHub #40801)."""
        for chunk_size in range(1, len(FC_START) + 2):
            p = make_parser()
            full_text = build_tool_call("fn", {"k": "v"})
            deltas = self._stream_chunked(p, full_text, chunk_size=chunk_size)
            content = "".join(d.content for d in deltas if d.content is not None)
            assert content == "", (
                f"Leaked content {content!r} at chunk_size={chunk_size}"
            )

    def test_false_partial_marker_emitted(self, parser):
        """Text ending with a prefix of the start token that turns out
        NOT to be a marker must still be emitted as content."""
        full_text = "<｜DSM some regular text"
        deltas = self._stream_chunked(parser, full_text, chunk_size=3)
        content = "".join(d.content for d in deltas if d.content is not None)
        assert content == full_text


class TestDelimiterPreservation:
    """Regression: fast detokenization skipping DSML delimiters (PR #33964)."""

    @pytest.fixture
    def parser(self):
        return make_parser()

    def test_delimiter_preserved_fast_detokenization(self, parser):
        """DSML delimiters as literal text must still be detected."""
        # Delimiters appear as regular text (fast detokenization scenario).
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}get_weather">\n'
            f'{PARAM_START}location" string="true">Tokyo{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )

        # Non-streaming: parser must detect the tool call
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "location": "Tokyo"
        }

        assert result.content is None

        # With content prefix
        prefixed_output = "Here is the weather: " + model_output
        result2 = parser.extract_tool_calls(prefixed_output, None)
        assert result2.tools_called
        assert result2.content == "Here is the weather: "

    def test_tool_detection_skip_special_tokens_false(self, parser):
        """Regression: skip_special_tokens must be False when tools are enabled."""
        # adjust_request must set skip_special_tokens=False
        tool = make_tool_param(
            "search",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )
        request = make_request(tools=[tool])
        request.tool_choice = "auto"
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False

        full_text = build_tool_call("search", {"query": "vllm documentation"})

        # Non-streaming extraction
        non_stream_result = parser.extract_tool_calls(full_text, request)
        assert non_stream_result.tools_called
        assert len(non_stream_result.tool_calls) == 1
        assert non_stream_result.tool_calls[0].function.name == "search"
        ns_args = json.loads(non_stream_result.tool_calls[0].function.arguments)
        assert ns_args == {"query": "vllm documentation"}

        # Streaming extraction: drive the parser line-by-line
        chunks: list[str] = []
        remaining = full_text
        while remaining:
            nl = remaining.find("\n")
            if nl == -1:
                chunks.append(remaining)
                break
            chunks.append(remaining[: nl + 1])
            remaining = remaining[nl + 1 :]

        reconstructor = run_tool_extraction_streaming(
            parser, chunks, request, assert_one_tool_per_delta=False
        )
        assert len(reconstructor.tool_calls) == 1
        assert reconstructor.tool_calls[0].function.name == "search"
        streamed_args = json.loads(reconstructor.tool_calls[0].function.arguments)
        assert streamed_args == ns_args


@pytest.fixture(scope="module")
def deepseekv32_tokenizer():
    return get_tokenizer(tokenizer_name="deepseek-ai/DeepSeek-V3.2")


@pytest.fixture
def parser(deepseekv32_tokenizer):
    return DeepSeekV32ToolParser(deepseekv32_tokenizer)


def test_convert_param_value_single_types(parser):
    """Test _convert_param_value with single type parameters."""
    # Test string type
    assert parser._convert_param_value("hello", "string") == "hello"
    assert parser._convert_param_value("123", "string") == "123"

    # Test integer type - valid integers
    assert parser._convert_param_value("123", "integer") == 123
    assert parser._convert_param_value("456", "int") == 456
    # Invalid integer should return original string (due to exception catch)
    assert parser._convert_param_value("abc", "integer") == "abc"

    # Test float/number type
    assert parser._convert_param_value("123.45", "float") == 123.45
    assert (
        parser._convert_param_value("123.0", "number") == 123
    )  # Should be int when whole number
    assert parser._convert_param_value("123.5", "number") == 123.5
    # Invalid float should return original string
    assert parser._convert_param_value("abc", "float") == "abc"

    # Test boolean type - valid boolean values
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("false", "bool") is False
    assert parser._convert_param_value("1", "boolean") is True
    assert parser._convert_param_value("0", "boolean") is False
    # Invalid boolean should return original string
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("no", "bool") == "no"

    # Test null value
    assert parser._convert_param_value("null", "string") is None
    assert parser._convert_param_value("null", "integer") is None

    # Test object/array type (JSON)
    assert parser._convert_param_value('{"key": "value"}', "object") == {"key": "value"}
    assert parser._convert_param_value("[1, 2, 3]", "array") == [1, 2, 3]
    # Invalid JSON should return original string
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"

    # Test fallback for unknown type (tries json.loads, then returns original)
    assert parser._convert_param_value('{"key": "value"}', "unknown") == {
        "key": "value"
    }
    assert parser._convert_param_value("plain text", "unknown") == "plain text"


def test_convert_param_value_multi_typed_values(parser):
    """Test _convert_param_value with multi-typed values (list of types)."""
    # Test with list of types where first type succeeds
    assert parser._convert_param_value("123", ["integer", "string"]) == 123
    assert parser._convert_param_value("true", ["boolean", "string"]) is True
    assert parser._convert_param_value('{"x": 1}', ["object", "string"]) == {"x": 1}

    # Test with list of types where first type fails but second succeeds
    # "abc" is not a valid integer, so should try string next
    assert parser._convert_param_value("abc", ["integer", "string"]) == "abc"

    # Test with list of types where all fail - should return original value
    # "invalid json" is not valid JSON, last type is "object" which will fail JSON parse
    result = parser._convert_param_value("invalid json", ["integer", "object"])
    assert result == "invalid json"  # Returns original value after all types fail

    # Test with three types
    assert parser._convert_param_value("123.5", ["integer", "float", "string"]) == 123.5
    assert parser._convert_param_value("true", ["integer", "boolean", "string"]) is True

    # Test with null in multi-type list
    assert parser._convert_param_value("null", ["integer", "string"]) is None
    assert parser._convert_param_value("null", ["boolean", "object"]) is None

    # Test nested type conversion - boolean fails, integer succeeds
    value = parser._convert_param_value("123", ["boolean", "integer", "string"])
    assert value == 123  # Should be integer, not boolean

    # Test that order matters
    assert (
        parser._convert_param_value("123", ["string", "integer"]) == "123"
    )  # String first
    assert (
        parser._convert_param_value("123", ["integer", "string"]) == 123
    )  # Integer first

    # Test with all types failing - returns original value
    assert (
        parser._convert_param_value("not_a_number", ["integer", "float", "boolean"])
        == "not_a_number"
    )


def test_convert_param_value_stricter_type_checking(parser):
    """Test stricter type checking in the updated implementation."""
    # Boolean now has stricter validation
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value("false", "boolean") is False
    assert parser._convert_param_value("1", "boolean") is True
    assert parser._convert_param_value("0", "boolean") is False

    # These should return original string (not valid boolean values)
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("no", "boolean") == "no"
    assert parser._convert_param_value("TRUE", "boolean") is True
    assert parser._convert_param_value("FALSE", "boolean") is False

    # Integer and float now raise exceptions for invalid values
    assert parser._convert_param_value("123abc", "integer") == "123abc"
    assert parser._convert_param_value("123.45.67", "float") == "123.45.67"

    # JSON parsing is stricter - invalid JSON returns original
    assert parser._convert_param_value("{invalid: json}", "object") == "{invalid: json}"
    assert parser._convert_param_value("[1, 2,", "array") == "[1, 2,"

    # Test multi-type with stricter checking
    # "yes" is not valid boolean, but string would accept it
    assert parser._convert_param_value("yes", ["boolean", "string"]) == "yes"

    # "123abc" is not valid integer or float, but string accepts it
    assert (
        parser._convert_param_value("123abc", ["integer", "float", "string"])
        == "123abc"
    )


def test_convert_param_value_edge_cases(parser):
    """Test edge cases for _convert_param_value."""
    # Empty string
    assert parser._convert_param_value("", "string") == ""
    assert (
        parser._convert_param_value("", "integer") == ""
    )  # Invalid int returns original

    # Whitespace - trimmed by conversion functions
    assert parser._convert_param_value("  123  ", "integer") == 123
    assert parser._convert_param_value("  true  ", "boolean") is True

    # Numeric strings with special characters
    assert parser._convert_param_value("123.45.67", "float") == "123.45.67"
    assert parser._convert_param_value("123abc", "integer") == "123abc"

    # JSON with whitespace - should parse correctly
    assert parser._convert_param_value('  { "key" : "value" }  ', "object") == {
        "key": "value"
    }

    # Invalid JSON returns original
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"
    assert parser._convert_param_value("[1, 2,", "array") == "[1, 2,"


def test_convert_param_value_checked_helper(parser):
    """Test the _convert_param_value_checked helper function indirectly."""
    # This tests the behavior through the main function
    # Valid conversions should work
    assert parser._convert_param_value("123", "integer") == 123
    assert parser._convert_param_value("123.45", "float") == 123.45
    assert parser._convert_param_value("true", "boolean") is True
    assert parser._convert_param_value('{"x": 1}', "object") == {"x": 1}

    # Invalid conversions should return original value (exception caught)
    assert parser._convert_param_value("abc", "integer") == "abc"
    assert parser._convert_param_value("abc", "float") == "abc"
    assert parser._convert_param_value("yes", "boolean") == "yes"
    assert parser._convert_param_value("{invalid}", "object") == "{invalid}"

    # Test that null handling works in checked function
    assert parser._convert_param_value("null", "integer") is None
    assert parser._convert_param_value("null", "boolean") is None
    assert parser._convert_param_value("null", "object") is None
