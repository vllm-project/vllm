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

    def test_type_conversion_in_non_streaming(self):
        """Non-streaming extraction must convert params using the tool schema."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="toggle",
                parameters={
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "count": {"type": "integer"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}toggle">\n'
            f'{PARAM_START}enabled" string="false">true{PARAM_END}\n'
            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"enabled": True, "count": 42}
        assert isinstance(args["enabled"], bool)
        assert isinstance(args["count"], int)

    def test_string_attr_true_preserves_literal_despite_schema(self):
        """string="true" must keep the value as a string even
        if the schema says integer."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="score",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}score">\n'
            f'{PARAM_START}value" string="true">42{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"value": "42"}
        assert isinstance(args["value"], str)

    def test_string_attr_false_allows_schema_conversion(self):
        """string="false" allows the parser to convert via the tool schema."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="score",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}score">\n'
            f'{PARAM_START}value" string="false">42{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"value": 42}
        assert isinstance(args["value"], int)

    @pytest.mark.skip_global_cleanup
    def test_composed_schema_converts_object_and_array_params(self):
        """Composed JSON Schema types must still drive DSML type coercion."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="set_timer",
                parameters={
                    "type": "object",
                    "properties": {
                        "wait": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "until"},
                                        "date": {"type": "string"},
                                    },
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "for"},
                                        "minutes": {"type": "number"},
                                    },
                                },
                            ],
                        },
                        "patches": {
                            "oneOf": [
                                {"type": "array", "items": {"type": "object"}},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}set_timer">\n'
            f'{PARAM_START}wait" string="false">'
            f'{{"type":"for","minutes":2880}}'
            f"{PARAM_END}\n"
            f'{PARAM_START}patches" string="false">'
            f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
            f"{PARAM_END}\n"
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "wait": {"type": "for", "minutes": 2880},
            "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
        }
        assert isinstance(args["wait"], dict)
        assert isinstance(args["patches"], list)

    @pytest.mark.skip_global_cleanup
    def test_string_attr_true_preserves_literal_for_composed_schema(self):
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="set_timer",
                parameters={
                    "type": "object",
                    "properties": {
                        "wait": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}set_timer">\n'
            f'{PARAM_START}wait" string="true">'
            f'{{"type":"for","minutes":2880}}'
            f"{PARAM_END}\n"
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"wait": '{"type":"for","minutes":2880}'}

    def test_arguments_wrapper_repaired(self):
        """A single 'arguments' wrapper parameter must be unwrapped when it
        is not part of the tool schema and the inner object matches schema fields."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}get_weather">\n'
            f'{PARAM_START}arguments" string="false">'
            f'{{"location":"Beijing"}}'
            f"{PARAM_END}\n"
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "Beijing"}

    def test_input_wrapper_repaired(self):
        """A single 'input' wrapper parameter must be unwrapped similarly."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}get_weather">\n'
            f'{PARAM_START}input" string="true">'
            f'{{"location":"Beijing"}}'
            f"{PARAM_END}\n"
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "Beijing"}

    def test_object_and_array_params(self):
        """Object and array schema types must be JSON-parsed, not left as strings."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="update",
                parameters={
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array"},
                        "meta": {"type": "object"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}update">\n'
            f'{PARAM_START}tags" string="false">["a", "b"]{PARAM_END}\n'
            f'{PARAM_START}meta" string="false">{{"k": 1}}{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["tags"] == ["a", "b"]
        assert isinstance(args["tags"], list)
        assert args["meta"] == {"k": 1}
        assert isinstance(args["meta"], dict)

    def test_number_param(self):
        """Number (float) schema type must be converted correctly."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="measure",
                parameters={
                    "type": "object",
                    "properties": {
                        "ratio": {"type": "number"},
                        "whole": {"type": "number"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}measure">\n'
            f'{PARAM_START}ratio" string="false">3.14{PARAM_END}\n'
            f'{PARAM_START}whole" string="false">5.0{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == pytest.approx(3.14)
        assert args["whole"] == 5
        assert isinstance(args["whole"], int)

    def test_multi_typed_schema(self):
        """Schema with type: ["integer", "null"] must handle both cases."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="set_val",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": ["integer", "null"]},
                        "label": {"type": ["string", "null"]},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}set_val">\n'
            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
            f'{PARAM_START}label" string="false">hello{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["count"] == 42
        assert isinstance(args["count"], int)
        assert args["label"] == "hello"

    def test_multi_typed_null_value(self):
        """Literal 'null' must become None when the schema includes 'null'."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="clear",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": ["integer", "null"]},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}clear">\n'
            f'{PARAM_START}value" string="false">null{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["value"] is None

    def test_null_not_coerced_without_null_in_schema(self):
        """Literal 'null' must stay as a string when the schema is just 'string'."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="echo",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}echo">\n'
            f'{PARAM_START}text" string="false">null{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["text"] == "null"
        assert isinstance(args["text"], str)

    def test_no_schema_keeps_strings(self):
        """Without a tool schema, all string='false' params default to string."""
        parser = make_parser(tools=None)
        model_output = (
            f"{FC_START}\n"
            f'{INV_START}unknown_fn">\n'
            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
            f'{PARAM_START}flag" string="false">true{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        result = parser.extract_tool_calls(model_output, None)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["count"] == "42"
        assert args["flag"] == "true"


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
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}add">\n'
            f'{PARAM_START}x" string="false">3{PARAM_END}\n'
            f'{PARAM_START}y" string="false">4{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args_str = self._reconstruct_args(deltas)
        assert json.loads(args_str) == {"x": 3, "y": 4}

    def test_string_attr_true_preserves_literal_in_streaming(self):
        """Streaming: string='true' must keep the value literal despite schema."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="score",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}score">\n'
            f'{PARAM_START}value" string="true">42{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args_str = self._reconstruct_args(deltas)
        args = json.loads(args_str)
        assert args == {"value": "42"}
        assert isinstance(args["value"], str)

    @pytest.mark.skip_global_cleanup
    def test_composed_schema_conversion_in_streaming(self):
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="set_timer",
                parameters={
                    "type": "object",
                    "properties": {
                        "wait": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ],
                        },
                        "patches": {
                            "oneOf": [
                                {"type": "array", "items": {"type": "object"}},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}set_timer">\n'
            f'{PARAM_START}wait" string="false">'
            f'{{"type":"for","minutes":2880}}'
            f"{PARAM_END}\n"
            f'{PARAM_START}patches" string="false">'
            f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
            f"{PARAM_END}\n"
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args = json.loads(self._reconstruct_args(deltas))
        assert args == {
            "wait": {"type": "for", "minutes": 2880},
            "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
        }

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

    def test_object_and_array_params_streaming(self):
        """Streaming: object/array params must be JSON-parsed."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="update",
                parameters={
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array"},
                        "meta": {"type": "object"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}update">\n'
            f'{PARAM_START}tags" string="false">["a", "b"]{PARAM_END}\n'
            f'{PARAM_START}meta" string="false">{{"k": 1}}{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args = json.loads(self._reconstruct_args(deltas))
        assert args["tags"] == ["a", "b"]
        assert args["meta"] == {"k": 1}

    def test_multi_typed_schema_streaming(self):
        """Streaming: type: ["integer", "null"] must coerce correctly."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="set_val",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": ["integer", "null"]},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}set_val">\n'
            f'{PARAM_START}count" string="false">42{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args = json.loads(self._reconstruct_args(deltas))
        assert args["count"] == 42
        assert isinstance(args["count"], int)

    def test_multi_typed_null_streaming(self):
        """Streaming: 'null' with ["integer", "null"] schema must become None."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="clear",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": ["integer", "null"]},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}clear">\n'
            f'{PARAM_START}value" string="false">null{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args = json.loads(self._reconstruct_args(deltas))
        assert args["value"] is None

    def test_number_param_streaming(self):
        """Streaming: number type must be converted."""
        tool = ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="measure",
                parameters={
                    "type": "object",
                    "properties": {
                        "ratio": {"type": "number"},
                    },
                },
            ),
        )
        parser = make_parser(tools=[tool])
        full_text = (
            f"{FC_START}\n"
            f'{INV_START}measure">\n'
            f'{PARAM_START}ratio" string="false">3.14{PARAM_END}\n'
            f"{INV_END}\n"
            f"{FC_END}"
        )
        deltas = self._stream(parser, full_text)
        args = json.loads(self._reconstruct_args(deltas))
        assert args["ratio"] == pytest.approx(3.14)


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
