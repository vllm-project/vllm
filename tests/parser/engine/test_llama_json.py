# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Llama 3.x/4 JSON parser (llama3_json /
llama4_json).

Staged commit 5/7: wires up schema-coercion safety and streaming, on top
of truncation repair and full format-shape coverage. Later commits grow
both llama_json.py and this file together.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.llama_json import LlamaJsonParser
from vllm.parser.parser_manager import ParserManager

PYTHON_TAG = "<|python_tag|>"

_LLAMA_VOCAB = {
    PYTHON_TAG: 128010,
    "<|eot_id|>": 128009,
    "<|eom_id|>": 128008,
}


def _mock_tokenizer():
    return make_mock_tokenizer(_LLAMA_VOCAB)


def _parser():
    return LlamaJsonParser(_mock_tokenizer())


def _request(tools=None, tool_choice=None):
    return ChatCompletionRequest(
        model="llama",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
    )


def _stream(parser, request, text: str, chunk_size: int):
    """Character-granularity streaming; markers aren't tokenized here yet
    (the mock-tokenizer marker plumbing lands with phantom-retraction in
    commit 2 -- this stage only needs plain text)."""
    results = []
    previous = ""
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        results.append(
            parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=previous + delta_text,
                delta_text=delta_text,
                previous_token_ids=(),
                current_token_ids=(),
                delta_token_ids=(),
                request=request,
            )
        )
        previous += delta_text
    results.append(parser.finish_streaming())
    return results


def _accumulate(results):
    content_parts = []
    calls: dict[int, dict] = {}
    for delta in results:
        if delta is None:
            continue
        if delta.content:
            content_parts.append(delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"name": None, "args": ""})
            if tc.function and tc.function.name:
                call["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                call["args"] += tc.function.arguments
    return "".join(content_parts), [{"index": i, **calls[i]} for i in sorted(calls)]


class TestBasicCall:
    def test_non_streaming_envelope_with_parameters(self):
        text = '{"name": "get_weather", "parameters": {"city": "SF"}}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == '{"city": "SF"}'

    def test_non_streaming_envelope_with_arguments_key(self):
        text = '{"name": "get_weather", "arguments": {"city": "SF"}}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tool_calls[0].function.arguments == '{"city": "SF"}'

    def test_non_streaming_plain_content_has_no_call(self):
        text = "This is just some text without any tool calls"
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tools_called is False

    def test_streaming_basic_call(self):
        text = '{"name": "get_weather", "parameters": {"city": "SF"}}'
        results = _stream(_parser(), _request(), text, chunk_size=3)
        content, calls = _accumulate(results)
        assert content == ""
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["args"] == '{"city": "SF"}'

    def test_streaming_single_char_chunks(self):
        text = '{"name": "f", "parameters": {"x": 1}}'
        results = _stream(_parser(), _request(), text, chunk_size=1)
        _, calls = _accumulate(results)
        assert len(calls) == 1
        assert calls[0]["name"] == "f"
        assert calls[0]["args"] == '{"x": 1}'


def _function_tool(name, parameters=None):
    function = {"name": name}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _delegating_parser_cls():
    cls = ParserManager.get_parser(
        tool_parser_name="llama3_json",
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert cls is not None
    return cls


class TestNestedNameDisambiguation:
    """The engine's default name extraction is a naive regex matching the
    first ``"name"`` key anywhere; envelope-aware scanning must not."""

    def test_nested_name_key_does_not_misfire(self):
        text = '{"parameters": {"name": "Alice"}, "name": "rename_user"}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tool_calls[0].function.name == "rename_user"
        assert json.loads(result.tool_calls[0].function.arguments) == {"name": "Alice"}

    def test_prose_json_with_a_name_field_is_not_a_call(self):
        text = '{"name": "John", "age": 30}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tools_called is False
        assert text in (result.content or "")


class TestPhantomRetraction:
    """A balanced ``{...}`` with no top-level ``"name"`` is prose, not a
    call, and must be restored as content -- not dropped."""

    def test_prose_before_a_real_call_is_kept_as_content(self):
        text = 'Sure, let me check. {"name": "f", "parameters": {}}'
        results = _stream(_parser(), _request(), text, chunk_size=4)
        content, calls = _accumulate(results)
        assert "Sure, let me check." in content
        assert len(calls) == 1
        assert calls[0]["name"] == "f"

    def test_prose_json_streamed_is_restored_as_content(self):
        text = '{"a": 1, "b": 2}'
        results = _stream(_parser(), _request(), text, chunk_size=3)
        content, calls = _accumulate(results)
        assert calls == []
        assert content == text


class TestMultipleCalls:
    def test_two_sequential_envelopes(self):
        text = '{"name": "f", "parameters": {}}{"name": "g", "parameters": {}}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        names = [tc.function.name for tc in result.tool_calls]
        assert names == ["f", "g"]


class TestToolChoiceNone:
    def test_none_choice_returns_envelope_as_content(self):
        text = '{"name": "f", "parameters": {"x": 1}}'
        req = _request(
            tools=[_function_tool("f", {"type": "object", "properties": {}})],
            tool_choice="none",
        )
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, req
        )
        assert result.tools_called is False
        assert text in (result.content or "")


class TestNamedChoiceBareArguments:
    """A named tool choice with a structural-constraining schema makes the
    model emit bare parameters, with no envelope at all -- ``_bare_args_tool_name``
    must detect this from the request and switch converters.
    """

    PARAMETERS = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    @pytest.fixture
    def parser_cls(self, monkeypatch):
        monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")
        return _delegating_parser_cls()

    def test_named_choice_returns_a_call_not_bare_parameters(self, parser_cls):
        from vllm.sampling_params import StructuredOutputsParams

        tokenizer = _mock_tokenizer()
        parser = parser_cls(tokenizer)
        request = ChatCompletionRequest(
            model="llama",
            messages=[{"role": "user", "content": "weather in SF?"}],
            tools=[_function_tool("get_weather", self.PARAMETERS)],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        # adjust_request is normally what installs this from tool_choice;
        # set it directly here rather than chasing the full server-side
        # request-processing pipeline that isn't part of llama_json itself.
        request.structured_outputs = StructuredOutputsParams(json=self.PARAMETERS)
        _, content, tool_calls = parser.parse(
            '{"city": "SF"}', request, enable_auto_tools=True
        )
        assert [(tc.name, json.loads(tc.arguments)) for tc in tool_calls or []] == [
            ("get_weather", {"city": "SF"})
        ]
        assert content is None


class TestTruncationRepair:
    """A value cut off mid-stream must come back as valid JSON, not raw
    unterminated text -- every streamed delta, not just the final one."""

    def test_mid_string_truncation_is_closed(self):
        from vllm.parser.llama_json import _llama_arg_converter

        raw = '{"name": "f", "parameters": {"city": "San Franci'
        assert _llama_arg_converter(raw, False) == '{"city": "San Franci"}'

    def test_mid_number_truncation_cuts_back_to_last_complete_digit(self):
        from vllm.parser.llama_json import _llama_arg_converter

        raw = '{"name": "f", "parameters": {"x": 1.'
        assert _llama_arg_converter(raw, False) == '{"x": 1}'

    def test_streamed_arguments_close_cleanly_even_char_by_char(self):
        """The accumulated argument text need not be valid JSON mid-stream
        (partial mode never appends closers), but once the call finishes
        the accumulated text must be exactly the closed, valid object --
        not left truncated the way it would be without repair."""
        text = '{"name": "f", "parameters": {"city": "San Francisco", "unit": "C"}}'
        results = _stream(_parser(), _request(), text, chunk_size=1)
        _, calls = _accumulate(results)
        assert json.loads(calls[0]["args"]) == {"city": "San Francisco", "unit": "C"}


class TestDeclaredTypes:
    """Groundwork for schema-coercion safety: which JSON types a schema
    actually permits, not the shared engine's string fallback."""

    def test_empty_schema_permits_every_type(self):
        from vllm.parser.llama_json import _declared_types

        assert _declared_types({}) == {
            "null",
            "boolean",
            "integer",
            "number",
            "string",
            "array",
            "object",
        }

    def test_typed_schema_restricts_to_declared_type(self):
        from vllm.parser.llama_json import _declared_types

        assert _declared_types({"type": "string"}) == {"string"}

    def test_number_type_also_permits_integer(self):
        from vllm.parser.llama_json import _declared_types

        assert _declared_types({"type": "number"}) == {"number", "integer"}

    def test_is_valid_for_schema_type_true_for_already_matching_value(self):
        from vllm.parser.llama_json import _is_valid_for_schema_type

        assert _is_valid_for_schema_type(7, {"type": "integer"}) is True
        assert _is_valid_for_schema_type("7", {"type": "integer"}) is False


class TestSchemaCoercionSafety:
    """Coercion must not corrupt an already-valid value -- the shared
    engine's ``_coerce_value`` rewrites unconditionally, e.g. turning an
    already-correct int into a string under an ambiguous ``{}`` schema."""

    def test_already_valid_int_is_not_rewritten_as_string(self):
        from vllm.parser.llama_json import _splice_types

        raw = '{"n": 7}'
        result = _splice_types(raw, 0, len(raw), {"properties": {"n": {}}})
        assert result == raw

    def test_wrong_type_value_is_coerced(self):
        from vllm.parser.llama_json import _splice_types

        raw = '{"n": "7"}'
        schema = {"properties": {"n": {"type": "integer"}}}
        result = _splice_types(raw, 0, len(raw), schema)
        assert result == '{"n": 7}'


class TestSchemaCoercionStreamingParity:
    """Schema coercion must not silently drop a correction -- the shared
    engine re-serialises at flush, which fails the append-only streaming
    guard and leaves the uncoerced verbatim text streamed instead."""

    def test_coerced_value_streams_matching_non_streaming(self):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        }
        tool = _function_tool("f", schema)
        text = '{"name": "f", "parameters": {"n": "7"}}'

        non_streaming = LlamaJsonParser(
            _mock_tokenizer()
        ).extract_tool_calls_from_content(text, _request(tools=[tool]))
        streamed_results = _stream(
            _parser(), _request(tools=[tool]), text, chunk_size=1
        )
        _, calls = _accumulate(streamed_results)

        assert json.loads(non_streaming.tool_calls[0].function.arguments) == {"n": 7}
        assert json.loads(calls[0]["args"]) == {"n": 7}
