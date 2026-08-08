# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Kimi K2 streaming tool-argument type coercion.

Kimi K2 emits tool-call arguments as raw JSON and (before this fix) was
the only parser engine backend without an ``arg_converter``.  The
converter-less streaming path skipped ``_fix_arg_types``, so streamed
argument deltas kept the model's raw value types (e.g. the string
``"3"``) while the non-streaming path coerced them to the schema type
(the integer ``3``).  Because streamed deltas are append-only, the raw
wrong-typed characters could not be retracted once emitted.

These tests assert that the concatenation of streamed argument deltas is
byte-for-byte identical to the non-streaming ``extract_tool_calls``
output, across several chunk boundaries.  They fail on the unpatched
backend (streaming keeps ``"3"``, non-streaming yields ``3``) and pass
once the arguments are normalised and coerced consistently.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.parser.kimi_k2 import KimiK2Parser

SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
TOOL_BEGIN = "<|tool_call_begin|>"
TOOL_END = "<|tool_call_end|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"

_VOCAB = {
    "<think>": 200,
    "</think>": 201,
    SECTION_BEGIN: 300,
    SECTION_END: 301,
    TOOL_BEGIN: 302,
    TOOL_END: 303,
    ARG_BEGIN: 304,
}


def _tokenizer() -> MagicMock:
    return make_mock_tokenizer(_VOCAB)


def _make_tool(name: str, properties: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name=name,
            parameters={"type": "object", "properties": properties},
        ),
    )


def _make_request(tools) -> ChatCompletionRequest:
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = tools
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def _sub_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _build_stream_chunks(specs: list[tuple[str, str]], arg_chunk: int) -> list[str]:
    """Build the streamed chunk list for one or more tool calls.

    Special tokens are delivered as their own chunks (mirroring how the
    real tokenizer emits them) while each argument body is split into
    ``arg_chunk``-sized pieces to vary the chunk boundaries.
    """
    chunks = [SECTION_BEGIN]
    for tool_id, args in specs:
        chunks.append(TOOL_BEGIN)
        chunks.append(f"{tool_id} ")
        chunks.append(ARG_BEGIN)
        chunks.extend(_sub_chunks(args, arg_chunk))
        chunks.append(TOOL_END)
    chunks.append(SECTION_END)
    return chunks


def _token_ids_for(chunk: str) -> list[int]:
    return [tid for text, tid in _VOCAB.items() if text in chunk]


def _non_streaming_args(tools, specs: list[tuple[str, str]]) -> list[str]:
    parser = KimiK2Parser(_tokenizer(), tools=tools)
    body = (
        SECTION_BEGIN
        + "".join(
            f"{TOOL_BEGIN}{tool_id} {ARG_BEGIN}{args}{TOOL_END}"
            for tool_id, args in specs
        )
        + SECTION_END
    )
    result = parser.extract_tool_calls(body, _make_request(tools))
    return [tc.function.arguments for tc in result.tool_calls]


def _streaming_args(tools, chunks: list[str]) -> list[str]:
    parser = KimiK2Parser(_tokenizer(), tools=tools)
    req = _make_request(tools)
    prev_text = ""
    prev_ids: list[int] = []
    collected: dict[int, str] = {}

    def _absorb(delta):
        if not delta or not delta.tool_calls:
            return
        for tc in delta.tool_calls:
            if tc.function and tc.function.arguments:
                collected[tc.index] = (
                    collected.get(tc.index, "") + tc.function.arguments
                )

    for chunk in chunks:
        cur_text = prev_text + chunk
        d_ids = _token_ids_for(chunk)
        cur_ids = prev_ids + d_ids
        _absorb(
            parser.extract_tool_calls_streaming(
                previous_text=prev_text,
                current_text=cur_text,
                delta_text=chunk,
                previous_token_ids=tuple(prev_ids),
                current_token_ids=tuple(cur_ids),
                delta_token_ids=tuple(d_ids),
                request=req,
            )
        )
        prev_text = cur_text
        prev_ids = cur_ids
    # Real serving performs a final flush when generation ends.
    _absorb(parser.finish_streaming())
    return [collected[idx] for idx in sorted(collected)]


ARG_CHUNK_SIZES = [1, 3, 1000]


class TestKimiK2StreamingCoercion:
    """Streaming argument bytes must equal the non-streaming output."""

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    @pytest.mark.parametrize(
        "schema_type, raw_value, py_value",
        [
            ("integer", '"3"', 3),
            ("boolean", '"true"', True),
            ("number", '"1.5"', 1.5),
        ],
        ids=["integer", "boolean", "number"],
    )
    def test_single_nonstring_field(self, schema_type, raw_value, py_value, arg_chunk):
        tools = [_make_tool("f", {"v": {"type": schema_type}})]
        specs = [("functions.f:0", '{"v":' + raw_value + "}")]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        # The coercion actually happened (not just that both agree raw).
        assert json.loads(st[0])["v"] == py_value
        assert type(json.loads(st[0])["v"]) is type(py_value)

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    def test_middle_nonstring_field(self, arg_chunk):
        """A non-string field followed by more JSON (the hard case).

        The coerced middle value shrinks (``"5"`` -> ``5``); the trailing
        field must still stream correctly after it.
        """
        tools = [
            _make_tool(
                "f",
                {
                    "count": {"type": "integer"},
                    "loc": {"type": "string"},
                },
            )
        ]
        specs = [("functions.f:0", '{"count":"5","loc":"NYC"}')]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        parsed = json.loads(st[0])
        assert parsed == {"count": 5, "loc": "NYC"}
        assert type(parsed["count"]) is int

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    def test_nested_nonstring_field(self, arg_chunk):
        tools = [
            _make_tool(
                "f",
                {
                    "inner": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                    },
                    "loc": {"type": "string"},
                },
            )
        ]
        specs = [("functions.f:0", '{"inner":{"n":"7"},"loc":"NYC"}')]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        parsed = json.loads(st[0])
        assert parsed == {"inner": {"n": 7}, "loc": "NYC"}
        assert type(parsed["inner"]["n"]) is int

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    def test_parallel_tool_calls(self, arg_chunk):
        tools = [_make_tool("get_weather", {"count": {"type": "integer"}})]
        specs = [
            ("functions.get_weather:0", '{"count":"3"}'),
            ("functions.get_weather:1", '{"count":"7"}'),
        ]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        assert [json.loads(a) for a in st] == [{"count": 3}, {"count": 7}]
        assert all(type(json.loads(a)["count"]) is int for a in st)

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    def test_string_field_with_numeric_literal(self, arg_chunk):
        """A string-typed field emitted as a bare number must gain quotes."""
        tools = [_make_tool("f", {"id": {"type": "string"}})]
        specs = [("functions.f:0", '{"id":42}')]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        assert json.loads(st[0]) == {"id": "42"}
        assert type(json.loads(st[0])["id"]) is str

    @pytest.mark.parametrize("arg_chunk", ARG_CHUNK_SIZES)
    def test_already_correct_types_unchanged(self, arg_chunk):
        """Correctly-typed input must still round-trip identically."""
        tools = [_make_tool("f", {"count": {"type": "integer"}})]
        specs = [("functions.f:0", '{"count":5}')]

        ns = _non_streaming_args(tools, specs)
        st = _streaming_args(tools, _build_stream_chunks(specs, arg_chunk))

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
        assert json.loads(st[0]) == {"count": 5}

    @pytest.mark.parametrize("arg_chunk", [1, 3, 1000])
    def test_truncated_arguments_flush_consistently(self, arg_chunk):
        """Stream ending mid-value (no closing token) must still agree.

        ``finish_streaming`` injects the TOOL_CALL_END the model never
        emitted; both paths complete and coerce the partial object.
        """
        tools = [_make_tool("f", {"count": {"type": "integer"}})]
        raw = '{"count":"3'  # never closed
        body = SECTION_BEGIN + TOOL_BEGIN + "functions.f:0 " + ARG_BEGIN + raw

        parser = KimiK2Parser(_tokenizer(), tools=tools)
        ns = [
            tc.function.arguments
            for tc in parser.extract_tool_calls(body, _make_request(tools)).tool_calls
        ]

        chunks = [
            SECTION_BEGIN,
            TOOL_BEGIN,
            "functions.f:0 ",
            ARG_BEGIN,
        ] + _sub_chunks(raw, arg_chunk)
        st = _streaming_args(tools, chunks)

        assert st == ns, f"streaming {st!r} != non-streaming {ns!r}"
