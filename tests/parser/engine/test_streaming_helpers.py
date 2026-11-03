# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared streaming simulation helpers."""

import json

import pytest

from tests.parser.engine.streaming_helpers import simulate_tool_streaming
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import TOOL_CALL_END, TOOL_CALL_START, qwen3_config

FUNC_OPEN = "<function=read_file>"
PARAM_DESC_OPEN = "<parameter=description>"
PARAM_PATH_OPEN = "<parameter=path>"
PARAM_CLOSE = "</parameter>"
FUNC_CLOSE = "</function>"


class ConsistentTokenizer:
    """Tool-call markers encode to single special ids; every other character
    encodes to its own regular id, so decode(encode(text)) == text."""

    def __init__(self) -> None:
        self._vocab = {TOOL_CALL_START: 100, TOOL_CALL_END: 101}
        self._by_id = {100: TOOL_CALL_START, 101: TOOL_CALL_END}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._vocab)

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._vocab.values())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for marker, tid in self._vocab.items():
                if text.startswith(marker, i):
                    ids.append(tid)
                    i += len(marker)
                    break
            else:
                ids.append(256 + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for tid in ids:
            if tid in self._by_id:
                parts.append(self._by_id[tid])
            elif 256 <= tid < 256 + 0x110000:
                parts.append(chr(tid - 256))
        return "".join(parts)


def _xml_call(description: str, path: str) -> str:
    return (
        TOOL_CALL_START
        + FUNC_OPEN
        + f"{PARAM_DESC_OPEN}{description}{PARAM_CLOSE}"
        + f"{PARAM_PATH_OPEN}{path}{PARAM_CLOSE}"
        + FUNC_CLOSE
        + TOOL_CALL_END
    )


def _two_call_input() -> tuple[str, list[dict]]:
    raw = _xml_call("Read first file", "/tmp/first.txt") + _xml_call(
        "Read second file", "/tmp/second.txt"
    )
    expected = [
        {
            "name": "read_file",
            "arguments": {
                "description": "Read first file",
                "path": "/tmp/first.txt",
            },
        },
        {
            "name": "read_file",
            "arguments": {
                "description": "Read second file",
                "path": "/tmp/second.txt",
            },
        },
    ]
    return raw, expected


def _new_parser() -> ParserEngine:
    return ParserEngine(
        ConsistentTokenizer(),
        parser_engine_config=qwen3_config(thinking=False),
    )


def _streamed_calls(parser, request, chunks, delta_token_ids_per_chunk=None):
    """Stream `chunks` through the helper and return the parsed calls."""
    deltas = [
        delta
        for delta, _current in simulate_tool_streaming(
            parser, request, chunks, delta_token_ids_per_chunk
        )
    ]
    deltas.append(parser.finish_streaming())
    calls: dict[int, dict] = {}
    for delta in deltas:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                entry = calls.setdefault(tc.index, {"name": "", "arguments": ""})
                if tc.function:
                    if tc.function.name:
                        entry["name"] += tc.function.name
                    if tc.function.arguments:
                        entry["arguments"] += tc.function.arguments
    return [
        {"name": calls[i]["name"], "arguments": json.loads(calls[i]["arguments"])}
        for i in sorted(calls)
    ]


def test_explicit_token_ids_parse_layout_that_synthesized_ids_drop(mock_request):
    """Streaming real per-token ids parses a layout where the synthesized
    marker-only ids drop the first tool call."""
    raw, expected = _two_call_input()
    # Chunk 1 ends just past the start marker: with the default marker-only
    # ids the engine has already switched to token-id-authoritative matching
    # and drops the first call (see #65328); explicit ids reproduce serving and
    # keep both.
    split = len(TOOL_CALL_START) + 1
    chunks = [raw[:split], raw[split:]]
    tokenizer = ConsistentTokenizer()
    assert (
        _streamed_calls(
            _new_parser(),
            mock_request,
            chunks,
            [tuple(tokenizer.encode(chunk)) for chunk in chunks],
        )
        == expected
    )


@pytest.mark.parametrize("layout", ["whole", "between_calls", "char_by_char"])
def test_explicit_ids_match_synthesized_ids_on_well_formed_layouts(
    mock_request, layout
):
    """Explicit per-chunk ids agree with the synthesized default and with the
    expected calls on layouts both modes parse correctly."""
    raw, expected = _two_call_input()
    first_len = len(_xml_call("Read first file", "/tmp/first.txt"))
    chunks = {
        "whole": [raw],
        "between_calls": [raw[:first_len], raw[first_len:]],
        "char_by_char": list(raw),
    }[layout]
    tokenizer = ConsistentTokenizer()
    per_chunk = [tuple(tokenizer.encode(chunk)) for chunk in chunks]
    synthesized = _streamed_calls(_new_parser(), mock_request, chunks)
    explicit = _streamed_calls(_new_parser(), mock_request, chunks, per_chunk)
    assert synthesized == explicit == expected
