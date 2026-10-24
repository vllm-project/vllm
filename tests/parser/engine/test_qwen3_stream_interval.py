# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 streaming tool-call tests across stream intervals (#45256).

With ``--stream-interval > 1`` several tokens are batched into one delta, so
the closing ``</parameter>``/``</function>``/``</tool_call>`` tags can arrive
together. These tests feed the full per-token ``delta_token_ids`` (as the
serving layer does) regrouped at various intervals and check that the streamed
arguments stay valid JSON regardless of where the delta boundaries fall.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import qwen3_config

# <tool_call>/</tool_call> are special tokens; everything else is sub-word text.
_SPECIAL_VOCAB = {"<tool_call>": 100, "</tool_call>": 101}


def _make_tokenizer(id_to_text: dict[int, str]) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(_SPECIAL_VOCAB)
    tokenizer.encode.return_value = [1]
    tokenizer.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, f"<{i}>") for i in ids
    )
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        setattr(tokenizer, attr, None)
    return tokenizer


def _stream_tool_call(
    token_stream: list[tuple[str, int]],
    interval: int,
) -> tuple[str | None, str]:
    """Drive the streaming parser over ``token_stream`` regrouped into deltas
    of ``interval`` tokens, returning ``(function_name, accumulated_args)``."""
    id_to_text: dict[int, str] = {}
    for text, tid in token_stream:
        id_to_text.setdefault(tid, text)

    parser = ParserEngine(
        _make_tokenizer(id_to_text),
        parser_engine_config=qwen3_config(thinking=False),
    )
    request = MagicMock()
    request.tools = []
    request.tool_choice = "auto"

    prev_text = ""
    prev_ids: list[int] = []
    name: str | None = None
    args = ""

    for start in range(0, len(token_stream), interval):
        group = token_stream[start : start + interval]
        delta_text = "".join(text for text, _ in group)
        delta_ids = [tid for _, tid in group]
        cur_text = prev_text + delta_text
        cur_ids = prev_ids + delta_ids

        delta = parser.extract_tool_calls_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=delta_text,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=tuple(delta_ids),
            request=request,
        )
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    name = tc.function.name
                if tc.function and tc.function.arguments:
                    args += tc.function.arguments

        prev_text = cur_text
        prev_ids = cur_ids

    return name, args


# read_pdf(page=1) from the #45256 repro, tags split into sub-word tokens.
_SINGLE_PARAM_TOKENS: list[tuple[str, int]] = [
    ("<tool_call>", 100),
    ("\n", 1),
    ("<function", 2),
    ("=read", 3),
    ("_pdf", 4),
    (">", 5),
    ("\n", 1),
    ("<parameter", 6),
    ("=page", 7),
    (">", 5),
    ("\n", 1),
    ("1", 8),
    ("\n", 1),
    ("</parameter", 9),
    (">", 5),
    ("\n", 1),
    ("</function", 10),
    (">", 5),
    ("\n", 1),
    ("</tool_call>", 101),
]

_MULTI_PARAM_TOKENS: list[tuple[str, int]] = [
    ("<tool_call>", 100),
    ("\n", 1),
    ("<function", 2),
    ("=get", 3),
    ("_weather", 4),
    (">", 5),
    ("\n", 1),
    ("<parameter", 6),
    ("=city", 7),
    (">", 5),
    ("\n", 1),
    ("Tokyo", 8),
    ("\n", 1),
    ("</parameter", 9),
    (">", 5),
    ("\n", 1),
    ("<parameter", 6),
    ("=days", 11),
    (">", 5),
    ("\n", 1),
    ("5", 12),
    ("\n", 1),
    ("</parameter", 9),
    (">", 5),
    ("\n", 1),
    ("</function", 10),
    (">", 5),
    ("\n", 1),
    ("</tool_call>", 101),
]

# From token-at-a-time up to batches that land several tags in one delta.
_INTERVALS = [1, 2, 3, 4, 5, 8, 13, 64]


@pytest.mark.parametrize("interval", _INTERVALS)
def test_single_param_valid_json_across_intervals(interval):
    name, args = _stream_tool_call(_SINGLE_PARAM_TOKENS, interval)
    assert name == "read_pdf"
    assert json.loads(args) == {"page": "1"}, (
        f"stream_interval={interval} produced invalid/incorrect args: {args!r}"
    )


@pytest.mark.parametrize("interval", _INTERVALS)
def test_multi_param_valid_json_across_intervals(interval):
    name, args = _stream_tool_call(_MULTI_PARAM_TOKENS, interval)
    assert name == "get_weather"
    assert json.loads(args) == {"city": "Tokyo", "days": "5"}, (
        f"stream_interval={interval} produced invalid/incorrect args: {args!r}"
    )
