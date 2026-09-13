# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import string

import pytest

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser

pytestmark = pytest.mark.skip_global_cleanup


class CharTokenizer:
    """Character-level tokenizer: MuseGlimmer's ATEM markers are multi-token,
    matching the property the parser is written against."""

    def __init__(self):
        self._chars = {c: i for i, c in enumerate(string.printable)}
        self._ids = {i: c for c, i in self._chars.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._chars[c] for c in text]

    def decode(self, token_ids) -> str:
        return "".join(self._ids[i] for i in token_ids)


@pytest.fixture
def parser_and_tokenizer():
    tokenizer = CharTokenizer()
    return MuseGlimmerReasoningParser(tokenizer), tokenizer


def test_counts_closed_reasoning_span(parser_and_tokenizer):
    parser, tok = parser_and_tokenizer
    text = "to=self<|message|>think hard<|eom|>to=user<|message|>hi<|eot|>"
    assert parser.count_reasoning_tokens(tok.encode(text)) == len("think hard")


def test_counts_multiple_reasoning_spans(parser_and_tokenizer):
    parser, tok = parser_and_tokenizer
    text = (
        "to=self<|message|>abc<|eom|>"
        "to=user<|message|>answer<|eot|>"
        "to=self<|message|>de<|eom|>"
    )
    assert parser.count_reasoning_tokens(tok.encode(text)) == len("abc") + len("de")


def test_counts_open_reasoning_span_mid_stream(parser_and_tokenizer):
    parser, tok = parser_and_tokenizer
    text = "to=self<|message|>partial thought"
    assert parser.count_reasoning_tokens(tok.encode(text)) == len("partial thought")


def test_content_only_counts_zero(parser_and_tokenizer):
    parser, tok = parser_and_tokenizer
    text = "to=user<|message|>plain answer<|eot|>"
    assert parser.count_reasoning_tokens(tok.encode(text)) == 0


def test_empty_counts_zero(parser_and_tokenizer):
    parser, _ = parser_and_tokenizer
    assert parser.count_reasoning_tokens([]) == 0


def test_mid_generation_turn_reopen(parser_and_tokenizer):
    """Regression: real generations re-open the assistant turn before the
    answer channel; the reasoning before it must still be counted."""
    parser, tok = parser_and_tokenizer
    text = (
        " to=self<|message|>Simple. 2+2 = 4<|eom|>"
        "<|start|>assistant to=user<|message|>2 + 2 = 4"
    )
    assert parser.count_reasoning_tokens(tok.encode(text)) == len("Simple. 2+2 = 4")


def test_tool_channel_not_counted(parser_and_tokenizer):
    parser, tok = parser_and_tokenizer
    text = (
        "to=self<|message|>pick a tool<|eom|>"
        'to=functions.get_weather<|message|>{"city": "SF"}<|eom|>'
    )
    assert parser.count_reasoning_tokens(tok.encode(text)) == len("pick a tool")
