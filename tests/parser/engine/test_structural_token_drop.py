# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for structural token (BOS/EOS/PAD) drop behavior.

Verifies that tokens in STRUCTURAL_DROP_TOKENS are correctly filtered
in streaming (parse_delta) and non-streaming (parse) paths for both
direct ParserEngine and DelegatingParser wiring.
"""

from __future__ import annotations

import pytest

from tests.parser.engine.replay_harness import (
    CHUNK_SIZES,
    MockTokenizer,
    collect_output,
    replay_streaming,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

BOS_ID = 1
BOS_TEXT = "<bos>"

THINK_END_ID = 51

_VOCAB: dict[str, int] = {
    "<think>": 50,
    "</think>": THINK_END_ID,
    BOS_TEXT: BOS_ID,
}

_THINK_CONFIG = ParserEngineConfig(
    name="think_test",
    terminals={
        "THINK_START": "<think>",
        "THINK_END": "</think>",
    },
    transitions={
        (ParserState.REASONING, "THINK_END"): Transition(
            ParserState.CONTENT,
            (EventType.REASONING_END,),
        ),
    },
    initial_state=ParserState.REASONING,
)

# Token sequence: reasoning → </think> → content with <bos> mid-stream
_TOKENS: list[tuple[int, str]] = [
    (100, "Let "),
    (101, "me "),
    (102, "think."),
    (THINK_END_ID, "</think>"),
    (103, "Hello "),
    (BOS_ID, BOS_TEXT),
    (104, "world"),
]

_FULL_TEXT = "".join(text for _, text in _TOKENS)
_TOKEN_IDS = [tid for tid, _ in _TOKENS]


def _make_tokenizer() -> MockTokenizer:
    tok = MockTokenizer(vocab=_VOCAB, tokens=_TOKENS)
    tok.bos_token_id = BOS_ID
    return tok


class _ThinkParser(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, parser_engine_config=_THINK_CONFIG, **kwargs)


_ThinkReasoningAdapter, _ThinkToolAdapter = make_adapters(_ThinkParser)


class _ThinkDelegating(DelegatingParser):
    reasoning_parser_cls = _ThinkReasoningAdapter
    tool_parser_cls = _ThinkToolAdapter


class TestDirectParser:
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_streaming_drops_bos(self, chunk_size):
        tokenizer = _make_tokenizer()
        parser = _ThinkParser(tokenizer, None)

        deltas = replay_streaming(
            parser, _TOKENS, chunk_size=chunk_size, finished_on_last=True
        )
        output = collect_output(deltas)

        assert output.content == "Hello world"
        assert output.reasoning == "Let me think."

    def test_parse_drops_bos(self, mock_request):
        tokenizer = _make_tokenizer()
        parser = _ThinkParser(tokenizer, None)
        reasoning, content, _tool_calls = parser.parse(
            _FULL_TEXT, mock_request, model_output_token_ids=_TOKEN_IDS
        )

        assert content == "Hello world"
        assert reasoning == "Let me think."


class TestDelegatingParser:
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_streaming_drops_bos(self, chunk_size):
        tokenizer = _make_tokenizer()
        parser = _ThinkDelegating(tokenizer, None)

        deltas = replay_streaming(
            parser, _TOKENS, chunk_size=chunk_size, finished_on_last=True
        )
        output = collect_output(deltas)

        assert output.content == "Hello world"
        assert output.reasoning == "Let me think."

    def test_parse_drops_bos(self, mock_request):
        tokenizer = _make_tokenizer()
        parser = _ThinkDelegating(tokenizer, None)
        reasoning, content, _tool_calls = parser.parse(
            _FULL_TEXT, mock_request, model_output_token_ids=_TOKEN_IDS
        )

        assert content == "Hello world"
        assert reasoning == "Let me think."


# ── EOS / PAD token drop ────────────────────────────────────────────

EOS_ID = 2
EOS_TEXT = "<eos>"
PAD_ID = 3
PAD_TEXT = "<pad>"

_VOCAB_EOS_PAD: dict[str, int] = {
    **_VOCAB,
    EOS_TEXT: EOS_ID,
    PAD_TEXT: PAD_ID,
}

_TOKENS_WITH_EOS: list[tuple[int, str]] = [
    (100, "Let "),
    (101, "me "),
    (102, "think."),
    (THINK_END_ID, "</think>"),
    (103, "Hello "),
    (104, "world"),
    (EOS_ID, EOS_TEXT),
]

_TOKENS_WITH_PAD: list[tuple[int, str]] = [
    (100, "Let "),
    (PAD_ID, PAD_TEXT),
    (101, "me "),
    (102, "think."),
    (THINK_END_ID, "</think>"),
    (103, "Hello "),
    (PAD_ID, PAD_TEXT),
    (104, "world"),
]

_TOKENS_WITH_ALL: list[tuple[int, str]] = [
    (BOS_ID, BOS_TEXT),
    (100, "Let "),
    (PAD_ID, PAD_TEXT),
    (101, "me "),
    (102, "think."),
    (THINK_END_ID, "</think>"),
    (103, "Hello "),
    (BOS_ID, BOS_TEXT),
    (104, "world"),
    (EOS_ID, EOS_TEXT),
]


_STRUCTURAL_DROP_CASES = [
    pytest.param(
        _TOKENS_WITH_EOS,
        {"eos_token_id": EOS_ID},
        id="eos",
    ),
    pytest.param(
        _TOKENS_WITH_PAD,
        {"pad_token_id": PAD_ID},
        id="pad",
    ),
    pytest.param(
        _TOKENS_WITH_ALL,
        {"bos_token_id": BOS_ID, "eos_token_id": EOS_ID, "pad_token_id": PAD_ID},
        id="bos+eos+pad",
    ),
]


def _make_drop_parser(tokens, id_overrides):
    tok = MockTokenizer(vocab=_VOCAB_EOS_PAD, tokens=tokens)
    for attr, value in id_overrides.items():
        setattr(tok, attr, value)
    return _ThinkParser(tok, None), tokens


class TestStructuralTokenDrop:
    @pytest.mark.parametrize("tokens,id_overrides", _STRUCTURAL_DROP_CASES)
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_streaming_drops(self, chunk_size, tokens, id_overrides):
        parser, tokens = _make_drop_parser(tokens, id_overrides)
        deltas = replay_streaming(
            parser, tokens, chunk_size=chunk_size, finished_on_last=True
        )
        output = collect_output(deltas)
        assert output.content == "Hello world"
        assert output.reasoning == "Let me think."

    @pytest.mark.parametrize("tokens,id_overrides", _STRUCTURAL_DROP_CASES)
    def test_parse_drops(self, mock_request, tokens, id_overrides):
        parser, tokens = _make_drop_parser(tokens, id_overrides)
        full_text = "".join(text for _, text in tokens)
        token_ids = [tid for tid, _ in tokens]
        reasoning, content, _ = parser.parse(
            full_text, mock_request, model_output_token_ids=token_ids
        )
        assert content == "Hello world"
        assert reasoning == "Let me think."


# ── Text/token ID consistency edge cases ─────────────────────────────


class TestTokenIdEdgeCases:
    def test_empty_text_with_structural_token_ids(self, mock_request):
        """Empty text with BOS token ID should not crash."""
        tokenizer = _make_tokenizer()
        parser = _ThinkParser(tokenizer, None)
        reasoning, content, _ = parser.parse(
            "", mock_request, model_output_token_ids=[BOS_ID]
        )
        assert reasoning is None
        assert content is None

    def test_text_without_token_ids_backward_compat(self, mock_request):
        """Text-only path (no token IDs) must still parse correctly."""
        tokenizer = _make_tokenizer()
        parser = _ThinkParser(tokenizer, None)
        text = "Let me think.</think>Hello world"
        reasoning, content, _ = parser.parse(
            text, mock_request, model_output_token_ids=()
        )
        assert reasoning == "Let me think."
        assert content == "Hello world"

    def test_drop_token_id_present_but_text_already_stripped(self, mock_request):
        """Token IDs contain BOS but text doesn't — no corruption."""
        tokenizer = _make_tokenizer()
        parser = _ThinkParser(tokenizer, None)
        text_without_bos = "Let me think.</think>Hello world"
        ids_with_bos = [BOS_ID, 100, 101, 102, THINK_END_ID, 103, 104]

        reasoning, content, _ = parser.parse(
            text_without_bos, mock_request, model_output_token_ids=ids_with_bos
        )
        assert reasoning == "Let me think."
        assert content == "Hello world"
