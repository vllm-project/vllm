# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for bounded _build_drop_info memoization (#58145)."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.parser.engine import streaming_parser_engine as spe
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)


def _make_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="drop_info_cache_test",
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
    )


def _make_tokenizer(*, n_extra: int = 64) -> MagicMock:
    special_tokens = [f"<extra_id_{i}>" for i in range(n_extra)] + [
        "<tool_call>",
        "</tool_call>",
    ]
    special_ids = list(range(1000, 1000 + len(special_tokens)))
    vocab = {t: i for t, i in zip(special_tokens, special_ids)}
    tok = MagicMock()
    tok.all_special_tokens = special_tokens
    tok.all_special_ids = special_ids
    tok.get_vocab.return_value = vocab
    tok.decode.side_effect = lambda ids: special_tokens[ids[0] - 1000]
    return tok


@pytest.fixture(autouse=True)
def _clear_drop_info_cache():
    spe._clear_drop_info_cache()
    yield
    spe._clear_drop_info_cache()


def test_build_drop_info_cache_hit_same_config_and_tokenizer():
    config = _make_config()
    tok = _make_tokenizer()
    call_count = {"n": 0}
    real = spe._compute_drop_info

    def counting_compute(cfg, tokenizer):
        call_count["n"] += 1
        return real(cfg, tokenizer)

    with patch.object(spe, "_compute_drop_info", side_effect=counting_compute):
        first = spe._build_drop_info(config, tok)
        second = spe._build_drop_info(config, tok)

    assert call_count["n"] == 1
    assert first is second
    assert first is not None
    assert first.extra_token_ids


def test_streaming_parser_engine_reuses_drop_info_across_ctors():
    config = _make_config()
    tok = _make_tokenizer()
    call_count = {"n": 0}
    real = spe._compute_drop_info

    def counting_compute(cfg, tokenizer):
        call_count["n"] += 1
        return real(cfg, tokenizer)

    with patch.object(spe, "_compute_drop_info", side_effect=counting_compute):
        for _ in range(8):
            spe.StreamingParserEngine(config, tok)

    assert call_count["n"] == 1


def test_build_drop_info_cache_miss_on_different_tokenizer():
    config = _make_config()
    tok_a = _make_tokenizer()
    tok_b = _make_tokenizer()
    call_count = {"n": 0}
    real = spe._compute_drop_info

    def counting_compute(cfg, tokenizer):
        call_count["n"] += 1
        return real(cfg, tokenizer)

    with patch.object(spe, "_compute_drop_info", side_effect=counting_compute):
        spe._build_drop_info(config, tok_a)
        spe._build_drop_info(config, tok_b)

    assert call_count["n"] == 2
