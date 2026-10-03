# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`PoolsideV1ReasoningParser`.

These tests use an in-process fake tokenizer to avoid downloading the
real Laguna tokenizer at test time. The fake exposes only the surface
area exercised by ``BaseThinkingReasoningParser.__init__`` and the
poolside override (``get_vocab``).
"""

import pytest

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.poolside_v1_reasoning_parser import PoolsideV1ReasoningParser

THINK_START = "<think>"
THINK_END = "</think>"
ASSISTANT = "<assistant>"

# Stable, distinguishable ids for the markers we care about.
THINK_START_ID = 1001
THINK_END_ID = 1002
ASSISTANT_ID = 1003
# A handful of "normal" filler ids that must never collide with the markers.
FILLER_IDS = [10, 11, 12, 13, 14]


class _FakeTokenizer:
    """Minimal tokenizer satisfying the reasoning parser constructors.

    ``BaseThinkingReasoningParser`` and ``PoolsideV1ReasoningParser`` only
    touch ``get_vocab()`` (cached as ``self.vocab``) and check that
    ``self.model_tokenizer`` is truthy.
    """

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)

    def __bool__(self) -> bool:  # truthy check in BaseThinkingReasoningParser
        return True

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def _full_vocab() -> dict[str, int]:
    vocab = {
        THINK_START: THINK_START_ID,
        THINK_END: THINK_END_ID,
        ASSISTANT: ASSISTANT_ID,
    }
    for i, tok_id in enumerate(FILLER_IDS):
        vocab[f"tok_{i}"] = tok_id
    return vocab


@pytest.fixture
def tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer(_full_vocab())


@pytest.fixture
def thinking_parser(tokenizer) -> PoolsideV1ReasoningParser:
    """Poolside parser whose inner parser is DeepSeekR1 (thinking enabled)."""
    return PoolsideV1ReasoningParser(tokenizer, chat_template_kwargs={"thinking": True})


@pytest.fixture
def identity_parser(tokenizer) -> PoolsideV1ReasoningParser:
    """Poolside parser whose inner parser is Identity (thinking disabled)."""
    return PoolsideV1ReasoningParser(
        tokenizer, chat_template_kwargs={"thinking": False}
    )


def test_registered_name_is_poolside_v1():
    parser_cls = ReasoningParserManager.get_reasoning_parser("poolside_v1")

    assert parser_cls is PoolsideV1ReasoningParser


def test_constructor_requires_assistant_token():
    vocab_without_assistant = {
        THINK_START: THINK_START_ID,
        THINK_END: THINK_END_ID,
    }
    bad_tokenizer = _FakeTokenizer(vocab_without_assistant)

    with pytest.raises(ValueError, match="<assistant>"):
        PoolsideV1ReasoningParser(
            bad_tokenizer, chat_template_kwargs={"thinking": True}
        )


def test_thinking_kwarg_selects_deepseek_r1_inner_parser(thinking_parser):
    assert isinstance(thinking_parser._parser, DeepSeekR1ReasoningParser)


def test_no_thinking_selects_identity_inner_parser(identity_parser):
    assert isinstance(identity_parser._parser, IdentityReasoningParser)


def test_is_reasoning_end_true_when_end_after_assistant(thinking_parser):
    # Standard assistant turn: <assistant> <think> ... </think>
    input_ids = [
        FILLER_IDS[0],
        ASSISTANT_ID,
        THINK_START_ID,
        FILLER_IDS[1],
        THINK_END_ID,
    ]

    assert thinking_parser.is_reasoning_end(input_ids) is True


def test_is_reasoning_end_false_when_start_after_end_in_current_turn(
    thinking_parser,
):
    # Reverse walk hits <think> before </think>: reasoning is still open.
    input_ids = [ASSISTANT_ID, THINK_END_ID, THINK_START_ID]

    assert thinking_parser.is_reasoning_end(input_ids) is False


def test_is_reasoning_end_false_when_no_tokens_after_assistant(thinking_parser):
    # <assistant> seen with no thinking markers after it.
    input_ids = [ASSISTANT_ID, FILLER_IDS[0], FILLER_IDS[1]]

    assert thinking_parser.is_reasoning_end(input_ids) is False


def test_is_reasoning_end_false_on_empty_input(thinking_parser):
    assert thinking_parser.is_reasoning_end([]) is False


def test_scoping_prevents_prior_turn_false_positive(thinking_parser):
    """Core contract: a stale ``</think>`` from a previous assistant turn
    must NOT trigger ``is_reasoning_end`` for the current turn.

    The parent ``DeepSeekV3ReasoningParser`` would scan the whole sequence
    and return ``True`` here; the poolside override stops at the most
    recent ``<assistant>`` and returns ``False``.
    """
    input_ids = [
        ASSISTANT_ID,  # prior assistant turn
        THINK_END_ID,  # stale </think> from prior turn
        ASSISTANT_ID,  # current assistant turn begins
        FILLER_IDS[0],
        FILLER_IDS[1],
    ]

    assert thinking_parser.is_reasoning_end(input_ids) is False


def test_is_reasoning_end_with_identity_parser_returns_true(identity_parser):
    # IdentityReasoningParser short-circuits to True regardless of input.
    assert identity_parser.is_reasoning_end([]) is True
    assert identity_parser.is_reasoning_end([FILLER_IDS[0], FILLER_IDS[1]]) is True
    assert (
        identity_parser.is_reasoning_end([ASSISTANT_ID, THINK_START_ID, FILLER_IDS[0]])
        is True
    )
