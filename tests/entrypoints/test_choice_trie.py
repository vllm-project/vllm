# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ChoiceTrie.

Run without GPU:
    python -m pytest tests/entrypoints/test_choice_trie.py -v --noconftest
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from unittest.mock import MagicMock

from vllm.entrypoints.choice_trie import ChoiceTrie

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EOS = 999  # dummy EOS token id for all tests


def make_tokenizer(vocab: dict[str, int]) -> MagicMock:
    tok = MagicMock()

    def _encode(text, add_special_tokens=True, **kwargs):
        return [vocab.get(c, 0) for c in text]

    tok.encode.side_effect = _encode
    return tok


def build(choices, tok):
    """Build a ChoiceTrie with EOS appended (production behaviour)."""
    return ChoiceTrie.build(choices, tok, eos_token_id=EOS)


VOCAB = {
    c: i + 1
    for i, c in enumerate(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
    )
}
VOCAB[" "] = 99
VOCAB["\n"] = 98


def tids(tok, s: str) -> list[int]:
    """Continuation-context token IDs for string s."""
    prefix_len = len(tok.encode("\n"))
    return tok.encode("\n" + s)[prefix_len:]


# ---------------------------------------------------------------------------
# Regression: prefix must not eat choice tokens
# ---------------------------------------------------------------------------


def test_prefix_does_not_eat_choice_tokens():
    choices = ["calc", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    prefix_len = len(tok.encode("\n"))
    assert prefix_len == 1
    for choice in choices:
        ids = tids(tok, choice)
        assert len(ids) > 0, f"Prefix ate all tokens for choice {choice!r}"
        # After full choice: should return [EOS], not None (EOS not consumed yet)
        result = trie.allowed_tokens_for(ids)
        assert result == [EOS], (
            f"After full choice {choice!r} should allow only EOS, got {result}"
        )
        # After choice + EOS: terminal → None
        result_after_eos = trie.allowed_tokens_for(ids + [EOS])
        assert result_after_eos is None, (
            f"After choice+EOS should be terminal, got {result_after_eos}"
        )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_all_choices_reachable():
    choices = ["cat", "car", "card", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    for choice in choices:
        ids = tids(tok, choice)
        result = trie.allowed_tokens_for(ids)
        # After a full choice, EOS must be in the allowed set (may also allow
        # continuation tokens if the choice is a prefix of a longer choice).
        assert result is not None and EOS in result, (
            f"After full choice {choice!r} EOS must be allowed, got {result}"
        )
        assert trie.allowed_tokens_for(ids + [EOS]) is None


def test_non_choice_not_terminal():
    choices = ["cat", "car"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    prefix_ids = tids(tok, "ca")
    result = trie.allowed_tokens_for(prefix_ids)
    assert result is not None and result != [EOS], (
        "'ca' prefix should not be a choice terminal"
    )
    assert len(result) > 0


def test_invalid_prefix_returns_empty():
    choices = ["cat"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    result = trie.allowed_tokens_for([VOCAB["z"]])
    assert result == [], f"Invalid prefix should return [], got {result}"


# ---------------------------------------------------------------------------
# allowed_tokens_for at each step
# ---------------------------------------------------------------------------


def test_allowed_tokens_at_root():
    choices = ["cat", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    allowed = trie.allowed_tokens_for([])
    assert allowed is not None
    assert set(allowed) == {VOCAB["c"], VOCAB["d"]}


def test_allowed_tokens_mid_prefix():
    choices = ["cat", "car", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    allowed = trie.allowed_tokens_for(tids(tok, "ca"))
    assert allowed is not None
    assert set(allowed) == {VOCAB["t"], VOCAB["r"]}


def test_after_full_choice_only_eos_allowed():
    """After generating all tokens for a choice, only EOS should be allowed."""
    choices = ["cat", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    cat_ids = tids(tok, "cat")
    allowed = trie.allowed_tokens_for(cat_ids)
    assert allowed == [EOS], f"After 'cat' only EOS should be allowed, got {allowed}"


# ---------------------------------------------------------------------------
# Stateless re-walk
# ---------------------------------------------------------------------------


def test_stateless_repeated_calls():
    choices = ["cat", "car", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    ids = tids(tok, "ca")
    assert trie.allowed_tokens_for(ids) == trie.allowed_tokens_for(ids)


def test_two_beams_same_prefix_same_result():
    choices = ["cat", "car", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    ids = tids(tok, "ca")
    assert set(trie.allowed_tokens_for(ids)) == set(trie.allowed_tokens_for(ids))


# ---------------------------------------------------------------------------
# Dedup by generated sequence
# ---------------------------------------------------------------------------


def test_sequence_dedup_eliminates_duplicates():
    tok = make_tokenizer(VOCAB)
    cat_ids = tids(tok, "cat")
    seen: set[tuple] = set()
    kept = []
    for gen in [cat_ids, cat_ids]:  # two identical beams
        key = tuple(gen)
        if key not in seen:
            seen.add(key)
            kept.append(gen)
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_choices_no_allowed_tokens():
    tok = make_tokenizer(VOCAB)
    trie = build([], tok)
    allowed = trie.allowed_tokens_for([])
    assert allowed is not None and len(allowed) == 0


def test_single_token_choices():
    choices = ["a", "b", "c"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)
    for choice in choices:
        ids = tids(tok, choice)
        assert len(ids) == 1
        result = trie.allowed_tokens_for(ids)
        assert result == [EOS], (
            f"Single-token choice '{choice}' should allow EOS next, got {result}"
        )
        assert trie.allowed_tokens_for(ids + [EOS]) is None


# ---------------------------------------------------------------------------
# Full beam search simulation
# ---------------------------------------------------------------------------


def test_beam_search_simulation():
    """Simulate beam search over choices: tokens → EOS → terminal."""
    choices = ["cat", "car", "cap", "dog"]
    tok = make_tokenizer(VOCAB)
    trie = build(choices, tok)

    # Root: {c, d}
    assert set(trie.allowed_tokens_for([])) == {VOCAB["c"], VOCAB["d"]}
    # After "c": {a}
    assert set(trie.allowed_tokens_for([VOCAB["c"]])) == {VOCAB["a"]}
    # After "ca": {t, r, p}
    assert set(trie.allowed_tokens_for([VOCAB["c"], VOCAB["a"]])) == {
        VOCAB["t"],
        VOCAB["r"],
        VOCAB["p"],
    }
    # After "cat": [EOS]
    cat = [VOCAB["c"], VOCAB["a"], VOCAB["t"]]
    assert trie.allowed_tokens_for(cat) == [EOS]
    # After "cat" + EOS: terminal
    assert trie.allowed_tokens_for(cat + [EOS]) is None
    # After "dog": [EOS]
    dog = [VOCAB["d"], VOCAB["o"], VOCAB["g"]]
    assert trie.allowed_tokens_for(dog) == [EOS]
    assert trie.allowed_tokens_for(dog + [EOS]) is None
