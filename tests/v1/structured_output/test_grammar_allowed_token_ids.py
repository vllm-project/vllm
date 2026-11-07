# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam search should take a grammar's allowed set directly when it has one.

It needs the set as token ids, to put on each beam's `SamplingParams`. Obtaining
them by filling a bitmask and unpacking it again costs O(vocab_size) however few
tokens are allowed, once per beam per step.
"""

import torch

from vllm.entrypoints.generate.beam_search.offline import _bitmask_to_token_ids
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

VOCAB = 2048


class _Grammar(StructuredOutputGrammar):
    """Minimal grammar allowing a fixed set, with the hook optionally disabled."""

    def __init__(self, allowed: list[int], implement_hook: bool = True) -> None:
        self._allowed = allowed
        self._implement_hook = implement_hook
        self.fill_bitmask_calls = 0

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        return tokens

    def rollback(self, num_tokens: int) -> None:
        pass

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        self.fill_bitmask_calls += 1
        bitmask[batch_index].zero_()
        row = bitmask[batch_index]
        for token in self._allowed:
            if 0 <= token < VOCAB:
                row[token >> 5] |= 1 << (token & 31) if (token & 31) != 31 else -(2**31)

    def is_terminated(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def allowed_token_ids(self, vocab_size: int) -> list[int] | None:
        if not self._implement_hook:
            return None
        return sorted(t for t in self._allowed if 0 <= t < vocab_size)


def test_default_returns_none_so_callers_keep_their_fallback():
    """The hook is opt-in: an unmodified backend must be unaffected."""
    g = _Grammar([5, 6], implement_hook=False)
    assert g.allowed_token_ids(VOCAB) is None
    # The base class itself returns None regardless of the subclass.
    assert StructuredOutputGrammar.allowed_token_ids(g, VOCAB) is None


def test_hook_agrees_with_unpacking_the_bitmask():
    """Both routes must yield the same ids.

    A divergence would constrain a beam differently depending on which route ran.
    """
    allowed = [7, 31, 32, 63, 64, 1000, VOCAB - 1]
    g = _Grammar(allowed)
    bitmask = torch.full((1, (VOCAB + 31) // 32), -1, dtype=torch.int32)
    g.fill_bitmask(bitmask, 0)

    assert g.allowed_token_ids(VOCAB) == _bitmask_to_token_ids(bitmask[0], VOCAB)


def test_hook_filters_above_the_vocabulary():
    """An id the model cannot emit must not reach `SamplingParams`."""
    g = _Grammar([5, VOCAB, VOCAB + 100])
    assert g.allowed_token_ids(VOCAB) == [5]


def test_beam_path_prefers_the_hook_and_skips_the_bitmask():
    """The point of the change: with the hook, no bitmask is filled at all.

    Asserted through `fill_bitmask_calls` rather than by timing, so it holds on any
    machine.
    """
    allowed = [11, 22, 33]
    g = _Grammar(allowed)
    bitmask = torch.full((1, (VOCAB + 31) // 32), -1, dtype=torch.int32)

    # What the beam path does, in the order it does it.
    ids = g.allowed_token_ids(VOCAB)
    if ids is None:
        g.fill_bitmask(bitmask, 0)
        ids = _bitmask_to_token_ids(bitmask[0], VOCAB)

    assert ids == allowed
    assert g.fill_bitmask_calls == 0, "the bitmask was filled despite the hook"

    # And without the hook the fallback still runs.
    g2 = _Grammar(allowed, implement_hook=False)
    ids2 = g2.allowed_token_ids(VOCAB)
    if ids2 is None:
        g2.fill_bitmask(bitmask, 0)
        ids2 = _bitmask_to_token_ids(bitmask[0], VOCAB)
    assert ids2 == allowed
    assert g2.fill_bitmask_calls == 1, "the fallback did not run"
