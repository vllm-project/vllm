# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cursor state machine for the FlashInfer ReplaySSM speculative SSU.

FlashInfer never writes ``ring_start`` or ``prev_num_accepted_tokens`` back, so
the host must mirror the kernel's internal checkpoint decision exactly. Two
things differ from the Triton commit and both are load-bearing:

* the ring is exactly ``B + T``, so wraparound is a subtraction, not a bitmask;
* the checkpoint predicate uses each row's **actual** length
  (``pnat + seq_len > max_window``), not the maximum ``T``.

The Triton kernels run on GPU, so these tests exercise a CPU reference that is
kept structurally identical to ``_commit_flashinfer_cursors_kernel``; the GPU
kernel itself is covered by the metadata-builder suite.
"""

import pytest

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

B = 16
NUM_SPEC = 3
T = 1 + NUM_SPEC
R = B + T  # 20 -- deliberately not a power of two


def commit_row(history, origin, prev_flushed, accepted, cur_len, max_window, ring_len):
    """CPU mirror of _commit_flashinfer_cursors_kernel for one valid row."""
    if accepted > 0:
        if prev_flushed:
            origin = origin + history
            if origin >= ring_len:
                origin -= ring_len
            history = accepted
        else:
            history = history + accepted
    # Recomputed unconditionally, including when nothing was accepted.
    is_flush = (history + cur_len) > max_window
    return history, origin, is_flush


def test_ring_len_is_not_a_power_of_two():
    assert (
        MambaStateShapeCalculator.replayssm_spec_flashinfer_ring_len(B, NUM_SPEC) == R
    )
    assert R & (R - 1) != 0, "this suite is pointless if R happens to be pow2"


def test_non_flush_accumulates_history():
    history, origin, is_flush = commit_row(2, 0, False, 3, T, B, R)
    assert (history, origin) == (5, 0)
    assert is_flush is False


def test_flush_advances_origin_by_the_replayed_count():
    history, origin, is_flush = commit_row(5, 0, True, 2, T, B, R)
    # The checkpoint absorbed the 5 replayed tokens, so the ring restarts past
    # them and only the freshly accepted tokens remain live.
    assert (history, origin) == (2, 5)
    assert is_flush is False


def test_origin_wraps_by_subtraction_on_a_non_pow2_ring():
    """origin=18, history=5 on R=20 must wrap to 3."""
    history, origin, _ = commit_row(5, 18, True, 3, T, B, R)
    assert origin == 3
    assert history == 3
    # The Triton path's `& (R - 1)` is only equivalent when R is a power of two;
    # here it would leave the origin outside the ring entirely.
    assert (18 + 5) & (R - 1) == 19
    assert (18 + 5) % R == 3


def test_rejected_drafts_only_commit_accepted_tokens():
    history, _, _ = commit_row(2, 0, False, 1, T, B, R)
    assert history == 3


@pytest.mark.parametrize(
    "history,cur_len,expected",
    [
        (12, 4, False),  # 12 + 4 == 16, not > B
        (13, 4, True),  # 13 + 4 == 17 > B
        (13, 1, False),  # same history, a shorter row does not flush
        (16, 1, True),
    ],
)
def test_flush_uses_the_actual_row_length_not_max_t(history, cur_len, expected):
    """The kernel's varlen predicate is pnat + seq_len > max_window.

    Using T here instead of the row's real length would flush row (13, 1) a
    step early and desynchronise the host's origin advance from the kernel's.
    """
    _, _, is_flush = commit_row(history, 0, False, 0, cur_len, B, R)
    assert is_flush is expected


def test_zero_acceptance_freezes_cursors_but_still_recomputes_flush():
    """A row that accepted nothing must not inherit the previous flag."""
    history, origin, is_flush = commit_row(14, 7, True, 0, 4, B, R)
    assert (history, origin) == (14, 7), "nothing accepted -> nothing committed"
    assert is_flush is True, "14 + 4 > 16, decided fresh for this call"

    _, _, is_flush_short = commit_row(14, 7, True, 0, 1, B, R)
    assert is_flush_short is False, "same row, shorter window -> no flush"


def test_history_and_ring_invariants_hold_across_a_long_sequence():
    """Drive several non-flush steps, a flush, a rollback and a wraparound."""
    history, origin, is_flush = 0, 0, False
    accepted_pattern = [4, 1, 4, 2, 4, 4, 3, 4, 1, 4, 4, 4]

    for step, accepted in enumerate(accepted_pattern):
        cur_len = T
        history, origin, is_flush = commit_row(
            history, origin, is_flush, accepted, cur_len, B, R
        )
        assert 0 <= history <= B, f"step {step}: history {history} out of range"
        assert 0 <= origin < R, f"step {step}: origin {origin} out of range"
        # The kernel appends cur_len tokens at (origin + history) % R, so the
        # live window must fit the physical ring.
        assert history + cur_len <= R, f"step {step}: ring overflow"

    assert origin != 0, "the sequence should have flushed at least once"


def test_reset_leaves_a_row_that_cannot_flush_on_entry():
    """After a reset the ring is empty, so 0 + cur_len > B is false for any
    admissible row (cur_len <= T <= B)."""
    for cur_len in range(1, T + 1):
        _, _, is_flush = commit_row(0, 0, False, 0, cur_len, B, R)
        assert is_flush is False
