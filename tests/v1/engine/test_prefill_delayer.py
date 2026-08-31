# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PrefillDelayer state machine.

The delayer holds no distributed state: it is fed the reduced cross-DP signals
(prefillable rank count and total queued prefill tokens) directly, so its
FIRE/HOLD logic can be exercised single-process and deterministically.
"""

from vllm.v1.engine.prefill_delayer import PrefillDelayer

DP_SIZE = 8
BUDGET = 1000


def _make(**kwargs) -> PrefillDelayer:
    kwargs.setdefault("prefill_token_budget", BUDGET)
    kwargs.setdefault("target_fill", 0.9)
    return PrefillDelayer(dp_size=DP_SIZE, **kwargs)


def _throttle(delayer, prefillable_count, pending_tokens) -> bool:
    delayer.update_throttle(prefillable_count, pending_tokens)
    return delayer.should_throttle


def test_no_rank_prefillable_allows():
    d = _make()
    # Nothing queued anywhere -> allow (vacuous), never throttle.
    assert _throttle(d, prefillable_count=0, pending_tokens=0) is False


def test_all_ranks_prefillable_and_full_allows():
    d = _make(target_fill=0.9)
    # Aligned and the queued batch fills 90% of aggregate capacity -> fire.
    full = int(DP_SIZE * BUDGET * 0.9)
    assert _throttle(d, prefillable_count=DP_SIZE, pending_tokens=full) is False


def test_all_ranks_prefillable_but_underfull_holds():
    d = _make(target_fill=0.9)
    # Aligned but only 10% full -> coalesce (hold), which is the whole point.
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, prefillable_count=DP_SIZE, pending_tokens=sparse) is True


def test_skewed_ranks_hold_even_when_full():
    d = _make(target_fill=0.9)
    # Only some ranks prefillable -> anti-skew hold regardless of fill.
    assert _throttle(d, prefillable_count=3, pending_tokens=DP_SIZE * BUDGET) is True


def test_underfull_times_out_by_passes():
    d = _make(target_fill=0.9, max_delay_passes=3, max_delay_ms=1e9)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    # First max_delay_passes calls hold, then force-allow to bound TTFT.
    assert _throttle(d, DP_SIZE, sparse) is True
    assert _throttle(d, DP_SIZE, sparse) is True
    assert _throttle(d, DP_SIZE, sparse) is True
    assert _throttle(d, DP_SIZE, sparse) is False


def test_becoming_full_releases_before_timeout():
    d = _make(target_fill=0.9, max_delay_passes=100, max_delay_ms=1e9)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, DP_SIZE, sparse) is True
    # Queue fills up on a later step -> fire immediately, no timeout needed.
    full = int(DP_SIZE * BUDGET * 0.95)
    assert _throttle(d, DP_SIZE, full) is False


def test_target_fill_zero_disables_coalescing():
    d = _make(target_fill=0.0)
    # With coalescing off, an aligned batch fires even when nearly empty.
    assert _throttle(d, prefillable_count=DP_SIZE, pending_tokens=1) is False
    # Skew still holds (alignment is independent of the fill target).
    assert _throttle(d, prefillable_count=1, pending_tokens=1) is True


def test_reset_clears_throttle():
    d = _make(target_fill=0.9, max_delay_passes=100, max_delay_ms=1e9)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, DP_SIZE, sparse) is True
    d.reset()
    assert d.should_throttle is False


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
