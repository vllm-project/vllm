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


def test_global_prefill_step_emitted_on_release():
    d = _make(target_fill=0.0)
    # Aligned + coalescing off -> fire; a release step with a prefillable rank
    # is a global-prefill step.
    d.update_throttle(prefillable_count=DP_SIZE, pending_prefill_tokens=1)
    assert d.should_throttle is False
    assert d.is_global_prefill_step is True


def test_no_global_prefill_step_when_none_prefillable():
    d = _make(target_fill=0.0)
    d.update_throttle(prefillable_count=0, pending_prefill_tokens=0)
    assert d.is_global_prefill_step is False


def test_no_global_prefill_step_when_throttling():
    d = _make(target_fill=0.9)
    # Skew -> hold (throttle) -> not a global-prefill step.
    d.update_throttle(prefillable_count=3, pending_prefill_tokens=DP_SIZE * BUDGET)
    assert d.should_throttle is True
    assert d.is_global_prefill_step is False


def test_consecutive_prefill_steps_bounded():
    d = _make(target_fill=0.0, max_consecutive_prefill_steps=3)
    steps = []
    for _ in range(5):
        d.update_throttle(prefillable_count=DP_SIZE, pending_prefill_tokens=1)
        steps.append(d.is_global_prefill_step)
    # Three global-prefill steps, then a forced global-decode step, then resume.
    assert steps == [True, True, True, False, True]


def test_low_watermark_force_allows_and_prefills():
    d = _make(target_fill=0.9)
    # Skew would normally hold, but a rank under the KV low watermark forces the
    # release (and thus a global-prefill step).
    d.update_throttle(
        prefillable_count=3,
        pending_prefill_tokens=1,
        token_watermark_force_allow=True,
    )
    assert d.should_throttle is False
    assert d.is_global_prefill_step is True


def test_slot_pressure_holds_when_decode_batch_full():
    d = _make(max_running_requests=10, max_prefill_bs=5)
    max_running = DP_SIZE * 10
    # Few free decode slots (< max_prefill_bs) -> hold to protect decode batch.
    d.update_throttle(
        prefillable_count=DP_SIZE,
        pending_prefill_tokens=1,
        running_count=max_running - 4,
    )
    assert d.should_throttle is True
    # Ample free slots -> fire.
    d2 = _make(max_running_requests=10, max_prefill_bs=5)
    d2.update_throttle(
        prefillable_count=DP_SIZE,
        pending_prefill_tokens=1,
        running_count=max_running - 20,
    )
    assert d2.should_throttle is False


def test_queue_length_holds_short_queue():
    d = _make(max_running_requests=100, max_prefill_bs=5, queue_min_ratio=0.1)
    # queue_min = min(running*0.1, max_prefill_bs) = min(4, 5) = 4; waiting < 4
    # and ample decode slots -> hold on the queue trigger.
    d.update_throttle(
        prefillable_count=DP_SIZE,
        pending_prefill_tokens=1,
        running_count=40,
        waiting_count=2,
    )
    assert d.should_throttle is True
    # Queue long enough -> fire.
    d2 = _make(max_running_requests=100, max_prefill_bs=5, queue_min_ratio=0.1)
    d2.update_throttle(
        prefillable_count=DP_SIZE,
        pending_prefill_tokens=1,
        running_count=40,
        waiting_count=10,
    )
    assert d2.should_throttle is False


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
