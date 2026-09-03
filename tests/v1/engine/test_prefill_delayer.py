# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PrefillDelayer decision table.

The delayer holds no distributed state: it is fed the reduced cross-DP signals
directly, so its FIRE/HOLD logic can be exercised single-process and
deterministically.
"""

from vllm.v1.engine.prefill_delayer import PrefillDelayer

DP_SIZE = 8
BUDGET = 1000


def _make(**kwargs) -> PrefillDelayer:
    kwargs.setdefault("prefill_token_budget", BUDGET)
    kwargs.setdefault("target_fill", 0.9)
    return PrefillDelayer(dp_size=DP_SIZE, **kwargs)


def _throttle(
    d: PrefillDelayer,
    n_prefillable: int,
    pending_tokens: int,
    running_decode: int = 1,
    kv_high: int = 0,
    kv_low: int = 0,
    has_partial: int = 0,
    queue_hot: int = 0,
) -> bool:
    d.update_throttle(
        n_prefillable,
        pending_tokens,
        running_decode,
        kv_high,
        kv_low,
        has_partial,
        queue_hot,
    )
    return d.should_throttle


def test_no_rank_prefillable_fires():
    d = _make()
    assert _throttle(d, n_prefillable=0, pending_tokens=0) is False


def test_alignment_gate_holds_until_all_ranks_ready():
    d = _make(target_fill=0.0)
    for n in range(1, DP_SIZE):
        assert _throttle(d, n_prefillable=n, pending_tokens=1) is True, n
    # Aligned (and coalescing off) -> fire.
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=1) is False


def test_aligned_and_full_fires():
    d = _make(target_fill=0.9)
    full = int(DP_SIZE * BUDGET * 0.9)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=full) is False


def test_aligned_but_underfull_holds():
    d = _make(target_fill=0.9)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True


def test_kv_high_fires_over_alignment_hold():
    d = _make()
    # Skewed would normally hold, but a full KV cache forces the release.
    assert _throttle(d, n_prefillable=3, pending_tokens=1, kv_high=1) is False


def test_kv_low_fires_over_alignment_hold():
    d = _make()
    assert _throttle(d, n_prefillable=3, pending_tokens=1, kv_low=1) is False


def test_no_running_decode_fires():
    d = _make()
    # No decode to hide the wait behind -> fire immediately.
    assert _throttle(d, n_prefillable=3, pending_tokens=1, running_decode=0) is False


def test_queue_hot_fires_over_alignment_hold():
    d = _make()
    assert _throttle(d, n_prefillable=3, pending_tokens=1, queue_hot=1) is False


def test_ttft_max_ticks_bound_fires():
    d = _make(target_fill=0.9, ttft_max_ticks=3)
    for _ in range(3):
        assert _throttle(d, n_prefillable=3, pending_tokens=1) is True
    assert _throttle(d, n_prefillable=3, pending_tokens=1) is False


def test_partial_max_ticks_bound_fires():
    d = _make(target_fill=0.9, ttft_max_ticks=1000, partial_max_ticks=2)
    assert _throttle(d, n_prefillable=3, pending_tokens=1, has_partial=1) is True
    assert _throttle(d, n_prefillable=3, pending_tokens=1, has_partial=1) is True
    assert _throttle(d, n_prefillable=3, pending_tokens=1, has_partial=1) is False


def test_becoming_full_releases_before_timeout():
    d = _make(target_fill=0.9, ttft_max_ticks=1000)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True
    full = int(DP_SIZE * BUDGET * 0.95)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=full) is False


def test_stall_gives_up_and_fires():
    d = _make(target_fill=0.9, ttft_max_ticks=1000, stall_ticks=3)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    # Pending stops growing -> after stall_ticks non-growing holds, fire.
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is False


def test_stall_counter_resets_when_pending_grows():
    d = _make(target_fill=0.9, ttft_max_ticks=1000, stall_ticks=2)
    base = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=base) is True
    # Growth resets the stall counter, so it keeps holding.
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=base + 1) is True
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=base + 1) is True
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=base + 1) is False


def test_reset_clears_state():
    d = _make(target_fill=0.9, ttft_max_ticks=1000)
    sparse = int(DP_SIZE * BUDGET * 0.1)
    assert _throttle(d, n_prefillable=DP_SIZE, pending_tokens=sparse) is True
    d.reset()
    assert d.should_throttle is False
    assert d._hold_ticks == 0
    assert d._stall_count == 0


def test_global_prefill_step_disabled_by_default():
    d = _make(target_fill=0.0)
    # Even on a fire step, without idle_non_prefill_ranks it is never a
    # global-prefill step.
    _throttle(d, n_prefillable=DP_SIZE, pending_tokens=1)
    assert d.should_throttle is False
    assert d.is_global_prefill_step is False


def test_global_prefill_step_on_fire_when_enabled():
    d = _make(target_fill=0.0, idle_non_prefill_ranks=True)
    _throttle(d, n_prefillable=DP_SIZE, pending_tokens=1)
    assert d.should_throttle is False
    assert d.is_global_prefill_step is True


def test_no_global_prefill_step_on_hold():
    d = _make(target_fill=0.9, idle_non_prefill_ranks=True)
    # Skewed -> hold -> not a global-prefill step.
    assert _throttle(d, n_prefillable=3, pending_tokens=1) is True
    assert d.is_global_prefill_step is False


def test_no_global_prefill_step_when_none_prefillable():
    d = _make(target_fill=0.0, idle_non_prefill_ranks=True)
    _throttle(d, n_prefillable=0, pending_tokens=0)
    assert d.is_global_prefill_step is False


def test_consecutive_prefill_steps_bounded():
    d = _make(
        target_fill=0.0,
        idle_non_prefill_ranks=True,
        max_consecutive_prefill_steps=3,
    )
    steps = []
    for _ in range(5):
        _throttle(d, n_prefillable=DP_SIZE, pending_tokens=1)
        steps.append(d.is_global_prefill_step)
    # Three global-prefill steps, then a forced global-decode step, then resume.
    assert steps == [True, True, True, False, True]


if __name__ == "__main__":
    import importlib
    import sys
    import traceback

    module = importlib.import_module(__name__)
    tests = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAILED: {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
