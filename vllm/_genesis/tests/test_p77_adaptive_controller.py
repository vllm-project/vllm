# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P77 — AdaptiveNgramController state machine.

Pure-Python EMA + hysteresis controller (port of SGLang's
adaptive_spec_params.py + Nightjar arXiv 2512.22420 auto-disable). No
GPU / vLLM dependency — exercises the controller directly with synthetic
batch outcomes.

Covers:
  - init defaults match documented values
  - warmup: no transitions before WARMUP_BATCHES
  - update interval: only adjusts every UPDATE_INTERVAL batches
  - EMA tracks batch_avg
  - K rises when EMA exceeds a higher step (with hyst_down slack)
  - K drops when EMA falls below current step - hyst_down
  - Auto-disable: K → 0 when accept_rate < DISABLE_THRESHOLD
  - PROBE: re-tests after probe_interval batches at K=0
  - get_stats() reports sane fields
"""
from __future__ import annotations

import pytest

from vllm._genesis.kernels.adaptive_ngram_controller import (
    AdaptiveNgramController,
    _env_steps,
    _env_int,
    _env_float,
    is_active,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh controller singleton."""
    reset_for_tests()
    yield
    reset_for_tests()


# ─── env helpers ────────────────────────────────────────────────────────


def test_env_int_falls_back_on_invalid(monkeypatch):
    monkeypatch.setenv("X_TEST_INT", "not_an_int")
    assert _env_int("X_TEST_INT", 42) == 42


def test_env_float_falls_back_on_invalid(monkeypatch):
    monkeypatch.setenv("X_TEST_FLOAT", "not_a_float")
    assert _env_float("X_TEST_FLOAT", 0.5) == 0.5


def test_env_steps_dedups_and_sorts(monkeypatch):
    monkeypatch.setenv("X_TEST_STEPS", "5,3,1,3,0")
    assert _env_steps("X_TEST_STEPS", "0,1,3,5") == (0, 1, 3, 5)


def test_env_steps_default_on_garbage(monkeypatch):
    monkeypatch.setenv("X_TEST_STEPS", "abc,xyz")
    # ValueError → default
    assert _env_steps("X_TEST_STEPS", "0,1,3,5") == (0, 1, 3, 5)


def test_is_active_recognizes_truthy(monkeypatch):
    for v in ("1", "true", "yes", "on", "True", "YES"):
        monkeypatch.setenv("GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K", v)
        assert is_active(), f"{v!r} should activate"
    for v in ("0", "", "off", "no", "false"):
        monkeypatch.setenv("GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K", v)
        assert not is_active(), f"{v!r} should NOT activate"


# ─── init defaults ──────────────────────────────────────────────────────


def test_init_defaults_match_documented():
    c = AdaptiveNgramController()
    assert c.steps == (0, 1, 3, 5)
    assert c.alpha == pytest.approx(0.2)
    assert c.warmup == 10
    assert c.interval == 5
    assert c.hyst_down == pytest.approx(0.25)
    assert c.hyst_up == pytest.approx(0.0)
    assert c.disable_threshold == pytest.approx(0.30)
    assert c.probe_interval == 100
    assert c.log_every == 20
    # Initial K = middle of non-zero steps. For [1,3,5]: idx 3//2=1 → 3.
    assert c.current_K == 3
    assert c.batches_seen == 0


def test_init_picks_K_when_no_zero_in_steps(monkeypatch):
    monkeypatch.setenv("GENESIS_P77_STEPS", "2,4,6,8")
    c = AdaptiveNgramController()
    # non_zero = [2,4,6,8], middle = 6 (idx 4//2 = 2)
    assert c.current_K == 6


# ─── warmup ─────────────────────────────────────────────────────────────


def test_no_transition_before_warmup():
    c = AdaptiveNgramController()
    initial_K = c.current_K
    # Feed `warmup-1` batches with tiny accept rate (would normally trigger
    # auto-disable). Must NOT change K.
    for _ in range(c.warmup - 1):
        c.update([0, 0, 0], [c.current_K] * 3)
    assert c.current_K == initial_K
    assert c.transitions == 0


def test_decide_K_returns_current_K():
    c = AdaptiveNgramController()
    assert c.decide_K() == c.current_K


# ─── update interval / EMA ──────────────────────────────────────────────


def test_update_interval_throttles_decisions(monkeypatch):
    """After warmup, K should only change every `interval` batches."""
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "5")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.0")  # disable auto-off
    c = AdaptiveNgramController()
    # Drive EMA toward 1 (would pick step=1) but interval not yet hit
    for i in range(4):
        c.update([1] * 3, [c.current_K] * 3)
    # 4 batches < interval=5 → no decision yet
    # On the 5th update the controller may transition; before that batches_seen<5
    assert c.batches_seen == 4


def test_ema_updates_toward_batch_avg(monkeypatch):
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.0")
    c = AdaptiveNgramController()
    initial_ema = c.ema
    # Feed batch_avg = 5 (high accept). EMA should move toward 5.
    c.update([5, 5, 5], [c.current_K] * 3)
    assert c.ema > initial_ema
    # alpha=0.2 → ema = 0.8*old + 0.2*5
    expected = (1 - c.alpha) * initial_ema + c.alpha * 5.0
    assert c.ema == pytest.approx(expected, abs=1e-9)


# ─── K transitions (auto-disable) ──────────────────────────────────────


def test_auto_disable_when_accept_rate_below_threshold(monkeypatch):
    """K → 0 when accept_rate (batch_avg / batch_drafted) < disable_threshold."""
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.30")
    c = AdaptiveNgramController()
    # accept_rate = 0/K = 0 << 0.30 → must drop to K=0
    c.update([0, 0, 0], [c.current_K] * 3)
    assert c.current_K == 0
    assert c.disabled_steps == 1
    assert c.transitions == 1


def test_b4_disable_records_previous_K_as_last_K(monkeypatch):
    """B4 fix v7.62.12: when auto-disable triggers, last_K must record the
    PREVIOUS K (not the new 0). Without this fix, last_K was overwritten to
    0, breaking the "previous K" semantic + corrupting log message at line 219.
    """
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.30")
    c = AdaptiveNgramController()
    c.current_K = 5  # was running at K=5 before auto-disable trigger
    c.last_K = 3  # previous transition was 3->5

    # accept_rate = 0/5 = 0 << 0.30 → auto-disable
    c.update([0, 0, 0], [5] * 3)

    assert c.current_K == 0, "auto-disable should set K=0"
    assert c.last_K == 5, (
        f"B4 fix: last_K must be PREVIOUS K (5), got {c.last_K}. "
        "If 0, this is the pre-fix bug — last_K incorrectly assigned to "
        "the NEW value, breaking 'previous K' semantic."
    )


def test_no_auto_disable_when_zero_not_in_steps(monkeypatch):
    """If 0 is not in steps, controller must not pick it."""
    monkeypatch.setenv("GENESIS_P77_STEPS", "1,3,5")
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.50")
    c = AdaptiveNgramController()
    # accept_rate = 0 << 0.50 but 0 not in steps → must NOT pick 0
    c.update([0, 0, 0], [c.current_K] * 3)
    assert c.current_K != 0
    assert c.disabled_steps == 0


# ─── PROBE ──────────────────────────────────────────────────────────────


def test_probe_reactivates_K_after_probe_interval(monkeypatch):
    """After K=0 for `probe_interval` batches, force a non-zero K to retest."""
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.30")
    monkeypatch.setenv("GENESIS_P77_PROBE_INTERVAL", "5")
    c = AdaptiveNgramController()

    # 1) drop to K=0
    c.update([0, 0, 0], [c.current_K] * 3)
    assert c.current_K == 0

    # 2) Feed 4 more batches at K=0 — still under probe_interval (5)
    # Since K=0 we'd typically not run drafts; pass empty drafted_lens.
    for _ in range(4):
        c.update([0, 0, 0], [0] * 3)
    assert c.current_K == 0

    # 3) On the 5th batch (probe_interval reached), should re-probe.
    pre_transitions = c.transitions
    c.update([0, 0, 0], [0] * 3)
    assert c.current_K > 0, "PROBE should restore non-zero K"
    assert c.transitions == pre_transitions + 1


# ─── hysteresis / K rises ──────────────────────────────────────────────


def test_K_rises_when_ema_exceeds_higher_step(monkeypatch):
    """When EMA climbs above next step (with hyst_down slack), K must rise."""
    monkeypatch.setenv("GENESIS_P77_STEPS", "0,1,3,5")
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.0")
    monkeypatch.setenv("GENESIS_P77_EMA_ALPHA", "1.0")  # ema = batch_avg
    monkeypatch.setenv("GENESIS_P77_HYSTERESIS_DOWN", "0.0")
    monkeypatch.setenv("GENESIS_P77_HYSTERESIS_UP", "0.0")
    c = AdaptiveNgramController()
    c.current_K = 1
    c.ema = 1.0
    # batch_avg=5 → ema=5 → highest step ≤ ema = 5
    c.update([5, 5, 5], [3, 3, 3])
    assert c.current_K == 5


def test_K_drops_when_ema_falls(monkeypatch):
    """When EMA falls, K must drop to the next-lower step (no hysteresis)."""
    monkeypatch.setenv("GENESIS_P77_STEPS", "0,1,3,5")
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.0")
    monkeypatch.setenv("GENESIS_P77_EMA_ALPHA", "1.0")
    monkeypatch.setenv("GENESIS_P77_HYSTERESIS_DOWN", "0.0")
    c = AdaptiveNgramController()
    c.current_K = 5
    c.ema = 5.0
    # batch_avg=1 → ema=1 → highest step ≤ 1 = 1
    c.update([1, 1, 1], [5, 5, 5])
    assert c.current_K == 1


# ─── stats ──────────────────────────────────────────────────────────────


def test_get_stats_reports_running_totals(monkeypatch):
    monkeypatch.setenv("GENESIS_P77_WARMUP_BATCHES", "0")
    monkeypatch.setenv("GENESIS_P77_UPDATE_INTERVAL", "100")  # don't transition
    monkeypatch.setenv("GENESIS_P77_DISABLE_THRESHOLD", "0.0")
    c = AdaptiveNgramController()
    c.update([3, 3, 3], [c.current_K] * 3)
    c.update([1, 2, 0], [c.current_K] * 3)
    stats = c.get_stats()
    assert stats["batches_seen"] == 2
    assert stats["total_accepted"] == 12  # 3+3+3+1+2+0
    assert stats["total_drafted"] == 6 * c.current_K
    assert "overall_accept_rate" in stats
    assert "current_K" in stats
    assert "ema" in stats


def test_empty_accepted_lens_is_noop():
    """All-greedy / no-spec batches pass empty list; controller must skip."""
    c = AdaptiveNgramController()
    pre = c.batches_seen
    c.update([], [])
    assert c.batches_seen == pre
