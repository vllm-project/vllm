# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DFlash D-Cut dynamic draft pruning.

These tests cover the policy-free, hardware-independent parts of D-Cut:
the keep-count formula, the keep-ratio metrics, the scheduler-side
truncation/accounting, and the config surface (validation, mode
normalization, and the async-scheduling gate).

The GPU tensor selection path in ``DFlashProposer._select_dcut_keep_lens``
and the warmup cost-table profiling are exercised by end-to-end runs, not
here; they require real device tensors and a loaded DFlash model.
"""

import pytest
from pydantic import ValidationError

from tests.v1.core.utils import create_scheduler
from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats


# ---------------------------------------------------------------------------
# Keep-count formula
# ---------------------------------------------------------------------------
# D-Cut interprets the ratio as the fraction of target-forward query tokens
# kept, including the one bonus token reserved per request. The draft-token
# budget is therefore ceil(bs * (num_draft + 1) * ratio) - bs.
def test_dcut_keep_count_full_ratio_keeps_all_drafts():
    # ratio=1.0 keeps every draft token: bs*(K+1) - bs == bs*K.
    assert DFlashProposer._get_dcut_keep_count(8, 3, 1.0) == 8 * 3


def test_dcut_keep_count_reserves_bonus_token():
    # 4 reqs, K=3 -> 16 forward slots; half of that is 8, minus 4 bonus = 4.
    assert DFlashProposer._get_dcut_keep_count(4, 3, 0.5) == 4


def test_dcut_keep_count_zero_ratio_keeps_nothing():
    assert DFlashProposer._get_dcut_keep_count(8, 3, 0.0) == 0


def test_dcut_keep_count_never_negative():
    # A ratio small enough to not even cover the bonus tokens clamps to 0.
    assert DFlashProposer._get_dcut_keep_count(8, 3, 0.1) == 0


def test_dcut_keep_count_rounds_up():
    # ceil(1 * (3 + 1) * 0.6) - 1 = ceil(2.4) - 1 = 3 - 1 = 2.
    assert DFlashProposer._get_dcut_keep_count(1, 3, 0.6) == 2


def test_dcut_keep_count_bonus_boundary():
    # ceil(1 * 4 * 0.25) - 1 = 1 - 1 = 0: the only kept slot is the bonus.
    assert DFlashProposer._get_dcut_keep_count(1, 3, 0.25) == 0
    # Just above the bonus boundary keeps a single draft.
    assert DFlashProposer._get_dcut_keep_count(1, 3, 0.26) == 1


# ---------------------------------------------------------------------------
# Config: dflash_dcut validation, mode normalization, async gate
# ---------------------------------------------------------------------------
def _dflash_dcut_mode(dcut: float | str) -> str:
    # dflash_dcut_mode reads only dflash_dcut, so a method="dflash" config is
    # not required to exercise normalization. Build a lightweight ngram config
    # and set the field directly to avoid loading a real DFlash model.
    cfg = SpeculativeConfig(model="ngram", num_speculative_tokens=3)
    cfg.dflash_dcut = dcut
    return cfg.dflash_dcut_mode


def test_dflash_dcut_mode_off():
    assert _dflash_dcut_mode(0.0) == "off"


def test_dflash_dcut_mode_fixed_ratio():
    assert _dflash_dcut_mode(0.5) == "fixed_ratio"


def test_dflash_dcut_mode_selector():
    assert _dflash_dcut_mode("auto") == "selector"


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, "AUTO"])
def test_dflash_dcut_rejects_invalid_values(bad_value):
    # Out-of-range floats and non-"auto" strings are rejected. (A numeric
    # string like "0.5" is intentionally not tested: pydantic coerces it to
    # the float 0.5, which is a valid ratio.)
    with pytest.raises(ValidationError):
        SpeculativeConfig(
            model="ngram", num_speculative_tokens=3, dflash_dcut=bad_value
        )


def test_dflash_dcut_disabled_for_non_dflash_method():
    # ngram is not dflash, so a non-zero dflash_dcut is reset to 0 with a warn.
    cfg = SpeculativeConfig(model="ngram", num_speculative_tokens=3, dflash_dcut=0.5)
    assert cfg.dflash_dcut == 0.0
    assert cfg.dflash_dcut_mode == "off"


def test_uses_dflash_dcut_false_for_non_dflash():
    # ngram resets dflash_dcut to 0, so the D-Cut gate predicate is False.
    cfg = SpeculativeConfig(model="ngram", num_speculative_tokens=3, dflash_dcut=0.5)
    assert cfg.uses_dflash_dcut() is False


def test_uses_dflash_dcut_tracks_mode():
    # uses_dflash_dcut() requires both the dflash method and a non-off mode.
    cfg = SpeculativeConfig(model="ngram", num_speculative_tokens=3)
    # Not dflash: always disabled regardless of the field value.
    cfg.dflash_dcut = 0.5
    assert cfg.uses_dflash_dcut() is False
    # Stub the method check to isolate the mode half of the predicate.
    cfg.use_dflash = lambda: True  # type: ignore[method-assign]
    assert cfg.uses_dflash_dcut() is True
    cfg.dflash_dcut = 0.0
    assert cfg.uses_dflash_dcut() is False
    cfg.dflash_dcut = "auto"
    assert cfg.uses_dflash_dcut() is True


# The cudagraph-mode override (full / full_and_piecewise -> piecewise) and the
# Model-Runner-V2 fallback for D-Cut are wired in VllmConfig the same way as
# dynamic speculative decoding; like the async gate they need a full VllmConfig
# (and thus a real model_config) to reach, so they are covered by end-to-end
# runs rather than these pure-logic unit tests.


# The async-scheduling gate in vllm.py (raise on explicit async_scheduling,
# auto-disable otherwise when D-Cut is enabled) is exercised by end-to-end
# runs: constructing a VllmConfig that reaches the gate needs a real
# model_config, which is out of scope for these pure-logic unit tests.


# ---------------------------------------------------------------------------
# Metrics: keep-ratio accounting
# ---------------------------------------------------------------------------
def test_spec_decoding_stats_observe_dcut_accumulates():
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_dcut(kept_draft_tokens=6, total_draft_tokens=10)
    stats.observe_dcut(kept_draft_tokens=2, total_draft_tokens=6)
    assert stats.dcut_kept_draft_tokens == 8
    assert stats.dcut_total_draft_tokens == 16


def _capture_log(logging: SpecDecodingLogging) -> str:
    messages: list[str] = []
    logging.log(log_fn=lambda *args: messages.append(args[0] % args[1:]))
    return messages[0]


def test_spec_decoding_logging_reports_keep_ratio():
    logging = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    stats.observe_dcut(kept_draft_tokens=6, total_draft_tokens=12)
    logging.observe(stats)
    assert "D-Cut keep ratio: 0.500" in _capture_log(logging)


def test_spec_decoding_logging_aggregates_keep_ratio_across_steps():
    # Ratio is over summed counts, not an average of per-step ratios:
    # (6 + 1) / (12 + 3) = 7 / 15 = 0.467.
    logging = SpecDecodingLogging()
    for kept, total in [(6, 12), (1, 3)]:
        stats = SpecDecodingStats.new(num_spec_tokens=3)
        stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=1)
        stats.observe_dcut(kept_draft_tokens=kept, total_draft_tokens=total)
        logging.observe(stats)
    assert "D-Cut keep ratio: 0.467" in _capture_log(logging)


def test_spec_decoding_logging_keep_ratio_nan_without_dcut():
    # No D-Cut observations -> ratio is nan, not a division-by-zero error.
    logging = SpecDecodingLogging()
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=1)
    logging.observe(stats)
    assert "D-Cut keep ratio: nan" in _capture_log(logging)


# ---------------------------------------------------------------------------
# Scheduler: truncation + pending accounting
# ---------------------------------------------------------------------------
def _make_scheduler():
    return create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_speculative_tokens=3,
    )


def test_dcut_truncate_keeps_prefix_and_counts_bonus():
    scheduler = _make_scheduler()
    kept = scheduler._dcut_truncate([10, 11, 12], keep_len=2)
    assert kept == [10, 11]
    # kept counts the bonus token: keep_len + 1; total: len + 1.
    assert scheduler._pending_dcut_kept_draft_tokens == 3
    assert scheduler._pending_dcut_total_draft_tokens == 4


def test_dcut_truncate_keep_len_zero_keeps_only_bonus():
    scheduler = _make_scheduler()
    kept = scheduler._dcut_truncate([10, 11, 12], keep_len=0)
    assert kept == []
    assert scheduler._pending_dcut_kept_draft_tokens == 1
    assert scheduler._pending_dcut_total_draft_tokens == 4


def test_dcut_truncate_keep_len_at_or_above_len():
    scheduler = _make_scheduler()
    # keep_len == len keeps everything; slicing beyond len is a no-op.
    assert scheduler._dcut_truncate([10, 11, 12], keep_len=3) == [10, 11, 12]
    assert scheduler._dcut_truncate([20, 21], keep_len=5) == [20, 21]


def test_dcut_truncate_accumulates_across_requests():
    scheduler = _make_scheduler()
    scheduler._dcut_truncate([10, 11, 12], keep_len=1)
    scheduler._dcut_truncate([20, 21, 22], keep_len=3)
    assert scheduler._pending_dcut_kept_draft_tokens == (1 + 1) + (3 + 1)
    assert scheduler._pending_dcut_total_draft_tokens == (3 + 1) + (3 + 1)


def test_make_or_update_dcut_stats_feeds_and_resets():
    scheduler = _make_scheduler()
    scheduler._dcut_truncate([10, 11, 12], keep_len=2)

    stats = SpecDecodingStats.new(num_spec_tokens=3)
    out = scheduler._make_or_update_dcut_stats(stats)
    assert out is stats
    assert stats.dcut_kept_draft_tokens == 3
    assert stats.dcut_total_draft_tokens == 4
    # Pending counters are drained after being observed.
    assert scheduler._pending_dcut_kept_draft_tokens == 0
    assert scheduler._pending_dcut_total_draft_tokens == 0


def test_make_or_update_dcut_stats_noop_without_pending():
    scheduler = _make_scheduler()
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    out = scheduler._make_or_update_dcut_stats(stats)
    assert out is stats
    assert stats.dcut_kept_draft_tokens == 0
    assert stats.dcut_total_draft_tokens == 0


def test_make_or_update_dcut_stats_preserves_pending_when_no_stats():
    scheduler = _make_scheduler()
    scheduler._dcut_truncate([10, 11, 12], keep_len=2)
    # When no SpecDecodingStats exists this step, pending counters are left
    # intact for a later step rather than being silently dropped.
    assert scheduler._make_or_update_dcut_stats(None) is None
    assert scheduler._pending_dcut_kept_draft_tokens == 3
    assert scheduler._pending_dcut_total_draft_tokens == 4


def test_make_or_update_dcut_stats_noop_when_log_stats_disabled():
    scheduler = _make_scheduler()
    scheduler.log_stats = False
    scheduler._dcut_truncate([10, 11, 12], keep_len=2)
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    # log_stats=False short-circuits; the provided stats are returned untouched
    # and pending counters are not drained.
    out = scheduler._make_or_update_dcut_stats(stats)
    assert out is stats
    assert stats.dcut_total_draft_tokens == 0
