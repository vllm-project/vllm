# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the DSpark scheduler policy helpers.

Covers the three torch-free / CPU-tensor pure functions extracted into
``dspark/scheduler.py``: the uniform-length decision, the dynamic-SD table
derivation, and the per-request width allocation. No GPU is touched.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.scheduler import (
    OVERHEAD_C1_CLAMP,
    READOUT_CADENCE,
    DSparkScheduler,
    allocate_widths,
    derive_dynamic_sd_table,
    schedule_uniform_length,
)

# --- schedule_uniform_length ------------------------------------------------


def test_schedule_uniform_length_argmax_across_cliff():
    # A shape-aware table with a hard cost cliff at L=4: the argmax must land
    # just before it (L=3), not at gamma.
    accept = [0.0, 0.7, 1.2, 1.5, 1.6, 1.65]
    times_by_l = [[0.010], [0.011], [0.012], [0.013], [0.030], [0.031]]
    best_l, _ = schedule_uniform_length(
        accept, num_reqs=100, gamma=5, r_grid=[100], times_by_l=times_by_l
    )
    assert best_l == 3


def test_schedule_uniform_length_per_token_overhead_flips_decision():
    # Nearly flat GPU cost => a per-STEP overhead amortizes over deep L and pins
    # gamma; a per-TOKEN overhead does not amortize and picks the shallow L that
    # minimizes time-per-accepted-token. Same table, opposite decisions.
    accept = [0.0, 1.0, 1.5, 1.8]
    times_by_l = [[0.008], [0.010], [0.018], [0.030]]
    kwargs = dict(
        accept=accept, num_reqs=10, gamma=3, r_grid=[10], times_by_l=times_by_l
    )

    per_step_l, _ = schedule_uniform_length(**kwargs, overhead=(0.1, 0.0))
    per_token_l, _ = schedule_uniform_length(**kwargs, overhead=(0.0, 1e-3))

    assert per_step_l == 3  # folding overhead into c0 wrongly pins gamma
    assert per_token_l == 1  # the correct per-token model trims
    assert per_token_l != per_step_l


def test_schedule_uniform_length_hysteresis():
    # L=3 beats the incumbent L=2 by < 5%, so the default margin keeps L=2; a
    # zero margin lets the challenger through.
    accept = [0.0, 0.7, 1.2, 1.5, 1.6, 1.65]
    times_by_l = [[0.010], [0.011], [0.012], [0.013], [0.030], [0.031]]
    kwargs = dict(
        accept=accept, num_reqs=100, gamma=5, r_grid=[100], times_by_l=times_by_l
    )

    kept, _ = schedule_uniform_length(**kwargs, current=2)  # default 0.05 margin
    switched, _ = schedule_uniform_length(**kwargs, current=2, hysteresis=0.0)

    assert kept == 2
    assert switched == 3


def test_schedule_uniform_length_returns_gamma_without_table():
    accept = [0.0, 0.7, 1.2]
    assert schedule_uniform_length(accept, 100, 2, None, None) == (2, 0.0)
    assert schedule_uniform_length(accept, 100, 2, [], []) == (2, 0.0)


# --- derive_dynamic_sd_table ------------------------------------------------


def test_derive_dynamic_sd_table_ranges_and_collapse():
    # Two small buckets both pick K=gamma (must collapse); the large bucket hits
    # an eager cliff on every K>=1 and picks K=0.
    r_grid = [8, 16, 256]
    times_by_l = [
        [0.010, 0.010, 0.010],  # K=0 baseline decode
        [0.011, 0.011, 0.500],  # K=1 (eager on r=256)
        [0.012, 0.012, 0.500],  # K=2
        [0.013, 0.013, 0.500],  # K=3
    ]
    table = derive_dynamic_sd_table(r_grid, times_by_l, gamma=3, max_num_reqs=256)

    assert table == [(1, 16, 3), (17, 256, 0)]

    # Contiguous cover of 1..max_num_reqs with non-overlapping ranges.
    assert table[0][0] == 1
    assert table[-1][1] == 256
    for (s0, e0, _), (s1, _, _) in zip(table, table[1:]):
        assert s0 <= e0
        assert s1 == e0 + 1
    # Adjacent buckets were collapsed (no two consecutive rows share a K).
    ks = [k for _, _, k in table]
    assert all(a != b for a, b in zip(ks, ks[1:]))
    # K=0 is a legal outcome.
    assert 0 in ks


# --- allocate_widths (CPU tensors) ------------------------------------------


def _monotone_survivals() -> torch.Tensor:
    # Each row non-increasing, mirroring a prefix-survival cumprod.
    return torch.tensor(
        [
            [0.90, 0.80, 0.70, 0.60],
            [0.50, 0.40, 0.30, 0.20],
            [0.95, 0.50, 0.10, 0.05],
        ]
    )


def _is_prefix(mask_row: torch.Tensor, keep: int) -> bool:
    # Kept positions must be exactly the first `keep` entries.
    expected = torch.arange(mask_row.numel()) < keep
    return bool(torch.equal(mask_row, expected))


def test_allocate_widths_budget_and_prefix_validity():
    sv = _monotone_survivals()
    cal_t = torch.ones(4)
    keep = allocate_widths(sv, cal_t, num_reqs=3, length=4, tau=0.0, budget_frac=0.5)

    # budget = int(3*4*0.5)+1 = 7 tokens; top-k threshold lands at 0.5.
    assert keep.tolist() == [4, 1, 2]
    assert int(keep.sum()) == 7  # budget respected exactly

    # Monotone survivals => each kept set is a valid leading prefix.
    th = 0.5
    for r in range(sv.shape[0]):
        assert _is_prefix(sv[r] >= th, int(keep[r]))


def test_allocate_widths_full_budget_is_lossless():
    sv = _monotone_survivals()
    cal_t = torch.ones(4)
    keep = allocate_widths(sv, cal_t, num_reqs=3, length=4, tau=0.0, budget_frac=1.0)
    # budget >= number of candidates => every request keeps the full width.
    assert keep.tolist() == [4, 4, 4]


def test_allocate_widths_confidence_threshold():
    sv = _monotone_survivals()
    cal_t = torch.ones(4)
    keep = allocate_widths(sv, cal_t, num_reqs=3, length=4, tau=0.6, budget_frac=1.0)
    # Static tau on WIDTHS: count positions with survival >= 0.6 (a prefix).
    assert keep.tolist() == [4, 0, 1]
    for r in range(sv.shape[0]):
        assert _is_prefix(sv[r] >= 0.6, int(keep[r]))


# --- DSparkScheduler stateful core (CPU) ------------------------------------
#
# These drive the scheduler object itself on device="cpu". The CUDA path uses a
# pinned host buffer + a torch.cuda.Event for its async survival readout; on CPU
# the scheduler falls back to a pageable buffer and an always-ready readout
# (self._is_cuda seam), so the online overhead regression, survival calibration,
# and realized-survival accounting are exercisable without a GPU.


def _make_scheduler(
    gamma=3, max_num_reqs=8, tau=0.0, perreq=False, budget=1.0
) -> DSparkScheduler:
    spec = SimpleNamespace(
        dspark_per_request=perreq,
        dspark_confidence_threshold=tau,
        dspark_budget_frac=budget,
    )
    return DSparkScheduler(spec, gamma, max_num_reqs, torch.device("cpu"))


# --- begin_step online engine-overhead regression ---------------------------


def _obs_overhead(sched, tokens, o, gpu_pred=0.005):
    # Feed one overhead observation: with _surv_ema unset, begin_step only runs
    # the regression, recording residual ``o`` against ``tokens``. Residual =
    # (now - _last_t) - _last_gpu_pred, so pick now = _last_t + gpu_pred + o.
    sched._last_t = 0.0
    sched._last_gpu_pred = gpu_pred
    sched._last_tokens = float(tokens)
    sched.begin_step(gpu_pred + o, num_reqs=1)


def test_begin_step_overhead_regression_converges():
    sched = _make_scheduler(gamma=4)
    c0_true, c1_true = 0.02, 3e-4
    for i in range(40):
        tokens = 10 + (i % 20) * 5  # 10..105, good leverage for the fit
        _obs_overhead(sched, tokens, c0_true + c1_true * tokens)
    # Exact linear residuals => least squares recovers (c0, c1).
    assert sched._o_c1 == pytest.approx(c1_true, abs=2e-5)
    assert sched._o_c0 == pytest.approx(c0_true, abs=2e-3)
    assert 0.0 <= sched._o_c1 <= OVERHEAD_C1_CLAMP
    assert sched._o_c0 >= 0.0


def test_begin_step_overhead_outlier_gate():
    sched = _make_scheduler(gamma=4)
    for i in range(30):
        tokens = 10 + (i % 20) * 5
        _obs_overhead(sched, tokens, 0.02 + 3e-4 * tokens)
    n_before = len(sched._o_samples)
    c0_before, c1_before = sched._o_c0, sched._o_c1
    # A residual above the 0.25s idle-gap gate is dropped: no sample, no refit.
    _obs_overhead(sched, tokens=50, o=5.0)
    assert len(sched._o_samples) == n_before
    assert sched._o_c0 == c0_before
    assert sched._o_c1 == c1_before


def test_begin_step_overhead_c1_clamped_high():
    sched = _make_scheduler(gamma=4)
    # Steep slope over a small token range: residuals stay under the gate but
    # cov/var exceeds the per-token slope clamp.
    for i in range(30):
        tokens = 1 + (i % 15)  # 1..15
        _obs_overhead(sched, tokens, 0.005 + 0.01 * tokens)  # <= 0.155 < gate
    assert sched._o_c1 == pytest.approx(OVERHEAD_C1_CLAMP)
    assert sched._o_c0 >= 0.0


def test_begin_step_overhead_c1_clamped_low():
    sched = _make_scheduler(gamma=4)
    # Anti-correlated residuals -> negative raw slope -> clamped up to 0.
    for i in range(30):
        tokens = 1 + (i % 15)
        _obs_overhead(sched, tokens, 0.2 - 0.01 * tokens)  # 0.05..0.19
    assert sched._o_c1 == 0.0


# --- update_survival online calibration -------------------------------------


def _drive_readout(sched, num_reqs=2):
    # Push enough survival updates to launch (at _ema_ct == READOUT_CADENCE) and
    # then consume one host readout of the currently-set accumulators. On CPU the
    # readout is always ready, so CADENCE+1 calls suffice.
    sv = torch.zeros(num_reqs, sched.gamma)
    idx = torch.arange(num_reqs, dtype=torch.int32)
    for _ in range(READOUT_CADENCE + 1):
        sched.update_survival(sv, idx)


def test_update_survival_calibration_blend():
    sched = _make_scheduler(gamma=3)
    sched._acc_counts = torch.tensor([80.0, 50.0, 20.0])
    sched._obs_counts = torch.tensor([100.0, 100.0, 100.0])
    sched._pred_counts = torch.tensor([40.0, 40.0, 40.0])
    _drive_readout(sched)
    # ratio = realized/predicted = [2.0, 1.25, 0.5]; cal = 0.8*1 + 0.2*ratio.
    assert sched._cal == pytest.approx([1.2, 1.05, 0.9])


def test_update_survival_calibration_min_obs_gate():
    sched = _make_scheduler(gamma=3)
    sched._acc_counts = torch.tensor([80.0, 50.0, 20.0])
    sched._obs_counts = torch.tensor([100.0, 50.0, 100.0])  # pos1 below min-obs 64
    sched._pred_counts = torch.tensor([40.0, 40.0, 40.0])
    _drive_readout(sched)
    assert sched._cal[1] == 1.0  # gated out -> unchanged
    assert sched._cal[0] == pytest.approx(1.2)
    assert sched._cal[2] == pytest.approx(0.9)


def test_update_survival_calibration_ratio_clamp():
    sched = _make_scheduler(gamma=2)
    sched._acc_counts = torch.tensor([400.0, 2.0])  # raw ratios 10.0 and 0.05
    sched._obs_counts = torch.tensor([100.0, 100.0])
    sched._pred_counts = torch.tensor([40.0, 40.0])
    _drive_readout(sched)
    # 10.0 clamps to 4.0 -> 0.8 + 0.2*4 = 1.6; 0.05 clamps to 0.25 -> 0.85.
    assert sched._cal[0] == pytest.approx(1.6)
    assert sched._cal[1] == pytest.approx(0.85)


# --- observe_verified realized-survival accounting --------------------------


def test_observe_verified_counts_full_width_only():
    sched = _make_scheduler(gamma=3, max_num_reqs=8)
    sched.commit_length(3)  # _prev_sched_l > 0 so the observation runs
    num_reqs = 4
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    prev_widths = torch.zeros(8, dtype=torch.int32)
    prev_widths[:4] = torch.tensor([3, 3, 2, 3], dtype=torch.int32)
    # Predicted prefix survival (req-state-indexed) feeding pred_counts.
    sched._surv_state[0] = torch.tensor([0.9, 0.8, 0.7])
    sched._surv_state[1] = torch.tensor([0.95, 0.9, 0.85])
    sched._surv_state[2] = torch.tensor([0.6, 0.4, 0.2])
    sched._surv_state[3] = torch.tensor([0.5, 0.5, 0.5])
    # req0: width 3, full, accepted 2   -> observed [1,2,3], survived [1,2]
    # req1: width 3, full, accepted 3   -> observed [1,2,3], survived [1,2,3]
    # req2: width 2, full, accepted 1   -> observed [1,2]  (width respected)
    # req3: width 3, SHORT (not full)   -> excluded entirely
    num_sampled = torch.tensor([3, 4, 2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    sched.observe_verified(num_reqs, num_sampled, num_rejected, prev_widths, idx)

    assert sched._obs_counts.tolist() == [3.0, 3.0, 2.0]
    assert sched._acc_counts.tolist() == [3.0, 2.0, 1.0]
    assert sched._pred_counts.tolist() == pytest.approx([2.45, 2.1, 1.55])


def test_observe_verified_noop_without_prev_length():
    sched = _make_scheduler(gamma=3, max_num_reqs=8)
    # _prev_sched_l defaults to 0 -> observe_verified is a no-op.
    idx = torch.tensor([0, 1], dtype=torch.int32)
    prev_widths = torch.zeros(8, dtype=torch.int32)
    num_sampled = torch.tensor([3, 3], dtype=torch.int32)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32)
    sched.observe_verified(2, num_sampled, num_rejected, prev_widths, idx)
    assert sched._acc_counts.sum().item() == 0.0
    assert sched._obs_counts.sum().item() == 0.0


# --- engine-side live re-derivation ------------------------------------------

# Flat cost rows: T is width-independent, so throughput argmax follows
# survival alone and longer drafts win while survival mass remains.
_R_GRID = [1, 8, 64]
_FLAT_T = [[10.0, 10.0, 10.0]] * 6  # L = 0..5


def _make_rederivation(max_num_seqs=64, num_spec_tokens=5):
    from vllm.v1.spec_decode.dynamic.utils import DSparkLiveRederivation

    return DSparkLiveRederivation(_R_GRID, _FLAT_T, max_num_seqs, num_spec_tokens)


def test_rederivation_seeds_prior_and_counters():
    from vllm.v1.spec_decode.dynamic.utils import DEFAULT_SURVIVAL_PRIOR

    red = _make_rederivation()
    gamma = len(_FLAT_T) - 1
    assert red._survival == list(DEFAULT_SURVIVAL_PRIOR[:gamma])
    assert red._acc_counts == [0.0] * gamma
    assert red._obs_counts == [0.0] * gamma


def test_observe_acceptance_counts_positions():
    red = _make_rederivation()
    # 3 drafted, 2 accepted: positions 0-2 observed, 0-1 accepted.
    assert red.observe(3, 2) is None  # below re-derive cadence
    assert red._obs_counts == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert red._acc_counts == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_rederive_replaces_prior_with_measured_survival():
    red = _make_rederivation()
    # Uniform traffic: 5 drafted, 1 accepted -> position 0 survives every
    # draft (1.0), positions 1-4 never do (0.0).
    lookup = None
    for _ in range(red.REDERIVE_DRAFTS):
        lookup = red.observe(5, 1)
    assert red._survival[0] == pytest.approx(1.0)
    assert red._survival[1:] == pytest.approx([0.0] * 4)
    # Lookup is rebuilt and indexable for batch sizes 0..max_num_seqs.
    assert lookup is not None
    assert len(lookup) >= 64
    # Cadence counter reset; counters decayed, not cleared: drift tracking
    # retains half the mass.
    assert red._rederive_ct == 0
    assert red._obs_counts[0] == pytest.approx(red.REDERIVE_DRAFTS * 0.5)


def test_rederive_skips_positions_below_min_obs():
    from vllm.v1.spec_decode.dynamic.utils import DEFAULT_SURVIVAL_PRIOR

    red = _make_rederivation()
    # All traffic drafts only 1 token: positions 1+ never reach the
    # min-observation gate and must keep the prior.
    for _ in range(red.REDERIVE_DRAFTS):
        red.observe(1, 1)
    assert red._survival[0] == pytest.approx(1.0)
    assert red._survival[1:] == pytest.approx(list(DEFAULT_SURVIVAL_PRIOR[1:5]))


# --- speculative-config dspark flag validation --------------------------------


def _spec_config_kwargs(**overrides):
    # No num_speculative_tokens/model: keeps __post_init__ off the draft-model
    # resolution paths so construction reaches the trailing flag validation.
    kwargs = dict(method="dspark")
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "flags,match",
    [
        (
            dict(method="eagle", dspark_scheduler=True),
            "require method='dspark'",
        ),
        (dict(dspark_per_request=True), "requires dspark_scheduler"),
        (
            dict(dspark_scheduler=True, dspark_pad_to_bucket=True),
            "requires dspark_per_request",
        ),
        (
            dict(dspark_confidence_threshold=0.5),
            "requires dspark_scheduler",
        ),
        (
            dict(dspark_scheduler=True, dspark_budget_frac=0.5),
            "requires dspark_per_request",
        ),
        (
            dict(
                dspark_scheduler=True,
                dspark_per_request=True,
                dspark_budget_frac=0.0,
            ),
            r"in \(0, 1\]",
        ),
        (
            dict(dspark_scheduler=True, dspark_confidence_threshold=1.5),
            r"in \[0, 1\]",
        ),
    ],
)
def test_dspark_config_flag_validation(flags, match):
    from vllm.config.speculative import SpeculativeConfig

    with pytest.raises(ValueError, match=match):
        SpeculativeConfig(**_spec_config_kwargs(**flags))
