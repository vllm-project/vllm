# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Dynamic SD controller (pure, no GPU)."""

import pytest

from vllm.v1.spec_decode.dynamic.controller import (
    DynamicSDController,
    controller_from_env,
)


def lookup(table: dict[int, int], max_b: int = 128) -> list[int]:
    """{first batch size with a new K: K} -> dense lookup (index 0 unused)."""
    out = [0] * (max_b + 1)
    cur = table[min(table)]
    for b in range(1, max_b + 1):
        cur = table.get(b, cur)
        out[b] = cur
    return out


def test_hysteresis_holds_k_near_a_table_edge():
    # table: K=3 up to 51, K=2 above; with hysteresis=8, K=3 is kept up to 59
    ctrl = DynamicSDController(lookup({51: 3, 52: 2}), 3, hysteresis=8)
    ks = [ctrl.choose_k(b) for b in (40, 51, 52, 55, 59, 60, 61)]
    assert ks == [3, 3, 3, 3, 3, 2, 2]
    assert ctrl.stats["held"] == 3 and ctrl.stats["switches"] == 1


def test_without_hysteresis_k_follows_the_table():
    ctrl = DynamicSDController(lookup({51: 3, 52: 2}), 3, hysteresis=0)
    assert [ctrl.choose_k(51), ctrl.choose_k(52)] == [3, 2]


def test_lookup_is_clamped_to_num_spec_tokens():
    ctrl = DynamicSDController([0, 5, 5], 3)  # table claims K=5, engine drafts 3
    assert ctrl.choose_k(1) == 3


def test_observe_clamps_impossible_counts():
    ctrl = DynamicSDController([0, 1, 1], 3)
    ctrl.observe(2, 5)  # more accepted than drafted
    ctrl.observe(-1, -1)  # negative
    assert ctrl.accepted == [1.0, 1.0, 0.0] and ctrl.drafted == [1.0, 1.0, 0.0]


def test_acceptance_blends_prior_with_live_counts():
    # prior_weight 0: only live counts count (position 3 is never drafted,
    # so it keeps the prior)
    ctrl = DynamicSDController([0, 1, 1], 3, prior_accept=[0.5, 0.5], prior_weight=0.0)
    ctrl.observe(2, 1)
    assert ctrl.acceptance_estimate() == [1.0, 0.0, 0.5]
    # huge prior weight: live counts barely move the estimate
    ctrl = DynamicSDController([0, 1, 1], 3, prior_accept=[0.5, 0.5], prior_weight=1e9)
    ctrl.observe(2, 1)
    assert ctrl.acceptance_estimate() == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)


def test_refit_rebuilds_the_table_from_live_acceptance():
    # costs: K=1 and K=2 equally fast; the table starts with both Ks
    # (K=2 below 5, K=1 above); live acceptance says nothing is ever
    # accepted, so every K yields one token per step and the cheaper-to-tie
    # K=1 must take over the whole range
    costs = {1: ([1, 8], [10.0, 10.0]), 2: ([1, 8], [10.0, 10.0])}
    ctrl = DynamicSDController(
        lookup({1: 2, 5: 1}),
        2,
        costs=costs,
        prior_accept=[0.8, 0.5],
        prior_weight=0.0,
        refit_steps=1,
    )
    for _ in range(100):
        ctrl.observe(2, 0)
    ctrl.refit()
    assert ctrl.lookup[1] == 1 and ctrl.lookup[8] == 1
    assert ctrl.stats["table_changes"] == 1


def test_refit_only_uses_ks_with_cuda_graphs():
    # live acceptance favors K=2 at every batch size, but the starting table
    # only has K=1: a re-fit must not switch to K=2
    costs = {1: ([1, 8], [10.0, 10.0]), 2: ([1, 8], [10.0, 10.0])}
    ctrl = DynamicSDController(
        lookup({8: 1}),
        2,
        costs=costs,
        prior_accept=[0.0, 0.0],
        prior_weight=0.0,
        refit_steps=1,
    )
    for _ in range(100):
        ctrl.observe(2, 2)
    ctrl.refit()
    assert ctrl.lookup[8] == 1


def test_failed_refits_keep_the_table_and_then_disable_refit():
    ctrl = DynamicSDController(
        [0, 1, 1],
        3,
        costs={1: ([1, 8], [10.0, 10.0])},
        prior_accept=[0.5, 0.5],
        refit_steps=1,
    )
    ctrl.refit = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    for _ in range(3):
        ctrl.choose_k(1)  # each step triggers a (failing) re-fit
    assert ctrl.lookup == [0, 1, 1]  # table unchanged
    assert ctrl.refit_steps == 0  # re-fit turned off
    ctrl.refit = lambda: None  # type: ignore[method-assign]
    ctrl.choose_k(1)  # no crash after disabling


def test_controller_from_env_gates_and_validates():
    # off by default
    assert controller_from_env([0, 1, 1], 3, env={}) is None
    # on, no files: hysteresis only
    ctrl = controller_from_env([0, 1, 1], 3, env={"VLLM_DSD_CONTROLLER": "1"})
    assert ctrl is not None and ctrl.refit_steps == 0 and ctrl.hysteresis == 4
    # one of costs/prior without the other: error naming both variables
    with pytest.raises(ValueError, match="VLLM_DSD_COSTS"):
        controller_from_env(
            [0, 1, 1], 3, env={"VLLM_DSD_CONTROLLER": "1", "VLLM_DSD_PRIOR": "x.json"}
        )
    # a bad switch value: error naming the variable
    with pytest.raises(ValueError, match="VLLM_DSD_HYSTERESIS"):
        controller_from_env(
            [0, 1, 1], 3, env={"VLLM_DSD_CONTROLLER": "1", "VLLM_DSD_HYSTERESIS": "x"}
        )
