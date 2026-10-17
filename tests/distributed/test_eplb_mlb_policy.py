# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MLB-backed EPLB placement policy.

These run on CPU and need no distributed group: the placement contract is a
pure function of the load statistics and the topology.

They are skipped unless the ``moe-load-balancer`` package is importable.
"""

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.policy import EPLB_POLICIES, DefaultEplbPolicy

pytest.importorskip("moe_load_balancer")

from vllm.distributed.eplb.policy.mlb import MlbEplbPolicy  # noqa: E402

# DeepSeek-R1 geometry: 58 MoE layers, 256 routed experts, EP=8, 32 redundant.
NUM_LAYERS = 58
NUM_LOGICAL = 256
NUM_RANKS = 8
NUM_REPLICAS = NUM_LOGICAL + 32
NUM_GROUPS = 8


def _skewed_load(seed: int = 0) -> torch.Tensor:
    """A realistic per-layer expert load: a hot head plus a long tail."""
    rng = np.random.default_rng(seed)
    base = rng.pareto(1.6, size=(NUM_LAYERS, NUM_LOGICAL)) + 1.0
    # Shuffle which experts are hot per layer, as real routing does.
    for layer in range(NUM_LAYERS):
        rng.shuffle(base[layer])
    return torch.from_numpy((base * 1000).astype(np.float32))


def _rank_loads(phy2log: torch.Tensor, weight: torch.Tensor) -> np.ndarray:
    """Per-layer, per-rank load if each replica takes an equal share."""
    phy = phy2log.numpy()
    w = weight.numpy()
    replicas = np.zeros_like(w)
    np.add.at(
        replicas,
        (np.arange(phy.shape[0])[:, None].repeat(phy.shape[1], 1), phy),
        1,
    )
    per_replica = w / np.maximum(replicas, 1)
    slots_per_rank = phy.shape[1] // NUM_RANKS
    loads = np.take_along_axis(per_replica, phy, axis=1)
    return loads.reshape(phy.shape[0], NUM_RANKS, slots_per_rank).sum(axis=2)


def _imbalance(phy2log: torch.Tensor, weight: torch.Tensor) -> float:
    loads = _rank_loads(phy2log, weight)
    return float(np.mean(loads.max(axis=1) / np.maximum(loads.mean(axis=1), 1e-9)))


def _assert_valid_placement(phy2log: torch.Tensor) -> None:
    assert phy2log.shape == (NUM_LAYERS, NUM_REPLICAS)
    assert phy2log.dtype == torch.int64
    assert phy2log.device.type == "cpu"
    arr = phy2log.numpy()
    assert arr.min() >= 0 and arr.max() < NUM_LOGICAL
    # Every logical expert must own at least one physical slot in every layer,
    # otherwise its tokens have nowhere to go.
    for layer in range(NUM_LAYERS):
        assert len(np.unique(arr[layer])) == NUM_LOGICAL


def test_mlb_policy_is_registered():
    assert EPLB_POLICIES["mlb"] is MlbEplbPolicy


def test_mlb_placement_is_valid():
    weight = _skewed_load()
    phy2log = MlbEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )
    _assert_valid_placement(phy2log)


def test_mlb_placement_is_deterministic():
    weight = _skewed_load()
    a = MlbEplbPolicy.rebalance_experts(weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS)
    b = MlbEplbPolicy.rebalance_experts(weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS)
    assert torch.equal(a, b), "placement must be reproducible across EP ranks"


def test_mlb_balances_at_least_as_well_as_default():
    """MLB's DeepSeek EPLB and vLLM's are ports of the same algorithm, so the
    balance quality should be equivalent -- not merely 'not much worse'."""
    weight = _skewed_load()
    mlb = MlbEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )
    default = DefaultEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )
    mlb_imb = _imbalance(mlb, weight)
    default_imb = _imbalance(default, weight)
    assert mlb_imb <= default_imb * 1.02, (
        f"MLB imbalance {mlb_imb:.4f} materially worse than "
        f"vLLM default {default_imb:.4f}"
    )


def test_slot_preservation_reduces_weight_movement():
    """The built-in policy preserves intra-GPU slots to avoid weight copies.
    MLB's L1 policies do not consume a previous placement, so the glue applies
    the same pass; without it, a re-plan churns far more slots."""
    weight = _skewed_load(seed=0)
    first = MlbEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )

    drifted = _skewed_load(seed=1) * 0.25 + weight * 0.75
    without = MlbEplbPolicy.rebalance_experts(
        drifted, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )
    with_preserve = MlbEplbPolicy.rebalance_experts(
        drifted, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS, old_global_expert_indices=first
    )

    _assert_valid_placement(with_preserve)
    moved_without = int((without != first).sum())
    moved_with = int((with_preserve != first).sum())
    assert moved_with < moved_without, (
        f"slot preservation did not reduce churn: {moved_with} vs {moved_without}"
    )

    # Preservation must only permute slots inside a rank, never move an expert
    # to a different rank than MLB placed it on.
    slots = NUM_REPLICAS // NUM_RANKS
    a = without.numpy().reshape(NUM_LAYERS, NUM_RANKS, slots)
    b = with_preserve.numpy().reshape(NUM_LAYERS, NUM_RANKS, slots)
    assert np.array_equal(np.sort(a, axis=2), np.sort(b, axis=2)), (
        "slot preservation changed which experts live on which rank"
    )


@pytest.mark.parametrize(
    "algorithm",
    ["auto", "deepseek", "deepseek_hierarchical", "deepseek_vec"],
)
def test_mlb_algorithms_selectable(monkeypatch, algorithm):
    monkeypatch.setenv("VLLM_MLB_L1_ALGORITHM", algorithm)
    weight = _skewed_load()
    phy2log = MlbEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS
    )
    _assert_valid_placement(phy2log)


def test_no_expert_groups():
    """VLLM passes num_groups=0 for models without expert groups."""
    weight = _skewed_load()
    phy2log = MlbEplbPolicy.rebalance_experts(weight, NUM_REPLICAS, 0, 1, NUM_RANKS)
    _assert_valid_placement(phy2log)


def test_ep_rank_reaches_the_placement_request(monkeypatch):
    """Ultraep's placement kernel solves once per EP rank and validates the
    value is in range, so the caller's rank must reach the request -- every
    other algorithm ignores it, but the plumbing is shared."""
    import vllm.distributed.eplb.mlb_runtime as mlb_runtime

    real_placement_request = mlb_runtime.placement_request
    captured = {}

    def spy(*args, **kwargs):
        captured["ep_rank"] = kwargs.get("ep_rank")
        return real_placement_request(*args, **kwargs)

    monkeypatch.setattr(mlb_runtime, "placement_request", spy)
    weight = _skewed_load()
    MlbEplbPolicy.rebalance_experts(
        weight, NUM_REPLICAS, NUM_GROUPS, 1, NUM_RANKS, ep_rank=5
    )
    assert captured["ep_rank"] == 5
