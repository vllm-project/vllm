# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the EvenMoE-backed EPLB placement policy.

These run on CPU and need no distributed group: the placement contract is a
pure function of the load statistics and the topology.

They are skipped unless the ``evenmoe`` package is importable.
"""

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.policy import EPLB_POLICIES, DefaultEplbPolicy, EplbStats

pytest.importorskip("evenmoe")

from vllm.distributed.eplb.policy.evenmoe import EvenMoeEplbPolicy  # noqa: E402

# DeepSeek-R1 geometry: 58 MoE layers, 256 routed experts, EP=8, 32 redundant.
NUM_LAYERS = 58
NUM_LOGICAL = 256
NUM_RANKS = 8
NUM_REPLICAS = NUM_LOGICAL + 32
NUM_GROUPS = 8


def _stats(weight: torch.Tensor, num_groups: int = NUM_GROUPS) -> EplbStats:
    """The planning snapshot vLLM hands a policy, for the geometry above, one node."""
    return EplbStats(weight, NUM_REPLICAS, num_groups, 1, NUM_RANKS)


def _plan(weight: torch.Tensor, num_groups: int = NUM_GROUPS, **kwargs) -> torch.Tensor:
    return EvenMoeEplbPolicy().rebalance_experts(_stats(weight, num_groups), **kwargs)


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


def test_evenmoe_policy_is_registered():
    assert EPLB_POLICIES["evenmoe"] is EvenMoeEplbPolicy


def test_evenmoe_placement_is_valid():
    weight = _skewed_load()
    phy2log = _plan(weight)
    _assert_valid_placement(phy2log)


def test_evenmoe_placement_is_deterministic():
    weight = _skewed_load()
    a = _plan(weight)
    b = _plan(weight)
    assert torch.equal(a, b), "placement must be reproducible across EP ranks"


def test_evenmoe_balances_at_least_as_well_as_default():
    """EvenMoE's DeepSeek EPLB and vLLM's are ports of the same algorithm, so the
    balance quality should be equivalent -- not merely 'not much worse'."""
    weight = _skewed_load()
    evenmoe = _plan(weight)
    default = DefaultEplbPolicy().rebalance_experts(_stats(weight))
    evenmoe_imb = _imbalance(evenmoe, weight)
    default_imb = _imbalance(default, weight)
    assert evenmoe_imb <= default_imb * 1.02, (
        f"EvenMoE imbalance {evenmoe_imb:.4f} materially worse than "
        f"vLLM default {default_imb:.4f}"
    )


def test_slot_preservation_reduces_weight_movement():
    """The built-in policy preserves intra-GPU slots to avoid weight copies.
    EvenMoE's L1 policies do not consume a previous placement, so the glue applies
    the same pass; without it, a re-plan churns far more slots."""
    weight = _skewed_load(seed=0)
    first = _plan(weight)

    drifted = _skewed_load(seed=1) * 0.25 + weight * 0.75
    without = _plan(drifted)
    with_preserve = _plan(drifted, old_global_expert_indices=first)

    _assert_valid_placement(with_preserve)
    moved_without = int((without != first).sum())
    moved_with = int((with_preserve != first).sum())
    assert moved_with < moved_without, (
        f"slot preservation did not reduce churn: {moved_with} vs {moved_without}"
    )

    # Preservation must only permute slots inside a rank, never move an expert
    # to a different rank than EvenMoE placed it on.
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
def test_evenmoe_algorithms_selectable(monkeypatch, algorithm):
    monkeypatch.setenv("VLLM_EVENMOE_L1_ALGORITHM", algorithm)
    weight = _skewed_load()
    phy2log = _plan(weight)
    _assert_valid_placement(phy2log)


def test_no_expert_groups():
    """VLLM passes num_groups=0 for models without expert groups."""
    weight = _skewed_load()
    phy2log = _plan(weight, num_groups=0)
    _assert_valid_placement(phy2log)


def test_ep_rank_reaches_the_placement_request(monkeypatch):
    """Ultraep's placement kernel solves once per EP rank and validates the
    value is in range, so the caller's rank must reach the request -- every
    other algorithm ignores it, but the plumbing is shared."""
    import vllm.distributed.eplb.evenmoe_runtime as evenmoe_runtime

    real_placement_request = evenmoe_runtime.placement_request
    captured = {}

    def spy(*args, **kwargs):
        captured["ep_rank"] = kwargs.get("ep_rank")
        return real_placement_request(*args, **kwargs)

    monkeypatch.setattr(evenmoe_runtime, "placement_request", spy)
    weight = _skewed_load()
    _plan(weight, ep_rank=5)
    assert captured["ep_rank"] == 5


@pytest.mark.parametrize(
    ("algorithm", "per_rank"),
    [
        ("auto", False),
        ("deepseek", False),
        ("deepseek_hierarchical", False),
        ("ultraep", True),
    ],
)
def test_per_rank_statistics_follow_the_planners_declaration(
    monkeypatch, algorithm, per_rank
):
    """Whether `rearrange` gathers or sum-reduces the load windows is the
    planner's own declaration (``logical_count`` = "per_source_rank"), read
    through the balancer -- not a name this module compares against, which
    would have to be kept in step with every planner making the same choice."""
    monkeypatch.setenv("VLLM_EVENMOE_L1_ALGORITHM", algorithm)
    assert EvenMoeEplbPolicy.wants_per_rank_weight() is per_rank


def test_initial_placement_is_the_planners_own_layout(monkeypatch):
    """A planner whose placement before any statistics is its own layout
    supplies it, and `EplbState` installs it as it installs a checkpoint. The
    EPLB variants have no such layout and keep vLLM's trivial one (None)."""
    geometry = dict(
        num_moe_layers=3,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_REPLICAS,
        num_ranks=NUM_RANKS,
        num_groups=NUM_GROUPS,
        num_nodes=1,
    )
    monkeypatch.setenv("VLLM_EVENMOE_L1_ALGORITHM", "auto")
    assert EvenMoeEplbPolicy.initial_placement(**geometry) is None

    monkeypatch.setenv("VLLM_EVENMOE_L1_ALGORITHM", "ultraep")
    phy2log = EvenMoeEplbPolicy.initial_placement(**geometry)
    assert phy2log is not None
    # The shape and placement a checkpoint is installed from.
    assert phy2log.shape == (3, NUM_REPLICAS)
    assert phy2log.dtype == torch.int64
    assert phy2log.device.type == "cpu"
    # ultraep's rank-local layout: every rank holds its block of master experts
    # first, then redundant slots filled with its own masters, so the first
    # traffic-driven solve only has to touch the redundant slots.
    slots_per_rank = NUM_REPLICAS // NUM_RANKS
    masters_per_rank = NUM_LOGICAL // NUM_RANKS
    for layer in range(3):
        for rank in range(NUM_RANKS):
            slots = phy2log[layer, rank * slots_per_rank : (rank + 1) * slots_per_rank]
            own = torch.arange(rank * masters_per_rank, (rank + 1) * masters_per_rank)
            assert torch.equal(slots[:masters_per_rank], own)
            assert set(slots[masters_per_rank:].tolist()) <= set(own.tolist())
        assert len(torch.unique(phy2log[layer])) == NUM_LOGICAL
