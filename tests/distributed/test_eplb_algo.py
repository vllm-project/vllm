# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.policy.default import DefaultEplbPolicy


def test_basic_rebalance():
    """Test basic rebalancing functionality"""
    # Example from https://github.com/deepseek-ai/eplb
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )

    num_layers = weight.shape[0]
    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify output shapes
    assert phy2log.shape == (
        2,
        16,
    ), f"Expected `phy2log` shape (2, 16), got {phy2log.shape}"
    assert log2phy.shape[0] == 2, (
        f"Expected `log2phy` first dimension 2, got {log2phy.shape[0]}"
    )
    assert log2phy.shape[1] == 12, (
        f"Expected `log2phy` second dimension 12, got {log2phy.shape[1]}"
    )
    assert logcnt.shape == (
        2,
        12,
    ), f"Expected `logcnt` shape (2, 12), got {logcnt.shape}"

    # Verify physical to logical expert mapping range is correct
    assert torch.all(phy2log >= 0) and torch.all(phy2log < 12), (
        "Physical to logical mapping should be in range [0, 12)"
    )

    # Verify expert count reasonableness
    assert torch.all(logcnt >= 1), "Each logical expert should have at least 1 replica"
    assert torch.sum(logcnt, dim=1).sum() == num_replicas * num_layers, (
        f"Total replicas should be {num_replicas * num_layers}"
    )

    # Verify expected output
    expected_phy2log = torch.tensor(
        [
            [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
            [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1],
        ]
    )
    assert torch.all(phy2log == expected_phy2log)

    expected_logcnt = torch.tensor(
        [[1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], [1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1]]
    )
    assert torch.all(logcnt == expected_logcnt)


def test_single_gpu_case():
    """Test single GPU case"""
    weight = torch.tensor([[10, 20, 30, 40]])
    num_replicas = 4
    num_groups = 1
    num_nodes = 1
    num_gpus = 1

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify shapes
    assert phy2log.shape == (1, 4)
    assert log2phy.shape[0] == 1
    assert log2phy.shape[1] == 4
    assert logcnt.shape == (1, 4)

    # Verify all logical experts are mapped
    assert set(phy2log[0].tolist()) == {0, 1, 2, 3}


def test_equal_weights():
    """Test case with equal weights"""
    weight = torch.tensor([[50, 50, 50, 50, 50, 50, 50, 50]])
    num_replicas = 8
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify shapes
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 8)

    # With equal weights, each expert should have exactly one replica
    assert torch.all(logcnt == 1), (
        "With equal weights and no replication, "
        "each expert should have exactly 1 replica"
    )


def test_extreme_weight_imbalance():
    """Test extreme weight imbalance case"""
    weight = torch.tensor([[1000, 1, 1, 1, 1, 1, 1, 1]])
    num_replicas = 12
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify shapes
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)

    # Expert with highest weight (index 0) should have more replicas
    assert logcnt[0, 0] > logcnt[0, 1], (
        "Expert with highest weight should have more replicas"
    )


def test_multiple_layers():
    """Test multiple layers case"""
    weight = torch.tensor(
        [
            [10, 20, 30, 40, 50, 60],  # First layer
            [60, 50, 40, 30, 20, 10],  # Second layer (opposite weight pattern)
            [25, 25, 25, 25, 25, 25],  # Third layer (equal weights)
        ]
    )
    num_replicas = 8
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify shapes
    assert phy2log.shape == (3, 8)
    assert logcnt.shape == (3, 6)

    # Verify expert allocation is reasonable for each layer
    for layer in range(3):
        assert torch.all(phy2log[layer] >= 0) and torch.all(phy2log[layer] < 6), (
            f"Layer {layer} physical to logical mappingshould be in range [0, 6)"
        )
        assert torch.sum(logcnt[layer]) == num_replicas, (
            f"Layer {layer} total replicas should be {num_replicas}"
        )


def test_parameter_validation():
    """Test parameter validation"""
    weight = torch.tensor([[10, 20, 30, 40]])

    # Test non-divisible case - this should handle normally without throwing
    # errors because the function will fall back to global load balancing
    # strategy
    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(weight, 8, 3, 2, 4)
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 4)

    # Test cases that will actually cause errors:
    # num_physical_experts not divisible by num_gpus
    with pytest.raises(AssertionError):
        DefaultEplbPolicy.rebalance_experts(weight, 7, 2, 2, 4)  # 7 not divisible by 4


def test_small_scale_hierarchical():
    """Test small-scale hierarchical load balancing"""
    weight = torch.tensor(
        [
            [100, 50, 200, 75, 150, 25, 300, 80],  # 8 experts
        ]
    )
    num_replicas = 12
    num_groups = 4  # 4 groups, 2 experts each
    num_nodes = 2  # 2 nodes
    num_gpus = 4  # 4 GPUs

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Verify basic constraints
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)
    assert torch.sum(logcnt) == num_replicas
    assert torch.all(logcnt >= 1)

    # Expert with highest weight should have more replicas
    max_weight_expert = torch.argmax(weight[0])
    assert logcnt[0, max_weight_expert] >= 2, (
        "Highest weight expert should have multiple replicas"
    )


def test_global_load_balance_fallback():
    """Test global load balancing fallback case"""
    # When num_groups % num_nodes != 0, should fall back to global load
    # balancing
    weight = torch.tensor([[10, 20, 30, 40, 50, 60]])
    num_replicas = 8
    num_groups = 3  # Cannot be divided evenly by num_nodes=2
    num_nodes = 2
    num_gpus = 4

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Should work normally, just using global load balancing strategy
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 6)
    assert torch.sum(logcnt) == num_replicas


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    """Test device compatibility"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    weight = torch.tensor([[10, 20, 30, 40]], device=device)
    num_replicas = 6
    num_groups = 2
    num_nodes = 1
    num_gpus = 2

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )

    # Function will convert to CPU internally, but should handle different
    # device inputs normally
    assert phy2log.shape == (1, 6)
    assert logcnt.shape == (1, 4)


def test_additional_cases():
    """Test more edge cases and different parameter combinations"""

    # Test case 1: Large-scale distributed setup
    weight1 = torch.tensor(
        [[50, 100, 75, 120, 90, 60, 80, 110, 40, 70, 95, 85, 65, 55, 45, 35]]
    )
    phy2log1, log2phy1, logcnt1 = DefaultEplbPolicy.rebalance_experts(
        weight1, 24, 8, 4, 8
    )

    assert phy2log1.shape == (1, 24)
    assert logcnt1.shape == (1, 16)
    assert torch.sum(logcnt1) == 24

    # Test case 2: Different weight distributions
    weight2 = torch.tensor(
        [
            [200, 150, 100, 50, 25, 12],  # Decreasing weights
            [12, 25, 50, 100, 150, 200],  # Increasing weights
        ]
    )
    phy2log2, log2phy2, logcnt2 = DefaultEplbPolicy.rebalance_experts(
        weight2, 10, 3, 1, 2
    )

    assert phy2log2.shape == (2, 10)
    assert logcnt2.shape == (2, 6)

    # Verify high-weight experts have more replicas
    for layer in range(2):
        max_weight_idx = torch.argmax(weight2[layer])
        assert logcnt2[layer, max_weight_idx] >= 2


if __name__ == "__main__":
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )

    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log, log2phy, logcnt = DefaultEplbPolicy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    print(phy2log)

    test_basic_rebalance()


def _make_phy_replicas_idx_from_phy2log(phy2log: np.ndarray) -> np.ndarray:
    """Create replicas indices mapping from phy2log."""
    pr = np.zeros_like(phy2log, dtype=np.int64)
    for layer in range(phy2log.shape[0]):
        seen: dict[int, int] = {}
        row = phy2log[layer].tolist()
        for i, expert in enumerate(row):
            r = seen.get(expert, 0)
            pr[layer, i] = r
            seen[expert] = r + 1
    return pr


def _validate_intragpu_rearrangement(
    old_global_expert_indices: np.ndarray,
    new_phy2log: np.ndarray,
    new_phy_replicas_idx: np.ndarray,
    post_phy2log: np.ndarray,
    post_phy_replicas_idx: np.ndarray,
    num_ranks: int,
    slots_per_gpu: int,
):
    # Per-GPU checks
    for gpu_idx in range(num_ranks):
        start = gpu_idx * slots_per_gpu
        end = start + slots_per_gpu
        old_seg = old_global_expert_indices[0, start:end]
        new_seg = new_phy2log[0, start:end]
        new_rnk = new_phy_replicas_idx[0, start:end]
        post_seg = post_phy2log[0, start:end]
        post_rnk = post_phy_replicas_idx[0, start:end]

        # Pairwise equality for (expert, rank) pairs to ensure nothing is lost
        def sorted_pairs(seg, rnk):
            pairs = list(zip(seg.tolist(), rnk.tolist()))
            pairs.sort()
            return pairs

        assert sorted_pairs(post_seg, post_rnk) == sorted_pairs(new_seg, new_rnk), (
            f"Per-GPU pairs of (expert,rank) must match new mapping for GPU {gpu_idx}"
        )

        # For experts that remain on the same GPU, the old slot is preserved
        # for at least one occurrence; rank at that slot must be valid for that expert
        old_list = old_seg.tolist()
        new_list = new_seg.tolist()
        post_list = post_seg.tolist()
        remained = set(old_list) & set(new_list)
        new_ranks_for_expert: dict[int, list[int]] = {}
        for v, r in zip(new_list, new_rnk.tolist()):
            new_ranks_for_expert.setdefault(v, []).append(r)
        for expert in remained:
            old_pos = old_list.index(expert)
            assert post_list[old_pos] == expert, (
                f"Expert {expert} on GPU {gpu_idx} should stay at old slot {old_pos}"
            )
            # Rank at preserved slot must be one of the ranks
            # the expert has in new mapping
            assert post_rnk.tolist()[old_pos] in new_ranks_for_expert[expert], (
                f"Rank for expert {expert} at preserved slot on GPU {gpu_idx} "
                "must come from new mapping"
            )


@pytest.mark.parametrize(
    "num_ranks, slots_per_gpu, old_phy2log, new_phy2log",
    [
        pytest.param(
            # Setup: 2 GPUs, 4 slots each, 1 layer
            # Old mapping: GPU0 -> [0,1,2,3], GPU1 -> [4,5,6,7]
            # New mapping shuffles within GPU0 and brings 4,5 into GPU0.
            # GPU0 new -> [1,5,0,4]; GPU1 new -> [6,2,7,3]
            2,
            4,
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
            np.array([[1, 5, 0, 4, 6, 2, 7, 3]]),
            id="simple",
        ),
        pytest.param(
            # Setup: 2 GPUs, 5 slots each (total 10 physical experts), 1 layer
            # Old mapping:
            #   GPU0 -> [0, 1, 0, 2, 3]  (expert 0 duplicated)
            #   GPU1 -> [4, 5, 6, 1, 2]
            # New mapping reorders within GPUs and moves some experts across GPUs,
            # while still including duplicates:
            #   GPU0 new -> [0, 5, 4, 0, 1]  (expert 0 duplicated, 4/5 incoming)
            #   GPU1 new -> [6, 2, 3, 2, 1]  (expert 2 duplicated)
            2,
            5,
            np.array([[0, 1, 0, 2, 3, 4, 5, 6, 1, 2]]),
            np.array([[0, 5, 4, 0, 1, 6, 2, 3, 2, 1]]),
            id="duplicates",
        ),
        pytest.param(
            # Setup: 3 GPUs, 4 slots each (total 12 physical experts), 1 layer
            # Old mapping:
            #   GPU0 -> [0, 1, 2, 3]
            #   GPU1 -> [0, 1, 2, 3]
            #   GPU2 -> [0, 1, 2, 3]
            # New mapping decides to use one expert on 2 GPUs and shuffles
            # experts on the third GPU,
            #   GPU0 new -> [0, 0, 0, 0]
            #   GPU1 new -> [0, 0, 0, 0]
            #   GPU2 new -> [1, 2, 3, 0]
            3,
            4,
            np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]),
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0]]),
            id="skewed_expert",
        ),
    ],
)
def test_preserve_intragpu_slots(
    num_ranks: int,
    slots_per_gpu: int,
    old_phy2log: torch.Tensor,
    new_phy2log: torch.Tensor,
):
    """Experts that stay on a GPU keep their old slots; incoming not lost."""
    phy_replicas_idx = _make_phy_replicas_idx_from_phy2log(new_phy2log)

    post_phy2log, post_phy_replicas_idx = DefaultEplbPolicy.preserve_intragpu_slots(
        new_phy2log, phy_replicas_idx, num_ranks, old_phy2log
    )

    # Shapes preserved
    assert post_phy2log.shape == new_phy2log.shape
    assert post_phy_replicas_idx.shape == phy_replicas_idx.shape

    _validate_intragpu_rearrangement(
        old_phy2log,
        new_phy2log,
        phy_replicas_idx,
        post_phy2log,
        post_phy_replicas_idx,
        num_ranks,
        slots_per_gpu,
    )
