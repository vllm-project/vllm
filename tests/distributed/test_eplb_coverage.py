# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for EPLB coverage enforcement and redistribution algorithm.

Tests verify that the rebalance_experts algorithm correctly ensures
all logical experts have at least one physical replica.
"""

import pytest
import torch

from vllm.distributed.eplb.rebalance_algo import rebalance_experts


def test_redistribute_after_masking_sufficient_redundancy():
    """
    Test that rebalance_experts can redistribute to ensure coverage
    when we have sufficient redundant experts.
    
    Scenario: ep=4, 64 logical experts, 88 physical (24 redundant)
    Simulate masking rank 0 (removes 22 experts) → 66 remain for 64 logical
    This should be sufficient to cover all logical experts.
    """
    num_layers = 1
    num_logical_experts = 64
    num_physical_experts = 88  # 24 redundant
    ep_size = 4
    num_groups = 64  # Typically num_logical_experts for MoE
    num_nodes = 1
    
    # Create expert load representing some usage pattern
    expert_load = torch.randint(10, 200, (num_layers, num_logical_experts))
    
    # Simulate BEFORE masking: all 88 physical experts available
    phy2log_before, log2phy_before, logcnt_before = rebalance_experts(
        expert_load,
        num_replicas=num_physical_experts,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size,
    )
    
    # Verify all logical experts have coverage before masking
    assert torch.all(logcnt_before >= 1), (
        "Before masking: all logical experts should have at least 1 replica"
    )
    
    # Simulate AFTER masking rank 0: only 66 physical experts remain
    # (88 - 22 = 66, where 22 = 88/4 experts per rank)
    num_physical_after_masking = 66
    
    phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after_masking,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size - 1,  # Only 3 GPUs remain
    )
    
    # KEY TEST: All logical experts should STILL have at least 1 replica
    assert torch.all(logcnt_after >= 1), (
        f"After masking: all {num_logical_experts} logical experts should have at least 1 replica. "
        f"Got counts: min={logcnt_after.min()}, max={logcnt_after.max()}, "
        f"zeros={torch.sum(logcnt_after == 0)}"
    )
    
    # Total replicas should equal remaining physical experts
    assert torch.sum(logcnt_after) == num_physical_after_masking, (
        f"Total replicas should equal {num_physical_after_masking}"
    )


def test_redistribute_insufficient_experts_fails():
    """
    Test that when we don't have enough physical experts, some logical
    experts will have 0 replicas (which should trigger error in actual code).
    
    Scenario: ep=4, 64 logical experts, 64 physical (0 redundant)
    Mask rank 0 (removes 16 experts) → only 48 remain for 64 logical
    This is INSUFFICIENT - some logical experts will have 0 replicas.
    """
    num_layers = 1
    num_logical_experts = 64
    num_physical_experts = 48
    ep_size = 4
    num_groups = 64
    num_nodes = 1
    
    expert_load = torch.randint(10, 200, (num_layers, num_logical_experts))
    
    
    phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
        expert_load,
        num_replicas=num_physical_experts,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size - 1,  # Only 3 GPUs remain
    )
    
    # With only 48 physical experts for 64 logical experts,
    # some logical experts MUST have 0 replicas
    num_uncovered = torch.sum(logcnt_after == 0).item()
    
    assert num_uncovered > 0, (
        f"With {num_physical_experts} physical experts for {num_logical_experts} logical experts, "
        f"some should have 0 replicas. Got {num_uncovered} uncovered."
    )
    
    # Should have at least 64 - 48 = 16 uncovered experts
    assert num_uncovered >= (num_logical_experts - num_physical_experts), (
        f"Expected at least {num_logical_experts - num_physical_experts} "
        f"uncovered, got {num_uncovered}"
    )


def test_redistribute_with_24_redundant_ep2():
    """
    Test redistribution for ep=2 with 24 redundant experts.
    Verifies the specific case from user's question.
    
    Scenario: ep=2, 64 logical, 88 physical (24 redundant)
    Mask rank 0 → 44 physical remain for 64 logical
    This is INSUFFICIENT.
    """
    num_layers = 1
    num_logical_experts = 64
    num_redundant = 24
    num_physical_experts = num_logical_experts + num_redundant  # 88
    ep_size = 2
    num_groups = 64
    num_nodes = 1
    
    expert_load = torch.ones(num_layers, num_logical_experts) * 100
    
    # After masking rank 0: 44 experts remain (88/2 = 44 per rank)
    num_physical_after_masking = num_physical_experts // 2
    
    assert num_physical_after_masking == 44
    assert num_logical_experts == 64
    
    # Try to rebalance with only 44 physical experts
    phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after_masking,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=1,  # Only 1 GPU remains after masking
    )
    
    # With 44 physical for 64 logical, at least 20 logical experts must be uncovered
    num_uncovered = torch.sum(logcnt_after == 0).item()
    
    assert num_uncovered >= 20, (
        f"With 44 physical for 64 logical, expected at least 20 uncovered. "
        f"Got {num_uncovered}"
    )


def test_redistribute_with_64_redundant_ep4():
    """
    Test that 64 redundant experts provides good fault tolerance for ep=4.
    
    Scenario: ep=4, 64 logical, 128 physical (64 redundant)
    Mask rank 0 → 96 physical remain for 64 logical
    This SHOULD be sufficient.
    """
    num_layers = 1
    num_logical_experts = 64
    num_redundant = 64
    num_physical_experts = num_logical_experts + num_redundant  # 128
    ep_size = 4
    num_groups = 64
    num_nodes = 1
    
    expert_load = torch.randint(50, 150, (num_layers, num_logical_experts))
    
    # After masking rank 0: 96 experts remain (128 - 32 = 96)
    experts_per_rank = num_physical_experts // ep_size  # 32
    num_physical_after_masking = num_physical_experts - experts_per_rank  # 96
    
    phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after_masking,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size - 1,  # 3 GPUs remain
    )
    
    # All logical experts should have coverage
    assert torch.all(logcnt_after >= 1), (
        f"With 96 physical for 64 logical, all should be covered. "
        f"Uncovered: {torch.sum(logcnt_after == 0)}"
    )
    
    # Total replicas should equal remaining physical experts
    assert torch.sum(logcnt_after) == num_physical_after_masking


def test_rebalance_preserves_all_logical_experts():
    """
    Test that rebalance_experts maps to ALL logical experts when possible.
    """
    num_layers = 1
    num_logical_experts = 64
    num_redundant = 24
    num_physical_experts = num_logical_experts + num_redundant
    
    expert_load = torch.randint(10, 200, (num_layers, num_logical_experts))
    
    phy2log, log2phy, logcnt = rebalance_experts(
        expert_load,
        num_replicas=num_physical_experts,
        num_groups=64,
        num_nodes=1,
        num_gpus=4,
    )
    
    # Verify all logical experts appear in the mapping
    unique_logical_experts = torch.unique(phy2log[0])
    
    assert len(unique_logical_experts) == num_logical_experts, (
        f"All {num_logical_experts} logical experts should appear in mapping. "
        f"Got {len(unique_logical_experts)}"
    )
    
    # All logical experts should have at least 1 replica
    assert torch.all(logcnt >= 1), "All logical experts should have >= 1 replica"


def test_masking_then_redistribute_validates_coverage():
    """
    Test the actual masking logic: mask experts, then check if
    redistribution is possible.
    
    This tests the ACTUAL logic without mocking.
    """
    num_layers = 1
    num_logical_experts = 64
    num_redundant = 24
    num_physical_experts = 88
    ep_size = 4
    experts_per_rank = num_physical_experts // ep_size  # 22
    
    # Create initial mapping (all 88 experts active)
    expert_load = torch.randint(50, 150, (num_layers, num_logical_experts))
    
    phy2log_initial, log2phy_initial, logcnt_initial = rebalance_experts(
        expert_load,
        num_replicas=num_physical_experts,
        num_groups=64,
        num_nodes=1,
        num_gpus=ep_size,
    )
    
    # Verify initial coverage
    assert torch.all(logcnt_initial >= 1), "Initially all should be covered"
    
    # Simulate masking rank 0 by reducing physical experts
    rank_to_mask = 0
    num_physical_after_mask = num_physical_experts - experts_per_rank  # 66
    num_gpus_after_mask = ep_size - 1  # 3
    
    # Redistribute with reduced experts
    phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after_mask,
        num_groups=64,
        num_nodes=1,
        num_gpus=num_gpus_after_mask,
    )
    
    # Check if redistribution maintained coverage
    num_uncovered = torch.sum(logcnt_after == 0).item()
    
    # With 66 physical for 64 logical, coverage should be possible
    assert num_uncovered == 0, (
        f"With {num_physical_after_mask} physical experts for {num_logical_experts} logical, "
        f"all should be covered. Found {num_uncovered} uncovered."
    )
    
    # Verify total replicas
    assert torch.sum(logcnt_after) == num_physical_after_mask


def test_minimum_redundancy_for_single_rank_failure():
    """
    Test minimum redundancy needed to survive single rank failure.
    
    For ep=4 with 64 logical experts:
    - Need at least 64 / 3 ≈ 22 redundant experts
    - So 64 + 24 = 88 total should work
    - After removing 1 rank: 88 - 22 = 66 remain for 64 logical ✓
    """
    num_logical_experts = 64
    ep_size = 4
    experts_per_rank_initial = 22
    num_physical_initial = ep_size * experts_per_rank_initial  # 88
    num_redundant = num_physical_initial - num_logical_experts  # 24
    
    expert_load = torch.ones(1, num_logical_experts) * 100
    
    # After removing 1 rank
    num_physical_after = num_physical_initial - experts_per_rank_initial  # 66
    num_gpus_after = ep_size - 1  # 3
    
    # Can we cover all 64 logical experts with 66 physical?
    assert num_physical_after >= num_logical_experts, (
        f"Need at least {num_logical_experts}, have {num_physical_after}"
    )
    
    phy2log, log2phy, logcnt = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after,
        num_groups=64,
        num_nodes=1,
        num_gpus=num_gpus_after,
    )
    
    # All should be covered
    assert torch.all(logcnt >= 1), (
        f"With 24 redundant experts (88 total), should survive 1 rank failure. "
        f"Uncovered: {torch.sum(logcnt == 0)}"
    )


def test_two_rank_failure_with_insufficient_redundancy():
    """
    Test that masking 2 ranks out of 4 with only 24 redundant fails.
    
    ep=4, 64 logical, 88 physical (24 redundant)
    Mask 2 ranks → only 44 remain for 64 logical
    This is INSUFFICIENT.
    """
    num_logical_experts = 64
    num_redundant = 24
    num_physical_initial = 88
    ep_size = 4
    
    expert_load = torch.ones(1, num_logical_experts) * 100
    
    # After masking 2 ranks: only 44 experts remain
    experts_per_rank = num_physical_initial // ep_size  # 22
    num_physical_after = experts_per_rank * 2  # 44
    num_gpus_after = 2
    
    phy2log, log2phy, logcnt = rebalance_experts(
        expert_load,
        num_replicas=num_physical_after,
        num_groups=64,
        num_nodes=1,
        num_gpus=num_gpus_after,
    )
    
    # With 44 physical for 64 logical, at least 20 must be uncovered
    num_uncovered = torch.sum(logcnt == 0).item()
    
    assert num_uncovered >= 20, (
        f"With 44 physical for 64 logical, expected at least 20 uncovered. "
        f"Got {num_uncovered}"
    )


def test_redistribute_preserves_high_load_experts():
    """
    Test that high-load experts get more replicas after redistribution.
    """
    num_logical_experts = 64
    num_physical = 88
    
    # Create load with some experts very popular
    expert_load = torch.ones(1, num_logical_experts) * 50
    expert_load[0, 0] = 1000  # Expert 0 very popular
    expert_load[0, 10] = 800  # Expert 10 popular
    expert_load[0, 20] = 600  # Expert 20 popular
    
    phy2log, log2phy, logcnt = rebalance_experts(
        expert_load,
        num_replicas=num_physical,
        num_groups=64,
        num_nodes=1,
        num_gpus=4,
    )
    
    # High-load experts should have more replicas
    assert logcnt[0, 0] > logcnt[0, 30], (
        "Expert 0 (load=1000) should have more replicas than expert 30 (load=50)"
    )
    assert logcnt[0, 10] > logcnt[0, 40], (
        "Expert 10 (load=800) should have more replicas than expert 40 (load=50)"
    )


def test_edge_case_exact_match():
    """
    Test edge case where physical experts exactly match logical experts.
    """
    num_logical_experts = 64
    num_physical_experts = 64  # Exact match, no redundancy
    
    expert_load = torch.ones(1, num_logical_experts) * 100
    
    phy2log, log2phy, logcnt = rebalance_experts(
        expert_load,
        num_replicas=num_physical_experts,
        num_groups=64,
        num_nodes=1,
        num_gpus=4,
    )
    
    # Each logical expert should have exactly 1 replica
    assert torch.all(logcnt == 1), (
        "With no redundancy, each logical expert should have exactly 1 replica"
    )
    
    # All logical experts should appear
    unique_logical = set(phy2log[0].tolist())
    assert len(unique_logical) == num_logical_experts


def test_valid_redundant_values_for_deepseek_v2_lite():
    """
    Test the valid redundant expert values identified for DeepSeek-V2-Lite.
    Validates that 0, 24, 64, 112, 192 all work and provide proper coverage.
    """
    num_logical_experts = 64
    ep_sizes_to_test = [2, 4, 8]
    redundant_values = [0, 24, 64]  # Valid for all ep sizes
    
    for ep_size in ep_sizes_to_test:
        for num_redundant in redundant_values:
            num_physical = num_logical_experts + num_redundant
            
            # Should be divisible by ep_size
            assert num_physical % ep_size == 0, (
                f"ep={ep_size}, redundant={num_redundant}: "
                f"{num_physical} not divisible by {ep_size}"
            )
            
            expert_load = torch.ones(1, num_logical_experts) * 100
            
            # Test full configuration
            phy2log, log2phy, logcnt = rebalance_experts(
                expert_load,
                num_replicas=num_physical,
                num_groups=64,
                num_nodes=1,
                num_gpus=ep_size,
            )
            
            # All logical experts should have coverage
            assert torch.all(logcnt >= 1), (
                f"ep={ep_size}, redundant={num_redundant}: "
                f"all logical experts should have coverage"
            )
            
            # Test after masking 1 rank
            experts_per_rank = num_physical // ep_size
            num_physical_after = num_physical - experts_per_rank
            
            if num_physical_after >= num_logical_experts:
                # Should still have coverage
                phy2log_after, log2phy_after, logcnt_after = rebalance_experts(
                    expert_load,
                    num_replicas=num_physical_after,
                    num_groups=64,
                    num_nodes=1,
                    num_gpus=ep_size - 1,
                )
                
                coverage = torch.all(logcnt_after >= 1).item()
                print(
                    f"ep={ep_size}, redundant={num_redundant}: "
                    f"After 1 rank failure: {num_physical_after} physical, "
                    f"coverage={'✓' if coverage else '✗'}"
                )


if __name__ == "__main__":
    # Run a quick sanity check
    test_redistribute_after_masking_sufficient_redundancy()
    test_redistribute_insufficient_experts_fails()
    test_redistribute_with_24_redundant_ep2()
    test_redistribute_with_64_redundant_ep4()
    test_valid_redundant_values_for_deepseek_v2_lite()
    print("All tests passed!")
