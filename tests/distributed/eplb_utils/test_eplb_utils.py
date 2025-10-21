# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import torch

# Import the functions to test
from vllm.distributed.eplb.eplb_utils.eplb_utils import (
    determine_default_log2phy_map,
    generate_log2phy_map,
    get_ep_ranks_with_expert,
    global_idx_to_rank,
    idx_global_to_local,
    idx_local_to_global,
)


# -------------------------- Test idx_local_to_global --------------------------
def test_idx_local_to_global_basic():
    """Test basic conversion from local to global index."""
    # Case 1: ep_rank=0, local_cnt=5, local_idx=3 → 0*5+3=3
    assert idx_local_to_global(local_idx=3, local_cnt=5, ep_rank=0) == 3
    # Case 2: ep_rank=2, local_cnt=4, local_idx=1 → 2*4+1=9
    assert idx_local_to_global(local_idx=1, local_cnt=4, ep_rank=2) == 9
    # Case 3: ep_rank=1, local_cnt=10, local_idx=9 → 1*10+9=19
    assert idx_local_to_global(local_idx=9, local_cnt=10, ep_rank=1) == 19


def test_idx_local_to_global_edge():
    """Test edge cases (local_idx=0, large ep_rank)."""
    # Local index = 0 (minimum valid value)
    assert idx_local_to_global(local_idx=0, local_cnt=100, ep_rank=5) == 500
    # Large ep_rank
    assert idx_local_to_global(local_idx=5, local_cnt=20, ep_rank=100) == 2005


# -------------------------- Test idx_global_to_local --------------------------
def test_idx_global_to_local_basic():
    """Test basic conversion from global to local index."""
    # Case 1: ep_rank=0, local_cnt=5, global_idx=3 → 3-0*5=3
    assert idx_global_to_local(global_idx=3, local_cnt=5, ep_rank=0) == 3
    # Case 2: ep_rank=2, local_cnt=4, global_idx=9 → 9-2*4=1
    assert idx_global_to_local(global_idx=9, local_cnt=4, ep_rank=2) == 1
    # Case 3: ep_rank=1, local_cnt=10, global_idx=19 → 19-1*10=9
    assert idx_global_to_local(global_idx=19, local_cnt=10, ep_rank=1) == 9


def test_idx_global_to_local_edge():
    """Test edge cases (global_idx=ep_rank*local_cnt, large values)."""
    # Global index = ep_rank*local_cnt (local index = 0)
    assert idx_global_to_local(global_idx=500, local_cnt=100, ep_rank=5) == 0
    # Large global index
    assert idx_global_to_local(global_idx=2005, local_cnt=20, ep_rank=100) == 5


# -------------------------- Test global_idx_to_rank --------------------------
def test_global_idx_to_rank_basic():
    """Test basic conversion from global index to rank."""
    # Case 1: local_cnt=5 → global_idx 0-4 → rank 0; 5-9 → rank 1
    assert global_idx_to_rank(global_idx=3, local_cnt=5) == 0
    assert global_idx_to_rank(global_idx=5, local_cnt=5) == 1
    # Case 2: local_cnt=4 → global_idx 8-11 → rank 2
    assert global_idx_to_rank(global_idx=9, local_cnt=4) == 2


def test_global_idx_to_rank_edge():
    """Test edge cases (global_idx=0, global_idx=local_cnt*N-1)."""
    # Global index = 0 (minimum valid value)
    assert global_idx_to_rank(global_idx=0, local_cnt=100) == 0
    # Global index = local_cnt*N - 1 (last index of rank N-1)
    assert global_idx_to_rank(global_idx=99, local_cnt=100) == 0
    # 0*100 ≤99 <1*100
    assert global_idx_to_rank(global_idx=199, local_cnt=100) == 1
    # 1*100 ≤199 <2*100


# ----------------------- Test get_ep_ranks_with_expert -----------------------
def test_get_ep_ranks_with_expert_basic():
    """Test basic scenario: get send/recv ranks for a target expert."""
    # Params: idx=2 (target expert),
    # num_local_experts=3 (each rank has 3 experts)
    old_indices = [1, 2, 2, 3]  # Global indices (rank = global_idx//3)
    new_indices = [2, 4, 2, 5]
    ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
        idx=2,
        num_local_experts=3,
        old_indices=old_indices,
        new_indices=new_indices,
    )

    assert ranks_to_send == [0]
    assert ranks_to_recv == []


def test_get_ep_ranks_with_expert_no_overlap():
    """Test scenario: recv ranks have no overlap with send ranks."""
    old_indices = [0, 0, 1]
    new_indices = [0, 3, 0]

    ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
        idx=0,
        num_local_experts=2,
        old_indices=old_indices,
        new_indices=new_indices,
    )

    assert ranks_to_send == [0]
    assert ranks_to_recv == [1]  # 1 not in send_set {0}


def test_get_ep_ranks_with_expert_duplicate_ranks():
    """
    Test scenario: old/new indices have duplicate ranks
    (should be deduplicated).
    """
    old_indices = [8, 11, 8]  # num_local_experts=3 → rank 0 (2//3=0)
    new_indices = [5, 8, 8]  # 5//3=1, 8//3=2 → ranks [1,2]

    ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
        idx=8,
        num_local_experts=3,
        old_indices=old_indices,
        new_indices=new_indices,
    )

    assert ranks_to_send == [0]  # duplicates deduplicated
    assert ranks_to_recv == []


# ------------------------- Test generate_log2phy_map -------------------------
def test_generate_log2phy_map_single_holding_rank():
    """
    Test case: Each expert is held by exactly 1 rank
    (fill others with its value).
    """
    # Expert map: shape (3 ranks, 2 global experts)
    # - Rank 0 holds expert 0 (value 0),
    # rank 1 holds expert 1 (value 1), others = -1
    expert_map = torch.tensor([[0, -1], [-1, 1], [-1, -1]], dtype=torch.int32)
    log2phy_map = generate_log2phy_map(expert_map)

    # Step 1: Verify row_indices addition (rank * num_local_experts)
    # num_local_experts = max(expert_map) +1 = 1+1=2
    # Rank 0: 0 + 0*2 = 0; Rank1:1 +1*2=3
    # Step 2: Fill -1 with the only holding rank's value
    expected = torch.tensor([[0, 3], [0, 3], [0, 3]], dtype=torch.int32)
    assert torch.equal(log2phy_map, expected)


def test_generate_log2phy_map_multiple_holding_ranks(monkeypatch):
    """
    Test case: Expert held by multiple ranks
    (fill -1 with random choices).
    """

    # Fix random seed to avoid non-determinism
    def mock_choice(arr):
        return arr[0]  # Always pick first element for consistency

    monkeypatch.setattr(random, "choice", mock_choice)

    # Expert map: shape (4 ranks, 1 global expert)
    # - Ranks 0 and 2 hold expert 0 (value 0)
    expert_map = torch.tensor([[0], [-1], [0], [-1]], dtype=torch.int32)
    log2phy_map = generate_log2phy_map(expert_map)

    # num_local_experts = 0+1=1 → row_indices = rank*1
    # Holding ranks (0,2) have values: 0+0*1=0, 0+2*1=2
    # Fill -1 ranks (1,3) with random choice (fixed to 0 here)
    expected = torch.tensor([[0], [0], [2], [0]], dtype=torch.int32)
    assert torch.equal(log2phy_map, expected)


# --------------------- Test determine_default_log2phy_map ---------------------
def test_determine_default_log2phy_map_equal_distribution():
    """Test case: Global experts are evenly distributed across ranks."""
    global_expert_num = 6  # 6 experts total
    world_size = 3  # 3 ranks → 2 experts per rank
    rank_id = 1  # Test rank 1

    log2phy_map_rank1 = determine_default_log2phy_map(
        global_expert_num,
        world_size,
        rank_id,
    )

    # Step 1: Expert_map_all (3 ranks ×6 experts)
    # Rank0: [0,1,-1,-1,-1,-1] → values 0+0*2=0, 1+0*2=1
    # Rank1: [-1,-1,0,1,-1,-1] → 0+1*2=2, 1+1*2=3
    # Rank2: [-1,-1,-1,-1,0,1] →0+2*2=4,1+2*2=5
    # Step 2: generate_log2phy_map fills -1 with the only holding rank's value
    # For rank1, log2phy_map should be [0,1,2,3,4,5]
    expected = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    assert torch.equal(log2phy_map_rank1, expected)


def test_determine_default_log2phy_map_unequal_distribution(monkeypatch):
    """
    Test case: Global experts are unevenly distributed
    (last rank has more).
    """

    def mock_choice(arr):
        return arr[0]

    monkeypatch.setattr(random, "choice", mock_choice)

    global_expert_num = 7  # 7 experts total
    world_size = 3  # 3 ranks → 2,2,3 experts (last rank has 3)
    rank_id = 2  # Test last rank

    log2phy_map_rank2 = determine_default_log2phy_map(
        global_expert_num,
        world_size,
        rank_id,
    )

    # Expert_map_all for rank2: [-1,-1,-1,-1,0,1,2] (3 local experts)
    # After generate_log2phy_map: all ranks get full expert values
    expected = torch.tensor([0, 1, 3, 4, 6, 7, 8], dtype=torch.int32)
    assert torch.equal(log2phy_map_rank2, expected)
