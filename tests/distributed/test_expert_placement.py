# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.fused_moe.layer import determine_expert_map


def verify_round_robin_pattern(expert_map, ep_rank, ep_size,
                               global_num_experts):
    """Verify that the expert map follows the round_robin pattern."""
    # Calculate expected local experts (supporting non-divisible cases)
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size

    if ep_rank < remainder:
        local_num_experts = base_experts + 1
    else:
        local_num_experts = base_experts

    # Expected expert IDs for this rank in round_robin pattern
    # For non-divisible cases, ranks with extra experts start earlier
    expected_expert_ids = []
    for expert_idx in range(local_num_experts):
        global_expert_id = ep_rank + expert_idx * ep_size
        expected_expert_ids.append(global_expert_id)

    # Check that only expected experts are mapped to this rank
    for global_expert_id in range(global_num_experts):
        if global_expert_id in expected_expert_ids:
            local_expert_id = expert_map[global_expert_id]
            expected_local_id = expected_expert_ids.index(global_expert_id)
            assert (
                local_expert_id == expected_local_id
            ), f"Global expert {global_expert_id} should map to local expert " \
                f"{expected_local_id}, got {local_expert_id}"
        else:
            assert (
                expert_map[global_expert_id] == -1
            ), f"Global expert {global_expert_id} should not be mapped to " \
                f"this rank"

    # Verify that all local expert IDs are consecutive starting from 0
    local_expert_ids = [
        expert_map[global_id] for global_id in expected_expert_ids
    ]
    expected_local_ids = list(range(local_num_experts))
    assert (
        local_expert_ids == expected_local_ids
    ), f"Expected local expert IDs {expected_local_ids}, got {local_expert_ids}"


@pytest.mark.parametrize("expert_placement_strategy", ["round_robin"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_expert_placement_various_sizes(expert_placement_strategy, world_size):
    """Test round_robin expert placement with various expert counts."""

    # Test with different global_num_experts values
    # Include both divisible and non-divisible cases
    if world_size == 2:
        test_cases = [
            (4, 2),  # 4 experts (divisible)
            (8, 2),  # 8 experts (divisible)
            (9, 2),  # 9 experts (non-divisible)
            (16, 2),  # 16 experts (divisible)
            (17, 2),  # 17 experts (non-divisible)
        ]
    elif world_size == 4:
        test_cases = [
            (8, 4),  # 8 experts (divisible)
            (16, 4),  # 16 experts (divisible)
            (18, 4),  # 18 experts (non-divisible)
            (32, 4),  # 32 experts (divisible)
            (33, 4),  # 33 experts (non-divisible)
        ]
    else:
        test_cases = []

    for test_global_experts, test_ep_size in test_cases:
        # Ensure ep_size matches world_size
        assert (test_ep_size == world_size
                ), f"ep_size {test_ep_size} must equal world_size {world_size}"

        # Test each rank
        for ep_rank in range(world_size):
            # Calculate expected local experts
            base_experts = test_global_experts // test_ep_size
            remainder = test_global_experts % test_ep_size
            if ep_rank < remainder:
                expected_test_local = base_experts + 1
            else:
                expected_test_local = base_experts

            test_local_experts, test_expert_map = determine_expert_map(
                ep_size=test_ep_size,
                ep_rank=ep_rank,
                global_num_experts=test_global_experts,
                expert_placement_strategy=expert_placement_strategy,
            )

            assert (
                test_local_experts == expected_test_local
            ), f"For {test_global_experts} experts on {test_ep_size} ranks, " \
                f"rank {ep_rank}: expected {expected_test_local} local" \
                f"experts, got {test_local_experts}"

            if test_expert_map is not None:
                assert test_expert_map.shape == (
                    test_global_experts,
                ), f"Expected expert map shape ({test_global_experts},), " \
                    f"got {test_expert_map.shape}"

                # Verify round_robin pattern for this test case
                verify_round_robin_pattern(test_expert_map, ep_rank,
                                           test_ep_size, test_global_experts)


@pytest.mark.parametrize("expert_placement_strategy", ["round_robin"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_expert_placement_edge_cases(expert_placement_strategy, world_size):
    """Test edge cases for round_robin expert placement."""

    # Test case 1: ep_size = 1 (should return None for expert_map)
    local_num_experts, expert_map = determine_expert_map(
        ep_size=1,
        ep_rank=0,
        global_num_experts=8,
        expert_placement_strategy=expert_placement_strategy,
    )
    assert local_num_experts == 8, "For ep_size=1, should get all experts"
    assert expert_map is None, "For ep_size=1, expert_map should be None"

    # Test case 2: ep_size = 0 (should raise assertion)
    with pytest.raises(AssertionError):
        determine_expert_map(
            ep_size=0,
            ep_rank=0,
            global_num_experts=8,
            expert_placement_strategy=expert_placement_strategy,
        )


def test_determine_expert_map_comprehensive():
    """Test of determine_expert_map function with various configurations."""

    # Test cases: (ep_size, ep_rank, global_num_experts,
    # expert_placement_strategy, expected_local, expected_map_pattern)
    test_cases = [
        # Round robin placement tests
        (2, 0, 8, "round_robin", 4, [0, -1, 1, -1, 2, -1, 3,
                                     -1]),  # rank 0 gets even experts
        (2, 1, 8, "round_robin", 4, [-1, 0, -1, 1, -1, 2, -1,
                                     3]),  # rank 1 gets odd experts
        (2, 0, 9, "round_robin", 5, [0, -1, 1, -1, 2, -1, 3, -1, 4
                                     ]),  # rank 0 gets 5 experts (even + last)
        (2, 1, 9, "round_robin", 4, [-1, 0, -1, 1, -1, 2, -1, 3,
                                     -1]),  # rank 1 gets 4 experts (odd)

        # 4-rank tests
        (4, 0, 8, "round_robin", 2, [0, -1, -1, -1, 1, -1, -1,
                                     -1]),  # rank 0 gets experts 0, 4
        (4, 1, 8, "round_robin", 2, [-1, 0, -1, -1, -1, 1, -1,
                                     -1]),  # rank 1 gets experts 1, 5
        (4, 2, 8, "round_robin", 2, [-1, -1, 0, -1, -1, -1, 1,
                                     -1]),  # rank 2 gets experts 2, 6
        (4, 3, 8, "round_robin", 2, [-1, -1, -1, 0, -1, -1, -1,
                                     1]),  # rank 3 gets experts 3, 7
    ]

    for ep_size, ep_rank, global_num_experts, expert_placement_strategy, \
        expected_local, expected_map_pattern in test_cases:
        local_num_experts, expert_map = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy=expert_placement_strategy,
        )

        assert local_num_experts == expected_local, \
            f"ep_size={ep_size}, ep_rank={ep_rank}, " \
            f"global_num_experts={global_num_experts}, " \
            f"expert_placement_strategy={expert_placement_strategy}: " \
            f"expected {expected_local} local experts, got {local_num_experts}"

        if expected_map_pattern is None:
            assert expert_map is None, "Expected expert_map to be None"
        else:
            assert expert_map is not None, "Expected expert_map to not be None"
            actual_map = expert_map.tolist()
            assert actual_map == expected_map_pattern, \
                f"ep_size={ep_size}, ep_rank={ep_rank}, " \
                f"global_num_experts={global_num_experts}, " \
                f"expert_placement_strategy={expert_placement_strategy}: " \
                f"expected map {expected_map_pattern}, got {actual_map}"
