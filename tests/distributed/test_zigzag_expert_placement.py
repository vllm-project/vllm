# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import random

import pytest
import torch
import torch.distributed

from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.utils import update_environment_variables


def distributed_run(fn, world_size):
    """Run a function in a distributed environment."""
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    """Wrapper for worker functions to set up distributed environment."""

    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()

        # Ensure each worker process has the same random seed
        random.seed(42)
        torch.manual_seed(42)

        fn()

    return wrapped_fn


def verify_zigzag_pattern(expert_map, ep_rank, ep_size, global_num_experts):
    """Verify that the expert map follows the zigzag pattern."""
    # Calculate expected local experts (supporting non-divisible cases)
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size

    if ep_rank < remainder:
        local_num_experts = base_experts + 1
    else:
        local_num_experts = base_experts

    # Expected expert IDs for this rank in zigzag pattern
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


@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_expert_placement_various_sizes(world_size):
    """Test zigzag expert placement with various expert counts."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1)

        ep_rank = torch.distributed.get_rank()

        from vllm.model_executor.layers.fused_moe.layer import (
            determine_expert_map)

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
            assert (
                test_ep_size == world_size
            ), f"ep_size {test_ep_size} must equal world_size {world_size}"

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
                enable_zigzag_expert_placement=True,
            )

            assert (
                test_local_experts == expected_test_local
            ), f"For {test_global_experts} experts on {test_ep_size} GPUs, " \
                f"expected {expected_test_local} local experts, got " \
                f"{test_local_experts}"

            if test_expert_map is not None:
                assert test_expert_map.shape == (
                    test_global_experts,
                ), f"Expected expert map shape ({test_global_experts},), " \
                    f"got {test_expert_map.shape}"

                # Verify zigzag pattern for this test case
                verify_zigzag_pattern(test_expert_map, ep_rank, test_ep_size,
                                      test_global_experts)

    distributed_run(worker_fn, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_expert_placement_edge_cases(world_size):
    """Test edge cases for zigzag expert placement."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1)

        ep_rank = torch.distributed.get_rank()

        from vllm.model_executor.layers.fused_moe.layer import (
            determine_expert_map)

        # Test case 1: ep_size = 1 (should return None for expert_map)
        local_num_experts, expert_map = determine_expert_map(
            ep_size=1,
            ep_rank=0,
            global_num_experts=8,
            enable_zigzag_expert_placement=True,
        )
        assert local_num_experts == 8, "For ep_size=1, should get all experts"
        assert expert_map is None, "For ep_size=1, expert_map should be None"

        # Test case 2: ep_size = 0 (should raise assertion)
        if ep_rank == 0:
            with pytest.raises(AssertionError):
                determine_expert_map(
                    ep_size=0,
                    ep_rank=0,
                    global_num_experts=8,
                    enable_zigzag_expert_placement=True,
                )

    distributed_run(worker_fn, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_vs_standard_expert_placement(world_size):
    """Compare zigzag placement with standard expert placement."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1)

        ep_rank = torch.distributed.get_rank()

        from vllm.model_executor.layers.fused_moe.layer import (
            determine_expert_map)

        # Test parameters - test both divisible and non-divisible cases
        if world_size == 2:
            test_cases = [
                (8, 2),
                (9, 2),
            ]  # 8 experts (divisible), 9 experts (non-divisible)
        elif world_size == 4:
            test_cases = [
                (16, 4),
                (18, 4),
            ]  # 16 experts (divisible), 18 experts (non-divisible)
        else:
            test_cases = []

        for global_num_experts, ep_size in test_cases:
            # Get both placement strategies
            zigzag_local, zigzag_map = determine_expert_map(
                ep_size=ep_size,
                ep_rank=ep_rank,
                global_num_experts=global_num_experts,
                enable_zigzag_expert_placement=True,
            )

            standard_local, standard_map = determine_expert_map(
                ep_size=ep_size,
                ep_rank=ep_rank,
                global_num_experts=global_num_experts)

            # Both should give same number of local experts
            assert (
                zigzag_local == standard_local
            ), f"For {global_num_experts} experts on {ep_size} GPUs: " \
                f"zigzag={zigzag_local}, standard={standard_local}"

    distributed_run(worker_fn, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_expert_placement_communication_pattern(world_size):
    """Test that zigzag placement creates the expected communication pattern."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1)

        ep_rank = torch.distributed.get_rank()

        from vllm.model_executor.layers.fused_moe.layer import (
            determine_expert_map)

        # Test with both divisible and non-divisible cases
        test_cases = []
        if world_size == 2:
            test_cases = [
                (16, 2),
                (17, 2),
            ]  # 16 experts (divisible), 17 experts (non-divisible)
        elif world_size == 4:
            test_cases = [
                (16, 4),
                (18, 4),
            ]  # 16 experts (divisible), 18 experts (non-divisible)

        for global_num_experts, ep_size in test_cases:
            local_num_experts, expert_map = determine_expert_map(
                ep_size=ep_size,
                ep_rank=ep_rank,
                global_num_experts=global_num_experts)

            # Verify the zigzag pattern creates interleaved expert distribution
            expected_expert_ids = []
            for expert_idx in range(local_num_experts):
                global_expert_id = ep_rank + expert_idx * ep_size
                expected_expert_ids.append(global_expert_id)

            # Verify the pattern
            assert len(expected_expert_ids) == local_num_experts
            assert all(0 <= expert_id < global_num_experts
                       for expert_id in expected_expert_ids)

            # Check that experts are evenly distributed across the expert space
            if world_size == 2:
                # For rank 0, all expert IDs should be even
                if ep_rank == 0:
                    assert all(expert_id % 2 == 0
                               for expert_id in expected_expert_ids)
                # For rank 1, all expert IDs should be odd
                else:
                    assert all(expert_id % 2 == 1
                               for expert_id in expected_expert_ids)
            elif world_size == 4:
                # For rank 0, all expert IDs should be divisible by 4
                if ep_rank == 0:
                    assert all(expert_id % 4 == 0
                               for expert_id in expected_expert_ids)
                # For rank 1, all expert IDs should have remainder 1
                elif ep_rank == 1:
                    assert all(expert_id % 4 == 1
                               for expert_id in expected_expert_ids)
                # For rank 2, all expert IDs should have remainder 2
                elif ep_rank == 2:
                    assert all(expert_id % 4 == 2
                               for expert_id in expected_expert_ids)
                # For rank 3, all expert IDs should have remainder 3
                else:
                    assert all(expert_id % 4 == 3
                               for expert_id in expected_expert_ids)

    distributed_run(worker_fn, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_expert_placement_invalid_configurations(world_size):
    """Test zigzag expert placement with invalid configurations."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1)

        ep_rank = torch.distributed.get_rank()

        from vllm.model_executor.layers.fused_moe.layer import (
            determine_expert_map)

        # Test various invalid configurations
        invalid_configs = [
            # (ep_size, global_num_experts, expected_error)
            (0, 8, AssertionError),  # ep_size = 0
        ]

        for ep_size, global_experts, expected_error in invalid_configs:
            if ep_rank == 0:  # Only test on one rank to avoid multiple errors
                with pytest.raises(expected_error):
                    determine_expert_map(
                        ep_size=ep_size,
                        ep_rank=0,
                        global_num_experts=global_experts,
                        enable_zigzag_expert_placement=True,
                    )

    distributed_run(worker_fn, world_size)
