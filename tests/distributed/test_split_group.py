# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for split_group in GroupCoordinator.

These tests verify that:
1. split_group is used for both device and CPU group creation.
2. Multiple subgroups work correctly with split_group.
3. Both GPU and CPU all-reduce work on split groups.
"""

import os
from typing import Any

import multiprocess as mp
import pytest
import torch
import torch.distributed

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

# The whole module exercises the split_group code path, which is opt-in
# behind VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1.
pytestmark = pytest.mark.skipif(
    not envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP,
    reason=("VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 not set; split_group path is opt-in."),
)

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[mp.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12346"
        # propagate the opt-in flag to the spawned child workers
        env["VLLM_DISTRIBUTED_USE_SPLIT_GROUP"] = "1"
        p = mp.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


def _verify_device_group(coordinator: GroupCoordinator):
    """Verify device group works via all-reduce."""
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(16, 16, dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, group=coordinator.device_group)
    torch.accelerator.synchronize()
    expected = coordinator.world_size
    assert torch.all(tensor == expected).cpu().item(), (
        f"Device group all-reduce failed: expected {expected}, "
        f"got {tensor.flatten()[0].item()}"
    )


def _verify_cpu_group(coordinator: GroupCoordinator):
    """Verify CPU group works via all-reduce."""
    tensor = torch.ones(16, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, group=coordinator.cpu_group)
    expected = coordinator.world_size
    assert torch.all(tensor == expected).cpu().item(), (
        f"CPU group all-reduce failed: expected {expected}, "
        f"got {tensor.flatten()[0].item()}"
    )


# ---------------------------------------------------------------------------
# Test 1: Basic split_group path with 2 GPUs
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def split_group_basic_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    group_ranks = [list(range(world_size))]

    coordinator = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="test_split_basic",
    )

    _verify_device_group(coordinator)
    _verify_cpu_group(coordinator)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_split_group_basic():
    """Test basic GroupCoordinator creation with split_group."""
    distributed_run(split_group_basic_worker, 2)


# ---------------------------------------------------------------------------
# Test 2: Multiple subgroups with split_group (4 GPUs)
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def split_group_multiple_subgroups_worker():
    rank = torch.distributed.get_rank()
    group_ranks = [[0, 1], [2, 3]]

    coordinator = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="test_split_multi",
    )

    assert coordinator.world_size == 2

    _verify_device_group(coordinator)
    _verify_cpu_group(coordinator)

    if rank in [0, 1]:
        assert coordinator.ranks == [0, 1]
    else:
        assert coordinator.ranks == [2, 3]


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="Need at least 4 GPUs to run the test.",
)
def test_split_group_multiple_subgroups():
    """Test GroupCoordinator with multiple independent subgroups."""
    distributed_run(split_group_multiple_subgroups_worker, 4)


# ---------------------------------------------------------------------------
# Test 3: split_group contract — every parent rank must enter with the same
# ``split_ranks``. NCCL happens to produce
# correct subgroups for disjoint partitions because the wrapper hashes
# ``my_group`` to derive the comm-split color, but the contract violation is
# real and would break under non-partition / non-NCCL backends. This test
# captures the actual ``split_ranks`` argument passed on every rank and
# asserts they match.
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def split_group_contract_worker():
    rank = torch.distributed.get_rank()
    group_ranks = [[0, 1], [2, 3]]

    captured: list[list[list[int]]] = []
    original_split_group = torch.distributed.split_group

    def capturing_split_group(*args, split_ranks=None, **kwargs):
        captured.append([list(g) for g in split_ranks])
        return original_split_group(*args, split_ranks=split_ranks, **kwargs)

    torch.distributed.split_group = capturing_split_group
    try:
        GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=rank,
            torch_distributed_backend="nccl",
            use_device_communicator=False,
            group_name="test_split_contract",
        )
    finally:
        torch.distributed.split_group = original_split_group

    # GroupCoordinator builds two subgroups (device + cpu) per coordinator,
    # so every rank must have made exactly two split_group calls.
    if len(captured) != 2:
        raise AssertionError(
            f"rank {rank} expected 2 split_group calls (device + cpu), "
            f"got {len(captured)}: {captured}"
        )

    world_size = torch.distributed.get_world_size()
    for call_idx in range(2):
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, captured[call_idx])
        # Normalize for stable comparison (sort each subgroup and the outer list).
        norm = [
            sorted([sorted(sg) for sg in per_rank_args]) for per_rank_args in gathered
        ]
        reference = norm[0]
        for r, args in enumerate(norm):
            if args != reference:
                raise AssertionError(
                    f"split_group contract violation on call #{call_idx}: "
                    f"rank {r} passed split_ranks={gathered[r]}, but rank 0 "
                    f"passed split_ranks={gathered[0]}. PyTorch requires every "
                    "parent rank to enter split_group with the same split_ranks."
                )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="Need at least 4 GPUs to run the test.",
)
def test_split_group_contract_same_split_ranks_on_all_ranks():
    """All parent ranks must call torch.distributed.split_group with the same
    ``split_ranks`` argument. This catches the bug where each rank passed
    only its own subgroup (``split_ranks=[ranks]``), which NCCL forgives for
    disjoint partitions but is a documented contract violation.
    """
    distributed_run(split_group_contract_worker, 4)
