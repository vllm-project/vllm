# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NCCL2 process-group initialization."""

import os

import multiprocess as mp
import pytest
import torch
import torch.distributed

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _default_group_options,
    _pp_device_backend,
    _use_lazy_device_group,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size):
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346",
            "TORCH_DIST_USE_NCCL2": "1",
            "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0",
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        assert process.exitcode == 0


def worker_fn_wrapper(fn):
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ["LOCAL_RANK"]
        torch.accelerator.set_device_index(torch.device(f"cuda:{local_rank}"))
        init_distributed_environment()
        fn()

    return wrapped_fn


@worker_fn_wrapper
def singleton_group_lazy_init_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    default_backend = (
        torch.distributed.distributed_c10d._get_default_group()._get_backend(
            torch.device("cuda")
        )
    )
    assert default_backend.comm_ptr == 0

    coordinator = GroupCoordinator(
        group_ranks=[[group_rank] for group_rank in range(world_size)],
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="singleton",
    )
    backend = coordinator.device_group._get_backend(torch.device("cuda"))
    assert backend.comm_ptr == 0

    tensor = torch.ones(1, device=f"cuda:{rank}")
    torch.distributed.all_reduce(tensor, group=coordinator.device_group)
    assert backend.comm_ptr != 0
    assert tensor.item() == 1


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_singleton_group_uses_lazy_init():
    distributed_run(singleton_group_lazy_init_worker, 2)


@worker_fn_wrapper
def device_communicator_group_lazy_init_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    coordinator = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=True,
        group_name="tp",
    )

    backend = coordinator.device_group._get_backend(torch.device("cuda"))
    assert backend.comm_ptr == 0

    tensor = torch.ones(1, device=f"cuda:{rank}")
    torch.distributed.all_reduce(tensor, group=coordinator.device_group)
    assert backend.comm_ptr != 0
    assert tensor.item() == world_size


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_device_communicator_group_uses_lazy_init():
    distributed_run(device_communicator_group_lazy_init_worker, 2)


@pytest.mark.parametrize(
    ("group_name", "use_device_communicator", "expected"),
    [
        ("world", False, True),
        ("tp", True, True),
        ("dp", True, True),
        ("ep", True, True),
        ("tp", False, False),
        ("pp", True, True),
    ],
)
def test_lazy_device_group_selection(
    group_name: str,
    use_device_communicator: bool,
    expected: bool,
):
    assert _use_lazy_device_group(group_name, use_device_communicator) is expected


def test_default_group_lazy_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TORCH_DIST_USE_NCCL2", "1")
    options = _default_group_options("nccl")
    assert options is not None
    assert options.lazy_init

    monkeypatch.setenv("TORCH_DIST_USE_NCCL2", "0")
    assert _default_group_options("nccl") is None


@worker_fn_wrapper
def pipeline_parallel_lazy_p2p_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    original_split_group = torch.distributed.split_group

    def unexpected_split_group(*args, **kwargs):
        raise AssertionError("pipeline-parallel groups must use new_group")

    torch.distributed.split_group = unexpected_split_group
    try:
        coordinator = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            torch_distributed_backend="nccl",
            use_device_communicator=False,
            group_name="pp",
        )
    finally:
        torch.distributed.split_group = original_split_group

    assert torch.distributed.get_backend(coordinator.device_group) == "nccl-lazy"


@pytest.mark.skipif(
    not torch.distributed.is_backend_available("nccl-lazy"),
    reason="NCCL2 lazy backend is unavailable.",
)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_pipeline_parallel_uses_lazy_p2p_group():
    """PP uses new_group with NCCL2's per-peer lazy backend."""
    distributed_run(pipeline_parallel_lazy_p2p_worker, 2)


@pytest.mark.parametrize(
    ("resolved_backend", "expected"),
    [
        ("nccl2", "nccl-lazy"),
        ("nccl", "nccl"),
    ],
)
def test_pipeline_parallel_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    resolved_backend: str,
    expected: str,
):
    class BackendImpl:
        def name(self):
            return resolved_backend

    class DefaultProcessGroup:
        def _get_backend(self, device):
            assert device.type == "cuda"
            return BackendImpl()

    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: DefaultProcessGroup(),
    )

    assert _pp_device_backend("pp", "nccl") == expected
