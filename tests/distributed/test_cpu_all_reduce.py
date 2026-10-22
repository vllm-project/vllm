# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cpu_all_reduce_ (gloo, CPU only)."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.utils import (
    cpu_all_reduce_,
    stateless_init_torch_distributed_process_group,
)
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.cpu_test


def _make_input(rank: int, world_size: int, step: int) -> torch.Tensor:
    # Same shape as the V2 DP metadata tensor; values differ per rank/step.
    t = torch.zeros(6, world_size, dtype=torch.int32)
    t[:, rank] = torch.arange(6, dtype=torch.int32) * (rank + 1) + step
    return t


def _worker(rank: int, world_size: int, port: int, stateless_port: int):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    coord = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="cpu_all_reduce_test",
    )

    # Several steps to exercise tag reuse across consecutive calls.
    for step in range(5):
        expected = _make_input(rank, world_size, step)
        dist.all_reduce(expected, group=coord.cpu_group)
        t = _make_input(rank, world_size, step)
        out = coord.cpu_all_reduce_(t)
        assert out is t
        torch.testing.assert_close(t, expected, rtol=0, atol=0)

    # Float sums must be bitwise identical on every rank.
    t = torch.randn(6, world_size, generator=torch.Generator().manual_seed(rank))
    expected = t.clone()
    dist.all_reduce(expected, group=coord.cpu_group)
    coord.cpu_all_reduce_(t)
    torch.testing.assert_close(t, expected)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=coord.cpu_group)
    for other in gathered:
        assert torch.equal(other, t)

    # Unregistered (stateless) process group, as used by the engine-core DP
    # group and elastic EP.
    stateless_pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", stateless_port, rank, world_size, backend="gloo"
    )
    expected = _make_input(rank, world_size, 0)
    dist.all_reduce(expected)
    t = _make_input(rank, world_size, 0)
    cpu_all_reduce_(t, stateless_pg)
    torch.testing.assert_close(t, expected, rtol=0, atol=0)

    dist.destroy_process_group()


# Non-power-of-two sizes exercise the cross-block exchange.
@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 8, 9, 12])
def test_cpu_all_reduce(world_size: int):
    mp.spawn(
        _worker,
        args=(world_size, get_open_port(), get_open_port()),
        nprocs=world_size,
        join=True,
    )
