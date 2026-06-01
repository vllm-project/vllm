# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
from typing import Any

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError as e:
    raise SystemExit(
        "This script requires a Python environment with torch installed."
    ) from e

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.parallel_state import (
    get_world_group,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables


def _skip(reason: str):
    if "PYTEST_CURRENT_TEST" in os.environ:
        try:
            import pytest

            pytest.skip(reason)
        except ImportError:
            pass
    raise SystemExit(reason)


def _has_nccl_comm_split() -> bool:
    try:
        return "ncclCommSplit" in NCCLLibrary()._funcs
    except Exception:
        return False


def _distributed_run(fn, world_size: int):
    ctx = mp.get_context("spawn")
    port = get_open_port()
    processes: list[Any] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
        p = ctx.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def _comm_split_worker(env):
    update_environment_variables(env)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_distributed_environment()

    parent = None
    try:
        parent = PyNcclCommunicator(
            get_world_group().cpu_group, device=get_world_group().device
        )
        rank = parent.rank
        device = parent.device

        # First split verifies multiple non-overlapping child groups can be created
        # directly from the parent communicator.
        pair_comm = parent.split(color=rank // 2, key=rank % 2)
        assert pair_comm is not None
        assert pair_comm.world_size == 2
        assert pair_comm.rank == rank % 2

        data = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        data = pair_comm.all_reduce(data)
        torch.accelerator.synchronize()
        expected = 3.0 if rank < 2 else 7.0
        assert data.item() == expected

        data = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        pair_comm.broadcast(data, src=0)
        torch.accelerator.synchronize()
        expected = 1.0 if rank < 2 else 3.0
        assert data.item() == expected

        data = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        gathered = torch.empty(pair_comm.world_size, dtype=torch.float32, device=device)
        pair_comm.all_gather(gathered, data)
        torch.accelerator.synchronize()
        expected_gather = [1.0, 2.0] if rank < 2 else [3.0, 4.0]
        assert gathered.cpu().tolist() == expected_gather

        data = torch.tensor(
            [10 * rank + 1, 10 * rank + 2], dtype=torch.float32, device=device
        )
        reduced = torch.empty(1, dtype=torch.float32, device=device)
        pair_comm.reduce_scatter(reduced, data)
        torch.accelerator.synchronize()
        expected = {0: 12.0, 1: 14.0, 2: 52.0, 3: 54.0}[rank]
        assert reduced.item() == expected
        pair_comm.destroy()

        # Second split models DP scale-down from 4 ranks to ranks 0 and 1.
        torch.accelerator.synchronize()
        active = rank < 2
        shrunken_comm = parent.split(color=0 if active else None, key=rank)

        if active:
            assert shrunken_comm is not None
            assert shrunken_comm.world_size == 2
            assert shrunken_comm.rank == rank

            data = torch.tensor([rank + 1], dtype=torch.float32, device=device)
            data = shrunken_comm.all_reduce(data)
            torch.accelerator.synchronize()
            assert data.item() == 3.0
            shrunken_comm.destroy()
        else:
            assert shrunken_comm is None

        dist.barrier()
        torch.accelerator.synchronize()
    finally:
        if parent is not None:
            parent.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_comm_split_test(world_size: int = 4):
    if torch.accelerator.device_count() < world_size:
        _skip(f"Need at least {world_size} GPUs to run the test.")
    if not _has_nccl_comm_split():
        _skip("NCCL lacks ncclCommSplit")
    _distributed_run(_comm_split_worker, world_size)


def test_pynccl_comm_split():
    _run_comm_split_test()


def main():
    _run_comm_split_test()


if __name__ == "__main__":
    main()
