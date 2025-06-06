# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
import traceback
from typing import Callable

import torch
from torch.multiprocessing import (
    spawn)  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    rank = node_rank * world_local_size + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            *args,
            **kwargs,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert not kwargs
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,
            "tcp://localhost:29500",
            worker,
        ) + args,
        nprocs=world_size,
        join=True,
    )


def parallel_launch_from_env(
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """
    Launches a worker function in parallel across all processes in the current
    environment. The environment must have the following variables set:
    - WORLD_SIZE: The total number of processes.
    - WORLD_LOCAL_SIZE: The number of processes on the current node.
    - NODE_RANK: The rank of the current
    - MASTER_ADDR: The address of the master process.
    - MASTER_PORT: The port of the master process.
    """
    assert not kwargs
    world_size = int(os.environ["WORLD_SIZE"])
    world_local_size = int(os.environ["WORLD_LOCAL_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_local_size,
            node_rank,
            "env://",
            worker,
        ) + args,
        nprocs=world_local_size,
        join=True,
    )
