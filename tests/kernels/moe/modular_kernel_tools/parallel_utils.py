# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import os
import traceback
from collections.abc import Callable
from typing import Any, Concatenate

import torch
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import ParamSpec

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.utils.network_utils import get_open_port

## Parallel Processes Utils

P = ParamSpec("P")


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _set_vllm_config(
    vllm_config: VllmConfig, world_size: int, rank: int, local_rank: int
):
    import tempfile

    temp_file = tempfile.mkstemp()[1]

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=local_rank,
            backend="nccl",
        )

        initialize_model_parallel(
            tensor_model_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        )
        cpu_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
    return cpu_group


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, VllmConfig | None, Any, P], None],
    vllm_config: VllmConfig | None,
    env_dict: dict | None,
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

    if env_dict is not None:
        os.environ.update(env_dict)

    cpu_group = None
    if vllm_config is not None:
        cpu_group = _set_vllm_config(vllm_config, world_size, rank, local_rank)

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
            vllm_config,
            cpu_group,
            *args,
            **kwargs,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch_with_config(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, VllmConfig, Any, P], None],
    vllm_config: VllmConfig,
    env_dict: dict[Any, Any],
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
            f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}",
            worker,
            vllm_config,
            env_dict,
        )
        + args,
        nprocs=world_size,
        join=True,
    )
