# SPDX-License-Identifier: Apache-2.0
"""
DeepEP test utilities
"""
import dataclasses
import os
import traceback
from typing import Callable, Optional, any

import torch
from torch.distributed import ProcessGroup
from torch.multiprocessing import (
    spawn)  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (init_distributed_environment,
                              initialize_model_parallel)
from vllm.utils import get_open_port, has_deep_ep

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (  # noqa: E501
        DeepEPHTPrepareAndFinalize)
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
        DeepEPLLPrepareAndFinalize)

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


def _set_vllm_config(vllm_config: VllmConfig, world_size: int, rank: int,
                     local_rank: int):

    import tempfile
    temp_file = tempfile.mkstemp()[1]

    set_current_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=local_rank,
            backend="nccl",
        )

        initialize_model_parallel(
            tensor_model_parallel_size=vllm_config.parallel_config.
            tensor_parallel_size,
            pipeline_model_parallel_size=vllm_config.parallel_config.
            pipeline_parallel_size,
        )
        cpu_group = torch.distributed.new_group(list(range(world_size)),
                                                backend="gloo")
    return cpu_group


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    vllm_config: Optional[VllmConfig],
    env_dict: Optional[dict],
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
        args = (vllm_config, cpu_group) + args

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
            f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}",
            worker,
            None,  # vllm_config
            None,  # env_dict
        ) + args,
        nprocs=world_size,
        join=True,
    )


def parallel_launch_with_config(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    vllm_config: VllmConfig,
    env_dict: dict[any, any],
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
            vllm_config,
            env_dict,
        ) + args,
        nprocs=world_size,
        join=True,
    )


## DeepEP specific utils


@dataclasses.dataclass
class DeepEPHTArgs:
    num_local_experts: int


@dataclasses.dataclass
class DeepEPLLArgs:
    max_tokens_per_rank: int
    hidden_size: int
    num_experts: int
    use_fp8_dispatch: bool


def make_deepep_ht_a2a(pg: ProcessGroup,
                       pgi: ProcessGroupInfo,
                       dp_size: int,
                       ht_args: DeepEPHTArgs,
                       q_dtype: Optional[torch.dtype] = None,
                       block_shape: Optional[list[int]] = None):

    import deep_ep

    # high throughput a2a
    num_nvl_bytes = 1024 * 1024 * 1024  # 1GB
    num_rdma_bytes, low_latency_mode, num_qps_per_rank = 0, False, 1
    buffer = deep_ep.Buffer(group=pg,
                            num_nvl_bytes=num_nvl_bytes,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=low_latency_mode,
                            num_qps_per_rank=num_qps_per_rank)
    return DeepEPHTPrepareAndFinalize(buffer=buffer,
                                      world_size=pgi.world_size,
                                      rank=pgi.rank,
                                      dp_size=dp_size,
                                      rank_expert_offset=pgi.rank *
                                      ht_args.num_local_experts)


def make_deepep_ll_a2a(pg: ProcessGroup,
                       pgi: ProcessGroupInfo,
                       dp_size: int,
                       deepep_ll_args: DeepEPLLArgs,
                       q_dtype: Optional[torch.dtype] = None,
                       block_shape: Optional[list[int]] = None):

    import deep_ep

    # low-latency a2a
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        deepep_ll_args.max_tokens_per_rank, deepep_ll_args.hidden_size,
        pgi.world_size, deepep_ll_args.num_experts)

    buffer = deep_ep.Buffer(group=pg,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=deepep_ll_args.num_experts //
                            pgi.world_size)

    return DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        world_size=pgi.world_size,
        dp_size=dp_size,
        max_tokens_per_rank=deepep_ll_args.max_tokens_per_rank,
        use_fp8_dispatch=deepep_ll_args.use_fp8_dispatch,
    )


def make_deepep_a2a(pg: ProcessGroup,
                    pgi: ProcessGroupInfo,
                    dp_size: int,
                    deepep_ht_args: Optional[DeepEPHTArgs],
                    deepep_ll_args: Optional[DeepEPLLArgs],
                    q_dtype: Optional[torch.dtype] = None,
                    block_shape: Optional[list[int]] = None):
    if deepep_ht_args is not None:
        assert deepep_ll_args is None
        return make_deepep_ht_a2a(pg, pgi, dp_size, deepep_ht_args, q_dtype,
                                  block_shape)

    assert deepep_ll_args is not None
    return make_deepep_ll_a2a(pg, pgi, dp_size, deepep_ll_args, q_dtype,
                              block_shape)
