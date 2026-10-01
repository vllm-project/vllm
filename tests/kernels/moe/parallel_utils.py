# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP test utilities."""

import dataclasses
import os
import traceback
from collections.abc import Callable
from typing import Concatenate

import torch
from torch.distributed import ProcessGroup
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import ParamSpec

from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep, has_deep_ep_v2
from vllm.utils.network_utils import get_open_port

from .utils import make_test_moe_config

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DeepEPLLPrepareAndFinalize,
    )

if has_deep_ep_v2():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
        DeepEPV2PrepareAndFinalize,
    )

## Parallel Processes Utils

P = ParamSpec("P")


class GINNotAvailableError(RuntimeError):
    pass


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
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
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
    try:
        spawn(
            _worker_parallel_launch,
            args=(
                world_size,
                world_size,
                0,
                f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}",
                worker,
            )
            + args,
            nprocs=world_size,
            join=True,
        )
    except Exception as exc:
        # pytest.skip cannot propagate directly through torch.multiprocessing.
        if "GINNotAvailableError" in str(exc):
            import pytest

            pytest.skip("NCCL GIN not available (no IBGDA-capable hardware)")
        raise


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


def make_deepep_ht_a2a(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    ht_args: DeepEPHTArgs,
    q_dtype: torch.dtype | None = None,
    block_shape: list[int] | None = None,
):
    import deep_ep

    # high throughput a2a
    num_nvl_bytes = 1024 * 1024 * 1024  # 1GB
    num_rdma_bytes, low_latency_mode, num_qps_per_rank = 0, False, 1
    buffer = deep_ep.Buffer(
        group=pg,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=low_latency_mode,
        num_qps_per_rank=num_qps_per_rank,
    )
    num_experts = ht_args.num_local_experts * pgi.world_size
    return DeepEPHTPrepareAndFinalize(
        make_test_moe_config(
            ep_rank=pgi.rank,
            ep_size=pgi.world_size,
            device=pgi.device,
            num_experts=num_experts,
            num_local_experts=ht_args.num_local_experts,
            hidden_size=1,
            max_num_tokens=1,
            dp_size=dp_size,
        ),
        FUSED_MOE_UNQUANTIZED_CONFIG,
        buffer=buffer,
        num_dispatchers=pgi.world_size,
    )


def make_deepep_ll_a2a(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    deepep_ll_args: DeepEPLLArgs,
    q_dtype: torch.dtype | None = None,
    block_shape: list[int] | None = None,
):
    import deep_ep

    # low-latency a2a
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        deepep_ll_args.max_tokens_per_rank,
        deepep_ll_args.hidden_size,
        pgi.world_size,
        deepep_ll_args.num_experts,
    )

    buffer = deep_ep.Buffer(
        group=pg,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=deepep_ll_args.num_experts // pgi.world_size,
    )
    quant_config = (
        FusedMoEQuantConfig.make(
            current_platform.fp8_dtype(),
            block_shape=[128, 128],
        )
        if deepep_ll_args.use_fp8_dispatch
        else FUSED_MOE_UNQUANTIZED_CONFIG
    )

    return DeepEPLLPrepareAndFinalize(
        make_test_moe_config(
            ep_rank=pgi.rank,
            ep_size=pgi.world_size,
            device=pgi.device,
            num_experts=deepep_ll_args.num_experts,
            num_local_experts=deepep_ll_args.num_experts // pgi.world_size,
            hidden_size=deepep_ll_args.hidden_size,
            max_num_tokens=deepep_ll_args.max_tokens_per_rank,
            all2all_backend="deepep_low_latency",
        ),
        quant_config,
        buffer=buffer,
        num_dispatchers=pgi.world_size,
    )


def make_deepep_a2a(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    deepep_ht_args: DeepEPHTArgs | None,
    deepep_ll_args: DeepEPLLArgs | None,
    q_dtype: torch.dtype | None = None,
    block_shape: list[int] | None = None,
):
    if deepep_ht_args is not None:
        assert deepep_ll_args is None
        return make_deepep_ht_a2a(
            pg, pgi, dp_size, deepep_ht_args, q_dtype, block_shape
        )

    assert deepep_ll_args is not None
    return make_deepep_ll_a2a(pg, pgi, deepep_ll_args, q_dtype, block_shape)


@dataclasses.dataclass
class DeepEPV2Args:
    num_local_experts: int
    num_experts: int
    num_topk: int
    hidden_size: int
    max_tokens_per_rank: int
    use_fp8_dispatch: bool


def make_deepep_v2_a2a(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    v2_args: DeepEPV2Args,
    use_cudagraph: bool = False,
):
    import deep_ep

    from vllm.utils.nccl import query_nccl_gin_type

    # ElasticBuffer can segfault when GIN is unavailable. Initialize the
    # lazy communicator and reject unsupported systems before entering DeepEP.
    probe = torch.zeros(1, device=pgi.device)
    torch.distributed.all_reduce(probe, group=pg)
    gin_type = query_nccl_gin_type(pg)
    if gin_type is None:
        raise RuntimeError("Failed to determine NCCL GIN support")
    if gin_type == 0:
        raise GINNotAvailableError("NCCL GIN not available")

    buffer = deep_ep.ElasticBuffer(
        group=pg,
        num_max_tokens_per_rank=v2_args.max_tokens_per_rank,
        hidden=v2_args.hidden_size,
        num_topk=v2_args.num_topk,
        use_fp8_dispatch=v2_args.use_fp8_dispatch,
        allow_hybrid_mode=False,
        explicitly_destroy=True,
    )
    quant_config = (
        FusedMoEQuantConfig.make(
            current_platform.fp8_dtype(),
            block_shape=[128, 128],
        )
        if v2_args.use_fp8_dispatch
        else FUSED_MOE_UNQUANTIZED_CONFIG
    )
    return DeepEPV2PrepareAndFinalize(
        make_test_moe_config(
            ep_rank=pgi.rank,
            ep_size=pgi.world_size,
            device=pgi.device,
            num_experts=v2_args.num_experts,
            num_local_experts=v2_args.num_local_experts,
            hidden_size=v2_args.hidden_size,
            max_num_tokens=v2_args.max_tokens_per_rank,
            dp_size=dp_size,
            experts_per_token=v2_args.num_topk,
            all2all_backend="deepep_v2",
        ),
        quant_config,
        buffer=buffer,
        num_dispatchers=pgi.world_size,
        use_cudagraph=use_cudagraph,
    )
