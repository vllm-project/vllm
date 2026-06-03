# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import _dp_sync_stats, _dp_sync_stats_lock
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

logger = init_logger(__name__)

_ucx_init_attempted = False


def _maybe_init_ucx(dp_rank: int, dp_size: int) -> None:
    """Lazily initialize the UCX DP communicator on first use.

    Gated on VLLM_DP_SYNC_BACKEND=ucx. Falls back to Gloo on
    failure.
    """
    global _ucx_init_attempted
    if _ucx_init_attempted:
        return
    _ucx_init_attempted = True

    if os.environ.get("VLLM_DP_SYNC_BACKEND", "").lower() != "ucx":
        return

    try:
        from vllm.distributed.device_communicators.ucx_dp_communicator import (
            try_init_ucx_dp,
        )

        gloo_group = get_dp_group().cpu_group
        try_init_ucx_dp(
            rank=dp_rank,
            world_size=dp_size,
            gloo_group=gloo_group,
            max_msg_bytes=1024,
        )
    except Exception:
        logger.warning("UCX DP init failed, using Gloo", exc_info=True)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"

    _maybe_init_ucx(dp_rank, dp_size)

    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)

    t0 = time.monotonic()

    from vllm.distributed.device_communicators.ucx_dp_communicator import (
        get_ucx_dp_communicator,
    )

    ucx = get_ucx_dp_communicator()
    if ucx is not None:
        ucx.allreduce_inplace(tensor)
    else:
        group = get_dp_group().cpu_group
        dist.all_reduce(tensor, group=group)

    elapsed = time.monotonic() - t0
    with _dp_sync_stats_lock:
        _dp_sync_stats.append(elapsed)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs
    synced_desc = cudagraph_manager.dispatch(
        num_reqs, synced_num_tokens, synced_uniform_token_count
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs, num_tokens, uniform_token_count
        )

    if dp_size == 1:
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
    )
