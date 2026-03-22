# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None:
    if dp_size == 1:
        return None
    return torch.full((dp_size,), num_tokens, dtype=torch.int32, device="cpu")


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    num_active_loras: int = 0,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    dist.all_reduce(tensor, group=group)

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
            num_active_loras=desired_batch_desc.num_active_loras,
        ), num_tokens_across_dp

    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp
