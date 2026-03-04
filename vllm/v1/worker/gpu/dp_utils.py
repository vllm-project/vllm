# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group

if TYPE_CHECKING:
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
    batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """Sync batch metadata across DP ranks and redispatch if using cudagraph.

    Args:
        cudagraph_manager: The cudagraph manager for redispatching.
        batch_desc: Initial batch descriptor from get_cudagraph_desc.
        num_tokens: Actual number of tokens (before padding).
        dp_size: Data parallel world size.
        dp_rank: Data parallel rank.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    if dp_size == 1:
        return batch_desc, None
    group = get_dp_group().cpu_group
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = batch_desc.num_reqs
    tensor[1][dp_rank] = num_tokens
    tensor[2][dp_rank] = batch_desc.cg_mode.value
    tensor[3][dp_rank] = int(batch_desc.uniform)
    dist.all_reduce(tensor, group=group)

    num_reqs_across_dp = tensor[0]
    num_tokens_across_dp = tensor[1]
    cg_mode_across_dp = tensor[2]
    uniform_across_dp = tensor[3]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0, uniform=False
        )
        return synced_desc, None

    # Sync by taking max for sizes, min for mode/uniform (conservative).
    synced_num_reqs = int(num_reqs_across_dp.max().item())
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))
    synced_uniform = bool(uniform_across_dp.min().item())

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_num_tokens

    # Redispatch to get correct descriptor if using cudagraph.
    if synced_cg_mode != CUDAGraphMode.NONE:
        # max_query_len is unused when is_uniform is explicitly passed.
        synced_desc = cudagraph_manager.get_cudagraph_desc(
            synced_num_reqs, synced_num_tokens, synced_uniform
        )
    else:
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=synced_num_tokens,
            num_reqs=synced_num_reqs,
            uniform=synced_uniform,
        )
    return synced_desc, num_tokens_across_dp
