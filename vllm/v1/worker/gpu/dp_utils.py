# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_dp_group


def get_batch_metadata_across_dp(
    num_tokens: int,
    cudagraph_size: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dp_size > 1
    # Use CPU group to avoid CPU-GPU synchronization.
    group = get_dp_group().cpu_group
    tensor = torch.zeros(2, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = cudagraph_size
    dist.all_reduce(tensor, group=group)
    return tensor[0], tensor[1]
