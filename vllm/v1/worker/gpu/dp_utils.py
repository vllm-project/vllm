# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_dp_group


def get_num_tokens_across_dp(
    local_num_tokens: int,
    dp_size: int,
    dp_rank: int,
) -> torch.Tensor:
    assert dp_size > 1
    # Use CPU group to avoid CPU-GPU synchronization.
    group = get_dp_group().cpu_group
    tensor = torch.zeros(dp_size, dtype=torch.int32, device="cpu")
    tensor[dp_rank] = local_num_tokens
    dist.all_reduce(tensor, group=group)
    return tensor
