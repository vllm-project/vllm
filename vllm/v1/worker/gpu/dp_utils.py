# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_dp_group


def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None:
    if dp_size == 1:
        return None
    return torch.full((dp_size,), num_tokens, dtype=torch.int32, device="cpu")


def get_batch_metadata_across_dp(
    num_tokens: int, cudagraph_size: int, dp_size: int, dp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dp_size > 1
    # Use CPU group to avoid CPU-GPU synchronization.
    group = get_dp_group().cpu_group
    tensor = torch.zeros(2, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = cudagraph_size
    dist.all_reduce(tensor, group=group)
    return tensor[0], tensor[1]


def get_cudagraph_and_dp_padding(
    num_tokens: int, cudagraph_size: int | None, dp_size: int, dp_rank: int
) -> tuple[bool, int, torch.Tensor | None]:
    if dp_size == 1:
        if cudagraph_size is not None:
            return True, cudagraph_size, None
        else:
            return False, num_tokens, None

    if num_tokens == 0:
        cudagraph_size = 0
    elif cudagraph_size is None:
        cudagraph_size = -1
    num_tokens_across_dp, cudagraph_size_across_dp = get_batch_metadata_across_dp(
        num_tokens, cudagraph_size, dp_size, dp_rank
    )
    if torch.all(num_tokens_across_dp == 0).item():
        # All ranks have zero tokens to run.
        return False, 0, None

    if torch.all(cudagraph_size_across_dp != -1).item():
        # All ranks use CUDA graph or have zero tokens.
        # Use CUDA graph for all ranks.
        # Pad all ranks to the maximum CUDA graph size.
        max_cudagraph_size = int(cudagraph_size_across_dp.max().item())
        num_tokens_across_dp[:] = max_cudagraph_size
        return True, max_cudagraph_size, num_tokens_across_dp
    else:
        # Some ranks do not use CUDA graph. Use eager mode for all ranks.
        # No padding is needed except for ranks that have no tokens to run.
        num_tokens_across_dp = torch.clamp(num_tokens_across_dp, min=1)
        num_tokens_after_padding = int(num_tokens_across_dp[dp_rank].item())
        return False, num_tokens_after_padding, num_tokens_across_dp
