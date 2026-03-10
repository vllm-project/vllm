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
    num_tokens: int,
    cudagraph_size: int,
    cudagraph_runtime_mode: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert dp_size > 1
    # Use CPU group to avoid CPU-GPU synchronization.
    group = get_dp_group().cpu_group
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = cudagraph_size
    tensor[2][dp_rank] = cudagraph_runtime_mode
    dist.all_reduce(tensor, group=group)
    return tensor[0], tensor[1], tensor[2]


def get_cudagraph_and_dp_padding(
    num_tokens: int,
    cudagraph_size: int | None,
    cudagraph_runtime_mode: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[int, torch.Tensor | None, int]:
    if dp_size == 1:
        if cudagraph_size is not None:
            return cudagraph_size, None, cudagraph_runtime_mode
        else:
            return num_tokens, None, cudagraph_runtime_mode

    # Convert None to -1 for sync (indicates no cudagraph available)
    if num_tokens == 0:
        cudagraph_size = 0
    elif cudagraph_size is None:
        cudagraph_size = -1

    num_tokens_across_dp, cudagraph_size_across_dp, cudagraph_mode_across_dp = (
        get_batch_metadata_across_dp(
            num_tokens, cudagraph_size, cudagraph_runtime_mode, dp_size, dp_rank
        )
    )
    if torch.all(num_tokens_across_dp == 0).item():
        # All ranks have zero tokens to run.
        return 0, None, 0

    # Synchronize cudagraph_runtime_mode across ranks by taking the minimum.
    synced_cudagraph_mode = int(cudagraph_mode_across_dp.min().item())
    # Check if all ranks have valid cudagraph_size.
    all_have_cudagraph = torch.all(cudagraph_size_across_dp != -1).item()

    if synced_cudagraph_mode != 0 and all_have_cudagraph:
        # All ranks use cudagraph. Pad to max cudagraph_size.
        max_cudagraph_size = int(cudagraph_size_across_dp.max().item())
        num_tokens_across_dp[:] = max_cudagraph_size
        return max_cudagraph_size, num_tokens_across_dp, synced_cudagraph_mode
    else:
        # Fall back to eager mode (no cudagraph).
        # Either some rank doesn't have cudagraph size or mode is NONE.
        synced_cudagraph_mode = 0
        num_tokens_across_dp = torch.clamp(num_tokens_across_dp, min=1)
        num_tokens_after_padding = int(num_tokens_across_dp[dp_rank].item())
        return num_tokens_after_padding, num_tokens_across_dp, synced_cudagraph_mode
