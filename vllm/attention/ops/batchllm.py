# Copyright (c) Microsoft Corporation.

import torch
import triton
import triton.language as tl


def BatchLLM_merge_attn_states(
    output: torch.Tensor,
    shared_output: torch.Tensor,
    shared_lse: torch.Tensor,
    non_shared_output: torch.Tensor,
    non_shared_lse: torch.Tensor,
    shared_q_lens: torch.Tensor,
    shared_q_start_loc: torch.Tensor,
    max_shared_q_len: int,
    is_prompt: bool = True,
) -> None:
    BLOCK_M = 128 if is_prompt else 32
    reduction_tiling_num = int(triton.cdiv(max_shared_q_len, BLOCK_M))
    shared_prefix_num = shared_q_lens.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    grid_merge = (reduction_tiling_num, shared_prefix_num, num_query_heads)
    cur_non_shared_lse = non_shared_lse if is_prompt \
        else non_shared_lse.squeeze(-1)
    # TODO (BatchLLM): Use CUDA kernel instead of Triton
    # to minimize CPU overhead.

    BatchLLM_merge_attn_states_kernel[grid_merge](
        output,
        shared_output,
        shared_lse,
        non_shared_output,
        cur_non_shared_lse,
        output.stride(0),
        output.stride(1),
        shared_lse.stride(0),
        cur_non_shared_lse.stride(0),
        shared_q_lens,
        shared_q_start_loc,
        is_prompt,
        BLOCK_M,
        head_size,
        padded_head_size,
    )


@triton.jit
def BatchLLM_merge_attn_states_kernel(
    output,  # [num_query_tokens, num_query_heads, head_size]
    shared_output,  # [num_query_tokens, num_query_heads, head_size]
    shared_lse,  # [num_query_heads, num_query_tokens]
    non_shared_output,  # [num_query_tokens, num_query_heads, head_size]
    non_shared_lse,  # [num_query_heads, num_query_tokens]
    stride_obs,  # int
    stride_oh,  # int
    stride_shared_lse_0,  # int
    stride_non_shared_lse_0,  # int
    shared_q_lens,  # [num_shared_prefixes]
    shared_q_start_loc,  # [num_shared_prefixes]
    IS_PROMPT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    shared_prefix_idx = tl.program_id(1)
    cur_head = tl.program_id(2)

    total_q_seq_len = tl.load(shared_q_lens + shared_prefix_idx)

    if start_m * BLOCK_M >= total_q_seq_len:
        return
    cur_q_start_loc = tl.load(shared_q_start_loc + shared_prefix_idx)

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, PADDED_HEAD_SIZE)

    offs_o = (cur_q_start_loc + off_m
              )[:, None] * stride_obs + cur_head * stride_oh + off_d[None, :]
    mask_o = (off_m[:, None] < total_q_seq_len) and (off_d[None, :]
                                                     < HEAD_SIZE)

    o_shared_tile = tl.load(shared_output + offs_o, mask=mask_o, other=0.0)
    o_non_shared_tile = tl.load(non_shared_output + offs_o,
                                mask=mask_o,
                                other=0.0)
    offs_shared_lse = cur_head * stride_shared_lse_0 + cur_q_start_loc + off_m
    lse_shared_tile = tl.load(shared_lse + offs_shared_lse,
                              mask=off_m < total_q_seq_len,
                              other=float("-inf"))

    if IS_PROMPT:
        offs_non_shared_lse = cur_head * stride_non_shared_lse_0 + (
            cur_q_start_loc + off_m)
    else:
        offs_non_shared_lse = (cur_q_start_loc +
                               off_m) * stride_non_shared_lse_0 + cur_head
    lse_non_shared_tile = tl.load(non_shared_lse + offs_non_shared_lse,
                                  mask=off_m < total_q_seq_len,
                                  other=float("-inf"))

    max_tile = tl.maximum(lse_shared_tile, lse_non_shared_tile)
    lse_shared_tile = lse_shared_tile - max_tile
    lse_non_shared_tile = lse_non_shared_tile - max_tile
    # correction weight
    shared_sumexp_scale = tl.math.exp(lse_shared_tile) / (
        tl.math.exp(lse_shared_tile) + tl.math.exp(lse_non_shared_tile))
    non_shared_sumexp_scale = tl.math.exp(lse_non_shared_tile) / (
        tl.math.exp(lse_shared_tile) + tl.math.exp(lse_non_shared_tile))

    fixed_shared_tile = o_shared_tile * shared_sumexp_scale[:, None]
    fixed_non_shared_tile = o_non_shared_tile * non_shared_sumexp_scale[:,
                                                                        None]
    final_o = fixed_shared_tile + fixed_non_shared_tile

    tl.store(output + offs_o, final_o, mask=mask_o)
