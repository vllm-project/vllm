# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton 合并注意力状态操作模块。

本模块实现了合并前后缀注意力状态的操作，使用 Triton kernel。
基于 https://www.arxiv.org/pdf/2501.01005 第 2.2 节实现。

主要用于合并 split-KV 场景下的部分注意力结果。

主要函数：
- merge_attn_states: 合并注意力状态
- merge_attn_states_kernel: Triton kernel 实现
"""

import torch

from vllm.triton_utils import tl, triton


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    """合并注意力状态（prefix 和 suffix）。

    基于 https://www.arxiv.org/pdf/2501.01005 第 2.2 节实现，
    用于合并 split-KV 场景下的部分注意力结果。

    Args:
        output: 输出张量 [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_output: prefix 输出张量
        prefix_lse: prefix LSE [NUM_HEADS, NUM_TOKENS]
        suffix_output: suffix 输出张量
        suffix_lse: suffix LSE [NUM_HEADS, NUM_TOKENS]
        output_lse: 输出 LSE（可选）
    """
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the attention
    # backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)
    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        head_size,
        padded_head_size,
        output_lse is not None,
    )


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    """合并注意力状态的 Triton kernel。

    将 prefix 和 suffix 的注意力输出和 LSE 合并为最终结果。
    使用数值稳定的方式计算加权和。

    Args:
        output: 输出张量指针
        output_lse: 输出 LSE 指针
        prefix_output: prefix 输出指针
        prefix_lse: prefix LSE 指针
        suffix_output: suffix 输出指针
        suffix_lse: suffix LSE 指针
        prefix_head_stride: prefix 头步幅
        output_head_stride: 输出头步幅
        HEAD_SIZE: 头维度
        PADDED_HEAD_SIZE: 填充后的头维度（2 的幂）
        OUTPUT_LSE: 是否输出 LSE
    """
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )
