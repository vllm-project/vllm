# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batched utility kernels used in the fused_moe operation.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def silu(x_tile):
    return x_tile * (1.0 / (1.0 + tl.exp(-x_tile)))


@triton.jit
def silu_and_mul(
        pid_d,
        output,  # [M, D]
        input,  # [M, D * 2]
        stride_om,
        stride_im,
        M,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        compute_type: tl.constexpr):

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    mask_m = offs_m < M

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    input_ptrs = input + offs_m * stride_im + pid_d * BLOCK_D + offs_d
    output_ptrs = output + offs_m * stride_om + pid_d * BLOCK_D + offs_d

    mask_tile = mask_m & mask_d

    x_tile = tl.load(input_ptrs, mask=mask_tile,
                     other=0.0).to(dtype=tl.float32)
    y_tile = tl.load(input_ptrs + D, mask=mask_tile, other=0.0)

    # silu and mul
    out_tile = silu(x_tile).to(dtype=compute_type)
    out_tile = out_tile * y_tile

    tl.store(output_ptrs, out_tile, mask=mask_tile)


@triton.jit
def batched_silu_and_mul_kernel(
        output,  # [E, MAX_NUM_TOKENS, D]
        input,  # [E, MAX_NUM_TOKENS, D * 2]
        expert_num_tokens,  # [E]
        stride_oe,
        stride_om,
        stride_ie,
        stride_im,
        compute_type: tl.constexpr,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr):

    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # early exit
        return

    pid_m = tl.program_id(axis=1)
    cta_m_start = pid_m * BLOCK_M
    if cta_m_start >= e_num_tokens:
        # early exit
        return

    cta_input_ptr = input + expert_id * stride_ie + cta_m_start * stride_im
    cta_output_ptr = output + expert_id * stride_oe + cta_m_start * stride_om

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)

    pid_d = tl.program_id(axis=2)
    silu_and_mul(
        pid_d,
        cta_output_ptr,
        cta_input_ptr,
        stride_om,
        stride_im,
        cta_m_size,  # M
        D,
        BLOCK_M,
        BLOCK_D,
        compute_type)


def invoke_batched_silu_and_mul(
        output: torch.Tensor,  #[E, MAX_TOKENS, D]
        input: torch.Tensor,  #[E, MAX_TOKENS, D * 2]
        expert_num_tokens: torch.Tensor):

    num_experts = output.size(0)
    max_num_tokens = output.size(1)
    D = output.size(2)

    BLOCK_D = 1024
    BLOCK_M = 1

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16
    }[output.dtype]

    grid = (num_experts, triton.cdiv(max_num_tokens,
                                     BLOCK_M), triton.cdiv(D, BLOCK_D))
    batched_silu_and_mul_kernel[grid](output, input, expert_num_tokens,
                                      output.stride(0), output.stride(1),
                                      input.stride(0), input.stride(1),
                                      compute_tl_dtype, D, BLOCK_M, BLOCK_D)
