# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batched utility kernels used in the fused_moe operation.
"""

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _do_per_token_group_quant_fp8, _do_per_token_group_quant_fp8_colmajor)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

## Batched Per Token Quant ####


@triton.jit
def _batched_per_token_group_quant_fp8(
        expert_num_tokens,
        stride_ye,
        stride_yqe,
        stride_yse,
        # Pointers to inputs and output
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        group_size,
        # Num columns of y
        y_num_columns,
        y_row_stride,
        # Avoid to divide zero
        eps,
        # Information for float8
        fp8_min,
        fp8_max,
        # Meta-parameters
        BLOCK: tl.constexpr):

    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # early exit
        return

    groups_per_row = y_num_columns // group_size
    valid_groups_in_experts = e_num_tokens * groups_per_row
    group_id = tl.program_id(axis=1)
    if group_id >= valid_groups_in_experts:
        # early exit
        return

    y_ptr = y_ptr + expert_id * stride_ye
    y_q_ptr = y_q_ptr + expert_id * stride_yqe
    y_s_ptr = y_s_ptr + expert_id * stride_yse

    _do_per_token_group_quant_fp8(
        group_id,  # group id
        # Pointers to inputs and output
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        group_size,
        # Num columns of y
        y_num_columns,
        y_row_stride,
        # Avoid to divide zero
        eps,
        # Information for float8
        fp8_min,
        fp8_max,
        # Meta-parameters
        BLOCK)


@triton.jit
def _batched_per_token_group_quant_fp8_colmajor(
    expert_num_tokens,
    stride_ye,
    stride_yqe,
    stride_yse,
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # early exit
        return

    groups_per_row = y_num_columns // group_size
    valid_groups_in_experts = e_num_tokens * groups_per_row
    group_id = tl.program_id(axis=1)
    if group_id >= valid_groups_in_experts:
        # early exit
        return

    y_ptr = y_ptr + expert_id * stride_ye
    y_q_ptr = y_q_ptr + expert_id * stride_yqe
    y_s_ptr = y_s_ptr + expert_id * stride_yse

    _do_per_token_group_quant_fp8_colmajor(
        group_id,
        # Pointers to inputs and output
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        group_size,
        # Num columns of y
        y_num_columns,
        y_row_stride,
        # Stride from one column to the next of y_s
        y_s_col_stride,
        # Avoid to divide zero
        eps,
        # Information for float8
        fp8_min,
        fp8_max,
        # Meta-parameters
        BLOCK)


def batched_per_token_group_quant_fp8(
        x: torch.Tensor,
        x_q: Optional[torch.Tensor],
        expert_num_tokens: torch.Tensor,
        group_size: int,
        column_major_scales: bool,
        eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:

    assert (x.size(-1) % group_size == 0), (
        f"the last dimension of `x` {x.size(-1)} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    dtype = current_platform.fp8_dtype()
    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    assert x_q is None or x_q.shape == x.shape
    if x_q is None:
        x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    E, MAX_TOKENS, HIDDEN_SIZE = x.shape
    shape = (E, MAX_TOKENS, HIDDEN_SIZE // group_size)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    if column_major_scales:
        x_s = x_s.permute(-1, -2)

    M = (MAX_TOKENS * HIDDEN_SIZE) // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    grid = (E, M)

    if column_major_scales:
        _batched_per_token_group_quant_fp8_colmajor[grid](
            expert_num_tokens,
            x.stride(0),
            x_q.stride(0),
            x_s.stride(0),
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _batched_per_token_group_quant_fp8[grid](
            expert_num_tokens,
            x.stride(0),
            x_q.stride(0),
            x_s.stride(0),
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


## Batched Silu and Mul Kernel ####


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
