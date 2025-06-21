# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Masked kernels used in the fused_moe operation. In the batched versions
of ModularKernel.FusedMoEPermuteExpertsUnpermute, where batch_size
is the number-of-experts, only some tokens in each batch are valid.
The kernels in this file, account for that and only operate on the
valid tokens.
"""

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _do_per_token_group_quant_fp8, _do_per_token_group_quant_fp8_colmajor)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

## Masked Per Token Quant ####


@triton.jit
def _masked_per_token_group_quant_fp8(
        valid_tokens_array,
        # Batch dimension strides
        stride_yb,
        stride_yqb,
        stride_ysb,
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

    batch_id = tl.program_id(axis=0)
    num_tokens = tl.load(valid_tokens_array + batch_id)
    if num_tokens == 0:
        # early exit
        return

    groups_per_row = y_num_columns // group_size
    valid_num_groups = num_tokens * groups_per_row
    group_id = tl.program_id(axis=1)
    if group_id >= valid_num_groups:
        # early exit
        return

    y_ptr = y_ptr + batch_id * stride_yb
    y_q_ptr = y_q_ptr + batch_id * stride_yqb
    y_s_ptr = y_s_ptr + batch_id * stride_ysb

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
def _masked_per_token_group_quant_fp8_colmajor(
    valid_tokens_array,
    # Batch strides
    stride_yb,
    stride_yqb,
    stride_ysb,
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
    batch_id = tl.program_id(axis=0)
    num_tokens = tl.load(valid_tokens_array + batch_id)
    if num_tokens == 0:
        # early exit
        return

    group_id = tl.program_id(axis=1)
    groups_per_row = y_num_columns // group_size
    valid_num_groups = num_tokens * groups_per_row
    if group_id >= valid_num_groups:
        # early exit
        return

    y_ptr = y_ptr + batch_id * stride_yb
    y_q_ptr = y_q_ptr + batch_id * stride_yqb
    y_s_ptr = y_s_ptr + batch_id * stride_ysb

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


def masked_per_token_group_quant_fp8(
    x: torch.Tensor,  # [B, MAX_TOKENS, HIDDEN_SIZE]
    valid_tokens_array: torch.Tensor,  # [B]
    group_size: int,
    column_major_scales: bool,
    x_q: Optional[torch.Tensor] = None,  # [B, MAX_TOKENS, HIDDEN_SIZE]
    eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.ndim == 3
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

    B, MAX_TOKENS, HIDDEN_SIZE = x.shape
    shape = (B, MAX_TOKENS, HIDDEN_SIZE // group_size)
    if column_major_scales:
        cms_shape = (shape[0], shape[2], shape[1])
        x_s = torch.empty(cms_shape, device=x.device, dtype=torch.float32)
        x_s = x_s.permute(0, 2, 1)
    else:
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    M = (MAX_TOKENS * HIDDEN_SIZE) // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    grid = (B, M)

    if column_major_scales:
        _masked_per_token_group_quant_fp8_colmajor[grid](
            valid_tokens_array,
            x.stride(0),
            x_q.stride(0),
            x_s.stride(0),
            x,
            x_q,
            x_s,
            group_size,
            x.size(2),  # num_columns
            x.stride(1),  # row_stride
            x_s.stride(2),  # col_stride
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _masked_per_token_group_quant_fp8[grid](
            valid_tokens_array,
            x.stride(0),
            x_q.stride(0),
            x_s.stride(0),
            x,
            x_q,
            x_s,
            group_size,
            x.size(2),  # num_columns
            x.stride(1),  # row_stride
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

    remaining_d = D - (pid_d * BLOCK_D)

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    mask_m = offs_m < M

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < remaining_d

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
def masked_silu_and_mul_kernel(
        output,  # [B, MAX_NUM_TOKENS, D]
        input,  # [B, MAX_NUM_TOKENS, D * 2]
        valid_tokens_array,  # [B]
        stride_oe,
        stride_om,
        stride_ie,
        stride_im,
        compute_type: tl.constexpr,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr):

    batch_id = tl.program_id(axis=0)
    num_tokens = tl.load(valid_tokens_array + batch_id)
    if num_tokens == 0:
        # early exit
        return

    pid_m = tl.program_id(axis=1)
    cta_m_start = pid_m * BLOCK_M
    if cta_m_start >= num_tokens:
        # early exit
        return

    cta_input_ptr = input + batch_id * stride_ie + cta_m_start * stride_im
    cta_output_ptr = output + batch_id * stride_oe + cta_m_start * stride_om

    cta_m_size = min(BLOCK_M, num_tokens - cta_m_start)

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


def invoke_masked_silu_and_mul(
        output: torch.Tensor,  #[B, MAX_TOKENS, D]
        input: torch.Tensor,  #[B, MAX_TOKENS, D * 2]
        valid_tokens_array: torch.Tensor):
    assert input.ndim == 3
    batch_size, max_num_tokens, D = output.size()

    BLOCK_D = 1024
    BLOCK_M = 1

    compute_tl_dtype = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16
    }[output.dtype]

    grid = (batch_size, triton.cdiv(max_num_tokens,
                                    BLOCK_M), triton.cdiv(D, BLOCK_D))
    masked_silu_and_mul_kernel[grid](output, input, valid_tokens_array,
                                     output.stride(0), output.stride(1),
                                     input.stride(0), input.stride(1),
                                     compute_tl_dtype, D, BLOCK_M, BLOCK_D)
