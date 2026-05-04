# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x Triton FP8 einsum kernels for DeepSeek V4."""

import torch

from vllm.triton_utils import tl, triton


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    fp32_bits = exp_bits << 23
    return fp32_bits.view(torch.float32)


@triton.jit
def _deepseek_v4_sm12_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group: tl.constexpr,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group: tl.constexpr,
    a_scale_stride_hidden: tl.constexpr,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
) -> None:
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank)
            & (hidden[:, None] < hidden_size),
            other=0.0,
        )
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    tl.store(
        out_ptr
        + token_offsets[:, None] * out_stride_token
        + group * out_stride_group
        + out_offsets[None, :] * out_stride_rank,
        accum,
        mask=(token_offsets[:, None] < num_tokens)
        & (out_offsets[None, :] < out_rank),
    )


def deepseek_v4_sm12_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute ``bhr,hdr->bhd`` with FP32 block scales on SM12x.

    ``a`` is the transposed output of ``fused_inv_rope_fp8_quant`` with shape
    ``[tokens, groups, hidden]``. ``b`` is ``wo_a`` reshaped to
    ``[groups, out_rank, hidden]``.
    """
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if a_scale.dtype == e8m0_dtype:
        a_scale = _upcast_e8m0_to_fp32(a_scale)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _deepseek_v4_sm12_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        num_warps=4,
        num_stages=3,
    )
