# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for the HY V4 Flash Engram layer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _lookup_norm_kernel(
    rows_ptr,
    scales_ptr,
    hashes_ptr,
    gamma_ptr,
    out_ptr,
    hashes_stride,
    eps,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SCALES: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    lanes = tl.arange(0, BLOCK_D)[None, :]
    head_mask = heads < HEADS
    mask = head_mask & (lanes < DIM)
    offsets = (token * HEADS + heads) * DIM + lanes

    x = tl.load(rows_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALES:
        scale = tl.load(scales_ptr + token * HEADS + heads, mask=head_mask, other=0.0)
        x *= scale.to(tl.float32)
    hashes = tl.load(
        hashes_ptr + token * hashes_stride + heads, mask=head_mask, other=-1
    )
    x = tl.where(mask & (hashes >= 0), x, 0.0)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / DIM + eps)[:, None]
    gamma = tl.load(gamma_ptr + heads * DIM + lanes, mask=mask, other=0.0)
    tl.store(
        out_ptr + offsets,
        (x * rstd * gamma.to(tl.float32)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def lookup_norm(
    rows: torch.Tensor,
    scales: torch.Tensor | None,
    hashes: torch.Tensor,
    gamma: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize, mask and RMS-normalize looked-up Engram rows per head.

    Args:
        rows: `[T, heads, dim]` table rows (FP8 or BF16), contiguous.
        scales: `[T, heads]` FP8 row scales, or None for BF16 tables.
        hashes: `[T, heads]` hash columns; a negative hash zeroes its head.
        gamma: `[heads, dim]` per-head norm weights.
        dtype: output dtype.

    Returns:
        `[T, heads * dim]` normalized rows.

    """
    num_tokens, heads, dim = rows.shape
    assert rows.is_contiguous() and gamma.is_contiguous() and hashes.stride(1) == 1
    out = torch.empty(num_tokens, heads * dim, dtype=dtype, device=rows.device)
    if num_tokens == 0 or heads == 0:
        return out.zero_()
    _lookup_norm_kernel[(num_tokens,)](
        rows,
        rows if scales is None else scales,
        hashes,
        gamma,
        out,
        hashes.stride(0),
        torch.finfo(torch.float32).eps,
        HEADS=heads,
        DIM=dim,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_D=triton.next_power_of_2(dim),
        HAS_SCALES=scales is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _gated_update_kernel(
    hidden_ptr,
    proj_ptr,
    hashes_ptr,
    key_norm_ptr,
    query_norm_ptr,
    pre_gamma_ptr,
    hc_scales_ptr,
    out_ptr,
    hidden_stride_t,
    hidden_stride_h,
    proj_stride,
    hashes_stride,
    out_stride_t,
    out_stride_h,
    num_hash_cols,
    eps,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    program = tl.program_id(0)
    token = (program // HC_MULT).to(tl.int64)
    channel = program % HC_MULT
    lanes = tl.arange(0, BLOCK_D)
    mask = lanes < DIM
    dtype: tl.constexpr = hidden_ptr.dtype.element_ty

    query = tl.load(
        hidden_ptr + token * hidden_stride_t + channel * hidden_stride_h + lanes,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    proj = proj_ptr + token * proj_stride
    key = tl.load(proj + channel * DIM + lanes, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(proj + HC_MULT * DIM + lanes, mask=mask, other=0.0).to(tl.float32)
    key_norm = tl.load(key_norm_ptr + channel * DIM + lanes, mask=mask, other=0.0)
    query_norm = tl.load(query_norm_ptr + channel * DIM + lanes, mask=mask, other=0.0)

    # Round at the same tensor boundaries as the eager BF16 chain.
    key = (key * tl.rsqrt(tl.sum(key * key) / DIM + eps) * key_norm.to(tl.float32)).to(
        dtype
    )
    normed_query = (
        query * tl.rsqrt(tl.sum(query * query) / DIM + eps) * query_norm.to(tl.float32)
    ).to(dtype)
    products = (key.to(tl.float32) * normed_query.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    dot = (dot * (1.0 / tl.sqrt(float(DIM)))).to(dtype).to(tl.float32)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6)).to(dtype).to(tl.float32)
    sign = tl.where(dot > 0, 1.0, tl.where(dot < 0, -1.0, 0.0))
    gate = tl.sigmoid(sign * magnitude).to(dtype).to(tl.float32)

    pre_gamma = tl.load(pre_gamma_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
    signal = pre_gamma * value * tl.rsqrt(tl.sum(value * value) / DIM + eps)
    hc_scale = tl.load(hc_scales_ptr + channel * DIM + lanes, mask=mask, other=0.0)
    update = ((gate + hc_scale.to(tl.float32)) * signal).to(dtype).to(tl.float32)

    columns = tl.arange(0, BLOCK_C)
    hashes = tl.load(
        hashes_ptr + token * hashes_stride + columns,
        mask=columns < num_hash_cols,
        other=-1,
    )
    live = tl.max((hashes >= 0).to(tl.int32)) > 0
    tl.store(
        out_ptr + token * out_stride_t + channel * out_stride_h + lanes,
        tl.where(live, query + update, query).to(dtype),
        mask=mask,
    )


def gated_update(
    hidden: torch.Tensor,
    proj: torch.Tensor,
    hashes: torch.Tensor,
    key_norm: torch.Tensor,
    query_norm: torch.Tensor,
    pre_gamma: torch.Tensor,
    hc_scales: torch.Tensor,
) -> torch.Tensor:
    """Apply the gated Engram update to the hyper-connection streams.

    Args:
        hidden: `[T, hc, dim]` streams.
        proj: `[T, (hc + 1) * dim]` reduced key projections followed by the
            value projection.
        hashes: `[T, n_hash_cols]`; tokens whose hashes are all negative pass
            through unchanged.
        key_norm: `[hc, dim]` key RMSNorm weights.
        query_norm: `[hc, dim]` query RMSNorm weights.
        pre_gamma: `[dim]` value RMSNorm weight.
        hc_scales: `[hc, dim]` per-stream update scales.

    Returns:
        `[T, hc, dim]` updated streams.

    """
    num_tokens, hc_mult, dim = hidden.shape
    assert hidden.stride(2) == 1 and proj.stride(1) == 1 and hashes.stride(1) == 1
    assert proj.shape == (num_tokens, (hc_mult + 1) * dim)
    out = torch.empty_like(hidden)
    if num_tokens == 0:
        return out
    block = triton.next_power_of_2(dim)
    _gated_update_kernel[(num_tokens * hc_mult,)](
        hidden,
        proj,
        hashes,
        key_norm,
        query_norm,
        pre_gamma,
        hc_scales,
        out,
        hidden.stride(0),
        hidden.stride(1),
        proj.stride(0),
        hashes.stride(0),
        out.stride(0),
        out.stride(1),
        hashes.shape[1],
        torch.finfo(torch.float32).eps,
        DIM=dim,
        HC_MULT=hc_mult,
        BLOCK_D=block,
        BLOCK_C=triton.next_power_of_2(hashes.shape[1]),
        num_warps=8 if block >= 2048 else 4,
    )
    return out
