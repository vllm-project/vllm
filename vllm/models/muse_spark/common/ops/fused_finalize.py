# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MuseSpark post-expert RMSNorm, route weighting and top-k sum."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _post_expert_finalize_kernel(
    x_ptr,
    w_ptr,
    tw_ptr,
    out_ptr,
    expert_ids_ptr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    stride_ids,
    idx_ptr,
    PERMUTED: tl.constexpr,
    stride_idx,
    K,
    H,
    eps,
    gain_center,
    stride_xt,
    stride_xk,
    stride_twt,
    stride_out,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    offs = tl.arange(0, BLOCK_H)
    k_mask = k < K
    h_mask = offs < H
    tw = tl.load(tw_ptr + t * stride_twt + k, mask=k_mask, other=0.0).to(tl.float32)
    local = k_mask
    if HAS_EXPERT_MAP:
        expert = tl.load(expert_ids_ptr + t * stride_ids + k, mask=k_mask, other=-1)
        local = k_mask & (
            tl.load(expert_map_ptr + expert, mask=k_mask & (expert >= 0), other=-1) >= 0
        )
    if PERMUTED:
        row = tl.load(idx_ptr + t * stride_idx + k, mask=k_mask, other=-1)
        local = local & (row >= 0)
        x_ptrs = x_ptr + row.to(tl.int64)[:, None] * stride_xt + offs[None, :]
    else:
        x_ptrs = x_ptr + t * stride_xt + k[:, None] * stride_xk + offs[None, :]
    x = tl.load(
        x_ptrs,
        mask=local[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=h_mask, other=0.0).to(tl.float32) + gain_center
        if WEIGHT_FIRST:
            x = x * w[None, :]
    scale = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps) * tw
    acc = tl.sum(x * scale[:, None], axis=0)
    if HAS_WEIGHT and not WEIGHT_FIRST:
        acc = acc * w
    tl.store(
        out_ptr + t * stride_out + offs, acc.to(out_ptr.dtype.element_ty), mask=h_mask
    )


@triton.jit
def _post_expert_finalize_loop_kernel(
    x_ptr,
    w_ptr,
    tw_ptr,
    out_ptr,
    expert_ids_ptr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    stride_ids,
    idx_ptr,
    PERMUTED: tl.constexpr,
    stride_idx,
    K,
    H,
    eps,
    gain_center,
    stride_xt,
    stride_xk,
    stride_twt,
    stride_out,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + gain_center
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    x_row = x_ptr + t * stride_xt
    for k in range(K):
        local = tl.full((), True, tl.int1)
        if HAS_EXPERT_MAP:
            expert = tl.load(expert_ids_ptr + t * stride_ids + k)
            local = tl.load(expert_map_ptr + expert, mask=expert >= 0, other=-1) >= 0
        if PERMUTED:
            row = tl.load(idx_ptr + t * stride_idx + k)
            local = local & (row >= 0)
            x_k = x_ptr + row.to(tl.int64) * stride_xt
        else:
            x_k = x_row + k * stride_xk
        x = tl.load(x_k + offs, mask=mask & local, other=0.0).to(tl.float32)
        if HAS_WEIGHT and WEIGHT_FIRST:
            x = x * w
        var = tl.sum(x * x, axis=0) / H
        y = x * (1.0 / tl.sqrt(var + eps))
        if HAS_WEIGHT and not WEIGHT_FIRST:
            y = y * w
        tw = tl.load(tw_ptr + t * stride_twt + k).to(tl.float32)
        acc += y * tw
    tl.store(
        out_ptr + t * stride_out + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
    )


def fused_post_expert_finalize(
    routed: torch.Tensor,
    weight: torch.Tensor | None,
    topk_weights: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_first: bool,
    out: torch.Tensor,
    *,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    permuted_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[t] = sum_k tw[t, k] * norm(routed[t, k])`` written in ``out.dtype``.

    Routes with a negative ``expert_map`` entry or ``permuted_idx == -1`` were
    never written and are masked before the read.
    """
    if expert_map is not None:
        assert topk_ids is not None and topk_ids.stride(1) == 1
    if permuted_idx is not None:
        assert permuted_idx.dtype == torch.int32 and permuted_idx.stride(1) == 1
        T, K = permuted_idx.shape
        H = routed.shape[-1]
        stride_xt, stride_xk = routed.stride(0), 0
    else:
        T, K, H = routed.shape
        stride_xt, stride_xk = routed.stride(0), routed.stride(1)
    assert routed.stride(-1) == 1 and out.stride(1) == 1
    assert topk_weights.stride(1) == 1
    assert out.shape == (T, H) and topk_weights.shape == (T, K)
    if T == 0:
        return out
    block_h = triton.next_power_of_2(H)
    assert block_h <= 16384, "row too wide for the single-tile finalize"
    block_k = triton.next_power_of_2(K)
    tiled = block_k * block_h <= 65536
    kernel = (
        _post_expert_finalize_kernel if tiled else _post_expert_finalize_loop_kernel
    )
    tile_kwargs = {"BLOCK_K": block_k} if tiled else {}
    kernel[(T,)](
        routed,
        weight if weight is not None else routed,
        topk_weights,
        out,
        topk_ids if topk_ids is not None else topk_weights,
        expert_map if expert_map is not None else topk_weights,
        expert_map is not None,
        topk_ids.stride(0) if topk_ids is not None else 0,
        permuted_idx if permuted_idx is not None else topk_weights,
        permuted_idx is not None,
        permuted_idx.stride(0) if permuted_idx is not None else 0,
        K,
        H,
        eps,
        gain_center,
        stride_xt,
        stride_xk,
        topk_weights.stride(0),
        out.stride(0),
        BLOCK_H=block_h,
        HAS_WEIGHT=weight is not None,
        WEIGHT_FIRST=weight_first,
        num_warps=8 if (block_k * block_h if tiled else block_h) >= 4096 else 4,
        **tile_kwargs,
    )
    return out
