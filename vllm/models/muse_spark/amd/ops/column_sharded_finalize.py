# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark post-expert finalize over a column shard of the routes.

When each TP rank holds every route's output on its own ``H / tp`` columns
(rather than a partial sum over all ``H`` columns), the per-route RMSNorm only
needs the sum of squares across ranks: each rank computes its ``[T, K]``
partial sums of squares (:func:`route_sum_squares`), the caller all-reduces
them, and :func:`column_sharded_finalize` normalizes, weights and sums the
routes on the local columns. Same math as ``fused_post_expert_finalize``.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _route_sum_squares_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K,
    H,
    gain_center,
    stride_xt,
    stride_xk,
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
    x = tl.load(
        x_ptr + t * stride_xt + k[:, None] * stride_xk + offs[None, :],
        mask=k_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_WEIGHT and WEIGHT_FIRST:
        w = tl.load(w_ptr + offs, mask=h_mask, other=0.0).to(tl.float32) + gain_center
        x = x * w[None, :]
    tl.store(out_ptr + t * stride_out + k, tl.sum(x * x, axis=1), mask=k_mask)


@triton.jit
def _column_sharded_finalize_kernel(
    x_ptr,
    w_ptr,
    tw_ptr,
    ss_ptr,
    out_ptr,
    K,
    H,
    hidden_size,
    eps,
    gain_center,
    stride_xt,
    stride_xk,
    stride_twt,
    stride_sst,
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
    ss = tl.load(ss_ptr + t * stride_sst + k, mask=k_mask, other=0.0)
    x = tl.load(
        x_ptr + t * stride_xt + k[:, None] * stride_xk + offs[None, :],
        mask=k_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=h_mask, other=0.0).to(tl.float32) + gain_center
        if WEIGHT_FIRST:
            x = x * w[None, :]
    scale = tl.rsqrt(ss / hidden_size + eps) * tw
    acc = tl.sum(x * scale[:, None], axis=0)
    if HAS_WEIGHT and not WEIGHT_FIRST:
        acc = acc * w
    tl.store(
        out_ptr + t * stride_out + offs, acc.to(out_ptr.dtype.element_ty), mask=h_mask
    )


def _blocks(K: int, H: int) -> tuple[int, int, int]:
    block_k = triton.next_power_of_2(K)
    block_h = triton.next_power_of_2(H)
    assert block_k * block_h <= 65536, "routes too wide for the single-tile finalize"
    return block_k, block_h, 8 if block_k * block_h >= 4096 else 4


def route_sum_squares(
    routed: torch.Tensor,
    weight: torch.Tensor | None,
    gain_center: float,
    weight_first: bool,
) -> torch.Tensor:
    """``[T, K]`` fp32 sums of squares of each route over the local columns.

    Args:
        routed: ``[T, K, H_local]`` per-route outputs on this rank's columns.
        weight: ``[H_local]`` slice of the norm weight, or ``None``.
        gain_center: Added to ``weight`` (0.0 when unused).
        weight_first: Square ``x * (w + gain_center)`` instead of ``x``.

    """
    T, K, H = routed.shape
    assert routed.stride(2) == 1
    out = torch.empty((T, K), dtype=torch.float32, device=routed.device)
    if T == 0:
        return out
    block_k, block_h, num_warps = _blocks(K, H)
    _route_sum_squares_kernel[(T,)](
        routed,
        weight if weight is not None else routed,
        out,
        K,
        H,
        gain_center,
        routed.stride(0),
        routed.stride(1),
        out.stride(0),
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        HAS_WEIGHT=weight is not None,
        WEIGHT_FIRST=weight_first,
        num_warps=num_warps,
    )
    return out


def column_sharded_finalize(
    routed: torch.Tensor,
    weight: torch.Tensor | None,
    topk_weights: torch.Tensor,
    sum_squares: torch.Tensor,
    hidden_size: int,
    eps: float,
    gain_center: float,
    weight_first: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out[t] = sum_k tw[t, k] * norm(routed[t, k])`` on the local columns.

    Args:
        routed: ``[T, K, H_local]`` per-route outputs on this rank's columns.
        weight: ``[H_local]`` slice of the norm weight, or ``None``.
        topk_weights: ``[T, K]`` router weights.
        sum_squares: ``[T, K]`` fp32 sums of squares over all ``hidden_size``
            columns (:func:`route_sum_squares` all-reduced over TP).
        hidden_size: The full norm width ``H``.
        eps: Norm epsilon.
        gain_center: Added to ``weight`` (0.0 when unused).
        weight_first: ``rms_norm(x * w)`` when True, else ``rms_norm(x) * w``.
        out: ``[T, H_local]`` destination.

    """
    T, K, H = routed.shape
    assert routed.stride(2) == 1 and out.stride(1) == 1 and topk_weights.stride(1) == 1
    assert out.shape == (T, H) and topk_weights.shape == (T, K)
    assert sum_squares.shape == (T, K) and sum_squares.dtype == torch.float32
    if T == 0:
        return out
    block_k, block_h, num_warps = _blocks(K, H)
    _column_sharded_finalize_kernel[(T,)](
        routed,
        weight if weight is not None else routed,
        topk_weights,
        sum_squares,
        out,
        K,
        H,
        hidden_size,
        eps,
        gain_center,
        routed.stride(0),
        routed.stride(1),
        topk_weights.stride(0),
        sum_squares.stride(0),
        out.stride(0),
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        HAS_WEIGHT=weight is not None,
        WEIGHT_FIRST=weight_first,
        num_warps=num_warps,
    )
    return out
