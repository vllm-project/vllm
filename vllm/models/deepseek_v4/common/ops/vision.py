# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _square_kernel(X, Squared, n_elements, BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx, idx < n_elements, other=0).to(tl.float32)
    tl.store(Squared + idx, x * x, idx < n_elements)


@triton.jit
def _norm_scale_kernel(
    X,
    Weight,
    Variance,
    Out,
    n_elements,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(X + idx, mask, other=0).to(tl.float32)
    w = tl.load(Weight + idx % DIM, mask, other=0).to(tl.float32)
    variance = tl.load(Variance + idx // DIM, mask, other=0)
    out = w * (x * tl.rsqrt(variance + EPS))
    tl.store(Out + idx, out, mask)


def vision_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not (
        current_platform.is_cuda()
        and x.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.dtype != torch.float64
        and x.numel()
    ):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return (weight * x).to(dtype)

    squared = torch.empty_like(x, dtype=torch.float32)
    grid = (triton.cdiv(x.numel(), 1024),)
    _square_kernel[grid](x, squared, x.numel(), BLOCK=1024)
    # Preserve the reference reduction order: even small rounding differences
    # can accumulate across the vision tower's BF16 residual blocks.
    variance = squared.mean(-1, keepdim=True)
    out = torch.empty_like(x)
    _norm_scale_kernel[grid](
        x,
        weight,
        variance,
        out,
        x.numel(),
        x.shape[-1],
        eps,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    return out


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


@triton.jit(do_not_specialize=["n_tokens"])
def _qk_rotary_kernel(
    QKV,
    Cos,
    Sin,
    Q,
    K,
    n_tokens,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    half: tl.constexpr = DIM // 2
    col = idx % half
    head = idx // half % HEADS
    row = idx // (half * HEADS)
    mask = row < n_tokens
    cos = tl.load(Cos + row * half + col, mask, other=0)
    sin = tl.load(Sin + row * half + col, mask, other=0)
    src = row * STRIDE + head * DIM + col
    q1 = tl.load(QKV + src, mask, other=0).to(tl.float32)
    q2 = tl.load(QKV + src + half, mask, other=0).to(tl.float32)
    k1 = tl.load(QKV + src + HEADS * DIM, mask, other=0).to(tl.float32)
    k2 = tl.load(QKV + src + HEADS * DIM + half, mask, other=0).to(tl.float32)
    dst = row * HEADS * DIM + head * DIM + col
    tl.store(Q + dst, q1 * cos - q2 * sin, mask)
    tl.store(Q + dst + half, q2 * cos + q1 * sin, mask)
    tl.store(K + dst, k1 * cos - k2 * sin, mask)
    tl.store(K + dst + half, k2 * cos + k1 * sin, mask)


def vision_qk_rotary(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = qkv.shape[0]
    if not (current_platform.is_cuda() and qkv.is_cuda):
        q, k, _ = (t.view(n, num_heads, head_dim) for t in qkv.chunk(3, -1))
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

    q, k = torch.empty(
        (2, n, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device
    ).unbind()
    _qk_rotary_kernel[(triton.cdiv(n * num_heads * head_dim // 2, 512),)](
        qkv,
        cos,
        sin,
        q,
        k,
        n,
        num_heads,
        head_dim,
        qkv.stride(0),
        BLOCK=512,
        enable_fp_fusion=False,
    )
    return q, k


@triton.jit(do_not_specialize=["n_rows"])
def _spatial_merge_kernel(
    X,
    Indices,
    Mask,
    Out,
    n_rows,
    DIM: tl.constexpr,
    R2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = idx // (DIM * R2)
    channel = idx // R2 % DIM
    patch = idx % R2
    valid = row < n_rows
    src = tl.load(Indices + row * R2 + patch, valid, other=0)
    scale = tl.load(Mask + row * R2 + patch, valid, other=0).to(tl.float32)
    x = tl.load(X + src * DIM + channel, valid, other=0).to(tl.float32)
    tl.store(Out + idx, x * scale, valid)


def vision_spatial_merge(
    x: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    rows, r2 = indices.shape
    if not (current_platform.is_cuda() and x.is_cuda and x.is_contiguous()):
        gathered = (x[indices] * mask).view(rows, r2, -1)
        return gathered.permute(0, 2, 1).reshape(rows, -1)

    dim = x.shape[-1]
    out = torch.empty((rows, dim * r2), device=x.device, dtype=x.dtype)
    _spatial_merge_kernel[(triton.cdiv(out.numel(), 1024),)](
        x,
        indices,
        mask,
        out,
        rows,
        dim,
        r2,
        BLOCK=1024,
    )
    return out
