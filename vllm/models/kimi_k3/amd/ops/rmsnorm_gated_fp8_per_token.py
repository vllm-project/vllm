# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused gated RMSNorm + per-token FP8 for Kimi-K3 KDA ``o_proj``.

Port of ATOM ``_rmsnorm_gated_fp8_per_token_kernel`` (ATOM#1752). RMSNorm
is per-head over ``head_dim``; the FP8 amax is **per token** across heads.
That is the PTPC activation layout AITER ``gemm_a8w8_bpreshuffle`` wants:
``y [tokens, heads*H] fp8``, ``scale [tokens, 1] float32``.

Do not use this for group-128 FP8 (vLLM PR 40710). That ABI is a
different GEMM.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_gated_fp8_per_token_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        y_ptr,
        s_ptr,
        H,
        eps,
        fp8_max,
        stride_xm,
        stride_xh,
        stride_g_outer,
        stride_g_head,
        stride_ym,
        HEADS: tl.constexpr,
        HEADS_POW2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tok = tl.program_id(0)
        head_ids = tl.arange(0, HEADS_POW2)
        cols = tl.arange(0, BLOCK)
        mask = (head_ids[:, None] < HEADS) & (cols[None, :] < H)
        # Padding heads are masked, but a raw offset of head_ids*stride can
        # still form an OOB pointer on ROCm/Triton. Clamp the address index.
        h_safe = tl.where(head_ids < HEADS, head_ids, 0)
        x_off = tok * stride_xm + h_safe[:, None] * stride_xh + cols[None, :]
        x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
        g_off = tok * stride_g_outer + h_safe[:, None] * stride_g_head + cols[None, :]
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        normed = (x * rstd[:, None] * w[None, :]) * tl.sigmoid(gate)
        amax = tl.max(tl.abs(normed))
        scale = amax / fp8_max
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = normed * inv
        q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
        y_off = tok * stride_ym + h_safe[:, None] * H + cols[None, :]
        tl.store(y_ptr + y_off, q.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(s_ptr + tok, scale)


def _rmsnorm_gated_torch(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    xn = x_f * torch.rsqrt(var + eps)
    return (xn.to(dtype) * weight.to(dtype)) * torch.sigmoid(gate)


def rmsnorm_gated_fp8_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated RMSNorm over last dim, then per-token FP8.

    Args:
        x: ``[tokens, heads, head_dim]`` (bf16)
        weight: ``[head_dim]``
        gate: ``[tokens, heads, head_dim]`` or a strided 3D view
        eps: RMSNorm epsilon
        quant_dtype: platform FP8 dtype (``float8_e4m3fn`` on gfx950)

    Returns:
        ``(y [tokens, heads*head_dim] fp8, scale [tokens, 1] float32)``
    """
    if x.ndim != 3:
        raise ValueError(f"expected [t, heads, head_dim], got {tuple(x.shape)}")
    t, heads, h = x.shape
    fp8_max = float(torch.finfo(quant_dtype).max)
    if not _HAS_TRITON or t == 0 or h > 8192:
        normed = _rmsnorm_gated_torch(x, weight, gate, eps).reshape(t, heads * h)
        amax = normed.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
        scale = amax / fp8_max
        q = (normed.float() / scale).clamp(-fp8_max, fp8_max).to(quant_dtype)
        return q, scale
    x = x.contiguous()
    out = torch.empty((t, heads * h), dtype=quant_dtype, device=x.device)
    scale = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    if gate.ndim == 3:
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        stride_g_outer, stride_g_head = gate.stride(0), 0
    block = triton.next_power_of_2(h)
    _rmsnorm_gated_fp8_per_token_kernel[(t,)](
        x,
        weight,
        gate,
        out,
        scale,
        h,
        float(eps),
        fp8_max,
        x.stride(0),
        x.stride(1),
        stride_g_outer,
        stride_g_head,
        out.stride(0),
        HEADS=heads,
        HEADS_POW2=triton.next_power_of_2(heads),
        BLOCK=block,
    )
    return out, scale


def per_token_fp8_quant(
    x: torch.Tensor, quant_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token amax FP8 of an already-normed activation ``[t, k]``.

    Used when HIP ``fused_kda_decode`` already applied gated RMSNorm.
    Re-running :func:`rmsnorm_gated_fp8_per_token` would double-norm.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [tokens, k], got {tuple(x.shape)}")
    fp8_max = float(torch.finfo(quant_dtype).max)
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / fp8_max
    q = (x.float() / scale).clamp(-fp8_max, fp8_max).to(quant_dtype)
    return q, scale
