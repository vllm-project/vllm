# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton GemmaRMSNorm kernels (plain and fused residual-add).

GemmaRMSNorm's native path recomputes ``weight.float() + 1.0`` on every call
and then decomposes into a chain of small eager kernels (the fp32 weight fails
the vllm_c ``supports_args`` dtype check). These kernels fold the ``+1`` into
the weight load, accumulate the variance in fp32, and emit a single downcast,
matching ``GemmaRMSNorm.forward_native`` numerics.
"""

import torch

from vllm.triton_utils import tl, triton


def _norm_launch_config(hidden: int) -> tuple[int, int]:
    block = triton.next_power_of_2(hidden)
    if block <= 2048:
        return block, 4
    if block <= 4096:
        return block, 8
    return block, 16


@triton.jit
def _gemma_rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    x_stride_t,
    out_stride_t,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden

    x = tl.load(x_ptr + row * x_stride_t + offs, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * (w + 1.0)
    tl.store(
        out_ptr + row * out_stride_t + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemma_fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    x_stride_t,
    residual_stride_t,
    out_stride_t,
    hidden,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden

    x = tl.load(x_ptr + row * x_stride_t + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(
        residual_ptr + row * residual_stride_t + offs, mask=mask, other=0.0
    ).to(tl.float32)
    x = x + res
    # In-place residual update, mirroring torch.ops._C.fused_add_rms_norm.
    # The load above has consumed the old values, so no read-after-write
    # hazard within this program; each row is owned by a single program.
    tl.store(
        residual_ptr + row * residual_stride_t + offs,
        x.to(residual_ptr.dtype.element_ty),
        mask=mask,
    )
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * (w + 1.0)
    tl.store(
        out_ptr + row * out_stride_t + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * (weight + 1)``, one Triton kernel.

    Args:
        x: (num_tokens, hidden), row-major (``x.stride(-1) == 1``).
        weight: (hidden,) any float dtype; read and offset in fp32.
        eps: variance epsilon.

    Returns:
        A fresh tensor with x's shape, dtype, and device. ``x`` is untouched.

    """
    assert x.ndim == 2 and x.stride(-1) == 1
    assert weight.ndim == 1 and weight.is_contiguous()
    num_tokens, hidden = x.shape
    assert weight.numel() == hidden

    out = torch.empty_like(x)
    if num_tokens == 0 or hidden == 0:
        return out

    block, num_warps = _norm_launch_config(hidden)
    _gemma_rms_norm_kernel[(num_tokens,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        hidden,
        eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def gemma_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += x`` (in place) + GemmaRMSNorm of the sum, one kernel.

    Mirrors the semantics of ``torch.ops._C.fused_add_rms_norm``: the residual
    buffer is updated in place with the fp32 sum downcast to the input dtype,
    and the normalized output is a fresh tensor. ``x`` is untouched.
    """
    assert x.ndim == 2 and x.stride(-1) == 1
    assert residual.shape == x.shape and residual.dtype == x.dtype
    assert residual.stride(-1) == 1
    assert weight.ndim == 1 and weight.is_contiguous()
    num_tokens, hidden = x.shape
    assert weight.numel() == hidden

    out = torch.empty_like(x)
    if num_tokens == 0 or hidden == 0:
        return out, residual

    block, num_warps = _norm_launch_config(hidden)
    _gemma_fused_add_rms_norm_kernel[(num_tokens,)](
        x,
        residual,
        weight,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        hidden,
        eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out, residual
