# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's RMSNorm, fused residual-add RMSNorm and SiLU-and-mul (Triton).

Numerics of the reference implementation's CUDA norm and activation kernels,
everything in fp32 with one output rounding:

- ``rms_norm``: ``bf16(x * rsqrt(mean(x^2) + eps) * w)``.
- ``fused_add_rms_norm``: ``s = x + residual`` in fp32,
  ``residual_out = bf16(s)``, ``out = bf16(s * rsqrt(mean(s^2) + eps) * w)``
  from the unrounded ``s``.
- ``silu_and_mul``: ``bf16(g / (1 + exp(-g)) * u)`` (precise ``expf`` and
  division).

``rsqrt`` is ``rsqrt.approx.ftz.f32`` (``rsqrtf`` under fast math). The fused
Flint kernels (``qkv_fused``, the decode MoE block's norm and shared expert, and
the GEMV SiLU epilogue) follow the same rules; the routed experts use the
approximate ``ex2`` / ``rcp`` SiLU of the TRTLLM MoE (see ``moe``).
"""

import torch
from torch import nn

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _rsqrt(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [x], tl.float32, True, 1
    )


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    x_stride,
    eps,
    H: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    s = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    if HAS_RESIDUAL:
        s += tl.load(res_ptr + row * H + offs).to(tl.float32)
        tl.store(res_out_ptr + row * H + offs, s.to(tl.bfloat16))
    rs = _rsqrt(tl.sum(s * s, axis=0) / H + eps)
    tl.store(out_ptr + row * H + offs, (s * rs * w).to(tl.bfloat16))


def _launch_norm(x, residual, weight, eps, out):
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1
    assert weight.shape == (h,) and h & (h - 1) == 0
    if m > 0:
        _rms_norm_kernel[(m,)](
            x,
            x if residual is None else residual,
            weight,
            out,
            residual,
            x.stride(0),
            eps,
            H=h,
            HAS_RESIDUAL=residual is not None,
            num_warps=4 if h <= 2048 else 8,
        )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _launch_norm(x, None, weight, eps, out)
    return out


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """In place: ``residual = bf16(x + residual)``, ``x = rms_norm(x +
    residual)``; returns ``(x, residual)``."""
    assert residual.shape == x.shape and residual.is_contiguous()
    assert x.is_contiguous()
    _launch_norm(x, residual, weight, eps, x)
    return x, residual


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, x_stride, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BD + tl.arange(0, BD)
    g = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    u = tl.load(x_ptr + row * x_stride + D + offs).to(tl.float32)
    silu = tl.math.div_rn(g, 1.0 + tldevice.exp(-g))
    tl.store(out_ptr + row * D + offs, (silu * u).to(tl.bfloat16))


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[:, :D]) * x[:, D:]`` for ``x`` [M, 2D]."""
    m, d2 = x.shape
    d = d2 // 2
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1
    bd = min(d, 1024)
    assert d % bd == 0
    out = torch.empty(m, d, dtype=x.dtype, device=x.device)
    if m > 0:
        _silu_and_mul_kernel[(m, d // bd)](x, out, x.stride(0), D=d, BD=bd, num_warps=4)
    return out


class RMSNorm(nn.Module):
    """Holds a norm ``weight``; ``rms_norm(x)``, or with a residual
    ``fused_add_rms_norm(x, residual)``."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.variance_epsilon)
        return fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
