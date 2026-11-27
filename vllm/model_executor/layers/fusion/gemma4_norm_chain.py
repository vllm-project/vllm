# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 decoder-layer residual / RMSNorm chain in three row-wise Triton kernels.

Every Gemma4 layer runs, between the attention output projection and the next
layer's attention, these memory-bound ops on [tokens, hidden] tensors:

  K_A  x1 = post_attn_norm(attn) + x0 ;  y = pre_ff_norm(x1)  (FP8 per-token quantized for the MLP)
  K_B  x2 = post_ff_norm(mlp) + x1
  K_C  x3 = (x2 + post_per_layer_norm(ple)) * layer_scalar ;  n = next layer input_layernorm(x3)

Unfused they are eight kernels and about 1.55 GB of HBM traffic per layer at 16k tokens; one program
per row holds the row in registers, so each kernel is a single pass. The bf16 rounding points of the
eager chain are kept (each norm output and each add is rounded to bf16); the compiled unfused graph fuses
the last add with the layer scalar without the intermediate rounding, so K_C differs from it by one bf16
ulp on some elements. The FP8 quantization follows AITER's rmsnorm_quant (fp32 norm output,
scale = amax / fp8_max, multiply by the reciprocal).
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_FP8_MAX = float(torch.finfo(current_platform.fp8_dtype()).max)


def _split(n: int) -> tuple[int, int] | None:
    """n = T0 + T1 with T0 the largest power of two below or equal to n and T1 zero or a power of two."""
    t0 = 1 << (n.bit_length() - 1)
    t1 = n - t0
    if t1 and (t1 & (t1 - 1)):
        return None
    return t0, t1


def supports_hidden_size(n: int) -> bool:
    return n >= 128 and _split(n) is not None


@triton.jit
def _load_row(ptr, o0, o1, T1: tl.constexpr):
    a0 = tl.load(ptr + o0).to(tl.float32)
    if T1 > 0:
        a1 = tl.load(ptr + o1).to(tl.float32)
    else:
        a1 = a0
    return a0, a1


@triton.jit
def _sumsq(a0, a1, T1: tl.constexpr):
    ss = tl.sum(a0 * a0, axis=0)
    if T1 > 0:
        ss += tl.sum(a1 * a1, axis=0)
    return ss


@triton.jit
def _norm_bf16(a0, a1, w_ptr, o0, o1, eps, N: tl.constexpr, T1: tl.constexpr):
    """RMSNorm * weight of a row held as fp32, rounded to bf16 and returned as fp32 (the unfused op boundary)."""
    r = tl.rsqrt(_sumsq(a0, a1, T1) * (1.0 / N) + eps)
    n0 = (a0 * r * tl.load(w_ptr + o0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if T1 > 0:
        n1 = (a1 * r * tl.load(w_ptr + o1).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        n1 = n0
    return n0, n1


@triton.jit
def _store_bf16(ptr, v0, v1, o0, o1, T1: tl.constexpr):
    tl.store(ptr + o0, v0.to(tl.bfloat16))
    if T1 > 0:
        tl.store(ptr + o1, v1.to(tl.bfloat16))


@triton.jit
def _attn_ff_kernel(
    a_ptr,
    x0_ptr,
    x1_ptr,
    y_ptr,
    ys_ptr,
    wpa_ptr,
    wpf_ptr,
    stride_a,
    stride_x0,
    stride_x1,
    stride_y,
    eps,
    FP8_MAX: tl.constexpr,
    N: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    QUANT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, T0)
    if T1 > 0:
        o1 = T0 + tl.arange(0, T1)
    else:
        o1 = o0
    a0, a1 = _load_row(a_ptr + row * stride_a, o0, o1, T1)
    n0, n1 = _norm_bf16(a0, a1, wpa_ptr, o0, o1, eps, N, T1)
    r0, r1 = _load_row(x0_ptr + row * stride_x0, o0, o1, T1)
    x0v = (n0 + r0).to(tl.bfloat16).to(tl.float32)
    if T1 > 0:
        x1v = (n1 + r1).to(tl.bfloat16).to(tl.float32)
    else:
        x1v = x0v
    _store_bf16(x1_ptr + row * stride_x1, x0v, x1v, o0, o1, T1)
    r = tl.rsqrt(_sumsq(x0v, x1v, T1) * (1.0 / N) + eps)
    y0 = x0v * r * tl.load(wpf_ptr + o0).to(tl.float32)
    if T1 > 0:
        y1 = x1v * r * tl.load(wpf_ptr + o1).to(tl.float32)
    else:
        y1 = y0
    yp = y_ptr + row * stride_y
    if QUANT:
        amax = tl.max(tl.abs(y0), axis=0)
        if T1 > 0:
            amax = tl.maximum(amax, tl.max(tl.abs(y1), axis=0))
        amax = tl.maximum(amax, 1e-10)
        s = amax * (1.0 / FP8_MAX)
        tl.store(ys_ptr + row, s)
        inv = 1.0 / s
        fp8_ty: tl.constexpr = y_ptr.dtype.element_ty
        tl.store(yp + o0, tl.minimum(tl.maximum(y0 * inv, -FP8_MAX), FP8_MAX).to(fp8_ty))
        if T1 > 0:
            tl.store(yp + o1, tl.minimum(tl.maximum(y1 * inv, -FP8_MAX), FP8_MAX).to(fp8_ty))
    else:
        _store_bf16(yp, y0, y1, o0, o1, T1)


@triton.jit
def _post_ff_kernel(
    m_ptr,
    x1_ptr,
    x2_ptr,
    w_ptr,
    stride_m,
    stride_x1,
    stride_x2,
    eps,
    N: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, T0)
    if T1 > 0:
        o1 = T0 + tl.arange(0, T1)
    else:
        o1 = o0
    a0, a1 = _load_row(m_ptr + row * stride_m, o0, o1, T1)
    n0, n1 = _norm_bf16(a0, a1, w_ptr, o0, o1, eps, N, T1)
    r0, r1 = _load_row(x1_ptr + row * stride_x1, o0, o1, T1)
    _store_bf16(x2_ptr + row * stride_x2, n0 + r0, n1 + r1, o0, o1, T1)


@triton.jit
def _ple_kernel(
    p_ptr,
    x2_ptr,
    x3_ptr,
    nx_ptr,
    w_ptr,
    sc_ptr,
    wn_ptr,
    stride_p,
    stride_x2,
    stride_x3,
    stride_nx,
    eps,
    N: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, T0)
    if T1 > 0:
        o1 = T0 + tl.arange(0, T1)
    else:
        o1 = o0
    a0, a1 = _load_row(p_ptr + row * stride_p, o0, o1, T1)
    n0, n1 = _norm_bf16(a0, a1, w_ptr, o0, o1, eps, N, T1)
    r0, r1 = _load_row(x2_ptr + row * stride_x2, o0, o1, T1)
    s = tl.load(sc_ptr).to(tl.float32)
    t0 = (n0 + r0).to(tl.bfloat16).to(tl.float32)
    z0 = (t0 * s).to(tl.bfloat16).to(tl.float32)
    if T1 > 0:
        t1 = (n1 + r1).to(tl.bfloat16).to(tl.float32)
        z1 = (t1 * s).to(tl.bfloat16).to(tl.float32)
    else:
        z1 = z0
    _store_bf16(x3_ptr + row * stride_x3, z0, z1, o0, o1, T1)
    if HAS_NEXT:
        q0, q1 = _norm_bf16(z0, z1, wn_ptr, o0, o1, eps, N, T1)
        _store_bf16(nx_ptr + row * stride_nx, q0, q1, o0, o1, T1)


def attn_ff_norm(
    attn: torch.Tensor,
    residual: torch.Tensor,
    w_post_attn: torch.Tensor,
    w_pre_ff: torch.Tensor,
    eps: float,
    quant: bool,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = attn.shape
    x1 = torch.empty_like(attn)
    if quant:
        y = torch.empty((m, n), dtype=current_platform.fp8_dtype(), device=attn.device)
        ys = torch.empty((m, 1), dtype=torch.float32, device=attn.device)
    else:
        y = torch.empty_like(attn)
        ys = torch.empty((0, 1), dtype=torch.float32, device=attn.device)
    if m:
        t0, t1 = _split(n)
        _attn_ff_kernel[(m,)](
            attn,
            residual,
            x1,
            y,
            ys,
            w_post_attn,
            w_pre_ff,
            attn.stride(0),
            residual.stride(0),
            x1.stride(0),
            y.stride(0),
            eps,
            FP8_MAX=_FP8_MAX,
            N=n,
            T0=t0,
            T1=t1,
            QUANT=quant,
            num_warps=num_warps,
        )
    return x1, y, ys


def post_ff_norm_add(
    mlp: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    num_warps: int = 1,
) -> torch.Tensor:
    m, n = mlp.shape
    out = torch.empty_like(mlp)
    if m:
        t0, t1 = _split(n)
        _post_ff_kernel[(m,)](
            mlp,
            residual,
            out,
            w,
            mlp.stride(0),
            residual.stride(0),
            out.stride(0),
            eps,
            N=n,
            T0=t0,
            T1=t1,
            num_warps=num_warps,
        )
    return out


def ple_norm_add_scale(
    proj: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    scalar: torch.Tensor,
    w_next: torch.Tensor | None,
    eps: float,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = proj.shape
    out = torch.empty_like(proj)
    nxt = torch.empty_like(proj) if w_next is not None else torch.empty((0, n), dtype=proj.dtype, device=proj.device)
    if m:
        t0, t1 = _split(n)
        _ple_kernel[(m,)](
            proj,
            residual,
            out,
            nxt,
            w,
            scalar,
            w_next if w_next is not None else w,
            proj.stride(0),
            residual.stride(0),
            out.stride(0),
            nxt.stride(0),
            eps,
            N=n,
            T0=t0,
            T1=t1,
            HAS_NEXT=w_next is not None,
            num_warps=num_warps,
        )
    return out, nxt


def _attn_ff_impl(
    attn: torch.Tensor, residual: torch.Tensor, w_post_attn: torch.Tensor, w_pre_ff: torch.Tensor, eps: float, quant: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info_once("gemma4_attn_ff_norm kernel running (first call)")
    return attn_ff_norm(attn, residual, w_post_attn, w_pre_ff, eps, quant)


def _attn_ff_fake(
    attn: torch.Tensor, residual: torch.Tensor, w_post_attn: torch.Tensor, w_pre_ff: torch.Tensor, eps: float, quant: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = attn.shape
    if quant:
        y = torch.empty((m, n), dtype=current_platform.fp8_dtype(), device=attn.device)
        ys = torch.empty((m, 1), dtype=torch.float32, device=attn.device)
    else:
        y = torch.empty_like(attn)
        ys = torch.empty((0, 1), dtype=torch.float32, device=attn.device)
    return torch.empty_like(attn), y, ys


def _post_ff_impl(mlp: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    logger.info_once("gemma4_post_ff_norm_add kernel running (first call)")
    return post_ff_norm_add(mlp, residual, w, eps)


def _post_ff_fake(mlp: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(mlp)


def _ple_impl(
    proj: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, scalar: torch.Tensor, w_next: torch.Tensor, eps: float, has_next: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info_once("gemma4_ple_norm_add_scale kernel running (first call)")
    return ple_norm_add_scale(proj, residual, w, scalar, w_next if has_next else None, eps)


def _ple_fake(
    proj: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, scalar: torch.Tensor, w_next: torch.Tensor, eps: float, has_next: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = proj.shape
    nxt = torch.empty_like(proj) if has_next else torch.empty((0, n), dtype=proj.dtype, device=proj.device)
    return torch.empty_like(proj), nxt


direct_register_custom_op(
    op_name="gemma4_attn_ff_norm",
    op_func=_attn_ff_impl,
    mutates_args=[],
    fake_impl=_attn_ff_fake,
)
direct_register_custom_op(
    op_name="gemma4_post_ff_norm_add",
    op_func=_post_ff_impl,
    mutates_args=[],
    fake_impl=_post_ff_fake,
)
direct_register_custom_op(
    op_name="gemma4_ple_norm_add_scale",
    op_func=_ple_impl,
    mutates_args=[],
    fake_impl=_ple_fake,
)
