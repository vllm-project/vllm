# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Kimi-K3 attention epilogue + MXFP4 activation quant.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

try:
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
except ImportError:  # CUDA CI / AITER not installed
    _mxfp4_quant_op = None

MXFP4_GROUP_SIZE = 32
# Triton only lets a @jit body reference globals declared as constexpr.
_GROUP = tl.constexpr(MXFP4_GROUP_SIZE)

PLAIN = "plain"
SWIZZLED = "swizzled"


@triton.jit
def _swizzled_scale_offset(row, col, SCALE_N: tl.constexpr):
    """Flat byte offset of scale element (row, col) in AITER's swizzled buffer.

    Scalar equivalent of `shuffle_scale`'s permute. `SCALE_N` is the *padded*
    column count (multiple of 8); the destination buffer is row-major
    (pad256(M), SCALE_N), so the flattened permuted index is the byte offset.
    """
    r0 = row // 32
    r1 = (row % 32) // 16
    r2 = row % 16
    c0 = col // 8
    c1 = (col % 8) // 4
    c2 = col % 4
    return ((((r0 * (SCALE_N // 8) + c0) * 4 + c2) * 16 + r2) * 2 + c1) * 2 + r1


def _alloc_outputs(T: int, K: int, scale_layout: str, device: torch.device):
    """Allocate (fp4, scale) exactly as the quantizer this layout replaces does.

    Returns the raw uint8 buffers plus the scale strides the kernels index
    with. `SWIZZLED` pads rows to a multiple of 32 because AITER's ASM
    `gemm_a4w4` tiles M in 32s and overreads the activation past M; the pad
    keeps that overread inside this allocation. (Eager execution tolerates the
    overread because adjacent memory is mapped; a CUDA-graph private pool does
    not, which is why an unpadded buffer faults only under the real model.)
    `PLAIN` needs no pad -- the Triton GEMM masks -- so it matches
    `dynamic_mxfp4_quant`'s exact allocation.
    """
    n_groups = K // MXFP4_GROUP_SIZE
    if scale_layout == SWIZZLED:
        m_pad = triton.cdiv(T, MXFP4_GROUP_SIZE) * MXFP4_GROUP_SIZE
        out_fp4 = torch.empty((m_pad, K // 2), dtype=torch.uint8, device=device)
        scale_n = triton.cdiv(n_groups, 8) * 8
        out_scale = torch.empty(
            (triton.cdiv(T, 256) * 256, scale_n), dtype=torch.uint8, device=device
        )
        # Swizzled stores compute their own offsets from SCALE_N.
        return out_fp4, out_scale, scale_n, 0, 0
    if scale_layout != PLAIN:
        raise ValueError(f"unknown scale_layout {scale_layout!r}")
    out_fp4 = torch.empty((T, K // 2), dtype=torch.uint8, device=device)
    # Column-major, as dynamic_mxfp4_quant allocates it.
    out_scale = torch.empty((n_groups, T), dtype=torch.uint8, device=device).T
    return out_fp4, out_scale, n_groups, out_scale.stride(0), out_scale.stride(1)


def _view_outputs(out_fp4, out_scale, T: int, scale_layout: str):
    if scale_layout == SWIZZLED:
        # per_1x32_f4_quant_hip's dtypes; gemm_a4w4 re-views its weights to match.
        return (
            out_fp4[:T].view(torch.float4_e2m1fn_x2),
            out_scale.view(torch.float8_e8m0fnu),
        )
    # dynamic_mxfp4_quant returns plain uint8.
    return out_fp4, out_scale


# --------------------------------------------------------------------------- #
# KDA: gated RMSNorm (per head) + MXFP4 quant
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_gated_rmsnorm_mxfp4_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    out_fp4_ptr,
    out_scale_ptr,
    eps,
    T,
    x_stride_t,
    g_stride_t,
    g_stride_h,
    out_fp4_stride_t,
    scale_stride_t,
    scale_stride_g,
    H,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SCALE_N: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)

    o_h = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
    o_d = tl.arange(0, D)
    m_h = o_h < H

    x = tl.load(
        x_ptr + i_t * x_stride_t + o_h[:, None] * D + o_d[None, :],
        mask=m_h[:, None],
        other=0.0,
    ).to(tl.float32)

    # RMSNorm is per (token, head) over head_dim -- the reduction axis is
    # entirely inside this tile.
    var = tl.sum(x * x, axis=1) / D
    y = x * (1.0 / tl.sqrt(var + eps))[:, None]
    if HAS_WEIGHT:
        y = y * tl.load(w_ptr + o_d).to(tl.float32)[None, :]

    g = tl.load(
        g_ptr + i_t * g_stride_t + o_h[:, None] * g_stride_h + o_d[None, :],
        mask=m_h[:, None],
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "sigmoid":
        y = y * tl.sigmoid(g)
    elif ACTIVATION == "swish" or ACTIVATION == "silu":
        y = y * g * tl.sigmoid(g)

    # Groups of 32 never straddle heads (D % 32 == 0), so the head-major
    # flattening below leaves every group intact.
    y = tl.reshape(y, (1, BLOCK_H * D))
    x_fp4, bs_e8m0 = _mxfp4_quant_op(y, BLOCK_H * D, 1, _GROUP)

    NBYTES: tl.constexpr = BLOCK_H * D // 2
    NGROUPS: tl.constexpr = BLOCK_H * D // 32

    o_b = tl.arange(0, NBYTES)
    byte_base = i_h * NBYTES
    tl.store(
        out_fp4_ptr + i_t * out_fp4_stride_t + byte_base + o_b,
        tl.reshape(x_fp4, (NBYTES,)),
        mask=(byte_base + o_b) < (H * D // 2),
    )

    o_s = i_h * NGROUPS + tl.arange(0, NGROUPS)
    m_s = o_s < (H * D // 32)
    bs = tl.reshape(bs_e8m0, (NGROUPS,))
    if SWIZZLE:
        tl.store(
            out_scale_ptr + _swizzled_scale_offset(i_t, o_s, SCALE_N), bs, mask=m_s
        )
    else:
        tl.store(
            out_scale_ptr + i_t * scale_stride_t + o_s * scale_stride_g, bs, mask=m_s
        )


def fused_gated_rmsnorm_mxfp4_quant(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    activation: str = "sigmoid",
    scale_layout: str = PLAIN,
    block_h: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA epilogue: per-head gated RMSNorm fused with MXFP4 quant.

    Replaces `FusedRMSNormGated(head_dim)(x, g)` followed by the o_proj
    activation quant.

    Args:
        x: `(T, H, D)` or `(1, T, H, D)` core attention output.
        g: gate, broadcastable to x's layout; `(T, H, D)` or `(T, H*D)`.
        weight: `(D,)` RMSNorm gain, or None.
        scale_layout: `"plain"` (Triton `gemm_afp4wfp4`) or `"swizzled"`
            (ASM `gemm_a4w4`).
        block_h: heads per program. Defaults to all heads (one program per
            token); smaller values trade redundant gate loads for occupancy.

    Returns:
        `(x_fp4, x_scale)` in exactly the layout the replaced quantizer
        returns.
    """
    if x.dim() == 4:
        assert x.shape[0] == 1, f"expected leading batch of 1, got {x.shape}"
        x = x[0]
    assert x.dim() == 3, f"expected (T, H, D), got {tuple(x.shape)}"
    T, H, D = x.shape
    assert D % MXFP4_GROUP_SIZE == 0, (
        f"head_dim={D} must be a multiple of {MXFP4_GROUP_SIZE}; otherwise a "
        "quant group would straddle two heads and could not be reduced "
        "inside one program"
    )
    x = x.contiguous()

    g = g.reshape(T, H, D)
    assert g.stride(2) == 1, "gate must be contiguous along head_dim"

    K = H * D

    if block_h is None:
        block_h = triton.next_power_of_2(H)
    assert block_h & (block_h - 1) == 0, "block_h must be a power of two"

    out_fp4, out_scale, scale_n, s_stride_t, s_stride_g = _alloc_outputs(
        T, K, scale_layout, x.device
    )

    if _mxfp4_quant_op is None:
        raise RuntimeError("fused KDA o_proj quant requires AITER's _mxfp4_quant_op")
    _fused_gated_rmsnorm_mxfp4_kernel[(T, triton.cdiv(H, block_h))](
        x,
        g,
        weight,
        out_fp4,
        out_scale,
        eps,
        T,
        x.stride(0),
        g.stride(0),
        g.stride(1),
        out_fp4.stride(0),
        s_stride_t,
        s_stride_g,
        H,
        D=D,
        BLOCK_H=block_h,
        SCALE_N=scale_n,
        HAS_WEIGHT=weight is not None,
        ACTIVATION=activation,
        SWIZZLE=scale_layout == SWIZZLED,
        num_warps=4,
    )
    return _view_outputs(out_fp4, out_scale, T, scale_layout)


# --------------------------------------------------------------------------- #
# MLA / DSpark: sigmoid output gate + MXFP4 quant
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_sigmoid_gate_mxfp4_kernel(
    x_ptr,
    g_ptr,
    out_fp4_ptr,
    out_scale_ptr,
    T,
    x_stride_t,
    g_stride_t,
    out_fp4_stride_t,
    scale_stride_t,
    scale_stride_g,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_N: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_k = tl.program_id(1)

    o_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
    o_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_t = o_t < T

    x = tl.load(
        x_ptr + o_t[:, None] * x_stride_t + o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)
    g = tl.load(
        g_ptr + o_t[:, None] * g_stride_t + o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)

    y = x * tl.sigmoid(g)

    x_fp4, bs_e8m0 = _mxfp4_quant_op(y, BLOCK_K, BLOCK_T, _GROUP)

    NBYTES: tl.constexpr = BLOCK_K // 2
    NGROUPS: tl.constexpr = BLOCK_K // 32

    o_b = i_k * NBYTES + tl.arange(0, NBYTES)
    tl.store(
        out_fp4_ptr + o_t[:, None] * out_fp4_stride_t + o_b[None, :],
        x_fp4,
        mask=m_t[:, None],
    )

    o_s = i_k * NGROUPS + tl.arange(0, NGROUPS)
    if SWIZZLE:
        tl.store(
            out_scale_ptr + _swizzled_scale_offset(o_t[:, None], o_s[None, :], SCALE_N),
            bs_e8m0,
            mask=m_t[:, None],
        )
    else:
        tl.store(
            out_scale_ptr
            + o_t[:, None] * scale_stride_t
            + o_s[None, :] * scale_stride_g,
            bs_e8m0,
            mask=m_t[:, None],
        )


def fused_sigmoid_gate_mxfp4_quant(
    x: torch.Tensor,
    g: torch.Tensor,
    *,
    scale_layout: str = PLAIN,
    block_t: int = 1,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MLA/DSpark epilogue: `x * sigmoid(g)` fused with MXFP4 quant.

    Replaces the ATen `sigmoid` + `mul` pair in
    `MultiHeadLatentAttentionWrapper.forward` followed by the o_proj activation
    quant.

    Args:
        x: `(T, K)` attention output.
        g: `(T, K)` pre-sigmoid gate (the raw `g_proj` output).
        scale_layout: `"plain"` (Triton `gemm_afp4wfp4`) or `"swizzled"`
            (ASM `gemm_a4w4`).
        block_t / block_k: tile shape; `block_k` defaults to the whole row.

    Returns:
        `(x_fp4, x_scale)` in exactly the layout the replaced quantizer
        returns.
    """
    assert x.dim() == 2 and g.shape == x.shape, (
        f"expected matching 2D (T, K), got x={tuple(x.shape)} g={tuple(g.shape)}"
    )
    T, K = x.shape
    assert K % MXFP4_GROUP_SIZE == 0

    if block_k is None:
        # Triton needs a power-of-two tile, and splitting K must not split a
        # group, so take the largest power of two that divides K (capped so a
        # tile still fits comfortably in registers).
        block_k = 1024
        while K % block_k != 0:
            block_k //= 2
    assert K % block_k == 0, "block_k must divide K so groups stay intact"
    assert block_k % MXFP4_GROUP_SIZE == 0
    assert block_k & (block_k - 1) == 0, "block_k must be a power of two"

    out_fp4, out_scale, scale_n, s_stride_t, s_stride_g = _alloc_outputs(
        T, K, scale_layout, x.device
    )

    if _mxfp4_quant_op is None:
        raise RuntimeError("fused MLA o_proj quant requires AITER's _mxfp4_quant_op")
    _fused_sigmoid_gate_mxfp4_kernel[(triton.cdiv(T, block_t), K // block_k)](
        x,
        g,
        out_fp4,
        out_scale,
        T,
        x.stride(0),
        g.stride(0),
        out_fp4.stride(0),
        s_stride_t,
        s_stride_g,
        K=K,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        SCALE_N=scale_n,
        SWIZZLE=scale_layout == SWIZZLED,
        num_warps=4,
    )
    return _view_outputs(out_fp4, out_scale, T, scale_layout)


# --------------------------------------------------------------------------- #
# Producers: wrap the pair as QuantizedActivation for o_proj
# --------------------------------------------------------------------------- #


def _o_proj_accepts_mxfp4(o_proj: torch.nn.Module) -> bool:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    return getattr(o_proj, "input_quant_key", None) == kMxfp4Dynamic


def kda_will_fuse_oproj_quant(o_proj: torch.nn.Module) -> bool:
    """Whether KDA should skip its own gated RMSNorm and let the fused producer run.

    Must match ``maybe_fused_kda_oproj_quant``'s decline predicate. The AMD
    decode path otherwise applies that epilogue inside ``fused_kda_decode``,
    which left the previous producer with nothing to fuse on 3 of every 4
    layers.
    """
    return _o_proj_accepts_mxfp4(o_proj) and _mxfp4_quant_op is not None


def maybe_fused_kda_oproj_quant(
    core_attn_out: torch.Tensor,
    g2: torch.Tensor,
    o_norm: torch.nn.Module,
    o_proj: torch.nn.Module,
):
    """KDA gated RMSNorm + MXFP4, or None to keep the unfused epilogue.
    """
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    if not kda_will_fuse_oproj_quant(o_proj):
        return None

    data, scale = fused_gated_rmsnorm_mxfp4_quant(
        core_attn_out,
        g2,
        o_norm.weight,
        eps=o_norm.eps,
        activation=o_norm.activation,
        scale_layout=PLAIN,
    )
    # core_attn_out is (1, T, H, D) or (T, H, D); flatten matches rearrange
    # "1 n h d -> n (h d)" on the unfused path.
    x = core_attn_out[0] if core_attn_out.dim() == 4 else core_attn_out
    t, h, d = x.shape
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=core_attn_out.dtype,
        orig_shape=torch.Size([t, h * d]),
        quant_key=kMxfp4Dynamic,
    )


def maybe_fused_mla_oproj_quant(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    o_proj: torch.nn.Module,
):
    """MLA/DSpark sigmoid gate + MXFP4, or None to keep the BF16 expression."""
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    if not _o_proj_accepts_mxfp4(o_proj):
        return None
    if _mxfp4_quant_op is None:
        return None

    data, scale = fused_sigmoid_gate_mxfp4_quant(attn_out, gate, scale_layout=PLAIN)
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=attn_out.dtype,
        orig_shape=attn_out.shape,
        quant_key=kMxfp4Dynamic,
    )
