# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse MLA sigmoid(gate) * attn_out with MXFP4 for ``o_proj``.

Kimi-K3 MLA's output-gate epilogue is three launches today:

1. ``g_proj`` BF16 GEMM (CK ``hgemm_bf16``) — kept. Closed kernel, no MXFP4
   epilogue hook. FINDINGS: do not fold quant into this GEMM.
2. ATen ``sigmoid`` then ATen ``mul`` — replaced.
3. Standalone ``dynamic_per_group_scaled_quant`` / Triton
   ``dynamic_mxfp4_quant`` — replaced.

The real predecessors of ``o_proj`` quant are the elementwise gate ops, not
``mla_a8w8`` / ``kn_mla_reduce`` (those write a different tensor). This module
is one Triton launch that does ``y = x * sigmoid(g)`` in fp32 and writes the
MXFP4 pair ``o_proj`` already knows how to consume (KDA decode's ABI).

Scale layout is ``o_proj.input_quant_layout`` (None → Triton column-major e8m0,
``"shuffled"`` → ASM ``shuffle_scale`` + pad32 A). QuantKey stays
``kMxfp4Dynamic``. Do not import aiter at module scope (HIP init).
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.models.kimi_k3.amd.ops.kda_decode import (
    alloc_kda_mxfp4,
    mxfp4_layout_for_oproj,
    wrap_kda_mxfp4,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

MXFP4_GROUP_SIZE = 32

# Nested @triton.jit kernel, compiled on first successful aiter import.
_KERNEL: Any = None


def mx_scale_shuffle_idx(scale_n_pad: int, row: int, col: int) -> int:
    """Python form of AITER ``shuffle_scale`` / ``mx_scale_shuffle_idx``.

    Must stay identical to the Triton ``_swizzled_scale_offset`` below and to
    ``tests/models/kimi_k3/test_amd_kda_decode_mxfp4.py``.
    """
    r0, r1, r2 = row // 32, (row % 32) // 16, row % 16
    c0, c1, c2 = col // 8, (col % 8) // 4, col % 4
    return ((((r0 * (scale_n_pad // 8) + c0) * 4 + c2) * 16 + r2) * 2 + c1) * 2 + r1


def _load_mxfp4_quant_op():
    """AITER's device MXFP4 op. None if aiter is missing (CUDA CI)."""
    try:
        from aiter.ops.triton._triton_kernels.quant.quant import (
            _mxfp4_quant_op,
        )
    except ImportError:
        return None
    return _mxfp4_quant_op


def fused_mla_mxfp4_available() -> bool:
    return HAS_TRITON and _load_mxfp4_quant_op() is not None


def _get_kernel():
    """Build the producer kernel once aiter is importable.

    Delayed so ``amd/linear.py`` can import this module without initializing
    HIP. The nested jit captures ``_mxfp4_quant_op`` from this frame.
    """
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if not HAS_TRITON:
        return None
    mxfp4_quant_op = _load_mxfp4_quant_op()
    if mxfp4_quant_op is None:
        return None

    @triton.jit
    def _swizzled_scale_offset(row, col, SCALE_N: tl.constexpr):
        r0 = row // 32
        r1 = (row % 32) // 16
        r2 = row % 16
        c0 = col // 8
        c1 = (col % 8) // 4
        c2 = col % 4
        return ((((r0 * (SCALE_N // 8) + c0) * 4 + c2) * 16 + r2) * 2 + c1) * 2 + r1

    @triton.jit
    def kernel(
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
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_N: tl.constexpr,
        GROUP: tl.constexpr,
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
        x_fp4, bs_e8m0 = mxfp4_quant_op(y, BLOCK_K, BLOCK_T, GROUP)

        nbytes: tl.constexpr = BLOCK_K // 2
        ngroups: tl.constexpr = BLOCK_K // GROUP
        o_b = i_k * nbytes + tl.arange(0, nbytes)
        tl.store(
            out_fp4_ptr + o_t[:, None] * out_fp4_stride_t + o_b[None, :],
            x_fp4,
            mask=m_t[:, None],
        )

        o_s = i_k * ngroups + tl.arange(0, ngroups)
        if SWIZZLE:
            tl.store(
                out_scale_ptr
                + _swizzled_scale_offset(o_t[:, None], o_s[None, :], SCALE_N),
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

    _KERNEL = kernel
    return _KERNEL


def fused_sigmoid_gate_mxfp4_quant(
    x: torch.Tensor,
    g: torch.Tensor,
    *,
    scale_layout: str = "plain",
    block_t: int = 1,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x * sigmoid(g)`` fused with MXFP4. Returns ``(data, scale)``.

    ``scale_layout`` is ``"plain"`` (Triton ``gemm_afp4wfp4``) or
    ``"shuffled"`` (ASM ``gemm_a4w4``). Allocation matches
    ``alloc_kda_mxfp4`` so KDA decode and MLA share one GEMM ABI.
    """
    if x.dim() != 2 or g.shape != x.shape:
        raise ValueError(
            f"expected matching 2D (T, K), got x={tuple(x.shape)} g={tuple(g.shape)}"
        )
    t, k = x.shape
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"K={k} must be a multiple of {MXFP4_GROUP_SIZE} so a quant group "
            "never straddles a tile"
        )
    kernel = _get_kernel()
    if kernel is None:
        raise RuntimeError(
            "fused MLA o_proj quant requires Triton and AITER _mxfp4_quant_op"
        )

    if block_k is None:
        block_k = 1024
        while k % block_k != 0:
            block_k //= 2
    if k % block_k != 0 or block_k % MXFP4_GROUP_SIZE != 0:
        raise ValueError("block_k must divide K and be a multiple of 32")
    if block_k & (block_k - 1):
        raise ValueError("block_k must be a power of two")

    x = x.contiguous()
    g = g.contiguous()
    out_fp4, out_scale = alloc_kda_mxfp4(t, k, scale_layout, x.device)
    n_groups = k // MXFP4_GROUP_SIZE
    if scale_layout == "shuffled":
        scale_n = (n_groups + 7) // 8 * 8
        s_stride_t = s_stride_g = 0
        swizzle = True
    elif scale_layout == "plain":
        scale_n = n_groups
        s_stride_t = out_scale.stride(0)
        s_stride_g = out_scale.stride(1)
        swizzle = False
    else:
        raise ValueError(f"unknown MXFP4 layout {scale_layout!r}")

    kernel[(triton.cdiv(t, block_t), k // block_k)](
        x,
        g,
        out_fp4,
        out_scale,
        t,
        x.stride(0),
        g.stride(0),
        out_fp4.stride(0),
        s_stride_t,
        s_stride_g,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        SCALE_N=scale_n,
        GROUP=MXFP4_GROUP_SIZE,
        SWIZZLE=swizzle,
        num_warps=4,
    )
    return out_fp4, out_scale


def maybe_fused_mla_oproj_quant(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    o_proj: torch.nn.Module,
):
    """MLA sigmoid gate + MXFP4, or None to keep the BF16 expression.

    Declines when ``o_proj`` does not advertise ``kMxfp4Dynamic``, AITER's
    quant op is missing, or K is not a multiple of 32. Call from
    ``_gated_o_proj`` after ``g_proj``; wrap with the same helpers KDA decode
    uses so ``as_quantized_activation`` sees one ABI.
    """
    layout = mxfp4_layout_for_oproj(o_proj)
    if layout is None:
        return None
    if not fused_mla_mxfp4_available():
        return None
    if attn_out.dim() != 2 or attn_out.shape[-1] % MXFP4_GROUP_SIZE != 0:
        return None

    data, scale = fused_sigmoid_gate_mxfp4_quant(
        attn_out, gate, scale_layout=layout
    )
    return wrap_kda_mxfp4(
        data,
        scale,
        attn_out.shape,
        attn_out.dtype,
        o_proj,
    )
