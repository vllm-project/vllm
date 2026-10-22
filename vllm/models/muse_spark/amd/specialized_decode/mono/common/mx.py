# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/mx.py
"""FP8 / MX arithmetic the mono kernels share: the E4M3 range, E8M0 scale
codes, the byte of a scale in aiter's ``shuffle_scale`` layout, an FP4 weight
tile's operand loads and the scaled MFMA that takes them.

An E8M0 code is a biased exponent; ``pow2(code)`` is the scale it stands for.

Muse Spark adds the MXFP4 activation quant (``e8m0_roundup``, ``pk_fp4_*``,
moved here from K3's ``stages/moe.py``) and the scale reads of the
``e8m0_shuffle``d w13 / w2 scales (``w13_scale_word``, ``scale_byte``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import CM_NT

# the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8, FP4 = 0, 4
UNIT_SCALE = 127  # the E8M0 code of 1.0


def pow2(code):
    """The fp32 power of two a biased exponent stands for."""
    return (code << 23).bitcast(fx.Float32)


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``shuffle_scale`` layout:
    ``view(rows/32, 2, 16, cols/8, 2, 4).permute(0, 3, 5, 2, 4, 1)``."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


def fp4_tile_load(r_w, base, rg, k, st, at):
    """A lane's 16 B of an FP4 weight's (16, 16)-shuffled 16-row group ``rg``
    (K = ``k``) at 128-k step ``st``: dword ``base`` + the group's 2 k + 256 a
    step + ``at`` (``lane * 4``: 32 fp4, K block lane / 16 of row lane % 16).
    Nontemporal: a weight a step reads once."""
    return fx.Vector(
        bo.buffer_load(
            r_w,
            base + rg * (2 * k) + st * 256 + at,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_NT,
        )
    )


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8, sa_byte=0, sb_byte=0):
    """One 16x16x128 scaled MFMA (gfx950 f8f6f4): A ``a`` (``a_fmt``) with E8M0
    ``sa`` (its byte ``sa_byte``), B ``b`` (``b_fmt``) with E8M0 ``sb`` (its
    byte ``sb_byte``)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, sa_byte, sa, sb_byte, sb]
        )
    )


def e8m0_roundup(amax):
    """The ``_e8m0_roundup(amax, 6)`` of aiter: ceil code of amax * f32(1/6),
    at most 254 (the RoundUp rule of aiter's HIP ``fused_mx_quant_moe_sort``;
    callers floor ``amax`` at ``spec.AMAX_FLOOR`` first, as that kernel does)."""
    wi = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    b = fx.Int32(fx.Uint32(wi + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(b, fx.Int32(254))


def pk_fp4_f32(old, a, b, scale, j):
    """Two f32 / scale -> E2M1 nibbles into byte j of ``old`` (RNE, saturating)."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_fp4_bf16(old, word, scale, j):
    """A packed bf16 pair word / scale -> E2M1 nibbles into byte j of ``old``."""
    v = fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(v),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def w13_scale_word(e, rg, st, lane, rows=1024, cols=128):
    """The dword of an ``e8m0_shuffle``d w13 scale ([experts * rows, cols]
    codes) holding expert ``e``'s (row rg 16 + lane % 16, K block 4 st +
    lane / 16): steps st, st ^ 1 and row groups rg, rg ^ 1 share it, the code
    at byte (st % 2) 2 + rg % 2. K3's ``w13_scale_word`` at Muse's shapes."""
    return scale_index(e * rows + rg * 16 + lane % 16, 4 * st + lane // 16, cols) // 4


def scale_byte(r_s, row, col, cols):
    """The e8m0 code (row, col) of an ``e8m0_shuffle``d scale with ``cols``
    codes a row, from the buffer resource ``r_s`` (K3's ``w2_scale``)."""
    idx = scale_index(row, col, cols)
    w = fx.Int32(bo.buffer_load(r_s, idx // 4, vec_width=1, dtype=T.i32))
    return (w >> ((idx % 4) * 8)) & 0xFF
