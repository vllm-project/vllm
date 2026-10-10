# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/mx.py
"""FP8 / MX arithmetic the mono kernels share: the E4M3 range, E8M0 scale
codes, the byte of a scale in aiter's ``shuffle_scale`` layout, an FP4 weight
tile's operand loads and the scaled MFMA that takes them.

An E8M0 code is a biased exponent; ``pow2(code)`` is the scale it stands for.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import CM_NT

FP8_MAX = 448.0
# the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8, FP4 = 0, 4
UNIT_SCALE = 127  # the E8M0 code of 1.0


def clamp_fp8(v):
    """``v`` clamped to the E4M3 range."""
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def code_ceil(v):
    """Biased exponent of the smallest power of two >= ``v`` (v >= 0, finite),
    clamped to [1, 254]: the codes a scale may take."""
    bits = v.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))


def pow2(code):
    """The fp32 power of two a biased exponent stands for."""
    return (code << 23).bitcast(fx.Float32)


def rcp_pow2(code):
    """1 / ``pow2(code)``, exact (code in [1, 253]): x times it is x / scale
    bit for bit, a multiply where the division is a dozen instructions."""
    return ((254 - code) << 23).bitcast(fx.Float32)


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
