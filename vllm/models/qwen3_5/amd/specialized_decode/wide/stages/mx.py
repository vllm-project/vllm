# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 activations on the scaled fp4 matrix instruction (gfx950).

A 32-value group is sixteen bytes of e2m1 codes (low nibble first) and one
E8M0 scale code. The scale follows AITER's activation quantization (the stock
MoE's, ``_mxfp4_quant_op``): ``amax`` rounded on its top mantissa bits
(``+ 0x200000``, the mantissa then dropped), the code that exponent minus 2,
so values past 6 x scale saturate. ``mfma_f4`` (``common.ops``) multiplies two
16 x 128 operands of such groups: a lane holds row (token) ``lane % 16`` and
the 32 values of K block ``lane // 16``, the block's scale code in the low byte
of a register.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T

GROUP = 32  # values a scale


def i32(v):
    return fx.Int32(v)


def scale_code(amax):
    """The E8M0 code of a group whose largest magnitude is ``amax`` (f32)."""
    bits = fx.Float32(amax).bitcast(fx.Int32)
    e = ((bits + i32(0x200000)) >> i32(23)) & i32(0xFF)
    return fx.min(fx.max(e - i32(2), i32(0)), i32(254))


def scale_value(code):
    """``2 ** (code - 127)`` as f32 (code 0 gives 0, as a zero group's)."""
    return (fx.Int32(code) << i32(23)).bitcast(fx.Float32)


def _pack_f32(word, a, b, scale, byte: int):
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32, word.ir_value(), a.ir_value(), b.ir_value(), scale.ir_value(), byte
        )
    )


def _pack_bf16(word, pair, scale, byte: int):
    v = fx.Vector.from_elements([pair], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32, word.ir_value(), v.ir_value(), scale.ir_value(), byte
        )
    )


def quant_bf16_group(pairs, code):
    """Sixteen packed bf16 pairs (Int32) of a 32-value group as four Int32 words
    of e2m1 codes over ``scale_value(code)``."""
    scale = scale_value(code)
    words = []
    for w in range(4):
        word = i32(0)
        for j in range(4):
            word = _pack_bf16(word, pairs[4 * w + j], scale, j)
        words.append(word)
    return words


def quant_f32_word(values, scale):
    """8 f32 values as one Int32 word of e2m1 codes over ``scale``
    (``scale_value`` of the group's code)."""
    lo = _pack_f32(
        _pack_f32(i32(0), values[0], values[1], scale, 0),
        values[2],
        values[3],
        scale,
        1,
    )
    hi = _pack_f32(
        _pack_f32(i32(0), values[4], values[5], scale, 0),
        values[6],
        values[7],
        scale,
        1,
    )
    return lo | (hi << i32(16))


def quant_f32_group(values, code):
    """32 f32 values as four Int32 words of e2m1 codes."""
    scale = scale_value(code)
    return [quant_f32_word(values[8 * w : 8 * w + 8], scale) for w in range(4)]


def amax(values):
    m = fx.Float32(0.0)
    for v in values:
        a = fmath.absf(v)
        m = (a > m).select(a, m)
    return m
