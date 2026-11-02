# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 (MI300X, MI325X) replacements for the gfx950 instructions the mono
kernels use.

FP8 on gfx942 is FNUZ: the same bit patterns as OCP e4m3 stand for half the
OCP value, and 0x80 is NaN instead of -0. The kernels quantize with an FP8
maximum of 224 there (``arch.GFX942``), which gives the same bit patterns as
OCP quantization with 448 and a scale code one higher. A product of two
bytes then needs 2^(code_a + code_b - 254) with the FNUZ codes, which is what
``mx_factor`` returns.

gfx942 has no scaled MFMA. A GEMV runs one ``v_mfma_f32_16x16x32_fp8_fp8``
per 32-wide K block (one scale block) with a zero accumulator and adds the
result times the block's scale factor (``mfma_fp8_scaled``). A lane holds 8
bytes of each block: K 8 (lane / 16) .. + 8 of the block, for its row (A) or
column (B).
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# 0x80808080 and 0x7F7F7F7F as signed 32-bit constants.
SIGN_BYTES = -0x7F7F7F80
MAG_BYTES = 0x7F7F7F7F


def pair_i64(lo, hi):
    """Two dwords as the 64-bit operand of an FP8 MFMA, lo in the low half."""
    return fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(
        fx.Int64
    )[0]


def mfma_fp8(a, b, c):
    """One v_mfma_f32_16x16x32_fp8_fp8: A and B are a lane's 8 FNUZ bytes as
    an i64 (``pair_i64``), C a Vector of 4 f32."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0])
    )


def mx_factor(code_sum):
    """2^(code_sum - 254) as f32, for code_sum the sum of two E8M0 codes. The
    f32 bits of 2^e are (e + 127) << 23, so this is (code_sum - 127) << 23.
    The sum must stay in [128, 381]: the factor is then a normal f32."""
    return ((code_sum - 127) << 23).bitcast(fx.Float32)


def acc_scaled(acc, d, f):
    """``acc`` + ``d`` * ``f`` for the 4 accumulator values of a lane."""
    return fx.Vector.from_elements(
        [fx.Float32(fmath.fma(d[i], f, acc[i])) for i in range(4)], fx.Float32
    )


def pin_acc(acc):
    """The 4 accumulator values of a lane through opaque moves with side
    effects. An unrolled GEMV loop pins its accumulators after each round.
    The round's FMAs must then finish before the moves, and the next round's
    LDS reads (and the MFMAs that need them) cannot move above them. Without
    it LLVM runs every MFMA of the loop first and the scaling FMAs at the end,
    which needs a register for each MFMA result and spills them all."""
    return fx.Vector.from_elements(
        [
            fx.Float32(
                llvm.inline_asm(
                    T.f32,
                    [fx.Float32(acc[i]).ir_value()],
                    "v_mov_b32 $0, $1",
                    "=v,v",
                    has_side_effects=True,
                )
            )
            for i in range(4)
        ],
        fx.Float32,
    )


def mfma_fp8_scaled(acc, a, b, code_sum):
    """``acc`` plus one 32-wide K block's product: the FP8 MFMA with a zero
    accumulator, then times the block's scale factor (``mx_factor``)."""
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    return acc_scaled(acc, mfma_fp8(a, b, zero), mx_factor(code_sum))


def ocp_to_fnuz(w):
    """Four OCP e4m3 bytes -> the FNUZ bytes for half their values. Every bit
    pattern keeps its meaning (halved) except 0x80 and the OCP NaNs. 0x80,
    OCP's -0, is the FNUZ NaN and becomes 0x00. The NaNs 0x7F and 0xFF stay
    as they are and read as 240 and -240. A byte's magnitude plus 0x7F
    carries into its sign bit exactly when the magnitude is not zero, without
    reaching the next byte, so the sign survives only on nonzero bytes."""
    w = fx.Int32(w)
    nonzero = ((w & MAG_BYTES) + MAG_BYTES) & SIGN_BYTES
    return w & (nonzero | MAG_BYTES)


def fp4x8_fnuz(w):
    """Eight e2m1 values in a dword (value i in bits 4 i .. 4 i + 3) -> two
    dwords of FNUZ e4m3 bytes: (values 0, 2, 4, 6) and (values 1, 3, 5, 7).

    Each byte is the e2m1 value times 2^-7, exactly, the subnormal 0.5
    included: the e2m1 sign goes to bit 7 and its exponent and mantissa bits
    to bits 4 .. 2, which are the low two exponent bits and the top mantissa
    bit of e4m3 with bias 8. A scale code for these bytes is the e8m0 code
    plus 7. A negative zero nibble (0x8) would give 0x80, the FNUZ NaN, so
    the weights must not hold one: ``weights942.fp4_drop_negative_zero_``
    rewrites them to +0 at load, which changes no product."""
    w = fx.Int32(w)
    even = ((w << 2) & 0x1C1C1C1C) | ((w << 4) & SIGN_BYTES)
    odd = ((w >> 2) & 0x1C1C1C1C) | (w & SIGN_BYTES)
    return even, odd


def fp8x2_f32(word, hi):
    """Two FNUZ bytes of a dword as f32: bytes 0 and 1, or 2 and 3 when ``hi``."""
    v = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), fx.Int32(word), hi))
    return v[0], v[1]


def bf16_hi_pair(a, b):
    """Two f32 that a bf16 holds exactly -> one dword of their bf16 (a low).
    The bf16 of such a value is the top half of its f32 bits."""
    return fx.Int32(
        rocdl.perm_b32(b.bitcast(fx.Int32), a.bitcast(fx.Int32), fx.Int32(0x07060302))
    )


def mfma_bf16_k32(a, b, c):
    """The 16x16x32 bf16 product of gfx950's ``mfma_f32_16x16x32_bf16`` as two
    16x16x16 bf16 MFMAs: lane l holds K 8 (l // 16) .. + 8 in ``a`` and
    ``b`` (8 bf16 each). The first MFMA takes elements 0 .. 3 of every lane,
    the second elements 4 .. 7, so together they cover all 32 K."""
    a16 = fx.Vector(a).bitcast(fx.Int16)
    b16 = fx.Vector(b).bitcast(fx.Int16)
    for h in range(2):
        ah = fx.Vector.from_elements([a16[4 * h + i] for i in range(4)], fx.Int16)
        bh = fx.Vector.from_elements([b16[4 * h + i] for i in range(4)], fx.Int16)
        c = fx.Vector(
            rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [ah, bh, c, 0, 0, 0])
        )
    return c
