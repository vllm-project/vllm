# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 x MXFP4 (A16W4) MFMA operands over AITER's post-load layouts.

Weights are ``shuffle_weight(layout=(16, 16))`` fp4x2 ``[E, N, K / 2]``: a
16-row group's 128-wide K step is 1 KiB, and lane ``l``'s 16 bytes hold row
``l % 16``, K ``32 (l / 16) ..`` -- one E8M0 group. Scales are
``e8m0_shuffle`` ``[E N, K / 32]``; lane ``l``'s dword at
``(r32 C8 + st / 2) 64 + l`` holds its row's codes of steps ``st`` (even, odd)
for both 16-row halves of the 32-row block. A dequantized dword is one
``16x16x32`` BF16 MFMA A operand; the B operand is the same 8 K values of the
lane's token (column ``l % 16``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import CM_NT, mfma_bf16, rsrc

STEP = 128  # K a lane group covers in one 16-byte weight load
STEP_DWORDS = 256  # a 16-row group's dwords per step


def weight_loads(w, row_group, k, st0, steps, cm=CM_NT):
    """Lane's 16-byte weight words of 16-row group ``row_group`` (counted over
    the flattened ``[E N]`` rows) for steps ``st0 ..``; loads only."""
    lane = fx.thread_idx.x % 64
    base = row_group * (2 * k) + lane * 4
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(w),
                base + (st0 + i) * STEP_DWORDS,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        for i in range(steps)
    ]


def scale_loads(ws, row_group, k, st0, steps, cm=CM_NT):
    """Lane's scale dwords for steps ``st0 .. st0 + steps`` (st0, steps even)."""
    lane = fx.thread_idx.x % 64
    c8 = k // 256
    base = row_group // 2 * c8 * 64 + lane
    return [
        fx.Int32(
            bo.buffer_load(
                rsrc(ws),
                base + (st0 // 2 + i) * 64,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        for i in range(steps // 2)
    ]


def scale_f32(words, row_group, st):
    """The lane's E8M0 scale of step ``st`` (relative to the first loaded
    word) as an f32 power of two."""
    byte = (st % 2) * 2 + row_group % 2
    word = words[st // 2]
    return (((word >> (8 * byte)) & 0xFF) << 23).bitcast(fx.Float32)


def dequant(word, scale):
    """One fp4x2 dword (8 values, low nibble first) -> 8 BF16 (as 4 i32)."""
    pairs = [
        fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16), word.ir_value(), scale.ir_value(), j
            )
        ).bitcast(fx.Int32)[0]
        for j in range(4)
    ]
    return fx.Vector.from_elements(pairs, fx.Int32)


def step_mfma(acc, wv, scale, xs):
    """``acc`` += one step: weight words ``wv`` (scale ``scale``) times the
    lane's four B operands ``xs`` (vector<4xi32> each)."""
    for d in range_constexpr(4):
        a = dequant(fx.Int32(wv[d]), scale).bitcast(fx.BFloat16)
        acc = mfma_bf16(a, xs[d].bitcast(fx.BFloat16), acc)
    return acc
