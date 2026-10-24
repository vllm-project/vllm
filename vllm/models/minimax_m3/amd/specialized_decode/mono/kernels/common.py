# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/common.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's own device helpers: bf16 row loads, the per-token FP8 scale of
aiter's ``dynamic_per_token_scaled_quant``, and the all-reduce epilogue in the
order the original path's fused all-reduce + RMSNorm reduces
(``support.device`` holds the model-independent primitives)."""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..support.device.mx import (
    FP8_MAX,
)
from ..support.device.ops import (
    butterfly,
    hw_rcp,
    traced,
)


def ld_bf16x4(r, k, cm=0):
    """4 bf16 from element ``k`` of buffer resource ``r``, as f32."""
    w = fx.Vector(
        bo.buffer_load(r, k // 2, vec_width=2, dtype=T.i32, cache_modifier=cm)
    )
    v = w.bitcast(fx.BFloat16).to(fx.Float32)
    return [v[j] for j in range(4)]


def ld_bf16x12(r, word, cm=0):
    """12 bf16 from ``word`` (a 16 B then an 8 B load)."""
    a8 = fx.Vector(
        bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm)
    ).bitcast(fx.BFloat16)
    a4 = fx.Vector(
        bo.buffer_load(r, word + 4, vec_width=2, dtype=T.i32, cache_modifier=cm)
    ).bitcast(fx.BFloat16)
    return [a8[e] for e in range(8)] + [a4[e] for e in range(4)]


def per_token_fp8_scale(amax):
    """A row's FP8 scale ``amax * (1 / FP8_MAX)`` and the multiplier its elements
    take, the scale's hardware reciprocal; an all-zero row (a pad row) quantizes
    to zeros, not 0 * inf."""
    x_scale = amax * (1.0 / FP8_MAX)
    return x_scale, (amax == 0.0).select(fx.Float32(0.0), hw_rcp(x_scale))


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc


@traced
def ar_block_sum(p0, p1, lane, wave, red):
    """Sum of the 768 per-pack partials of a 6144-wide bf16 row in the order the
    1-stage fused all-reduce + RMSNorm reduces them: 24 warps of 32, a 16..1
    butterfly per warp, then the 24 warp sums butterflied again in one warp.

    Logical warp ``3 * wave + lane // 32`` holds ``p0``; logical warp
    ``3 * wave + 2`` holds ``p1`` in lanes < 32 (``WAVES`` = 8). ``red`` needs
    24 words. Matching this order keeps rstd bit-exact with the original path.
    """
    return ar_block_sums([(p0, p1)], lane, wave, red)[0]


@traced
def ar_block_sums(parts, lane, wave, red):
    """``ar_block_sum`` of several rows ``parts = [(p0, p1), ...]`` behind one set
    of barriers; ``red`` needs 24 words per row."""
    bs = [
        (butterfly(p0, (16, 8, 4, 2, 1)), butterfly(p1, (16, 8, 4, 2, 1)))
        for p0, p1 in parts
    ]
    gpu.barrier()
    for k in range_constexpr(len(bs)):
        b0, b1 = bs[k]
        if lane % 32 == 0:
            fx.ptr_store(b0, red + (24 * k + wave * 3 + lane // 32))
        if lane == 0:
            fx.ptr_store(b1, red + (24 * k + wave * 3 + 2))
    gpu.barrier()
    li = lane % 32
    tots = [
        butterfly(
            (li < 24).select(
                fx.ptr_load(red + (24 * k + fx.min(li, 23))), fx.Float32(0.0)
            ),
            (16, 8, 4, 2, 1),
        )
        for k in range(len(parts))
    ]
    gpu.barrier()
    return tots
