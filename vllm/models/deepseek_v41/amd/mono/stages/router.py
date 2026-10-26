# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/model_ops/deepseek_v41/router.py
"""V4.1's router GEMV tile for decode: bf16 x [T, 5120] times the gate weight
[384, 5120] -> f32 logits, in one fixed summation order.

Rows 16 a tile, K in ``PARTS`` contiguous parts, 16x16x32 bf16 MFMA: in part
p wave w takes the part's K steps w, w + 8, ... (32 columns each) in that
order, and the eight wave partials are summed in wave order
(``common.ops.row_sum``); the MoE's route stage sums the parts in part
order. The MoE stages spread a tile's parts over PARTS CTAs, and past 16 tokens
a token tile a CTA too, which the order allows: a CTA alone cannot read a
tile, or every token's x, fast enough.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import mfma_bf16, rsrc
from vllm.models.deepseek_v41.amd.mono.common.plan import WAVES

HIDDEN = 5120
ROWS = 16  # a tile's rows: the MFMA's M
PARTS = 4  # contiguous K parts of a tile, summed in order


def _k0s(part, lane, wave):
    """The K offsets of this lane's loads in part ``part`` (8 bf16 each)."""
    per_wave = HIDDEN // 32 // WAVES // PARTS
    assert per_wave * 32 * WAVES * PARTS == HIDDEN
    return [
        (part * per_wave * WAVES + wave + WAVES * i) * 32 + 8 * (lane // 16)
        for i in range(per_wave)
    ]


def router_weight_loads(w, task, part, lane, wave, w_cm=0):
    """``router_tile``'s weight operands (rows 16 task .., K part ``part``).
    ``w_cm``: their load cache policy (nontemporal where nothing else of the
    step reuses them)."""
    row = task * ROWS + lane % 16
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(w),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=w_cm,
            )
        )
        for k0 in _k0s(part, lane, wave)
    ]


def router_x_loads(x, part, s, lane, wave, x_cm=0, t0=0):
    """``router_tile``'s x operands: the s tokens from ``t0`` (column
    lane % 16; ``t0`` traced or an int). ``x_cm``: their load cache policy
    (device scope when another CTA of the same launch wrote x)."""
    tok = fx.min(lane % 16, s - 1)
    if const_expr(not isinstance(t0, int) or t0 != 0):
        tok = tok + t0
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(x),
                (tok * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=x_cm,
            )
        )
        for k0 in _k0s(part, lane, wave)
    ]


def router_tile_mfma(wvs, xvs, lane, wave, red):
    """``router_tile``'s MFMAs, K steps in order -> ``red[wave][lane]``."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(wvs)):
        acc = mfma_bf16(wvs[i].bitcast(fx.BFloat16), xvs[i].bitcast(fx.BFloat16), acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
