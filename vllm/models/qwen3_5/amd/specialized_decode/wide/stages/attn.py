# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A full-attention layer's projections around vLLM's attention: the input
projection ``qkv = x @ W_qkv.T`` (bf16, ``[q | gate interleaved a head | k |
v]`` as ``Qwen3NextAttention.qkv_proj`` returns it) as the tail of the
launch before it, in three K slices a 64-row group folded by the group's
last arriver into bf16 rows; its output projection is ``gdn.out`` over
``gdn.GatedCore``.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    pack_bf16,
    st,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.layout import (
    BLOCKS,
    HIDDEN,
    QKV,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import tiled

QKV_SPLITS = 3  # 72 row groups x 3 = 216 units


def qkv_plan():
    return tiled.SplitK(QKV, HIDDEN, QKV_SPLITS)


def qkv_weights(w):
    return tiled.Bf16Weights(w, QKV, HIDDEN)


def qkv_proj(g, gemm, w, x, qkv_out, part, rows: int, fold0: int, pending=None):
    """``qkv_out`` ([rows, QKV] bf16) of ``x`` ([rows, HIDDEN] bf16)."""
    plan = qkv_plan()

    def finish(group, first, lane, acc):
        row = first + lane // 16 * 4
        for t in range_constexpr(gemm.tiles):
            token = fx.Int32(t * 16) + lane % 16
            if token < rows:
                words = [pack_bf16(acc[t][0], acc[t][1]), pack_bf16(acc[t][2], acc[t][3])]
                st(
                    qkv_out,
                    (token * QKV + row) // 2,
                    fx.Vector.from_elements(words, fx.Int32),
                    CM_DEV,
                )

    epilogue, after = tiled.fold(g, gemm, plan, part, fold0, finish)
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        tiled.Source(x, HIDDEN),
        qkv_weights(w),
        plan,
        epilogue,
        after,
        pending=pending,
    )
