# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA layer's projections in `qsa_front`: vLLM's `qkv_proj` and the
indexer's `index_qk_proj` (both bf16, read in place) on the launch's `x`,
as one grid phase.

- `qkv`: `x @ W_qkv.T` ([q | gate] per head, then k, v: QKVParallelLinear's
  rank shard, `Shard.QKV` rows) in `QKV_SPLITS` K slices a 64-row group,
  folded by the group's last arriving unit, which stores the group's bf16
  columns of `qkv_out` [rows, QKV].
- `ix`: `x @ W_ix.T` (the replicated indexer Q heads then K, `IX` rows)
  likewise in `IX_SPLITS` slices, on the CTAs after the `qkv` units, into
  `ix_out` [rows, IX].

Each output is the f32 sum of its splits' partials in split order, rounded
to bf16 once, as the stock bf16 GEMMs round their f32 accumulators.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    pack_bf16,
    st,
    traced,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    HIDDEN,
    TPS,
    shard,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tiled

IX_HEADS, IX_DIM = 4, 128  # the indexer's query heads (one key head), dims
IX = (IX_HEADS + 1) * IX_DIM  # 640, replicated
QKV_SPLITS = {1: 1, 2: 2}  # by TP: 208 row groups x 1, 104 x 2 = 208 units
IX_SPLITS = 4  # 10 row groups x 4 = 40 units, on CTAs 208 ..
# by TP: the ix groups' first fold counter (after the qkv groups')
FOLD_IX = {1: 256, 2: 128}


def i32(v):
    return fx.Int32(v)


def qkv_plan(sh):
    return tiled.SplitK(sh.QKV, HIDDEN, QKV_SPLITS[sh.TP])


def ix_plan():
    return tiled.SplitK(IX, HIDDEN, IX_SPLITS)


def partial_words(sh, rows: int) -> int:
    """f32 words of split-K partials the phase needs: qkv's, then ix's."""
    return qkv_plan(sh).partial_words(rows) + ix_plan().partial_words(rows)


assert all(qkv_plan(shard(tp)).units + ix_plan().units <= BLOCKS for tp in TPS)
assert all(qkv_plan(shard(tp)).groups <= FOLD_IX[tp] for tp in TPS)


@traced
def _store_rows(out, cols: int, rows: int, first, lane, acc):
    """A fold's summed rows `r .. r + 3` (output columns) of each token,
    rounded to bf16, at `out[token, r]`."""
    col = first + lane // i32(16) * i32(4)
    for t in range_constexpr(len(acc)):
        token = i32(t * 16) + lane % i32(16)
        if token < i32(rows):
            v = acc[t]
            words = fx.Vector.from_elements(
                [pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3])], fx.Int32
            )
            st(out, (token * i32(cols) + col) // i32(2), words)


def _store_bf16(out, cols: int, rows: int):
    def finish(group, first, lane, acc):
        _store_rows(out, cols, rows, first, lane, acc)

    return finish


@traced
def proj(g, gemm, sh, w_qkv, w_ix, x, qkv_out, ix_out, part, rows: int):
    """Both projections of the step's `rows` rows of `x` [rows, HIDDEN]
    bf16 (written earlier in the launch); `part`: the scratch's partials."""
    source = tiled.Source(x, HIDDEN)
    plan = qkv_plan(sh)
    epi, after = tiled.fold(g, gemm, plan, part, 0, _store_bf16(qkv_out, sh.QKV, rows))
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        source,
        tiled.Bf16Weights(w_qkv, sh.QKV, HIDDEN),
        plan,
        epi,
        after,
    )
    iplan = ix_plan()
    ipart = fx.Int64(part) + fx.Int64(plan.partial_words(rows) * 4)
    iepi, iafter = tiled.fold(
        g, gemm, iplan, ipart, FOLD_IX[sh.TP], _store_bf16(ix_out, IX, rows)
    )
    weights = tiled.Bf16Weights(w_ix, IX, HIDDEN)
    for unit in range(g.bid - i32(plan.units), i32(iplan.units), i32(BLOCKS)):
        if unit >= i32(0):
            group, split, firsts, rws = tiled.unit_rows(iplan, unit, gemm.tid)
            acc = gemm.run(source, weights, rws, split * i32(iplan.per), iplan.per)
            iepi(unit, split, firsts[0], gemm.lane, acc[0])
            iafter(group, split)
