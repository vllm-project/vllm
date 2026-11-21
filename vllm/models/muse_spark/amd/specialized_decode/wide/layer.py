# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Muse layer's decode step on one TP8 = EP8 rank of an MI300X, as one
304-CTA kernel a step width: from the attention output of layer `L` to
the QKVG output of layer `L + 1`.

The phases, with a grid barrier where a phase reads what other CTAs of the
rank wrote, and exchange flags where it reads what other ranks pushed:

 1. o-path gated norm + FP8 quant (a wave a token)           | barrier
 2. o_proj (split-K 2), each 64-column group scattered to its owner
 3. the owner's sum of the o_proj copies, gathered to every rank
 4. attention boundary: post_attn norm, gated residual, pre_ffn norm (a CTA a
    token)                                                    | barrier
 5. router (bf16x3 planes, split-K 16) and pooler (split-K 32) slices,
    gathered
 6. route + pre-expert norm + FP8 quant (a wave a token)      | barrier
 7. expert up pieces                                          | barrier
 8. expert down owners                                        | barrier
 9. finalize lines of the expert sum, scattered to their owners
10. the owner's sum of the expert copies, gathered
11. unpooler slice (split-K 16) off the gathered lines, gathered
12. FFN boundary: post_ffn norm, gated residual, next pre_qkv norm + FP8
    quant (a CTA a token)                                     | barrier
13. next layer's QKVG (split-K 8) to the QKVG buffer

Weights come from a per-layer table (`WEIGHTS`), the rank's buffers from a
per-width table (`SCRATCH`), both i64 byte addresses the runtime fills.
"""

import hashlib
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.muse_spark.amd.specialized_decode.wide.common.grid import (
    Grid,
    grid_sync,
    prologue,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.common.ops import (
    ld,
    ld_ptr64,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.layout import (
    BLOCKS,
    HIDDEN,
    INTER,
    KV_ROWS,
    POOL_ROWS,
    POOLED,
    POST_EPS,
    Q_ROWS,
    QKVG,
    THREADS,
    TOPK,
    TP,
    UNPOOL_ROWS,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages import dense, moe
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.exchange import (
    COLS,
    Exchange,
    GroupTile,
    Kind,
    LineSource,
    Mailbox,
    reduce_lines,
    wait_groups,
)
from vllm.models.muse_spark.amd.specialized_decode.wide.stages.tiled import (
    TILE,
    Gemm,
    Source,
    SplitK,
    bf16_weights,
    fold,
    fp8_weights,
    gemm_units,
    stage_words,
    token_tiles,
)

_STREAM = fx.Stream(None)


def _source_key() -> str:
    """A hash of this package's sources. FlyDSL's compile-cache key hashes
    only helpers in the launcher's own directory, and of closure values only
    scalars and tuples: the kernel closes over this (and its phases as a
    tuple), so an edit under `stages/` or `common/` or another phase
    subset compiles anew."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


_SOURCE_KEY = _source_key()

# the per-layer weight table (byte addresses)
WEIGHTS = (
    "o_w",
    "o_ws",  # FP8 [8192, 1024], f32 [8192]
    "attn_alpha",
    "attn_beta",
    "pre_ffn_w1",  # f32 [8192]
    "router",
    "bias",  # bf16 planes [128, 8192], f32 [256]
    "pooler",
    "pre_expert_w1",  # bf16 [512, 8192] (the rank's rows), f32 [4096]
    "w13",
    "w13s",
    "w2",
    "w2s",  # the local experts'
    "post_expert_w1",  # f32 [4096]
    "unpooler",  # bf16 [1024, 4096] (the rank's rows)
    "post_ffn_w1",
    "ffn_alpha",
    "ffn_beta",  # f32 [8192]
    "pre_qkv_w1",  # f32 [8192], layer L + 1's
    "qkvg_w",
    "qkvg_ws",  # FP8 [2304, 8192], f32 [2304], layer L + 1's
)
# the per-width buffers (byte addresses); ``rows`` the width
SCRATCH = (
    "stream",  # f32 [rows, 8192], the residual, in place
    "xo",
    "xs",  # FP8 [rows, 1024], f32 [rows]
    "x2",  # bf16 [rows, 8192]
    "xq",
    "xqs",  # FP8 [rows, 8192], f32 [rows]
    "ids",
    "wts",
    "hq",
    "hs",
    "act",
    "amax",
    "y",
    "sq",  # moe.MoEBufs
    "part_o",
    "part_router",
    "part_pool",
    "part_unpool",
    "part_qkvg",
)

# split-K splits a width: (o, router, pool, unpool, qkvg)
NARROW_SPLITS = (2, 16, 32, 16, 8)  # 128 x 2, 2 x 16, 8 x 32, 16 x 16, 36 x 8 units
WIDE_SPLITS = (2, 16, 16, 8, 8)  # the pooler fold of 32 splits: ~50 us at 128 rows


class Plans:
    """The split-K plans of width `rows`: a last arriver sums `splits *
    rows` partials a row, so wide steps cut K fewer ways."""

    def __init__(self, rows: int):
        o, router, pool, unpool, qkvg = NARROW_SPLITS if rows <= 16 else WIDE_SPLITS
        self.splits = (o, router, pool, unpool, qkvg)
        self.o = SplitK(HIDDEN, Q_ROWS, 1, splits=o)
        self.router = SplitK(dense.ROUTER_ROWS, HIDDEN, 2, splits=router)
        self.pool = SplitK(POOL_ROWS, HIDDEN, 2, splits=pool)
        self.unpool = SplitK(UNPOOL_ROWS, POOLED, 2, splits=unpool)
        self.qkvg = SplitK(QKVG, HIDDEN, 1, splits=qkvg)
        plans = (self.o, self.router, self.pool, self.unpool, self.qkvg)
        # the fold counters: each plan's row groups, in order
        firsts = [sum(p.groups for p in plans[:i]) for i in range(len(plans))]
        (
            self.fold_o,
            self.fold_router,
            self.fold_pool,
            self.fold_unpool,
            self.fold_qkvg,
        ) = firsts
        assert self.router.units + self.pool.units <= BLOCKS


# the layer's exchanges: groups of 64 bf16 columns (the logits: 32 f32)
O_SC, O_GA = Kind("o_sc", HIDDEN // COLS), Kind("o_ga", HIDDEN // COLS)
LOGITS = Kind("logits", dense.ROUTER_GROUPS * TP)
POOL_GA = Kind("pool_ga", POOLED // COLS)
MOE_SC, MOE_GA = Kind("moe_sc", POOLED // COLS), Kind("moe_ga", POOLED // COLS)
UNPOOL_GA = Kind("unpool_ga", HIDDEN // COLS)
MAILBOX = Mailbox((O_SC, O_GA, LOGITS, POOL_GA, MOE_SC, MOE_GA, UNPOOL_GA))

O_OWNED = HIDDEN // COLS // TP  # 16 groups a rank
TAG_LAYERS = 64  # tags a step: epoch * 64 + layer + 1
POOL_OWNED = POOLED // COLS // TP  # 8


def scratch_bytes(rows: int) -> dict[str, int]:
    """Each `SCRATCH` buffer's bytes at width `rows`."""
    a = rows * TOPK
    p = Plans(rows)
    return {
        "stream": rows * HIDDEN * 4,
        "xo": rows * Q_ROWS,
        "xs": rows * 4,
        "x2": rows * HIDDEN * 2,
        "xq": rows * HIDDEN,
        "xqs": rows * 4,
        "ids": a * 4,
        "wts": a * 4,
        "hq": rows * POOLED,
        "hs": rows * 4,
        "act": a * INTER * 2,
        "amax": a * 4,
        "y": a * POOLED * 2,
        "sq": a * moe.DOWN_OWNERS * 4,
        "part_o": p.o.partial_words(rows) * 4,
        "part_router": p.router.partial_words(rows) * 4,
        "part_pool": p.pool.partial_words(rows) * 4,
        "part_unpool": p.unpool.partial_words(rows) * 4,
        "part_qkvg": p.qkvg.partial_words(rows) * 4,
    }


class LdsPlan:
    """The LDS page of a width (i32 words): the dense GEMMs' stage, then their
    group tile; the MoE page over the same words; the boundary's 8 words
    after both."""

    def __init__(self, rows: int):
        self.moe = moe.MoEShape(rows)
        self.stage = 0
        self.tile = stage_words(rows)
        dense_words = self.tile + GroupTile.words(rows)
        self.red = max(dense_words, self.moe.words)
        self.words = self.red + 8
        # the router's mid halves (2 waves x 16 words a token) over the stage
        assert 2 * token_tiles(rows) * TILE * 16 <= self.tile


def _owner(owned):
    return lambda grp: grp // fx.Int32(owned)


ALL_PHASES = frozenset(range(1, 14))


def build(
    rows: int,
    tp: int = TP,
    phases=ALL_PHASES,
    route_scale: float = 1.0,
    w13_ld: int = POOLED,
    w2_ld: int = INTER,
):
    """The launcher of width `rows`: `launch(wtab, stab, peers, attn,
    qkv_in, qkv_out, epoch, layer, rank, sync, sync_next, folds, health,
    stream)`. `phases`: a subset of the module doc's phases to build (the
    others are left out: a measurement aid, not a working layer). The
    checkpoint's `route_scale` and the expert matrices' row strides (bytes)
    are compiled in."""
    lds = LdsPlan(rows)
    sh = lds.moe
    plans = Plans(rows)
    O_PLAN, ROUTER_PLAN, POOL_PLAN, UNPOOL_PLAN, QKVG_PLAN = (
        plans.o,
        plans.router,
        plans.pool,
        plans.unpool,
        plans.qkvg,
    )
    FOLD_O, FOLD_ROUTER, FOLD_POOL, FOLD_UNPOOL, FOLD_QKVG = (
        plans.fold_o,
        plans.fold_router,
        plans.fold_pool,
        plans.fold_unpool,
        plans.fold_qkvg,
    )
    key = (_SOURCE_KEY, *sorted(phases))
    split_key = plans.splits

    @fx.struct
    class Smem:
        word: fx.Array[fx.Int32, 4, 16]
        page: fx.Array[fx.Int32, lds.words, 16]

    name = f"muse942_layer_s{rows}_k" + "_".join(str(s) for s in plans.splits)
    if phases != ALL_PHASES:
        name += "_p" + "_".join(str(p) for p in sorted(phases))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        wtab: Int64,
        stab: Int64,
        peers: Int64,
        attn: Int64,
        qkv_in: Int64,
        qkv_out: Int64,
        epoch: Int64,
        layer: Int32,
        rank: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
    ):
        assert const_expr(len(split_key) == 5)  # keys the compile cache
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Smem).peek()
        g = Grid(tid, bid, sync, sync_next, folds, health, Int64(0), sm.word.ptr)
        prologue(g)
        w = {n: ld_ptr64(wtab, i) for i, n in enumerate(WEIGHTS)}
        s = {n: ld_ptr64(stab, i) for i, n in enumerate(SCRATCH)}
        bases = [ld_ptr64(peers, r) for r in range(tp)]
        own = ld_ptr64(peers, rank)
        # a step's tags differ from every earlier step's: the epoch is bumped
        # on the device before the step's launches (graph replays included)
        tag = ld(epoch, 0, fx.Int32) * fx.Int32(TAG_LAYERS) + layer + fx.Int32(1)
        parity = layer % fx.Int32(2)

        def xch(kind, mode, owner=None):
            return Exchange(
                MAILBOX, kind, mode, bases, own, rank, parity, tag, rows, owner=owner
            )

        page = sm.page.ptr
        tile = GroupTile(page + lds.tile, rows)
        red = page + lds.red
        waves = fx.Int32(BLOCKS * WAVES)

        # the exchanges and buffers (no code until used)
        o_sc = xch(O_SC, "scatter", _owner(O_OWNED))
        o_ga = xch(O_GA, "gather", _owner(O_OWNED))
        logits = xch(LOGITS, "gather", _owner(dense.ROUTER_GROUPS))
        pool_ga = xch(POOL_GA, "gather", _owner(POOL_OWNED))
        b = moe.MoEBufs(
            s["ids"],
            s["wts"],
            s["hq"],
            s["hs"],
            s["act"],
            s["amax"],
            s["y"],
            s["sq"],
            w["w13"],
            w["w13s"],
            w["w2"],
            w["w2s"],
            w["bias"],
            w["pre_expert_w1"],
            w["post_expert_w1"],
            route_scale=route_scale,
            w13_ld=w13_ld,
            w2_ld=w2_ld,
        )
        moe_sc = xch(MOE_SC, "scatter", _owner(POOL_OWNED))
        moe_ga = xch(MOE_GA, "gather", _owner(POOL_OWNED))
        unpool_ga = xch(UNPOOL_GA, "gather", _owner(O_OWNED))

        # 1. o-path gated norm + quant
        if const_expr(1 in key):
            dense.o_gate_quant(
                g, rows, attn, qkv_in, QKVG, Q_ROWS + 2 * KV_ROWS, s["xo"], s["xs"]
            )
            grid_sync(g)

        # 2. o_proj, scattered
        if const_expr(2 in key):
            gemm = Gemm(page + lds.stage, rows, tid, 1)
            fin, done = dense.push_finish(
                g,
                gemm,
                tile,
                o_sc,
                lambda grp: grp,
                lambda grp: dense.dequant(
                    s["xs"], w["o_ws"], lambda col: grp * COLS + col
                ),
            )
            ep, after = fold(g, gemm, O_PLAN, s["part_o"], FOLD_O, fin, done)
            gemm_units(
                g.bid,
                fx.Int32(BLOCKS),
                gemm,
                Source(s["xo"], Q_ROWS),
                fp8_weights(w["o_w"], HIDDEN, Q_ROWS),
                O_PLAN,
                ep,
                after,
            )

        # 3. the owner's sum, gathered
        if const_expr(3 in key):
            reduce_lines(g, o_sc, o_ga, O_OWNED, g.spread, waves)

        # 4. attention boundary
        if const_expr(4 in key):
            bd = dense.Boundary(
                None,
                w["attn_alpha"],
                w["attn_beta"],
                w["pre_ffn_w1"],
                POST_EPS,
                quant=False,
            )
            dense.boundary_tokens(
                g, rows, bd, o_ga, HIDDEN, s["stream"], s["x2"], None, red
            )
            grid_sync(g)

        # 5. router + pooler slices
        if const_expr(5 in key):
            x2 = Source(s["x2"], HIDDEN * 2)
            rgemm = Gemm(page + lds.stage, rows, tid, 2)
            fin, done = dense.router_finish(
                g, rgemm, tile, page + lds.stage, logits, rank
            )
            ep, after = fold(
                g, rgemm, ROUTER_PLAN, s["part_router"], FOLD_ROUTER, fin, done
            )
            gemm_units(
                g.bid,
                fx.Int32(BLOCKS),
                rgemm,
                x2,
                bf16_weights(w["router"], dense.ROUTER_ROWS, HIDDEN),
                ROUTER_PLAN,
                ep,
                after,
            )
            pgemm = Gemm(page + lds.stage, rows, tid, 2)
            fin, done = dense.push_finish(
                g, pgemm, tile, pool_ga, lambda grp: rank * POOL_OWNED + grp
            )
            ep, after = fold(g, pgemm, POOL_PLAN, s["part_pool"], FOLD_POOL, fin, done)
            shifted = (g.bid + fx.Int32(BLOCKS - ROUTER_PLAN.units)) % fx.Int32(BLOCKS)
            gemm_units(
                shifted,
                fx.Int32(BLOCKS),
                pgemm,
                x2,
                bf16_weights(w["pooler"], POOL_ROWS, HIDDEN),
                POOL_PLAN,
                ep,
                after,
            )

        # 6. route + expert input
        if const_expr(6 in key):
            moe.route_tokens(g, sh, b, logits, pool_ga)
            grid_sync(g)

        # 7, 8. experts
        if const_expr(7 in key or 8 in key):
            moe.schedule(g, sh, page, b, rank)
            egemm = moe.expert_gemm(sh, page, tid)
            if const_expr(7 in key):
                moe.up_units(g, sh, page, b, egemm)
                grid_sync(g)
            if const_expr(8 in key):
                moe.down_units(g, sh, page, b, egemm)
                grid_sync(g)

        # 9, 10. finalize, the owner's sum
        if const_expr(9 in key or 10 in key):
            moe.finalize_lines(g, sh, page, b, moe_sc, rank)
            reduce_lines(g, moe_sc, moe_ga, POOL_OWNED, g.spread, waves)

        # 11. unpooler slice
        if const_expr(11 in key):
            ugemm = Gemm(page + lds.stage, rows, tid, 2)
            fin, done = dense.push_finish(
                g, ugemm, tile, unpool_ga, lambda grp: rank * O_OWNED + grp
            )
            ep, after = fold(
                g, ugemm, UNPOOL_PLAN, s["part_unpool"], FOLD_UNPOOL, fin, done
            )
            per = UNPOOL_PLAN.per

            def before(group, split):
                wait_groups(g, moe_ga, split * fx.Int32(per), per)

            gemm_units(
                g.bid,
                fx.Int32(BLOCKS),
                ugemm,
                LineSource(moe_ga),
                bf16_weights(w["unpooler"], UNPOOL_ROWS, POOLED),
                UNPOOL_PLAN,
                ep,
                after,
                before=before,
            )

        # 12. FFN boundary + next pre_qkv norm + quant
        if const_expr(12 in key):
            bd = dense.Boundary(
                w["post_ffn_w1"],
                w["ffn_alpha"],
                w["ffn_beta"],
                w["pre_qkv_w1"],
                POST_EPS,
                quant=True,
            )
            dense.boundary_tokens(
                g, rows, bd, unpool_ga, HIDDEN, s["stream"], s["xq"], s["xqs"], red
            )
            grid_sync(g)

        # 13. next layer's QKVG
        if const_expr(13 in key):
            qgemm = Gemm(page + lds.stage, rows, tid, 1)
            fin = dense.store_finish(qgemm, qkv_out, QKVG, s["xqs"], w["qkvg_ws"])
            ep, after = fold(g, qgemm, QKVG_PLAN, s["part_qkvg"], FOLD_QKVG, fin)
            gemm_units(
                g.bid,
                fx.Int32(BLOCKS),
                qgemm,
                Source(s["xq"], HIDDEN),
                fp8_weights(w["qkvg_w"], QKVG, HIDDEN),
                QKVG_PLAN,
                ep,
                after,
            )

    @flyc.jit
    def launch(
        wtab: Int64,
        stab: Int64,
        peers: Int64,
        attn: Int64,
        qkv_in: Int64,
        qkv_out: Int64,
        epoch: Int64,
        layer: Int32,
        rank: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(
            wtab,
            stab,
            peers,
            attn,
            qkv_in,
            qkv_out,
            epoch,
            layer,
            rank,
            sync,
            sync_next,
            folds,
            health,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch, lds


def compile_only(rows: int, phases=ALL_PHASES) -> None:
    """Trace and compile width `rows` without a device (`COMPILE_ONLY=1`)."""
    launch, _ = build(rows, phases=phases)
    flyc.compile(launch, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, _STREAM)
