# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dense_post``: one MiniMax-M3 dense layer past its attention, o_proj through
the MLP's FFN reduce.

A dense layer is three launches: K1 without the indexer (``dense_pre``), the
original attention kernel, then this one. One launch per layer and rank,
``BLOCKS`` CTAs x ``THREADS`` threads, for a step of S = 1..MAX_TOKENS tokens::

    merge : (S > 1) a task per token: the attention output's per-token FP8
    o     : o_proj GEMV -> push bf16 partial rows to every peer -> rank-ordered
            sum -> acc = bf16(sum) + h (h_mid = bf16(acc)), as K4's o stage
    norm  : (S > 2) a task per token: gemma RMSNorm(acc) -> per-token FP8
    ug    : gate / up GEMV (16 mid columns a task) -> swiglu -> bf16 mid
    down  : the token's mids -> per-token FP8 -> down GEMV -> push bf16 partial
            rows -> rank-ordered sum -> ar_out = bf16(sum), as K4's FFN reduce

Its output is K4's (ar_out, h_mid): the next layer, dense or sparse, takes it as
the original path's fused all-reduce + RMSNorm takes (partials, residual).
The stages' functions are in ``stages`` (``attention.o_defs``, ``dense_mlp``,
``ffn_reduce``)."""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from ..config import (
    HIDDEN,
    MAX_TOKENS,
)
from ..layout import (
    AG_RS_FROM,
    DMID_ROW,
    DX_ROW,
    O_ROW,
    O_ROWS,
    SCRATCH,
    dense_stage_bases,
    diag_region_names,
    sym_layout,
)
from ..sources import SOURCES
from ..support.device.ops import (
    block_max as block_max_of,
)
from ..support.device.ops import (
    kernel_symbol,
    launcher,
    load_ptr64,
)
from ..support.device.stamps import (
    stamp as stamp_point,
)
from ..support.device.stamps import (
    stamp_begin,
    stamp_flush,
)
from ..support.device.sync import (
    Mailbox,
    preg,
    sreg,
)
from ..support.plan.build_key import (
    key_tuple,
    symbol_params,
)
from ..support.plan.execution import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
)
from ..support.plan.trace import (
    enter_stage,
)
from ..support.runtime.abi import (
    KernelAbi,
)
from ..support.runtime.debug import (
    region_ids,
)
from .stages.attention import (
    o_defs,
)
from .stages.dense_mlp import (
    dense_mlp_defs,
)
from .stages.ffn_reduce import (
    ffn_reduce_defs,
)

# The launch goes on the caller's current stream (graph capture included).
_CURRENT_STREAM = fx.Stream(None)

# dense_post's arguments, in order (checked at build time, packed by name)
DENSE_POST_ABI = KernelAbi(
    (
        "attn", "h_in", "w_o", "s_o", "g_post", "w_gu", "s_gu", "w_dn", "s_dn",
        "h_mid", "ar_out", "scratch", "sym", "peers", "rank", "layer", "tl",
    )
)  # fmt: skip
# ``timeline``: per CTA, the s_memrealtime of passing each stage's end (tl
# [BLOCKS][DENSE_TL_POINTS], int64): 1 merge, 3 o got its input, 4 o, 5 norm,
# 6 ug, 7 mq, 8 down
DENSE_TL_POINTS = 16


@dataclass(frozen=True)
class DensePostBuild:
    """Every parameter of a dense_post build (``support.plan.build_key``)."""

    npes: int = field(metadata={"sym": "tp"})
    tokens: int = field(metadata={"sym": "s"})
    index_heads: int = field(metadata={"sym": "ih"})
    timeline: bool = field(metadata={"sym": "tl"})
    eps: float
    swiglu_alpha: float
    swiglu_beta: float
    swiglu_limit: float
    debug: bool


def build_dense_post_kernel(
    npes: int,
    eps: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    tokens: int = 1,
    debug: bool = False,
    index_heads: int = 1,
    timeline: bool = False,
):
    """``@flyc.jit`` launcher of dense_post for one rank of an ``npes``-way TP
    group and a decode step of ``tokens`` (<= MAX_TOKENS) rows. ``debug``: every
    mailbox wait bounded, a wait that gives up recorded in scratch ``diag``,
    its region numbered as the step's sparse layer kernels (built for
    ``index_heads``) number theirs. ``timeline``: stage stamps into ``tl``
    (``DENSE_TL_POINTS``)."""
    assert 1 <= tokens <= MAX_TOKENS
    key = DensePostBuild(
        npes=npes, tokens=tokens, index_heads=index_heads, timeline=timeline, eps=eps,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit,
        debug=debug,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)  # the launcher references it: keyed
    # a debug build's wait records name a region by its ``region_id``
    REGION_IDS = region_ids(diag_region_names(tokens, index_heads))
    W = npes
    G = BLOCKS
    BASE = dense_stage_bases(tokens)
    SY = sym_layout(W)
    AG_RS = tokens >= AG_RS_FROM
    # the o stage's names: K1 is a separate launch, its outputs plain loads
    CM_K1 = 0
    fuse_k1 = False
    k1_poll = rdone_mb = None

    @fx.struct
    class Smem:
        # every token's FP8 row of the stage's GEMV input (the o stage's
        # attention, the gate / up input, the down stage's mids)
        x: fx.Array[fx.Int32, max(DX_ROW, O_ROW, DMID_ROW) * tokens, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        red2: fx.Array[fx.Float32, WAVES * 64 * 4, 16]  # the up rows' partials
        misc: fx.Array[fx.Float32, O_ROWS * MAX_TOKENS, 16]  # a task's output rows
        tls: fx.Array[fx.Int64, DENSE_TL_POINTS if timeline else 1, 16]

    kernel_name = kernel_symbol("minimax_m3_mono_dense_post", **symbol_params(key))

    # every builder name, for the stage factories
    _bc = dict(locals())

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def dense_post_kernel(
        attn: Int64,
        h_in: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_dn: Int64,
        s_dn: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        tl: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        red = lds.red.ptr
        red2 = lds.red2.ptr
        misc = lds.misc.ptr
        tls = lds.tls.ptr
        v4f = fx.Vector.make_type(4, fx.Float32)

        diag = scratch + fx.Int64(SCRATCH["diag"]) if debug else None
        mbox = Mailbox(layer, diag, REGION_IDS)
        # Plain names, not method calls: flydsl's AST rewriter carries every local
        # whose method is called inside a dynamic if/for as loop state.
        mb_put = mbox.put
        mb_put_bf = mbox.put_bf
        mb_poll = mbox.poll

        def mb(name):
            """Scratch region ``name``'s raw address (plain loads / stores)."""
            return scratch + fx.Int64(SCRATCH[name])

        def mr(name):
            """Scratch mailbox region ``name`` (puts / polls)."""
            return sreg(scratch, SCRATCH[name], name)

        def acquire():
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )

        def stamp(k):
            """``timeline``: this CTA passing point k (``support.device.stamps``)."""
            stamp_point(timeline, tls, tid, k)

        def block_max(v):
            return block_max_of(v, lane, wave, red)

        # each wave pushes to one peer
        peer_dst = load_ptr64(peers, fx.min(wave, W - 1))

        def peer_addr(w):
            return load_ptr64(peers, w)

        def start(name):
            return first_task(bid, BASE[name])

        def push_rows(region, row0, n_rows, only=None):
            """This task's bf16 partial rows (misc: token k's n_rows from k n_rows) to
            the peers' ``region``, a pair a lane, wave w -> peer w (itself too);
            ``only``: to that peer alone."""
            half = n_rows // 2
            to = (wave < W) if only is None else (wave == only)
            for c in range_constexpr((half * tokens + 63) // 64):
                i = lane + 64 * c
                if to & (i < half * tokens):
                    mb_put_bf(
                        preg(peer_dst, SY[region], region),
                        (rank * MAX_TOKENS + i // half) * HIDDEN
                        + row0
                        + (i % half) * 2,
                        [fx.ptr_load(misc + i * 2), fx.ptr_load(misc + (i * 2 + 1))],
                    )

        dctx = dict(_bc)
        dctx.update(locals())
        dctx.update(ffn_reduce_defs(dctx))
        dctx.update(locals())
        dctx.update(dense_mlp_defs(dctx))
        dctx["merge_token"] = dctx["quant_token"]
        dctx.update(o_defs(dctx))

        stamp_begin(timeline, tls, tid, DENSE_TL_POINTS)
        enter_stage("merge")
        dctx["stage_quant"]()
        stamp(1)
        enter_stage("o")
        dctx["stage_o"]()
        stamp(4)
        enter_stage("norm")
        dctx["stage_norm"]()
        stamp(5)
        enter_stage("ug")
        dctx["stage_ug"]()
        stamp(6)
        enter_stage("mq")
        dctx["stage_mq"]()
        stamp(7)
        enter_stage("down")
        dctx["stage_down"]()
        stamp(8)
        stamp_flush(timeline, tls, tl, tid, bid, DENSE_TL_POINTS)

    @launcher(kernel_name)
    def launch_dense_post(
        attn: Int64,
        h_in: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_dn: Int64,
        s_dn: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key  # every build parameter in the JIT cache key
        dense_post_kernel(
            attn, h_in, w_o, s_o, g_post, w_gu, s_gu, w_dn, s_dn, h_mid, ar_out,
            scratch, sym, peers, rank, layer, tl,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)  # fmt: skip

    DENSE_POST_ABI.check(dense_post_kernel, launch_dense_post)
    return launch_dense_post
