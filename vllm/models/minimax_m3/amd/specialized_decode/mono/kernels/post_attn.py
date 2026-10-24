# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/post_attn.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 ``mono_post``: one MiniMax-M3 sparse MoE layer, attention through FFN reduce.

One launch per layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, for a step of
S = 1..MAX_TOKENS tokens (each token's stage tasks side by side); for one token::

    index : K1, fused in, scored every index-K block (the Triton decode scorer's
            block score; with indexer context parallelism a long context's
            blocks of this rank, for every index head); every split CTA then
            runs the Triton selector's top-k and page-16 table emission itself
            (ranking, no sort; for a long request the split tasks' shares'
            best TOPK_BLOCKS are ranked, gathered from the peers under CP)
    split : one 256-key context partition x 16 heads per task (the gluon decode's
            partitioning) over that page-16 table, FP8 QK / PV with its scale
            placement
    o     : combine the partitions (the gluon reduce), per-token FP8 (standalone
            quant formula) -> o_proj GEMV
            -> push bf16 partial rows to every peer -> rank-ordered sum
            -> mailbox ``a`` = bf16(sum), h_mid = bf16(bf16(sum) + h)
    router: add residual in fp32; gemma RMSNorm in the fused all-reduce's order
            -> bf16
            gate GEMV -> logits; publishes its slice of the normalized input
    moe   : sigmoid + bias top-4 (+ the fused shared expert) -> MXFP8 x MXFP4
            up/gate on aiter's shuffled layout -> swiglu -> MXFP8 mid;
            the CTA's down task reuses that routing, its weights prefetched
            under the up/gate GEMV: route-weighted down GEMV -> push bf16
            partial rows -> rank-ordered sum -> ar_out = bf16(sum)

From ``WIDE_FROM`` tokens the MoE groups its work by expert (``wide_moe``): the
router tasks publish xn as MXFP8, each up / gate task pair runs one distinct
expert's 32 rows for every token routed to it (a8w4 MFMA, a B column per token)
and publishes MXFP8 mid rows, and a down task sums over the step's (expert,
token) mids. From ``AG_RS_FROM`` tokens both all-reduces are a reduce-scatter to
the row group's owner rank, which sums in rank order and all-gathers the result.

The next layer's K1 takes (ar_out, h_mid) exactly as the original path's fused
all-reduce + RMSNorm takes (partials, residual).

Every stage is a task list; task t runs on CTA (base + t) % BLOCKS and every CTA
walks the stages in order. Dependencies only point to earlier stages and all
CTAs are co-resident, so every spin wait makes progress (``contract`` checks the
traced hand-offs against ``layout.mailbox_regions``).

This module builds the kernel and runs its program: the stages' functions live in
``stages`` (a module per group), defined over the kernel's names and called here
in program order.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from ..config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
)
from ..layout import (
    AG_RS_FROM,
    MID_ROW,
    O_K,
    O_ROWS,
    SCRATCH,
    SPLIT_KEYS,
    UG_PER_SLOT,
    XN8_ROW,
    XSC_ROW,
    H,
    mailbox_regions,
    pool_words,
    stage_bases,
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)
from ..sources import SOURCES
from ..support.device.ops import (
    CM_DEV,
    kernel_symbol,
    launcher,
    load_ptr64,
)
from ..support.device.ops import (
    block_max as block_max_of,
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
    shift,
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
from .index_score import (
    index_scale_log2e,
    step_rows,
)
from .pre_attn import (
    K1_ARGS,
    emit_k1,
)
from .pre_attn import (
    SCRATCH_HDONE as K1_SCRATCH_HDONE,
)
from .pre_attn import (
    SCRATCH_RDONE as K1_SCRATCH_RDONE,
)
from .stages.attention import (
    merge_defs,
    o_defs,
    split_defs,
)
from .stages.ffn_reduce import (
    ffn_reduce_defs,
)
from .stages.moe_common import (
    moe_common_defs,
)
from .stages.moe_narrow import (
    moe_narrow_defs,
)
from .stages.moe_wide import (
    moe_wide_defs,
)
from .stages.router import (
    router_defs,
)
from .stages.select import (
    select_defs,
)
from .stages.shared import (
    K1_STAMPS,
    TL_POINTS,
)

# The launch goes on the caller's current stream (graph capture included).
_CURRENT_STREAM = fx.Stream(None)


# K4's arguments, in order: the kernel's and the launcher's parameters are checked
# against it at build time, and the runner packs its arguments by these names
K4_ABI = KernelAbi(
    (
        "h_in", "q", "block_table", "seq_lens", "k_cache", "v_cache", "k_scale",
        "v_scale", "w_o", "s_o", "g_post", "w_gate", "bias", "w13", "s13", "w2", "s2",
        "h_mid", "ar_out", "scratch", "sym", "peers", "rank", "layer", "bt_width",
        "q_len", "tl", "k1_args", "positions", "slot_mapping", "res", "batch_ids",
        "index_block_table", "index_bt_width", "index_slot_mapping", "ar_in", "mode",
    )
)  # fmt: skip


@dataclass(frozen=True)
class K4Build:
    """Every parameter of a K4 build (``support.plan.build_key``: all of them
    reach the JIT cache key); the ``sym`` ones name the kernel symbol."""

    npes: int = field(metadata={"sym": "tp"})
    tokens: int = field(metadata={"sym": "s"})
    init_blocks: int = field(metadata={"sym": "ib"})
    local_blocks: int = field(metadata={"sym": "lb"})
    index_heads: int = field(metadata={"sym": "ih"})
    index_own: int = field(metadata={"sym": "io"})
    timeline: bool = field(metadata={"sym": "tl"})
    index_topk: bool = field(metadata={"sym": "it"})
    fuse_k1: bool
    sm_scale: float
    eps: float
    route_scale: float
    shared_weight: float
    swiglu_limit: float
    debug: bool


def build_post_attn_kernel(
    npes: int,
    sm_scale: float,
    eps: float,
    route_scale: float,
    shared_weight: float,
    swiglu_limit: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    timeline: bool = False,
    fuse_k1: bool = False,
    heads: IndexHeads = ONE_INDEX_HEAD,
    debug: bool = False,
    index_topk: bool = True,
):
    """``@flyc.jit`` launcher of K4 for one rank of an ``npes``-way TP group and a
    decode step of ``tokens`` (<= MAX_TOKENS) rows, ``q_len`` consecutive rows a
    request (a speculative verify's), with contexts up to MAX_CONTEXT.
    ``fuse_k1``: the layer's K1 runs first in the same launch (``K1_ARGS``,
    ``positions``, ``slot_mapping``, ``res``; 0 otherwise), for a fused projection
    of index q ``heads`` (``IndexHeads``). ``debug``: every mailbox wait bounded,
    a wait that gives up recorded in scratch ``diag`` (``Mailbox``).
    ``index_topk``: the layer scores and selects its blocks; else (the original
    path's ``skip_index_topk`` layer) it attends over the sparse table the last
    selecting layer of the step left in scratch."""
    assert 1 <= tokens <= MAX_TOKENS
    key = K4Build(
        npes=npes, tokens=tokens, init_blocks=init_blocks, local_blocks=local_blocks,
        index_heads=heads.count, index_own=heads.own, timeline=timeline,
        index_topk=index_topk, fuse_k1=fuse_k1, sm_scale=sm_scale, eps=eps,
        route_scale=route_scale, shared_weight=shared_weight,
        swiglu_limit=swiglu_limit, debug=debug,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)  # the launcher references it: keyed
    # a debug build's wait records name a region by its ``region_id``
    REGION_IDS = region_ids(
        d.name for d in mailbox_regions(tokens, heads.count, fuse_k1, index_topk)
    )
    W = npes
    G = BLOCKS
    BASE = stage_bases(tokens)
    SY = sym_layout(W)
    # fuse_k1: K1's outputs are read device-coherent (K1 stores them so)
    CM_K1 = CM_DEV if fuse_k1 else 0
    WIDE = wide_moe(tokens)
    POOL = pool_words(tokens)
    U_MAX = TOP_K * tokens + 1  # distinct routed experts at most, then the shared

    @fx.struct
    class Smem:
        # activations, bf16 pairs / fp8 quads: every token's xn, o_proj input or
        # mid; wide: also the split's PV partials (the stages take turns)
        x: fx.Array[fx.Int32, POOL, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        p8: fx.Array[fx.Int32, H * SPLIT_KEYS // 4, 16]
        q8: fx.Array[fx.Int32, O_K // 4, 16]
        opart: fx.Array[fx.Float32, 1 if WIDE else WAVES * O_K, 16]  # PV partials
        # wide MoE: the step's distinct experts (the shared one last), token masks
        uexp: fx.Array[fx.Int32, U_MAX + 2, 16]
        umask: fx.Array[fx.Int32, U_MAX + 2, 16]
        # and per (expert, B column) the down stage's mid row and route weight
        urow: fx.Array[fx.Int32, U_MAX * 16 if WIDE else 1, 16]
        uwt: fx.Array[fx.Float32, U_MAX * 16 if WIDE else 1, 16]
        hst: fx.Array[
            fx.Float32, 3 * WAVES * H + WAVES, 16
        ]  # head max / sum, max, v_scale max
        misc: fx.Array[fx.Float32, O_ROWS * MAX_TOKENS, 16]  # a task's output rows
        route: fx.Array[fx.Int32, 8 * MAX_TOKENS, 16]  # token k's slots at 8 k
        rwt: fx.Array[fx.Float32, 8 * MAX_TOKENS, 16]  # and their routing weights
        blk: fx.Array[fx.Int32, 32, 16]  # this split's 16 pages, n_ctx, tail flag
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]  # timeline stamps
        # fuse_k1: index q (every head's: in the pool)
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens if heads.count == 1 else 1, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_mono_layer" if fuse_k1 else "minimax_m3_post_attn",
        **symbol_params(key),
    )

    # the kernel body rebuilds IndexHeads from ints: an object in a closure is not
    # a cache-key value (``build_key`` keys them anyway)
    ih_count, ih_own = heads.count, heads.own
    # fuse_k1 with every index head: K1's index q of every head (qs) after its x8
    # rows in the pool (K1 is done with the pool before K4's stages use it)
    QS_CP = tokens * HIDDEN // 4
    assert not (fuse_k1 and heads.count > 1) or (
        QS_CP + HEAD_DIM // 2 * tokens * heads.count <= POOL
    )

    # every builder name, for the stage factories (``stages``)
    _bc = dict(locals())

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def post_attn_kernel(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        batch_ids: Int64,
        index_block_table: Int64,
        index_bt_width: Int32,
        index_slot_mapping: Int64,
        ar_in: Int64,
        mode: Int32,
    ):
        if mode != 0:
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            g4 = lane // 16
            l16 = lane % 16
            lds = fx.SharedAllocator().allocate(Smem).peek()
            xs = lds.x.ptr
            red = lds.red.ptr
            p8 = lds.p8.ptr
            q8 = lds.q8.ptr
            # the pool read and written as f32 (wide: the split's PV partials live
            # there)
            fpool = fx.recast_iter(fx.Float32, xs)
            opart = fpool if WIDE else lds.opart.ptr
            uexp = lds.uexp.ptr
            umask = lds.umask.ptr
            urow = lds.urow.ptr
            uwt = lds.uwt.ptr
            hst = lds.hst.ptr
            misc = lds.misc.ptr
            route = lds.route.ptr
            rwt = lds.rwt.ptr
            # the selection's sort keys live in the pool: K1 is done with it (its
            # barrier), and a split task's attention fills it only once the task's
            # blocks are placed
            keys = xs
            blk = lds.blk.ptr
            tls = lds.tls.ptr
            v4f = fx.Vector.make_type(4, fx.Float32)
            v2i = fx.Vector.make_type(2, fx.Int32)

            diag = scratch + fx.Int64(SCRATCH["diag"]) if debug else None
            mbox = Mailbox(layer, diag, REGION_IDS)
            # Plain names, not method calls: flydsl's AST rewriter carries every local
            # whose method is called inside a dynamic if/for as loop state.
            mb_put = mbox.put
            mb_put_bf = mbox.put_bf
            mb_put_words = mbox.put_words
            mb_poll = mbox.poll

            def mb(name):
                """Scratch region ``name``'s raw address (plain loads / stores)."""
                return scratch + fx.Int64(SCRATCH[name])

            def mr(name):
                """Scratch mailbox region ``name`` (puts / polls)."""
                return sreg(scratch, SCRATCH[name], name)

            def acquire():
                fx.memory_fence(
                    syncscope=rocdl.SyncScope.Workgroup,
                    ordering=fx.AtomicOrdering.Acquire,
                )

            def scores_copy(c, k=0):
                """Copy c of token k's router outputs: expert e's (routing key, sigmoid)
                at pairs 2 e, 2 e + 1."""
                return shift(
                    mr("scores"),
                    (fx.Int64(c) * MAX_TOKENS + fx.Int64(k)) * (N_ROUTED * 16),
                )

            def route_key(v, e):
                """Signed-orderable i32 of score ``v`` with its low 7 bits replaced by
                127 - expert ``e``: a pick is one integer max, a tie goes to the lower
                expert."""
                b = v.bitcast(fx.Int32)
                b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
                return (b & fx.Int32(-128)) | (127 - e)

            # each wave pushes to one peer
            peer_dst = load_ptr64(peers, fx.min(wave, W - 1))
            AG_RS = tokens >= AG_RS_FROM

            def peer_addr(w):
                return load_ptr64(peers, w)

            def start(name):
                return first_task(bid, BASE[name])

            def stamp(k):
                """Record this CTA passing timeline point k."""
                stamp_point(timeline, tls, tid, k)

            def block_max(v):
                return block_max_of(v, lane, wave, red)

            def push_rows(region, row0, n_rows, only=None):
                """Push this task's bf16 partial rows from misc to
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
                            [
                                fx.ptr_load(misc + i * 2),
                                fx.ptr_load(misc + (i * 2 + 1)),
                            ],
                        )

            stamp_begin(timeline, tls, tid, TL_POINTS)

            if const_expr(fuse_k1):
                # this layer's K1 first: its GEMV / head / norm tasks and the scores
                def k1_stamp(k):
                    """K1's timeline points 1-4 and 6 on K4's free stamps 27-31."""
                    if const_expr(k in K1_STAMPS):
                        stamp(K1_STAMPS[k])

                (
                    g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik,
                    cos_sin, index_cache,
                    iq_out, scratch1,
                ) = [load_ptr64(k1_args, i) for i in range(len(K1_ARGS))]  # fmt: skip
                qs_k1 = lds.qs.ptr
                if const_expr(ih_count > 1):
                    qs_k1 = fpool + QS_CP
                emit_k1(
                    tid, bid, lane, wave, xs, red, qs_k1, k1_stamp, layer,
                    eps, index_scale_log2e(sm_scale), tokens, q_len, init_blocks,
                    local_blocks, IndexHeads(ih_count, ih_own),
                    (
                        ar_in, res, h_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq,
                        g_ik, cos_sin,
                        positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
                        index_cache, q, iq_out, block_table, seq_lens,
                        mb("iscore"), scratch1,
                    ),
                    bt_width,
                    True,
                    diag,
                    REGION_IDS,
                    index_topk,
                    plain_norm=True,
                    index_block_table=index_block_table,
                    index_bt_width=index_bt_width,
                    index_slot_mapping=index_slot_mapping,
                    scalar_kv=True,
                )  # fmt: skip
                gpu.barrier()
                k1_mbox = Mailbox(layer, diag, REGION_IDS)
                k1_poll: object = k1_mbox.poll
                hdone_mb = sreg(scratch1, K1_SCRATCH_HDONE, "k1.hdone")
                rdone_mb = sreg(scratch1, K1_SCRATCH_RDONE, "k1.rdone")
            else:
                # K1's hand-offs: the stage factories bind every name they read, used
                # only under fuse_k1 or not (``stages``)
                k1_poll = hdone_mb = rdone_mb = None

            # ============================================================ 0. index
            # Indexer: K1 scored every block the selection does not pin (index_score);
            # here only the selection runs.
            # a graph's pad rows carry seq_len 0 and an all-zero block-table row: as one
            # key of page 0 they stay finite and select a real page
            seq_lens_k, n_blks, long_rows = step_rows(
                seq_lens, tokens, q_len, IndexHeads(ih_count, ih_own)
            )
            # Up to TOPK_BLOCKS blocks the top-k keeps every block and the scores would
            # only order them. The original selector's own contract calls either order
            # equally correct (index_topk._pack_score_key), so a short context is not
            # scored and takes the blocks in id order -- the tail block is last either
            # way -- instead of putting the scoring chain in front of the attention.
            scored = [nb > TOPK_BLOCKS for nb in n_blks]

            # every name the stage factories read: the builder's, the body's so far,
            # and the functions the factories define
            k4ctx = dict(_bc)
            k4ctx.update(locals())
            k4ctx.update(
                {
                    "h_in": h_in,
                    "q": q,
                    "block_table": block_table,
                    "seq_lens": seq_lens,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "k_scale": k_scale,
                    "v_scale": v_scale,
                    "w_o": w_o,
                    "s_o": s_o,
                    "g_post": g_post,
                    "w_gate": w_gate,
                    "bias": bias,
                    "w13": w13,
                    "s13": s13,
                    "w2": w2,
                    "s2": s2,
                    "h_mid": h_mid,
                    "ar_out": ar_out,
                    "scratch": scratch,
                    "sym": sym,
                    "peers": peers,
                    "rank": rank,
                    "layer": layer,
                    "bt_width": bt_width,
                    "q_len": q_len,
                    "tl": tl,
                    "k1_args": k1_args,
                    "positions": positions,
                    "slot_mapping": slot_mapping,
                    "res": res,
                    "batch_ids": batch_ids,
                    "index_block_table": index_block_table,
                    "index_bt_width": index_bt_width,
                    "index_slot_mapping": index_slot_mapping,
                }
            )
            k4ctx.update(select_defs(k4ctx))
            stage_select_long = k4ctx["stage_select_long"]

            k4ctx.update(locals())
            k4ctx.update(split_defs(k4ctx))
            stage_split = k4ctx["stage_split"]

            if const_expr(index_topk):
                enter_stage("select_long")
                stage_select_long()
            enter_stage("split")
            stage_split()
            stamp(1)

            k4ctx.update(locals())
            k4ctx.update(merge_defs(k4ctx))
            stage_merge = k4ctx["stage_merge"]

            enter_stage("merge")
            stage_merge()

            k4ctx.update(locals())
            k4ctx.update(o_defs(k4ctx))
            stage_o = k4ctx["stage_o"]

            enter_stage("o")
            stage_o()
            stamp(4)

            # a8w4 operands in xs (Int32 words, never through f32 arithmetic): the up /
            # gate stage holds every token's MXFP8 xn then its E8M0 scales; the down
            # stage every token's MXFP8 mid then its scales
            XN8_WORDS = HIDDEN // 4
            XSC_WORDS = HIDDEN // 32
            XSC_LDS = tokens * XN8_ROW
            MID16_WORDS = MOE_SLOTS * INTER // 2  # bf16 pairs, as the mids land
            MID8_WORDS = MOE_SLOTS * INTER // 4
            MID8_LDS = tokens * MID16_WORDS
            MIDSC_WORDS = MOE_SLOTS * INTER // 32
            MIDSC_LDS = MID8_LDS + tokens * MID8_WORDS

            k4ctx.update(locals())
            k4ctx.update(router_defs(k4ctx))
            stage_router = k4ctx["stage_router"]

            enter_stage("router")
            stage_router()
            stamp(5)

            # ================================ 5. experts: up / gate -> down -> FFN
            # reduce
            UG_KC = HIDDEN // 128  # 128-k chunks
            UG_CPW = UG_KC // WAVES  # 128-k chunks per wave: each wave a k eighth
            DN_KC = INTER // 128  # 128-k chunks of one slot
            DN_UNITS = TOP_K * DN_KC  # routed (slot, 128-k chunk) units per row group
            DN_UPW = (DN_UNITS + 3) // 4  # 4 waves per row group
            SH_UPW = (DN_KC + 3) // 4  # the shared expert's chunks, once for all tokens
            # routed (token, slot, row group) tasks t < ROUTED_TASKS; the shared
            # expert's
            # row groups (one GEMV with a column per token) are tasks ROUTED_TASKS + j,
            # run as task j of stage "shared" while the routing is still on its way, so
            # the routed tasks alone set the rounds (S = 4: 768 = 3 x 256)
            ROUTED_TASKS = TOP_K * UG_PER_SLOT * tokens
            TPT = TOP_K * UG_PER_SLOT  # routed tasks per token
            CPT = ug_ctas_per_token(tokens)
            UG_ITERS = max(ug_tasks_of(c, tokens) for c in range(G))  # per CTA, at most

            k4ctx.update(locals())
            k4ctx.update(moe_common_defs(k4ctx))
            k4ctx.update(ffn_reduce_defs(k4ctx))

            # ======================================== 5w. experts, wide (S > 4)
            # LDS pool words: MXFP8 xn, its scales, the up partials, router sigmoids;
            # the
            # down stage reuses the pool for MXFP8 mid rows (token k's slot s at row
            # k MOE_SLOTS + s, a zero row last) and their scales
            W_UPP = tokens * (XN8_ROW + XSC_ROW)
            W_SIG = W_UPP + WAVES * 64 * 4
            MID_ROWS = MOE_SLOTS * tokens  # the zero row's index
            MID_ROW_WORDS = INTER // 4
            MIDSC_ROW_WORDS = INTER // 32
            W_MIDSC = (MID_ROWS + 1) * MID_ROW
            # up / gate tasks go in pairs, a CTA runs both: an MXFP8 block is 32 rows
            UG_PAIRS = UG_PER_SLOT // 2
            # Up / gate tasks per CTA.
            UGW_ITERS = 2 * ((U_MAX * UG_PAIRS + G - 1) // G)
            # a wave takes chunks q and q + 4 of an expert (``wide_down``)
            assert 4 < DN_KC <= 8
            DN_WITERS = 2 * U_MAX  # (expert, chunk) units a wave runs, at most
            DN_DEPTH = 12  # units whose weights are in flight
            ROUTE_ITERS = (tokens + WAVES - 1) // WAVES  # tokens a wave routes

            k4ctx.update(locals())
            k4ctx.update(moe_wide_defs(k4ctx))
            stage_moe_wide = k4ctx["stage_moe_wide"]

            k4ctx.update(locals())
            k4ctx.update(moe_narrow_defs(k4ctx))
            stage_moe = k4ctx["stage_moe"]

            # routing, shared + routed up / gate ("ug"); each CTA's down task enters
            # "down" (moe_down / wide_down)
            enter_stage("ug")
            (stage_moe_wide if WIDE else stage_moe)()
            stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)

    @launcher(kernel_name)
    def launch_post_attn(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        batch_ids: Int64,
        index_block_table: Int64,
        index_bt_width: Int32,
        index_slot_mapping: Int64,
        ar_in: Int64,
        mode: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key  # every build parameter in the JIT cache key
        post_attn_kernel(
            h_in,
            q,
            block_table,
            seq_lens,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            w_o,
            s_o,
            g_post,
            w_gate,
            bias,
            w13,
            s13,
            w2,
            s2,
            h_mid,
            ar_out,
            scratch,
            sym,
            peers,
            rank,
            layer,
            bt_width,
            q_len,
            tl,
            k1_args,
            positions,
            slot_mapping,
            res,
            batch_ids,
            index_block_table,
            index_bt_width,
            index_slot_mapping,
            ar_in,
            mode,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    K4_ABI.check(post_attn_kernel, launch_post_attn)
    return launch_post_attn
