# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
            -> acc = bf16(sum) + h   (h_mid = bf16(acc), mailbox ``a`` = acc)
    router: gemma RMSNorm(acc) in the fused all-reduce's reduction order -> bf16
            gate GEMV -> logits; publishes its slice of the normalized input
    moe   : sigmoid + bias top-4 (+ the fused shared expert) -> MXFP4 a16w4
            up/gate on aiter's shuffled layout -> swiglu -> mid (bf16);
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
CTAs are co-resident, so every spin wait makes progress.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from vllm.models.minimax_m3.amd.mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    INDEX_CP_FROM_BLOCKS,
    INTER,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    PAGE16,
    PAGE16_SIDES,
    SHARED_EXPERT,
    SPARSE_BLOCK,
    THREADS,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    WAVES,
    IndexHeads,
)
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    CM_SYS,
    Mailbox,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    kv_cache_scale,
    memrealtime,
    permlane_swap,
    rsrc,
    uniform,
    unlikely,
    wave_sum,
    xshfl,
)
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    block_max as block_max_of,
)
from vllm.models.minimax_m3.amd.mono.kernels.index_score import (
    index_scale_log2e,
    step_rows,
)
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import (
    K1_ARGS,
    emit_k1,
)
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import (
    N_HEAD_TASKS as K1_HEAD_TASKS,
)
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import (
    SCRATCH_HDONE as K1_SCRATCH_HDONE,
)
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import (
    SCRATCH_RDONE as K1_SCRATCH_RDONE,
)
from vllm.models.minimax_m3.amd.mono.layout import (
    AG_RS_FROM,
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_O,
    N_ROUTER,
    N_SPLIT,
    O_K,
    O_ROW,
    O_ROWS,
    PAGES_PER_BLOCK,
    SCORE_COPIES,
    SCRATCH,
    SH_ROWS,
    SH_TASKS,
    SPARSE_TABLE_WORDS,
    SPLIT_KEYS,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
    H,
    pool_words,
    shared_after_routing,
    stage_bases,
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)

FP8_MAX = 448.0
NEG = -3.4e38  # the gluon decode's masked-score value
PAGE_BYTES = PAGE16 * HEAD_DIM
XN_SLICE = HIDDEN // N_ROUTER  # normalized-input elements each router task publishes
W13_ROWS = 2 * INTER  # gate rows then up rows (GGUU)
W13_BYTES = W13_ROWS * HIDDEN // 2  # one expert, fp4
W2_BYTES = HIDDEN * INTER // 2
W13_SCALE_COLS = HIDDEN // 32
W2_SCALE_COLS = INTER // 32
S2_BYTES = (N_ROUTED + 1) * HIDDEN * W2_SCALE_COLS  # e8m0, every expert
S13_BYTES = (N_ROUTED + 1) * W13_ROWS * W13_SCALE_COLS

# The launch goes on the caller's current stream (graph capture included).
_CURRENT_STREAM = fx.Stream(None)


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


TL_POINTS = 32  # timeline stamps per CTA
K1_STAMPS = {1: 27, 2: 28, 3: 29, 4: 30, 6: 31}  # fuse_k1: K1's points -> K4's
RANK_SPLIT = 8  # threads ranking one indexer block
RANK_LANES = (4, 2, 1)  # butterfly over those RANK_SPLIT lanes
RANK_CANDS = THREADS // (WAVES * TOPK_BLOCKS)  # threads ranking one candidate
RANK_CAND_LANES = (2, 1)  # butterfly over those RANK_CANDS lanes
KEY_NONE = -(2**31)  # the sort key below every block's
# blocks a thread keys in its split task's eighth of a context (<= MAX_INDEX_BLOCKS)
CAND_BATCH = 2
assert N_SPLIT * THREADS * CAND_BATCH >= MAX_INDEX_BLOCKS
# Indexer context parallelism: a long context's split task p scans index head
# p // 2's scores of half p % 2 of this rank's blocks -- as many as a TP split task
# scans -- for the head's rank; a request is long past INDEX_CP_FROM_BLOCKS blocks
# (``step_rows``; every long request context-parallel: no TP long selection in that
# build), its rows ending less than a block apart are >= INDEX_CP_FROM_BLOCKS blocks
# each, and a split task then scans >= TOPK_BLOCKS blocks
assert N_SPLIT == 2 * TP
assert N_SPLIT * TOPK_BLOCKS <= INDEX_CP_FROM_BLOCKS <= THREADS
assert MAX_TOKENS <= SPARSE_BLOCK


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``shuffle_scale`` layout:
    ``view(rows/32, 2, 16, cols/8, 2, 4).permute(0, 3, 5, 2, 4, 1)``."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


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
):
    """``@flyc.jit`` launcher of K4 for one rank of an ``npes``-way TP group and a
    decode step of ``tokens`` (<= MAX_TOKENS) rows, ``q_len`` consecutive rows a
    request (a speculative verify's), with contexts up to MAX_CONTEXT.
    ``fuse_k1``: the layer's K1 runs first in the same launch (``K1_ARGS``,
    ``positions``, ``slot_mapping``, ``res``; 0 otherwise), for a fused projection
    of index q ``heads`` (``IndexHeads``)."""
    assert 1 <= tokens <= MAX_TOKENS
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
        hst: fx.Array[fx.Float32, 3 * WAVES * H, 16]  # head max / sum, max
        misc: fx.Array[fx.Float32, O_ROWS * MAX_TOKENS, 16]  # a task's output rows
        route: fx.Array[fx.Int32, 8 * MAX_TOKENS, 16]  # token k's slots at 8 k
        rwt: fx.Array[fx.Float32, 8 * MAX_TOKENS, 16]  # and their routing weights
        # indexer sort keys, a block a thread (a long request: the waves'
        # candidate (key, block) pairs) and their ranks
        keys: fx.Array[fx.Int32, THREADS, 16]
        ranks: fx.Array[fx.Int32, THREADS, 16]
        blk: fx.Array[fx.Int32, 32, 16]  # this split's 16 pages, n_ctx, tail flag
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]  # timeline stamps
        # fuse_k1: index q (every head's: in the pool)
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens if heads.count == 1 else 1, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_mono_layer" if fuse_k1 else "minimax_m3_post_attn",
        tp=npes,
        s=tokens,
        ib=init_blocks,
        lb=local_blocks,
        ih=heads.count,
        io=heads.own,
        tl=timeline,
    )

    # the JIT cache key holds the kernel's scalar closure values, not objects: the
    # index heads go in as ints (a shared cache otherwise serves one rank's build
    # to every rank)
    ih_count, ih_own = heads.count, heads.own
    # fuse_k1 with every index head: K1's index q of every head (qs) after its x8
    # rows in the pool (K1 is done with the pool before K4's stages use it)
    QS_CP = tokens * HIDDEN // 4
    assert not (fuse_k1 and heads.count > 1) or (
        QS_CP + HEAD_DIM // 2 * tokens * heads.count <= POOL
    )

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
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
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
        p8 = lds.p8.ptr
        q8 = lds.q8.ptr
        # the pool read and written as f32 (wide: the split's PV partials live there)
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
        keys = lds.keys.ptr
        ranks = lds.ranks.ptr
        blk = lds.blk.ptr
        tls = lds.tls.ptr
        v4f = fx.Vector.make_type(4, fx.Float32)
        v2i = fx.Vector.make_type(2, fx.Int32)

        mbox = Mailbox(scratch, step, layer)
        # Plain names, not method calls: flydsl's AST rewriter carries every local
        # whose method is called inside a dynamic if/for as loop state.
        mb_put = mbox.put
        mb_put_bf = mbox.put_bf
        mb_put_words = mbox.put_words
        mb_poll = mbox.poll

        def mb(name):
            return scratch + fx.Int64(SCRATCH[name])

        def acquire():
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )

        def scores_copy(c, k=0):
            """Copy c of token k's router outputs: expert e's (routing key, sigmoid)
            at pairs 2 e, 2 e + 1."""
            return mb("scores") + (fx.Int64(c) * MAX_TOKENS + fx.Int64(k)) * (
                N_ROUTED * 16
            )

        def route_key(v, e):
            """Signed-orderable i32 of score ``v`` with its low 7 bits replaced by
            127 - expert ``e``: a pick is one integer max, a tie goes to the lower
            expert."""
            b = v.bitcast(fx.Int32)
            b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
            return (b & fx.Int32(-128)) | (127 - e)

        # each wave pushes to one peer
        pv = fx.Vector(
            bo.buffer_load(
                rsrc(peers), fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32
            )
        )
        peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(uniform(pv[0]))
        )
        AG_RS = tokens >= AG_RS_FROM

        def peer_addr(w):
            pw = fx.Vector(bo.buffer_load(rsrc(peers), w * 2, vec_width=2, dtype=T.i32))
            return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(
                fx.Uint32(uniform(pw[0]))
            )

        def start(name):
            return (bid + (G - BASE[name])) % G

        def stamp(k):
            """``timeline``: s_memrealtime (100 MHz) of this CTA passing point k, kept
            in LDS until the kernel ends -- a global store per stamp would put its
            write latency on every later ``s_waitcnt vmcnt(0)``."""
            # compile-time gate outside, traced condition inside: they cannot be
            # one `and`
            if const_expr(timeline):  # noqa: SIM102
                if tid == 0:
                    fx.ptr_store(memrealtime(), tls + k)

        def flush_stamps():
            if const_expr(timeline):  # noqa: SIM102
                if tid < TL_POINTS:
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(
                                fx.Int64.ir_type, fx.AddressSpace.Global, 8
                            ),
                            tl + fx.Int64((bid * TL_POINTS + tid) * 8),
                        ),
                        fx.ptr_load(tls + tid),
                    )

        def ld_bf16x4(r, k, cm=0):
            w = fx.Vector(
                bo.buffer_load(r, k // 2, vec_width=2, dtype=T.i32, cache_modifier=cm)
            )
            v = w.bitcast(fx.BFloat16).to(fx.Float32)
            return [v[j] for j in range(4)]

        def mfma_fp8(a64, b64, c):
            return fx.Vector(
                rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a64, b64, c, 0, 0, 0])
            )

        def block_max(v):
            return block_max_of(v, lane, wave, red)

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
                        peer_dst + fx.Int64(SY[region]),
                        (rank * MAX_TOKENS + i // half) * HIDDEN
                        + row0
                        + (i % half) * 2,
                        [fx.ptr_load(misc + i * 2), fx.ptr_load(misc + (i * 2 + 1))],
                        CM_SYS,
                    )

        if const_expr(timeline):  # noqa: SIM102
            if tid < TL_POINTS:  # same wave as the stamping thread: ordered
                fx.ptr_store(fx.Int64(0), tls + tid)
        stamp(0)

        if const_expr(fuse_k1):
            # this layer's K1 first: its GEMV / head / norm tasks and the scores
            def k1_ptr(i):
                v = fx.Vector(
                    bo.buffer_load(rsrc(k1_args), 2 * i, vec_width=2, dtype=T.i32)
                )
                return (fx.Int64(uniform(v[1])) << 32) | fx.Int64(
                    fx.Uint32(uniform(v[0]))
                )

            def k1_stamp(k):
                """K1's timeline points 1-4 and 6 on K4's free stamps 27-31."""
                if const_expr(k in K1_STAMPS):
                    stamp(K1_STAMPS[k])

            (
                ar_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, index_cache,
                iq_out, scratch1,
            ) = [k1_ptr(i) for i in range(len(K1_ARGS))]  # fmt: skip
            qs_k1 = lds.qs.ptr
            if const_expr(ih_count > 1):
                qs_k1 = fpool + QS_CP
            emit_k1(
                tid, bid, lane, wave, xs, red, qs_k1, k1_stamp, step, layer,
                eps, index_scale_log2e(sm_scale), tokens, q_len, init_blocks,
                local_blocks, IndexHeads(ih_count, ih_own),
                (
                    ar_in, res, h_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
                    positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
                    index_cache, q, iq_out, block_table, seq_lens, mb("iscore"),
                    scratch1,
                ),
                bt_width,
                True,
            )  # fmt: skip
            gpu.barrier()
            k1_mbox = Mailbox(scratch1, step, layer)
            k1_poll = k1_mbox.poll
            hdone_mb = k1_mbox.addr(K1_SCRATCH_HDONE)
            rdone_mb = k1_mbox.addr(K1_SCRATCH_RDONE)

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

        def bt_row(k):
            """Token k's block-table row (rows are ``bt_width`` apart)."""
            return block_table + fx.Int64(k) * fx.Int64(bt_width) * 4

        def pick(vals, k):
            """vals[k] for a traced token index k."""
            v = vals[0]
            for i in range_constexpr(1, tokens):
                v = (k == i).select(vals[i], v)
            return v

        def index_key(b, word, n_blk):
            """Sort key of block b's raw score word: forced blocks pinned, then a
            signed int ordered like the float (-0 == 0, NaN below -inf)."""
            sc = word.bitcast(fx.Float32)
            sc = (b < init_blocks).select(fx.Float32(1e30), sc)
            sc = (b >= n_blk - local_blocks).select(fx.Float32(1e29), sc)
            bits = sc.bitcast(fx.Int32)
            bits = (bits == fx.Int32(-(2**31))).select(fx.Int32(0), bits)
            key = (bits < 0).select(bits ^ 0x7FFFFFFF, bits)
            return ((bits & 0x7FFFFFFF) > 0x7F800000).select(fx.Int32(-(2**31)), key)

        def rank_blocks(k, iscore, n_blk, r_bt, scored_k):
            """Up to THREADS blocks: thread b keys block b, and a block's rank
            counts the blocks that beat it. -> this thread's (key, block, rank,
            live, the tail block's key, page). ``scored_k``: the scores are read
            (else the blocks rank by id)."""
            b = tid
            live = b < n_blk
            bb = fx.min(b, n_blk - 1)
            # the table is logical: a block's first page id is PAGE16_SIDES wider
            # than its id when both K/V sides live in the block
            page = PAGE16_SIDES * fx.Int32(
                bo.buffer_load(r_bt, bb, vec_width=1, dtype=T.i32)
            )
            # K1 (or the harnesses' scorer) wrote every score before this launch, so
            # a plain load reads them: it goes out before the context length is in.
            # A pinned block's slot holds no score (index_key forces its key); past
            # n_blk the keys are never read
            if const_expr(fuse_k1):
                # the scorers are CTAs of this launch: poll (a pinned block's slot
                # polls a scored block's)
                word = fx.Int32(0)
                if scored_k:
                    word = mb_poll(
                        [
                            (
                                iscore,
                                fx.min(
                                    fx.max(tid, init_blocks),
                                    n_blk - 1 - local_blocks,
                                ),
                                1,
                            )
                        ]
                    )[0][0]
            else:
                word = fx.Int32(
                    bo.buffer_load(rsrc(iscore), 2 * tid, vec_width=1, dtype=T.i32)
                )
            fx.ptr_store(scored_k.select(index_key(tid, word, n_blk), -tid), keys + tid)
            stamp(18)
            gpu.barrier()
            stamp(20)
            # rank = the keys that beat it. RANK_SPLIT threads share a block, each
            # comparing every RANK_SPLIT-th key: one lane walking all 61 keys of a
            # 7.8k context took 1.9 us (a serial compare -> SGPR mask chain)
            part = tid % RANK_SPLIT
            for b0 in range(0, n_blk, THREADS // RANK_SPLIT):
                rb = fx.Int32(b0) + tid // RANK_SPLIT
                rbc = fx.min(rb, n_blk - 1)
                kb = fx.ptr_load(keys + rbc)
                cnt = fx.Int32(0)
                for j0 in range(0, n_blk, RANK_SPLIT):
                    j = fx.Int32(j0) + part
                    kj = fx.ptr_load(keys + fx.min(j, n_blk - 1))
                    beats = (kj > kb) | ((kj == kb) & (j > rbc))
                    cnt = cnt + (beats & (j < n_blk)).select(fx.Int32(1), fx.Int32(0))
                cnt = butterfly(cnt, RANK_LANES)
                if (part == 0) & (rb < n_blk):
                    fx.ptr_store(cnt, ranks + rb)
            gpu.barrier()
            key = fx.ptr_load(keys + bb)
            rank = fx.ptr_load(ranks + bb)
            stamp(19)
            kt = fx.ptr_load(keys + (n_blk - 1))
            return key, bb, rank, live, kt, page

        def ballot(pred):
            return fx.Int64(rocdl.ballot(T.i64, pred))

        def beats(ah, al, bh, bl):
            """(key, block) pair a orders before b."""
            return (ah > bh) | ((ah == bh) & (al > bl))

        def pair_best(bh, bl, off):
            """The better (key, block) pair of this lane's and lane ^ off's; offsets
            32 / 16 by v_permlane*_swap (no LDS round trip)."""
            if const_expr(off < 16):
                oh, ol = xshfl(bh, off), xshfl(bl, off)
                up = beats(oh, ol, bh, bl)
                return up.select(oh, bh), up.select(ol, bl)
            ah, xh = permlane_swap(off, bh, bh)
            al, xl = permlane_swap(off, bl, bl)
            up = beats(xh, xl, ah, al)
            return up.select(xh, ah), up.select(xl, al)

        def wave_best(h0, l0, h1, l1):
            """The wave's TOPK_BLOCKS best of its lanes' two (key, block) pairs, best
            first in lanes 0..15: round r moves the best pair left to lane r and
            drops it from its holder. A runtime loop: unrolled, rare code like this
            raised the kernel's SGPR spills (S = 16, 3k context: +3 us)."""
            none_h, none_l = fx.Int32(KEY_NONE), fx.Int32(-1)
            rh, rl = none_h, none_l
            for r in range(TOPK_BLOCKS):
                up = beats(h1, l1, h0, l0)
                bh, bl = up.select(h1, h0), up.select(l1, l0)
                for off in (32, 16, 8, 4, 2, 1):
                    bh, bl = pair_best(bh, bl, off)
                rh = (lane == r).select(bh, rh)
                rl = (lane == r).select(bl, rl)
                gone = (h0 == bh) & (l0 == bl)
                h0, l0 = gone.select(none_h, h0), gone.select(none_l, l0)
                gone = (h1 == bh) & (l1 == bl)
                h1, l1 = gone.select(none_h, h1), gone.select(none_l, l1)
            return rh, rl

        def rank_cands():
            """Candidate tid / RANK_CANDS of the WAVES TOPK_BLOCKS (key, block)
            pairs in ``keys`` and its rank, the count of candidates that beat it ->
            (key, block, rank, held: this thread reports it and it is a block)."""
            c = tid // RANK_CANDS
            part = tid % RANK_CANDS
            ch = fx.ptr_load(keys + 2 * c)
            cl = fx.ptr_load(keys + (2 * c + 1))
            cnt = fx.Int32(0)
            for j0 in range_constexpr(0, WAVES * TOPK_BLOCKS, RANK_CANDS):
                j = part + j0
                up = beats(
                    fx.ptr_load(keys + 2 * j), fx.ptr_load(keys + (2 * j + 1)), ch, cl
                )
                cnt = cnt + up.select(fx.Int32(1), fx.Int32(0))
            cnt = butterfly(cnt, RANK_CAND_LANES)
            return ch, cl, cnt, (part == 0) & (cl >= 0)

        def place(key, bb, rank, live, kt, n_blk, seq_len, active):
            """The ranked blocks' slots, as the Triton selector emits them: full
            blocks by rank, the tail block (the one holding the current token)
            last. -> (selected, slot, sparse context length); ``active``: this
            CTA places (a uniform flag)."""
            n_sel = fx.min(n_blk, TOPK_BLOCKS)
            tail = n_blk - 1
            tail_first = (kt > key) | ((kt == key) & (tail > bb))
            sel = live & (rank < n_sel) & active
            is_tail = sel & (bb == tail)
            if is_tail:
                fx.ptr_store(fx.Int32(1), blk + 17)
            gpu.barrier()
            tail_sel = fx.ptr_load(blk + 17) != 0
            n_full = n_sel - tail_sel.select(fx.Int32(1), fx.Int32(0))
            slot = is_tail.select(
                n_full, rank - tail_first.select(fx.Int32(1), fx.Int32(0))
            )
            n_ctx = tail_sel.select(
                n_full * SPARSE_BLOCK + seq_len - tail * SPARSE_BLOCK,
                fx.min(n_sel * SPARSE_BLOCK, seq_len),
            )
            return sel, slot, n_ctx

        def table_of(k):
            return rsrc(mb("sparse_table") + fx.Int64(k) * (SPARSE_TABLE_WORDS * 4))

        def select_blocks(k, t):
            """blk[0:16] := the pages of selection slots 2t, 2t+1 and blk[16] := the
            sparse context length, exactly as the Triton selector emits them: top
            TOPK_BLOCKS by (score, block id) descending with the init blocks pinned
            to 1e30 and the local ones to 1e29, full blocks in that order and the
            tail block (the one holding the current token) last. A block's slot
            follows from its rank, the count of blocks that beat it. k: token.
            For a long request ``stage_select_long`` left blk: a one-block
            stand-in is ranked, placing nothing."""
            iscore = mb("iscore") + fx.Int64(k) * (MAX_INDEX_BLOCKS * 8)
            short = ~pick(long_rows, k)
            n_blk = short.select(pick(n_blks, k), fx.Int32(1))
            seq_len = pick(seq_lens_k, k)
            r_bt = rsrc(bt_row(k))
            table = table_of(k)
            if (tid < 2 * PAGES_PER_BLOCK + 2) & short:
                fx.ptr_store(fx.Int32(0), blk + tid)
            key, bb, rank, live, kt, page = rank_blocks(
                k, iscore, n_blk, r_bt, pick(scored, k) & short
            )
            sel, slot, n_ctx = place(key, bb, rank, live, kt, n_blk, seq_len, short)
            if sel & (slot // 2 == t):
                for j in range_constexpr(PAGES_PER_BLOCK):
                    fx.ptr_store(
                        page * PAGES_PER_BLOCK + j,
                        blk + ((slot % 2) * PAGES_PER_BLOCK + j),
                    )
            if sel & (t == 0):
                for j in range_constexpr(PAGES_PER_BLOCK):
                    bo.buffer_store(
                        page * PAGES_PER_BLOCK + j,
                        table,
                        slot * PAGES_PER_BLOCK + j,
                    )
            if (tid == 0) & short:
                fx.ptr_store(n_ctx, blk + 16)
                if t == 0:
                    bo.buffer_store(n_ctx, table, TOPK_BLOCKS * PAGES_PER_BLOCK)
            gpu.barrier()

        def select_long(k, part):
            """Split task (k, part) of a long request's token, before the
            split stage. Its share of the blocks (a thread keys CAND_BATCH) -> each
            wave's TOPK_BLOCKS best -> the CTA's, to a candidate mailbox; then the
            N_SPLIT shares' candidates (the context's TOPK_BLOCKS best are among
            them) are ranked and placed -> blk as ``select_blocks`` leaves it, and
            (part 0) the sparse table. The share: an eighth of the blocks, to this
            rank's sel_long; with every index head (indexer context parallelism)
            head part // 2's scores of half part % 2 of this rank's blocks, to the
            head's rank's sel_cp -- the shares its own head's candidates come in."""
            n_blk = pick(n_blks, k)
            iscore = mb("iscore") + fx.Int64(k) * (MAX_INDEX_BLOCKS * 8)
            first = k * (N_SPLIT * TOPK_BLOCKS)  # token k's candidates (pair pairs)
            table = table_of(k)
            if tid == 0:
                fx.ptr_store(fx.Int32(0), blk + 17)
            if const_expr(ih_count > 1):
                own0 = init_blocks + (ih_own - init_blocks) % TP  # this rank's first
                n_own = (n_blk - own0 + TP - 1) // TP
                # the last scored one: the local blocks' slots hold no score
                last = own0 + TP * ((n_blk - local_blocks - own0 + TP - 1) // TP - 1)
                span = (n_own + 1) // 2
                lo = (part % 2) * span
                hi = fx.min(lo + span, n_own)
                js = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
                blocks = [own0 + TP * j for j in js]
                live = [j < hi for j in js]
                src = iscore + fx.Int64(part // 2) * (MAX_INDEX_BLOCKS // TP * 8)
                slots = [fx.min(b, last) // TP for b in blocks]
                out, out_slot = peer_addr(part // 2) + fx.Int64(SY["sel_cp"]), 2 * rank
                out_slot = out_slot + part % 2
                got_from, got_scope = sym + fx.Int64(SY["sel_cp"]), "system"
            else:
                span = (n_blk + N_SPLIT - 1) // N_SPLIT
                lo = part * span
                hi = fx.min(lo + span, n_blk)
                blocks = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
                live = [b < hi for b in blocks]
                src = iscore
                slots = [
                    fx.min(fx.max(b, init_blocks), n_blk - 1 - local_blocks)
                    for b in blocks
                ]
                out, out_slot = mb("sel_long"), part
                got_from, got_scope = out, "agent"
            if const_expr(fuse_k1):
                words = [w[0] for w in mb_poll([(src, sl, 1) for sl in slots])]
            else:
                words = [
                    fx.Int32(
                        bo.buffer_load(rsrc(src), 2 * sl, vec_width=1, dtype=T.i32)
                    )
                    for sl in slots
                ]
            hs = [
                lv.select(index_key(b, w, n_blk), fx.Int32(KEY_NONE))
                for b, w, lv in zip(blocks, words, live)
            ]
            ls = [lv.select(b, fx.Int32(-1)) for b, lv in zip(blocks, live)]
            stamp(18)
            rh, rl = wave_best(hs[0], ls[0], hs[1], ls[1])
            if lane < TOPK_BLOCKS:
                fx.ptr_store(rh, keys + 2 * (wave * TOPK_BLOCKS + lane))
                fx.ptr_store(rl, keys + (2 * (wave * TOPK_BLOCKS + lane) + 1))
            gpu.barrier()
            ch, cl, cnt, held = rank_cands()
            # a share holds >= TOPK_BLOCKS blocks: its 16 best are blocks of
            # distinct ranks
            if held & (cnt < TOPK_BLOCKS):
                mb_put_words(
                    out,
                    2 * (first + out_slot * TOPK_BLOCKS + cnt),
                    [ch, cl],
                    CM_SYS if ih_count > 1 else CM_DEV,
                )
            gpu.barrier()
            stamp(20)
            if tid < N_SPLIT * TOPK_BLOCKS:
                got = mb_poll([(got_from, 2 * (first + tid), 2)], got_scope)[0]
                fx.ptr_store(got[0], keys + 2 * tid)
                fx.ptr_store(got[1], keys + (2 * tid + 1))
            gpu.barrier()
            ch, cl, cnt, held = rank_cands()
            page = PAGE16_SIDES * fx.Int32(
                bo.buffer_load(rsrc(bt_row(k)), fx.max(cl, 0), vec_width=1, dtype=T.i32)
            )
            stamp(19)
            # the tail block is a local one: its key is forced
            kt = index_key(n_blk - 1, fx.Int32(0), n_blk)
            sel, slot, n_ctx = place(
                ch, cl, cnt, held, kt, n_blk, pick(seq_lens_k, k), pick(long_rows, k)
            )
            if sel & (slot // 2 == part):
                for j in range_constexpr(PAGES_PER_BLOCK):
                    fx.ptr_store(
                        page * PAGES_PER_BLOCK + j,
                        blk + ((slot % 2) * PAGES_PER_BLOCK + j),
                    )
            if sel & (part == 0):
                for j in range_constexpr(PAGES_PER_BLOCK):
                    bo.buffer_store(
                        page * PAGES_PER_BLOCK + j, table, slot * PAGES_PER_BLOCK + j
                    )
            if tid == 0:
                fx.ptr_store(n_ctx, blk + 16)
                if part == 0:
                    bo.buffer_store(n_ctx, table, TOPK_BLOCKS * PAGES_PER_BLOCK)
            gpu.barrier()

        def stage_select_long():
            """The split tasks of a long request's token select before the
            split stage (``select_long``), each leaving its blk for its split task
            (a CTA runs at most one). Out of the split stage and cold: inline
            there, this code slowed the split for every context (64k: +1 us)."""
            ts = start("split")
            k = ts // N_SPLIT
            if unlikely((ts < N_SPLIT * tokens) & pick(long_rows, k)):
                select_long(k, ts % N_SPLIT)

        # Each stage is its own function: flydsl carries a local reassigned inside a
        # dynamic if / for as region state, so names must not be shared across stages.
        # ============================================================ 1. split
        def stage_split():
            r_k, r_v = rsrc(k_cache), rsrc(v_cache)
            # one scale a cache, so both fold into scalars the whole stage shares:
            # the QK scale is uniform over keys, and the PV requant below needs no
            # max over the partition's V scales because they are all this one
            ks, vs = kv_cache_scale(k_scale), kv_cache_scale(v_scale)
            qk_scale = sm_scale * ks
            pscale = vs * (1.0 / FP8_MAX)
            PPW = SPLIT_KEYS // WAVES // PAGE16  # pages per wave
            for ts in range(start("split"), N_SPLIT * tokens, G):
                ts = fx.Int32(ts)  # token tok's partition t: ts = tok N_SPLIT + t
                tok = ts // N_SPLIT
                t = ts % N_SPLIT
                select_blocks(tok, t)
                stamp(17)
                if const_expr(fuse_k1):
                    # this token's q heads and every token's new K / V: a request's
                    # tokens of one step (speculative decode) attend to each other's
                    if wave == 0:
                        kv = lane - H
                        k1_poll(
                            [
                                (
                                    hdone_mb,
                                    (lane < H).select(
                                        tok * K1_HEAD_TASKS + lane,
                                        fx.min(kv // 2, tokens - 1) * K1_HEAD_TASKS
                                        + H
                                        + kv % 2,
                                    ),
                                    1,
                                )
                            ]
                        )
                    gpu.barrier()
                    acquire()
                # Every independent load goes out first: this wave's pages and q.
                # Pages past the context read junk masked below.
                n_ctx = uniform(fx.ptr_load(blk + 16))
                pages = [
                    uniform(fx.ptr_load(blk + (wave * PPW + j))) for j in range(PPW)
                ]
                qv = ld_bf16x4(rsrc(q), tok * O_K + tid * 4, CM_K1)
                # K (QK B operand, key = lane % 16 of page j), V (PV B operand: dims
                # 16 jd + lane % 16, keys 8 (lane / 16) .. + 8)
                kw = []
                for j in range_constexpr(PPW):
                    for s in range_constexpr(HEAD_DIM // 32):
                        kw.append(
                            fx.Vector(
                                bo.buffer_load(
                                    r_k,
                                    (
                                        pages[j] * PAGE_BYTES
                                        + (2 * s + g4 // 2) * 256
                                        + l16 * 16
                                        + (g4 % 2) * 8
                                    )
                                    // 4,
                                    vec_width=2,
                                    dtype=T.i32,
                                    cache_modifier=CM_K1,
                                )
                            ).bitcast(fx.Int64)[0]
                        )
                vpg = (g4 // 2 == 0).select(pages[0], pages[1])
                vw = [
                    fx.Vector(
                        bo.buffer_load(
                            r_v,
                            (vpg * PAGE_BYTES + (16 * jd + l16) * PAGE16 + (g4 % 2) * 8)
                            // 4,
                            vec_width=2,
                            dtype=T.i32,
                            cache_modifier=CM_K1,
                        )
                    ).bitcast(fx.Int64)[0]
                    for jd in range(HEAD_DIM // 16)
                ]
                fx.ptr_store(fp8_pack4(qv[0], qv[1], qv[2], qv[3]), q8 + tid)
                gpu.barrier()
                stamp(12)
                # QK per page j: A = q (heads), B = K (keys); lane holds heads
                # 4 g4 + e, key l16
                key0 = t * SPLIT_KEYS + wave * (PPW * PAGE16)
                valid = [key0 + j * PAGE16 + l16 < n_ctx for j in range(PPW)]
                sc = []
                for j in range_constexpr(PPW):
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for s in range_constexpr(HEAD_DIM // 32):
                        qw = fx.Vector(
                            fx.ptr_load(
                                q8 + (l16 * 32 + 8 * s + 2 * g4), result_type=v2i
                            )
                        )
                        c = mfma_fp8(
                            qw.bitcast(fx.Int64)[0], kw[j * (HEAD_DIM // 32) + s], c
                        )
                    sc.append(
                        [
                            valid[j].select(qk_scale * c[e], fx.Float32(NEG))
                            for e in range(4)
                        ]
                    )
                # partition max per head: in-wave butterfly, then LDS
                for e in range_constexpr(4):
                    mh = butterfly(fx.max(sc[0][e], sc[1][e]), (8, 4, 2, 1), fx.max)
                    if l16 == 0:
                        fx.ptr_store(mh, hst + (wave * H + g4 * 4 + e))
                gpu.barrier()
                stamp(13)
                # every wave partial is read before the first p8 store below: LDS
                # stores in between serialize each read behind its own wait
                mxs = []
                for e in range_constexpr(4):
                    mx = fx.ptr_load(hst + (g4 * 4 + e))
                    for w in range_constexpr(1, WAVES):
                        mx = fx.max(mx, fx.ptr_load(hst + (w * H + g4 * 4 + e)))
                    mxs.append(mx)
                for e in range_constexpr(4):
                    head = g4 * 4 + e
                    mx = mxs[e]
                    ls = fx.Float32(0.0)
                    for j in range_constexpr(PPW):
                        pj = hw_exp2((sc[j][e] - mx) * LOG2E)
                        ls = ls + pj
                        # P spans the whole FP8 range: p is in (0, 1] after the max
                        # subtraction, and V's scale comes back out through pscale
                        # rather than being folded in a key at a time
                        b8 = (
                            fp8_pack4(
                                valid[j].select(fx.Float32(FP8_MAX), fx.Float32(0.0))
                                * pj,
                                0.0,
                                0.0,
                                0.0,
                            )
                            & 0xFF
                        )
                        w8 = b8 | (xshfl(b8, 1) << 8)
                        w8 = w8 | (xshfl(w8, 2) << 16)
                        if l16 % 4 == 0:
                            kloc = wave * (PPW * PAGE16) + j * PAGE16 + l16
                            fx.ptr_store(
                                w8, p8 + (head * (SPLIT_KEYS // 4) + kloc // 4)
                            )
                    ls = butterfly(ls, (8, 4, 2, 1))
                    if l16 == 0:
                        fx.ptr_store(ls, hst + (WAVES * H + wave * H + head))
                        fx.ptr_store(mx, hst + (2 * WAVES * H + head))
                gpu.barrier()
                stamp(14)
                # PV over this wave's 32 keys: A = P (heads), B = V (dims)
                pw = fx.Vector(
                    fx.ptr_load(
                        p8
                        + (
                            l16 * (SPLIT_KEYS // 4)
                            + (wave * (PPW * PAGE16) + 8 * g4) // 4
                        ),
                        result_type=v2i,
                    )
                ).bitcast(fx.Int64)[0]
                # keys past the context carry P = 0, but a NaN V byte would still poison
                nv = n_ctx - (key0 + 8 * g4)
                vmask = (nv >= 8).select(
                    fx.Int64(-1),
                    (nv <= 0).select(
                        fx.Int64(0), (fx.Int64(1) << (fx.Int64(nv) * 8)) - 1
                    ),
                )
                for jd in range_constexpr(HEAD_DIM // 16):
                    cv = mfma_fp8(
                        pw, vw[jd] & vmask, fx.Vector.filled(4, 0.0, fx.Float32)
                    )
                    for e in range_constexpr(4):
                        fx.ptr_store(
                            cv[e],
                            opart
                            + (wave * O_K + (g4 * 4 + e) * HEAD_DIM + 16 * jd + l16),
                        )
                gpu.barrier()
                stamp(15)
                # sum the waves' partials; gluon: acc = prob_scale * PV, then
                # * (1 / exp_sum), bf16
                e0 = tid * 4
                head = e0 // HEAD_DIM
                lsum = fx.ptr_load(hst + (WAVES * H + head))
                for w in range_constexpr(1, WAVES):
                    lsum = lsum + fx.ptr_load(hst + (WAVES * H + w * H + head))
                inv_l = 1.0 / (lsum > 0.0).select(lsum, fx.Float32(1.0))
                ov = []
                for k in range_constexpr(4):
                    acc = fx.ptr_load(opart + (e0 + k))
                    for w in range_constexpr(1, WAVES):
                        acc = acc + fx.ptr_load(opart + (w * O_K + e0 + k))
                    ov.append((pscale * acc) * inv_l)
                mb_put_bf(mb("sp_o"), ts * O_K + e0, ov)
                if tid < H:
                    mb_put(
                        mb("sp_m"),
                        ts * H + tid,
                        fx.ptr_load(hst + (2 * WAVES * H + tid)),
                    )
                    lt = fx.ptr_load(hst + (WAVES * H + tid))
                    for w in range_constexpr(1, WAVES):
                        lt = lt + fx.ptr_load(hst + (WAVES * H + w * H + tid))
                    mb_put(mb("sp_l"), ts * H + tid, lt)
                gpu.barrier()

        stage_select_long()
        stage_split()
        stamp(1)

        def merge_token(k):
            """Token k's partitions merged (the gluon decode's reduce; thread = 4 dims
            of one head) and quantized per token (standalone quant: scale = amax *
            (1 / FP8_MAX), x * rcp(scale)) -> (this thread's fp8 word, scale)."""
            hm = tid // (HEAD_DIM // 4)
            e0 = tid * 4
            sp0 = k * N_SPLIT  # token k's partitions
            # wave 0 waits on every partition's l, then the CTA polls once: 512
            # threads x 24 loads spinning through the whole split load the memory
            # system the split itself reads (-0.4 us)
            if wave == 0:
                mb_poll(
                    [
                        (mb("sp_l"), sp0 * H + lane + 64 * i, 1)
                        for i in range(N_SPLIT * H // 64)
                    ]
                )
            gpu.barrier()
            # one batch: every partition's (m, l) and this thread's 4 outputs
            got = mb_poll(
                [(mb("sp_m"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
                + [(mb("sp_l"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
                + [
                    (mb("sp_o"), ((sp0 + i) * O_K + e0) // 2, 2) for i in range(N_SPLIT)
                ],
                batch=3 * N_SPLIT,
            )
            mls = got[: 2 * N_SPLIT]
            ovs = got[2 * N_SPLIT :]
            stamp(3)
            m_i = [mls[i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
            l_i = [mls[N_SPLIT + i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
            gm = m_i[0]
            for i in range_constexpr(1, N_SPLIT):
                gm = fx.max(gm, m_i[i])
            sl = [l_i[i] * hw_exp2((m_i[i] - gm) * LOG2E) for i in range(N_SPLIT)]
            gs = sl[0]
            for i in range_constexpr(1, N_SPLIT):
                gs = gs + sl[i]
            gs = (gs > 0.0).select(gs, fx.Float32(1.0))
            av = [fx.Float32(0.0) for _ in range(4)]
            for i in range_constexpr(N_SPLIT):
                wi = sl[i] / gs
                o01 = bf2_f32(ovs[i][0])
                o23 = bf2_f32(ovs[i][1])
                av = [
                    av[0] + wi * o01[0],
                    av[1] + wi * o01[1],
                    av[2] + wi * o23[0],
                    av[3] + wi * o23[1],
                ]
            a0, a1, a2, a3 = (bf16_round(x) for x in av)
            amax = block_max(
                fx.max(
                    fx.max(fmath.absf(a0), fmath.absf(a1)),
                    fx.max(fmath.absf(a2), fmath.absf(a3)),
                )
            )
            x_scale = amax * (1.0 / FP8_MAX)
            # an all-zero row (a pad row) quantizes to zeros, not 0 * inf
            inv = (amax == 0.0).select(fx.Float32(0.0), hw_rcp(x_scale))
            return fp8_pack4(a0 * inv, a1 * inv, a2 * inv, a3 * inv), x_scale

        # ======================================= 2b. attention merge (S > 1)
        # One task per token, in parallel on its own CTA: merged in the o tasks, the
        # tokens' merges ran one after another in every one of them (+14 us at S = 4)
        def stage_merge():
            if const_expr(tokens > 1):
                for k in range(start("merge"), tokens, G):
                    k = fx.Int32(k)
                    word, x_scale = merge_token(k)
                    # plain words, then the scale: its pair is the flag
                    bo.buffer_store(
                        word,
                        rsrc(mb("attnq")),
                        k * (O_K // 4) + tid,
                        cache_modifier=CM_DEV,
                    )
                    fx.memory_fence(
                        syncscope=rocdl.SyncScope.Workgroup,
                        ordering=fx.AtomicOrdering.Release,
                    )
                    gpu.barrier()
                    if tid == 0:
                        mb_put(mb("attnq_s"), k, x_scale)

        stage_merge()

        # ================================================ 3. o_proj + attn reduce
        def stage_o():
            r_wo = rsrc(w_o)
            O_CPW = O_K // 64 // 4  # 64-k chunks per wave: 4 waves per 16-row group
            for t in range(start("o"), N_O, G):
                t = fx.Int32(t)
                rg = t * 2 + wave // 4
                wts = []
                for cc in range_constexpr(O_CPW):
                    kc = (wave % 4) * O_CPW + cc
                    wts.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_wo,
                                ((rg * (O_K // 32) + kc * 2) * 512 + lane * 16) // 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
                # this task's row scales, before the wait
                ws = fx.Float32(
                    bo.buffer_load(
                        rsrc(s_o), t * O_ROWS + tid % O_ROWS, vec_width=1, dtype=T.f32
                    )
                )

                if const_expr(tokens == 1):
                    # the merge folded in: the partitions -> this CTA's fp8 input
                    word, x_scale = merge_token(0)
                    x_scales = [x_scale]
                    fx.ptr_store(word, xs + tid)
                else:
                    # the merge tasks' fp8 inputs: every token's scale (published
                    # after its plain words), then the words, 16 B a load
                    got = mb_poll([(mb("attnq_s"), k, 1) for k in range(tokens)])
                    stamp(3)
                    x_scales = [got[k][0].bitcast(fx.Float32) for k in range(tokens)]
                    acquire()
                    n_v = tokens * O_K // 16
                    for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                        v = tid + THREADS * i
                        if v < n_v:
                            fx.ptr_store(
                                fx.Vector(
                                    bo.buffer_load(
                                        rsrc(mb("attnq")),
                                        v * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                        cache_modifier=CM_DEV,
                                    )
                                ),
                                xs + (v * 4 // (O_K // 4) * O_ROW + v * 4 % (O_K // 4)),
                            )
                gpu.barrier()
                # B column l % 16 is token l % 16 (token 0 past the last): C column k
                # is token k's rows, the weight stream is the same for any token count
                col_tok = fx.min(l16, tokens - 1)
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                # chunks kc, kc + 1 in one 16x16x128 FP8 MFMA (unit scales): its lane
                # holds k 16 (l / 16) .. + 16 of each 64-k half -- the chunks' packing
                for p in range_constexpr(O_CPW // 2):
                    xb = [
                        fx.Vector(
                            fx.ptr_load(
                                xs
                                + (
                                    col_tok * O_ROW
                                    + ((wave % 4) * O_CPW + 2 * p + h) * 16
                                    + g4 * 4
                                ),
                                result_type=v4f,
                            )
                        ).bitcast(fx.Int32)
                        for h in range(2)
                    ]
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                fx.Vector.from_elements(
                                    [
                                        wts[2 * p + h][e]
                                        for h in range(2)
                                        for e in range(4)
                                    ],
                                    fx.Int32,
                                ),
                                fx.Vector.from_elements(
                                    [xb[h][e] for h in range(2) for e in range(4)],
                                    fx.Int32,
                                ),
                                c,
                                0,
                                0,
                                0,
                                fx.Int32(127),
                                0,
                                fx.Int32(127),
                            ],
                        )
                    )
                fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                if tid < O_ROWS * tokens:
                    tk = tid // O_ROWS
                    grp = (tid % O_ROWS) // 16
                    rr = tid % 16
                    tot = fx.Float32(0.0)
                    for w in range_constexpr(4):
                        tot = tot + fx.ptr_load(
                            red
                            + (((grp * 4 + w) * 64 + 16 * (rr // 4) + tk) * 4 + rr % 4)
                        )
                    fx.ptr_store(bf16_round(tot * pick(x_scales, tk) * ws), misc + tid)
                gpu.barrier()
                # push bf16 partial rows, wave w -> peer w (to itself too); with AG_RS
                # to the row group's owner rank only
                push_rows("attn", t * O_ROWS, O_ROWS, t % W if AG_RS else None)
                gpu.barrier()
                if const_expr(AG_RS):  # noqa: SIM102
                    # the owner sums every peer's partials in rank order and gathers
                    # the sums to every rank's a_ag (h_mid: the router tasks)
                    if (t % W == rank) & (tid < O_ROWS // 2 * tokens):
                        ptk = tid // (O_ROWS // 2)
                        row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                        own = sym + fx.Int64(SY["attn"])
                        prt = mb_poll(
                            [
                                (own, ((src * MAX_TOKENS + ptk) * HIDDEN + row) // 2, 1)
                                for src in range(W)
                            ],
                            "system",
                        )
                        t0 = fx.Float32(0.0)
                        t1 = fx.Float32(0.0)
                        for src in range_constexpr(W):
                            f0, f1 = bf2_f32(prt[src][0])
                            t0 = t0 + f0
                            t1 = t1 + f1
                        for w in range_constexpr(W):
                            mb_put_bf(
                                peer_addr(w) + fx.Int64(SY["a_ag"]),
                                ptk * HIDDEN + row,
                                [t0, t1],
                                CM_SYS,
                            )
                if const_expr(not AG_RS):  # noqa: SIM102
                    # this task's rows of the all-reduce, finished here (GLM's
                    # peer_reduce): every peer's partials summed in rank order,
                    # bf16, plus the residual -> h_mid and the 'a' mailbox. A
                    # router task then reads 24 KB of device-scope bf16, not 4
                    # peers' 48 KB of system-scope ones
                    if tid < O_ROWS // 2 * tokens:
                        ptk = tid // (O_ROWS // 2)
                        row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                        if const_expr(fuse_k1):  # the residual (K1's norm / GEMV task)
                            k1_poll([(rdone_mb, ptk, 1)])
                            acquire()
                        hr = fx.Int32(
                            bo.buffer_load(
                                rsrc(h_in),
                                (ptk * HIDDEN + row) // 2,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_K1,
                            )
                        )
                        own = sym + fx.Int64(SY["attn"])
                        prt = mb_poll(
                            [
                                (own, ((src * MAX_TOKENS + ptk) * HIDDEN + row) // 2, 1)
                                for src in range(W)
                            ],
                            "system",
                        )
                        t0 = fx.Float32(0.0)
                        t1 = fx.Float32(0.0)
                        for src in range_constexpr(W):
                            f0, f1 = bf2_f32(prt[src][0])
                            t0 = t0 + f0
                            t1 = t1 + f1
                        r0, r1 = bf2_f32(hr)
                        a0 = bf16_round(t0) + r0
                        a1 = bf16_round(t1) + r1
                        # the rank sum only: the router adds the residual in f32, so the
                        # norm reads bf16(sum) + residual unrounded, as the
                        # original path
                        mb_put_bf(mb("a"), ptk * HIDDEN + row, [t0, t1])
                        bo.buffer_store(
                            fx.Vector.from_elements([a0, a1], fx.Float32).to(
                                fx.BFloat16
                            ),
                            rsrc(h_mid),
                            ptk * HIDDEN + row,
                        )

        stage_o()
        stamp(4)

        # ======================================================= MoE helpers
        def e8m0_load(r_s, byte_idx):
            """An e8m0 scale byte, still in flight (the MFMA takes it as is)."""
            return fx.Int32(bo.buffer_load(r_s, byte_idx, vec_width=1, dtype=T.i8))

        def group_scale_byte(expert, rows, rg, kc, cols):
            """scale_index(expert rows + 16 rg + l16, 4 kc + g4, cols): this lane's
            byte for row group rg, 128-k chunk kc, folded to wave-uniform terms plus
            the lane (the general form's per-lane arithmetic cost the S = 16 up /
            gate loop ~8 us)."""
            base = (expert * (rows // 32) + rg // 2) * (cols // 8) + kc // 2
            return (base * 64 + lane) * 4 + (kc % 2) * 2 + rg % 2

        def route_top4(k, k0, sc0, k1, sc1, sig, wave_mask=None):
            """route[8k : 8k+4] / rwt[..] := token k's sigmoid top-k gating from the
            router tasks' routing keys ``k0`` / ``k1`` (experts ``lane`` / ``lane +
            64``, sigmoid + bias, see route_key) and unbiased sigmoids (wave k % WAVES
            computes it, looking the sigmoids up in LDS at ``sig``): a pick is one
            wave max; the weights are the picks' sigmoids renormalized and scaled by
            route_scale. Slot 4 is the fused shared expert. ``wave_mask``: this
            wave's expert -> token mask, the picks' bit k set (the picks are
            distinct, and the wave's tokens run in program order). The caller
            syncs."""
            if wave == k % WAVES:
                # the unbiased weights, looked up by expert once the picks are known
                fx.ptr_store(sc0, sig + (k * N_ROUTED + lane))
                fx.ptr_store(sc1, sig + (k * N_ROUTED + 64 + lane))
                pid = fx.Int32(0)
                for p in range_constexpr(TOP_K):
                    m = butterfly(fx.max(k0, k1), (32, 16, 8, 4, 2, 1), fx.max)
                    hit0 = k0 == m
                    k0 = hit0.select(fx.Int32(-(2**31)), k0)
                    k1 = ((k1 == m) & (k0 != m)).select(fx.Int32(-(2**31)), k1)
                    pid = (lane == p).select(127 - (m & 127), pid)
                if lane < TOP_K:
                    w = fx.ptr_load(sig + (k * N_ROUTED + pid))
                    tot = butterfly(w, (1, 2))
                    fx.ptr_store(pid, route + (8 * k + lane))
                    fx.ptr_store(
                        w * (route_scale / fx.max(tot, 1e-20)), rwt + (8 * k + lane)
                    )
                    if const_expr(wave_mask is not None):
                        was = fx.Int32(fx.ptr_load(wave_mask + pid))
                        fx.ptr_store(was | (fx.Int32(1) << k), wave_mask + pid)
                if lane == 0:
                    fx.ptr_store(fx.Int32(SHARED_EXPERT), route + (8 * k + TOP_K))
                    fx.ptr_store(fx.Float32(shared_weight), rwt + (8 * k + TOP_K))

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

        def mx8_scale(vals, offsets):
            """E8M0 byte (>= 1) of the 32-block the lanes xor-``offsets`` apart share,
            aiter's MXFP8 RoundUp rule ceil_pow2(amax / 448), and its reciprocal."""
            amax = fx.max(vals[0], -vals[0])
            for v in vals[1:]:
                amax = fx.max(amax, fx.max(v, -v))
            amax = butterfly(amax, offsets, fx.max)
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            e = (bits >> 23) & 0xFF
            e = e + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
            e = fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))
            return e, hw_rcp((e << 23).bitcast(fx.Float32))

        def mx8_q(v, inv):
            return fx.max(fx.min(v * inv, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))

        def lds_b8(chunk_word):
            """This lane's B operand of the 128-k chunk at LDS word ``chunk_word``:
            fp8 k 16 g .. + 16 then 64 + 16 g .. + 16 (g = lane // 16), the
            f8f6f4 MFMA's B layout (harness/mfma_f4f8_probe.py)."""
            h = [
                fx.Vector(
                    fx.ptr_load(xs + (chunk_word + 16 * j + 4 * g4), result_type=v4f)
                ).bitcast(fx.Int32)
                for j in range(2)
            ]
            return fx.Vector.from_elements(
                [h[j][e] for j in range(2) for e in range(4)], fx.Int32
            )

        def bf16x2(word):
            return fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)

        def lds_i32(word):
            return fx.ptr_load(xs + word).bitcast(fx.Int32)

        def mfma_f4f8(w4, sa, b8, sb, c, sa_byte=None):
            """16x16x128 scaled MFMA: A = this lane's 32 fp4 (row l % 16, k block
            l / 16) with E8M0 ``sa``, B = its 32 fp8 with E8M0 ``sb``. ``sa_byte``:
            ``sa`` is a scale dword, the MFMA takes that byte of it."""
            if const_expr(sa_byte is None):
                sa, sa_byte = sa & 0xFF, 0
            return fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32), [w4, b8, c, 4, 0, sa_byte, sa, 0, sb]
                )
            )

        # ============================================================ 4. router
        def stage_router():
            r_g = rsrc(g_post)
            r_gate = rsrc(w_gate)
            # this thread's 12 consecutive elements of the row (512 x 12 = HIDDEN):
            # the poll is 3 specs per rank, 12 in one batch, every thread busy
            RE = HIDDEN // THREADS
            rw0 = tid * (RE // 2)

            def ld_bf16x12(r, word, cm=0):
                """12 bf16 from ``word`` (a 16 B then an 8 B load)."""
                a8 = fx.Vector(
                    bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm)
                ).bitcast(fx.BFloat16)
                a4 = fx.Vector(
                    bo.buffer_load(
                        r, word + 4, vec_width=2, dtype=T.i32, cache_modifier=cm
                    )
                ).bitcast(fx.BFloat16)
                return [a8[e] for e in range(8)] + [a4[e] for e in range(4)]

            # one task per (token, 8 experts): the tokens' norms and dot products run
            # on different CTAs, not one after another in each (S = 4: 74.7 -> 64.1 us)
            for tt in range(start("router"), N_ROUTER * tokens, G):
                tt = fx.Int32(tt)
                k = tt // N_ROUTER
                t = tt % N_ROUTER
                e_r = t * WAVES + wave
                bias_r = fx.Float32(
                    bo.buffer_load(rsrc(bias), e_r, vec_width=1, dtype=T.f32)
                )
                gw = [  # bf16 pairs
                    fx.Vector(
                        bo.buffer_load(
                            r_gate,
                            (e_r * HIDDEN + (i * 64 + lane) * 8) // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for i in range(HIDDEN // 512)
                ]
                gp = ld_bf16x12(r_g, rw0)
                if const_expr(fuse_k1):
                    # the residual (K1's norm / GEMV task)
                    if wave == 0:
                        k1_poll([(rdone_mb, k, 1)])
                    gpu.barrier()
                    acquire()
                hres = ld_bf16x12(rsrc(h_in), k * (HIDDEN // 2) + rw0, CM_K1)
                # the all-reduced post-attention row (the o tasks finished it; with
                # AG_RS the owner ranks gathered it into this rank's a_ag)
                a_src, a_scope = (
                    (sym + fx.Int64(SY["a_ag"]), "system")
                    if AG_RS
                    else (mb("a"), "agent")
                )
                arow = mb_poll(
                    [
                        (a_src, (k * HIDDEN + tid * RE) // 2 + 2 * h, 2)
                        for h in range(RE // 4)
                    ],
                    a_scope,
                )
                stamp(22)
                v = [
                    f + fx.Float32(hres[4 * h + 2 * qq + i])
                    for h in range(RE // 4)
                    for qq in range(2)
                    for i, f in enumerate(bf2_f32(arow[h][qq]))
                ]
                if const_expr(AG_RS):  # noqa: SIM102
                    # h_mid = bf16(bf16(sum) + residual): task t writes slice t
                    if tid // (XN_SLICE // RE) == t:
                        hw = (k * HIDDEN + tid * RE) // 2
                        hb = fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16)
                        hi = hb.bitcast(fx.Int32)
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [hi[e] for e in range(4)], fx.Int32
                            ),
                            rsrc(h_mid),
                            hw,
                        )
                        bo.buffer_store(
                            fx.Vector.from_elements([hi[4], hi[5]], fx.Int32),
                            rsrc(h_mid),
                            hw + 4,
                        )
                # the row's sum of squares: a wave sum each, one barrier, every lane
                # adds the 8
                ssw = fx.Float32(0.0)
                for e in range_constexpr(RE):
                    ssw = fmath.fma(v[e], v[e], ssw)
                ssw = wave_sum(ssw)
                if lane == 0:
                    fx.ptr_store(ssw, red + wave)
                gpu.barrier()
                ss = fx.ptr_load(red)
                for w in range_constexpr(1, WAVES):
                    ss = ss + fx.ptr_load(red + w)
                stamp(24)
                rstd = hw_rsq(ss / float(HIDDEN) + eps)
                for j in range_constexpr(RE // 2):
                    fx.ptr_store(
                        bf16_pair(
                            v[2 * j] * rstd * (fx.Float32(gp[2 * j]) + 1.0),
                            v[2 * j + 1] * rstd * (fx.Float32(gp[2 * j + 1]) + 1.0),
                        ),
                        xs + (rw0 + j),
                    )
                gpu.barrier()
                stamp(25)
                if tid < XN_SLICE // 4:  # 4 elements a lane, 8 lanes a 32-block
                    w0 = t * (XN_SLICE // 2) + tid * 2
                    f = [
                        v
                        for j in range(2)
                        for v in bf2_f32(fx.ptr_load(xs + (w0 + j)).bitcast(fx.Int32))
                    ]
                    e, inv = mx8_scale(f, (1, 2, 4))
                    word = fp8_pack4(*[mx8_q(v, inv) for v in f])
                    xw = k * XN8_WORDS + t * (XN_SLICE // 4) + tid
                    xsw = k * XSC_WORDS + t * (XN_SLICE // 32) + tid // 8
                    if const_expr(WIDE):
                        # plain words, flagged once the task is done (below)
                        bo.buffer_store(
                            word, rsrc(mb("xn8p")), xw, cache_modifier=CM_DEV
                        )
                        if tid % 8 == 0:
                            bo.buffer_store(
                                e, rsrc(mb("xscp")), xsw, cache_modifier=CM_DEV
                            )
                    else:
                        mb_put_words(mb("xn8"), xw, [word])
                        if tid % 8 == 0:
                            mb_put_words(mb("xsc"), xsw, [e])
                stamp(26)
                # bf16 pair dot products, one accumulator per pair position: four
                # 12-long chains instead of one 96-long FMA chain behind 192 widenings
                accs = [fx.Float32(0.0) for _ in range(4)]
                for i in range_constexpr(HIDDEN // 512):
                    xv = fx.Vector(
                        fx.ptr_load(xs + (i * 64 + lane) * 4, result_type=v4f)
                    ).bitcast(fx.Int32)
                    for j in range_constexpr(4):
                        accs[j] = fx.Float32(
                            rocdl.fdot2_f32_bf16(
                                T.f32,
                                as_ir_value(bf16x2(gw[i][j])),
                                as_ir_value(bf16x2(xv[j])),
                                as_ir_value(accs[j]),
                                clamp=False,
                            ).result
                        )
                logit = wave_sum((accs[0] + accs[1]) + (accs[2] + accs[3]))
                stamp(23)
                # the routing key and sigmoid go out, not the logit: the consumers'
                # routing starts at the max rounds
                sc = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
                rk = route_key(sc + bias_r, e_r)
                if lane < SCORE_COPIES:
                    mb_put_words(
                        scores_copy(lane, k), e_r * 2, [rk, sc.bitcast(fx.Int32)]
                    )
                if const_expr(WIDE):
                    # this task's xn slice is out: its flag (after the logits, not
                    # in front of them)
                    fx.memory_fence(
                        syncscope=rocdl.SyncScope.Workgroup,
                        ordering=fx.AtomicOrdering.Release,
                    )
                    gpu.barrier()
                    if tid == 0:
                        mb_put(mb("xndone"), tt, fx.Int32(1))

        stage_router()
        stamp(5)

        # ================================ 5. experts: up / gate -> down -> FFN reduce
        UG_KC = HIDDEN // 128  # 128-k chunks
        UG_CPW = UG_KC // WAVES  # 128-k chunks per wave: each wave a k eighth
        DN_KC = INTER // 128  # 128-k chunks of one slot
        DN_UNITS = TOP_K * DN_KC  # routed (slot, 128-k chunk) units per row group
        DN_UPW = (DN_UNITS + 3) // 4  # 4 waves per row group
        SH_UPW = (DN_KC + 3) // 4  # the shared expert's chunks, once for all tokens
        # routed (token, slot, row group) tasks t < ROUTED_TASKS; the shared expert's
        # row groups (one GEMV with a column per token) are tasks ROUTED_TASKS + j,
        # run as task j of stage "shared" while the routing is still on its way, so
        # the routed tasks alone set the rounds (S = 4: 768 = 3 x 256)
        ROUTED_TASKS = TOP_K * UG_PER_SLOT * tokens
        TPT = TOP_K * UG_PER_SLOT  # routed tasks per token
        CPT = ug_ctas_per_token(tokens)
        UG_ITERS = max(ug_tasks_of(c, tokens) for c in range(G))  # per CTA, at most

        def ug_task_id(it):
            """(this CTA's it-th routed task, clamped; whether it exists)."""
            if const_expr(CPT is not None):
                u = bid % CPT + it * CPT
                return (bid // CPT) * TPT + fx.min(u, TPT - 1), u < TPT
            t = start("ug") + it * G
            return fx.min(t, ROUTED_TASKS - 1), t < ROUTED_TASKS

        def ug_task(t):
            """(shared expert?, token, slot, row group) of up / gate task t."""
            shared = t >= ROUTED_TASKS
            r = fx.min(t, ROUTED_TASKS - 1)
            tk = r // (TOP_K * UG_PER_SLOT)
            slot = shared.select(
                fx.Int32(TOP_K), (r % (TOP_K * UG_PER_SLOT)) // UG_PER_SLOT
            )
            return shared, tk, slot, t % UG_PER_SLOT

        def ug_load(t):
            """Up / gate weights and scales of task t."""
            shared, tk, slot, cg = ug_task(t)
            # the bound keeps a corrupt route from addressing past the expert table
            e_u = shared.select(
                fx.Int32(SHARED_EXPERT),
                fx.min(uniform(fx.ptr_load(route + (8 * tk + slot))), SHARED_EXPERT),
            )
            r_w = rsrc(w13 + fx.Int64(e_u) * W13_BYTES)
            r_s = rsrc(s13)
            wts = []  # [gate, up][chunk]
            scs = []
            for gu in range_constexpr(2):
                rg = gu * UG_PER_SLOT + cg
                wts.append([])
                scs.append([])
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    wts[gu].append(
                        fx.Vector(
                            bo.buffer_load(
                                r_w,
                                ((rg * (HIDDEN // 64) + kc * 2) * 512 + lane * 16) // 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
                    scs[gu].append(
                        e8m0_load(
                            r_s,
                            group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS),
                        )
                    )
            return wts, scs

        def ug_compute(t, wts, scs):
            """Up / gate GEMV of task t -> swiglu -> its 16 mid values (of every token
            for the shared expert: B column l % 16 is token l % 16)."""
            shared, tk, slot, cg = ug_task(t)
            col = shared.select(fx.min(l16, tokens - 1), tk)
            cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
            cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
            # every chunk's B scale up front: read at use, each MFMA waited out an LDS
            # latency (S = 4 up / gate 3.5 us over the unscaled floor)
            sbs = [
                lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                for i in range(UG_CPW)
            ]
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                xb = lds_b8(col * XN8_ROW + kc * 32)
                cg_ = mfma_f4f8(wts[0][i], scs[0][i], xb, sbs[i], cg_)
                cu_ = mfma_f4f8(wts[1][i], scs[1][i], xb, sbs[i], cu_)
            stamp(9)
            # gate partials in red, up partials in opart (free once the split is done)
            fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
            fx.ptr_store(cu_, opart + (wave * 64 + lane) * 4)
            gpu.barrier()
            # row tid % 16; a shared task's thread tid reads token tid / 16's column
            if tid < shared.select(fx.Int32(UG_ROWS * tokens), fx.Int32(UG_ROWS)):
                r = tid % UG_ROWS
                col = tid // UG_ROWS
                kk = shared.select(col, tk)
                gsum = fx.Float32(0.0)
                usum = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    gsum = gsum + fx.ptr_load(
                        red + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                    )
                    usum = usum + fx.ptr_load(
                        opart + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                    )
                y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
                mb_put(
                    mb("mid"),
                    (kk * MOE_SLOTS + slot) * INTER + cg * UG_ROWS + r,
                    bf16_round(y),
                )
            gpu.barrier()

        def sh_load(j, has=None):
            """The shared expert's half task j: gate rows 8 j .. + 8 on lanes l16 < 8,
            the up rows alike on the rest -- one MFMA tile; the waves split k."""
            row = (l16 < 8).select(SH_ROWS * j + l16, INTER + SH_ROWS * j + l16 - 8)
            rg = row // 16
            if const_expr(has is None):
                r_w = rsrc(w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES)
                r_s = rsrc(s13)
            else:  # a CTA without the task reads through zero-sized buffers
                r_w = rsrc(
                    w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES,
                    has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
                )
                r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
            wts = []
            scs = []
            # the row's part of scale_index, once: a chunk adds 256 (kc / 2)
            # + 2 (kc % 2)
            sc_row = scale_index(SHARED_EXPERT * W13_ROWS + row, g4, W13_SCALE_COLS)
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts.append(
                    fx.Vector(
                        bo.buffer_load(
                            r_w,
                            (
                                (rg * (HIDDEN // 64) + kc * 2) * 512
                                + (g4 * 16 + row % 16) * 16
                            )
                            // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                )
                scs.append(e8m0_load(r_s, sc_row + kc // 2 * 256 + kc % 2 * 2))
            return wts, scs

        def sh_compute(j, wts, scs):
            """Half task j's GEMV (B column l % 16 = token l % 16) -> swiglu -> the
            shared slot's mid rows 8 j .. + 8 of every token."""
            col = fx.min(l16, tokens - 1)
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            sbs = [
                lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                for i in range(UG_CPW)
            ]
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                c = mfma_f4f8(
                    wts[i], scs[i], lds_b8(col * XN8_ROW + kc * 32), sbs[i], c
                )
            fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            # mid row r = tid % 8 of token tid / 8: gate in tile row r, up in r + 8
            if tid < SH_ROWS * tokens:
                r = tid % SH_ROWS
                col = tid // SH_ROWS
                gsum = fx.Float32(0.0)
                usum = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    gsum = gsum + fx.ptr_load(
                        red + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                    )
                    usum = usum + fx.ptr_load(
                        red + ((w * 64 + 16 * (r // 4 + 2) + col) * 4 + r % 4)
                    )
                y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
                mb_put(
                    mb("mid"),
                    (col * MOE_SLOTS + TOP_K) * INTER + j * SH_ROWS + r,
                    bf16_round(y),
                )
            gpu.barrier()

        def down_load(k, td, has_dn):
            """Token k's route-weighted down units of row group td; a CTA without a
            down task reads through zero-sized buffers."""
            dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
            dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
            rg_d = td * 2 + wave // 4
            units = []
            for i in range_constexpr(DN_UPW):
                j = (wave % 4) + 4 * i
                jj = fx.min(j, DN_UNITS - 1)
                sl = jj // DN_KC
                kc = jj % DN_KC
                e_d = fx.min(uniform(fx.ptr_load(route + (8 * k + sl))), SHARED_EXPERT)
                wv = fx.Vector(
                    bo.buffer_load(
                        rsrc(w2 + fx.Int64(e_d) * W2_BYTES, dn_bytes),
                        ((rg_d * (INTER // 64) + kc * 2) * 512 + lane * 16) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                sc = e8m0_load(
                    rsrc(s2, dn_scale_bytes),
                    group_scale_byte(e_d, HIDDEN, rg_d, kc, W2_SCALE_COLS),
                )
                wgt = (j < DN_UNITS).select(
                    fx.ptr_load(rwt + (8 * k + sl)), fx.Float32(0.0)
                )
                units.append((wv, sc, sl, kc, wgt))
            return units

        def down_load_shared(td, has_dn):
            """The shared expert's down units of row group td (every token's)."""
            dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
            dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
            rg_d = td * 2 + wave // 4
            units = []
            for i in range_constexpr(SH_UPW):
                j = (wave % 4) + 4 * i
                kc = fx.min(j, DN_KC - 1)
                wv = fx.Vector(
                    bo.buffer_load(
                        rsrc(w2 + fx.Int64(SHARED_EXPERT) * W2_BYTES, dn_bytes),
                        ((rg_d * (INTER // 64) + kc * 2) * 512 + lane * 16) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                sc = e8m0_load(
                    rsrc(s2, dn_scale_bytes),
                    group_scale_byte(SHARED_EXPERT, HIDDEN, rg_d, kc, W2_SCALE_COLS),
                )
                wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
                units.append((wv, sc, kc, wgt))
            return units

        def ffn_finish(t):
            """``misc`` (row group t's bf16 partial rows, token k's DN_ROWS from k
            DN_ROWS) -> every peer -> rank-ordered sum -> ar_out."""
            # every token's rows: reduce threads take token tid // 16, pair tid % 16
            row_tok = tid // (DN_ROWS // 2)
            row_pair = tid % (DN_ROWS // 2)
            if const_expr(W > 1):  # push bf16 partial rows: wave w -> peer w
                # (with AG_RS to the row group's owner rank only)
                push_rows("ffn", t * DN_ROWS, DN_ROWS, t % W if AG_RS else None)
                gpu.barrier()
            if tid < DN_ROWS // 2 * tokens:
                row = t * DN_ROWS + row_pair * 2
                if const_expr(W == 1):
                    t0 = fx.ptr_load(misc + tid * 2)
                    t1 = fx.ptr_load(misc + (tid * 2 + 1))
                else:
                    own = sym + fx.Int64(SY["ffn"])

                    def rank_sum():
                        parts = mb_poll(
                            [
                                (
                                    own,
                                    ((src * MAX_TOKENS + row_tok) * HIDDEN + row) // 2,
                                    1,
                                )
                                for src in range(W)
                            ],
                            "system",
                        )
                        s0 = fx.Float32(0.0)
                        s1 = fx.Float32(0.0)
                        for src in range_constexpr(W):
                            f0, f1 = bf2_f32(parts[src][0])
                            s0 = s0 + f0
                            s1 = s1 + f1
                        return s0, s1

                    if const_expr(AG_RS):
                        t0 = fx.Float32(0.0)
                        t1 = fx.Float32(0.0)
                        if t % W == rank:  # the owner: sum, gather to every rank
                            t0, t1 = rank_sum()
                            for w in range_constexpr(W):
                                mb_put_bf(
                                    peer_addr(w) + fx.Int64(SY["ffn_ag"]),
                                    row_tok * HIDDEN + row,
                                    [t0, t1],
                                    CM_SYS,
                                )
                        else:  # the owner's sums (bf16, as ar_out stores them)
                            got = mb_poll(
                                [
                                    (
                                        sym + fx.Int64(SY["ffn_ag"]),
                                        (row_tok * HIDDEN + row) // 2,
                                        1,
                                    )
                                ],
                                "system",
                            )
                            t0, t1 = bf2_f32(got[0][0])
                    else:
                        t0, t1 = rank_sum()
                bo.buffer_store(
                    fx.Vector.from_elements([t0, t1], fx.Float32).to(fx.BFloat16),
                    rsrc(ar_out),
                    row_tok * HIDDEN + row,
                )

        # ======================================== 5w. experts, wide (S > 4)
        # LDS pool words: MXFP8 xn, its scales, the up partials, router sigmoids; the
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
        UGW_ITERS = 2 * ((U_MAX * UG_PAIRS + G - 1) // G)  # up / gate tasks a CTA runs
        DN_WITERS = (U_MAX * DN_KC + 3) // 4  # (expert, chunk) units a wave runs
        DN_DEPTH = 12  # units whose weights are in flight
        ROUTE_ITERS = (tokens + WAVES - 1) // WAVES  # tokens a wave routes

        def popc64(x):
            return fx.Int32(fx.ctpop(x))

        def slot_of(k, e):
            """Token k's slot routed to expert e (TOP_K: the shared expert)."""
            s = fx.Int32(TOP_K)
            for j in range_constexpr(TOP_K):
                s = (fx.Int32(fx.ptr_load(route + (8 * k + j))) == e).select(
                    fx.Int32(j), s
                )
            return s

        def build_experts():
            """``uexp`` / umask[0 .. n_u) := the tokens' distinct routed experts in
            expert order and their token masks, [n_u] := the shared expert (every
            token), uexp[U_MAX + 1] := n_u. The waves' masks must be in (synced)."""
            m = fx.Int32(0)
            if tid < N_ROUTED:
                for w in range_constexpr(WAVES):
                    m = m | fx.Int32(fx.ptr_load(xs + (W_UPP + w * N_ROUTED + tid)))
            bal = ballot(m != 0)
            below = popc64(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))
            if tid == 0:
                fx.ptr_store(popc64(bal), umask + (U_MAX + 1))
            gpu.barrier()
            if tid < N_ROUTED:
                off = (wave == 0).select(
                    fx.Int32(0), fx.Int32(fx.ptr_load(umask + (U_MAX + 1)))
                )
                rank_ = off + below
                if m != 0:
                    fx.ptr_store(fx.Int32(tid), uexp + rank_)
                    fx.ptr_store(m, umask + rank_)
                if tid == N_ROUTED - 1:
                    nu_ = off + popc64(bal)
                    fx.ptr_store(nu_, uexp + (U_MAX + 1))
                    fx.ptr_store(fx.Int32(SHARED_EXPERT), uexp + nu_)
                    fx.ptr_store(fx.Int32((1 << tokens) - 1), umask + nu_)
            gpu.barrier()

        def wug_load(t, has, scales=True):
            """Up / gate weights (and ``scales``: the scale dwords, which hold the
            pair's both halves) of task t = (distinct expert t / 48, row group
            t % 48); without the task, reads go through zero-sized buffers."""
            u = t // UG_PER_SLOT
            cg = t % UG_PER_SLOT
            e_u = fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT)
            r_w = rsrc(
                w13 + fx.Int64(e_u) * W13_BYTES,
                has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
            )
            r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
            wts = []
            scs = []
            for gu in range_constexpr(2):
                rg = gu * UG_PER_SLOT + cg
                wts.append([])
                scs.append([])
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    wts[gu].append(
                        fx.Vector(
                            bo.buffer_load(
                                r_w,
                                ((rg * (HIDDEN // 64) + kc * 2) * 512 + lane * 16) // 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
                    if const_expr(scales and i % 2 == 0):
                        # the dword holding this lane's scales of chunks kc, kc + 1
                        # of both 16-row halves of the 32-row block (bytes 2 d + a)
                        scs[gu].append(
                            fx.Int32(
                                bo.buffer_load(
                                    r_s,
                                    group_scale_byte(
                                        e_u, W13_ROWS, rg, kc, W13_SCALE_COLS
                                    )
                                    // 4,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                        )
            return wts, scs

        def wug_y(wts, scs, half):
            """A task's up / gate GEMV for every token (B column l % 16 = token
            l % 16) -> swiglu: thread (row r, token k)'s mid value. ``half``: the
            task's 16-row half of its 32-row scale block (compile time)."""
            col = fx.min(l16, tokens - 1)
            cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
            cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
            sbs = [
                lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                for i in range(UG_CPW)
            ]
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                xb = lds_b8(col * XN8_ROW + kc * 32)
                sel = (i % 2) * 2 + half
                cg_ = mfma_f4f8(wts[0][i], scs[0][i // 2], xb, sbs[i], cg_, sel)
                cu_ = mfma_f4f8(wts[1][i], scs[1][i // 2], xb, sbs[i], cu_, sel)
            fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
            fx.ptr_store(cu_, fpool + (W_UPP + (wave * 64 + lane) * 4))
            gpu.barrier()
            r = tid % UG_ROWS
            k = fx.min(tid // UG_ROWS, tokens - 1)
            gsum = fx.Float32(0.0)
            usum = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                gsum = gsum + fx.ptr_load(
                    red + ((w * 64 + 16 * (r // 4) + k) * 4 + r % 4)
                )
                usum = usum + fx.ptr_load(
                    fpool + (W_UPP + (w * 64 + 16 * (r // 4) + k) * 4 + r % 4)
                )
            gpu.barrier()  # red and the up partials are free for the next task
            return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

        def fp8_lanes4(q):
            """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
            lo = fx.Int32(
                rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False)
            )
            return lo | (xshfl(lo, 2) << 16)

        def wug_publish(pr, y_lo, y_hi):
            """Task pair pr = (distinct expert pr / UG_PAIRS, 32-row block
            pr % UG_PAIRS): its mid rows (thread (r, k) holds rows r and 16 + r of
            the block) as the down stage's MXFP8 -- the rounding its bf16 staging
            did, one 32-block a 16-lane group -- then the pair's flag."""
            u = pr // UG_PAIRS
            blk_ = pr % UG_PAIRS
            r = tid % UG_ROWS
            k = fx.min(tid // UG_ROWS, tokens - 1)
            lo = bf16_round(y_lo)
            hi = bf16_round(y_hi)
            sc, inv = mx8_scale([lo, hi], (1, 2, 4, 8))
            words = [fp8_lanes4(mx8_q(v, inv)) for v in (lo, hi)]
            e_u = fx.Int32(fx.ptr_load(uexp + u))
            m_u = fx.Int32(fx.ptr_load(umask + u))
            row = k * MOE_SLOTS + slot_of(k, e_u)
            if (tid < UG_ROWS * tokens) & (((m_u >> k) & 1) != 0):
                if r % 4 == 0:
                    for h in range_constexpr(2):
                        bo.buffer_store(
                            words[h],
                            rsrc(mb("mid8p")),
                            row * MID_ROW_WORDS + blk_ * 8 + h * 4 + r // 4,
                            cache_modifier=CM_DEV,
                        )
                if r == 0:
                    bo.buffer_store(
                        sc,
                        rsrc(mb("midsp")),
                        row * MIDSC_ROW_WORDS + blk_,
                        cache_modifier=CM_DEV,
                    )
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
            )
            gpu.barrier()
            if tid == 0:
                mb_put(mb("ugdone"), pr, fx.Int32(1))

        def down_tables():
            """``urow`` / uwt := per (distinct expert, B column) the down stage's mid
            row (the zero row where the column's token is not routed there) and
            route weight. Routing only: built while the first up / gate weights
            land."""
            nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
            for i in range_constexpr((U_MAX * 16 + THREADS - 1) // THREADS):
                ix = tid + THREADS * i
                if ix < U_MAX * 16:
                    xu = ix // 16
                    xk = ix % 16
                    xe = fx.Int32(fx.ptr_load(uexp + fx.min(xu, nu)))
                    xm = fx.Int32(fx.ptr_load(umask + fx.min(xu, nu)))
                    xt = fx.min(xk, tokens - 1)
                    xs_ = slot_of(xt, xe)
                    xh = (((xm >> xk) & 1) != 0) & (xu <= nu)
                    fx.ptr_store(
                        xh.select(xt * MOE_SLOTS + xs_, fx.Int32(MID_ROWS)), urow + ix
                    )
                    fx.ptr_store(
                        xh.select(fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)),
                        uwt + ix,
                    )

        def stage_moe_wide():
            td = start("down")  # this CTA's down task (the split tasks come first)
            has_dn = td < N_DN
            td = fx.min(td, N_DN - 1)
            # every token's routing: wave w routes tokens w, w + WAVES, .., their
            # scores polled in one batch
            mine = bid % SCORE_COPIES
            # wave w's expert -> mask of the tokens it routed there, in the up
            # partials' words (free until the first up / gate task)
            wave_mask = xs + (W_UPP + wave * N_ROUTED)
            fx.ptr_store(fx.Int32(0), wave_mask + lane)
            fx.ptr_store(fx.Int32(0), wave_mask + (lane + 64))
            wave_toks = [
                fx.min(wave + WAVES * j, tokens - 1) for j in range(ROUTE_ITERS)
            ]
            sg = mb_poll(
                [
                    (scores_copy(mine, k), (lane + 64 * h) * 2, 2)
                    for k in wave_toks
                    for h in range(2)
                ]
            )
            for j in range_constexpr(ROUTE_ITERS):
                if wave + WAVES * j < tokens:
                    route_top4(
                        wave_toks[j],
                        sg[2 * j][0],
                        sg[2 * j][1].bitcast(fx.Float32),
                        sg[2 * j + 1][0],
                        sg[2 * j + 1][1].bitcast(fx.Float32),
                        fpool + W_SIG,
                        wave_mask,
                    )
            stamp(2)
            gpu.barrier()
            build_experts()
            stamp(6)
            if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
                sk = tid // MOE_SLOTS
                sj = tid % MOE_SLOTS
                mb_put(
                    mb("sel"),
                    sk * 2 * MOE_SLOTS + sj,
                    fx.ptr_load(route + (8 * sk + sj)),
                )
                mb_put(
                    mb("sel"),
                    sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                    fx.ptr_load(rwt + (8 * sk + sj)),
                )
            n_pairs = (uniform(fx.ptr_load(uexp + (U_MAX + 1))) + 1) * UG_PAIRS

            def task(it):
                """Iteration it's task: pair bid + (it / 2) G's half it % 2."""
                pr = fx.Int32(bid + it // 2 * G)
                return fx.min(pr, n_pairs - 1) * 2 + it % 2, pr < n_pairs

            t0, h0 = task(0)
            cur = wug_load(t0, h0)  # task 0: a lo half, with the pair's scales
            down_tables()
            stamp(7)
            # the first task's weights are in flight: now the xn
            # every token's MXFP8 xn and scales -> the pool, once every router task
            # flagged its slice (plain words, 16 B a load)
            n_rt = N_ROUTER * tokens
            if wave == 0:
                mb_poll(
                    [
                        (mb("xndone"), fx.min(lane + 64 * c, n_rt - 1), 1)
                        for c in range((n_rt + 63) // 64)
                    ]
                )
            gpu.barrier()
            acquire()
            for base_w, n_row, lds_w, lds_row in (
                (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
            ):
                n_v = tokens * n_row // 4
                for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                    v = tid + THREADS * i
                    if v < n_v:
                        fx.ptr_store(
                            fx.Vector(
                                bo.buffer_load(
                                    rsrc(base_w),
                                    v * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            ),
                            xs + (lds_w + v * 4 // n_row * lds_row + v * 4 % n_row),
                        )
            gpu.barrier()
            y_lo = fx.Float32(0.0)  # the pair's first half, until the second's
            for it in range_constexpr(UGW_ITERS):
                t, has_t = task(it)
                if const_expr(it + 1 < UGW_ITERS):
                    tn, hn = task(it + 1)
                    # a hi half reads no scales: its lo half's dwords hold them
                    nxt = wug_load(tn, hn, scales=(it + 1) % 2 == 0)
                if const_expr(it % 2 == 0):
                    pair_scs = cur[1]
                if has_t:
                    y = wug_y(cur[0], pair_scs, it % 2)  # a pair: row groups cg, cg + 1
                    if const_expr(it % 2 == 0):
                        y_lo = y
                    else:
                        wug_publish(t // 2, y_lo, y)
                if const_expr(it + 1 < UGW_ITERS):
                    cur = nxt
            stamp(8)
            if has_dn:
                wide_down(td)
                stamp(11)

        def wide_down(td):
            """Row group td's down GEMV over every distinct expert (B column l % 16 =
            token l % 16's mid for it, the zero row where the token is not routed
            there), route-weighted per column -> ffn_finish."""
            nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
            rg_d = td * 2 + wave // 4
            n_units = (nu + 1) * DN_KC

            def unit_load(i):
                """(weights, scale, unit) of this wave's i-th (expert, 128-k chunk)
                unit, zero-sized reads past the last."""
                j = wave % 4 + 4 * i
                ok = j < n_units
                # clamped into the table: a unit past the last reads a real or zero
                # row, never past urow (a garbage row is garbage B, and 0 * NaN)
                du = fx.min(j // DN_KC, U_MAX - 1)
                dkc = j % DN_KC
                de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
                dwv = fx.Vector(
                    bo.buffer_load(
                        rsrc(
                            w2 + fx.Int64(de) * W2_BYTES,
                            ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                        ),
                        ((rg_d * (INTER // 64) + dkc * 2) * 512 + lane * 16) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                dsc = e8m0_load(
                    rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                    group_scale_byte(de, HIDDEN, rg_d, dkc, W2_SCALE_COLS),
                )
                return dwv, dsc, du, dkc, ok

            def with_rows(u):
                """A loaded unit plus its B row and route weight (the tables are in),
                read ahead so the unit's B loads wait on one LDS trip."""
                dwv, dsc, du, dkc, ok = u
                drow = fx.Int32(fx.ptr_load(urow + (du * 16 + l16)))
                dw = ok.select(fx.ptr_load(uwt + (du * 16 + l16)), fx.Float32(0.0))
                return dwv, dsc, drow, dkc, dw, ok

            # software pipeline: the first units' weights go out before the mids are
            # in; unit i + DN_DEPTH's before unit i's MFMA
            acc = [fx.Float32(0.0) for _ in range(4)]
            inflight = [unit_load(i) for i in range(min(DN_DEPTH, DN_WITERS))]
            # every up / gate task pair's flag, then the MXFP8 mid rows and their
            # scales, the pool's layout (plain words, 16 B a load)
            n_ug = (nu + 1) * UG_PAIRS
            mb_poll(
                [
                    (mb("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                    for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
                ]
            )
            gpu.barrier()
            acquire()
            for base_w, n_row, lds_w, lds_row in (
                (mb("mid8p"), MID_ROW_WORDS, 0, MID_ROW),
                (mb("midsp"), MIDSC_ROW_WORDS, W_MIDSC, MIDSC_ROW),
            ):
                n_v = MID_ROWS * n_row // 4
                for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                    v = tid + THREADS * i
                    if v < n_v:
                        fx.ptr_store(
                            fx.Vector(
                                bo.buffer_load(
                                    rsrc(base_w),
                                    v * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            ),
                            xs + (lds_w + v * 4 // n_row * lds_row + v * 4 % n_row),
                        )
            if tid < MID_ROW_WORDS:
                fx.ptr_store(fx.Int32(0), xs + (MID_ROWS * MID_ROW + tid))
            if tid < MIDSC_ROW_WORDS:
                fx.ptr_store(fx.Int32(0), xs + (W_MIDSC + MID_ROWS * MIDSC_ROW + tid))
            gpu.barrier()
            stamp(10)
            for i in range_constexpr(DN_WITERS):
                if const_expr(i == 0):  # the tables are in: the early units' rows
                    inflight = [with_rows(u) for u in inflight]
                dwv, dsc, drow, dkc, dw, ok = inflight.pop(0)
                if const_expr(i + DN_DEPTH < DN_WITERS):
                    inflight.append(with_rows(unit_load(i + DN_DEPTH)))
                dcu = mfma_f4f8(
                    dwv,
                    dsc,
                    lds_b8(drow * MID_ROW + dkc * 32),
                    lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                    fx.Vector.filled(4, 0.0, fx.Float32),
                )
                # a unit past the last adds nothing, whatever its MFMA gave
                acc = [
                    acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0)) for e in range(4)
                ]
            a0, a1, a2, a3 = acc
            fx.ptr_store(
                fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < DN_ROWS * tokens:
                k = tid // DN_ROWS
                grp = (tid % DN_ROWS) // 16
                rr = tid % 16
                tot = fx.Float32(0.0)
                for w in range_constexpr(4):
                    tot = tot + fx.ptr_load(
                        red + (((grp * 4 + w) * 64 + 16 * (rr // 4) + k) * 4 + rr % 4)
                    )
                fx.ptr_store(bf16_round(tot), misc + tid)
            gpu.barrier()
            ffn_finish(td)

        def stage_moe():
            td = bid - BASE["down"]  # this CTA's down task
            has_dn = (td >= 0) & (td < N_DN)
            td = fx.min(fx.max(td, 0), N_DN - 1)
            has_sh = start("shared") < SH_TASKS
            n_xw = XN8_WORDS // THREADS

            def xn_specs(k):
                return [
                    (mb("xn8"), k * XN8_WORDS + tid + i * THREADS, 1)
                    for i in range(n_xw)
                ] + [(mb("xsc"), k * XSC_WORDS + fx.min(tid, XSC_WORDS - 1), 1)]

            def stage_xn(k, got):
                for i in range_constexpr(n_xw):
                    fx.ptr_store(got[i][0], xs + (k * XN8_ROW + tid + i * THREADS))
                if tid < XSC_WORDS:
                    fx.ptr_store(got[n_xw][0], xs + (XSC_LDS + k * XSC_ROW + tid))

            def fetch_xn(tok=None):
                """Token tok's (default: every token's) MXFP8 xn and scales -> xs. Wave
                0 first polls the last word of each router task's slice: all 512
                threads of 256 CTAs spinning on the whole of it (S = 4) held the
                shared-expert CTAs' xn back 3 us."""
                gate_tok = fx.min(lane // N_ROUTER, tokens - 1) if tok is None else tok
                if wave == 0:
                    mb_poll(
                        [
                            (
                                mb("xn8"),
                                gate_tok * XN8_WORDS
                                + (lane % N_ROUTER + 1) * (XN_SLICE // 4)
                                - 1,
                                1,
                            )
                        ]
                    )
                gpu.barrier()
                if tok is None:
                    got = mb_poll([sp for k in range(tokens) for sp in xn_specs(k)])
                    for k in range_constexpr(tokens):
                        stage_xn(k, got[k * (n_xw + 1) : (k + 1) * (n_xw + 1)])
                else:
                    stage_xn(tok, mb_poll(xn_specs(tok)))

            sh_late = shared_after_routing(tokens)
            if const_expr(sh_late):
                sh_j = fx.min(start("shared"), SH_TASKS - 1)
                sh_w = sh_load(sh_j, has_sh)
                if has_sh:
                    fetch_xn()
            elif has_sh:  # the shared expert needs xn, not the routing
                sh_j0 = start("shared")
                sh_w0 = sh_load(sh_j0)
                fetch_xn()
                gpu.barrier()
                sh_compute(sh_j0, sh_w0[0], sh_w0[1])
            if ug_task_id(0)[1] | has_dn:
                # xn lands before the logits: stage it while waiting (a shared-expert
                # CTA has it already); a CTA's routed tasks are one token's
                if start("shared") >= SH_TASKS:
                    fetch_xn(bid // CPT if CPT is not None else None)
                # wave k polls token k's router logits and routes it right away
                mine = bid % SCORE_COPIES
                for k in range_constexpr(tokens):
                    if wave == k:
                        got = mb_poll(
                            [
                                (scores_copy(mine, k), (lane + 64 * h) * 2, 2)
                                for h in range(2)
                            ]
                        )
                        stamp(21)
                        route_top4(
                            k,
                            got[0][0],
                            got[0][1].bitcast(fx.Float32),
                            got[1][0],
                            got[1][1].bitcast(fx.Float32),
                            opart,
                        )
                stamp(2)
                gpu.barrier()
                stamp(6)
                if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
                    sk = tid // MOE_SLOTS
                    sj = tid % MOE_SLOTS
                    mb_put(
                        mb("sel"),
                        sk * 2 * MOE_SLOTS + sj,
                        fx.ptr_load(route + (8 * sk + sj)),
                    )
                    mb_put(
                        mb("sel"),
                        sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                        fx.ptr_load(rwt + (8 * sk + sj)),
                    )
                cur = ug_load(ug_task_id(0)[0])
                stamp(7)
                if const_expr(sh_late):  # noqa: SIM102
                    if has_sh:  # while the first routed task's weights are in flight
                        sh_compute(sh_j, sh_w[0], sh_w[1])
                # task i + 1's weights go out before task i's GEMV (software pipeline);
                # past the last task the load index is clamped and the result dropped
                for it in range_constexpr(UG_ITERS):
                    t, has_t = ug_task_id(it)
                    if const_expr(it + 1 < UG_ITERS):
                        nxt = ug_load(ug_task_id(it + 1)[0])
                    if has_t:
                        ug_compute(t, cur[0], cur[1])
                    if const_expr(it == 0):
                        # token 0's down weights stream under the mid exchange; issued
                        # with the first task's up / gate weights they share the CU's
                        # bandwidth and delay them (K4 on 4 GPUs 34.3 -> 32.8 us)
                        dn_units = {0: down_load(0, td, has_dn)}
                        sh_units = down_load_shared(td, has_dn)
                    if const_expr(tokens > 1 and it == UG_ITERS - 1):
                        # two tokens in flight when the down GEMVs start
                        dn_units[1] = down_load(1, td, has_dn)
                    if const_expr(it + 1 < UG_ITERS):
                        cur = nxt
                stamp(8)
                if has_dn:
                    moe_down(td, has_dn, dn_units, sh_units)
                    stamp(11)

        def moe_down(t, has_dn, dn_units, sh_units):
            """Every token's route-weighted down GEMV of row group ``t`` -> push bf16
            partial rows -> rank-ordered sum -> ar_out."""

            def stage_mids(k):
                """Token k's mid rows (every slot) -> xs, as bf16 pairs."""
                mids = mb_poll(
                    [
                        (
                            mb("mid"),
                            k * MOE_SLOTS * INTER
                            + fx.min(tid * 2 + i * 2 * THREADS, MOE_SLOTS * INTER - 2),
                            2,
                        )
                        for i in range(4)
                    ]
                )
                for i in range_constexpr(4):
                    e = tid * 2 + i * 2 * THREADS
                    if e < MOE_SLOTS * INTER:
                        fx.ptr_store(
                            bf16_pair(
                                mids[i][0].bitcast(fx.Float32),
                                mids[i][1].bitcast(fx.Float32),
                            ),
                            xs + (k * (MOE_SLOTS * INTER // 2) + e // 2),
                        )

            # every token's mid in one go: they land together (S = 4: -1.1 us)
            for k in range_constexpr(tokens):
                stage_mids(k)
            stamp(10)
            gpu.barrier()
            # then MXFP8 in a pass of its own: quantizing inside stage_mids (after
            # each poll) cost S = 1 / 4 1.75 / 6 us, though the arithmetic is tiny
            for k in range_constexpr(tokens):
                if tid < MID8_WORDS // 2:  # 8 elements a lane, 4 lanes a 32-block
                    w = fx.Vector(
                        fx.ptr_load(xs + (k * MID16_WORDS + tid * 4), result_type=v4f)
                    ).bitcast(fx.Int32)
                    f = [v for j in range(4) for v in bf2_f32(w[j])]
                    sc, inv = mx8_scale(f, (1, 2))
                    qv = [mx8_q(v, inv) for v in f]
                    fx.ptr_store(
                        fp8_pack4(*qv[:4]), xs + (MID8_LDS + k * MID8_WORDS + tid * 2)
                    )
                    fx.ptr_store(
                        fp8_pack4(*qv[4:]),
                        xs + (MID8_LDS + k * MID8_WORDS + tid * 2 + 1),
                    )
                    if tid % 4 == 0:
                        fx.ptr_store(sc, xs + (MIDSC_LDS + k * MIDSC_WORDS + tid // 4))
            gpu.barrier()
            # the shared expert once, B column l % 16 = token l % 16's mid -> opart
            # (free once the split is done); each token's reduce adds its column
            col_tok = fx.min(l16, tokens - 1)
            sh = [fx.Float32(0.0) for _ in range(4)]
            for wv, sc, kc, wgt in sh_units:
                cu = mfma_f4f8(
                    wv,
                    sc,
                    lds_b8(
                        MID8_LDS
                        + col_tok * MID8_WORDS
                        + (TOP_K * INTER + kc * 128) // 4
                    ),
                    lds_i32(
                        MIDSC_LDS
                        + col_tok * MIDSC_WORDS
                        + (TOP_K * INTER + kc * 128) // 32
                        + g4
                    ),
                    fx.Vector.filled(4, 0.0, fx.Float32),
                )
                sh = [sh[j4] + cu[j4] * wgt for j4 in range(4)]
            fx.ptr_store(
                fx.Vector.from_elements(sh, fx.Float32), opart + (wave * 64 + lane) * 4
            )
            # every token's GEMV first, partials in opart past the shared ones: one
            # barrier and one reduce for all tokens, not two barriers per token
            for k in range_constexpr(tokens):
                # token k + 2's weights go out before token k's GEMV
                if const_expr(k + 2 < tokens):
                    dn_units[k + 2] = down_load(k + 2, t, has_dn)
                acc = [fx.Float32(0.0) for _ in range(4)]
                for wv, sc, sl, kc, wgt in dn_units.pop(k):
                    cu = mfma_f4f8(
                        wv,
                        sc,
                        lds_b8(
                            MID8_LDS + k * MID8_WORDS + (sl * INTER + kc * 128) // 4
                        ),
                        lds_i32(
                            MIDSC_LDS
                            + k * MIDSC_WORDS
                            + (sl * INTER + kc * 128) // 32
                            + g4
                        ),
                        fx.Vector.filled(4, 0.0, fx.Float32),
                    )
                    acc = [acc[j4] + cu[j4] * wgt for j4 in range(4)]
                fx.ptr_store(
                    fx.Vector.from_elements(acc, fx.Float32),
                    opart + ((k + 1) * WAVES * 64 + wave * 64 + lane) * 4,
                )
            gpu.barrier()
            if tid < DN_ROWS * tokens:
                k = tid // DN_ROWS
                grp = (tid % DN_ROWS) // 16
                rr = tid % 16
                tot = fx.Float32(0.0)
                for w in range_constexpr(4):
                    tot = tot + fx.ptr_load(
                        opart
                        + (
                            ((k + 1) * WAVES * 64 + (grp * 4 + w) * 64 + 16 * (rr // 4))
                            * 4
                            + rr % 4
                        )
                    )
                for w in range_constexpr(4):
                    tot = tot + fx.ptr_load(
                        opart + (((grp * 4 + w) * 64 + 16 * (rr // 4) + k) * 4 + rr % 4)
                    )
                fx.ptr_store(bf16_round(tot), misc + tid)
            gpu.barrier()
            ffn_finish(t)

        (stage_moe_wide if WIDE else stage_moe)()
        flush_stamps()

    @flyc.jit
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
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        tl: Int64,
        k1_args: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
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
            step,
            rank,
            layer,
            bt_width,
            q_len,
            tl,
            k1_args,
            positions,
            slot_mapping,
            res,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch_post_attn
