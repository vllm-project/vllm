# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the GLM-5 MonoKernel of ROCm/FlyDSL#1204 (@coderfeli),
# kernels/monokernel/glm/kernel.py, and ROCm/ATOM#2435 (@XiaobingSuper),
# atom/model_ops/monokernel/glm/kernel.py (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors.
# ruff: noqa: B008, B023, E501, SIM102

"""GLM-5.3-Flash FFN-layer MonoKernel: one persistent launch per TP rank.

``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs one MoE layer's
FFN for this rank's TP shard::

    seam: TP sum of vLLM's attention partials -> mHC post -> R' + mix partials
      -> gate: mixes -> pre / post / comb (Sinkhorn) -> RMSNorm'd FFN input row
      -> router GEMV (bf16) -> sigmoid scores + the rows' MXFP8 activation blocks
      -> top-8 of 288 (sigmoid + bias) -> 1 shared + 8 routed expert up/gate,
         clamped SiLU -> mid MXFP8 quant -> expert down + route weighting
      -> MoE TP peer reduce -> x_out

Every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order. There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every bounded spin wait makes progress.

Mailboxes are tagged pairs ``(value, tag)`` with this launch's epoch as the tag;
a consumer polls the payload until the tags match. Each rank pushes its MoE
partial rows into every peer's symmetric buffer and sums the TP partials in
rank order, so every rank produces bit-identical outputs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.glm5next.amd.specialized_decode.mono import abi
from vllm.models.glm5next.amd.specialized_decode.mono.config import (
    EPS,
    FP8_MAX,
    FWHT_SCALE,
    HC_EPS,
    HC_MULT,
    HC_POST_MULT,
    HIDDEN,
    INDEX_DIM,
    INDEX_EPS,
    INDEX_HEADS,
    KDA_DIM,
    KDA_LOWER_BOUND,
    KPOOL,
    KV_LORA,
    L2_EPS,
    MLA_SCALE,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    O_NORM_EPS,
    Q_LORA,
    ROUTE_SCALE,
    ROUTER_CANDIDATES,
    SHARED_EXPERT,
    SINKHORN_ITERS,
    SWIGLU_LIMIT,
    TOP_K,
    TOPK_POOLS,
    TOPK_TOKENS,
    V_DIM,
    FlashShard,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
    DSA_TAB,
    FGB_TILE,
    IN_TILE,
    N_MIX,
    N_QKV,
    N_QKV_FP8,
    N_ROUTER,
    N_ROW_TILES,
    N_SPLIT,
    O_TILE,
    PMIX,
    POLL_STAGES,
    POOLS_PER_TASK,
    QB_TILE,
    QKV_TILE,
    ROW_BT,
    ROW_IBT,
    ROW_POOL_SLOT,
    ROW_POS,
    ROW_SLOT,
    ROW_TAIL_SLOT,
    ROW_WORDS,
    SPLIT_KEYS,
    UK_TILE,
    UV_TILE,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    layout,
    sample_wave_batches,
    stage_tasks,
    ug_task_rounds,
)
from vllm.models.glm5next.amd.specialized_decode.mono.layout import (
    BLOCKS,
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    ROUTER_TILE,
    ROW_TILE,
    THREADS,
    UG_TILE,
    WAVES,
    atom_mxfp4_scale_index,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ops import (
    REALTIME_PER_US,
    bpermute_i32,
    div_rn,
    fp8_pack4,
    mem_realtime,
    read_lane_i32,
    spin_pause,
    wave_umax_dpp,
    write_lane_i32,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ops import exp as _exp
from vllm.models.glm5next.amd.specialized_decode.mono.ops import (
    fp8_to_bf16x8 as _fp8_to_bf16x8,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ops import rcp as _rcp
from vllm.models.glm5next.amd.specialized_decode.mono.ops import rsq as _rsq
from vllm.models.glm5next.amd.specialized_decode.mono.ops import rsrc as _rsrc
from vllm.models.glm5next.amd.specialized_decode.mono.ops import uniform as _uniform
from vllm.models.glm5next.amd.specialized_decode.mono.ops import (
    uniform_f32 as _uniform_f32,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ops import xred as _xred
from vllm.models.glm5next.amd.specialized_decode.mono.ops import xshfl as _xshfl

# 5-key descending sorting network (lane's router candidates)
SORT5 = ((0, 1), (3, 4), (2, 4), (2, 3), (0, 3), (0, 2), (1, 4), (1, 3), (1, 2))
# ID encoding for the tie-break: the larger of (ID_BASE - id) is the smaller id
ID_BASE = 511
assert N_EXPERTS < ID_BASE and ROUTER_CANDIDATES == 5


def build_flash_ffn_kernel(
    S: int,
    rank: int = 0,
    npes: int = 4,
    launches_per_step: int = 1,
    inter: int = 512,
    poll_timeout_us: int | None = None,
    poll_early_out: bool = False,
    kda: bool = False,
    conv_dim_first: bool = True,
    dsa: bool = False,
    max_pools: int = 0,
    dsa_fp8: tuple[bool, bool, bool] = (True, True, True),
    kv_fp8: bool = False,
    tpr: int = 1,
):
    """Return the ``@flyc.jit`` launcher of one rank's FFN layer for S rows; with
    ``kda`` the whole KDA decoder layer (attention-side mHC, projections, short conv,
    gated delta rule, gated RMSNorm, o_proj and its TP reduce) runs first; with
    ``dsa`` the whole sparse-attention decoder layer (attention-side mHC, q / kv
    LoRA projections and norms, KV cache write, kpool indexer, sparse MLA, o_proj)
    for rows of at most ``max_pools`` kpool pools. ``dsa_fp8`` says which of q_a | kv_a,
    q_b and o_proj are FP8 with 128 x 128 block scales (else row-major bf16).
    ``kv_fp8``: the MLA latent cache holds e4m3 bytes over vLLM's per-tensor
    ``k_scale`` (written as vLLM's ``concat_and_cache_mla`` does, read back
    dequantized to bf16 for the attention).

    ``tpr`` > 1 is a speculative verify step: every request owns ``tpr``
    consecutive rows (its token and the drafts after it, in position order). KDA
    then walks a request's rows as one chain, as vLLM's spec-decode conv and
    recurrent kernels do: ``state_idx`` holds every row's state slot, then a
    request's accepted-token count; the conv state has ``2 + tpr`` columns.

    ``poll_timeout_us`` bounds every mailbox wait in time. An expired wait
    stores 1 into its stage's error word (``poll_err_addr``) and, with TP peers,
    into this rank's flag on rank 0 (``poll_xrank_addr``), then continues, so a
    stalled peer finishes the launch with wrong values instead of hanging. With
    ``poll_early_out`` every later wait of the same step returns at its first retry.
    The shared expert is read from ``w_sh_*`` (one expert each), not the routed
    tensors, so it may be fused into them or stored apart.
    """
    assert 2 <= S <= 16 and 1 <= launches_per_step <= LAYER_SLOTS
    assert poll_timeout_us is None or poll_timeout_us > 0
    assert inter % 128 == 0
    assert not kda or S <= 8
    assert not (kda and dsa)
    assert not dsa or (S <= 8 and max_pools % POOLS_PER_TASK == 0)
    assert tpr >= 1 and S % tpr == 0
    # kpool tail ring slots a request: vLLM's ``Glm5NextTailCache`` size for
    # tpr - 1 draft tokens (KPOOL without speculative decoding)
    TAIL_RING = KPOOL * (1 << ((KPOOL + tpr - 1 + KPOOL - 1) // KPOOL - 1).bit_length())
    NREQ = S // tpr
    W = npes
    SH = FlashShard(W)
    P, H, N_IN = SH.kda_proj, SH.kda_heads, SH.kda_in_rows
    MH = SH.mla_heads
    HG = MH // WAVES
    QB_ROWS = MH * NOPE_DIM
    QKV8, QB8, O8 = dsa_fp8
    whole = kda or dsa
    POLL_TICKS = 0 if poll_timeout_us is None else poll_timeout_us * REALTIME_PER_US
    FAULT = abi.fault_spec()
    F_RANK, F_STEP, F_US = (
        (FAULT.rank, FAULT.step, FAULT.us) if FAULT is not None else (0, 0, 0)
    )
    expert_inter = inter
    SHARED = SHARED_EXPERT
    G = BLOCKS
    SC, SY = layout(S, W, expert_inter, kda, dsa, max_pools)
    DN_TILE = dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    # split attention LDS: bf16 q of the task's 8 heads, then its 64 keys' latent
    # rows (bf16 pairs); rows are padded by 4 words to spread MFMA operand banks
    QS = KV_LORA // 2 + 4
    KT_OFF = WAVES * QS
    SPLIT_WORDS = KT_OFF + SPLIT_KEYS * QS
    # top-512 pool select: a 4096-bin histogram, then (key, pool) candidates
    SEL_BINS = 4096
    SEL_ITEMS = 8
    SEL_CHUNK = THREADS * SEL_ITEMS
    LDS_KEYS = max(S * MOE_SLOTS, 320 if dsa else 0, SPLIT_KEYS if dsa else 0)
    # LDS X users: one router row (bf16), every row's MXFP8 FFN input (ug), the
    # seam tile (x_attn + R', and R after the layer's own attention), every row's
    # MXFP8 mid (down); with KDA every row's bf16 attention input (inproj) and o
    # (oproj); with DSA the attention input, q_a, a split tile, the pool select
    X_WORDS = max(
        HIDDEN // 2,
        S * HIDDEN // 4,
        S * ROW_TILE * (1 + 2 * HC_MULT) if whole else S * ROW_TILE * (1 + HC_MULT),
        down_x_words(S, expert_inter),
        S * HIDDEN // 2 if whole else 0,
        S * P // 2 if kda else 0,
        S * KDA_DIM if kda else 0,
        S * MH * V_DIM // 2 if dsa else 0,
        SPLIT_WORDS if dsa else 0,
        SEL_BINS + 2 * 2048 if dsa else 0,
    )
    SEL_CAP = (X_WORDS - SEL_BINS) // 2
    # fp8 KV write: a row's 512 normed values, past the kpool compress's words
    KVX = 2 * INDEX_DIM
    assert not kv_fp8 or (dsa and KVX + KV_LORA <= X_WORDS)
    MISC_OFF = X_WORDS
    MISC_WORDS = max(8 + S * XQ_GROUPS, S * MOE_SLOTS * (expert_inter // 32))
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE, S * DN_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    order = stage_tasks(S, expert_inter, W, kda, dsa, tpr)
    base, acc = {}, 0
    for name, n in order:
        base[name] = acc % G
        acc += n

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def flash_ffn_kernel(
        attn_part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        g_norm: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        x_out: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        s_sh_ug: Int64,
        w_sh_dn: Int64,
        s_sh_dn: Int64,
        hc_a_fn: Int64,
        hc_a_scale: Int64,
        hc_a_base: Int64,
        g_in: Int64,
        w_in: Int64,
        w_fb: Int64,
        w_gb: Int64,
        conv_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        w_on: Int64,
        w_o: Int64,
        conv_state: Int64,
        rec_state: Int64,
        state_idx: Int64,
        conv_stride: Int64,
        rec_stride: Int64,
        dsa_tab: Int64,
        dsa_rows: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step; ``layer`` makes it unique
        # per layer within the step.
        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * launches_per_step + layer + 1
        peer_slot = (step_value * launches_per_step + layer) & 1
        r_peers = _rsrc(peers)
        # One wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(_uniform(pv[0]))
        )
        if const_expr(FAULT is not None):
            if (rank == F_RANK) & (step_value == F_STEP) & (layer == 0):
                t_fault = mem_realtime()
                while mem_realtime() - t_fault < fx.Int64(F_US * REALTIME_PER_US):
                    spin_pause()

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(
                fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16))
            )

        def ld_pair(r, elem):
            """bf16 elements elem, elem + 1 (elem even) of a plain tensor -> 2 f32."""
            return bf2_f32(
                fx.Int32(bo.buffer_load(r, elem // 2, vec_width=1, dtype=T.i32))
            )

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def div(a, b):
            return div_rn(a, b, _rcp(b))

        def sigmoid(a):
            return div(fx.Float32(1.0), 1.0 + _exp(-a))

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        # ---- tagged-pair mailboxes
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(
                fx.Vector.from_elements([bits, tag], fx.Int32),
                _rsrc(base_addr),
                i * 2,
                cache_modifier=cm,
            )

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [
                    fx.Float32(v0).bitcast(fx.Int32),
                    tag,
                    fx.Float32(v1).bitcast(fx.Int32),
                    tag,
                ],
                fx.Int32,
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                _rsrc(base_addr),
                i,
                cache_modifier=cm,
            )

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(
                fx.Float32
            )

        stage_now = ["seam"]  # trace-time stage name; names an expired poll

        def addr_at(name):
            w = fx.Vector(bo.buffer_load(_rsrc(mb(name)), 0, vec_width=2, dtype=T.i32))
            return (fx.Int64(w[1]) << 32) | fx.Int64(fx.Uint32(w[0]))

        def store_flag(addr):
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                    addr,
                ),
                fx.Int32(1),
                memory_order=fx.AtomicOrdering.Monotonic,
            )

        def poll_expired():
            """Set this stage's error word and, with TP peers, this rank's flag on
            rank 0, fenced before anything derived from the expired wait is posted."""
            stage_index = POLL_STAGES.index(stage_now[0])
            store_flag(addr_at("poll_err_addr") + fx.Int64(4 * stage_index))
            if const_expr(W > 1):
                store_flag(addr_at("poll_xrank_addr"))
                fx.memory_fence(ordering=fx.AtomicOrdering.Release)

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index,
            npairs in {1, 2})]; re-load the batch while any tag is stale."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(
                    specs[batch:], scope, batch
                )
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(
                            _rsrc(b),
                            fx.Int32(i) * 2,
                            vec_width=2 * n,
                            dtype=T.i32,
                            cache_modifier=cm,
                        )
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            v = load_all()
            if const_expr(poll_timeout_us is None):
                while pending(v):
                    spin_pause()
                    v = load_all()
            else:
                # the clock starts at the first re-poll, so a wait that is
                # already satisfied reads no timer
                timeout = fx.Int64(POLL_TICKS)
                t0 = fx.Int64(0)
                waited = fx.Int64(0)
                if const_expr(poll_early_out):
                    abort_mark = step_value + 1
                    r_abort = _rsrc(mb("poll_abort"))
                    ab = fx.Int32(0)
                    while pending(v) & (waited < timeout) & (ab != abort_mark):
                        spin_pause()
                        v = load_all()
                        now = mem_realtime()
                        t0 = (t0 == fx.Int64(0)).select(now, t0)
                        waited = now - t0
                        ab = fx.Int32(
                            bo.buffer_load(
                                r_abort,
                                0,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                else:
                    while pending(v) & (waited < timeout):
                        spin_pause()
                        v = load_all()
                        now = mem_realtime()
                        t0 = (t0 == fx.Int64(0)).select(now, t0)
                        waited = now - t0
                if pending(v):
                    poll_expired()
                    if const_expr(poll_early_out):
                        bo.buffer_store(abort_mark, r_abort, 0, cache_modifier=CM_DEV)
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [
                v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])
            ]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [
                (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
                for v in poll([(b, i, 2) for b, i in specs])
            ]

        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ MFMA GEMV machinery
        def unit_native_mxfp4(
            w_rsrc, s_rsrc, rg, kc, K, b_word, b_scale_word, coef=None, ln=None
        ):
            """One 16-byte/lane AITER-shuffled W4 tile for the scaled MFMA."""
            ln = lane if ln is None else ln
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale = fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    atom_mxfp4_scale_index(
                        rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                    ),
                    vec_width=1,
                    dtype=T.i8,
                )
            )
            return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

        def lds_mxfp8(chunk_word):
            """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
            words = [
                fx.Vector(
                    fx.ptr_load(
                        xs + chunk_word + 16 * half + 4 * (lane // 16),
                        result_type=v4f,
                    )
                ).bitcast(fx.Int32)
                for half in range(2)
            ]
            return fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)],
                fx.Int32,
            )

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt == "native_mxfp4"):
                    raw, weight_scale, activation_scale_word = wv
                    activation_scale = lds_ld(misc, activation_scale_word).bitcast(
                        fx.Int32
                    )
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                raw,
                                lds_mxfp8(bw),
                                c,
                                4,
                                0,
                                0,
                                weight_scale,
                                0,
                                activation_scale,
                            ],
                        )
                    )
                if const_expr(fmt == "bf16"):
                    for sp in range_constexpr(2):
                        a = wv[sp].bitcast(fx.BFloat16)
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(
                            fx.BFloat16
                        )
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(fmt == "fp8rm"):
                    # lane group g holds K 32 g .. 32 g + 31 of the block; MFMA i
                    # takes its values 8 i .. 8 i + 7 (exactly widened to bf16)
                    for i in range_constexpr(4):
                        a = _fp8_to_bf16x8(
                            wv[i // 2][(i % 2) * 2], wv[i // 2][(i % 2) * 2 + 1]
                        )
                        b = fx.ptr_load(xs + (bw + i * 4), result_type=v4f).bitcast(
                            fx.BFloat16
                        )
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                elif const_expr(isinstance(coef, tuple)):
                    # (column mask, factor): a select, so a non-finite row stays
                    # in its own column
                    mask, factor = coef
                    acc = [
                        acc[e] + mask.select(c[e] * factor, fx.Float32(0.0))
                        for e in range(4)
                    ]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(4)]
            return acc

        def run_units(make_unit, cpw, batch, pre=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            ``pre`` = the already-issued first batch (prefetched before a wait)."""
            acc = [fx.Float32(0.0) for _ in range(4)]
            starts = list(range(0, cpw, batch))
            cur = (
                pre
                if pre is not None
                else [make_unit(c) for c in range(0, min(batch, cpw))]
            )
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                acc = mma_units(acc, cur)
                cur = nxt
            return acc

        def reduce_rows(R, acc, emit, count=S):
            """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
            wpr = WAVES // R
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            gpu.barrier()
            n_out = R * 16 * count
            for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                t = tid + i * THREADS
                if t < n_out:
                    rl = t % (R * 16)
                    n = t // (R * 16)
                    r = rl % 16
                    tot = fx.Float32(0.0)
                    for j in range_constexpr(wpr):
                        ww = (rl // 16) * wpr + j
                        tot = tot + lds_ld(
                            red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4
                        )
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def quant_mxfp8(a0, a1):
            """Per-32 E4M3 values and E8M0 scale used by AITER W4A8."""
            amax = fx.max(fmath.absf(a0), fmath.absf(a1))
            for offset in (1, 2, 4, 8):
                amax = _xred(amax, offset, fx.max)
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            exponent = (bits >> 23) & fx.Int32(0xFF)
            exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
            inverse = _rcp((exponent << 23).bitcast(fx.Float32))
            q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
            return q0, q1, exponent

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of ``samples``
            into LDS."""
            nxw = HIDDEN // 4 // THREADS
            nscale = XQ_GROUPS
            got = poll(
                [
                    (mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1)
                    for sx in samples
                    for i in range(nxw)
                ]
                + [
                    (mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1)
                    for sx in samples
                ]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = tid + i * THREADS
                    lds_st(
                        xs,
                        j * (HIDDEN // 4) + wd,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )
                if tid < nscale:
                    lds_st(
                        misc,
                        8 + j * nscale + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )

        def st_mxfp8(k, q0, q1):
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

        def candidate(i):
            """This lane's router candidate i: expert lane + 64 i, or None past N_EXPERTS."""
            return lane + 64 * i

        def load_bias():
            """This lane's candidate biases (issue before the scores wait)."""
            return [
                ld_f32(_rsrc(bias), fx.min(candidate(i), N_EXPERTS - 1))
                for i in range(ROUTER_CANDIDATES)
            ]

        def route_top8(s, bs):
            """Top-8 of sample s over 288 experts (one whole wave, after the scores).

            Every FP32 bit of (sigmoid + bias) is kept; equal keys go to the smaller
            expert id. Returns (expert id, route weight = raw score / sum of the 8 raw
            scores * ROUTE_SCALE) of pick ``lane``, valid in lanes < TOP_K."""
            raws = getf_many(
                [
                    (mb("scores"), s * N_EXPERTS + fx.min(candidate(i), N_EXPERTS - 1))
                    for i in range(ROUTER_CANDIDATES)
                ]
            )
            ks, ids = [], []
            for i in range_constexpr(ROUTER_CANDIDATES):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                valid = candidate(i) < N_EXPERTS
                ks.append(valid.select(fx.Uint32(ok), fx.Uint32(0)))
                ids.append(valid.select(candidate(i), fx.Int32(N_EXPERTS)))
            for a, b in SORT5:
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax_dpp(ks[0])
                winner = wave_umax_dpp(
                    (ks[0] == m).select(ID_BASE - ids[0], fx.Int32(0))
                )
                expert = ID_BASE - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                n = ROUTER_CANDIDATES
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(n)] + [ks[n]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(n)] + [ids[n]]
                mv = write_lane_i32(expert, k, mv)
            e = mv
            src = (e % 64) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, ROUTER_CANDIDATES):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, out_fn, tile):
            """Push BF16 partials to every peer, then sum them in rank order."""
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + region_base,
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [
                                    lds_ld(outs, si * tile + ri),
                                    lds_ld(outs, si * tile + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                else:
                    own = sym + region_base
                    specs = [
                        (own, ((src * S + s) * HIDDEN + row) // 2, 1)
                        for src in range(W)
                    ]
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got]
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, t0, t1)

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name):
            stage_now[0] = name

        def n_sel(count=S):
            """This lane's local MFMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # ========= 1. seam: attention TP reduce + mHC post -> R', partial mixes
        # One task a 32-column tile, every row: x = the TP sum of the attention
        # partials (bf16) or, on the attention side, the previous layer's output;
        # R'[j] = bf16(sum_i comb[i][j] R[i] + post[j] x) -> res_out and mailbox
        # ``rnew``, and the tile's share of the mix projection (fn . R') and of
        # sum(R'^2) -> mailbox ``pmix``.
        XA = 0
        RN = S * ROW_TILE
        RA = RN + S * HC_MULT * ROW_TILE
        NPAIR = S * ROW_TILE // 2
        MIX_A = HC_MULT * (1 + HC_MULT)

        def run_seam(stage, mode):
            """``mode``: "ffn" (vLLM's attention partials), "attn" (the previous
            layer's output, attention-side mHC) or "ffn_own" (this launch's o_proj
            partials, R / post / comb from its attention-side mHC)."""
            r_attn = _rsrc(attn_part)
            r_res = _rsrc(res_in)
            r_fn = _rsrc(hc_a_fn if mode == "attn" else hc_fn)
            r_post = _rsrc(post_in)
            r_comb = _rsrc(comb_in)
            rnew = mb("rnew_a" if mode == "attn" else "rnew")
            pmix = mb("pmix_a" if mode == "attn" else "pmix")

            def seam_post(item, col0):
                """R'[j] for one (row, stream, column pair) item of a seam tile."""
                s = item // (HC_MULT * (ROW_TILE // 2))
                j = (item // (ROW_TILE // 2)) % HC_MULT
                c = (item % (ROW_TILE // 2)) * 2
                if const_expr(mode == "ffn_own"):
                    pj = lds_ld(misc, s * MIX_A + j)
                else:
                    pj = ld_f32(r_post, s * HC_MULT + j)
                acc0 = fx.Float32(0.0)
                acc1 = fx.Float32(0.0)
                for i in range_constexpr(HC_MULT):
                    if const_expr(mode == "ffn_own"):
                        cij = lds_ld(misc, s * MIX_A + HC_MULT + i * HC_MULT + j)
                        ra = RA + (s * HC_MULT + i) * ROW_TILE + c
                        r0, r1 = lds_ld(xs, ra), lds_ld(xs, ra + 1)
                    else:
                        cij = ld_f32(r_comb, (s * HC_MULT + i) * HC_MULT + j)
                        r0, r1 = ld_pair(r_res, (s * HC_MULT + i) * HIDDEN + col0 + c)
                    acc0 = acc0 + cij * r0
                    acc1 = acc1 + cij * r1
                n0 = bf16_round(acc0 + pj * lds_ld(xs, XA + s * ROW_TILE + c))
                n1 = bf16_round(acc1 + pj * lds_ld(xs, XA + s * ROW_TILE + c + 1))
                elem = (s * HC_MULT + j) * HIDDEN + col0 + c
                if const_expr(mode != "attn"):
                    bo.buffer_store(
                        fx.Vector.from_elements([n0, n1], fx.Float32).to(fx.BFloat16),
                        _rsrc(res_out),
                        elem,
                    )
                put(rnew, elem // 2, bf16_pair(n0, n1))
                lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c, n0)
                lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c + 1, n1)

            for t in range(start(stage), N_ROW_TILES, G):
                t = fx.Int32(t)
                stamp(stage)
                col0 = t * ROW_TILE
                attn_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(
                    SY["_part_stride"]
                )
                if const_expr(mode == "ffn" and W > 1):
                    if wave < W:
                        for batch in range_constexpr((NPAIR + 63) // 64):
                            pair = lane + batch * 64
                            if pair < NPAIR:
                                si = pair // (ROW_TILE // 2)
                                ci = (pair % (ROW_TILE // 2)) * 2
                                word = fx.Int32(
                                    bo.buffer_load(
                                        r_attn,
                                        (si * HIDDEN + col0 + ci) // 2,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                bo.buffer_store(
                                    fx.Vector.from_elements([word, tag], fx.Int32),
                                    _rsrc(peer_dst + attn_base),
                                    (rank * S + si) * HIDDEN + col0 + ci,
                                    cache_modifier=CM_SYS,
                                )
                    gpu.barrier()
                if const_expr(mode == "ffn_own"):
                    NRA = S * HC_MULT * ROW_TILE // 2
                    for ra_i in range_constexpr((NRA + THREADS - 1) // THREADS):
                        q = tid + ra_i * THREADS
                        if q < NRA:
                            sj = q // (ROW_TILE // 2)
                            c = (q % (ROW_TILE // 2)) * 2
                            a0, a1 = bf2_f32(
                                get(mb("rnew_a"), (sj * HIDDEN + col0 + c) // 2)
                            )
                            lds_st(xs, RA + sj * ROW_TILE + c, a0)
                            lds_st(xs, RA + sj * ROW_TILE + c + 1, a1)
                    if tid < S * MIX_A:
                        lds_st(misc, tid, getf_many([(mb("mix_a"), tid)])[0])
                if tid < NPAIR:
                    s = tid // (ROW_TILE // 2)
                    c = (tid % (ROW_TILE // 2)) * 2
                    if const_expr(mode == "attn" or (mode == "ffn" and W == 1)):
                        v0, v1 = ld_pair(r_attn, s * HIDDEN + col0 + c)
                    else:
                        own = sym + attn_base
                        got = poll(
                            [
                                (own, ((src * S + s) * HIDDEN + col0 + c) // 2, 1)
                                for src in range(W)
                            ],
                            "one-as",
                        )
                        v0 = fx.Float32(0.0)
                        v1 = fx.Float32(0.0)
                        for src in range_constexpr(W):
                            a0, a1 = bf2_f32(got[src][0])
                            v0 = v0 + a0
                            v1 = v1 + a1
                        v0, v1 = bf16_round(v0), bf16_round(v1)
                    lds_st(xs, XA + s * ROW_TILE + c, v0)
                    lds_st(xs, XA + s * ROW_TILE + c + 1, v1)
                gpu.barrier()
                RN_ITEMS = S * HC_MULT * (ROW_TILE // 2)
                for rn_i in range_constexpr((RN_ITEMS + THREADS - 1) // THREADS):
                    item = tid + rn_i * THREADS
                    if item < RN_ITEMS:
                        seam_post(item, col0)
                gpu.barrier()
                if tid < S * PMIX:
                    s = tid // PMIX
                    k = tid % PMIX
                    is_mix = k < N_MIX
                    krow = fx.min(k, N_MIX - 1) * (HC_MULT * HIDDEN)
                    acc = fx.Float32(0.0)
                    for j in range_constexpr(HC_MULT):
                        for c in range_constexpr(ROW_TILE):
                            v = lds_ld(xs, RN + (s * HC_MULT + j) * ROW_TILE + c)
                            w = ld_f32(r_fn, krow + j * HIDDEN + col0 + c)
                            acc = acc + v * is_mix.select(w, v)
                    put(pmix, (s * N_ROW_TILES + t) * PMIX + k, acc)
                gpu.barrier()

        # ========= 2. gate: mixes -> pre / post / comb (Sinkhorn), layer input row
        # One CTA a row. The tile partials are summed in a fixed order, so every
        # rank computes bit-identical mixes.
        NP = 16
        TPP = N_ROW_TILES // NP
        CPT = HIDDEN // THREADS

        def run_gate(stage, mode):
            """``mode`` "attn" writes post / comb to mailbox ``mix_a`` and the
            attention input to ``a_in``; "ffn" writes them out and the FFN input
            to ``a``."""
            attn = mode == "attn"
            r_scale = _rsrc(hc_a_scale if attn else hc_scale)
            r_base = _rsrc(hc_a_base if attn else hc_base)
            r_g = _rsrc(g_in if attn else g_norm)
            pmix = mb("pmix_a" if attn else "pmix")
            rnew = mb("rnew_a" if attn else "rnew")
            a_out = mb("a_in" if attn else "a")
            for t in range(start(stage), S, G):
                s = fx.Int32(t)
                stamp(stage)
                gamma = [ld_bf16(r_g, tid * CPT + q) for q in range(CPT)]
                if tid < PMIX * NP:
                    k = tid // NP
                    part = tid % NP
                    vals = getf_many(
                        [
                            (pmix, (s * N_ROW_TILES + part * TPP + q) * PMIX + k)
                            for q in range(TPP)
                        ]
                    )
                    tot = vals[0]
                    for q in range_constexpr(1, TPP):
                        tot = tot + vals[q]
                    lds_st(red, k * NP + part, tot)
                gpu.barrier()
                if tid < PMIX:
                    tot = lds_ld(red, tid * NP)
                    for p_ in range_constexpr(1, NP):
                        tot = tot + lds_ld(red, tid * NP + p_)
                    lds_st(misc, tid, tot)
                gpu.barrier()
                if tid == 0:
                    rinv = _rsq(lds_ld(misc, N_MIX) * (1.0 / (HC_MULT * HIDDEN)) + EPS)
                    m = [lds_ld(misc, k) * rinv for k in range(N_MIX)]
                    sc = [ld_f32(r_scale, i) for i in range(3)]
                    hb = [ld_f32(r_base, k) for k in range(N_MIX)]
                    for i in range_constexpr(HC_MULT):
                        pre_i = sigmoid(m[i] * sc[0] + hb[i]) + HC_EPS
                        lds_st(misc, 32 + i, pre_i)
                        post_i = (
                            sigmoid(m[HC_MULT + i] * sc[1] + hb[HC_MULT + i])
                            * HC_POST_MULT
                        )
                        if const_expr(attn):
                            put(mb("mix_a"), s * MIX_A + i, post_i)
                        else:
                            bo.buffer_store(post_i, _rsrc(post_out), s * HC_MULT + i)
                    o = 2 * HC_MULT
                    comb = []
                    for i in range_constexpr(HC_MULT):
                        row = [
                            m[o + i * HC_MULT + j] * sc[2] + hb[o + i * HC_MULT + j]
                            for j in range(HC_MULT)
                        ]
                        mx = row[0]
                        for j in range_constexpr(1, HC_MULT):
                            mx = fx.max(mx, row[j])
                        ex = [_exp(v - mx) for v in row]
                        den = ex[0]
                        for j in range_constexpr(1, HC_MULT):
                            den = den + ex[j]
                        comb.append([div(v, den) + HC_EPS for v in ex])

                    def norm_cols(cm):
                        for j in range_constexpr(HC_MULT):
                            den = cm[0][j]
                            for i in range_constexpr(1, HC_MULT):
                                den = den + cm[i][j]
                            den = den + HC_EPS
                            for i in range_constexpr(HC_MULT):
                                cm[i][j] = div(cm[i][j], den)
                        return cm

                    def norm_rows(cm):
                        for i in range_constexpr(HC_MULT):
                            den = cm[i][0]
                            for j in range_constexpr(1, HC_MULT):
                                den = den + cm[i][j]
                            den = den + HC_EPS
                            cm[i] = [div(v, den) for v in cm[i]]
                        return cm

                    comb = norm_cols(comb)
                    for _ in range_constexpr(SINKHORN_ITERS - 1):
                        comb = norm_cols(norm_rows(comb))
                    for i in range_constexpr(HC_MULT):
                        for j in range_constexpr(HC_MULT):
                            if const_expr(attn):
                                put(
                                    mb("mix_a"),
                                    s * MIX_A + HC_MULT + i * HC_MULT + j,
                                    comb[i][j],
                                )
                            else:
                                bo.buffer_store(
                                    comb[i][j],
                                    _rsrc(comb_out),
                                    (s * HC_MULT + i) * HC_MULT + j,
                                )
                gpu.barrier()
                pre = [lds_ld(misc, 32 + i) for i in range(HC_MULT)]
                got = poll(
                    [
                        (
                            rnew,
                            ((s * HC_MULT + i) * HIDDEN + tid * CPT + 4 * q) // 2,
                            2,
                        )
                        for i in range(HC_MULT)
                        for q in range(CPT // 4)
                    ]
                )
                li = []
                for q in range_constexpr(CPT // 4):
                    for h in range_constexpr(2):
                        for e in range_constexpr(2):
                            v = fx.Float32(0.0)
                            for i in range_constexpr(HC_MULT):
                                word = got[i * (CPT // 4) + q][h]
                                v = v + pre[i] * bf2_f32(word)[e]
                            li.append(bf16_round(v))
                ss = fx.Float32(0.0)
                for v in li:
                    ss = ss + v * v
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for q in range_constexpr(CPT // 2):
                    put(
                        a_out,
                        (s * HIDDEN + tid * CPT + 2 * q) // 2,
                        bf16_pair(
                            li[2 * q] * rstd * gamma[2 * q],
                            li[2 * q + 1] * rstd * gamma[2 * q + 1],
                        ),
                    )

        # ================================================= KDA attention stages
        KD = KDA_DIM
        XW = HIDDEN // 2
        CONV_TAPS = 4

        def unit_rm(w_rsrc, row0, kc, K, rows, b_word):
            """One 64-K bf16 MFMA unit of a row-major [rows, K] weight; rows past the
            end load the last row and are dropped by the caller."""
            r = fx.min(row0 + lane % 16, rows - 1)
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (r * K + kc * 64 + sp * 32 + (lane // 16) * 8) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def stage_rows(box, K):
            """Poll every row's bf16 [K] vector of mailbox ``box`` into LDS words
            ``n * K / 2 ..``, the B operand layout of ``unit_rm``."""
            nq = S * K // 4
            full = nq // THREADS
            if const_expr(full > 0):
                got = poll([(box, (tid + i * THREADS) * 2, 2) for i in range(full)])
                for i in range_constexpr(full):
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (tid + i * THREADS) * 2 + j,
                            got[i][j].bitcast(fx.Float32),
                        )
            if const_expr(nq % THREADS):
                q = tid + full * THREADS
                if q < nq:
                    v = poll([(box, q * 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, q * 2 + j, v[j].bitcast(fx.Float32))

        # ===================== 3. merged q | k | v | beta | f_a | g_a projection
        # One task a 16-row tile of this rank's row-major bf16 weight, every row;
        # outputs rounded to bf16 as vLLM's GEMM stores them.
        IN_CPW = HIDDEN // 64 // WAVES

        def run_inproj():
            r_w = _rsrc(w_in)
            for t in range(start("inproj"), (N_IN + IN_TILE - 1) // IN_TILE, G):
                t = fx.Int32(t)
                stamp("inproj")
                row0 = t * IN_TILE

                def u_in(c):
                    kc = wave * IN_CPW + c
                    return unit_rm(r_w, row0, kc, HIDDEN, N_IN, n_sel() * XW + kc * 32)

                pre = [u_in(c) for c in range(4)]
                stage_rows(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(u_in, IN_CPW, 4, pre)

                def emit(rl, n, v):
                    row = row0 + rl
                    if row < N_IN:
                        put(mb("proj"), n * N_IN + row, bf16_round(v))

                reduce_rows(1, acc, emit)
                gpu.barrier()

        # ===================== 4. f_b / g_b: the per-channel decay and output gates
        # One task 128 rows (a 16-row tile a wave, K = 128) of f_b or of g_b.
        FA = 3 * P + H

        def run_fgb():
            for t in range(start("fgb"), 2 * P // FGB_TILE, G):
                t = fx.Int32(t)
                stamp("fgb")
                half = t >= P // FGB_TILE
                src = FA + half.select(fx.Int32(KD), fx.Int32(0))
                if tid < S * KD // 2:
                    n = tid // (KD // 2)
                    w = tid % (KD // 2)
                    v0, v1 = get2_many([(mb("proj"), n * N_IN + src + 2 * w)])[0]
                    lds_st(xs, n * (KD // 2) + w, bf16_pair(v0, v1))
                gpu.barrier()
                row0 = t * FGB_TILE + wave * 16
                r_w = _rsrc(half.select(w_gb, w_fb))
                lr = row0 - half.select(fx.Int32(P), fx.Int32(0))
                units = [
                    unit_rm(r_w, lr, kc, KD, P, n_sel() * (KD // 2) + kc * 32)
                    for kc in range(KD // 64)
                ]
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], units)
                # C[r][n] sits in lane n + 16 (r // 4), element r % 4
                if lane % 16 < S:
                    for e in range_constexpr(4):
                        put(
                            mb("gates"),
                            (lane % 16) * 2 * P + row0 + 4 * (lane // 16) + e,
                            bf16_round(acc[e]),
                        )
                gpu.barrier()

        # ===================== 5. short conv + gated delta rule + gated RMSNorm
        # One task a (request, head); it walks the request's ``tpr`` rows in order
        # (one row without speculative decoding), carrying the conv window and the
        # recurrent state, as vLLM's spec-decode kernels do. A row whose state
        # index is not positive (padding) writes no state and outputs zeros.
        QO, KO, VO, G1O, G2O, OO = 0, KD, 2 * KD, 3 * KD, 4 * KD, 5 * KD
        # conv state columns: the 3 taps, plus one a draft token
        STATE_LEN = CONV_TAPS - 1 + tpr - 1

        def conv_at(ch, j):
            """Element of channel ``ch``, column ``j`` in a conv state slot, laid out
            [3P, columns] (dim first) or [columns, 3P]."""
            if const_expr(conv_dim_first):
                return ch * STATE_LEN + j
            return j * (3 * P) + ch

        def run_kda():
            r_cw = _rsrc(conv_w)
            r_a_log = _rsrc(a_log)
            r_dtb = _rsrc(dt_bias)
            r_won = _rsrc(w_on)
            r_idx = _rsrc(state_idx)

            def row_idx(row):
                return _uniform(
                    fx.Int32(bo.buffer_load(r_idx, row, vec_width=1, dtype=T.i32))
                )

            for t in range(start("kda"), NREQ * H, G):
                t = fx.Int32(t)
                stamp("kda")
                s0 = (t // H) * tpr
                h = t % H
                # the window starts at the request's last accepted token: column
                # (accepted - 1) of the conv state, recurrent slot of that row
                if const_expr(tpr == 1):
                    off = fx.Int32(0)
                else:
                    acc_n = row_idx(S + t // H)
                    off = fx.min(fx.max(acc_n - 1, fx.Int32(0)), fx.Int32(tpr - 1))
                cidx = row_idx(s0)
                cvalid = cidx > 0
                r_cs = _rsrc(
                    conv_state
                    + fx.Int64(fx.max(cidx, fx.Int32(0))) * conv_stride * fx.Int64(2)
                )
                # every thread loads (the last conv channel past 3 * KD) so the
                # window values exist outside the stores' branches
                tc = fx.min(tid, fx.Int32(3 * KD - 1))
                ch = (tc // KD) * P + h * KD + tc % KD
                win = [ld_bf16(r_cs, conv_at(ch, off + j)) for j in range(3)]
                keep = [win[1], win[2]]
                cw = [ld_f32(r_cw, ch * CONV_TAPS + j) for j in range(CONV_TAPS)]
                amp = _exp(ld_f32(r_a_log, h))
                vr = tid // 4
                k0 = (tid % 4) * 32
                rd = fx.Int64(fx.max(row_idx(s0 + off), fx.Int32(0)))
                r_rd = _rsrc(
                    rec_state + (rd * rec_stride + fx.Int64(h * KD * KD)) * fx.Int64(4)
                )
                hv = []
                for q in range_constexpr(8):
                    w4 = fx.Vector(
                        bo.buffer_load(
                            r_rd, vr * KD + k0 + 4 * q, vec_width=4, dtype=T.f32
                        )
                    )
                    hv += [w4[e] for e in range(4)]
                xin = []
                for j in range_constexpr(tpr):
                    s = s0 + j
                    idx = row_idx(s)
                    valid = idx > 0
                    x = getf_many([(mb("proj"), s * N_IN + ch)])[0]
                    a = cw[CONV_TAPS - 1] * x
                    for i in range_constexpr(CONV_TAPS - 1):
                        a = a + cw[i] * win[i]
                    act = bf16_round(div(a, 1.0 + _exp(-a)))
                    if tid < 3 * KD:
                        lds_st(xs, tid, act)
                    win = [win[1], win[2], x]
                    xin.append(x)
                    if tid < 2 * KD:
                        g = getf_many(
                            [
                                (
                                    mb("gates"),
                                    s * 2 * P + (tid // KD) * P + h * KD + tid % KD,
                                )
                            ]
                        )[0]
                        lds_st(xs, G1O + tid, g)
                    if tid == 0:
                        lds_st(
                            misc, 0, getf_many([(mb("proj"), s * N_IN + 3 * P + h)])[0]
                        )
                    gpu.barrier()
                    if wave < 2:
                        a0 = lds_ld(xs, wave * KD + lane)
                        a1 = lds_ld(xs, wave * KD + lane + 64)
                        ss = wave_sum(a0 * a0 + a1 * a1)
                        if lane == 0:
                            lds_st(misc, 1 + wave, ss)
                    gpu.barrier()
                    rq = _rsq(lds_ld(misc, 1) + L2_EPS) * (KD**-0.5)
                    rk = _rsq(lds_ld(misc, 2) + L2_EPS)
                    beta = sigmoid(lds_ld(misc, 0))
                    kk = [lds_ld(xs, KO + k0 + i) * rk for i in range(32)]
                    dot = fx.Float32(0.0)
                    for i in range_constexpr(32):
                        g = lds_ld(xs, G1O + k0 + i) + ld_f32(r_dtb, h * KD + k0 + i)
                        hv[i] = hv[i] * _exp(
                            div(KDA_LOWER_BOUND, 1.0 + _exp(-(amp * g)))
                        )
                        dot = dot + hv[i] * kk[i]
                    for o_ in (1, 2):
                        dot = _xred(dot, o_, lambda a_, b_: a_ + b_)
                    vn = beta * (lds_ld(xs, VO + vr) - dot)
                    o = fx.Float32(0.0)
                    for i in range_constexpr(32):
                        hv[i] = hv[i] + vn * kk[i]
                        o = o + hv[i] * (lds_ld(xs, QO + k0 + i) * rq)
                    for o_ in (1, 2):
                        o = _xred(o, o_, lambda a_, b_: a_ + b_)
                    if valid:
                        r_rs = _rsrc(
                            rec_state
                            + (fx.Int64(idx) * rec_stride + fx.Int64(h * KD * KD))
                            * fx.Int64(4)
                        )
                        for q in range_constexpr(8):
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    hv[4 * q : 4 * q + 4], fx.Float32
                                ),
                                r_rs,
                                vr * KD + k0 + 4 * q,
                            )
                    if tid % 4 == 0:
                        lds_st(
                            xs, OO + vr, valid.select(bf16_round(o), fx.Float32(0.0))
                        )
                    gpu.barrier()
                    if wave == 0:
                        e0 = 2 * lane
                        o0 = lds_ld(xs, OO + e0)
                        o1 = lds_ld(xs, OO + e0 + 1)
                        rstd = _rsq(
                            wave_sum(o0 * o0 + o1 * o1) * (1.0 / KD) + O_NORM_EPS
                        )
                        y0 = (
                            o0
                            * rstd
                            * ld_f32(r_won, e0)
                            * sigmoid(lds_ld(xs, G2O + e0))
                        )
                        y1 = (
                            o1
                            * rstd
                            * ld_f32(r_won, e0 + 1)
                            * sigmoid(lds_ld(xs, G2O + e0 + 1))
                        )
                        put(mb("o_in"), (s * P + h * KD + e0) // 2, bf16_pair(y0, y1))
                    gpu.barrier()
                # the window after the request: the two newest history columns, then
                # every row's input (vLLM keeps the drafts until acceptance is known)
                if cvalid:
                    if tid < 3 * KD:
                        for j, v in enumerate(keep + xin):
                            bo.buffer_store(
                                fx.Float32(v).to(fx.BFloat16), r_cs, conv_at(ch, j)
                            )

        # ===================== 6. o_proj partial -> every peer's attention region
        # One task a 16-row tile of the row-major [HIDDEN, K] weight; the FFN seam
        # sums the TP partials as it does vLLM's. ``unit_of(row0, chunk)`` issues
        # one K chunk (``cpw`` a wave, prefetched ``ahead`` at a time) of the weight.
        def run_oproj(unit_of, K, cpw, ahead):
            attn_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            for t in range(start("oproj"), HIDDEN // O_TILE, G):
                t = fx.Int32(t)
                stamp("oproj")
                row0 = t * O_TILE

                def u_o(c):
                    return unit_of(row0, wave * cpw + c)

                pre = [u_o(c) for c in range(ahead)]
                stage_rows(mb("o_in"), K)
                gpu.barrier()
                acc = run_units(u_o, cpw, ahead, pre)
                reduce_rows(1, acc, emit_out(O_TILE))
                gpu.barrier()
                if wave < W:
                    pair_count = S * O_TILE // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (O_TILE // 2)
                            ri = (pair % (O_TILE // 2)) * 2
                            put_bf(
                                peer_dst + attn_base,
                                (rank * S + si) * HIDDEN + row0 + ri,
                                [
                                    lds_ld(outs, si * O_TILE + ri),
                                    lds_ld(outs, si * O_TILE + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()

        # ================================================= sparse-attention stages
        # The step table ``dsa_rows`` holds, a row, (position, MLA cache slot, slot of
        # the index-cache pool the row completes, kpool tail slot, block-table row);
        # a padding row has position -1 and writes nothing.
        def row_word(s, k):
            return _uniform(
                fx.Int32(
                    bo.buffer_load(
                        _rsrc(dsa_rows), s * ROW_WORDS + k, vec_width=1, dtype=T.i32
                    )
                )
            )

        def ld_i32(r, i):
            return fx.Int32(bo.buffer_load(r, i, vec_width=1, dtype=T.i32))

        def start_at(b):
            """First task of a stage whose task 0 runs on CTA ``b``."""
            return (bid + (G - b % G)) & (G - 1)

        def wave_max(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, fx.max)
            return v

        def unit_fp8_rm(w_rsrc, s_rsrc, row0, kb, K, rows, b_word, scale_cols):
            """One 128-K block ``kb`` of a 16-row tile of a row-major FP8 [rows, K]
            weight with f32 [rows / 128, K / 128] block scales (``row0 % 16 == 0``)."""
            r = fx.min(row0 + lane % 16, rows - 1)
            word = (r * K + kb * 128 + (lane // 16) * 32) // 4
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, word + 4 * h, vec_width=4, dtype=T.i32)
                )
                for h in range(2)
            ]
            sc = ld_f32(s_rsrc, (row0 // 128) * scale_cols + kb)
            return ("fp8rm", wv, sc, b_word + (lane // 16) * 16)

        def proj_unit(fp8, w_rsrc, s_rsrc, row0, kc, K, rows, n_words):
            """K chunk ``kc`` (128 FP8 or 64 bf16 values) of a 16-row tile of a
            row-major [rows, K] projection; the B row starts at LDS word n_words."""
            if const_expr(fp8):
                return unit_fp8_rm(
                    w_rsrc, s_rsrc, row0, kc, K, rows, n_words + kc * 64, K // 128
                )
            return unit_rm(w_rsrc, row0, kc, K, rows, n_words + kc * 32)

        def fwht128(x0, x1):
            """Hadamard-128 of one wave's vector (lane holds d = lane and lane + 64):
            butterflies of stride 1 .. 64 in vLLM's order, then the 1/sqrt(128)."""
            for sh in range_constexpr(6):
                st = 1 << sh
                upper = (lane & st) != 0
                p0, p1 = _xshfl(x0, st), _xshfl(x1, st)
                x0 = upper.select(p0 - x0, x0 + p0)
                x1 = upper.select(p1 - x1, x1 + p1)
            x0, x1 = x0 + x1, x0 - x1
            return x0 * FWHT_SCALE, x1 * FWHT_SCALE

        def ue8m0_quant(x0, x1):
            """Per-vector FP8 (vLLM's) of one wave's 128 values: scale =
            2^ceil(log2(max(absmax, 1e-4) / 448)); returns (q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(x0), fmath.absf(x1)))
            y = fx.max(amax, fx.Float32(1.0e-4)) * fx.Float32(1.0 / FP8_MAX)
            bits = y.bitcast(fx.Int32)
            e = (bits >> 23) & fx.Int32(0xFF)
            e = e + ((bits & fx.Int32(0x7FFFFF)) != 0).select(fx.Int32(1), fx.Int32(0))
            scale = (e << 23).bitcast(fx.Float32)
            inv = ((fx.Int32(254) - e) << 23).bitcast(fx.Float32)
            q0 = fx.min(fx.max(x0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(x1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, scale

        # ========= q_a | kv_a (FP8 or bf16) and wk | weights_proj | kpool gate (bf16)
        # One task a 16-row tile, every row. Outputs are bf16-rounded as vLLM's
        # GEMMs store them, except the indexer head weights (fp32 in vLLM).
        def dsa_qkv_fp8(tv, t32):
            r_wqkv, r_sqkv = _rsrc(tv["w_qkv"]), _rsrc(tv["s_qkv"])
            QKV_CPW = HIDDEN // (128 if QKV8 else 64) // WAVES
            for t in range(start("qkv_a"), N_QKV_FP8, G):
                t = fx.Int32(t)
                stamp("qkv_a")
                row0 = t * QKV_TILE

                def u8(c):
                    return proj_unit(
                        QKV8,
                        r_wqkv,
                        r_sqkv,
                        row0,
                        wave * QKV_CPW + c,
                        HIDDEN,
                        Q_LORA + KV_LORA,
                        n_sel() * XW,
                    )

                pre = [u8(c) for c in range(4)]
                stage_rows(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(u8, QKV_CPW, 4, pre)

                def emit_qkv(rl, n, v):
                    row = row0 + rl
                    v = bf16_round(v)
                    if row < Q_LORA:
                        put(mb("qa"), n * Q_LORA + row, v)
                    else:
                        put(mb("kva"), n * KV_LORA + row - Q_LORA, v)

                reduce_rows(1, acc, emit_qkv)
                gpu.barrier()

        def dsa_qkv_bf16(tv, t32):
            N_IK = (INDEX_DIM + INDEX_HEADS) // QKV_TILE
            IDX_CPW = HIDDEN // 64 // WAVES
            for tb in range(start_at(base["qkv_a"] + N_QKV_FP8), N_QKV - N_QKV_FP8, G):
                tb = fx.Int32(tb)
                stamp("qkv_a")
                in_ik = tb < N_IK
                r_wb = _rsrc(in_ik.select(tv["w_idx"], tv["w_gate"]))
                lrow0 = in_ik.select(tb, tb - N_IK) * QKV_TILE
                nrows = in_ik.select(
                    fx.Int32(INDEX_DIM + INDEX_HEADS), fx.Int32(INDEX_DIM)
                )

                def ub(c):
                    kc = wave * IDX_CPW + c
                    return unit_rm(
                        r_wb, lrow0, kc, HIDDEN, nrows, n_sel() * XW + kc * 32
                    )

                pre = [ub(c) for c in range(4)]
                stage_rows(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(ub, IDX_CPW, 4, pre)

                def emit_idx(rl, n, v):
                    row = tb * QKV_TILE + rl
                    if row < INDEX_DIM:
                        put(mb("ik"), n * INDEX_DIM + row, bf16_round(v))
                    elif row < INDEX_DIM + INDEX_HEADS:
                        put(mb("iw"), n * INDEX_HEADS + row - INDEX_DIM, v)
                    else:
                        put(
                            mb("ig"),
                            n * INDEX_DIM + row - INDEX_DIM - INDEX_HEADS,
                            bf16_round(v),
                        )

                reduce_rows(1, acc, emit_idx)
                gpu.barrier()

        # ======================================= q_a RMSNorm, one task a row
        def dsa_qnorm(tv, t32):
            r_gqa = _rsrc(tv["g_qa"])
            NQ4 = Q_LORA // 4
            for t in range(start("qnorm"), S, G):
                s = fx.Int32(t)
                stamp("qnorm")
                live = tid < NQ4
                e0 = fx.min(tid, NQ4 - 1) * 4
                gamma = [ld_bf16(r_gqa, e0 + j) for j in range(4)]
                v2 = get2_many(
                    [(mb("qa"), s * Q_LORA + e0), (mb("qa"), s * Q_LORA + e0 + 2)]
                )
                xv = [v2[0][0], v2[0][1], v2[1][0], v2[1][1]]
                ss = xv[0] * xv[0] + xv[1] * xv[1] + xv[2] * xv[2] + xv[3] * xv[3]
                total = block_sum(live.select(ss, fx.Float32(0.0)))
                rstd = _rsq(total * (1.0 / Q_LORA) + EPS)
                if live:
                    put_bf(
                        mb("q_an"),
                        s * Q_LORA + e0,
                        [xv[j] * rstd * gamma[j] for j in range(4)],
                    )
                gpu.barrier()

        # ===== KV RMSNorm -> MLA cache; index k LayerNorm; kpool update, a row
        # A row whose position completes a pool (pos % 4 == 3) compresses the
        # pool into the index cache: per dim softmax(gate + ape) over its four
        # tokens' (k, gate) (three from the tail ring, the row's own), the
        # weighted k, bf16, Hadamard, bf16, ue8m0 FP8. Then the row's (k, gate)
        # goes to ring slot pos % 4. ``kv_ready`` releases both caches.
        def dsa_cache(tv, t32):
            KVB = t32["kv_block_tokens"]
            KVE = tv["kv_block_elems"]
            IPB = t32["idx_block_pools"]
            IBB = t32["idx_block_bytes"]
            TAIL_BLK = t32["tail_block_elems"]
            TAIL_GATE = t32["tail_gate_elems"]
            r_gkv = _rsrc(tv["g_kv"])
            r_lnw, r_lnb = _rsrc(tv["ln_w"]), _rsrc(tv["ln_b"])
            r_ape = _rsrc(tv["ape"])
            # a request's rows in position order: a draft that completes a pool
            # takes the earlier drafts' (k, gate) from this task, older tokens from
            # the ring; its own then goes to ring slot pos % TAIL_RING
            for tq in range(start("cache"), NREQ, G):
                s0 = fx.Int32(tq) * tpr
                kns, grs = [], []
                for jr in range_constexpr(tpr):
                    s = s0 + jr
                    stamp("cache")
                    pos = row_word(s, ROW_POS)
                    slot = row_word(s, ROW_SLOT)
                    pslot = row_word(s, ROW_POOL_SLOT)
                    tslot = row_word(s, ROW_TAIL_SLOT)
                    g = ld_bf16(r_gkv, tid)
                    d = fx.min(tid, INDEX_DIM - 1)
                    lk = tid < INDEX_DIM
                    lnw, lnb = ld_f32(r_lnw, d), ld_f32(r_lnb, d)
                    kv, kraw, graw = getf_many(
                        [
                            (mb("kva"), s * KV_LORA + tid),
                            (mb("ik"), s * INDEX_DIM + d),
                            (mb("ig"), s * INDEX_DIM + d),
                        ]
                    )
                    rstd = _rsq(block_sum(kv * kv) * (1.0 / KV_LORA) + EPS)
                    row_off = fx.Int64(slot // KVB) * KVE + fx.Int64(
                        (slot % KVB) * KV_LORA
                    )
                    if const_expr(kv_fp8):
                        # vLLM's write: the bf16 normed row / k_scale, to e4m3
                        # (round to nearest even, saturating at 448), four a thread
                        lds_st(xs, KVX + tid, bf16_round(kv * rstd * g))
                        gpu.barrier()
                        ks = ld_f32(_rsrc(tv["k_scale"]), 0)
                        e4 = [
                            fx.min(
                                fx.max(
                                    div(lds_ld(xs, KVX + tid * 4 + i), ks), -FP8_MAX
                                ),
                                FP8_MAX,
                            )
                            for i in range(4)
                        ]
                        if (slot >= 0) & (tid < KV_LORA // 4):
                            bo.buffer_store(
                                fp8_pack4(*e4), _rsrc(tv["kv_cache"] + row_off), tid
                            )
                    else:
                        if slot >= 0:
                            bo.buffer_store(
                                fx.Float32(kv * rstd * g).to(fx.BFloat16),
                                _rsrc(tv["kv_cache"] + row_off * fx.Int64(2)),
                                tid,
                            )
                    mean = block_sum(lk.select(kraw, fx.Float32(0.0))) * (
                        1.0 / INDEX_DIM
                    )
                    cen = lk.select(kraw - mean, fx.Float32(0.0))
                    var = block_sum(cen * cen) * (1.0 / INDEX_DIM)
                    kn = bf16_round(cen * _rsq(var + INDEX_EPS) * lnw + lnb)
                    kns.append(kn)
                    grs.append(graw)
                    ppos = fx.max(pos, 0)
                    phase = ppos % KPOOL
                    # vLLM's caches can be views of pools past 4 GB: a 64-bit base
                    r_tail = _rsrc(
                        tv["tail_cache"]
                        + fx.Int64(fx.max(tslot, 0) // TAIL_RING)
                        * fx.Int64(TAIL_BLK)
                        * fx.Int64(2)
                    )
                    complete = (pos >= 0) & (pslot >= 0) & (phase == KPOOL - 1)
                    if complete:
                        if lk:
                            sj, kj = [], []
                            for i in range_constexpr(KPOOL - 1):
                                back = KPOOL - 1 - i
                                if const_expr(jr - back >= 0):
                                    sj.append(grs[jr - back])
                                    kj.append(kns[jr - back])
                                else:
                                    ring = ((ppos - back) % TAIL_RING) * INDEX_DIM + d
                                    sj.append(ld_bf16(r_tail, ring + TAIL_GATE))
                                    kj.append(ld_bf16(r_tail, ring))
                            sj.append(graw)
                            kj.append(kn)
                            sj = [
                                sj[i] + ld_f32(r_ape, i * INDEX_DIM + d)
                                for i in range(KPOOL)
                            ]
                            mx = sj[0]
                            for i in range_constexpr(1, KPOOL):
                                mx = fx.max(mx, sj[i])
                            num = fx.Float32(0.0)
                            den = fx.Float32(0.0)
                            for i in range_constexpr(KPOOL):
                                p = _exp(sj[i] - mx)
                                den = den + p
                                num = num + kj[i] * p
                            lds_st(xs, d, bf16_round(div(num, den)))
                        gpu.barrier()
                        if wave == 0:
                            x0, x1 = fwht128(lds_ld(xs, lane), lds_ld(xs, lane + 64))
                            q0, q1, kscale = ue8m0_quant(bf16_round(x0), bf16_round(x1))
                            lds_st(xs, INDEX_DIM + lane, q0)
                            lds_st(xs, INDEX_DIM + lane + 64, q1)
                            if lane == 0:
                                lds_st(misc, 0, kscale)
                        gpu.barrier()
                        r_icache = _rsrc(
                            tv["idx_cache"] + fx.Int64(pslot // IPB) * fx.Int64(IBB)
                        )
                        poff = pslot % IPB
                        if tid < INDEX_DIM // 4:
                            d4 = tid * 4
                            word = fp8_pack4(
                                *[lds_ld(xs, INDEX_DIM + d4 + j) for j in range(4)]
                            )
                            # 16-pool x 16-byte tiles (a block holds 16 n pools)
                            byte = (
                                (poff // 16) * (16 * INDEX_DIM)
                                + (d4 // 16) * 256
                                + (poff % 16) * 16
                                + d4 % 16
                            )
                            bo.buffer_store(word, r_icache, byte // 4)
                        if tid == 0:
                            bo.buffer_store(
                                lds_ld(misc, 0),
                                r_icache,
                                IPB * INDEX_DIM // 4 + poff,
                            )
                    if (pos >= 0) & (tslot >= 0) & lk:
                        e = (ppos % TAIL_RING) * INDEX_DIM + d
                        bo.buffer_store(kn.to(fx.BFloat16), r_tail, e)
                        bo.buffer_store(graw.to(fx.BFloat16), r_tail, e + TAIL_GATE)
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Release, syncscope="agent"
                    )
                    gpu.barrier()
                    if tid == 0:
                        put(mb("kv_ready"), s, fx.Int32(1))

        # ======================= q_b (FP8 or bf16): two 16-row tiles a task
        def dsa_q_b(tv, t32):
            r_wqb, r_sqb = _rsrc(tv["w_qb"]), _rsrc(tv["s_qb"])
            QB_CPW = Q_LORA // (128 if QB8 else 64) // (WAVES // 2)
            for t in range(start("q_b"), QB_ROWS // QB_TILE, G):
                t = fx.Int32(t)
                stamp("q_b")
                row0 = t * QB_TILE + (wave // 4) * 16

                def u_qb(c):
                    return proj_unit(
                        QB8,
                        r_wqb,
                        r_sqb,
                        row0,
                        (wave % 4) * QB_CPW + c,
                        Q_LORA,
                        QB_ROWS,
                        n_sel() * (Q_LORA // 2),
                    )

                QB_AHEAD = min(QB_CPW, 4)
                pre = [u_qb(c) for c in range(QB_AHEAD)]
                stage_rows(mb("q_an"), Q_LORA)
                gpu.barrier()
                acc = run_units(u_qb, QB_CPW, QB_AHEAD, pre)
                reduce_rows(2, acc, emit_out(QB_TILE))
                gpu.barrier()
                if tid < S * QB_TILE // 4:
                    s = tid // (QB_TILE // 4)
                    r = (tid % (QB_TILE // 4)) * 4
                    put_bf(
                        mb("q_nope"),
                        s * QB_ROWS + t * QB_TILE + r,
                        [lds_ld(outs, s * QB_TILE + r + j) for j in range(4)],
                    )
                gpu.barrier()

        # ===== index q (bf16, one head a task): wq_b, Hadamard, ue8m0 FP8, and
        # the head weight w * q_scale * 128^-0.5 * 32^-0.5
        def dsa_index_q(tv, t32):
            r_wiq = _rsrc(tv["w_iq"])
            IQ_CPW = Q_LORA // 64
            IQW_SCALE = INDEX_DIM**-0.5 * INDEX_HEADS**-0.5
            for t in range(start("index_q"), INDEX_HEADS, G):
                head = fx.Int32(t)
                stamp("index_q")
                row0 = head * INDEX_DIM + wave * 16

                def u_iq(c):
                    return unit_rm(
                        r_wiq,
                        row0,
                        c,
                        Q_LORA,
                        INDEX_HEADS * INDEX_DIM,
                        n_sel() * (Q_LORA // 2) + c * 32,
                    )

                pre = [u_iq(c) for c in range(4)]
                stage_rows(mb("q_an"), Q_LORA)
                gpu.barrier()
                acc = run_units(u_iq, IQ_CPW, 4, pre)
                gpu.barrier()
                # C[r][n] sits in lane n + 16 (r // 4), element r % 4
                if lane % 16 < S:
                    for e in range_constexpr(4):
                        lds_st(
                            xs,
                            (lane % 16) * INDEX_DIM + wave * 16 + 4 * (lane // 16) + e,
                            bf16_round(acc[e]),
                        )
                gpu.barrier()
                if wave < S:
                    x0, x1 = fwht128(
                        lds_ld(xs, wave * INDEX_DIM + lane),
                        lds_ld(xs, wave * INDEX_DIM + lane + 64),
                    )
                    q0, q1, qscale = ue8m0_quant(bf16_round(x0), bf16_round(x1))
                    lds_st(xs, (S + wave) * INDEX_DIM + lane, q0)
                    lds_st(xs, (S + wave) * INDEX_DIM + lane + 64, q1)
                    w_h = getf_many([(mb("iw"), wave * INDEX_HEADS + head)])[0]
                    if lane == 0:
                        put(
                            mb("iqw"),
                            wave * INDEX_HEADS + head,
                            w_h * qscale * IQW_SCALE,
                        )
                gpu.barrier()
                if tid < S * INDEX_DIM // 4:
                    s = tid // (INDEX_DIM // 4)
                    j = tid % (INDEX_DIM // 4)
                    q8 = (S + s) * INDEX_DIM + 4 * j
                    put(
                        mb("iq8"),
                        (s * INDEX_HEADS + head) * (INDEX_DIM // 4) + j,
                        fp8_pack4(*[lds_ld(xs, q8 + i) for i in range(4)]),
                    )
                gpu.barrier()

        # ========== absorbed query q_lat = W_UK^T q_nope: 128 latent rows a task
        def dsa_uk(tv, t32):
            r_wuk = _rsrc(tv["w_uk"])
            UK_CPW = NOPE_DIM // 64
            UK_PER_HEAD = KV_LORA // UK_TILE
            for t in range(start("uk"), MH * UK_PER_HEAD, G):
                t = fx.Int32(t)
                stamp("uk")
                head = t // UK_PER_HEAD
                row0 = t * UK_TILE + wave * 16

                def u_uk(c):
                    return unit_rm(
                        r_wuk,
                        row0,
                        c,
                        NOPE_DIM,
                        MH * KV_LORA,
                        n_sel() * (NOPE_DIM // 2) + c * 32,
                    )

                pre = [u_uk(c) for c in range(UK_CPW)]
                if tid < S * NOPE_DIM // 4:
                    n = tid // (NOPE_DIM // 4)
                    k4 = (tid % (NOPE_DIM // 4)) * 4
                    v = poll(
                        [(mb("q_nope"), (n * QB_ROWS + head * NOPE_DIM + k4) // 2, 2)]
                    )[0]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            n * (NOPE_DIM // 2) + k4 // 2 + j,
                            v[j].bitcast(fx.Float32),
                        )
                gpu.barrier()
                acc = run_units(u_uk, UK_CPW, UK_CPW, pre)
                if lane % 16 < S:
                    put_bf(
                        mb("q_lat"),
                        ((lane % 16) * MH + head) * KV_LORA
                        + (t % UK_PER_HEAD) * UK_TILE
                        + wave * 16
                        + 4 * (lane // 16),
                        [acc[e] for e in range(4)],
                    )
                gpu.barrier()

        # ===== index scores of a row's complete pools: 256 pools a task
        # score(p) = kscale(p) * sum_h w'_h relu(q8_h . k8_p), FP8 x FP8 MFMAs. Rows
        # of at most 2048 tokens select every token and are not scored.
        MAXP = max(max_pools, 1)
        QW8 = INDEX_DIM // 4  # words of one head's FP8 query

        def dsa_iscore(tv, t32):
            IPB = t32["idx_block_pools"]
            IBB = t32["idx_block_bytes"]
            r_ibt = _rsrc(tv["idx_block_table"])
            one = fx.Int32(127)  # E8M0 1.0
            for s in range_constexpr(S):
                pos = row_word(s, ROW_POS)
                n_pools = (pos >= TOPK_TOKENS).select((pos + 1) // KPOOL, fx.Int32(0))
                n_tiles = (n_pools + POOLS_PER_TASK - 1) // POOLS_PER_TASK
                first = start_at(base["iscore"] + s * (G // S))
                if first < n_tiles:
                    stamp("iscore")
                    bt = row_word(s, ROW_IBT)
                    # the row's FP8 query -> LDS words h * 32 .., weights -> misc
                    NQW = INDEX_HEADS * QW8 // THREADS
                    qv = poll(
                        [
                            (mb("iq8"), s * INDEX_HEADS * QW8 + tid + i * THREADS, 1)
                            for i in range(NQW)
                        ]
                    )
                    for i in range_constexpr(NQW):
                        lds_st(xs, tid + i * THREADS, qv[i][0].bitcast(fx.Float32))
                    hw = getf_many(
                        [(mb("iqw"), s * INDEX_HEADS + fx.min(tid, INDEX_HEADS - 1))]
                    )[0]
                    if tid < INDEX_HEADS:
                        lds_st(misc, tid, hw)
                    # the pool this launch completed is in the cache
                    get(mb("kv_ready"), s)
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Acquire, syncscope="agent"
                    )
                    gpu.barrier()
                    g4 = lane // 16
                    for tile in range(first, n_tiles, G):
                        tile = fx.Int32(tile)
                        for half in range_constexpr(POOLS_PER_TASK // (16 * WAVES)):
                            p0 = tile * POOLS_PER_TASK + (half * WAVES + wave) * 16

                            # the wave's 16 pools (clamped to the row's) share one
                            # block of 16 n pools: a uniform 64-bit block base
                            blk = _uniform(
                                ld_i32(
                                    r_ibt,
                                    bt * t32["idx_bt_stride"]
                                    + fx.min(p0, n_pools - 1) // IPB,
                                )
                            )
                            r_blk = _rsrc(
                                tv["idx_cache"] + fx.Int64(blk) * fx.Int64(IBB)
                            )

                            def off_of(p):
                                return fx.min(p, n_pools - 1) % IPB

                            # A: pool lane % 16, bytes 32 g4 .. of its FP8 vector
                            po = off_of(p0 + lane % 16)
                            b0 = (
                                (po // 16) * (16 * INDEX_DIM)
                                + (2 * g4) * 256
                                + (po % 16) * 16
                            )
                            kw = [
                                fx.Vector(
                                    bo.buffer_load(
                                        r_blk, bb // 4, vec_width=4, dtype=T.i32
                                    )
                                )
                                for bb in (b0, b0 + 256)
                            ]
                            a = fx.Vector.from_elements(
                                [kw[h][e] for h in range(2) for e in range(4)],
                                fx.Int32,
                            )
                            parts = [fx.Float32(0.0) for _ in range(4)]
                            for hh in range_constexpr(INDEX_HEADS // 16):
                                hq = hh * 16 + lane % 16
                                qb = [
                                    fx.Vector(
                                        fx.ptr_load(
                                            xs + (hq * QW8 + 8 * g4 + 4 * h),
                                            result_type=v4f,
                                        )
                                    ).bitcast(fx.Int32)
                                    for h in range(2)
                                ]
                                b = fx.Vector.from_elements(
                                    [qb[h][e] for h in range(2) for e in range(4)],
                                    fx.Int32,
                                )
                                c = fx.Vector(
                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        T.vec(4, T.f32),
                                        [
                                            a,
                                            b,
                                            fx.Vector.filled(4, 0.0, fx.Float32),
                                            0,
                                            0,
                                            0,
                                            one,
                                            0,
                                            one,
                                        ],
                                    )
                                )
                                # C[pool][head] in lane head + 16 (pool // 4), elem pool % 4
                                w_h = lds_ld(misc, hq)
                                for e in range_constexpr(4):
                                    parts[e] = (
                                        parts[e] + fx.max(c[e], fx.Float32(0.0)) * w_h
                                    )
                            for off in (1, 2, 4, 8):
                                parts = [
                                    _xred(v, off, lambda x, y: x + y) for v in parts
                                ]
                            e_w = lane % 16
                            val = parts[0]
                            for e in range_constexpr(1, 4):
                                val = (e_w == e).select(parts[e], val)
                            pw = p0 + 4 * g4 + e_w
                            ksc = ld_f32(r_blk, IPB * INDEX_DIM // 4 + off_of(pw))
                            if (e_w < 4) & (pw < n_pools):
                                put(mb("iscores"), s * MAXP + pw, val * ksc)
                    gpu.barrier()

        # ====== top-512 pools of a row (one CTA a row), then the 2048 attended
        # tokens: the pools' tokens and the row's tail tokens past the last
        # complete pool; when there is a tail, the last positions of the last
        # selected pool (one of the lowest-scored) make room for it.
        def dsa_iselect(tv, t32):
            hist = fx.recast_iter(fx.Int32, xs)
            r_scores = _rsrc(mb("iscores"))
            r_sel = _rsrc(mb("tok_sel"))

            def radix_key(bits):
                return (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)

            def select_digit(shift, prefix, remain):
                digit = 255 - fx.min(tid, 255)
                cnt = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    cnt, fx.ReductionOp.ADD, width=64
                )
                wave_total = read_lane_i32(inclusive, 63)
                if (wave < 4) & (lane == 63):
                    lds_st(keys, 256 + wave, wave_total)
                gpu.barrier()
                before_wave = fx.Int32(0)
                for w in range_constexpr(4):
                    before_wave = before_wave + (wave > w).select(
                        lds_ld(keys, 256 + w), fx.Int32(0)
                    )
                above = before_wave + inclusive - cnt
                hit = (tid < 256) & (above < remain) & (above + cnt >= remain)
                gpu.barrier()
                if hit:
                    lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift)))
                    lds_st(keys, 257, remain - above)
                gpu.barrier()
                return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

            for t in range(start("iselect"), S, G):
                s = fx.Int32(t)
                stamp("iselect")
                pos = row_word(s, ROW_POS)
                if pos >= TOPK_TOKENS:
                    bound = (pos + 1) // KPOOL
                    count = fx.Int32(TOPK_POOLS)
                    n_chunks = (bound + SEL_CHUNK - 1) // SEL_CHUNK

                    def chunk_items(chunk):
                        return [
                            chunk * SEL_CHUNK + tid * SEL_ITEMS + j
                            for j in range_constexpr(SEL_ITEMS)
                        ]

                    def poll_keys(items):
                        vals = poll(
                            [
                                (mb("iscores"), s * MAXP + fx.min(i, bound - 1), 1)
                                for i in items
                            ]
                        )
                        return [fx.Uint32(radix_key(v[0])) for v in vals]

                    def load_keys(items):
                        """Keys of consecutive item pairs, two (value, tag) pairs a load."""
                        out = []
                        for j in range_constexpr(0, len(items), 2):
                            w = fx.Vector(
                                bo.buffer_load(
                                    r_scores,
                                    (s * MAXP + fx.min(items[j], MAXP - 2)) * 2,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                            out += [
                                fx.Uint32(radix_key(w[0])),
                                fx.Uint32(radix_key(w[2])),
                            ]
                        return out

                    def clear_hist():
                        for b in range_constexpr(SEL_BINS // THREADS):
                            lds_st(hist, tid + b * THREADS, fx.Int32(0))
                        gpu.barrier()

                    def select_wide(shift, prefix, remain):
                        """The 12-bit digit at ``shift`` holding the remain-th largest
                        key: (prefix with it, keys still needed, keys in that bin)."""
                        per = SEL_BINS // THREADS
                        cs = [
                            lds_ld(hist, SEL_BINS - 1 - (tid * per + j))
                            for j in range(per)
                        ]
                        local = cs[0]
                        for c in cs[1:]:
                            local = local + c
                        inclusive = fx.coop.warp_inclusive_scan(
                            local, fx.ReductionOp.ADD, width=64
                        )
                        wave_total = read_lane_i32(inclusive, 63)
                        if lane == 63:
                            lds_st(keys, 256 + wave, wave_total)
                        gpu.barrier()
                        running = inclusive - local
                        for w in range_constexpr(WAVES):
                            running = running + (wave > w).select(
                                lds_ld(keys, 256 + w), fx.Int32(0)
                            )
                        for j in range_constexpr(per):
                            digit = SEL_BINS - 1 - (tid * per + j)
                            hit = (running < remain) & (running + cs[j] >= remain)
                            if hit:
                                lds_st(
                                    keys,
                                    280,
                                    fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                                )
                                lds_st(keys, 281, remain - running)
                                lds_st(keys, 282, cs[j])
                            running = running + cs[j]
                        gpu.barrier()
                        return (
                            fx.Uint32(lds_ld(keys, 280)),
                            lds_ld(keys, 281),
                            lds_ld(keys, 282),
                        )

                    def scan_pair(first_flags, second_flags, parity):
                        """Exclusive block offsets of two flag lists in one scan
                        (running bases double-buffered by ``parity``)."""
                        local = fx.Int32(0)
                        local_offsets = []
                        for a, b in zip(first_flags, second_flags):
                            local_offsets.append(local)
                            local = (
                                local
                                + a.select(fx.Int32(1), fx.Int32(0))
                                + b.select(fx.Int32(1 << 16), fx.Int32(0))
                            )
                        inclusive = fx.coop.warp_inclusive_scan(
                            local, fx.ReductionOp.ADD, width=64
                        )
                        slot = 256 + parity * WAVES
                        if lane == 63:
                            lds_st(keys, slot + wave, read_lane_i32(inclusive, 63))
                        gpu.barrier()
                        before = fx.Int32(0)
                        total = fx.Int32(0)
                        for w in range_constexpr(WAVES):
                            wave_count = lds_ld(keys, slot + w)
                            before = before + (wave > w).select(wave_count, fx.Int32(0))
                            total = total + wave_count
                        thread_base = before + inclusive - local
                        base_a = lds_ld(keys, 272 + parity * 2)
                        base_b = lds_ld(keys, 273 + parity * 2)
                        offs = [thread_base + o for o in local_offsets]
                        n_a = total & fx.Int32(0xFFFF)
                        n_b = total >> 16
                        if tid == 0:
                            lds_st(keys, 272 + (1 - parity) * 2, base_a + n_a)
                            lds_st(keys, 273 + (1 - parity) * 2, base_b + n_b)
                        return (
                            [base_a + (o & fx.Int32(0xFFFF)) for o in offs],
                            [base_b + (o >> 16) for o in offs],
                        )

                    def set_bases(a, b):
                        if tid == 0:
                            lds_st(keys, 272, a)
                            lds_st(keys, 273, b)
                        gpu.barrier()

                    def gather(
                        chunk, valid, item_keys, values, threshold, out_gt, remain
                    ):
                        """Keys above ``threshold`` append to red from the running
                        base; the first ``remain`` ties fill red[out_gt:]."""
                        gt = [v & (k > threshold) for v, k in zip(valid, item_keys)]
                        eq = [v & (k == threshold) for v, k in zip(valid, item_keys)]
                        gt_offsets, eq_offsets = scan_pair(gt, eq, chunk & 1)
                        for j in range_constexpr(len(values)):
                            if gt[j]:
                                lds_st(
                                    red, gt_offsets[j], values[j].bitcast(fx.Float32)
                                )
                            if eq[j] & (eq_offsets[j] < remain):
                                lds_st(
                                    red,
                                    out_gt + eq_offsets[j],
                                    values[j].bitcast(fx.Float32),
                                )

                    def publish_all():
                        """red[0 .. 512) holds the pools; write the row's tokens."""
                        gpu.barrier()
                        tail_n = (pos + 1) - bound * KPOOL
                        keep = TOPK_TOKENS - tail_n
                        for b in range_constexpr(TOPK_TOKENS // THREADS):
                            j = tid + b * THREADS
                            pool = lds_ld(red, j // KPOOL).bitcast(fx.Int32)
                            tok = (j < keep).select(
                                pool * KPOOL + j % KPOOL, bound * KPOOL + (j - keep)
                            )
                            bo.buffer_store(
                                tok,
                                r_sel,
                                s * TOPK_TOKENS + j,
                                cache_modifier=CM_DEV,
                            )

                    def candidates(level, prefix, remain, n_cand):
                        """Keys above the ``level`` bin go to red, the bin's to LDS."""
                        set_bases(fx.Int32(0), fx.Int32(0))
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            chunk = fx.Int32(chunk)
                            items = chunk_items(chunk)
                            item_keys = load_keys(items)
                            high = [
                                (i < bound) & ((k >> level) > (prefix >> level))
                                for i, k in zip(items, item_keys)
                            ]
                            same = [
                                (i < bound) & ((k >> level) == (prefix >> level))
                                for i, k in zip(items, item_keys)
                            ]
                            win_offsets, cand_offsets = scan_pair(high, same, chunk & 1)
                            for j in range_constexpr(SEL_ITEMS):
                                if high[j]:
                                    lds_st(
                                        red,
                                        win_offsets[j],
                                        items[j].bitcast(fx.Float32),
                                    )
                                c = SEL_BINS + cand_offsets[j] * 2
                                if same[j]:
                                    lds_st(xs, c, item_keys[j].bitcast(fx.Float32))
                                    lds_st(xs, c + 1, items[j].bitcast(fx.Float32))
                        gpu.barrier()
                        n_win = count - remain

                        def cand_pass(shift, mask, prefix, remain):
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for c in range(tid, n_cand, fx.Int32(THREADS)):
                                key = fx.Uint32(
                                    lds_ld(xs, SEL_BINS + fx.Int32(c) * 2).bitcast(
                                        fx.Int32
                                    )
                                )
                                if (key & fx.Uint32(mask)) == prefix:
                                    digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                    fx.atomic_add(
                                        keys + digit,
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                            gpu.barrier()
                            return select_digit(shift, prefix, remain)

                        if const_expr(level == 20):
                            prefix, remain = cand_pass(12, 0xFFF00000, prefix, remain)
                            prefix, remain = cand_pass(4, 0xFFFFF000, prefix, remain)
                            prefix, remain = cand_pass(0, 0xFFFFFFF0, prefix, remain)
                        else:
                            prefix, remain = cand_pass(0, 0xFFFFFF00, prefix, remain)
                        set_bases(n_win, fx.Int32(0))
                        out_gt = count - remain
                        n_cand_chunks = (n_cand + SEL_CHUNK - 1) // SEL_CHUNK
                        for chunk in range(fx.Int32(0), n_cand_chunks, fx.Int32(1)):
                            chunk = fx.Int32(chunk)
                            idx = chunk_items(chunk)
                            safe = [fx.min(c, n_cand - 1) for c in idx]
                            gather(
                                chunk,
                                [c < n_cand for c in idx],
                                [
                                    fx.Uint32(
                                        lds_ld(xs, SEL_BINS + c * 2).bitcast(fx.Int32)
                                    )
                                    for c in safe
                                ],
                                [
                                    lds_ld(xs, SEL_BINS + c * 2 + 1).bitcast(fx.Int32)
                                    for c in safe
                                ],
                                prefix,
                                out_gt,
                                remain,
                            )
                        publish_all()

                    def full_radix():
                        """Byte-wise radix over every key, when the threshold bin
                        overflows LDS."""

                        def radix_pass(shift, prefix_mask, prefix, remain):
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                                items = chunk_items(fx.Int32(chunk))
                                item_keys = load_keys(items)
                                for j in range_constexpr(SEL_ITEMS):
                                    key = item_keys[j]
                                    if (items[j] < bound) & (
                                        (key & fx.Uint32(prefix_mask)) == prefix
                                    ):
                                        digit = fx.Int32(
                                            (key >> shift) & fx.Uint32(255)
                                        )
                                        fx.atomic_add(
                                            keys + digit,
                                            fx.Int32(1),
                                            syncscope="workgroup",
                                        )
                            gpu.barrier()
                            return select_digit(shift, prefix, remain)

                        prefix, remain = radix_pass(24, 0, fx.Uint32(0), count)
                        prefix, remain = radix_pass(16, 0xFF000000, prefix, remain)
                        prefix, remain = radix_pass(8, 0xFFFF0000, prefix, remain)
                        prefix, remain = radix_pass(0, 0xFFFFFF00, prefix, remain)
                        set_bases(fx.Int32(0), fx.Int32(0))
                        out_gt = count - remain
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            chunk = fx.Int32(chunk)
                            items = chunk_items(chunk)
                            gather(
                                chunk,
                                [i < bound for i in items],
                                load_keys(items),
                                items,
                                prefix,
                                out_gt,
                                remain,
                            )
                        publish_all()

                    clear_hist()
                    for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        items = chunk_items(fx.Int32(chunk))
                        item_keys = poll_keys(items)
                        for j in range_constexpr(SEL_ITEMS):
                            if items[j] < bound:
                                fx.atomic_add(
                                    hist + fx.Int32(item_keys[j] >> 20),
                                    fx.Int32(1),
                                    syncscope="workgroup",
                                )
                    gpu.barrier()
                    p12, r12, c12 = select_wide(20, fx.Uint32(0), count)
                    if c12 <= SEL_CAP:
                        candidates(20, p12, r12, c12)
                    if c12 > SEL_CAP:
                        clear_hist()
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            items = chunk_items(fx.Int32(chunk))
                            item_keys = load_keys(items)
                            for j in range_constexpr(SEL_ITEMS):
                                k = item_keys[j]
                                if (items[j] < bound) & ((k >> 20) == (p12 >> 20)):
                                    fx.atomic_add(
                                        hist + fx.Int32((k >> 8) & fx.Uint32(0xFFF)),
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                        gpu.barrier()
                        p24, r24, c24 = select_wide(8, p12, r12)
                        if c24 <= SEL_CAP:
                            candidates(8, p24, r24, c24)
                        if c24 > SEL_CAP:
                            full_radix()
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Release, syncscope="agent"
                    )
                    gpu.barrier()
                    if tid == 0:
                        put(mb("tok_ready"), s, fx.Int32(1))

        # ================= sparse MLA split: 64 selected keys x 8 heads a task
        # Rows of at most 2048 tokens attend to every token, longer rows the 2048
        # the index selected. Splits past the row's keys do nothing (split 0
        # always publishes, so empty rows merge to zero).
        def dsa_split(tv, t32):
            KVB = t32["kv_block_tokens"]
            KVE = tv["kv_block_elems"]
            r_bt = _rsrc(tv["block_table"])
            KPW = SPLIT_KEYS // WAVES
            ktile = xs + KT_OFF
            kscale = ld_f32(_rsrc(tv["k_scale"]), 0) if kv_fp8 else None
            pl = xs  # q is dead once the scores are in ``red``
            for tt in range(start("split"), S * HG * N_SPLIT, G):
                tt = fx.Int32(tt)
                stamp("split")
                s = tt // (HG * N_SPLIT)
                hg = (tt // N_SPLIT) % HG
                t = tt % N_SPLIT
                pos = row_word(s, ROW_POS)
                nk = fx.min(pos + 1, fx.Int32(TOPK_TOKENS))
                n_used = fx.max((nk + SPLIT_KEYS - 1) // SPLIT_KEYS, fx.Int32(1))
                if t < n_used:
                    bt = row_word(s, ROW_BT)
                    long_row = pos >= TOPK_TOKENS
                    if long_row:
                        if wave == 0:
                            if lane == 0:
                                get(mb("tok_ready"), s)
                            # only wave 0 reads the selection; its divergent
                            # ready poll reconverges before this acquire
                            fx.memory_fence(
                                ordering=fx.AtomicOrdering.Acquire, syncscope="agent"
                            )
                    if wave == 0:
                        kidx = t * SPLIT_KEYS + lane
                        tok = (kidx < nk).select(kidx, fx.Int32(0))
                        picked = fx.Int32(
                            bo.buffer_load(
                                _rsrc(mb("tok_sel")),
                                s * TOPK_TOKENS + fx.min(kidx, TOPK_TOKENS - 1),
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        tok = long_row.select(picked, tok)
                        tok_blk = ld_i32(r_bt, bt * t32["bt_stride"] + tok // KVB)
                        lds_st(keys, lane, tok_blk)
                        lds_st(keys, SPLIT_KEYS + lane, tok % KVB)
                    # the row's own token is written by this launch's cache task
                    get(mb("kv_ready"), s)
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Acquire, syncscope="agent"
                    )
                    gpu.barrier()
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        blk = _uniform(lds_ld(keys, j))
                        off = _uniform(lds_ld(keys, SPLIT_KEYS + j))
                        row_off = fx.Int64(blk) * KVE + fx.Int64(off * KV_LORA)
                        if const_expr(kv_fp8):
                            # 8 e4m3 a lane, dequantized into the bf16 tile
                            raw8 = fx.Vector(
                                bo.buffer_load(
                                    _rsrc(tv["kv_cache"] + row_off),
                                    lane * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                )
                            )
                            row8 = _fp8_to_bf16x8(raw8[0], raw8[1], kscale)
                            fx.ptr_store(
                                row8.bitcast(fx.Float32), ktile + (j * QS + lane * 4)
                            )
                        else:
                            raw = fx.Vector(
                                bo.buffer_load(
                                    _rsrc(tv["kv_cache"] + row_off * fx.Int64(2)),
                                    lane * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(
                                raw.bitcast(fx.Float32), ktile + (j * QS + lane * 4)
                            )
                    # q_lat of the task's 8 heads -> Q[h] (words h * QS ..)
                    NQ = WAVES * KV_LORA // 4 // THREADS
                    qv = poll(
                        [
                            (
                                mb("q_lat"),
                                (
                                    (s * MH + hg * WAVES) * KV_LORA
                                    + (tid + i * THREADS) * 4
                                )
                                // 2,
                                2,
                            )
                            for i in range(NQ)
                        ]
                    )
                    for i in range_constexpr(NQ):
                        q4 = tid + i * THREADS
                        qw = (q4 // (KV_LORA // 4)) * QS + (q4 % (KV_LORA // 4)) * 2
                        lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
                    gpu.barrier()
                    # scores = K Q^T: wave (16-key row group wave % 4, K half wave / 4)
                    hn = fx.min(lane % 16, WAVES - 1)
                    rgk = wave % 4
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for st in range_constexpr(KV_LORA // 32 // 2):
                        kst = (wave // 4) * (KV_LORA // 32 // 2) + st
                        key = rgk * 16 + lane % 16
                        a = fx.ptr_load(
                            ktile + (key * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        b = fx.ptr_load(
                            xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    # split-local softmax: wave = head, lane = key
                    valid = (t * SPLIT_KEYS + lane) < nk
                    r16 = lane % 16
                    cl = wave + 16 * (r16 // 4)
                    krg = lane // 16
                    raw_s = lds_ld(red, (krg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                        red, ((krg + 4) * 64 + cl) * 4 + r16 % 4
                    )
                    sc_v = valid.select(raw_s * MLA_SCALE, fx.Float32(NEG))
                    m = wave_max(sc_v)
                    p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                    lsum = wave_sum(p)
                    p_n = _xshfl(p, 1)
                    if lane % 2 == 0:
                        lds_st(
                            pl, wave * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n)
                        )
                    gpu.barrier()
                    # O = P V: heads M, keys K, latent dim pairs N (even / odd MFMAs)
                    for gq in range_constexpr(KV_LORA // 32 // WAVES):
                        dw = (wave * (KV_LORA // 32 // WAVES) + gq) * 16 + lane % 16
                        c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                        c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                        for js in range_constexpr(SPLIT_KEYS // 32):
                            a = fx.ptr_load(
                                pl
                                + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                                result_type=v4f,
                            ).bitcast(fx.BFloat16)
                            ws = [
                                fx.ptr_load(
                                    ktile + ((js * 32 + (lane // 16) * 8 + i) * QS + dw)
                                ).bitcast(fx.Int32)
                                for i in range(8)
                            ]
                            w_lo = [
                                (ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16)
                                for i in range(4)
                            ]
                            w_hi = [
                                fx.Int32(fx.Uint32(ws[2 * i]) >> 16)
                                | (ws[2 * i + 1] & -65536)
                                for i in range(4)
                            ]
                            b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(
                                fx.BFloat16
                            )
                            b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(
                                fx.BFloat16
                            )
                            c0 = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32), [a, b0, c0]
                                )
                            )
                            c1 = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32), [a, b1, c1]
                                )
                            )
                        if lane < 32:
                            for e in range_constexpr(4):
                                hh = hg * WAVES + (lane // 16) * 4 + e
                                put_bf(
                                    mb("sp_acc"),
                                    ((s * N_SPLIT + t) * MH + hh) * KV_LORA + dw * 2,
                                    [c0[e], c1[e]],
                                )
                    if lane == 0:
                        ml = (s * N_SPLIT + t) * MH + hg * WAVES + wave
                        put(mb("sp_m"), ml, m)
                        put(mb("sp_l"), ml, lsum)
                    gpu.barrier()

        # ===== split merge + W_UV: 64 v rows of one (row, head) a task
        def dsa_uv(tv, t32):
            r_wkvb = _rsrc(tv["w_kvb"])
            UV_R = UV_TILE // 16
            UV_WPR = WAVES // UV_R
            UV_CPW = KV_LORA // 64 // UV_WPR
            SPH = N_SPLIT // 2
            for tt in range(start("uv"), S * MH * (V_DIM // UV_TILE), G):
                tt = fx.Int32(tt)
                stamp("uv")
                s = tt // (MH * (V_DIM // UV_TILE))
                head = (tt // (V_DIM // UV_TILE)) % MH
                tile = tt % (V_DIM // UV_TILE)
                pos = row_word(s, ROW_POS)
                nk = fx.min(pos + 1, fx.Int32(TOPK_TOKENS))
                n_used = fx.max((nk + SPLIT_KEYS - 1) // SPLIT_KEYS, fx.Int32(1))
                row0 = head * (NOPE_DIM + V_DIM) + NOPE_DIM + tile * UV_TILE
                row0 = row0 + (wave // UV_WPR) * 16

                def u_uv(c):
                    kc = (wave % UV_WPR) * UV_CPW + c
                    return unit_rm(
                        r_wkvb, row0, kc, KV_LORA, MH * (NOPE_DIM + V_DIM), kc * 32
                    )

                pre = [u_uv(c) for c in range(UV_CPW)]
                if wave == 0:  # split weights exp(m - M) / L -> misc[split]
                    spi = fx.min(lane, n_used - 1)
                    sp_ml = (s * N_SPLIT + spi) * MH + head
                    mlv = poll([(mb("sp_m"), sp_ml, 1), (mb("sp_l"), sp_ml, 1)])
                    ok_sp = lane < n_used
                    m_sp = ok_sp.select(mlv[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    l_sp = ok_sp.select(mlv[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                    m_all = wave_max(m_sp)
                    w_sp = _exp(m_sp - m_all)
                    den = wave_sum(l_sp * w_sp)
                    if lane < N_SPLIT:
                        lds_st(
                            misc,
                            lane,
                            ok_sp.select(
                                w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                                fx.Float32(0.0),
                            ),
                        )
                gpu.barrier()
                dp = tid % (KV_LORA // 2)
                hf = tid // (KV_LORA // 2)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            (
                                (s * N_SPLIT + fx.min(hf * SPH + q, n_used - 1)) * MH
                                + head
                            )
                            * (KV_LORA // 2)
                            + dp,
                            1,
                        )
                        for q in range(SPH)
                    ]
                )
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + q)
                    a0, a1 = bf2_f32(got[q][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
                gpu.barrier()
                if tid < KV_LORA // 2:
                    o0 = lds_ld(red, tid * 2) + lds_ld(red, (KV_LORA // 2 + tid) * 2)
                    o1 = lds_ld(red, tid * 2 + 1) + lds_ld(
                        red, (KV_LORA // 2 + tid) * 2 + 1
                    )
                    lds_st(xs, tid, bf16_pair(o0, o1))
                gpu.barrier()
                acc = run_units(u_uv, UV_CPW, UV_CPW, pre)
                reduce_rows(UV_R, acc, emit_out(UV_TILE), 1)
                gpu.barrier()
                if tid < UV_TILE // 4:
                    r = tid * 4
                    put_bf(
                        mb("o_in"),
                        (s * MH + head) * V_DIM + tile * UV_TILE + r,
                        [lds_ld(outs, r + j) for j in range(4)],
                    )
                gpu.barrier()

        # ======================= o_proj (FP8 or bf16) partial -> every peer
        def dsa_oproj(tv, t32):
            r_wo, r_so = _rsrc(tv["w_o"]), _rsrc(tv["s_o"])
            OK = MH * V_DIM
            O8_CPW = OK // (128 if O8 else 64) // WAVES
            run_oproj(
                lambda row0, kc: proj_unit(
                    O8, r_wo, r_so, row0, kc, OK, HIDDEN, n_sel() * (OK // 2)
                ),
                OK,
                O8_CPW,
                min(4, O8_CPW),
            )

        def run_dsa():
            # the layer's argument table: pointers, then int32 geometry
            r_tab = _rsrc(dsa_tab)
            words = [
                fx.Vector(bo.buffer_load(r_tab, i * 2, vec_width=2, dtype=T.i32))
                for i in range(len(DSA_TAB))
            ]
            t32 = {n: _uniform(w[0]) for n, w in zip(DSA_TAB, words)}
            tv = {
                n: (fx.Int64(_uniform(w[1])) << 32) | fx.Int64(fx.Uint32(t32[n]))
                for n, w in zip(DSA_TAB, words)
            }

            run_seam("aseam", "attn")
            run_gate("agate", "attn")
            dsa_qkv_fp8(tv, t32)
            dsa_qkv_bf16(tv, t32)
            dsa_qnorm(tv, t32)
            dsa_cache(tv, t32)
            dsa_q_b(tv, t32)
            dsa_index_q(tv, t32)
            dsa_uk(tv, t32)
            dsa_iscore(tv, t32)
            dsa_iselect(tv, t32)
            dsa_split(tv, t32)
            dsa_uv(tv, t32)
            dsa_oproj(tv, t32)

        if const_expr(kda):
            r_wo_kda = _rsrc(w_o)
            O_CPW = P // 64 // WAVES
            run_seam("aseam", "attn")
            run_gate("agate", "attn")
            run_inproj()
            run_fgb()
            run_kda()
            run_oproj(
                lambda row0, kc: unit_rm(
                    r_wo_kda, row0, kc, P, HIDDEN, n_sel() * (P // 2) + kc * 32
                ),
                P,
                O_CPW,
                min(4, O_CPW),
            )
            run_seam("seam", "ffn_own")
        elif const_expr(dsa):
            run_dsa()
            run_seam("seam", "ffn_own")
        else:
            run_seam("seam", "ffn")
        run_gate("gate", "ffn")
        # ====== 2. router scores + the row's MXFP8 activation blocks (one sample a CTA)
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        R_CPW = R_NKC // WAVES // 2
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            router_sample = tt // N_ROUTER
            stamp("router")
            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            r_fold = (lane % 16) // ROUTER_TILE

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, kc * 32, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            # the row, bf16, into LDS words 0 .. HIDDEN / 2
            nq = HIDDEN // 4
            vals = poll(
                [
                    (
                        mb("a"),
                        (router_sample * HIDDEN + (tid + i * THREADS) * 4) // 2,
                        2,
                    )
                    for i in range(nq // THREADS)
                ]
            )
            for i in range_constexpr(nq // THREADS):
                for j in range_constexpr(2):
                    lds_st(
                        xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32)
                    )
            # this task's FP8 activation blocks: wave w quantizes block w * N_ROUTER + t
            x_blk = wave * N_ROUTER + t
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            gpu.barrier()
            if x_ok:
                a0, a1 = bf2_f32(lds_ld(xs, xk // 2).bitcast(fx.Int32))
                q0, q1, qs = quant_mxfp8(a0, a1)
                w8 = (
                    fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                    & 0xFFFF
                )
                w8n = _xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (router_sample * HIDDEN + xk) // 4, w8 | (w8n << 16))
                if lane % 16 == 0:
                    put(
                        mb("xqs"),
                        router_sample * XQ_GROUPS + x_blk * 4 + lane // 16,
                        qs,
                    )
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(
                            red,
                            (w * 64 + f * ROUTER_TILE + 16 * (m // 4)) * 4 + m % 4,
                        )
                put(
                    mb("scores"),
                    router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                    _rcp(1.0 + _exp(-logit)),
                )
            gpu.barrier()

        def dn_route(bs):
            """Expert routing: one whole wave per sample, in wave-sized batches.
            Slot 0 is the shared expert (weight 1), slot k + 1 pick k."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if route_sample < S:
                    e, w = route_top8(route_sample, bs)
                    if lane < MOE_SLOTS:
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ===================== 3. expert up/gate + clamped SiLU
        UG_W_BYTES = 2 * expert_inter * HIDDEN // 2
        UG_S_BYTES = 2 * expert_inter * (HIDDEN // 32)
        UG8 = 8
        UG8_UNITS = (HIDDEN // 128) // WAVES
        XW = HIDDEN // 4
        stamp("ug")
        dn_route(load_bias())
        gpu.barrier()
        stage_xq(list(range(S)))
        gpu.barrier()
        u0 = start("ug")
        for task_round in range_constexpr(ug_task_rounds(expert_inter)):
            u = fx.Int32(u0 + task_round * G)
            if u < expert_inter:
                c = u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(
                    fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8)
                )
                w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)

                def ug8_units(e, sample, shared_live=None):
                    if const_expr(shared_live is None):
                        rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                        rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            w_sh_ug,
                            num_records_bytes=shared_live.select(
                                fx.Int32(UG_W_BYTES), fx.Int32(0)
                            ),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            s_sh_ug,
                            num_records_bytes=shared_live.select(
                                fx.Int32(UG_S_BYTES), fx.Int32(0)
                            ),
                        )
                    sn = n_sel() if sample is None else fx.Int32(sample)
                    units = []
                    for cc in range_constexpr(UG8_UNITS):
                        unit = wave * UG8_UNITS + cc
                        units.append(
                            unit_native_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 32,
                                8 + sn * XQ_GROUPS + unit * 4 + lane // 16,
                                ln=w_ln,
                            )
                        )
                    return units

                def ug8_emit(sample, shared):
                    if tid < (S if shared else 1) * UG8 // 2:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0 = fx.min(lds_ld(outs, n * 16 + r), SWIGLU_LIMIT)
                        g1 = fx.min(lds_ld(outs, n * 16 + r + 1), SWIGLU_LIMIT)
                        v0 = fx.min(
                            fx.max(lds_ld(outs, n * 16 + UG8 + r), -SWIGLU_LIMIT),
                            SWIGLU_LIMIT,
                        )
                        v1 = fx.min(
                            fx.max(lds_ld(outs, n * 16 + UG8 + r + 1), -SWIGLU_LIMIT),
                            SWIGLU_LIMIT,
                        )
                        sn = n if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                    if (c == 0) & (tid < S if shared else tid == 0):
                        sn = tid if shared else fx.Int32(sample)
                        sl = fx.Int32(0) if shared else slot
                        put(
                            mb("sel"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(keys, sn * MOE_SLOTS + sl),
                        )
                        put(
                            mb("prob"),
                            sn * MOE_SLOTS + sl,
                            lds_ld(dnw, sn * MOE_SLOTS + sl),
                        )

                shared_pre = ug8_units(None, None, has_sh)
                cur = ug8_units(_uniform(lds_ld(keys, slot)), 0)
                if has_sh:
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(0, True)
                for sample in range_constexpr(S):
                    pre = cur
                    if const_expr(sample + 1 < S):
                        cur = ug8_units(
                            _uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)),
                            sample + 1,
                        )
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(sample, False)
                if const_expr(task_round + 1 < ug_task_rounds(expert_inter)):
                    gpu.barrier()

        # ======== 4. mid MXFP8 quant + expert down + route weighting + MoE TP reduce
        DN_R = (DN_TILE + 15) // 16
        DN_WPR = WAVES // DN_R
        DN_BATCH = down_prefetch_batch(S)
        DN_W_BYTES = HIDDEN * expert_inter // 2
        DN_S_BYTES = HIDDEN * (expert_inter // 32)
        DN_NKC = expert_inter // 64
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down")
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # rows outside the tile load their lane ^ 8 twin and are dropped
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            DN_NU = S * MOE_SLOTS * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            DN_BLK = S * MOE_SLOTS * expert_inter // 128

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                is_sh = e == SHARED
                wb = bo.create_buffer_resource_from_addr(
                    is_sh.select(w_sh_dn, w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES)),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    is_sh.select(s_sh_dn, s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES)),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )

                def route_coef():
                    return (
                        lane % 16 == s_q,
                        _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                    )

                return unit_native_mxfp4(
                    wb,
                    sb,
                    dn_rg + gu,
                    unit % (expert_inter // 128),
                    expert_inter,
                    unit * 32,
                    unit * 4 + lane // 16,
                    route_coef,
                    dn_ln,
                )

            if const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    q0, q1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                    st_mxfp8(blk * 128 + lane * 2, q0, q1)
                    if lane % 16 == 0:
                        lds_st(misc, blk * 4 + lane // 16, qs.bitcast(fx.Float32))
            gpu.barrier()
            if const_expr(S > 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            peer_reduce("ffn", t, store_x, DN_TILE)
            gpu.barrier()

    @flyc.jit
    def launch(
        attn_part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        g_norm: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        x_out: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        s_sh_ug: Int64,
        w_sh_dn: Int64,
        s_sh_dn: Int64,
        hc_a_fn: Int64,
        hc_a_scale: Int64,
        hc_a_base: Int64,
        g_in: Int64,
        w_in: Int64,
        w_fb: Int64,
        w_gb: Int64,
        conv_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        w_on: Int64,
        w_o: Int64,
        conv_state: Int64,
        rec_state: Int64,
        state_idx: Int64,
        conv_stride: Int64,
        rec_stride: Int64,
        dsa_tab: Int64,
        dsa_rows: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        flash_ffn_kernel(
            attn_part,
            res_in,
            post_in,
            comb_in,
            hc_fn,
            hc_scale,
            hc_base,
            g_norm,
            res_out,
            post_out,
            comb_out,
            x_out,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sh_ug,
            s_sh_ug,
            w_sh_dn,
            s_sh_dn,
            hc_a_fn,
            hc_a_scale,
            hc_a_base,
            g_in,
            w_in,
            w_fb,
            w_gb,
            conv_w,
            a_log,
            dt_bias,
            w_on,
            w_o,
            conv_state,
            rec_state,
            state_idx,
            conv_stride,
            rec_stride,
            dsa_tab,
            dsa_rows,
            scratch,
            sym,
            peers,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch, SC, SY
