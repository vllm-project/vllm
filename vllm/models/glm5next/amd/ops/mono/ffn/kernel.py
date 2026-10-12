# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/kernel.py
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

from vllm.models.common.mono import abi
from vllm.models.glm5next.amd.ops.mono.config import (
    EPS,
    FP8_MAX,
    HC_EPS,
    HC_MULT,
    HC_POST_MULT,
    HIDDEN,
    KDA_DIM,
    KDA_LOWER_BOUND,
    L2_EPS,
    MOE_SLOTS,
    N_EXPERTS,
    O_NORM_EPS,
    ROUTE_SCALE,
    ROUTER_CANDIDATES,
    SHARED_EXPERT,
    SINKHORN_ITERS,
    SWIGLU_LIMIT,
    TOP_K,
    FlashShard,
)
from vllm.models.glm5next.amd.ops.mono.ffn.layout import (
    FGB_TILE,
    IN_TILE,
    N_MIX,
    N_ROUTER,
    N_ROW_TILES,
    O_TILE,
    PMIX,
    POLL_STAGES,
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
from vllm.models.glm5next.amd.ops.mono.layout import (
    BLOCKS,
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    POLL_MAX,
    ROUTER_TILE,
    ROW_TILE,
    THREADS,
    UG_TILE,
    WAVES,
    atom_mxfp4_scale_index,
)
from vllm.models.glm5next.amd.ops.mono.ops import (
    bpermute_i32,
    div_rn,
    mem_realtime,
    spin_pause,
    wave_umax_dpp,
    write_lane_i32,
)
from vllm.models.glm5next.amd.ops.mono.ops import exp as _exp
from vllm.models.glm5next.amd.ops.mono.ops import rcp as _rcp
from vllm.models.glm5next.amd.ops.mono.ops import rsq as _rsq
from vllm.models.glm5next.amd.ops.mono.ops import rsrc as _rsrc
from vllm.models.glm5next.amd.ops.mono.ops import uniform as _uniform
from vllm.models.glm5next.amd.ops.mono.ops import uniform_f32 as _uniform_f32
from vllm.models.glm5next.amd.ops.mono.ops import xred as _xred
from vllm.models.glm5next.amd.ops.mono.ops import xshfl as _xshfl

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
    poll_limit: int | None = None,
    poll_early_out: bool = False,
    kda: bool = False,
):
    """Return the ``@flyc.jit`` launcher of one rank's FFN layer for S rows; with
    ``kda`` the whole KDA decoder layer (attention-side mHC, projections, short conv,
    gated delta rule, gated RMSNorm, o_proj and its TP reduce) runs first.

    ``poll_limit`` bounds the re-polls of every mailbox wait. An expired wait
    stores 1 into its stage's error word (``poll_err_addr``) and, with TP peers,
    into this rank's flag on rank 0 (``poll_xrank_addr``), then continues, so a
    stalled peer finishes the launch with wrong values instead of hanging. With
    ``poll_early_out`` every later wait of the same step returns at its first retry.
    The shared expert is read from ``w_sh_*`` (one expert each), not the routed
    tensors, so it may be fused into them or stored apart.
    """
    assert 2 <= S <= 16 and 1 <= launches_per_step <= LAYER_SLOTS
    assert poll_limit is None or poll_limit > 0
    assert inter % 128 == 0
    assert not kda or S <= 8
    W = npes
    SH = FlashShard(W)
    P, H, NIN = SH.kda_proj, SH.kda_heads, SH.kda_in_rows
    FAULT = abi.fault_spec()
    expert_inter = inter
    SHARED = SHARED_EXPERT
    G = BLOCKS
    SC, SY = layout(S, W, expert_inter, kda)
    DN_TILE = dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = S * MOE_SLOTS
    # LDS X users: one router row (bf16), every row's MXFP8 FFN input (ug), the
    # seam tile (x_attn + R', and R after KDA), every row's MXFP8 mid (down); with
    # KDA every row's bf16 attention input (inproj) and o (oproj)
    X_WORDS = max(
        HIDDEN // 2,
        S * HIDDEN // 4,
        S * ROW_TILE * (1 + 2 * HC_MULT) if kda else S * ROW_TILE * (1 + HC_MULT),
        down_x_words(S, expert_inter),
        S * HIDDEN // 2 if kda else 0,
        S * P // 2 if kda else 0,
        S * KDA_DIM if kda else 0,
    )
    MISC_OFF = X_WORDS
    MISC_WORDS = max(8 + S * XQ_GROUPS, S * MOE_SLOTS * (expert_inter // 32))
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE, S * DN_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    order = stage_tasks(S, expert_inter, W, kda)
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
            if (rank == FAULT.rank) & (step_value == FAULT.step) & (layer == 0):
                t_fault = mem_realtime()
                while mem_realtime() - t_fault < fx.Int64(FAULT.us * 100):
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
            if const_expr(poll_limit is None):
                while pending(v):
                    spin_pause()
                    v = load_all()
            else:
                retries = fx.Int32(0)
                if const_expr(poll_early_out):
                    abort_mark = step_value + 1
                    r_abort = _rsrc(mb("poll_abort"))
                    ab = fx.Int32(0)
                    while pending(v) & (retries < poll_limit) & (ab != abort_mark):
                        spin_pause()
                        v = load_all()
                        retries = retries + 1
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
                    while pending(v) & (retries < poll_limit):
                        spin_pause()
                        v = load_all()
                        retries = retries + 1
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
            layer's output, attention-side mHC) or "ffn_kda" (this launch's o_proj
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
                if const_expr(mode == "ffn_kda"):
                    pj = lds_ld(misc, s * MIX_A + j)
                else:
                    pj = ld_f32(r_post, s * HC_MULT + j)
                acc0 = fx.Float32(0.0)
                acc1 = fx.Float32(0.0)
                for i in range_constexpr(HC_MULT):
                    if const_expr(mode == "ffn_kda"):
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
                if const_expr(mode == "ffn_kda"):
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
                gam = [ld_bf16(r_g, tid * CPT + q) for q in range(CPT)]
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
                            li[2 * q] * rstd * gam[2 * q],
                            li[2 * q + 1] * rstd * gam[2 * q + 1],
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
            assert nq % THREADS == 0
            got = poll(
                [(box, (tid + i * THREADS) * 2, 2) for i in range(nq // THREADS)]
            )
            for i in range_constexpr(nq // THREADS):
                for j in range_constexpr(2):
                    lds_st(
                        xs, (tid + i * THREADS) * 2 + j, got[i][j].bitcast(fx.Float32)
                    )

        # ===================== 3. merged q | k | v | beta | f_a | g_a projection
        # One task a 16-row tile of this rank's row-major bf16 weight, every row;
        # outputs rounded to bf16 as vLLM's GEMM stores them.
        IN_CPW = HIDDEN // 64 // WAVES

        def run_inproj():
            r_w = _rsrc(w_in)
            for t in range(start("inproj"), (NIN + IN_TILE - 1) // IN_TILE, G):
                t = fx.Int32(t)
                stamp("inproj")
                row0 = t * IN_TILE

                def u_in(c):
                    kc = wave * IN_CPW + c
                    return unit_rm(r_w, row0, kc, HIDDEN, NIN, n_sel() * XW + kc * 32)

                pre = [u_in(c) for c in range(4)]
                stage_rows(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(u_in, IN_CPW, 4, pre)

                def emit(rl, n, v):
                    row = row0 + rl
                    if row < NIN:
                        put(mb("proj"), n * NIN + row, bf16_round(v))

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
                    v0, v1 = get2_many([(mb("proj"), n * NIN + src + 2 * w)])[0]
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
        # One task a (row, head). A row whose state index is not positive (padding)
        # writes no state and outputs zeros.
        QO, KO, VO, G1O, G2O, OO = 0, KD, 2 * KD, 3 * KD, 4 * KD, 5 * KD

        def run_kda():
            r_cw = _rsrc(conv_w)
            r_alog = _rsrc(a_log)
            r_dtb = _rsrc(dt_bias)
            r_won = _rsrc(w_on)
            r_idx = _rsrc(state_idx)
            for t in range(start("kda"), S * H, G):
                t = fx.Int32(t)
                stamp("kda")
                s = t // H
                h = t % H
                idx = _uniform(
                    fx.Int32(bo.buffer_load(r_idx, s, vec_width=1, dtype=T.i32))
                )
                valid = idx > 0
                slot = fx.Int64(fx.max(idx, fx.Int32(0)))
                r_cs = _rsrc(conv_state + slot * conv_stride * fx.Int64(2))
                if tid < 3 * KD:
                    ch = (tid // KD) * P + h * KD + tid % KD
                    x = getf_many([(mb("proj"), s * NIN + ch)])[0]
                    st = [
                        ld_bf16(r_cs, ch * (CONV_TAPS - 1) + j)
                        for j in range(CONV_TAPS - 1)
                    ]
                    cw = [ld_f32(r_cw, ch * CONV_TAPS + j) for j in range(CONV_TAPS)]
                    a = cw[CONV_TAPS - 1] * x
                    for j in range_constexpr(CONV_TAPS - 1):
                        a = a + cw[j] * st[j]
                    lds_st(xs, tid, bf16_round(div(a, 1.0 + _exp(-a))))
                    if valid:
                        for j in range_constexpr(CONV_TAPS - 1):
                            nxt = st[j + 1] if j + 1 < CONV_TAPS - 1 else x
                            bo.buffer_store(
                                fx.Float32(nxt).to(fx.BFloat16),
                                r_cs,
                                ch * (CONV_TAPS - 1) + j,
                            )
                if tid < 2 * KD:
                    g = getf_many(
                        [(mb("gates"), s * 2 * P + (tid // KD) * P + h * KD + tid % KD)]
                    )[0]
                    lds_st(xs, G1O + tid, g)
                if tid == 0:
                    lds_st(misc, 0, getf_many([(mb("proj"), s * NIN + 3 * P + h)])[0])
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
                amp = _exp(ld_f32(r_alog, h))
                vr = tid // 4
                k0 = (tid % 4) * 32
                r_rs = _rsrc(
                    rec_state
                    + (slot * rec_stride + fx.Int64(h * KD * KD)) * fx.Int64(4)
                )
                hv = []
                for q in range_constexpr(8):
                    w4 = fx.Vector(
                        bo.buffer_load(
                            r_rs, vr * KD + k0 + 4 * q, vec_width=4, dtype=T.f32
                        )
                    )
                    hv += [w4[e] for e in range(4)]
                kk = [lds_ld(xs, KO + k0 + i) * rk for i in range(32)]
                dot = fx.Float32(0.0)
                for i in range_constexpr(32):
                    g = lds_ld(xs, G1O + k0 + i) + ld_f32(r_dtb, h * KD + k0 + i)
                    hv[i] = hv[i] * _exp(div(KDA_LOWER_BOUND, 1.0 + _exp(-(amp * g))))
                    dot = dot + hv[i] * kk[i]
                for off in (1, 2):
                    dot = _xred(dot, off, lambda a, b: a + b)
                vn = beta * (lds_ld(xs, VO + vr) - dot)
                o = fx.Float32(0.0)
                for i in range_constexpr(32):
                    hv[i] = hv[i] + vn * kk[i]
                    o = o + hv[i] * (lds_ld(xs, QO + k0 + i) * rq)
                for off in (1, 2):
                    o = _xred(o, off, lambda a, b: a + b)
                if valid:
                    for q in range_constexpr(8):
                        bo.buffer_store(
                            fx.Vector.from_elements(hv[4 * q : 4 * q + 4], fx.Float32),
                            r_rs,
                            vr * KD + k0 + 4 * q,
                        )
                if tid % 4 == 0:
                    lds_st(xs, OO + vr, valid.select(bf16_round(o), fx.Float32(0.0)))
                gpu.barrier()
                if wave == 0:
                    e0 = 2 * lane
                    o0 = lds_ld(xs, OO + e0)
                    o1 = lds_ld(xs, OO + e0 + 1)
                    rstd = _rsq(wave_sum(o0 * o0 + o1 * o1) * (1.0 / KD) + O_NORM_EPS)
                    y0 = o0 * rstd * ld_f32(r_won, e0) * sigmoid(lds_ld(xs, G2O + e0))
                    y1 = (
                        o1
                        * rstd
                        * ld_f32(r_won, e0 + 1)
                        * sigmoid(lds_ld(xs, G2O + e0 + 1))
                    )
                    put(mb("o_in"), (s * P + h * KD + e0) // 2, bf16_pair(y0, y1))
                gpu.barrier()

        # ===================== 6. o_proj partial -> every peer's attention region
        # One task a 16-row tile of the row-major [HIDDEN, P] weight; the FFN seam
        # sums the TP partials as it does vLLM's.
        O_CPW = P // 64 // WAVES
        O_BATCH = min(4, O_CPW)

        def run_oproj():
            r_w = _rsrc(w_o)
            attn_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            for t in range(start("oproj"), HIDDEN // O_TILE, G):
                t = fx.Int32(t)
                stamp("oproj")
                row0 = t * O_TILE

                def u_o(c):
                    kc = wave * O_CPW + c
                    return unit_rm(
                        r_w, row0, kc, P, HIDDEN, n_sel() * (P // 2) + kc * 32
                    )

                pre = [u_o(c) for c in range(O_BATCH)]
                stage_rows(mb("o_in"), P)
                gpu.barrier()
                acc = run_units(u_o, O_CPW, O_BATCH, pre)
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

        if const_expr(kda):
            run_seam("aseam", "attn")
            run_gate("agate", "attn")
            run_inproj()
            run_fgb()
            run_kda()
            run_oproj()
            run_seam("seam", "ffn_kda")
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
            scratch,
            sym,
            peers,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch, SC, SY
