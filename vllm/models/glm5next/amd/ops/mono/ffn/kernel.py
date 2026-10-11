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
    MOE_SLOTS,
    N_EXPERTS,
    ROUTE_SCALE,
    ROUTER_CANDIDATES,
    SHARED_EXPERT,
    SINKHORN_ITERS,
    SWIGLU_LIMIT,
    TOP_K,
)
from vllm.models.glm5next.amd.ops.mono.ffn.layout import (
    N_MIX,
    N_ROUTER,
    N_ROW_TILES,
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
):
    """Return the ``@flyc.jit`` launcher of one rank's FFN layer for S rows.

    ``poll_limit`` bounds the re-polls of every mailbox wait. An expired wait
    stores 1 into its stage's error word (``poll_err_addr``) and, with TP peers,
    into this rank's flag on rank 0 (``poll_xrank_addr``), then continues, so a
    stalled peer finishes the launch with wrong values instead of hanging. With
    ``poll_early_out`` every later wait of the same step returns at its first retry.
    """
    assert 2 <= S <= 16 and 1 <= launches_per_step <= LAYER_SLOTS
    assert poll_limit is None or poll_limit > 0
    assert inter % 128 == 0
    W = npes
    FAULT = abi.fault_spec()
    expert_inter = inter
    SHARED = SHARED_EXPERT
    G = BLOCKS
    SC, SY = layout(S, W, expert_inter)
    DN_TILE = dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = S * MOE_SLOTS
    # LDS X users: one router row (bf16), every row's MXFP8 FFN input (ug), the
    # seam tile (x_attn + R'), every row's MXFP8 mid (down)
    X_WORDS = max(
        HIDDEN // 2,
        S * HIDDEN // 4,
        S * ROW_TILE * (1 + HC_MULT),
        down_x_words(S, expert_inter),
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

    tasks = dict(stage_tasks(S, expert_inter))
    base, acc = {}, 0
    for name in ("seam", "gate", "router", "ug", "down"):
        base[name] = acc % G
        acc += tasks[name]

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
        # One task a 32-column tile, every row: x_attn = the TP sum of vLLM's
        # unreduced attention output (bf16), R'[j] = bf16(sum_i comb[i][j] R[i]
        # + post[j] x_attn) -> res_out and mailbox ``rnew``, and the tile's share of
        # the mix projection (fn . R') and of sum(R'^2) -> mailbox ``pmix``.
        r_attn = _rsrc(attn_part)
        r_res = _rsrc(res_in)
        r_fn = _rsrc(hc_fn)
        r_post = _rsrc(post_in)
        r_comb = _rsrc(comb_in)
        XA = 0
        RN = S * ROW_TILE
        NPAIR = S * ROW_TILE // 2

        def seam_post(item, col0):
            """R'[j] for one (row, stream, column pair) item of a seam tile."""
            s = item // (HC_MULT * (ROW_TILE // 2))
            j = (item // (ROW_TILE // 2)) % HC_MULT
            c = (item % (ROW_TILE // 2)) * 2
            pj = ld_f32(r_post, s * HC_MULT + j)
            acc0 = fx.Float32(0.0)
            acc1 = fx.Float32(0.0)
            for i in range_constexpr(HC_MULT):
                cij = ld_f32(r_comb, (s * HC_MULT + i) * HC_MULT + j)
                r0, r1 = ld_pair(r_res, (s * HC_MULT + i) * HIDDEN + col0 + c)
                acc0 = acc0 + cij * r0
                acc1 = acc1 + cij * r1
            n0 = bf16_round(acc0 + pj * lds_ld(xs, XA + s * ROW_TILE + c))
            n1 = bf16_round(acc1 + pj * lds_ld(xs, XA + s * ROW_TILE + c + 1))
            elem = (s * HC_MULT + j) * HIDDEN + col0 + c
            bo.buffer_store(
                fx.Vector.from_elements([n0, n1], fx.Float32).to(fx.BFloat16),
                _rsrc(res_out),
                elem,
            )
            put(mb("rnew"), elem // 2, bf16_pair(n0, n1))
            lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c, n0)
            lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c + 1, n1)

        for t in range(start("seam"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("seam")
            col0 = t * ROW_TILE
            attn_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
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
            if tid < NPAIR:
                s = tid // (ROW_TILE // 2)
                c = (tid % (ROW_TILE // 2)) * 2
                if const_expr(W == 1):
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
                put(mb("pmix"), (s * N_ROW_TILES + t) * PMIX + k, acc)
            gpu.barrier()

        # ========= 2. gate: mixes -> pre / post / comb (Sinkhorn), FFN input row
        # One CTA a row. The tile partials are summed in a fixed order, so every
        # rank computes bit-identical mixes.
        NP = 16
        TPP = N_ROW_TILES // NP
        CPT = HIDDEN // THREADS
        r_scale = _rsrc(hc_scale)
        r_base = _rsrc(hc_base)
        r_g = _rsrc(g_norm)
        for t in range(start("gate"), S, G):
            s = fx.Int32(t)
            stamp("gate")
            gam = [ld_bf16(r_g, tid * CPT + q) for q in range(CPT)]
            if tid < PMIX * NP:
                k = tid // NP
                part = tid % NP
                vals = getf_many(
                    [
                        (mb("pmix"), (s * N_ROW_TILES + part * TPP + q) * PMIX + k)
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
                        sigmoid(m[HC_MULT + i] * sc[1] + hb[HC_MULT + i]) * HC_POST_MULT
                    )
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
                        mb("rnew"),
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
                    mb("a"),
                    (s * HIDDEN + tid * CPT + 2 * q) // 2,
                    bf16_pair(
                        li[2 * q] * rstd * gam[2 * q],
                        li[2 * q + 1] * rstd * gam[2 * q + 1],
                    ),
                )

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

                def ug8_units(e, sample, live=None):
                    if const_expr(live is None):
                        rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                        rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                            num_records_bytes=live.select(
                                fx.Int32(UG_W_BYTES), fx.Int32(0)
                            ),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                            num_records_bytes=live.select(
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

                shared_pre = ug8_units(fx.Int32(SHARED), None, has_sh)
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
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
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
            scratch,
            sym,
            peers,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch, SC, SY
