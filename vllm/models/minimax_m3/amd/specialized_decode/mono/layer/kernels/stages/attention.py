# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/attention.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the split-K sparse attention (one 256-key partition x 16 heads a task);
the partitions' merge and per-token FP8 quantization; o_proj and the attention
all-reduce."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...config import (
    HEAD_DIM,
    HIDDEN,
    MAX_TOKENS,
    PAGE16,
)
from ...layout import (
    N_O,
    N_SPLIT,
    O_K,
    O_ROW,
    O_ROWS,
    SPLIT_KEYS,
    H,
)
from ...support.device.mx import (
    FP8_MAX,
    UNIT_SCALE,
    mfma_scaled,
)
from ...support.device.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    mfma_fp8,
    row_sum,
    rows_to_lds,
    rsrc,
    traced,
    uniform,
    wave_max,
    xshfl,
)
from ...support.device.ranks import (
    sum_partials,
)
from ...support.device.sync import (
    preg,
    publish,
)
from ...support.plan.arith import (
    pick,
)
from ...support.plan.execution import (
    WAVES,
)
from ..common import (
    ld_bf16x4,
    per_token_fp8_scale,
)
from ..pre_attn import (
    N_HEAD_TASKS as K1_HEAD_TASKS,
)
from .shared import (
    NEG,
    PAGE_BYTES,
)


@traced
def split_defs(k4ctx):
    """The functions of the split-K sparse attention (one 256-key partition x 16 heads a
    task)."""
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    acquire = k4ctx["acquire"]
    blk = k4ctx["blk"]
    fuse_k1 = k4ctx["fuse_k1"]
    g4 = k4ctx["g4"]
    hdone_mb = k4ctx["hdone_mb"]
    hst = k4ctx["hst"]
    k1_poll = k4ctx["k1_poll"]
    k_cache = k4ctx["k_cache"]
    k_scale = k4ctx["k_scale"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    mb_put = k4ctx["mb_put"]
    mb_put_bf = k4ctx["mb_put_bf"]
    mr = k4ctx["mr"]
    opart = k4ctx["opart"]
    p8 = k4ctx["p8"]
    q = k4ctx["q"]
    q8 = k4ctx["q8"]
    # a layer reusing a selection reads it instead (``index_topk``)
    task_blocks = k4ctx["select_blocks" if k4ctx["index_topk"] else "reuse_selection"]
    sm_scale = k4ctx["sm_scale"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    v2i = k4ctx["v2i"]
    v_cache = k4ctx["v_cache"]
    v_scale = k4ctx["v_scale"]
    wave = k4ctx["wave"]

    # Each stage is its own function: flydsl carries a local reassigned inside a
    # dynamic if / for as region state, so names must not be shared across stages.
    # ============================================================ 1. split
    def stage_split():
        r_k, r_v = rsrc(k_cache), rsrc(v_cache)
        r_ks, r_vs = rsrc(k_scale), rsrc(v_scale)
        PPW = SPLIT_KEYS // WAVES // PAGE16  # pages per wave
        for ts in range(start("split"), N_SPLIT * tokens, G):
            ts = fx.Int32(ts)  # token tok's partition t: ts = tok N_SPLIT + t
            tok = ts // N_SPLIT
            t = ts % N_SPLIT
            task_blocks(tok, t)
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
            pages = [uniform(fx.ptr_load(blk + (wave * PPW + j))) for j in range(PPW)]
            qv = ld_bf16x4(rsrc(q), tok * O_K + tid * 4, CM_K1)
            # K (QK B operand, key = lane % 16 of page j), V (PV B operand: dims
            # 16 jd + lane % 16, keys 8 (lane / 16) .. + 8), per-token scales
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
            kss = [
                fx.Float32(
                    bo.buffer_load(
                        r_ks,
                        0,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=CM_K1,
                    )
                )
                for j in range(PPW)
            ]
            vss = [
                fx.Float32(
                    bo.buffer_load(
                        r_vs,
                        0,
                        vec_width=1,
                        dtype=T.f32,
                        cache_modifier=CM_K1,
                    )
                )
                for j in range(PPW)
            ]
            fx.ptr_store(fp8_pack4(qv[0], qv[1], qv[2], qv[3]), q8 + tid)
            gpu.barrier()
            stamp(12)
            # QK per page j: A = q (heads), B = K (keys); lane holds heads 4 g4 + e, key
            # l16
            key0 = t * SPLIT_KEYS + wave * (PPW * PAGE16)
            valid = [key0 + j * PAGE16 + l16 < n_ctx for j in range(PPW)]
            sc = []
            for j in range_constexpr(PPW):
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for s in range_constexpr(HEAD_DIM // 32):
                    qw = fx.Vector(
                        fx.ptr_load(q8 + (l16 * 32 + 8 * s + 2 * g4), result_type=v2i)
                    )
                    c = mfma_fp8(
                        qw.bitcast(fx.Int64)[0], kw[j * (HEAD_DIM // 32) + s], c
                    )
                qk = sm_scale * kss[j]
                sc.append(
                    [valid[j].select(qk * c[e], fx.Float32(NEG)) for e in range(4)]
                )
            # partition max per head and max v_scale: in-wave butterfly, then LDS
            vloc = fx.max(
                valid[0].select(vss[0], fx.Float32(0.0)),
                valid[1].select(vss[1], fx.Float32(0.0)),
            )
            vmax_w = wave_max(vloc)
            for e in range_constexpr(4):
                mh = butterfly(fx.max(sc[0][e], sc[1][e]), (8, 4, 2, 1), fx.max)
                if l16 == 0:
                    fx.ptr_store(mh, hst + (wave * H + g4 * 4 + e))
            if lane == 0:
                fx.ptr_store(vmax_w, hst + (3 * WAVES * H + wave))
            gpu.barrier()
            stamp(13)
            # every wave partial is read before the first p8 store below: LDS
            # stores in between serialize each read behind its own wait
            vmax = fx.ptr_load(hst + 3 * WAVES * H)
            for w in range_constexpr(1, WAVES):
                vmax = fx.max(vmax, fx.ptr_load(hst + (3 * WAVES * H + w)))
            mxs = []
            for e in range_constexpr(4):
                mx = fx.ptr_load(hst + (g4 * 4 + e))
                for w in range_constexpr(1, WAVES):
                    mx = fx.max(mx, fx.ptr_load(hst + (w * H + g4 * 4 + e)))
                mxs.append(mx)
            fscale = FP8_MAX * hw_rcp(vmax + 1e-8)
            pscale = vmax * (1.0 / FP8_MAX)
            for e in range_constexpr(4):
                head = g4 * 4 + e
                mx = mxs[e]
                ls = fx.Float32(0.0)
                for j in range_constexpr(PPW):
                    pj = hw_exp2((sc[j][e] - mx) * LOG2E)
                    ls = ls + pj
                    b8 = (
                        fp8_pack4(
                            (valid[j].select(vss[j], fx.Float32(0.0)) * fscale) * pj,
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
                        fx.ptr_store(w8, p8 + (head * (SPLIT_KEYS // 4) + kloc // 4))
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
                    + (l16 * (SPLIT_KEYS // 4) + (wave * (PPW * PAGE16) + 8 * g4) // 4),
                    result_type=v2i,
                )
            ).bitcast(fx.Int64)[0]
            # keys past the context carry P = 0, but a NaN V byte would still poison
            nv = n_ctx - (key0 + 8 * g4)
            vmask = (nv >= 8).select(
                fx.Int64(-1),
                (nv <= 0).select(fx.Int64(0), (fx.Int64(1) << (fx.Int64(nv) * 8)) - 1),
            )
            for jd in range_constexpr(HEAD_DIM // 16):
                cv = mfma_fp8(pw, vw[jd] & vmask, fx.Vector.filled(4, 0.0, fx.Float32))
                for e in range_constexpr(4):
                    fx.ptr_store(
                        cv[e],
                        opart + (wave * O_K + (g4 * 4 + e) * HEAD_DIM + 16 * jd + l16),
                    )
            gpu.barrier()
            stamp(15)
            # sum the waves' partials; gluon: acc = prob_scale * PV, then * (1 /
            # exp_sum), bf16
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
            mb_put_bf(mr("sp_o"), ts * O_K + e0, ov)
            if tid < H:
                mb_put(
                    mr("sp_m"),
                    ts * H + tid,
                    fx.ptr_load(hst + (2 * WAVES * H + tid)),
                )
                lt = fx.ptr_load(hst + (WAVES * H + tid))
                for w in range_constexpr(1, WAVES):
                    lt = lt + fx.ptr_load(hst + (WAVES * H + w * H + tid))
                mb_put(mr("sp_l"), ts * H + tid, lt)
            gpu.barrier()

    return {"stage_split": stage_split}


@traced
def merge_defs(k4ctx):
    """The functions of the partitions' merge and per-token FP8 quantization."""
    G = k4ctx["G"]
    block_max = k4ctx["block_max"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    mr = k4ctx["mr"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    wave = k4ctx["wave"]

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
                    (mr("sp_l"), sp0 * H + lane + 64 * i, 1)
                    for i in range(N_SPLIT * H // 64)
                ]
            )
        gpu.barrier()
        # one batch: every partition's (m, l) and this thread's 4 outputs
        got = mb_poll(
            [(mr("sp_m"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
            + [(mr("sp_l"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
            + [(mr("sp_o"), ((sp0 + i) * O_K + e0) // 2, 2) for i in range(N_SPLIT)],
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
        x_scale, inv = per_token_fp8_scale(amax)
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
                publish(mb_put, mr("attnq_s"), k, x_scale, tid == 0)

    return {"merge_token": merge_token, "stage_merge": stage_merge}


@traced
def o_defs(k4ctx):
    """The functions of o_proj and the attention all-reduce."""
    AG_RS = k4ctx["AG_RS"]
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    SY = k4ctx["SY"]
    W = k4ctx["W"]
    acquire = k4ctx["acquire"]
    fuse_k1 = k4ctx["fuse_k1"]
    g4 = k4ctx["g4"]
    h_in = k4ctx["h_in"]
    h_mid = k4ctx["h_mid"]
    k1_poll = k4ctx["k1_poll"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put_bf = k4ctx["mb_put_bf"]
    merge_token = k4ctx["merge_token"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    peer_addr = k4ctx["peer_addr"]
    push_rows = k4ctx["push_rows"]
    rank = k4ctx["rank"]
    rdone_mb = k4ctx["rdone_mb"]
    red = k4ctx["red"]
    s_o = k4ctx["s_o"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    v4f = k4ctx["v4f"]
    w_o = k4ctx["w_o"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]

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
                            cache_modifier=CM_NT,
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
                got = mb_poll([(mr("attnq_s"), k, 1) for k in range(tokens)])
                stamp(3)
                x_scales = [got[k][0].bitcast(fx.Float32) for k in range(tokens)]
                acquire()
                rows_to_lds(tid, mb("attnq"), tokens, O_K // 4, xs, O_ROW)
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
                c = mfma_scaled(
                    fx.Vector.from_elements(
                        [wts[2 * p + h][e] for h in range(2) for e in range(4)],
                        fx.Int32,
                    ),
                    fx.Vector.from_elements(
                        [xb[h][e] for h in range(2) for e in range(4)], fx.Int32
                    ),
                    c,
                    fx.Int32(UNIT_SCALE),
                    fx.Int32(UNIT_SCALE),
                )
            fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            if tid < O_ROWS * tokens:
                tk = tid // O_ROWS
                grp = (tid % O_ROWS) // 16
                rr = tid % 16
                tot = row_sum(red, rr, tk, [grp * 4 + w for w in range(4)])
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
                    own = preg(sym, SY["attn"], "attn")
                    t0, t1 = sum_partials(
                        mb_poll,
                        own,
                        lambda src, k=ptk, r=row: (
                            ((src * MAX_TOKENS + k) * HIDDEN + r) // 2
                        ),
                        W,
                    )
                    for w in range_constexpr(W):
                        mb_put_bf(
                            preg(peer_addr(w), SY["a_ag"], "a_ag"),
                            ptk * HIDDEN + row,
                            [t0, t1],
                        )
            if const_expr(not AG_RS):  # noqa: SIM102
                # this task's rows of the all-reduce, finished here (GLM's
                # peer_reduce): every peer's partials summed in rank order, bf16, plus
                # the residual -> h_mid and the 'a' mailbox. A router task then reads
                # 24 KB of device-scope bf16, not 4 peers' 48 KB of system-scope ones
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
                    own = preg(sym, SY["attn"], "attn")
                    t0, t1 = sum_partials(
                        mb_poll,
                        own,
                        lambda src, k=ptk, r=row: (
                            ((src * MAX_TOKENS + k) * HIDDEN + r) // 2
                        ),
                        W,
                    )
                    r0, r1 = bf2_f32(hr)
                    a0 = bf16_round(t0) + r0
                    a1 = bf16_round(t1) + r1
                    # the rank sum only: the router adds the residual in f32, so the
                    # norm reads bf16(sum) + residual unrounded, as the original path
                    mb_put_bf(mr("a"), ptk * HIDDEN + row, [t0, t1])
                    bo.buffer_store(
                        fx.Vector.from_elements([a0, a1], fx.Float32).to(fx.BFloat16),
                        rsrc(h_mid),
                        ptk * HIDDEN + row,
                    )

    return {"stage_o": stage_o}
