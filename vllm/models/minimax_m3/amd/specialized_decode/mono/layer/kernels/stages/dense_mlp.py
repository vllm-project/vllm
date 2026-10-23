# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense-layer stages (``dense_post``): the attention output's per-token FP8, the
post-attention norm, the MLP's gate / up GEMV + swiglu and its down GEMV.

The numerics follow the original dense layer's ops: ``dynamic_per_token_scaled_
quant`` (scale = amax * (1 / FP8_MAX), x * rcp(scale)) in front of o_proj and
down_proj, ``fused_allreduce_gemma_rms_norm_quant`` (scale = amax / FP8_MAX, a
true divide) in front of gate_up, ``gemm_a8w8_bpreshuffle``'s bf16(acc * x_scale *
w_scale), and Triton's ``_swiglu_oai_kernel``."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...config import (
    DENSE_INTER,
    HIDDEN,
)
from ...layout import (
    DMID_ROW,
    DN_ROWS,
    DX_ROW,
    N_DENSE_UG,
    N_DN,
    O_K,
    UG_ROWS,
)
from ...support.device.mx import (
    FP8_MAX,
    UNIT_SCALE,
    mfma_scaled,
)
from ...support.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    div_rn,
    fp8_pack4,
    hw_exp2,
    hw_rsq,
    mfma_fp8,
    row_sum,
    rows_to_lds,
    rsrc,
    traced,
    wave_max,
    wave_sum,
)
from ...support.device.sync import (
    preg,
    publish,
)
from ...support.plan.arith import (
    pick,
)
from ...support.plan.execution import (
    THREADS,
    WAVES,
)
from ..common import (
    ld_bf16x12,
    per_token_fp8_scale,
)

NE = HIDDEN // THREADS  # a norm thread's elements of the row
DE = 8  # a down thread's mid elements of a token (DENSE_INTER // DE threads)
MID_GROUP = 4  # tokens whose mids a down task holds at once


@traced
def dense_mlp_defs(dctx):
    """The dense layer's stages after the attention."""
    AG_RS = dctx["AG_RS"]
    G = dctx["G"]
    SY = dctx["SY"]
    attn = dctx["attn"]
    block_max = dctx["block_max"]
    eps = dctx["eps"]
    ffn_finish = dctx["ffn_finish"]
    g_post = dctx["g_post"]
    h_in = dctx["h_in"]
    h_mid = dctx["h_mid"]
    lane = dctx["lane"]
    l16 = dctx["l16"]
    g4 = dctx["g4"]
    mb = dctx["mb"]
    mb_poll = dctx["mb_poll"]
    mb_put = dctx["mb_put"]
    mb_put_bf = dctx["mb_put_bf"]
    misc = dctx["misc"]
    mr = dctx["mr"]
    red = dctx["red"]
    red2 = dctx["red2"]
    s_dn = dctx["s_dn"]
    s_gu = dctx["s_gu"]
    start = dctx["start"]
    swiglu_alpha = dctx["swiglu_alpha"]
    swiglu_beta = dctx["swiglu_beta"]
    swiglu_limit = dctx["swiglu_limit"]
    sym = dctx["sym"]
    tid = dctx["tid"]
    tokens = dctx["tokens"]
    v4f = dctx["v4f"]
    w_dn = dctx["w_dn"]
    w_gu = dctx["w_gu"]
    wave = dctx["wave"]
    xs = dctx["xs"]

    def acquire():
        fx.memory_fence(
            syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
        )

    # ============================================ the attention output's FP8
    def quant_token(k):
        """Token k's attention output (thread = 4 of its O_K values), per-token
        FP8 -> (this thread's word, scale): ``merge_token``'s interface."""
        w = fx.Vector(
            bo.buffer_load(
                rsrc(attn), (k * O_K + tid * 4) // 2, vec_width=2, dtype=T.i32
            )
        )
        a = w.bitcast(fx.BFloat16).to(fx.Float32)
        amax = block_max(
            fx.max(
                fx.max(fmath.absf(a[0]), fmath.absf(a[1])),
                fx.max(fmath.absf(a[2]), fmath.absf(a[3])),
            )
        )
        x_scale, inv = per_token_fp8_scale(amax)
        return fp8_pack4(a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv), x_scale

    def stage_quant():
        """S > 1: a task per token publishes its FP8 row (plain words, then the
        scale: its pair is the flag) for the o tasks, as K4's merge does."""
        if const_expr(tokens > 1):
            for qk in range(start("merge"), tokens, G):
                qk = fx.Int32(qk)
                word, x_scale = quant_token(qk)
                bo.buffer_store(
                    word,
                    rsrc(mb("attnq")),
                    qk * (O_K // 4) + tid,
                    cache_modifier=CM_DEV,
                )
                publish(mb_put, mr("attnq_s"), qk, x_scale, tid == 0)

    # ======================================== the post-attention norm's FP8

    def norm_token(k):
        """Token k: acc = bf16(attention sum) + residual (h_mid where this task
        writes it), gemma RMSNorm, per-token FP8 (the fused all-reduce + norm +
        quant) -> (this thread's NE / 4 words, scale). Thread = NE consecutive
        elements."""
        e0 = tid * NE
        gp = ld_bf16x12(rsrc(g_post), e0 // 2)
        hres = ld_bf16x12(rsrc(h_in), (k * HIDDEN + e0) // 2)
        a_src = preg(sym, SY["a_ag"], "a_ag") if AG_RS else mr("a")
        arow = mb_poll(
            [(a_src, (k * HIDDEN + e0) // 2 + 2 * h, 2) for h in range(NE // 4)]
        )
        v = [
            f + fx.Float32(hres[4 * h + 2 * qq + i])
            for h in range(NE // 4)
            for qq in range(2)
            for i, f in enumerate(bf2_f32(arow[h][qq]))
        ]
        if const_expr(AG_RS):  # the o tasks wrote no h_mid: the norm task does
            hi = (
                fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            )
            hw = (k * HIDDEN + e0) // 2
            bo.buffer_store(
                fx.Vector.from_elements([hi[e] for e in range(4)], fx.Int32),
                rsrc(h_mid),
                hw,
            )
            bo.buffer_store(
                fx.Vector.from_elements([hi[4], hi[5]], fx.Int32), rsrc(h_mid), hw + 4
            )
        ss = fx.Float32(0.0)
        for e in range_constexpr(NE):
            ss = fmath.fma(v[e], v[e], ss)
        ss = wave_sum(ss)
        gpu.barrier()
        if lane == 0:
            fx.ptr_store(ss, red + wave)
        gpu.barrier()
        tot = fx.ptr_load(red)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        rstd = hw_rsq(tot / float(HIDDEN) + eps)
        normed = [v[e] * rstd * (fx.Float32(gp[e]) + 1.0) for e in range(NE)]
        am = fmath.absf(normed[0])
        for e in range_constexpr(1, NE):
            am = fx.max(am, fmath.absf(normed[e]))
        amax = block_max(am)
        x_scale = (amax == 0.0).select(fx.Float32(1.0), amax / FP8_MAX)
        x_rcp = 1.0 / x_scale
        q = [div_rn(normed[e], x_scale, x_rcp) for e in range(NE)]
        words = [fp8_pack4(*q[4 * j : 4 * j + 4]) for j in range(NE // 4)]
        return words, x_scale

    def stage_norm():
        """S > 2: a task per token publishes its FP8 row (plain words, then the
        scale) for every gate / up task: they would each norm every token."""
        if const_expr(tokens > 2):
            for nk in range(start("norm"), tokens, G):
                nk = fx.Int32(nk)
                nwords, nscale = norm_token(nk)
                for j in range_constexpr(NE // 4):
                    bo.buffer_store(
                        nwords[j],
                        rsrc(mb("dx8")),
                        nk * (HIDDEN // 4) + tid * (NE // 4) + j,
                        cache_modifier=CM_DEV,
                    )
                publish(mb_put, mr("dxs"), nk, nscale, tid == 0)

    def stage_xn():
        """Every token's FP8 row into xs (row k at k DX_ROW) -> the scales: S <= 2
        normed here, else the norm tasks'."""
        if const_expr(tokens <= 2):
            scales = []
            for k in range_constexpr(tokens):
                kwords, kscale = norm_token(fx.Int32(k))
                for j in range_constexpr(NE // 4):
                    fx.ptr_store(
                        kwords[j].bitcast(fx.Float32),
                        xs + (k * DX_ROW + tid * (NE // 4) + j),
                    )
                scales.append(kscale)
        else:
            got = mb_poll([(mr("dxs"), k, 1) for k in range(tokens)])
            scales = [got[k][0].bitcast(fx.Float32) for k in range(tokens)]
            acquire()
            rows_to_lds(tid, mb("dx8"), tokens, HIDDEN // 4, xs, DX_ROW)
        gpu.barrier()
        return scales

    # ================================================ gate / up -> swiglu
    UG_KC = HIDDEN // 64  # 64-k chunks of a row: one dwordx4 of weight per lane
    UG_CPW = UG_KC // WAVES

    def swiglu(g, u):
        """Triton's _swiglu_oai_kernel on the bf16 gate / up values (f32)."""
        g = fx.min(g, fx.Float32(swiglu_limit))
        u = fx.min(fx.max(u, fx.Float32(-swiglu_limit)), fx.Float32(swiglu_limit))
        z = g * swiglu_alpha
        sig = 1.0 / (1.0 + hw_exp2(-z * LOG2E))
        return g * sig * (u + swiglu_beta)

    def stage_ug():
        r_w = rsrc(w_gu)
        for ut in range(start("ug"), N_DENSE_UG, G):
            ut = fx.Int32(ut)
            # gate rows UG_ROWS ut .., up rows DENSE_INTER + UG_ROWS ut ..: row
            # group rg's 64-k chunk kc is 1 KB at ((rg K / 32 + 2 kc) 512)
            wts = []
            for rg in (ut, N_DENSE_UG + ut):
                for c in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + c
                    wts.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_w,
                                ((rg * (HIDDEN // 32) + kc * 2) * 512 + lane * 16) // 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
            # thread tid < 8 S: token tid / 8, mid columns r0, r0 + 1
            r0 = (tid % 8) * 2
            wsc = [
                fx.Float32(
                    bo.buffer_load(
                        rsrc(s_gu),
                        base + ut * UG_ROWS + r0 + i,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                for base in (0, DENSE_INTER)
                for i in range(2)
            ]
            x_scales = stage_xn()
            col_tok = fx.min(l16, tokens - 1)
            accs = []
            for g_ in range_constexpr(2):
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for cc in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + cc
                    xb = fx.Vector(
                        fx.ptr_load(
                            xs + (col_tok * DX_ROW + kc * 16 + g4 * 4),
                            result_type=v4f,
                        )
                    ).bitcast(fx.Int64)
                    wa = wts[g_ * UG_CPW + cc].bitcast(fx.Int64)
                    for h in range_constexpr(2):
                        c = mfma_fp8(wa[h], xb[h], c)
                accs.append(c)
            fx.ptr_store(accs[0], red + (wave * 64 + lane) * 4)
            fx.ptr_store(accs[1], red2 + (wave * 64 + lane) * 4)
            gpu.barrier()
            if tid < 8 * tokens:
                tk = tid // 8
                xsk = pick(x_scales, tk)
                mids = []
                for i in range_constexpr(2):
                    r = r0 + i
                    gs = row_sum(red, r, tk)
                    us = row_sum(red2, r, tk)
                    gate = bf16_round(gs * xsk * wsc[i])
                    up = bf16_round(us * xsk * wsc[2 + i])
                    mids.append(swiglu(gate, up))
                mb_put_bf(mr("dmid"), tk * DENSE_INTER + ut * UG_ROWS + r0, mids)
            gpu.barrier()

    # ===================================================== the mids' FP8
    def mid_fp8(ks):
        """Tokens ``ks``' mids (thread tid < DENSE_INTER / DE holds DE of each) ->
        per token (this thread's DE / 4 FP8 words, scale), as
        dynamic_per_token_scaled_quant (amax over the row)."""
        mine = tid < DENSE_INTER // DE
        mt = fx.min(tid, DENSE_INTER // DE - 1)
        got = mb_poll(
            [
                (mr("dmid"), (k * DENSE_INTER + mt * DE) // 2 + p, 1)
                for k in ks
                for p in range(DE // 2)
            ]
        )
        mv = [
            [f for p in range(DE // 2) for f in bf2_f32(got[j * (DE // 2) + p][0])]
            for j in range(len(ks))
        ]
        for j in range_constexpr(len(ks)):
            am = fmath.absf(mv[j][0])
            for e in range_constexpr(1, DE):
                am = fx.max(am, fmath.absf(mv[j][e]))
            am = wave_max(mine.select(am, fx.Float32(0.0)))
            if lane == 0:
                fx.ptr_store(am, red + (j * WAVES + wave))
        gpu.barrier()
        out = []
        for j in range_constexpr(len(ks)):
            amax = fx.ptr_load(red + j * WAVES)
            for w in range_constexpr(1, WAVES):
                amax = fx.max(amax, fx.ptr_load(red + (j * WAVES + w)))
            x_scale, inv = per_token_fp8_scale(amax)
            words = [
                fp8_pack4(*[mv[j][4 * q + e] * inv for e in range(4)])
                for q in range(DE // 4)
            ]
            out.append((words, x_scale))
        gpu.barrier()
        return out

    def stage_mq():
        """S > 2: a task per token publishes its mids' FP8 (plain words, then the
        scale: its pair is the flag) for every down task -- each polling every
        token's 3072 mids itself read ~75 MB at S = 16."""
        if const_expr(tokens > 2):
            for qk in range(start("mq"), tokens, G):
                qk = fx.Int32(qk)
                mt = fx.min(tid, DENSE_INTER // DE - 1)
                words, x_scale = mid_fp8([qk])[0]
                if tid < DENSE_INTER // DE:
                    for q in range_constexpr(DE // 4):
                        bo.buffer_store(
                            words[q],
                            rsrc(mb("dmid8")),
                            qk * (DENSE_INTER // 4) + mt * (DE // 4) + q,
                            cache_modifier=CM_DEV,
                        )
                publish(mb_put, mr("dmids"), qk, x_scale, tid == 0)

    # ============================================================== down
    DN_CPW = DENSE_INTER // 64 // 4  # 64-k chunks a wave: 4 waves a 16-row group

    def stage_down():
        r_w = rsrc(w_dn)
        for dt in range(start("down"), N_DN, G):
            dt = fx.Int32(dt)
            rg = dt * 2 + wave // 4
            dwts = []
            for cc in range_constexpr(DN_CPW):
                kc = (wave % 4) * DN_CPW + cc
                dwts.append(
                    fx.Vector(
                        bo.buffer_load(
                            r_w,
                            ((rg * (DENSE_INTER // 32) + kc * 2) * 512 + lane * 16)
                            // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                )
            dws = fx.Float32(
                bo.buffer_load(
                    rsrc(s_dn), dt * DN_ROWS + tid % DN_ROWS, vec_width=1, dtype=T.f32
                )
            )
            # every token's mid as per-token FP8 into xs row k at k DMID_ROW, its
            # scale at red2[k]: S > 2 the mid quant tasks', else quantized here,
            # MID_GROUP tokens at a time (all 16 held at once: S >= 13 spilled)
            mt = fx.min(tid, DENSE_INTER // DE - 1)
            if const_expr(tokens > 2):
                got = mb_poll([(mr("dmids"), k, 1) for k in range(tokens)])
                if tid < tokens:
                    fx.ptr_store(
                        pick([g[0] for g in got], tid).bitcast(fx.Float32), red2 + tid
                    )
                acquire()
                rows_to_lds(tid, mb("dmid8"), tokens, DENSE_INTER // 4, xs, DMID_ROW)
            else:
                for k0 in range_constexpr(0, tokens, MID_GROUP):
                    ks = list(range(k0, min(k0 + MID_GROUP, tokens)))
                    for k, (words, x_scale) in zip(ks, mid_fp8(ks)):
                        if tid == 0:
                            fx.ptr_store(x_scale, red2 + k)
                        if tid < DENSE_INTER // DE:
                            for q in range_constexpr(DE // 4):
                                fx.ptr_store(
                                    words[q].bitcast(fx.Float32),
                                    xs + (k * DMID_ROW + mt * (DE // 4) + q),
                                )
            gpu.barrier()
            col_tok = fx.min(l16, tokens - 1)
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            # chunks kc, kc + 1 in one 16x16x128 FP8 MFMA (unit scales), as o_proj
            for p in range_constexpr(DN_CPW // 2):
                xb = [
                    fx.Vector(
                        fx.ptr_load(
                            xs
                            + (
                                col_tok * DMID_ROW
                                + ((wave % 4) * DN_CPW + 2 * p + h) * 16
                                + g4 * 4
                            ),
                            result_type=v4f,
                        )
                    ).bitcast(fx.Int32)
                    for h in range(2)
                ]
                c = mfma_scaled(
                    fx.Vector.from_elements(
                        [dwts[2 * p + h][e] for h in range(2) for e in range(4)],
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
            if tid < DN_ROWS * tokens:
                tk = tid // DN_ROWS
                grp = (tid % DN_ROWS) // 16
                rr = tid % 16
                tot = row_sum(red, rr, tk, [grp * 4 + w for w in range(4)])
                fx.ptr_store(bf16_round(tot * fx.ptr_load(red2 + tk) * dws), misc + tid)
            gpu.barrier()
            ffn_finish(dt)

    return {
        "quant_token": quant_token,
        "stage_quant": stage_quant,
        "stage_norm": stage_norm,
        "stage_ug": stage_ug,
        "stage_mq": stage_mq,
        "stage_down": stage_down,
    }
