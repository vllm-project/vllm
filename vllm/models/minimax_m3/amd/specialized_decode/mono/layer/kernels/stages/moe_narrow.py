# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/moe_narrow.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the per-(token, slot) MoE (S <= 4)."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ...config import (
    INTER,
    MOE_SLOTS,
    TOP_K,
)
from ...layout import (
    DN_ROWS,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    SH_TASKS,
    XN8_ROW,
    XSC_ROW,
    shared_after_routing,
)
from ...support.device.mx import (
    FP4,
    mfma_scaled,
)
from ...support.device.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_pack4,
    row_sum,
    traced,
)
from ...support.plan.execution import (
    THREADS,
    WAVES,
)
from ...support.plan.trace import (
    enter_stage,
)
from .shared import (
    XN_SLICE,
)


@traced
def moe_narrow_defs(k4ctx):
    """The functions of the per-(token, slot) MoE (S <= 4)."""
    BASE = k4ctx["BASE"]
    CPT = k4ctx["CPT"]
    MID16_WORDS = k4ctx["MID16_WORDS"]
    MID8_LDS = k4ctx["MID8_LDS"]
    MID8_WORDS = k4ctx["MID8_WORDS"]
    MIDSC_LDS = k4ctx["MIDSC_LDS"]
    MIDSC_WORDS = k4ctx["MIDSC_WORDS"]
    UG_ITERS = k4ctx["UG_ITERS"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_LDS = k4ctx["XSC_LDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    bid = k4ctx["bid"]
    down_load = k4ctx["down_load"]
    down_load_shared = k4ctx["down_load_shared"]
    ffn_finish = k4ctx["ffn_finish"]
    g4 = k4ctx["g4"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_i32 = k4ctx["lds_i32"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    opart = k4ctx["opart"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    scores_copy = k4ctx["scores_copy"]
    sh_compute = k4ctx["sh_compute"]
    sh_load = k4ctx["sh_load"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    ug_compute = k4ctx["ug_compute"]
    ug_load = k4ctx["ug_load"]
    ug_task_id = k4ctx["ug_task_id"]
    v4f = k4ctx["v4f"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]

    def stage_moe():
        td = bid - BASE["down"]  # this CTA's down task
        has_dn = (td >= 0) & (td < N_DN)
        td = fx.min(fx.max(td, 0), N_DN - 1)
        has_sh = start("shared") < SH_TASKS
        n_xw = XN8_WORDS // THREADS

        def xn_specs(k):
            return [
                (mr("xn8"), k * XN8_WORDS + tid + i * THREADS, 1) for i in range(n_xw)
            ] + [(mr("xsc"), k * XSC_WORDS + fx.min(tid, XSC_WORDS - 1), 1)]

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
                            mr("xn8"),
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
                    mr("sel"),
                    sk * 2 * MOE_SLOTS + sj,
                    fx.ptr_load(route + (8 * sk + sj)),
                )
                mb_put(
                    mr("sel"),
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
        enter_stage("down")
        """Every token's route-weighted down GEMV of row group ``t`` -> push bf16
        partial rows -> rank-ordered sum -> ar_out."""

        def stage_mids(k):
            """Token k's mid rows (every slot) -> xs, as bf16 pairs."""
            mids = mb_poll(
                [
                    (
                        mr("mid"),
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
            cu = mfma_scaled(
                a=wv,
                sa=sc,
                b=lds_b8(
                    MID8_LDS + col_tok * MID8_WORDS + (TOP_K * INTER + kc * 128) // 4
                ),
                sb=lds_i32(
                    MIDSC_LDS
                    + col_tok * MIDSC_WORDS
                    + (TOP_K * INTER + kc * 128) // 32
                    + g4
                ),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP4,
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
                cu = mfma_scaled(
                    a=wv,
                    sa=sc,
                    b=lds_b8(MID8_LDS + k * MID8_WORDS + (sl * INTER + kc * 128) // 4),
                    sb=lds_i32(
                        MIDSC_LDS + k * MIDSC_WORDS + (sl * INTER + kc * 128) // 32 + g4
                    ),
                    c=fx.Vector.filled(4, 0.0, fx.Float32),
                    a_fmt=FP4,
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
            waves = [grp * 4 + w for w in range(4)]
            tot = row_sum(opart + (k + 1) * WAVES * 64 * 4, rr, 0, waves) + row_sum(
                opart, rr, k, waves
            )
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        ffn_finish(t)

    return {"stage_moe": stage_moe, "moe_down": moe_down}
