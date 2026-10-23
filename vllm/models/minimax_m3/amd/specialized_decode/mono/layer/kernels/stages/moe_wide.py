# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the expert-grouped MoE (S > 4)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    N_ROUTED,
    SHARED_EXPERT,
    TOP_K,
)
from ...layout import (
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
)
from ...support.device.mx import (
    FP4,
    fp4_tile_load,
    mfma_scaled,
)
from ...support.device.ops import (
    CM_DEV,
    ballot,
    bf16_round,
    lanes_below,
    popcount,
    row_live,
    row_sum,
    rows_to_lds,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from ...support.device.sync import (
    publish,
)
from ...support.plan.execution import (
    THREADS,
    WAVES,
)
from ...support.plan.trace import (
    enter_stage,
)
from .shared import (
    S2_BYTES,
    S13_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
)


@traced
def moe_wide_defs(k4ctx):
    """The functions of the expert-grouped MoE (S > 4)."""
    BASE = k4ctx["BASE"]
    DN_DEPTH = k4ctx["DN_DEPTH"]
    DN_KC = k4ctx["DN_KC"]
    DN_WITERS = k4ctx["DN_WITERS"]
    G = k4ctx["G"]
    MIDSC_ROW_WORDS = k4ctx["MIDSC_ROW_WORDS"]
    MID_ROWS = k4ctx["MID_ROWS"]
    MID_ROW_WORDS = k4ctx["MID_ROW_WORDS"]
    ROUTE_ITERS = k4ctx["ROUTE_ITERS"]
    UGW_ITERS = k4ctx["UGW_ITERS"]
    UG_CPW = k4ctx["UG_CPW"]
    UG_PAIRS = k4ctx["UG_PAIRS"]
    U_MAX = k4ctx["U_MAX"]
    W_MIDSC = k4ctx["W_MIDSC"]
    W_SIG = k4ctx["W_SIG"]
    W_UPP = k4ctx["W_UPP"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_LDS = k4ctx["XSC_LDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    acquire = k4ctx["acquire"]
    batch_ids = k4ctx["batch_ids"]
    bid = k4ctx["bid"]
    e8m0_load = k4ctx["e8m0_load"]
    ffn_finish = k4ctx["ffn_finish"]
    fpool = k4ctx["fpool"]
    g4 = k4ctx["g4"]
    group_scale_byte = k4ctx["group_scale_byte"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_i32 = k4ctx["lds_i32"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    scores_copy = k4ctx["scores_copy"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    swiglu_limit = k4ctx["swiglu_limit"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    uexp = k4ctx["uexp"]
    umask = k4ctx["umask"]
    urow = k4ctx["urow"]
    uwt = k4ctx["uwt"]
    w13 = k4ctx["w13"]
    w2 = k4ctx["w2"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]

    def slot_of(k, e):
        """Token k's slot routed to expert e (TOP_K: the shared expert)."""
        s = fx.Int32(TOP_K)
        for j in range_constexpr(TOP_K):
            s = (fx.Int32(fx.ptr_load(route + (8 * k + j))) == e).select(fx.Int32(j), s)
        return s

    def build_experts():
        """Uexp / umask[0 .. n_u) := the tokens' distinct routed experts in
        expert order and their token masks, [n_u] := the shared expert (every
        token), uexp[U_MAX + 1] := n_u. The waves' masks must be in (synced)."""
        m = fx.Int32(0)
        if tid < N_ROUTED:
            for w in range_constexpr(WAVES):
                m = m | fx.Int32(fx.ptr_load(xs + (W_UPP + w * N_ROUTED + tid)))
        bal = ballot(m != 0)
        below = lanes_below(bal, lane)
        if tid == 0:
            fx.ptr_store(popcount(bal), umask + (U_MAX + 1))
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
                nu_ = off + popcount(bal)
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
        wts: list = []
        scs: list = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(fp4_tile_load(r_w, 0, rg, HIDDEN, kc, lane * 4))
                if const_expr(scales and i % 2 == 0):
                    # the dword holding this lane's scales of chunks kc, kc + 1
                    # of both 16-row halves of the 32-row block (bytes 2 d + a)
                    scs[gu].append(
                        fx.Int32(
                            bo.buffer_load(
                                r_s,
                                group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS)
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
            cg_ = mfma_scaled(
                wts[0][i], xb, cg_, scs[0][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
            cu_ = mfma_scaled(
                wts[1][i], xb, cu_, scs[1][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, fpool + (W_UPP + (wave * 64 + lane) * 4))
        gpu.barrier()
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        gsum = row_sum(red, r, k)
        usum = row_sum(fpool + W_UPP, r, k)
        gpu.barrier()  # red and the up partials are free for the next task
        return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

    def fp8_lanes4(q):
        """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False))
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
        publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    def down_tables():
        """Urow / uwt := per (distinct expert, B column) the down stage's mid row
        (the zero row where the column's token is not routed there) and route
        weight. Routing only: built while the first up / gate weights land."""
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
        wave_toks = [fx.min(wave + WAVES * j, tokens - 1) for j in range(ROUTE_ITERS)]
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
                    row_live(batch_ids, wave_toks[j]).select(
                        fx.Int32(1) << wave_toks[j], fx.Int32(0)
                    ),
                )
        stamp(2)
        gpu.barrier()
        build_experts()
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
                    (mr("xndone"), fx.min(lane + 64 * c, n_rt - 1), 1)
                    for c in range((n_rt + 63) // 64)
                ]
            )
        gpu.barrier()
        acquire()
        for base_w, n_row, lds_w, lds_row in (
            (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
            (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
        ):
            rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
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
        enter_stage("down")
        """Row group td's down GEMV over every distinct expert (B column l % 16 =
        token l % 16's mid for it, the zero row where the token is not routed
        there), route-weighted per column -> ffn_finish."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        rg_d = td * 2 + wave // 4
        # wave q of the row group takes chunks q and q + 4 of every expert, unit i
        # being expert i // 2 for all 4: which wave adds a chunk, and in what
        # order, follows from the expert alone, not its rank in the step's union,
        # so another row's experts cannot regroup this row's sums
        q = wave % 4

        def unit_load(i):
            """(weights, scale, unit) of this wave's i-th (expert, 128-k chunk)
            unit, zero-sized reads past the last."""
            dkc = q + 4 * (i % 2)
            ok = (dkc < DN_KC) & (fx.Int32(i // 2) <= nu)
            # clamped into the table: a unit past the last reads a real or zero
            # row, never past urow (a garbage row is garbage B, and 0 * NaN)
            du = fx.Int32(min(i // 2, U_MAX - 1))
            dkc = fx.min(dkc, DN_KC - 1)
            de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
            dwv = fp4_tile_load(
                rsrc(
                    w2 + fx.Int64(de) * W2_BYTES,
                    ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                ),
                0,
                rg_d,
                INTER,
                dkc,
                lane * 4,
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
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        for base_w, n_row, lds_w, lds_row in (
            (mb("mid8p"), MID_ROW_WORDS, 0, MID_ROW),
            (mb("midsp"), MIDSC_ROW_WORDS, W_MIDSC, MIDSC_ROW),
        ):
            rows_to_lds(tid, base_w, MID_ROWS, n_row, xs + lds_w, lds_row)
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
            dcu = mfma_scaled(
                a=dwv,
                sa=dsc,
                b=lds_b8(drow * MID_ROW + dkc * 32),
                sb=lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP4,
            )
            # a unit past the last (or a wave's missing second chunk) adds
            # nothing, whatever its MFMA gave
            acc = [acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0)) for e in range(4)]
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
            tot = row_sum(red, rr, k, [grp * 4 + w for w in range(4)])
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        ffn_finish(td)

    return {
        "slot_of": slot_of,
        "build_experts": build_experts,
        "wug_load": wug_load,
        "wug_y": wug_y,
        "fp8_lanes4": fp8_lanes4,
        "wug_publish": wug_publish,
        "down_tables": down_tables,
        "stage_moe_wide": stage_moe_wide,
        "wide_down": wide_down,
    }
