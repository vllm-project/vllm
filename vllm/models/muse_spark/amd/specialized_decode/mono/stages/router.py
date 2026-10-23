# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 2b (README, "K2 program order"): the router and the routing.

    router  (CTA R0 + i, i < 32: expert row e = 32 rank + i) the fp32 logit of
            e for every row t of ffn_in (``xrow``, announced slice by slice by
            the seam's XRDY flags): a VALU dot over the fp32 router weight
            (16 K a thread, fma in K order; no bf16 split of the weight), the
            waves' partials added in wave order -> every rank's RLOG (the
            logits all-gather: rank src's 32 experts at (src MT + t) 32 + i).
    route   (CTA t < S, wave 0, after its seam slice) row t's 256 logits ->
            sigmoid with an IEEE divide (``tl.sigmoid``-exact) + the
            selection bias -> the top-8 by a radix select (the 8th largest key
            by a 32-step ballot bit search; equal keys: the lower experts) ->
            the 8 ranked by key, descending, the lower expert first on a tie
            (``tl.argmax``'s rule) -> the unbiased sigmoids summed serially in
            that order, + ROUTE_EPS, ROUTE_SCALE / sum (IEEE) -> ROUTE.

The routing is ``fused_musespark1x_routing``'s (common/ops/fused_routing.py)
bit for bit on finite logits: the same sigmoid, the same keys, the same pick
order, the same serial sum and divide. A pad row gets ids -1 and weights 0 (by
``select``), as the stock kernel does with ``is_padding``. A NaN logit (a
warm-up's garbage row) ranks last and adds 0 to the sum; the stock argmax's
result there depends on its reduction tree.

Adapted from K3's ``stage_route`` (pr60683 ``mono/stages/moe.py``) with E = 256,
TOPK = 8, the IEEE sigmoid, the epsilon and the pad rows.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf2_f32,
    ieee_sigmoid,
    lanes_below,
    ld_pad,
    popcount,
    rsrc,
    traced,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    EXPERTS,
    EXPERTS_RANK,
    HIDDEN,
    MAX_TOKENS,
    NA,
    ROUTE_EPS,
    ROUTE_SCALE,
    THREADS,
    TOPK,
    TP,
    WAVES,
    XROW,
    rlog_pair,
    route_id_pair,
    route_weight_pair,
)

K_THREAD = HIDDEN // THREADS  # 16 K a thread of a router row
PER = EXPERTS // 64  # 4 logits a lane in the routing (expert lane + 64 u)
NAN_KEY = -3.0e38  # a NaN score's key: below every finite one
assert K_THREAD == 16 and PER * 64 == EXPERTS and TOPK <= 64


def ld_f32(ptr, i):
    """Element ``i`` of the fp32 buffer at ``ptr`` (a plain load)."""
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- router


def router_loads(c, i):
    """CTA R0 + i's router row (expert ``32 rank + i``): thread tid's 16 fp32
    weights of K ``16 tid ..`` as 4 dwordx4, nontemporal (the row is read once
    a step). Issued before ``stage_router``'s XRDY waits."""
    e = c["rank"] * EXPERTS_RANK + i
    r_w = rsrc(c["args"]["w_router"])
    base = e * HIDDEN + K_THREAD * c["tid"]
    return [
        fx.Vector(
            bo.buffer_load(
                r_w, base + 4 * q, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
            )
        )
        for q in range(K_THREAD // 4)
    ]


@traced
def stage_router(c, i, ops):
    """Expert ``32 rank + i``'s fp32 logits of rows 0 .. S-1 -> every rank's
    RLOG. Thread tid: 16 products of K ``16 tid ..`` (bf16 x -> fp32, exact) in
    one fma chain in K order; ``wave_sum``; the 8 waves' sums in wave order."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    # every seam slice of xrow: all of K
    if tid < NA:
        c["poll"]([(c["scr"]["xrdy"], tid, 1)])
    gpu.barrier()
    r_x = rsrc(c["scr"][XROW])
    wv = [ops[q][j].bitcast(fx.Float32) for q in range(K_THREAD // 4) for j in range(4)]
    # all rows' x first: 2 dwordx4 a row (16 bf16), device scope (the seam's
    # plain stores, published by XRDY)
    xs = []
    for t in range_constexpr(s):
        base = t * (HIDDEN // 2) + (K_THREAD // 2) * tid
        xs.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_x,
                        base + 4 * h,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                for h in range(2)
            ]
        )
    for t in range_constexpr(s):
        acc = fx.Float32(0.0)
        for h in range_constexpr(2):
            for j in range_constexpr(4):
                lo, hi = bf2_f32(xs[t][h][j])
                k = 8 * h + 2 * j
                acc = fx.Float32(fmath.fma(lo, wv[k], acc))
                acc = fx.Float32(fmath.fma(hi, wv[k + 1], acc))
        tot = wave_sum(acc)
        if lane == 0:
            fx.ptr_store(tot, red + (wave * MAX_TOKENS + t))
    gpu.barrier()
    if tid < s:
        logit = fx.ptr_load(red + tid)
        for w in range_constexpr(1, WAVES):
            logit = logit + fx.ptr_load(red + (w * MAX_TOKENS + tid))
        at = rlog_pair(c["rank"], tid, i)
        for p in range_constexpr(TP):
            c["put"](c["peer"]["rlog"][p], at, logit)
    gpu.barrier()


# ---------------------------------------------------------------- route


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, t):
    """Row t's top-8 (wave 0; the other waves only take the barrier): RLOG
    (every rank's 32 logits of row t) -> ``ieee_sigmoid`` + bias -> the radix
    select of the 8 largest keys -> ranked by key, lower expert first -> the
    renormalized unbiased sigmoids -> ROUTE ``t 16 + k`` (id) and
    ``t 16 + 8 + k`` (weight bits). A pad row: ids -1, weights 0."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    if wave == 0:
        # the plain loads go out before the wait
        pad_t = ld_pad(a["pad"], t)
        bias = [ld_f32(a["bias"], lane + 64 * u) for u in range(PER)]
        # expert e = lane + 64 u is rank e // 32's router row e % 32
        specs = [
            (
                c["peer"]["rlog_own"],
                rlog_pair(
                    lane // EXPERTS_RANK + (64 // EXPERTS_RANK) * u, t, lane % 32
                ),
                1,
            )
            for u in range(PER)
        ]
        got = c["poll"](specs)
        keys, sigs = [], []
        for u in range_constexpr(PER):
            sg = ieee_sigmoid(got[u][0].bitcast(fx.Float32))
            sc = sg + bias[u]
            # sg is never -0, so neither is sc: the integer key orders it as
            # the float compare does. A NaN (a warm-up's garbage row) ranks
            # last and weighs nothing.
            keys.append(_okey((sc == sc).select(sc, fx.Float32(NAN_KEY))))
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))

        def count_ge(th):
            n = fx.Int32(0)
            for u in range_constexpr(PER):
                n = n + popcount(ballot(keys[u] >= th))
            return n

        th = fx.Uint32(0)
        for bit in range_constexpr(31, -1, -1):
            cand = th | fx.Uint32(1 << bit)
            th = (count_ge(cand) >= TOPK).select(cand, th)
        # selected: key > th, then the lowest experts with key == th
        n_gt = fx.Int32(0)
        for u in range_constexpr(PER):
            n_gt = n_gt + popcount(ballot(keys[u] > th))
        need = fx.Int32(TOPK) - n_gt
        eq_before = fx.Int32(0)
        sel_before = fx.Int32(0)
        for u in range_constexpr(PER):
            eqb = ballot(keys[u] == th)
            eq_rank = eq_before + lanes_below(eqb, lane)
            sel = (keys[u] > th) | ((keys[u] == th) & (eq_rank < need))
            selb = ballot(sel)
            pos = sel_before + lanes_below(selb, lane)
            if sel:
                fx.ptr_store(keys[u].bitcast(fx.Float32), red + pos * 3)
                fx.ptr_store(
                    fx.Int32(lane + 64 * u).bitcast(fx.Float32), red + (pos * 3 + 1)
                )
                fx.ptr_store(sigs[u], red + (pos * 3 + 2))
            eq_before = eq_before + popcount(eqb)
            sel_before = sel_before + popcount(selb)
        # rank the 8 by key descending, the lower expert first on a tie: the
        # order the stock kernel's 8 argmax steps pick them in
        q = fx.min(lane, TOPK - 1)
        mk = fx.Uint32(fx.ptr_load(red + q * 3).bitcast(fx.Int32))
        me = fx.ptr_load(red + (q * 3 + 1)).bitcast(fx.Int32)
        ms = fx.ptr_load(red + (q * 3 + 2))
        rank = fx.Int32(0)
        for j in range_constexpr(TOPK):
            ok = fx.Uint32(fx.ptr_load(red + j * 3).bitcast(fx.Int32))
            oe = fx.ptr_load(red + (j * 3 + 1)).bitcast(fx.Int32)
            before = (ok > mk) | ((ok == mk) & (oe < me))
            rank = rank + before.select(fx.Int32(1), fx.Int32(0))
        # the sigmoids in pick order, summed serially from 0, + eps, then one
        # IEEE divide (fused_routing.py: score_sum, inv_sum_scaled)
        if lane < TOPK:
            fx.ptr_store(ms, red + (64 + rank))
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (64 + j))
        total = total + fx.Float32(ROUTE_EPS)
        inv = fx.Float32(ROUTE_SCALE) / total
        rid = fx.Int32(pad_t.select(fx.Int32(-1), me))
        rw = fx.Float32(pad_t.select(fx.Float32(0.0), ms * inv))
        if lane < TOPK:
            c["put"](c["scr"]["route"], route_id_pair(t, rank), rid)
            c["put"](c["scr"]["route"], route_weight_pair(t, rank), rw)
    gpu.barrier()
