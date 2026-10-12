# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2b ``moe``: one Kimi-K3 latent-MoE layer, from the MoE input row x to the
layer's output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 rows (a
spec-verify step). The original path (vLLM ROCm, see MOE_SPEC.md) replicates the
router and the latent down projection on every rank; here each rank computes
its eighth and the rows are all-gathered over the peer buffer.

    early (CTAs 0 .. 130, one task each, x from the input row; weights first):
      router  (7 x 16 rows of this rank's 112 experts): fp32 logits -> every
              rank's RLOG
      latent  (28 x 16 rows of this rank's 448): bf16 -> every rank's LAT
      sgu     (96 x 16 rows of the shared gate | up): bf16 -> SGU
    route   (S, CTAs 131 ..): sigmoid (v_rcp), + bias, top-16, the pre-bias
            sigmoids renormalized (IEEE 1 / sum) -> ROUTE (every rank alike)
    sdown   (448 x 16 rows, K 768, CTAs 131 ..): SiTU of SGU (vLLM's kernel) ->
            bf16 -> GEMV -> bf16 partial; this rank's up columns -> SHP (plain,
            SHRDY flags), the rest -> every rank's FINAL
    ug      (|U| x 12, every CTA): the union U of the step's picks; an expert's
            32 gate + 32 up rows against the MXFP4 latent (aiter's inline quant),
            SiTUv2 (aiter's fp32 epilogue), MXFP4 -> INTER / ISC a pick
    down    (224 x 16 latent rows, every CTA): w2 of every expert of U against
            its picks' INTER, bf16(w acc) folded in each token's top-k order with
            a bf16 rounding an add -> every rank's LATR
    lnorm   (28 x 128 columns): the ranks' LATR in rank order, bf16, RMSNorm
            (the fused AR + RMSNorm form) -> LN (plain, LNRDY flags)
    up      (56 x 16 rows of this rank's 896): LN GEMV + SHP -> bf16 -> every
            rank's FINAL
    reduce  (56 x 128 columns, ``reduce``): FINAL in rank order -> bf16 out

Peer regions are double-buffered by the epoch's parity.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from vllm.models.kimi_k3.amd.ops.mono.common.mx import (
    FP4,
    UNIT_SCALE,
    fp4_tile_load,
    mfma_scaled,
    scale_index,
)
from vllm.models.kimi_k3.amd.ops.mono.common.ops import (
    CM_DEV,
    ballot,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    butterfly,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    lanes_below,
    popcount,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from vllm.models.kimi_k3.amd.ops.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    pair_layout,
)
from vllm.models.kimi_k3.amd.ops.mono.common.ranks import sum_partials
from vllm.models.kimi_k3.amd.ops.mono.common.sync import publish
from vllm.models.kimi_k3.amd.ops.mono.stages import gemv as gemv

REGIONS = (
    "sgu",
    "h",
    "rpart",
    "lpart",
    "route",
    "inter",
    "isc",
    "lmsq",
    "lnrdy",
    "shrdy",
    "rlog",
    "lat",
    "latr",
    "final",
)


TP = 8
HIDDEN = 7168
LAT = 3584
E = 896
TOPK = 16
RI = 384  # routed intermediate a rank
SI = 768  # shared intermediate a rank
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
BETA, LBETA = 4.0, 25.0

R_GROUPS = E // TP // ROWS  # 7 router row groups a rank
R_PARTS = 4  # K quarters: the router is on the path to the route
L_GROUPS = LAT // TP // ROWS  # 28 latent row groups a rank
L_PARTS = 2  # K halves: the latent is on the path to ug
R_TASKS = R_GROUPS * R_PARTS  # 28
L_TASKS = L_GROUPS * L_PARTS  # 56
G_TASKS = 2 * SI // ROWS  # 96
EARLY = R_TASKS + L_TASKS + G_TASKS  # 180
LATE0 = EARLY  # route from here
H_CTA = LATE0 + 8  # the shared experts' SiTU
SPARE0 = H_CTA + 8  # 196: CTAs with no early task run the shared down first
SD_TASKS = HIDDEN // ROWS  # 448
UP_N = HIDDEN // TP  # 896 columns a rank up-projects
UP_TASKS = UP_N // ROWS  # 56
LN_TASKS = LAT // 128  # 28
LN0 = 56  # lnorm's CTAs
RED_TASKS = HIDDEN // 128  # 56
RED0 = 200
UG_GROUPS = RI // 32  # 12
W13_K = LAT  # 3584
W13_STEPS = W13_K // 128  # 28
W2_STEPS = RI // 128  # 3
DOWN_TASKS = LAT // ROWS  # 224
DOWN_BATCH = 5  # a wave's experts whose w2 is in flight together
MAX_U = 8 * TOPK
TL_POINTS = 8  # start, early, sdown, ug, down, lnorm, up, reduce

_STREAM = fx.Stream(None)


def peer_pairs(s):
    """One parity's peer regions, pairs: name -> count."""
    return {
        "rlog": TP * s * (E // TP),
        "lat": s * LAT // 2,
        "latr": TP * s * LAT // 2,
        "final": TP * s * HIDDEN // 2,
    }


def scratch_layout(s: int) -> dict:
    pairs = {
        "sgu": s * SI,
        "h": s * SI // 2,
        "rpart": R_GROUPS * R_PARTS * s * ROWS,
        "lpart": L_GROUPS * L_PARTS * s * ROWS,
        "route": s * 2 * TOPK,
        "inter": s * TOPK * 48,
        "isc": s * TOPK * UG_GROUPS,
        "lmsq": LN_TASKS * s,
        "lnrdy": LN_TASKS,
        "shrdy": UP_TASKS,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["ln"] = ((end + 15) // 16 * 16, s * LAT * 2)
    end = lay["ln"][0] + lay["ln"][1]
    lay["shp"] = ((end + 15) // 16 * 16, s * UP_N * 2)
    return lay


def f32_of(w):
    return w.bitcast(fx.Float32)


def tanh_libm(x):
    """The ``tanhf`` of x to a few ulp: (1 - e) / (1 + e), e = exp(-2|x|), x's sign."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) / (fx.Float32(1.0) + e)
    return (x < fx.Float32(0.0)).select(-t, t)


def situ_vllm(g, u):
    """The ``situ_and_mul`` of vLLM on bf16 g, u:
    ((2 tanh(g/4)) (1 + tanh(g/2))) (25 tanh(u/25))."""
    a = fx.Float32(0.5 * BETA) * tanh_libm(g * fx.Float32(1.0 / BETA))
    a = a * (fx.Float32(1.0) + tanh_libm(g * fx.Float32(0.5)))
    return a * (fx.Float32(LBETA) * tanh_libm(u * fx.Float32(1.0 / LBETA)))


def tanh_aiter(x):
    """The ``act.tanh_f32`` of aiter: (1 - e) rcp(1 + e), e = exp2(|x| (-2 log2 e))."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) * hw_rcp(fx.Float32(1.0) + e)
    return (x > fx.Float32(0.0)).select(t, -t)


def situ_aiter(g, u):
    """The situv2 epilogue of aiter on fp32 accumulators."""
    sg = hw_rcp(fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E)))
    a = (fx.Float32(BETA) * tanh_aiter(g * fx.Float32(1.0 / BETA))) * sg
    return (a * fx.Float32(LBETA)) * tanh_aiter(u * fx.Float32(1.0 / LBETA))


def e8m0_roundup(amax):
    """The ``_e8m0_roundup(amax, 6)`` of aiter: ceil code of amax * f32(1/6),
    at most 254."""
    wi = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    b = fx.Int32(fx.Uint32(wi + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(b, fx.Int32(254))


def pk_fp4_f32(old, a, b, scale, j):
    """Two f32 / scale -> E2M1 nibbles into byte j of ``old`` (RNE, saturating)."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_fp4_bf16(old, word, scale, j):
    """A packed bf16 pair word / scale -> E2M1 nibbles into byte j of ``old``."""
    v = fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(v),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


# ---------------------------------------------------------------- early
@traced
def early_out(c, kind, i, q, red):
    """Task (i, q) of ``kind`` (0 router, 1 latent, 2 sgu): its 16 rows. A K-split
    kind's parts q > 0 publish fp32 partials; part 0 adds them in part order and
    publishes the rows (router: fp32 to every rank's RLOG, latent: bf16 to every
    rank's LAT); sgu bf16 to SGU."""
    s, tid = c["S"], c["tid"]
    if const_expr(kind == 2):
        if tid < (ROWS // 2) * s:
            rp = tid % (ROWS // 2)
            t = tid // (ROWS // 2)
            v0 = bf16_round(row_sum(red, 2 * rp, t))
            v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
            c["put_bf"](c["sgu"], t * 2 * SI + i * ROWS + 2 * rp, [v0, v1])
    else:
        parts = R_PARTS if kind == 0 else L_PARTS
        region = c["rpart"] if kind == 0 else c["lpart"]
        if tid < ROWS * s:
            r = tid % ROWS
            t = tid // ROWS
            v = row_sum(red, r, t)
            at = lambda qq: ((i * parts + qq) * s + t) * ROWS + r  # noqa: E731
            if q > 0:
                c["put"](region, at(q), v)
            if q == 0:
                got = c["poll"]([(region, at(qq), 1) for qq in range(1, parts)])
                for qq in range_constexpr(parts - 1):
                    v = v + f32_of(got[qq][0])
                fx.ptr_store(v, c["av"] + tid)
        gpu.barrier()
        if const_expr(kind == 0):
            if (q == 0) & (tid < ROWS * s):
                r = tid % ROWS
                t = tid // ROWS
                v = fx.ptr_load(c["av"] + tid)
                for p in range_constexpr(TP):
                    c["put"](
                        c["peer"]["rlog"][p],
                        (c["rank"] * s + t) * (E // TP) + i * ROWS + r,
                        v,
                    )
        else:
            if (q == 0) & (tid < (ROWS // 2) * s):
                rp = tid % (ROWS // 2)
                t = tid // (ROWS // 2)
                v0 = bf16_round(fx.ptr_load(c["av"] + (t * ROWS + 2 * rp)))
                v1 = bf16_round(fx.ptr_load(c["av"] + (t * ROWS + 2 * rp + 1)))
                col = c["rank"] * (LAT // TP) + i * ROWS + 2 * rp
                for p in range_constexpr(TP):
                    c["put_bf"](c["peer"]["lat"][p], t * LAT + col, [v0, v1])
    gpu.barrier()


def early_shape(kind):
    """(K a part, parts) of an early kind: router a quarter, latent a half, sgu all."""
    return {
        0: (HIDDEN // R_PARTS, R_PARTS),
        1: (HIDDEN // L_PARTS, L_PARTS),
        2: (HIDDEN, 1),
    }[kind]


def early_loads(c, kind, i, q):
    """Early task (i, q) of ``kind``'s weights for this wave: issued as early as
    the CTA can, then ``early_compute`` takes them."""
    a = c["args"]
    klen, _ = early_shape(kind)
    if kind == 0:
        w, row0 = a["w_gate"], c["rank"] * (E // TP) + i * ROWS
    elif kind == 1:
        w, row0 = a["w_ld"], c["rank"] * (LAT // TP) + i * ROWS
    else:
        w, row0 = a["w_sgu"], i * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, klen, ldk=HIDDEN, k0=q * klen)


@traced
def early_compute(c, kind, i, q, ops):
    """Early task (i, q): x's K slice (the input row, or -- ``xflags`` -- the
    row this launch's AttnRes stage publishes) against ``ops`` -> ``early_out``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    klen, _ = early_shape(kind)
    xrow = klen + gemv.LDS_PAD
    if const_expr(c.get("xflags") is not None):
        if tid < c["xflags"]:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
        gemv.rows_to_lds(
            tid, c["xbuf"], klen, s, c["xl"], xrow, CM_DEV, ldk=HIDDEN, k0=q * klen
        )
    else:
        gemv.rows_to_lds(
            tid, c["args"]["x"], klen, s, c["xl"], xrow, ldk=HIDDEN, k0=q * klen
        )
    acc = gemv.mfmas(lane, wave, c["xl"], xrow, klen, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    early_out(c, kind, i, q, red)


@traced
def stage_early(c, bid):
    """K2b alone: CTA bid's early task (router parts, latent parts, sgu)."""
    lb = bid - R_TASKS
    gb = bid - R_TASKS - L_TASKS
    if bid < R_TASKS:
        early_compute(
            c,
            0,
            bid // R_PARTS,
            bid % R_PARTS,
            early_loads(c, 0, bid // R_PARTS, bid % R_PARTS),
        )
    if (bid >= R_TASKS) & (bid < R_TASKS + L_TASKS):
        early_compute(
            c,
            1,
            lb // L_PARTS,
            lb % L_PARTS,
            early_loads(c, 1, lb // L_PARTS, lb % L_PARTS),
        )
    if bid >= R_TASKS + L_TASKS:
        early_compute(c, 2, gb, 0, early_loads(c, 2, gb, 0))


# ---------------------------------------------------------------- route
def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, t):
    """Token t (wave 0): 896 logits, 14 a lane (expert lane + 64 i) -> sigmoid
    (v_rcp) + bias -> the top-16 by a radix select (the 16th largest key by a
    32-step bit search of ballot counts; equal keys: the lower experts) -> the
    16 ranked by score, descending (aiter's slot order) -> the pre-bias
    sigmoids renormalized in that order (IEEE 1 / sum)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    if wave == 0:
        s = c["S"]
        per = E // 64
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            src = e // (E // TP)
            specs.append(
                (c["peer"]["rlog_own"], (src * s + t) * (E // TP) + e % (E // TP), 1)
            )
        got = c["poll"](specs)
        keys, sigs = [], []
        for i in range_constexpr(per):
            e = lane + 64 * i
            lg = f32_of(got[i][0])
            sg = hw_rcp(fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bo_ld_f32(a["bias"], e)
            # a NaN logit (a warmup's garbage row) ranks last
            keys.append(_okey((sc == sc).select(sc, fx.Float32(-3.0e38))))
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))

        def count_ge(th):
            n = fx.Int32(0)
            for i in range_constexpr(per):
                n = n + popcount(ballot(keys[i] >= th))
            return n

        th = fx.Uint32(0)
        for bit in range_constexpr(31, -1, -1):
            cand = th | fx.Uint32(1 << bit)
            th = (count_ge(cand) >= TOPK).select(cand, th)
        # selected: key > th, then the lowest experts with key == th
        n_gt = fx.Int32(0)
        for i in range_constexpr(per):
            n_gt = n_gt + popcount(ballot(keys[i] > th))
        need = fx.Int32(TOPK) - n_gt
        eq_before = fx.Int32(0)
        sel_before = fx.Int32(0)
        for i in range_constexpr(per):
            eqb = ballot(keys[i] == th)
            eq_rank = eq_before + lanes_below(eqb, lane)
            sel = (keys[i] > th) | ((keys[i] == th) & (eq_rank < need))
            selb = ballot(sel)
            pos = sel_before + lanes_below(selb, lane)
            if sel:
                fx.ptr_store(keys[i].bitcast(fx.Float32), red + pos * 3)
                fx.ptr_store(
                    fx.Int32(lane + 64 * i).bitcast(fx.Float32), red + (pos * 3 + 1)
                )
                fx.ptr_store(sigs[i], red + (pos * 3 + 2))
            eq_before = eq_before + popcount(eqb)
            sel_before = sel_before + popcount(selb)
        # rank the 16 by key descending, the lower expert first on a tie
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
        # the sigmoids in rank order, summed serially (aiter's renorm)
        if lane < TOPK:
            fx.ptr_store(ms, red + (64 + rank))
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (64 + j))
        inv = fx.Float32(1.0) / total
        if lane < TOPK:
            c["put"](
                c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1))
            )
            c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ms * inv)
    gpu.barrier()


def bo_ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U: slot_of[e], expert_of[slot],
    kof[slot][t] (the token's top-k index, -1 if not picked); returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    e0 = fx.min(2 * tid, E - 1)
    f0 = (2 * tid < E).select(fx.ptr_load(rt["flag"] + e0), fx.Int32(0))
    f1 = (2 * tid + 1 < E).select(
        fx.ptr_load(rt["flag"] + fx.min(2 * tid + 1, E - 1)), fx.Int32(0)
    )
    off, tot = block_excl_scan(f0 + f1, lane, wave, c["scan"])
    gpu.barrier()
    if f0 == 1:
        fx.ptr_store(off, rt["flag"] + 2 * tid)
        fx.ptr_store(fx.Int32(2 * tid), rt["expert"] + off)
    if f1 == 1:
        fx.ptr_store(off + f0, rt["flag"] + 2 * tid + 1)
        fx.ptr_store(fx.Int32(2 * tid + 1), rt["expert"] + off + f0)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
    gpu.barrier()
    return uniform(tot)


@traced
def load_latq(c):
    """LAT (every rank's 448 columns) -> MXFP4 as aiter's inline quant: a
    thread a (token, 32-block): amax of |bf16| bits, the ceil code of amax / 6,
    v_cvt_scalef32_pk_fp4_bf16 -> latq words, latsc codes."""
    s, tid = c["S"], c["tid"]
    nblk = s * (LAT // 32)
    for i in range_constexpr((nblk + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nblk - 1)
        t = b // (LAT // 32)
        kb = b % (LAT // 32)
        base = (t * LAT + kb * 32) // 2
        got = c["poll"]([(c["peer"]["lat_own"], base + j, 1) for j in range(16)])
        ws = [got[j][0] for j in range(16)]
        am = fx.Int32(0)
        for w in ws:
            am = fx.max(am, w & 0x7FFF)
            am = fx.max(am, (w >> 16) & 0x7FFF)
        amax = (am << 16).bitcast(fx.Float32)
        code = e8m0_roundup(amax)
        scale = (code << 23).bitcast(fx.Float32)
        if tid + THREADS * i < nblk:
            for q in range_constexpr(4):
                pk = fx.Int32(0)
                for j in range_constexpr(4):
                    pk = pk_fp4_bf16(pk, ws[4 * q + j], scale, j)
                fx.ptr_store(pk, c["latq"] + (t * (LAT // 8) + kb * 4 + q))
            fx.ptr_store(code, c["latsc"] + (t * (LAT // 32) + kb))
    gpu.barrier()


# ---------------------------------------------------------------- ug
def w13_scale_word(e, rg, st, lane):
    """The dword of aiter's gate_up=False ``shuffle_scale`` holding (row rg 16 +
    lane % 16, K block 4 st + lane / 16) for st's pair and rg's pair: byte
    (st % 2) 2 + rg % 2."""
    n1 = rg // 2
    k1 = st // 2
    return (
        e * (2 * RI * (W13_K // 32) // 4)
        + ((n1 * (W13_STEPS // 2) + k1) * 4 + lane // 16) * 16
        + lane % 16
    )


def ug_loads(c, task):
    """Task ``task``'s w13 operands for this wave (its row group, its K half)
    and their scale words: issued a task ahead (``stage_ug`` takes them)."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    rgl = wave % 4  # 0, 1 gate; 2, 3 up
    half = wave // 4
    rg = (rgl // 2) * (RI // ROWS) + 2 * g + rgl % 2
    r_w = rsrc(a["w13"])
    base = e * (2 * RI * W13_K // 8)
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = [fp4_tile_load(r_w, base, rg, W13_K, st, lane * 4) for st in steps]
    r_s = rsrc(a["w13s"])
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, st, lane), vec_width=1, dtype=T.i32
            )
        )
        for st in steps[::2]
    ]
    return av + sw


@traced
def stage_ug(c, task, ops):
    """Slot task / 12, 32-column group task % 12: gate rows 32 g .. and up rows
    384 + 32 g .. of the slot's expert (``ops``: ``ug_loads``) against the MXFP4
    latent; SiTUv2; MXFP4 a 32-column group -> INTER / ISC of each picking
    token."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    rgl = wave % 4
    half = wave // 4
    rg = (rgl // 2) * (RI // ROWS) + 2 * g + rgl % 2
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = ops[: len(steps)]
    sw = ops[len(steps) :]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(steps)):
        st = steps[i]
        bw = fx.Vector(
            fx.ptr_load(
                c["latq"] + (t * (LAT // 8) + st * 16 + kb * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = fx.ptr_load(c["latsc"] + (t * (LAT // 32) + st * 4 + kb))
        # the code's byte (st % 2) 2 + rg % 2 of the word; rg % 2 is the wave's
        sa = (sw[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP4, 0, 0)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (token, column): g / u summed over the K halves, SiTUv2
    tt = tid // 32
    col = tid % 32
    live = tt < s
    tt = fx.min(tt, s - 1)
    first = col < 16  # gate / up row group 0 (waves 0, 2) or 1 (1, 3); +4: K's 2nd half
    r = col % 16
    gv = first.select(
        row_sum(red, r, tt, waves=(0, 4)), row_sum(red, r, tt, waves=(1, 5))
    )
    uv = first.select(
        row_sum(red, r, tt, waves=(2, 6)), row_sum(red, r, tt, waves=(3, 7))
    )
    v = situ_aiter(gv, uv)
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = e8m0_roundup(amax)
    gpu.barrier()
    fx.ptr_store(v, c["av"] + tid)
    gpu.barrier()
    k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
    if live & (col < 4) & (k >= 0):
        scale = (code << 23).bitcast(fx.Float32)
        pk = fx.Int32(0)
        for j in range_constexpr(4):
            x0 = fx.ptr_load(c["av"] + (tt * 32 + col * 8 + 2 * j))
            x1 = fx.ptr_load(c["av"] + (tt * 32 + col * 8 + 2 * j + 1))
            pk = pk_fp4_f32(pk, x0, x1, scale, j)
        pick = tt * TOPK + k
        c["put_words"](c["inter"], pick * 48 + g * 4 + col, [pk])
        if col == 0:
            c["put"](c["isc"], pick * UG_GROUPS + g, code)
    gpu.barrier()


# ---------------------------------------------------------------- queues
@traced
def load_inter(c):
    """Every pick's INTER words and ISC codes -> LDS."""
    s, tid = c["S"], c["tid"]
    nw = s * TOPK * 48
    per = (nw + THREADS - 1) // THREADS
    got = c["poll"](
        [(c["inter"], fx.min(tid + THREADS * i, nw - 1), 1) for i in range(per)]
    )
    for i in range_constexpr(per):
        if tid + THREADS * i < nw:
            fx.ptr_store(got[i][0], c["interq"] + (tid + THREADS * i))
    nc = s * TOPK * UG_GROUPS
    perc = (nc + THREADS - 1) // THREADS
    gc = c["poll"](
        [(c["isc"], fx.min(tid + THREADS * i, nc - 1), 1) for i in range(perc)]
    )
    for i in range_constexpr(perc):
        if tid + THREADS * i < nc:
            fx.ptr_store(gc[i][0], c["intsc"] + (tid + THREADS * i))
    gpu.barrier()


def w2_scale(c, e, rgd, st, lane):
    """The e8m0 code of w2 (row e 3584 + rgd 16 + lane % 16, K block 4 st +
    lane / 16) in ``e8m0_shuffle``'s layout (16 codes a row)."""
    row = e * LAT + rgd * ROWS + lane % 16
    idx = scale_index(row, st * 4 + lane // 16, 16)
    w = fx.Int32(
        bo.buffer_load(rsrc(c["args"]["w2s"]), idx // 4, vec_width=1, dtype=T.i32)
    )
    return (w >> ((idx % 4) * 8)) & 0xFF


def poll_inter(c, pick, kb):
    """A lane's INTER words (3 steps x 4) and ISC codes of ``pick``, polled at
    its use: a slot whose ug tasks are done is multiplied while later slots'
    still run (no wait for all of U)."""
    return c["poll"](
        [
            (c["inter"], pick * 48 + st * 16 + kb * 4 + d, 1)
            for st in range(W2_STEPS)
            for d in range(4)
        ]
        + [(c["isc"], pick * UG_GROUPS + st * 4 + kb, 1) for st in range(W2_STEPS)]
    )


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale codes of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's): flat, ``stage_down`` unpacks."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"])
    rt = c["rt"]
    slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(DOWN_BATCH)]
    es = [uniform(fx.ptr_load(rt["expert"] + sl)) for sl in slots]
    av = [
        fp4_tile_load(r_w, es[j] * (LAT * RI // 8), rgd, RI, st, lane * 4)
        for j in range(DOWN_BATCH)
        for st in range(W2_STEPS)
    ]
    sa = [
        w2_scale(c, es[j], rgd, st, lane)
        for j in range(DOWN_BATCH)
        for st in range(W2_STEPS)
    ]
    return av + sa


@traced
def stage_down(c, rgd, nu, ops0):
    """Latent rows 16 rgd ..: every slot's w2 against its picks (wave w the slots
    w, w + 8, ...; a batch of DOWN_BATCH in flight, the next issued under this
    one's MFMAs; ``ops0`` the first, issued before INTER was waited for) ->
    bf16(w acc) a (token, k) in LDS -> folded in top-k order with a bf16
    rounding an add -> every rank's LATR."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    nbs = DOWN_BATCH * W2_STEPS
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DOWN_BATCH
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(DOWN_BATCH)]
        for j in range_constexpr(DOWN_BATCH):
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (slots[j] * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * TOPK + fx.max(k, 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            # this slot's picks' INTER, polled here (not all of U first): a slot
            # whose ug tasks are done is multiplied while later ones still run
            for st in range_constexpr(W2_STEPS):
                bw = fx.Vector(
                    fx.ptr_load(
                        c["interq"] + (pick * 48 + st * 16 + kb * 4),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                scode = fx.ptr_load(c["intsc"] + (pick * UG_GROUPS + st * 4 + kb))
                bw = fx.Vector.from_elements(
                    [picked.select(bw[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
                sb = picked.select(scode, fx.Int32(UNIT_SCALE))
                acc = mfma_scaled(
                    ops[j * W2_STEPS + st],
                    bw,
                    acc,
                    ops[nbs + j * W2_STEPS + st],
                    sb,
                    FP4,
                    FP4,
                    0,
                    0,
                )
            # lane: rows 4 (lane / 16) + i of token lane % 16
            if live & picked:
                wt = fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + k)).bitcast(
                    fx.Float32
                )
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        bf16_round(acc[i] * wt),
                        c["contrib"] + ((t * TOPK + k) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        y0 = fx.Float32(0.0)
        y1 = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            y0 = bf16_round(
                y0 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp))
            )
            y1 = bf16_round(
                y1 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp + 1))
            )
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["latr"][p],
                (c["rank"] * s + tt) * LAT + rgd * ROWS + 2 * rp,
                [y0, y1],
            )
    gpu.barrier()


# ---------------------------------------------------------------- shared down
@traced
def stage_h(c, t):
    """Token t's shared h = SiTU(g, u) (vLLM's situ_and_mul, bf16) of SGU, once
    -> H; a thread a column pair, its polls in one batch."""
    tid = c["tid"]
    n = SI // 2
    if tid < n:
        got = c["poll"](
            [
                (c["sgu"], (t * 2 * SI) // 2 + tid, 1),
                (c["sgu"], (t * 2 * SI + SI) // 2 + tid, 1),
            ]
        )
        g0, g1 = bf_lo(got[0][0]), bf_hi(got[0][0])
        u0, u1 = bf_lo(got[1][0]), bf_hi(got[1][0])
        c["put_bf"](c["h"], t * SI + 2 * tid, [situ_vllm(g0, u0), situ_vllm(g1, u1)])
    gpu.barrier()


@traced
def load_h(c):
    """H -> LDS rows (for the shared down GEMV), once a CTA."""
    s, tid = c["S"], c["tid"]
    hrow = SI + gemv.LDS_PAD
    n = s * SI // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"]([(c["h"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)])
    for i in range_constexpr(per):
        pidx = tid + THREADS * i
        t = fx.min(pidx, n - 1) // (SI // 2)
        cp = fx.min(pidx, n - 1) % (SI // 2)
        if pidx < n:
            fx.ptr_store(got[i][0], c["xl"] + (t * hrow + 2 * cp) // 2)
    gpu.barrier()


@traced
def stage_sdown(c, j, ops):
    """Shared down rows 16 j .. (K 768) against h in LDS (``load_h``): bf16
    partial -> SHP (this rank's up columns) or every rank's FINAL."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    hrow = SI + gemv.LDS_PAD
    acc = gemv.mfmas(lane, wave, c["xl"], hrow, SI, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    lo = c["rank"] * UP_TASKS
    mine = (j >= lo) & (j < lo + UP_TASKS)
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        if mine:
            w = (
                fx.Vector.from_elements([v0, v1], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(
                w,
                rsrc(c["shp"]),
                (tt * UP_N + (j - lo) * ROWS + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        else:
            for p in range_constexpr(TP):
                c["put_bf"](
                    c["peer"]["final"][p],
                    (c["rank"] * s + tt) * HIDDEN + j * ROWS + 2 * rp,
                    [v0, v1],
                )
    publish(
        c["put"],
        c["shrdy"],
        fx.max(fx.min(j - lo, UP_TASKS - 1), 0),
        fx.Int32(1),
        (tid == 0) & mine,
    )
    gpu.barrier()


@traced
def run_sdown(c, bid, w_sd):
    """This rank's 56 shared-down tasks whose columns the up stage adds to (k ->
    row group (k + 56 rank) % 448), on the CTAs past the down tasks: up waits
    on them."""
    load_h(c)
    lo = c["rank"] * UP_TASKS
    if bid >= DOWN_TASKS:
        for k in range(bid - DOWN_TASKS, UP_TASKS, BLOCKS - DOWN_TASKS):
            j = (k + lo) % SD_TASKS
            stage_sdown(c, j, gemv.loads(c["lane"], c["wave"], w_sd, j * ROWS, SI))


@traced
def run_sdown_rest(c, bid, w_sd):
    """The other 392 shared-down tasks (only the final reduce waits on them), on
    the CTAs past lnorm / up's (84 ..), after their down: off the latent-norm ->
    up critical path."""
    r0 = LN0 + LN_TASKS  # 84: past the lnorm / up CTAs
    if bid >= r0:
        load_h(c)
        lo = c["rank"] * UP_TASKS
        for k in range(UP_TASKS + bid - r0, SD_TASKS, BLOCKS - r0):
            j = (k + lo) % SD_TASKS
            stage_sdown(c, j, gemv.loads(c["lane"], c["wave"], w_sd, j * ROWS, SI))


# ---------------------------------------------------------------- lnorm / up / reduce
@traced
def stage_lnorm(c, j):
    """Columns 128 j ..: the ranks' LATR (rank order, fp32 -> bf16), every
    task's sum of squares -> rstd -> bf16(acc rstd w) -> LN, LNRDY."""
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    cp = tid % 64
    col = j * 128 + 2 * cp
    lo, hi = sum_partials(
        c["poll"],
        c["peer"]["latr_own"],
        lambda src: ((src * s + t) * LAT + col) // 2,
        TP,
    )
    lo, hi = bf16_round(lo), bf16_round(hi)
    sq = wave_sum(lo * lo + hi * hi)
    if mine & (lane == 0):
        c["put"](c["lmsq"], j * s + t, sq)
    gpu.barrier()
    if mine & (lane < LN_TASKS):
        v = c["poll"]([(c["lmsq"], lane * s + t, 1)])[0][0]
        fx.ptr_store(f32_of(v), red + (t * 32 + lane))
    gpu.barrier()
    tot = fx.ptr_load(red + t * 32)
    for q in range_constexpr(1, LN_TASKS):
        tot = tot + fx.ptr_load(red + (t * 32 + q))
    r = hw_rsq(tot * fx.Float32(1.0 / LAT) + fx.Float32(c["ln_eps"]))
    w0 = fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(a["ln_w"]), col, vec_width=1, dtype=T.bf16))
    )
    w1 = fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(a["ln_w"]), col + 1, vec_width=1, dtype=T.bf16))
    )
    y = (
        fx.Vector.from_elements([lo * r * w0, hi * r * w1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    if mine:
        bo.buffer_store(y, rsrc(c["ln"]), (t * LAT + col) // 2, cache_modifier=CM_DEV)
    publish(c["put"], c["lnrdy"], j, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def stage_up(c, j):
    """This rank's up rows 16 j .. of its shard (hidden rows 896 rank ..): LN GEMV
    + SHP -> bf16 -> FINAL."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    row0 = c["rank"] * UP_N + j * ROWS  # the hidden column; w_up is the shard
    ops = gemv.loads(lane, wave, a["w_up"], j * ROWS, LAT)
    if tid < LN_TASKS:
        c["poll"]([(c["lnrdy"], tid, 1)])
    if tid == 0:
        c["poll"]([(c["shrdy"], j, 1)])
    gpu.barrier()
    lrow = LAT + gemv.LDS_PAD
    gemv.dev_rows_to_lds(tid, c["ln"], LAT, s, c["xl"], lrow)
    acc = gemv.mfmas(lane, wave, c["xl"], lrow, LAT, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        w = fx.Int32(
            bo.buffer_load(
                rsrc(c["shp"]),
                (tt * UP_N + j * ROWS + 2 * rp) // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        v0 = bf16_round(bf_lo(w) + row_sum(red, 2 * rp, tt))
        v1 = bf16_round(bf_hi(w) + row_sum(red, 2 * rp + 1, tt))
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["final"][p],
                (c["rank"] * s + tt) * HIDDEN + row0 + 2 * rp,
                [v0, v1],
            )
    gpu.barrier()


@traced
def stage_reduce(c, j):
    """Columns 128 j ..: FINAL in rank order -> bf16 out."""
    s, tid = c["S"], c["tid"]
    if tid < s * 64:
        t = tid // 64
        cp = tid % 64
        col = j * 128 + 2 * cp
        lo, hi = sum_partials(
            c["poll"],
            c["peer"]["final_own"],
            lambda src: ((src * s + t) * HIDDEN + col) // 2,
            TP,
        )
        w = (
            fx.Vector.from_elements([lo, hi], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(w, rsrc(c["args"]["out"]), (t * HIDDEN + col) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel
