# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 3b's down stage under TEP8 (``K2Build.moe == "ep"``): this rank's
32 experts are whole (w2 K = 4096), so a CTA's 16 pooled rows of a local pick
need the pick's whole a2 row: 32 128-K steps where TP8 has 4.

    local_picks  (every CTA) this rank's live local picks in pick order:
                 ``lpof[pick]`` (its local index, -1 otherwise), ``pickof[lp]``,
                 n_lp; ``res`` zeroed
    load_inter   (a chunk of EP_STAGE_PICKS local picks, ``load_inter_ep``)
                 their whole a2 rows -> LDS ``interq`` / ``intsc``; no other
                 pick is polled
    down         (CTA b: pooled rows 16 b .., ``stage_down_ep``) wave w the
                 K steps 4 w .. 4 w + 3 of every local slot's w2
                 (DOWN_BATCH_EP slots in flight) -> its fp32 ``part``; the 8
                 waves' parts summed in wave order -> ``res[pick]``
    push         every pick's 16 rows, rounded once to bf16, to the column
                 owner's RSX (``b // 32``): a local pick its row, any other
                 exact zeros, so the owner's 8-rank fp32 sum
                 (``reduce.stage_owner``, unchanged) is the producing rank's row

The chunk loop runs once unless more than EP_STAGE_PICKS of the step's picks
are this rank's (each further chunk streams the w2 again). The union
(``experts.load_route(c, ep=True)``) and ug (``experts.run_ug(c, nu,
EP_GEOM)``, a2 into ``inter_ep`` / ``isc_ep``) are TP8's stages at EP's
geometry. Pad rule and clamps as ``experts``: a dead or non-local pick is never
polled and never multiplied in (``select``, never a mask multiply); every
slot / expert / local index is clamped at each address use.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.mono.common.mx import (
    FP4,
    UNIT_SCALE,
    fp4_tile_load,
    mfma_scaled,
    scale_byte,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    block_excl_scan,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import peer_at
from vllm.models.muse_spark.amd.specialized_decode.mono.common.route_lds import (
    id_index,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import Addr
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    DOWN_BATCH_EP,
    EP_GEOM,
    EP_PICK_CODES,
    EP_PICK_WORDS,
    EP_STAGE_PICKS,
    EP_WAVE_STEPS,
    MAX_PICKS,
    MOE_HIDDEN,
    RED_TASKS,
    ROWS,
    THREADS,
    TOPK,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages.experts import (
    _live,
    _local,
    _pick_id,
    rsx_elem,
    slot_expert,
)

P = EP_STAGE_PICKS
CODE_WAVES = EP_PICK_CODES // 64  # 2: waves a pick's codes span
# This rank's w2 and its e8m0 codes, bytes; past them a load is out of range.
W2_BYTES = EP_GEOM.experts * EP_GEOM.w2_expert_dwords * 4  # 256 MiB
W2S_BYTES = EP_GEOM.experts * MOE_HIDDEN * EP_GEOM.ug_groups  # 16 MiB
OOB_DWORD = 1 << 28  # a w2 dword index past W2_BYTES (its byte offset < 2^32)
OOB_ROW = 1 << 20  # a w2 scale row past W2S_BYTES (scale_index < 2^32)
assert OOB_DWORD * 4 > W2_BYTES + (1 << 26) and (OOB_DWORD + (1 << 26)) * 4 < 1 << 32
assert OOB_ROW * EP_GEOM.ug_groups > W2S_BYTES
assert EP_PICK_CODES % 64 == 0 and THREADS % EP_PICK_CODES == 0

_DOWN_LDS: dict = {}


def down_lds_struct(s: int):
    """The EP down stage's LDS: a chunk's whole a2 rows (``interq``, 512 words a
    local pick; ``intsc``, 128 codes), every wave's fp32 partial rows of each
    staged pick (``part``), every pick's summed rows (``res``) and the local
    pick tables (``lpof``, ``pickof``)."""
    if s not in _DOWN_LDS:

        @fx.struct
        class DownLdsEp:
            interq: fx.Array[fx.Int32, P * EP_PICK_WORDS, 16]
            intsc: fx.Array[fx.Int32, P * EP_PICK_CODES, 16]
            part: fx.Array[fx.Float32, WAVES * P * ROWS, 16]
            res: fx.Array[fx.Float32, s * TOPK * ROWS, 16]
            lpof: fx.Array[fx.Int32, MAX_PICKS, 16]
            pickof: fx.Array[fx.Int32, MAX_PICKS, 16]

        _DOWN_LDS[s] = DownLdsEp
    return _DOWN_LDS[s]


def down_loads_ep(c, rgd, nu, b0):
    """The w2 tiles of a wave's K steps 4 w .. 4 w + 3 and their scale codes for
    the slots b0 .. b0 + DOWN_BATCH_EP (past |U| the last slot's, local expert
    0 when U is empty): flat, tiles then codes. Pooled rows 16 rgd .."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    # bounded resources: a slot past |U| addresses past the end, and an
    # out-of-range buffer load returns 0 without touching memory (its MFMA
    # result is never stored: ``picked`` is false for it)
    r_w = rsrc(a["w2"], W2_BYTES)
    r_s = rsrc(a["w2s"], W2S_BYTES)
    sls = [b0 + j for j in range(DOWN_BATCH_EP)]
    es = [slot_expert(c, fx.min(sl, nu - 1), True) for sl in sls]
    bases = [
        (sl < nu).select(e * EP_GEOM.w2_expert_dwords, fx.Int32(OOB_DWORD))
        for sl, e in zip(sls, es)
    ]
    rows = [
        (sl < nu).select(e * MOE_HIDDEN + rgd * ROWS + lane % 16, fx.Int32(OOB_ROW))
        for sl, e in zip(sls, es)
    ]
    sts = [fx.Int32(wave) * EP_WAVE_STEPS + q for q in range(EP_WAVE_STEPS)]
    av = [
        fp4_tile_load(r_w, bases[j], rgd, EP_GEOM.inter, st, lane * 4)
        for j in range(DOWN_BATCH_EP)
        for st in sts
    ]
    sa = [
        scale_byte(r_s, rows[j], 4 * st + lane // 16, EP_GEOM.ug_groups)
        for j in range(DOWN_BATCH_EP)
        for st in sts
    ]
    return av + sa


@traced
def local_picks(c):
    """This rank's live local picks in pick order: ``lpof[pick]`` (its local
    index, -1 for any other pick), ``pickof[lp]``; ``res`` zeroed. Returns n_lp
    (CTA-uniform). Ends with a barrier."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, d = c["rt"], c["lds"]["down"]
    tid32 = fx.Int32(tid)
    npk = s * TOPK
    e = _pick_id(rt, fx.min(tid32, npk - 1))
    ok = (tid32 < npk) & _live(e) & _local(c, e)
    off, tot = block_excl_scan(
        ok.select(fx.Int32(1), fx.Int32(0)), lane, wave, rt["scan"]
    )
    if tid < npk:
        fx.ptr_store(ok.select(off, fx.Int32(-1)), d["lpof"] + tid32)
    if ok:
        fx.ptr_store(tid32, d["pickof"] + off)
    for i in range_constexpr((npk * ROWS + THREADS - 1) // THREADS):
        idx = tid32 + THREADS * i
        if idx < npk * ROWS:
            fx.ptr_store(fx.Float32(0.0), d["res"] + idx)
    gpu.barrier()
    return uniform(tot)


def _ep_spec(c, route, a, idx, pick, ok):
    """(poll spec) of the 2 pairs at ``idx`` of region ``a`` for a WAVE-UNIFORM
    ``ok``: a staging slot past n_lp has no producer, so the spec names pick
    ``pick``'s ROUTE pairs instead (its id and the next pair, produced for every
    row; ``load_route`` saw them current), whose values the caller drops. The
    base select is scalar (``ok`` is wave-uniform)."""
    at = Addr(a.region, a.space, fx.Int64(ok.select(a.value, route.value)))
    rix = id_index(pick // TOPK, pick % TOPK)
    return (at, ok.select(idx, rix), 2)


POLL_GROUP = 16  # local picks a poll round trip stages (10 specs of 2 pairs)
WORD_THREADS = EP_PICK_WORDS // 2  # 256: threads a pick's words (2 a thread)
CODE_THREADS = EP_PICK_CODES // 2  # 64: threads a pick's codes (one wave)
assert THREADS % WORD_THREADS == 0 and CODE_THREADS == 64
assert POLL_GROUP % (THREADS // WORD_THREADS) == 0 and POLL_GROUP % WAVES == 0


@traced
def load_inter_ep(c, c0, n_lp):
    """Local picks c0 .. c0 + P (those below n_lp): their whole a2 rows ->
    ``interq`` (words), ``intsc`` (codes), POLL_GROUP picks a round trip (a
    dynamic loop: n_lp <= 16 costs one); a staging slot past n_lp gets zero
    words and the unit code without a poll. A wave's 64 threads poll one
    pick's pairs, so every spec is wave-uniform. Ends with a barrier."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    d, scr = c["lds"]["down"], c["scr"]
    tid32 = fx.Int32(tid)
    route = scr["route"]
    end = fx.min(n_lp - c0, fx.Int32(P))
    wp = THREADS // WORD_THREADS  # 2 picks' words a pass
    g0 = fx.Int32(0)
    while g0 < end:
        specs, oks, lps = [], [], []
        for i in range_constexpr(POLL_GROUP // wp):  # words 2 (tid % 256), + 1
            lpl = uniform(g0 + wp * i + tid32 // WORD_THREADS)  # a wave's 64 words
            ok = lpl < end
            pick = uniform(fx.ptr_load(d["pickof"] + fx.min(c0 + lpl, MAX_PICKS - 1)))
            pick = ok.select(pick, fx.Int32(0))
            pair = pick * EP_PICK_WORDS + 2 * (tid32 % WORD_THREADS)
            specs.append(_ep_spec(c, route, scr["inter_ep"], pair, pick, ok))
            oks.append(ok)
            lps.append(fx.min(lpl, P - 1))
        for i in range_constexpr(POLL_GROUP // WAVES):  # codes 2 lane, + 1
            lpl = uniform(g0 + WAVES * i + fx.Int32(wave))
            ok = lpl < end
            pick = uniform(fx.ptr_load(d["pickof"] + fx.min(c0 + lpl, MAX_PICKS - 1)))
            pick = ok.select(pick, fx.Int32(0))
            pair = pick * EP_PICK_CODES + 2 * fx.Int32(lane)
            specs.append(_ep_spec(c, route, scr["isc_ep"], pair, pick, ok))
            oks.append(ok)
            lps.append(fx.min(lpl, P - 1))
        got = c["poll"](specs)
        nwp = POLL_GROUP // wp
        for i in range_constexpr(nwp):
            at = d["interq"] + (lps[i] * EP_PICK_WORDS + 2 * (tid32 % WORD_THREADS))
            for q in range_constexpr(2):
                fx.ptr_store(oks[i].select(got[i][q], fx.Int32(0)), at + q)
        for i in range_constexpr(POLL_GROUP // WAVES):
            j = nwp + i
            at = d["intsc"] + (lps[j] * EP_PICK_CODES + 2 * fx.Int32(lane))
            for q in range_constexpr(2):
                fx.ptr_store(oks[j].select(got[j][q], fx.Int32(UNIT_SCALE)), at + q)
        g0 = g0 + POLL_GROUP
    gpu.barrier()


@traced
def stage_down_ep(c, rgd, nu, c0, ops0):
    """Wave w's K steps 4 w .. 4 w + 3 of every local slot's w2 against the
    staged a2 of its picks in this chunk (DOWN_BATCH_EP slots in flight, the
    next batch issued under this one's MFMAs; ``ops0`` the first, issued
    before the a2 poll) -> the wave's fp32 ``part``; then the waves' parts
    summed in wave order into ``res[pick]``. Ends with a barrier."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, d = c["rt"], c["lds"]["down"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    tok = lane % 16 < s
    nbs = DOWN_BATCH_EP * EP_WAVE_STEPS
    b0 = fx.Int32(0)
    ops = ops0
    while b0 < nu:
        nb = b0 + DOWN_BATCH_EP
        ops_n = down_loads_ep(c, rgd, nu, nb)
        for j in range_constexpr(DOWN_BATCH_EP):
            sl = b0 + j
            slot = fx.max(fx.min(sl, nu - 1), 0)
            k = fx.ptr_load(rt["kof"] + (slot * 8 + t))
            pick = t * TOPK + fx.max(fx.min(k, TOPK - 1), 0)
            lp = fx.ptr_load(d["lpof"] + pick) - c0
            picked = (k >= 0) & tok & (sl < nu) & (lp >= 0) & (lp < P)
            lpc = fx.max(fx.min(lp, P - 1), 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for q in range_constexpr(EP_WAVE_STEPS):
                st = fx.Int32(wave) * EP_WAVE_STEPS + q
                bw = fx.Vector(
                    fx.ptr_load(
                        d["interq"] + (lpc * EP_PICK_WORDS + 16 * st + 4 * kb),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                bw = fx.Vector.from_elements(
                    [picked.select(bw[x], fx.Int32(0)) for x in range(4)], fx.Int32
                )
                code = fx.ptr_load(d["intsc"] + (lpc * EP_PICK_CODES + 4 * st + kb))
                sb = picked.select(code, fx.Int32(UNIT_SCALE))
                acc = mfma_scaled(
                    ops[j * EP_WAVE_STEPS + q],
                    bw,
                    acc,
                    ops[nbs + j * EP_WAVE_STEPS + q],
                    sb,
                    FP4,
                    FP4,
                    0,
                    0,
                )
            # lane: rows 4 (lane / 16) + i of row t's pick
            if picked:
                for i in range_constexpr(4):
                    fx.ptr_store(
                        acc[i],
                        d["part"] + ((fx.Int32(wave) * P + lpc) * ROWS + 4 * kb + i),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    # thread (staged pick lp, row r): the 8 waves' parts in wave order
    lp = fx.Int32(tid) // ROWS
    r = fx.Int32(tid) % ROWS
    if c0 + lp < c["n_lp"]:
        v = fx.ptr_load(d["part"] + (lp * ROWS + r))
        for w in range_constexpr(1, WAVES):
            v = v + fx.ptr_load(d["part"] + ((w * P + lp) * ROWS + r))
        pick = fx.ptr_load(d["pickof"] + fx.min(c0 + lp, MAX_PICKS - 1))
        fx.ptr_store(v, d["res"] + (pick * ROWS + r))
    gpu.barrier()


@traced
def push_ep(c, rgd):
    """Every pick's 16 rows of ``res`` (zeros past the local picks), rounded
    once to bf16, to the column owner's RSX (``rgd // 32``). Ends with a
    barrier."""
    s, tid = c["S"], c["tid"]
    d = c["lds"]["down"]
    if tid < s * TOPK * (ROWS // 2):
        pick = fx.Int32(tid) // (ROWS // 2)
        rp = fx.Int32(tid) % (ROWS // 2)
        v0 = bf16_round(fx.ptr_load(d["res"] + (pick * ROWS + 2 * rp)))
        v1 = bf16_round(fx.ptr_load(d["res"] + (pick * ROWS + 2 * rp + 1)))
        c["put_bf"](
            peer_at(c["peer"]["rsx"], rgd // RED_TASKS),
            rsx_elem(c["rank"], pick, rgd, rp),
            [v0, v1],
        )
    gpu.barrier()


@traced
def run_down_ep(c, rgd, nu):
    """CTA ``rgd``'s EP down task: the local picks, then a chunk of them at a
    time (once, unless more than EP_STAGE_PICKS): the first w2 batch issued
    before the a2 polls, ``load_inter_ep``, ``stage_down_ep``; then the
    pushes."""
    n_lp = local_picks(c)
    c["n_lp"] = n_lp
    c0 = fx.Int32(0)
    while c0 < n_lp:
        ops = down_loads_ep(c, rgd, nu, fx.Int32(0))
        load_inter_ep(c, c0, n_lp)
        stage_down_ep(c, rgd, nu, c0, ops)
        c0 = c0 + P
    push_ep(c, rgd)
