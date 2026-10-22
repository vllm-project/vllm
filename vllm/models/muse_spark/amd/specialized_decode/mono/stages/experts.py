# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 3b: the routed experts of one Muse Spark layer and rank, from the
MXFP4 pre-expert-normed pooled rows (``xq`` / ``xsc`` in LDS, ``pool.load_xq``)
to every pick's bf16 rows pushed to their column owner (``rsx``).

Adapted from K3's latent MoE (pr60683 ``kimi_k3/amd/specialized_decode/mono/
stages/moe.py``: ``load_route``, ``ug_loads`` / ``stage_ug``, ``load_inter``,
``down_loads`` / ``stage_down``) to Muse's shapes (256 experts, top-8, I_local
512, pooled K 4096), its stage-1 numerics (aiter's SiLU, a bf16 h, the RoundUp
MXFP4 a2) and its reduce (unweighted rows, reduced by the column owner).

    union  (every CTA, ``load_route``): ROUTE -> the LDS route table; the union U
           of the step's live picks (``0 <= id < 256``; pad rows carry -1, a
           garbage id after a timed-out poll is dead too): ``slot_of[e]``,
           ``expert[slot]`` ascending (zeroed past |U|), ``kof[slot 8 + t]``
    ug     (|U| x 16 tasks, every CTA, reversed placement, ``run_ug``): 32 gate +
           32 up rows of a slot's expert against the MXFP4 rows (FP4 x FP4
           scaled MFMA, K halves on waves 0-3 / 4-7), SiLU (aiter: hw exp2 /
           rcp), h rounded to bf16, MXFP4 a 32-column group (RoundUp) ->
           INTER / ISC of each picking row
    down   (CTA b: 16 of the 4096 pooled rows, ``run_down``): w2 of every slot
           against its picks' INTER (live picks only polled) -> a bf16 row a
           pick, UNWEIGHTED (the post-expert norm comes before the route weight)
           -> the owner rank's RSX (``b // 32``), zeros included

Pad rule: a dead pick (pad row, garbage id) is never in U and never polled; its
contribution rows stay the zeros ``load_inter`` stores (``select``, never a mask
multiply). Every expert id is clamped into [0, 256) and every slot into
[0, MAX_U) at each address use, so a timed-out step keeps its addresses in
bounds. Every stage here touching the stage LDS or ``red`` ends with a barrier
(the LDS hand-over rule).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.mx import (
    FP4,
    UNIT_SCALE,
    e8m0_roundup,
    fp4_tile_load,
    mfma_scaled,
    pk_fp4_f32,
    pow2,
    scale_byte,
    w13_scale_word,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    block_excl_scan,
    butterfly,
    hw_sigmoid_approx,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import peer_at
from vllm.models.muse_spark.amd.specialized_decode.mono.common.route_lds import (
    clamp_expert,
    id_index,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import Addr
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    AMAX_FLOOR,
    BLOCKS,
    DOWN_BATCH,
    EXPERTS,
    INTER,
    MAX_PICKS,
    MAX_U,
    MOE_HIDDEN,
    OWN_COLS,
    PAIR_BYTES,
    RED_TASKS,
    ROWS,
    THREADS,
    TOPK,
    UG_GROUPS,
    W2_STEPS,
    W13_K,
    W13_STEPS,
    WAVES,
    XQ_WORDS,
    XSC_WORDS,
    scratch_layout,
)

HALF_STEPS = W13_STEPS // 2  # 16 K steps a ug wave (its K half)
W13_ROWS = 2 * INTER  # an expert's w13 rows: gate 512 | up 512
W13_EXPERT_DWORDS = W13_ROWS * W13_K // 8  # 524288: an expert's fp4 w13
W2_EXPERT_DWORDS = MOE_HIDDEN * INTER // 8  # 262144: an expert's fp4 w2
W2_SCALE_COLS = INTER // 32  # 16 e8m0 codes a w2 row
INTER_WORDS = INTER // 8  # 64 fp4 words of a pick's a2 row
UP_GROUP0 = INTER // ROWS  # 32: the first up row group of an expert
_LAY = scratch_layout()
# ISC's first pair counted from ROUTE's (``load_inter`` polls both over one base)
ISC_FROM_ROUTE = (_LAY["isc"][0] - _LAY["route"][0]) // PAIR_BYTES

assert MAX_U * TOPK == THREADS and EXPERTS <= THREADS and MAX_U <= THREADS
assert W13_STEPS % 2 == 0 and HALF_STEPS % 2 == 0 and WAVES == 8
# load_inter: a wave's 64 INTER words are one pick's (its spec is wave-uniform)
assert INTER_WORDS == 64 and THREADS % INTER_WORDS == 0
assert ISC_FROM_ROUTE > 0 and (_LAY["isc"][0] - _LAY["route"][0]) % PAIR_BYTES == 0


# ---------------------------------------------------------------- LDS

_UG_LDS: dict = {}
_DOWN_LDS: dict = {}


def ug_lds_struct(s: int):
    """The ug epilogue's buffer: ``av`` f32[THREADS], a (row, column) thread's bf16
    h, read back 8 at a time for the fp4 pack. K2 lays it out after
    ``pool.xq_lds_struct`` (the ``xq_ug`` composite: both live through the ug
    tasks)."""
    if s not in _UG_LDS:

        @fx.struct
        class UgLds:
            av: fx.Array[fx.Float32, THREADS, 16]

        _UG_LDS[s] = UgLds
    return _UG_LDS[s]


def down_lds_struct(s: int):
    """The down stage's operands: ``interq`` (a pick's 64 a2 words), ``intsc`` (its 16
    codes) and ``contrib`` (its 16 bf16-rounded rows of this CTA), s 8 picks."""
    if s not in _DOWN_LDS:

        @fx.struct
        class DownLds:
            interq: fx.Array[fx.Int32, s * TOPK * INTER_WORDS, 16]
            intsc: fx.Array[fx.Int32, s * TOPK * UG_GROUPS, 16]
            contrib: fx.Array[fx.Float32, s * TOPK * ROWS, 16]

        _DOWN_LDS[s] = DownLds
    return _DOWN_LDS[s]


# ---------------------------------------------------------------- indexes
# Plain integer arithmetic: the kernels' traced Int32 math and the host tests
# read the same formulas.


def inter_word(pick, g, w):
    """``inter`` pair of a pick's a2 word w (columns 32 g + 8 w .. + 8) of
    32-column group g."""
    return pick * INTER_WORDS + 4 * g + w


def inter_read_word(pick, st, kb, d):
    """The a2 word a down lane (K block ``kb`` = lane / 16) feeds its MFMA at
    128-K step ``st``: the K order of w2's ``fp4_tile_load`` (K 128 st + 32 kb
    + 8 d ..)."""
    return pick * INTER_WORDS + 16 * st + 4 * kb + d


def isc_index(pick, g):
    """``isc`` pair of a pick's group-g a2 code."""
    return pick * UG_GROUPS + g


def xq_read_word(t, st, kb, d):
    """The a1 word (``pool.load_xq``'s ``xq[t 512 + kb' 4 + w]``, kb' the
    32-column block) a ug lane feeds its MFMA at step ``st``, K block ``kb``."""
    return t * XQ_WORDS + 16 * st + 4 * kb + d


def ug_row_group(wave, g):
    """The w13 16-row group wave ``wave`` multiplies for 32-column group ``g``:
    waves 0 / 4 gate rows 32 g .., 1 / 5 gate 32 g + 16 .., 2 / 6 up rows
    512 + 32 g .., 3 / 7 up 512 + 32 g + 16 .. (a 32-row scale block holds two
    gate or two up groups, never one of each)."""
    rgl = wave % 4
    return (rgl // 2) * UP_GROUP0 + 2 * g + rgl % 2


def rsx_elem(rank, pick, rgd, rp):
    """``put_bf`` element index of down CTA ``rgd``'s column pair ``rp`` of a
    pick in its owner's ``rsx`` (2 ``spec.rsx_pair``)."""
    return (rank * MAX_PICKS + pick) * OWN_COLS + (rgd % RED_TASKS) * ROWS + 2 * rp


def slot_expert(c, slot):
    """Union slot ``slot``'s expert id, wave-uniform, the slot clamped into
    [0, MAX_U) and the id into [0, EXPERTS): an in-bounds weight address for
    any slot (past |U| ``expert`` holds 0) or garbage id."""
    sl = fx.max(fx.min(slot, MAX_U - 1), 0)
    return clamp_expert(uniform(fx.ptr_load(c["rt"]["expert"] + sl)))


def _pick_id(rt, pick):
    """The expert id of pick ``pick`` = t 8 + k in the LDS route table."""
    return fx.ptr_load(rt["route"] + id_index(pick // TOPK, pick % TOPK))


def _live(e):
    return (e >= 0) & (e < EXPERTS)


# ---------------------------------------------------------------- union


@traced
def load_route(c):
    """ROUTE -> the LDS route table (ids, weight bits) and the union U of the
    live picks: ``flag[e]`` (U's slot of e), ``expert[slot]`` (ascending, 0
    past |U|), ``kof[slot 8 + t]`` (row t's top-k index of the slot's expert,
    -1 when t did not pick it). LDS is not zeroed at launch: every field read
    later is written here first. Returns |U| (wave-uniform), 0 when every row
    is a pad."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    tid32 = fx.Int32(tid)
    if tid < s * 16:
        v = c["poll"]([(c["scr"]["route"], tid32, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid32)
    if tid < EXPERTS:
        fx.ptr_store(fx.Int32(0), rt["flag"] + tid32)
    if tid < MAX_U:
        fx.ptr_store(fx.Int32(0), rt["expert"] + tid32)
    fx.ptr_store(fx.Int32(-1), rt["kof"] + tid32)  # MAX_U 8 == THREADS words
    gpu.barrier()
    if tid < s * TOPK:
        e = _pick_id(rt, tid32)
        if _live(e):
            fx.ptr_store(fx.Int32(1), rt["flag"] + clamp_expert(e))
    gpu.barrier()
    f = (tid < EXPERTS).select(
        fx.ptr_load(rt["flag"] + fx.min(tid32, EXPERTS - 1)), fx.Int32(0)
    )
    off, tot = block_excl_scan(f, lane, wave, rt["scan"])
    gpu.barrier()  # every flag read before a slot overwrites one
    if f == 1:
        fx.ptr_store(off, rt["flag"] + tid32)
        fx.ptr_store(tid32, rt["expert"] + fx.min(off, MAX_U - 1))
    gpu.barrier()
    if tid < s * TOPK:
        e = _pick_id(rt, tid32)
        if _live(e):
            sl = fx.min(fx.ptr_load(rt["flag"] + clamp_expert(e)), MAX_U - 1)
            fx.ptr_store(tid32 % TOPK, rt["kof"] + (sl * 8 + tid32 // TOPK))
    gpu.barrier()
    return fx.min(uniform(tot), fx.Int32(MAX_U))


# ---------------------------------------------------------------- ug


def ug_loads(c, task):
    """Task ``task``'s w13 operands for this wave (its row group, its K half:
    16 dwordx4) and their scale words (8: a word serves 2 steps x 2 row
    groups), issued a task ahead (``stage_ug`` takes them). The task is the
    caller's, clamped to the step's last (``run_ug``)."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    e = slot_expert(c, task // UG_GROUPS)
    rg = ug_row_group(wave, task % UG_GROUPS)
    st0 = (wave // 4) * HALF_STEPS
    r_w = rsrc(a["w13"])
    base = e * W13_EXPERT_DWORDS
    av = [
        fp4_tile_load(r_w, base, rg, W13_K, st0 + i, lane * 4)
        for i in range(HALF_STEPS)
    ]
    r_s = rsrc(a["w13s"])
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, st0 + i, lane), vec_width=1, dtype=T.i32
            )
        )
        for i in range(0, HALF_STEPS, 2)
    ]
    return av + sw


@traced
def stage_ug(c, task, ops):
    """Slot task / 16, 32-column group g = task % 16: the gate rows 32 g .. and
    up rows 512 + 32 g .. of the slot's expert (``ops``: ``ug_loads``) against
    the MXFP4 rows; thread (row tt = tid / 32, column) sums its gate / up over
    the K halves, h = bf16((g sigmoid(g)) u) (aiter's ``silu_mul``: hw exp2 and
    rcp), the 32-column group's RoundUp code, the fp4 pack -> INTER / ISC of
    every row that picked the expert."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt, xq = c["rt"], c["lds"]["xq"]
    av_l = c["lds"]["ug"]["av"]
    slot = fx.max(fx.min(task // UG_GROUPS, MAX_U - 1), 0)
    g = task % UG_GROUPS
    rg = ug_row_group(wave, g)
    st0 = (wave // 4) * HALF_STEPS
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(HALF_STEPS):
        st = st0 + i
        bw = fx.Vector(
            fx.ptr_load(
                xq["xq"] + xq_read_word(t, st, kb, 0),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = fx.ptr_load(xq["xsc"] + (t * XSC_WORDS + st * 4 + kb))
        # the code's byte (st % 2) 2 + rg % 2 of the word (st % 2 == i % 2)
        sa = (ops[HALF_STEPS + i // 2] >> ((i % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(ops[i], bw, acc, sa, sb, FP4, FP4, 0, 0)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    tt = tid // 32
    col = tid % 32
    on = tt < s
    tt = fx.min(tt, s - 1)
    first = col < 16  # row group 2 g (waves 0 / 2, + 4 the K's 2nd half) or 2 g + 1
    r = col % 16
    gv = first.select(
        row_sum(red, r, tt, waves=(0, 4)), row_sum(red, r, tt, waves=(1, 5))
    )
    uv = first.select(
        row_sum(red, r, tt, waves=(2, 6)), row_sum(red, r, tt, waves=(3, 7))
    )
    h = bf16_round((gv * hw_sigmoid_approx(gv)) * uv)
    amax = fx.max(butterfly(abs(h), (16, 8, 4, 2, 1), fx.max), fx.Float32(AMAX_FLOOR))
    code = e8m0_roundup(amax)
    fx.ptr_store(h, av_l + tid)
    gpu.barrier()
    k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
    if on & (col < 4) & (k >= 0):
        scale = pow2(code)
        pk = fx.Int32(0)
        for j in range_constexpr(4):
            x0 = fx.ptr_load(av_l + (tt * 32 + col * 8 + 2 * j))
            x1 = fx.ptr_load(av_l + (tt * 32 + col * 8 + 2 * j + 1))
            pk = pk_fp4_f32(pk, x0, x1, scale, j)
        pick = tt * TOPK + k
        c["put_words"](c["scr"]["inter"], inter_word(pick, g, col), [pk])
        if col == 0:
            c["put"](c["scr"]["isc"], isc_index(pick, g), code)
    gpu.barrier()


@traced
def run_ug(c, nu):
    """Every ug task of the step (|U| x 16) over the grid, reversed placement
    (CTA b: tasks 255 - b, + 256, ...: the leftover tasks of the last round go
    to the high CTAs, whose early work ends first), the next task's weights in
    flight under this one's MFMAs. ``nu == 0`` runs no task (task index 0's
    loads, expert 0, are issued and unused)."""
    total = nu * UG_GROUPS
    last = fx.max(total - 1, 0)
    task = BLOCKS - 1 - fx.Int32(c["bid"])
    ops = ug_loads(c, fx.min(task, last))
    while task < total:
        nxt = task + BLOCKS
        ops_n = ug_loads(c, fx.min(nxt, last))
        stage_ug(c, task, ops)
        task = nxt
        ops = ops_n


# ---------------------------------------------------------------- down


def down_loads(c, rgd, nu, b0):
    """The w2 tiles (4 steps) and scale codes of a wave's batch of slots b0,
    b0 + 8, ... (DOWN_BATCH; past |U| the last slot's, and expert 0 when U is
    empty): flat, tiles then codes. Pooled rows 16 rgd .. of each expert."""
    lane = c["lane"]
    a = c["args"]
    r_w = rsrc(a["w2"])
    r_s = rsrc(a["w2s"])
    es = [slot_expert(c, fx.min(b0 + WAVES * j, nu - 1)) for j in range(DOWN_BATCH)]
    av = [
        fp4_tile_load(r_w, es[j] * W2_EXPERT_DWORDS, rgd, INTER, st, lane * 4)
        for j in range(DOWN_BATCH)
        for st in range(W2_STEPS)
    ]
    sa = [
        scale_byte(
            r_s,
            es[j] * MOE_HIDDEN + rgd * ROWS + lane % 16,
            4 * st + lane // 16,
            W2_SCALE_COLS,
        )
        for j in range(DOWN_BATCH)
        for st in range(W2_STEPS)
    ]
    return av + sa


def _inter_spec(rt, route, a, idx, pick):
    """(poll spec, live) of INTER pair ``idx`` of a WAVE-UNIFORM pick ``pick``
    (``wave + 8 i``: a wave's 64 words are one pick's): a dead pick's pair has
    no producer, so the spec names the pick's own ROUTE pair instead (produced
    for every row; ``load_route`` saw it current), whose value the caller
    drops. The id goes through ``uniform``, so the base select is scalar and
    the buffer resource stays in SGPRs."""
    ok = _live(uniform(_pick_id(rt, pick)))
    at = Addr(a.region, a.space, fx.Int64(ok.select(a.value, route.value)))
    rix = id_index(pick // TOPK, pick % TOPK)
    return (at, ok.select(idx, rix), 1), ok


def _isc_spec(rt, route, idx, pick):
    """(poll spec, live) of ISC pair ``idx`` of pick ``pick``, which differs
    across a wave (16 codes a pick): as ``_inter_spec``, but the dead / live
    choice is made in the INDEX over one wave-uniform base, ``route`` (ISC
    sits ``ISC_FROM_ROUTE`` pairs past it in the scratch). A per-lane base
    would make the buffer resource divergent, and the compiler then wraps every
    load of the poll loop in a readfirstlane waterfall. The spec keeps the
    region name ``isc``; a give-up record never names it (a batch's record
    names its first spec, an INTER one)."""
    ok = _live(_pick_id(rt, pick))
    at = Addr("isc", route.space, route.value)
    rix = id_index(pick // TOPK, pick % TOPK)
    return (at, ok.select(idx + ISC_FROM_ROUTE, rix), 1), ok


@traced
def load_inter(c):
    """The step's INTER words and ISC codes -> LDS ``interq`` / ``intsc``, live
    picks only: a dead pick (pad row, garbage id) has no producer, so it gets
    zero words and the unit code without a poll (its pair is replaced in the
    poll batch by the pick's own ROUTE pair, which ``load_route`` already saw
    current, so the batch stays one round trip). Also zeroes ``contrib``."""
    s, tid, wave = c["S"], c["tid"], c["wave"]
    rt, d, scr = c["rt"], c["lds"]["down"], c["scr"]
    tid32 = fx.Int32(tid)
    inter, route = scr["inter"], scr["route"]

    nw = s * TOPK * INTER_WORDS  # a multiple of THREADS
    nc = s * TOPK * UG_GROUPS
    specs, oks = [], []
    for i in range_constexpr(nw // THREADS):
        idx = tid32 + THREADS * i  # pick idx // 64 = wave + 8 i
        pick = fx.Int32(wave) + (THREADS // INTER_WORDS) * i
        sp, ok = _inter_spec(rt, route, inter, idx, pick)
        specs.append(sp)
        oks.append(ok)
    for i in range_constexpr((nc + THREADS - 1) // THREADS):
        idx = fx.min(tid32 + THREADS * i, nc - 1)
        sp, ok = _isc_spec(rt, route, idx, idx // UG_GROUPS)
        specs.append(sp)
        oks.append(ok)
    got = c["poll"](specs)
    for i in range_constexpr(nw // THREADS):
        idx = tid32 + THREADS * i
        fx.ptr_store(oks[i].select(got[i][0], fx.Int32(0)), d["interq"] + idx)
    for i in range_constexpr((nc + THREADS - 1) // THREADS):
        j = nw // THREADS + i
        idx = tid32 + THREADS * i
        if idx < nc:
            fx.ptr_store(
                oks[j].select(got[j][0], fx.Int32(UNIT_SCALE)), d["intsc"] + idx
            )
    for i in range_constexpr((s * TOPK * ROWS + THREADS - 1) // THREADS):
        idx = tid32 + THREADS * i
        if idx < s * TOPK * ROWS:
            fx.ptr_store(fx.Float32(0.0), d["contrib"] + idx)
    gpu.barrier()


@traced
def stage_down(c, rgd, nu, ops0):
    """Pooled rows 16 rgd ..: every slot's w2 against its picks' a2 (wave w the
    slots w, w + 8, ...; DOWN_BATCH in flight, the next batch issued under this
    one's MFMAs; ``ops0`` the first, issued before INTER was polled) -> one
    bf16 row a pick, unweighted (stock stage 2: one top-1 row a route, unit
    weight, bf16) -> the owner rank's RSX (``rgd // 32``). Every CTA pushes all
    its S 8 picks, zeros included, so the owner never waits on a producer that
    skipped."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, d = c["rt"], c["lds"]["down"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    tok = lane % 16 < s
    nbs = DOWN_BATCH * W2_STEPS
    b0 = fx.Int32(wave)
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DOWN_BATCH
        ops_n = down_loads(c, rgd, nu, nb)
        for j in range_constexpr(DOWN_BATCH):
            sl = b0 + WAVES * j
            slot = fx.max(fx.min(sl, nu - 1), 0)
            k = fx.ptr_load(rt["kof"] + (slot * 8 + t))
            picked = (k >= 0) & tok & (sl < nu)
            pick = t * TOPK + fx.max(fx.min(k, TOPK - 1), 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(W2_STEPS):
                bw = fx.Vector(
                    fx.ptr_load(
                        d["interq"] + inter_read_word(pick, st, kb, 0),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                bw = fx.Vector.from_elements(
                    [picked.select(bw[q], fx.Int32(0)) for q in range(4)], fx.Int32
                )
                code = fx.ptr_load(d["intsc"] + isc_index(pick, 4 * st + kb))
                sb = picked.select(code, fx.Int32(UNIT_SCALE))
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
            # lane: rows 4 (lane / 16) + i of row t's pick
            if picked:
                for i in range_constexpr(4):
                    fx.ptr_store(
                        bf16_round(acc[i]),
                        d["contrib"] + (pick * ROWS + 4 * kb + i),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < s * TOPK * (ROWS // 2):
        pick = fx.Int32(tid) // (ROWS // 2)
        rp = fx.Int32(tid) % (ROWS // 2)
        v0 = fx.ptr_load(d["contrib"] + (pick * ROWS + 2 * rp))
        v1 = fx.ptr_load(d["contrib"] + (pick * ROWS + 2 * rp + 1))
        c["put_bf"](
            peer_at(c["peer"]["rsx"], rgd // RED_TASKS),
            rsx_elem(c["rank"], pick, rgd, rp),
            [v0, v1],
        )
    gpu.barrier()


def run_down(c, rgd, nu):
    """CTA ``rgd``'s down task (5.2 order): its first w2 batch issued before the
    INTER polls, then ``load_inter``, then ``stage_down``."""
    ops = down_loads(c, rgd, nu, fx.Int32(c["wave"]))
    load_inter(c)
    stage_down(c, rgd, nu, ops)
