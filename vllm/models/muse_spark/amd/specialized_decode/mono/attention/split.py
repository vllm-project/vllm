# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split) phase 1, Phase C (DESIGN_BC 7): the split half of an in-kernel
split-KV GQA decode core for the q = 1 widths, and the core load that reads
the combine's gated rows.

    split       one unit a CTA: unit u = (t 2 + kvh) N + p (N =
                ``K2Build.nsplit()``) attends token t's 8 q heads of
                KV head kvh over part p of its pages, flash-decoding style ->
                its unnormalized partial ``opart`` and (max, sum) ``pstat``
                (fp32, device scope), then the ``srdy`` flag of u
    load_core   every CTA: the 16 S ``crdy`` flags (the combine's, one a
                (token, q head)), then the gated rows ``xo`` -> LDS ``xl``
                (back's padded rows), a pad row as zeros

Keys, from the layer's KV group (``block_table``, ``seq_lens``) and kind
(``K2Build.window``), all in-kernel: L = ``seq_lens[t]`` clamped to the
table's row (0 on a pad row), keys lo <= k < L with lo = L - WINDOW on a
sliding layer (else 0); pages b0 = lo // 16 .. (L - 1) // 16, nb of them, cut
into N parts of ``per = max(cdiv(nb, N), MIN_PER)`` pages: the combine's own
``unit_plan``, so a unit holds keys iff the combine waits
for it (p < live).
Pages outside [b0, b0 + nb) are never read: a sliding group's table nulls only
entries below b0 (DESIGN_BC 1.5). A page is one (block, KV head) of the cache
``kv`` [blocks, 2, 16, 128] bf16 at byte strides ``kv_sb, kv_sh, kv_st``: its
base is 64-bit (a layer's cache passes 4 GiB), a buffer resource of 16 rows.

Within a unit wave w takes pages w, w + 8, ... (``uniform``: the loop count is
wave-uniform), two in flight (the next page's K and V issued before this one's
math), the block ids lane-parallel, 64 a load. Per page (lane (j, g) = (lane %
16, lane // 16); MFMA maps as DESIGN_BC 7.3):

    QK   2 x mfma_f32_16x16x32_bf16, A = the page's 16 keys (lane: key j, dims
         16 g ..), B = q (column j = q head 8 kvh + j, zero for j >= 8),
         contraction s element 8 g + e <-> dim 16 g + 8 s + e on both sides
         -> lane (j, g) holds head j's scores of keys 4 g + e
    soft base 2, ``s = qk QK_SCALE_LOG2E``, a masked key NEG (select, never a
         multiply); m, alpha per head (xshfl 16, 32); P = bf16(exp2(s - m)),
         ``l`` the sum of the bf16 P the MFMA consumes (E9)
    PV   4 x mfma_f32_16x16x16bf16_1k, B = P as the lane holds it (no LDS),
         A = V^T, the masked keys' V words zeroed (NaN-safe: 0 x NaN is NaN)
         -> lane (j, g): head j's dims 16 g + 4 e + dt

The 8 waves merge in LDS (``lds_struct``) in wave order; thread (j, d) = (tid
// 64, tid % 64) stores its head's dim d. An empty unit or wave carries m =
NEG, l = 0, o = 0. Plain regions read in this launch are stored ``CM_DEV``
(rule 9), and the flag is published after them (``publish``).

Order (rule 8, DESIGN_BC 7.2): ``stage_split`` polls nothing and holds no
o_proj weights; its first loads are the pad word, ``seq_lens[t]`` and q.
``load_core`` polls ``crdy`` with only the CTA's own o_proj weights in flight
(their stage needs them next).
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.attention.combine import (
    unit_plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_round,
    hw_exp2,
    ld_dev,
    ld_i32,
    ld_pad,
    mfma_bf16,
    readlane,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    THREADS,
    WAVES,
    sel,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    publish,
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    CORE,
    HEAD_DIM,
    HEADS,
    KV_BLOCK,
    KV_HEADS,
    OPART,
    PSTAT,
    Q_PER_KV,
    QK_SCALE_LOG2E,
    QKV_ROWS,
    V_OFF,
    XO,
    K2Build,
    crdy_pair,
    opart_word,
    pstat_word,
    srdy_pair,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv

STAGE = "split"

NEG = -3.4028234663852886e38  # finite: NEG - NEG = 0, never inf - inf
BID_LANES = 64  # block ids a wave loads at once (a lane each)
XL_ROW = CORE + gemv.LDS_PAD  # bf16 a padded LDS core row (back.XL_ROW)
OKW = CORE // 8  # dwordx4 a core row
V_BYTES = 2 * V_OFF  # V's byte offset in the row
LANE_HEADS = 16  # MFMA columns: q heads j < Q_PER_KV, the rest zero
assert HEADS == KV_HEADS * Q_PER_KV and Q_PER_KV == WAVES and HEAD_DIM == 64
assert KV_BLOCK == 16  # a page is one MFMA M: 16 keys


@functools.cache
def lds_struct(s: int):
    """The split's LDS: each wave's per-head max ``ms`` and sum ``ls`` (16
    columns, heads 0 .. 7 used) and its 8 heads' 64-dim rows ``os``, fp32:
    17,408 B, the same at every width (``s`` keys the stage union only).
    Cached: one struct type a width."""

    @fx.struct
    class SplitLds:
        ms: fx.Array[fx.Float32, WAVES * LANE_HEADS, 16]
        ls: fx.Array[fx.Float32, WAVES * LANE_HEADS, 16]
        os: fx.Array[fx.Float32, WAVES * Q_PER_KV * HEAD_DIM, 16]

    return SplitLds


def key_of(c) -> K2Build:
    """The build's key, a K2Build(core="split") (its window, N and MIN_PER
    are static)."""
    key = c.get("key")
    if not isinstance(key, K2Build) or key.core != "split":
        raise ValueError(f"the split core needs a K2Build(core='split') key: {key!r}")
    return key


def entry_loads(c, t, kvh) -> dict:
    """The unit's first loads, raw (anything computed from them here would
    wait for them before the rest are issued): row t's pad word and sequence
    length, then this lane's q operand (``_q_loads``)."""
    a = c["args"]
    return {
        "pad": ld_i32(a["pad"], t // 4),
        "n": ld_i32(a["seq_lens"], t),
        "q": _q_loads(c, t, kvh),
    }


def row_len(c, t, pre):
    """Row t's context length L from ``entry_loads``' words (wave-uniform): 0
    on a pad row, else ``seq_lens[t]`` clamped to [0, the table's row of
    ``bt_stride`` pages)."""
    pad_t = ((uniform(pre["pad"]) >> (8 * (t % 4))) & 0xFF) != 0
    cap = fx.Int32(c["args"]["bt_stride"]) * KV_BLOCK
    L = fx.max(fx.min(uniform(pre["n"]), cap), 0)
    return pad_t.select(fx.Int32(0), L)


def pages(L, window: bool, n: int, p):
    """Part p of N = ``n`` at context length L: (lo, pa, pe), keys lo <= k <
    L, pages [pa, pe), empty when pe <= pa (p >= the combine's live). The
    combine's ``unit_plan``; Python ints or traced Int32."""
    lo, b0, nb, per, _ = unit_plan(L, window, n)
    pa = b0 + p * per
    pe = sel(pa + per < b0 + nb, pa + per, b0 + nb)
    return lo, pa, pe


def _q_loads(c, t, kvh):
    """This lane's q operand, two dwordx4 (MFMA s = 0, 1): head 8 kvh + j,
    dims 16 g + 8 s .. + 8 of qkvo row t (a lane j >= 8 loads head 8 kvh + 7;
    ``stage_split`` zeroes it). Raw: the CTA's entry loads."""
    lane = c["lane"]
    j = fx.min(lane % LANE_HEADS, Q_PER_KV - 1)
    g = lane // LANE_HEADS
    col = HEAD_DIM * (Q_PER_KV * kvh + j) + 16 * g
    r_q = rsrc(c["args"]["qkvo"])
    return [
        fx.Vector(
            bo.buffer_load(
                r_q, (t * QKV_ROWS + col + 8 * s) // 2, vec_width=4, dtype=T.i32
            )
        )
        for s in range(2)
    ]


def _bid_chunk(c, t, first, stop, wave, k):
    """Chunk k of a wave's block ids: lane i holds the table entry of page
    first + wave + 8 (64 k + i) of row t, or 0 past ``stop`` (a masked load:
    the resource is sized by the table's buffer, ``bt_rows`` rows)."""
    a = c["args"]
    stride = fx.Int32(a["bt_stride"])
    nbytes = fx.Int64(fx.Int32(a["bt_rows"])) * fx.Int64(stride) * 4
    r_bt = rsrc(a["block_table"], fx.min(nbytes, fx.Int64(0xFFFFFFFF)))
    pg = first + wave + WAVES * (BID_LANES * k + c["lane"])
    ok = pg < stop
    return fx.Int32(
        bo.buffer_load(
            r_bt,
            t * stride + fx.min(pg, fx.max(stop - 1, 0)),
            vec_width=1,
            dtype=T.i32,
            mask=ok,
        )
    )


def _page_loads(c, kvh, blk):
    """Page ``blk``'s (KV head kvh) operands of this lane: K of key lane % 16,
    dims 16 g .. + 16 (2 dwordx4, the QK A operands), and V of keys 4 g + e,
    dims 4 (lane % 16) .. + 4 (4 dwordx2, e = 0 .. 3). The page's base is
    64-bit and wave-uniform; the offsets are 32-bit within its 16 rows."""
    a = c["args"]
    lane = c["lane"]
    r16 = lane % 16
    g = lane // 16
    st = fx.Int32(a["kv_st"])
    page = (
        a["kv"]
        + fx.Int64(blk) * fx.Int64(a["kv_sb"])
        + fx.Int64(kvh) * fx.Int64(a["kv_sh"])
    )
    rp = rsrc(page, st * KV_BLOCK)
    kk = [
        fx.Vector(
            bo.buffer_load(
                rp, (r16 * st + 32 * g + 16 * s) // 4, vec_width=4, dtype=T.i32
            )
        )
        for s in range(2)
    ]
    vv = [
        fx.Vector(
            bo.buffer_load(
                rp,
                ((4 * g + e) * st + V_BYTES + 8 * r16) // 4,
                vec_width=2,
                dtype=T.i32,
            )
        )
        for e in range(4)
    ]
    return kk + vv


def _v_operand(vv, dt):
    """The PV A operand of MFMA dt: V^T row i (lane % 16) = dim 4 i + dt, keys
    4 g + e (e = 0 .. 3): bf16 dt of each key's 4 dims, as i16 x 4."""
    lo = dt % 2 == 0
    w = [vv[e][dt // 2] for e in range(4)]
    if lo:
        pk = [(w[2 * h] & 0xFFFF) | (w[2 * h + 1] << 16) for h in range(2)]
    else:
        pk = [
            ((w[2 * h] >> 16) & 0xFFFF) | (w[2 * h + 1] & fx.Int32(-65536))
            for h in range(2)
        ]
    return fx.Vector.from_elements(pk, fx.Int32).bitcast(fx.Int16)


def _scale(v, al):
    return fx.Vector.from_elements([v[i] * al for i in range(4)], fx.Float32)


def _attend(c, ops, qa, i, base, lo, L, live, m, den, acc):
    """Page i of the wave's (page ``base + 8 i``; ``ops``: its
    ``_page_loads``) into this lane's running max ``m``, sum ``den`` (7.3's
    l) and ``acc`` of head j: QK, the masked base-2 scores, the rescale, PV.
    ``live``: whether the page is the wave's (False: every key masked, the
    state unchanged). -> (m, den, acc)."""
    g = c["lane"] // LANE_HEADS
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    # keys 16 pg + 4 g + e of this lane: in [lo, L)?
    kb = KV_BLOCK * (base + WAVES * i) + 4 * g
    ok = [live & (kb + e >= lo) & (kb + e < L) for e in range(4)]
    sc = mfma_bf16(ops[0].bitcast(fx.BFloat16), qa[0], zero4)
    sc = mfma_bf16(ops[1].bitcast(fx.BFloat16), qa[1], sc)
    sv = [
        fx.Float32(ok[e].select(sc[e] * fx.Float32(QK_SCALE_LOG2E), fx.Float32(NEG)))
        for e in range(4)
    ]
    mt = fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3]))
    mt = fx.max(mt, xshfl(mt, 16))
    mt = fx.max(mt, xshfl(mt, 32))
    mn = fx.max(m, mt)
    al = hw_exp2(m - mn)
    pr = [
        bf16_round(ok[e].select(hw_exp2(sv[e] - mn), fx.Float32(0.0))) for e in range(4)
    ]
    ln = den * al + ((pr[0] + pr[1]) + (pr[2] + pr[3]))
    pb = fx.Vector.from_elements(pr, fx.Float32).to(fx.BFloat16).bitcast(fx.Int16)
    # a masked key's V words zeroed in the operand (key 4 g + e: ok[e])
    vv = [
        fx.Vector.from_elements(
            [ok[e].select(ops[2 + e][h], fx.Int32(0)) for h in range(2)], fx.Int32
        )
        for e in range(4)
    ]
    accn = [
        fx.Vector(
            rocdl.mfma_f32_16x16x16bf16_1k(
                T.vec(4, T.f32), [_v_operand(vv, dt), pb, _scale(acc[dt], al)]
            )
        )
        for dt in range(4)
    ]
    return mn, ln, accn


def _block(bid, i, kv_last):
    """Page i's block id from the wave's id chunk, clamped into the cache."""
    return fx.max(fx.min(readlane(bid, i % BID_LANES), kv_last), 0)


@traced
def stage_split(c, u):
    """Split unit ``u`` (this CTA's: u < ``spec.n_units``) -> ``opart`` rows u
    8 .., ``pstat`` (m, l) of its 8 q heads, then ``srdy[u]``. Polls nothing;
    ends with ``publish``'s barrier (the LDS hand-over)."""
    tid, lane = c["tid"], c["lane"]
    a, scr = c["args"], c["scr"]
    sl = c["lds"]["split"]
    key = key_of(c)
    window, n = key.window, key.nsplit()
    u = fx.Int32(u)
    t = u // (2 * n)
    kvh = (u // n) % KV_HEADS
    p = u % n
    wave = uniform(c["wave"])  # the page loop's count is wave-uniform
    j = lane % LANE_HEADS
    g = lane // LANE_HEADS
    # FIRST: the pad word, the length and q, all in flight before any wait
    rocdl.sched_barrier(0)
    pre = entry_loads(c, t, kvh)
    rocdl.sched_barrier(0)
    L = row_len(c, t, pre)
    lo, pa, pe = pages(L, window, n, p)
    nw = (fx.max(pe - pa - wave, 0) + (WAVES - 1)) // WAVES  # this wave's pages
    base = pa + wave
    bid = _bid_chunk(c, t, pa, pe, wave, 0)
    kv_last = fx.Int32(a["kv_blocks"]) - 1
    live_q = j < Q_PER_KV
    qa = [
        fx.Vector.from_elements(
            [live_q.select(q[i], fx.Int32(0)) for i in range(4)], fx.Int32
        ).bitcast(fx.BFloat16)
        for q in pre["q"]
    ]
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    m = fx.Float32(NEG)
    den = fx.Float32(0.0)
    acc = [zero4, zero4, zero4, zero4]
    # pages two at a time, two in flight: page i + 1's loads go out before
    # page i's math, page i + 2's before page i + 1's. Unrolled, so the one
    # buffer the back-edge carries (``ca``, page i + 2) keeps its registers
    # (a rotated pair cur = fol is copied there, which waits for the loads).
    # A page past the wave's last reloads the last one (in bounds, unused).
    ca = _page_loads(c, kvh, _block(bid, 0, kv_last))
    i = fx.Int32(0)
    while i < nw:
        i1 = i + 1
        i2 = i + 2
        cb = _page_loads(c, kvh, _block(bid, fx.min(i1, nw - 1), kv_last))
        rocdl.sched_barrier(0)
        m, den, acc = _attend(c, ca, qa, i, base, lo, L, i < nw, m, den, acc)
        # i is even: pages i, i + 1 share a chunk of ids; i + 2 may start one.
        # The id is read in both branches: only a reload waits for its load
        # (a read after the join would wait there on every path, draining
        # page i + 1's loads). (Defined first: the branches yield it.)
        blk2 = fx.Int32(0)
        if (i2 % BID_LANES == 0) & (i2 < nw):
            bid = _bid_chunk(c, t, pa, pe, wave, i2 // BID_LANES)
            blk2 = _block(bid, i2, kv_last)
        else:
            blk2 = _block(bid, fx.min(i2, nw - 1), kv_last)
        ca = _page_loads(c, kvh, blk2)
        rocdl.sched_barrier(0)
        m, den, acc = _attend(c, cb, qa, i1, base, lo, L, i1 < nw, m, den, acc)
        i = i2
    den = den + xshfl(den, 16)
    den = den + xshfl(den, 32)
    # the waves' (m, l, o) -> LDS: lanes g == 0 the per-head m and l, lanes
    # j < 8 their head's dims 16 g .. + 16 (4 x 16 B)
    if g == 0:
        fx.ptr_store(m, sl["ms"] + (c["wave"] * LANE_HEADS + j))
        fx.ptr_store(den, sl["ls"] + (c["wave"] * LANE_HEADS + j))
    if live_q:
        row = sl["os"] + ((c["wave"] * Q_PER_KV + j) * HEAD_DIM + 16 * g)
        for e in range_constexpr(4):
            fx.ptr_store(
                fx.Vector.from_elements([acc[dt][e] for dt in range(4)], fx.Float32),
                row + 4 * e,
            )
    gpu.barrier()
    # thread (head jj, dim d): the 8 waves in wave order
    jj = tid // HEAD_DIM
    d = tid % HEAD_DIM
    mw = [fx.ptr_load(sl["ms"] + (w * LANE_HEADS + jj)) for w in range(WAVES)]
    big = mw[0]
    for w in range_constexpr(1, WAVES):
        big = fx.max(big, mw[w])
    lsum = fx.Float32(0.0)
    osum = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        wt = hw_exp2(mw[w] - big)
        lsum = lsum + fx.ptr_load(sl["ls"] + (w * LANE_HEADS + jj)) * wt
        osum = osum + fx.ptr_load(sl["os"] + ((w * Q_PER_KV + jj) * HEAD_DIM + d)) * wt
    bo.buffer_store(osum, rsrc(scr[OPART]), opart_word(u, jj, d), cache_modifier=CM_DEV)
    if d == 0:
        bo.buffer_store(
            fx.Vector.from_elements([big, lsum], fx.Float32),
            rsrc(scr[PSTAT]),
            pstat_word(u, jj, 0),
            cache_modifier=CM_DEV,
        )
    publish(c["put"], scr["srdy"], srdy_pair(u), 1, tid == 0)


@traced
def load_core(c):
    """The combine's gated rows -> LDS ``xl`` (back's padded rows), after
    every (token, q head)'s ``crdy``: thread tid < 16 S polls pair tid (a
    fresh wave-uniform base: a combine CTA has just put to the region), then
    S x 128 dwordx4 at device scope; a pad row as zeros (``select``). Ends
    with the hand-over barrier."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    xl = c["lds"]["back"]["xl"]
    if tid < HEADS * s:
        crdy = uniform_addr(c["scr"]["crdy"])
        c["poll"]([(crdy, crdy_pair(tid // HEADS, tid % HEADS), 1)])
    gpu.barrier()
    words = s * OKW
    for r in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * r, words - 1)
        v = fx.Vector(ld_dev(c["scr"][XO], e * 4, 4))
        t = e // OKW
        k = e % OKW * 8
        dead = ld_pad(a["pad"], t)
        v = fx.Vector.from_elements(
            [dead.select(fx.Int32(0), v[q]) for q in range(4)], fx.Int32
        )
        if tid + THREADS * r < words:
            fx.ptr_store(v, xl + (t * XL_ROW + k) // 2)
    gpu.barrier()
