# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``front``: one attention layer (the counts below are TP8's) from its
normed input row to the rotated query and the step's KV rows in the
sliding-window cache.

One launch a layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, S rows.
Every CTA issues its weight loads (its wqkv part and its wq_b tasks) at the
kernel start, before it waits on anything; the stages:

    wqkv_a (128 row tiles x 2 K halves, 16 rows; a CTA each): the input's MXFP8
           half (X8, from the seam's norm) a token tile at a time (the next
           tile's rows in flight) times wqkv's -> fp32 partials -> QKVP
    qkv    (S): the halves summed -> bf16; q_norm -> bf16 -> MXFP8 -> QX8
           (published, a flag a token); kv_norm -> bf16 -> GPT-J RoPE -> bf16
           -> the fp8_ds_mla record at the token's slot
    wq_b   (H x 512 / 16, 16 rows): QX8 times wq_b -> bf16 -> GPT-J RoPE of
           each head's dims 448.. -> q (global, bf16)
    kt     (S, run by K2 after vLLM's indexer): the token's attention keys --
           top-k compressed slots through the kv source's block table (or,
           ``global_keys``, every compressed slot of vLLM's c128a table), then
           window slots -> KT

Each elementwise step follows vLLM's ROCm path (``mxfp8_quantize``,
``FusedQKVRMSNorm``, ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``);
the GEMV reductions differ from the Triton GEMMs' only in their order.
Adapted from ATOM's V4.1 ``attn_pre`` (K1), less its mHC seam.
"""

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..dims import (
    HALF,
    HEAD_DIM,
    HIDDEN,
    KV_DIM,
    NOPE,
    NORM_EPS,
    Q_RANK,
    QKV_ROWS,
    ROPE,
    ROWS,
    WINDOW,
    WQKV_KPARTS,
    WQKV_TASKS,
    X_GROUPS,
    X_WORDS,
    Dims,
    ld,
    tile_rows,
    token_tiles,
)
from .device import (
    BLOCKS,
    CM_DEV,
    THREADS,
    Mailbox,
    bf16_round,
    butterfly,
    ceil_exp,
    clamp_fp8,
    div_rn,
    first_task,
    fp8_pack4,
    fresh,
    gstore,
    hw_rsq,
    ld_f32,
    ld_i32,
    mx_code,
    mx_mul,
    rsrc,
    stamp,
    traced,
    xshfl,
)
from .gemv import RED, gemv_loads, gemv_mfmas, lds_row, row_sum, rows_loads, rows_store
from .plan import DATA, front_scratch

_STREAM = fx.Stream(None)
QX_WORDS = Q_RANK // 4
KSTEPS = HIDDEN // 128 // WQKV_KPARTS  # a wqkv part's 128-steps
KWORDS = KSTEPS * 32
KGROUPS = KSTEPS * 4
XH = HIDDEN // WQKV_KPARTS  # a K half's values
XH_COLS = XH // 8 // 2  # 16 B items (8 values) a half of a K half
XH_RPR = THREADS // XH_COLS  # half-rows a round
XQ_ITEMS = XH // 8  # an xq unit's 16 B items (one a thread)
FRONT_POINTS = 6  # start, kt done, wqkv done, qkv done, wq_b done, (spare)


@dataclass(frozen=True)
class FrontBuild:
    tokens: int
    tp: int
    ratio: int  # the layer's compress ratio (0: no top-k keys)
    timeline: bool = False
    topk: int = 0  # a token's compressed keys at most (``Dims.topk``)


def plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


@traced
def publish(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))


@traced
def await_flags(c, region, n, index):
    """Thread u < n polls flag ``index(u)``; then the CTA proceeds."""
    tid = c["tid"]
    if tid < n:
        c["mb"].poll([(region, index(fx.min(tid, n - 1)), 1)])
    gpu.barrier()


# ---------------------------------------------------------------- kt


@traced
def _kt_round(c, t, k, ntopk, kv_len, width, req, cm):
    """``stage_kt``'s key k of token t."""
    a, ratio, d = c["args"], c["ratio"], c["d"]
    sidx = fx.min(fx.max(k - ntopk, 0), WINDOW - 1)
    s_slot = ld_i32(a["swa_idx"], t * WINDOW + sidx)
    val = (k < kv_len).select(s_slot, fx.Int32(-1))
    if const_expr(ratio > 0):
        local = ld_i32(a["topk"], t * width + fx.min(k, width - 1))
        if const_expr(c.get("global_keys", False)):
            c_slot = local | fx.Int32(-2147483648)
        else:
            lc = fx.max(local, 0)
            e = a["comp_block"]
            blk = ld_i32(a["comp_bt"], req * a["bt_stride"] + lc // e)
            c_slot = (blk * e + lc % e) | fx.Int32(-2147483648)
        c_val = (local >= 0).select(c_slot, fx.Int32(-1))
        val = (k < ntopk).select(c_val, val)
    if k < d.keys:
        bo.buffer_store(val, rsrc(a["kt"]), t * d.keys + k, cache_modifier=cm)


@traced
def stage_kt(c, t):
    """Token t's keys for K2: ``KT[t][k] = slot | 1 << 31`` for its top-k compressed
    rows (k < ntopk), the window slot for k in [ntopk, kv_len), -1 for an
    invalid or absent key; KLEN[t] = kv_len (0 for a pad row), KLEN[S + t] =
    ntopk. The top-k table holds request-local compressed indices (through the
    block table) or, ``global_keys``, the slots themselves (a row of the
    runtime width ``bt_stride``: vLLM's every compressed key of a C128 layer)."""
    tid = c["tid"]
    a, ratio = c["args"], c["ratio"]
    d, glob = c["d"], c.get("global_keys", False)
    width = a["bt_stride"] if glob else fx.Int32(d.topk_width)
    # device scope where this launch reads the table (K2's)
    cm = CM_DEV if const_expr(c.get("dev", False)) else 0
    slot = ld_i32(a["slot"], 2 * t)
    live = slot >= 0
    nswa = live.select(ld_i32(a["swa_lens"], t), fx.Int32(0))
    if const_expr(ratio > 0):
        pos = ld_i32(a["pos"], 2 * t)
        ntopk = live.select(fx.min((pos + 1) // ratio, width), fx.Int32(0))
        req = ld_i32(a["t2r"], t)
    else:
        ntopk = fx.Int32(0)
        req = 0
    kv_len = ntopk + nswa
    if const_expr(glob):
        # a wide table: the rounds over the token's keys only (the split's
        # chunks past them skipped, ``attention.back.stage_split``)
        for i in range(0, (kv_len + THREADS - 1) // THREADS):
            _kt_round(c, t, fresh(tid) + THREADS * i, ntopk, kv_len, width, req, cm)
    else:
        for i in range_constexpr((d.keys + THREADS - 1) // THREADS):
            _kt_round(c, t, fresh(tid) + THREADS * i, ntopk, kv_len, width, req, cm)
    if tid == 0:
        bo.buffer_store(kv_len, rsrc(a["klen"]), t, cache_modifier=cm)
        bo.buffer_store(ntopk, rsrc(a["klen"]), c["S"] + t, cache_modifier=cm)


# ---------------------------------------------------------------- wqkv_a


@traced
def stage_wqkv(c, task, part, ops, t0, n, live):
    """Rows 16 task .. of wqkv over K half ``part`` against tokens t0 .. t0 + n
    -> fp32 partials QKVP[part] (``live``: a CTA with a wqkv part)."""
    tid, red = c["tid"], c["red"]
    gemv_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, ops, rows=n, nsteps=KSTEPS)
    gpu.barrier()
    if (tid < ROWS * n) & live:
        r = tid % ROWS
        tl = tid // ROWS
        t = plus(t0, tl)
        v = row_sum(red, r, tl)
        c["mb"].put(c["qkvp"], (part * c["S"] + t) * QKV_ROWS + task * ROWS + r, v)
    gpu.barrier()


# ---------------------------------------------------------------- q / kv norms


@traced
def stage_qkv(c, t):
    """Token t: the K halves summed -> bf16; q_norm -> bf16 -> MXFP8 (QX8);
    kv_norm -> bf16 -> RoPE -> bf16 -> the fp8_ds_mla record at
    slot_mapping[t]. 256 threads (8 q columns and 2 kv columns each); threads
    256.. mirror thread 255 and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a, s = c["args"], c["S"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # independent loads first: they overlap the partials' wait
    qw = [c["ldbf"](a["qn_w"], qc[j]) for j in range(8)]
    kw = [c["ldbf"](a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    slot = ld_i32(a["slot"], 2 * t)
    jr = tt - NOPE // 2
    is_rope = jr >= 0
    jj = fx.max(jr, 0)
    cs = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + jj), fx.Float32(1.0))
    sn = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + HALF + jj), fx.Float32(0.0))
    specs = []
    for p in range_constexpr(2):
        specs += [(c["qkvp"], (p * s + t) * QKV_ROWS + qc[j], 1) for j in range(8)]
        specs += [
            (c["qkvp"], (p * s + t) * QKV_ROWS + Q_RANK + tt * 2 + j, 1)
            for j in range(2)
        ]
    got = c["mb"].poll(specs)
    vals = [
        bf16_round(got[j][0].bitcast(fx.Float32) + got[10 + j][0].bitcast(fx.Float32))
        for j in range(10)
    ]
    xq = [((tt * 8 + j) < Q_RANK).select(vals[j], fx.Float32(0.0)) for j in range(8)]
    xk = vals[8:10]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly(ssq, (32, 16, 8, 4, 2, 1))
    ssk = butterfly(xk[0] * xk[0] + xk[1] * xk[1], (32, 16, 8, 4, 2, 1))
    if (lane == 0) & mine:
        fx.ptr_store(ssq, red + wave)
        fx.ptr_store(ssk, red + 4 + wave)
    gpu.barrier()
    tq = (fx.ptr_load(red + 0) + fx.ptr_load(red + 1)) + (
        fx.ptr_load(red + 2) + fx.ptr_load(red + 3)
    )
    tk = (fx.ptr_load(red + 4) + fx.ptr_load(red + 5)) + (
        fx.ptr_load(red + 6) + fx.ptr_load(red + 7)
    )
    rq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(NORM_EPS)
    )
    rk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(NORM_EPS)
    )
    # q: bf16 normed row, then vLLM's MXFP8 (a 32-group = 4 threads)
    ys = [bf16_round((xq[j] * rq) * qw[j]) for j in range(8)]
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = mx_code(butterfly(amax, (1, 2), fx.max))
    mul = mx_mul(code)
    # the bf16 qr, vLLM's indexer's input
    if mine & (tt * 8 < Q_RANK):
        words = [
            fx.Vector.from_elements([ys[2 * i], ys[2 * i + 1]], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
            for i in range(4)
        ]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(a["qr_out"]),
            t * (Q_RANK // 2) + 4 * tt,
        )
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
        w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["qx8"]),
            t * QX_WORDS + 2 * tt,
            cache_modifier=CM_DEV,
        )
        if tt % 4 == 0:
            bo.buffer_store(
                code,
                rsrc(c["qx8s"]),
                t * (Q_RANK // 32) + tt // 4,
                cache_modifier=CM_DEV,
            )
    # kv: this thread's pair (2 tt, 2 tt + 1)
    kn = [bf16_round((xk[j] * rk) * kw[j]) for j in range(2)]
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    # record quant of the nope dims: a 64-block = 32 threads (half a wave)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8, 16), fx.max)
    rcode = ceil_exp(
        div_rn(
            fx.max(amax, fx.Float32(1e-4)), fx.Float32(448.0), fx.Float32(1.0 / 448.0)
        )
    )
    inv = mx_mul(rcode)
    q0 = clamp_fp8(e0 * inv)
    q1 = clamp_fp8(e1 * inv)
    n0 = xshfl(q0, 1)
    n1 = xshfl(q1, 1)
    word = fp8_pack4(q0, q1, n0, n1)
    rope_word = (
        fx.Vector.from_elements([e0, e1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    blk = a["swa_block"]
    sblock = slot // blk
    soff = slot % blk
    base = a["swa"] + fx.Int64(sblock) * fx.Int64(a["swa_stride"])
    rec = base + fx.Int64(soff * DATA)
    if mine & (tt % 32 == 0) & (tt < NOPE // 2):
        fx.ptr_store(rcode.bitcast(fx.Float32), red + 8 + tt // 32)
    gpu.barrier()
    if mine & (slot >= 0):
        if tt < NOPE // 2:
            if tt % 2 == 0:
                gstore(rec + fx.Int64(2 * tt), word, words=1)
        else:
            gstore(rec + fx.Int64(NOPE + 4 * (tt - NOPE // 2)), rope_word, words=1)
        if tt == 0:
            sb = [fx.ptr_load(red + 8 + b).bitcast(fx.Int32) for b in range(7)]
            w0 = sb[0] | (sb[1] << 8) | (sb[2] << 16) | (sb[3] << 24)
            w1 = sb[4] | (sb[5] << 8) | (sb[6] << 16)
            sc = base + fx.Int64(blk * DATA + soff * 8)
            gstore(sc, fx.Vector.from_elements([w0, w1], fx.Int32), words=2)
    publish(c, c["qrdy"], t)


# ---------------------------------------------------------------- wq_b


def wqb_rope(c, task0, pos_t, n):
    """(cos, sin) of this thread's pair in ``stage_wqb``'s epilogue over a tile
    of n rows (``pos_t``: its token's position; every task of the CTA shares
    the pair's dims: tasks BLOCKS apart are whole heads apart)."""
    a = c["args"]
    rp = c["tid"] % (ROWS // 2)
    dim = (task0 * ROWS) % HEAD_DIM + 2 * rp
    jj = fx.max((dim - NOPE) >> 1, 0)
    return ld_f32(a["cos_sin"], pos_t * ROPE + jj), ld_f32(
        a["cos_sin"], pos_t * ROPE + HALF + jj
    )


def wqb_pos(c, t0, n):
    """This thread's token position for ``stage_wqb``'s epilogue of tile
    t0 .. t0 + n."""
    per = ROWS * n // 2
    tl = fx.min((c["tid"] % per) // (ROWS // 2), n - 1)
    return ld_i32(c["args"]["pos"], 2 * (t0 + tl))


@traced
def stage_wqb(c, task0, bops, t0, n, rope):
    """Rows 16 (task0 + BLOCKS k) .. of the rank's H x 512 query rows, every
    task k of the CTA: GEMVs -> bf16 -> GPT-J RoPE of a head's dims 448..
    (fp32) -> bf16 -> q."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    nt = len(bops)
    for k in range_constexpr(nt):
        gemv_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red + k * RED, bops[k], rows=n)
    gpu.barrier()
    per = ROWS * n // 2  # bf16 pairs a task
    if tid < nt * per:
        k = tid // per
        u = tid % per
        rp = u % (ROWS // 2)
        tl = u // (ROWS // 2)
        t = plus(t0, tl)
        r0 = (task0 + BLOCKS * k) * ROWS + 2 * rp
        rk = red + k * RED
        v0 = bf16_round(row_sum(rk, 2 * rp, tl))
        v1 = bf16_round(row_sum(rk, 2 * rp + 1, tl))
        is_rope = (task0 * ROWS) % HEAD_DIM + 2 * rp >= NOPE
        cs, sn = rope
        e0 = is_rope.select(v0 * cs - v1 * sn, v0)
        e1 = is_rope.select(v0 * sn + v1 * cs, v1)
        word = (
            fx.Vector.from_elements([e0, e1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(word, rsrc(a["q"]), (t * d.q_rows + r0) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def front_lds(s: int, d: Dims):
    """K1's LDS (the attention front's stages)."""
    rows = tile_rows(s)
    wqb_per_cta = d.wqb_tasks // BLOCKS

    @fx.struct
    class FrontLds:
        red: fx.Array[fx.Float32, wqb_per_cta * RED + 16, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        xl: fx.Array[fx.Int32, rows * lds_row(KWORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(KGROUPS), 16]

    return FrontLds


def front_context(key, lds, args: dict, scratch, tag, layout=None) -> dict:
    """The front stages' context: LDS views, mailbox (tag ``tag``), scratch
    regions, arguments (``args``: the kernel's, by name)."""
    s, d = key.tokens, Dims(key.tp, key.topk)
    tid = fx.thread_idx.x
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "xl": lds.xl.ptr,
        "xsl": lds.xsl.ptr,
        "qxl": lds.qxl.ptr,
        "qxsl": lds.qxsl.ptr,
        "mb": Mailbox(tag),
        "d": d,
        "ratio": key.ratio,
        "ldbf": lambda ptr, i: fx.Float32(
            fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
        ),
        "args": args,
    }
    for region, (off, _) in (layout or front_scratch(s, d)).items():
        c[region] = scratch + fx.Int64(off)
    return c


@traced
def run_front(c, key, bid, before_wqkv=None, after_wqkv=None):
    """The front's stages on this CTA, from the input's MXFP8 X8 / XRDY that a
    stage before them (the mono layer's seam norm) publishes for every token.
    ``before_wqkv()`` runs ahead of the wqkv weights' loads (a latency-critical
    stage of the caller's), ``after_wqkv()`` right after. The step's key table
    is K2's (``stage_kt``): an index source's top-k comes after this launch."""
    s = key.tokens
    d = c["d"]
    wqb_tasks = d.wqb_tasks
    a = c["args"]
    tiles = token_tiles(s)
    wqb_per_cta = wqb_tasks // BLOCKS
    assert wqb_tasks % BLOCKS == 0
    n_wqkv = WQKV_TASKS * WQKV_KPARTS  # 256: every CTA's
    assert n_wqkv == BLOCKS
    QKV0 = 0  # on wqkv CTAs
    WQB0 = 0
    stamp(c, 0)

    # ---- Loads complete in issue order, so every CTA issues its latency-
    # critical loads before its later stages' weights: its wqkv part, its
    # first input tile, the next tiles while it multiplies, then its wq_b
    # tasks. The q / kv norms run on wqkv CTAs, their partials' wait behind
    # the wq_b weights that are due by then anyway.
    wqkv_unit = first_task(bid, 0)
    has_wqkv = wqkv_unit < n_wqkv
    task = fx.min(wqkv_unit, n_wqkv - 1) % WQKV_TASKS
    part = fx.min(wqkv_unit, n_wqkv - 1) // WQKV_TASKS
    wqb_task = first_task(bid, WQB0)
    if const_expr(before_wqkv is not None):
        before_wqkv()
    kops = gemv_loads(
        c,
        a["wqkv"],
        a["wqkv_s"],
        HIDDEN,
        task * ROWS,
        kbeg=part * KSTEPS,
        nsteps=KSTEPS,
        pred=has_wqkv,
        shuffled=True,
    )
    if const_expr(after_wqkv is not None):
        after_wqkv()
    stamp(c, 1)
    poss = [wqb_pos(c, t0, n) for t0, n in tiles]

    # ---- wqkv_a: one (row tile, K half) a CTA, every token tile; the wq_b
    # weights issued behind the input rows' loads
    def wqb_weights():
        return [
            gemv_loads(
                c,
                a["wqb"],
                a["wqb_s"],
                Q_RANK,
                (wqb_task + BLOCKS * i) * ROWS,
                ld=ld(Q_RANK),
            )
            for i in range(wqb_per_cta)
        ]

    await_flags(c, c["xrdy"], s, lambda u: WQKV_KPARTS * u + part)
    xr = (
        c["x8"],
        c["x8s"],
        X_WORDS,
        part * KWORDS,
        KWORDS,
        X_GROUPS,
        part * KGROUPS,
        KGROUPS,
    )
    regs = rows_loads(c, *xr, tiles[0][0], tiles[0][1], pred=has_wqkv)
    if const_expr(len(tiles) == 1):
        bops = wqb_weights()
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        rows_store(c, regs, KWORDS, KGROUPS, c["xl"], c["xsl"], n)
        if const_expr(ti + 1 < len(tiles)):
            t1, n1 = tiles[ti + 1]
            regs = rows_loads(c, *xr, t1, n1, pred=has_wqkv)
        if const_expr(ti + 2 == len(tiles)):
            bops = wqb_weights()
        stage_wqkv(c, task, part, kops, t0, n, has_wqkv)
    stamp(c, 2)

    # ---- q / kv norms, KV insert
    for t in range(first_task(bid, QKV0), s, BLOCKS):
        stage_qkv(c, t)
    stamp(c, 3)

    # ---- wq_b: every task's rows of a token tile at a time, the next
    # tile's rows in flight
    ropes = [wqb_rope(c, wqb_task, poss[ti], tiles[ti][1]) for ti in range(len(tiles))]
    await_flags(c, c["qrdy"], s, lambda u: u)
    qx = (
        c["qx8"],
        c["qx8s"],
        QX_WORDS,
        0,
        QX_WORDS,
        Q_RANK // 32,
        0,
        Q_RANK // 32,
    )
    regs = rows_loads(c, *qx, *tiles[0])
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        rows_store(c, regs, QX_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"], n)
        if const_expr(ti + 1 < len(tiles)):
            regs = rows_loads(c, *qx, *tiles[ti + 1])
        stage_wqb(c, wqb_task, bops, t0, n, ropes[ti])
    stamp(c, 4)
