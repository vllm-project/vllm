# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's 5 layers and final norm as one launch for 9..32 requests.

A draft step of up to `7 max_reqs` query rows on one TP4 rank (`WideRLayout`),
the work the multi-launch path (launches of <= 8 requests on `wide` / `layer`)
does in several launches, each of which pays the whole chain of dependent stages
again. Per layer:

- A CTAs (192..255, `j = bid - 192`) own row `4 j + rank` (its residual stays in
  LDS): input norm -> MXFP8 row in this rank's buffer + a row flag. Every CTA b
  then copies gathered row b (flag poll, one read) into scratch, in the MFMA
  operand order of the GEMV loads (`pk_off`: lane-contiguous 1 KB blocks; the
  row-strided layout ran ~4x slower), and flags it.
- G CTAs (0..191): QKV over 16-row token tiles (activations of the next tile in
  flight, one LDS reduction barrier a tile, outputs staged in LDS and stored at
  the end: a store in the tile loop stalls the next tile's loads); the same wave
  split and order as `wide`, so a row's result is bitwise `wide`'s.
- Attention: G CTA b runs partition slot b (its first two K / V blocks loaded at
  the layer's start). `split="chunk"`: request r of chunk `r // 8` (k
  requests in it) gets `48 // k` partitions (k >= 3, as `wide` gives it; 6
  for a 1- or 2-request tail), so every request of a chunk of 3..8 sees the block
  ranges and merge order of `wide`; `split="even"`: `min(16, 192 // n)`
  each. Merge tasks (row, 4 heads) round-robin over all 256 CTAs.
- o_proj, gate / up (act quantized once to MXFP8: CTA pairs share their rows'
  amax), down: as QKV. Reduce-scatter by push, payload + flag: each G CTA stores
  its bf16 partials in the row owners' buffers, waits for the acknowledgements,
  then stores its flag in every rank's flag array; owners poll local flags and
  sum the 4 ranks' partials in rank order (bitwise `wide`'s tagged sum).
- After the last layer the owners' final bf16 rows are gathered into `h_out`.

Polling stays local or touches 8-byte flags: tagged rows polled across XGMI by
every CTA flooded the links (2-4x slower stages).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.minimax_m3.amd.mono.drafter import attention as att
from vllm.models.minimax_m3.amd.mono.drafter import device as dev
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.drafter.device import F32, I32, I64
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    CM_SYS,
    butterfly,
    fp8_pack4,
    kernel_symbol,
    memrealtime,
    traced,
    uniform,
    xshfl,
)

A0 = layout.NG  # A CTAs 192..255
CHUNK = 8  # requests a chunk of the multi-launch path
CH_PARTS = layout.W_NA  # partition slots a chunk (48)
# LDS (words): attention tiles, merge staging; then G: the GEMV reductions + the
# stage's outputs, A: the owned row's residual (3072 words) + 8 floats
RES0 = 5 * att.TILE_WORDS + 512
G_RED = RES0
G_STAGE = G_RED + 8192
POOL_WORDS = G_STAGE + 7168 + 16
NEG_INF = float("-inf")
_STREAM = fx.Stream(None)
TL_SLOTS = 128  # timeline build: int64 stamps a CTA (64..123: layer 1's partition)
# token tiles a GEMV issues ahead of the one it computes (activation VGPRs a tile:
# QKV / gate-up 54, o_proj 18, down 27); gate / up 1 spills (~140 VGPRs)
D_QKV, D_O, D_GU, D_DOWN = 1, 1, 0, 1


class WideRLayout:
    """Scratch, flag and symmetric regions of the wide-R kernel for up to
    `max_reqs` requests."""

    def __init__(self, max_reqs: int):
        if not 9 <= max_reqs <= 32:
            raise ValueError(f"wide-R max_reqs {max_reqs}: 9..32")
        mr = layout.QPR * max_reqs
        self.max_reqs, self.mr = max_reqs, mr
        self.nch = (max_reqs + CHUNK - 1) // CHUNK
        # partition slots: 48 a chunk ("chunk" split) or one a G CTA ("even")
        self.nparts = max(CH_PARTS * self.nch, layout.NG)
        # GEMV activations in the MFMA operand order (``pk_off`` / ``pk_soff``):
        # 16-row tiles, a wave's chunk as 2 x 1 KB lane-contiguous blocks
        mt = (mr + 15) // 16
        self.l_off, self.layer_bytes = layout._pack(
            [
                ("x8", mt * 16 * 6144),
                ("xs", mt * PK_S_TILE),
                ("qkv", mr * layout.QKV_N * 2),
                ("ap", self.nparts * layout.AP_STRIDE),
                ("a8", mt * 16 * layout.QS),
                ("as", mt * PK_S_TILE),
                ("h8", mt * 16 * 6144),
                ("hs", mt * PK_S_TILE),
                ("act", mt * 16 * layout.IR),  # MXFP8, packed (K 3072)
                ("acs", mt * PK_S_TILE),
                ("am", layout.NG * mt * 16 * 4),  # a G CTA's act row amax
            ]
        )
        self.f_off, self.scratch_bytes = layout._pack(
            [
                (name, count * 8)
                for name, count in (
                    ("x8f", mr),
                    ("qkvf", layout.NG),
                    ("apf", self.nparts),
                    ("attf", 4 * mr),
                    ("h8f", mr),
                    ("actf", layout.NG),
                    ("amf", layout.NG),
                )
            ],
            self.layer_bytes * layout.LAYERS,
        )
        # reduce-scatter: G pushes untagged bf16 partials to the row's owner rank,
        # [src rank][owned row][6144], then its flag to every rank's [src rank][G CTA]
        # all-gathers, payload + flag: the owner's untagged x8 rows [row][1584 words]
        # (e4m3, then 48 e8m0 words) and bf16 final rows [row][3072 words] in its own
        # buffer, a tagged flag a row; readers poll the 8-byte flag, then read once
        agx = mr * AGW * 4
        self.own = (mr + 3) // 4
        part = 4 * self.own * layout.HIDDEN * 2
        self.y_po, self.y_pd = 0, part
        # flags [src rank][G CTA]: a G CTA stores its flag in every rank's array, so
        # owners poll local memory and read only the payload across XGMI
        self.y_fo = 2 * part
        self.y_fd = self.y_fo + 4 * layout.NG * 8
        self.y_agx = self.y_fd + 4 * layout.NG * 8
        self.y_agh = self.y_agx + agx
        self.y_agf = self.y_agh + agx
        self.y_agxf = self.y_agf + mr * layout.ROW_PAIRS * 4
        self.y_aghf = self.y_agxf + mr * 8
        self.y_agff = self.y_aghf + mr * 8
        self.sym_bytes = self.y_agff + mr * 8


@traced
def wait_flags_n(base, n, tag, tid, dl, health, rank, site):
    """`dev.wait_flags` for n <= 1024 flags (two flags a thread)."""
    last = fx.max(I32(n) - 1, 0)
    _, fl = dev.poll_pairs(base, fx.min(tid, last), 1, tag, dl, health)
    dev.fail(health, fl, rank, site)
    if n > 512:
        _, fl2 = dev.poll_pairs(base, fx.min(tid + 512, last), 1, tag, dl, health)
        dev.fail(health, fl2, rank, site)
    gpu.barrier()
    dev.fence_acquire()


@traced
def wait_flags4(sym, Yf, tag, tid, dl, health, rank, site):
    """Wait for the 4 x 192 G flags at `Yf` of this rank's symmetric buffer
    (thread i < 192: each source rank's flag i), then a CTA barrier."""
    tag = I32(tag)
    i = fx.min(tid, layout.NG - 1)

    def load_all():
        words = []
        for src in range_constexpr(4):
            v = dev.ld(sym, Yf // 4 + (src * layout.NG + i) * 2, 2, CM_SYS)
            words += [v[0], v[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        return (v[1] != tag) | (v[3] != tag) | (v[5] != tag) | (v[7] != tag)

    _, fl = dev._spin(load_all, pending, dl, health)
    dev.fail(health, fl, rank, site)
    gpu.barrier()
    dev.fence_acquire()


def rs_read(sym, own, Yp, k, tid):
    """Owned row k: the 4 source ranks' bf16 partials pushed to `Yp` (read after
    `wait_flags4`), summed in rank order (bf16 each step, as `dev.row_sum4`),
    16 values a thread."""
    got = []
    for src in range_constexpr(4):
        idx = (Yp // 2 + (src * own + k) * layout.HIDDEN + fx.min(tid, 383) * 16) // 2
        a = dev.ld(sym, idx, 4, CM_SYS)
        b = dev.ld(sym, idx + 4, 4, CM_SYS)
        got.append([a[e] for e in range(4)] + [b[e] for e in range(4)])
    vals = []
    for k in range_constexpr(8):
        lo = 0.0 + dev.bf_lo(got[0][k])
        hi = 0.0 + dev.bf_hi(got[0][k])
        for r in range_constexpr(1, 4):
            lo = lo + dev.bf_lo(got[r][k])
            hi = hi + dev.bf_hi(got[r][k])
        vals += [dev.bf16r(lo), dev.bf16r(hi)]
    return vals


# ---------------------------------------------------------------- packed tiles
# MXFP8 [rows][K]: byte (t, j) of row t lives at tile t // 16, wave w = j // (K / 8),
# chunk c, 16-byte unit u = g + 4 h of the chunk, as lane (g, r = t % 16)'s operand
# half h: [tile][w][c][h][64 lanes][16 B]. Its e8m0 scale of 32-k block (c, g):
# [tile][w][64 lanes][8 B] byte c. bf16 act [rows][3072]: 16-element units the same
# way, a unit's two 16-byte halves q as separate 1 KB blocks:
# [tile][w][c][h][q][64 lanes][16 B].
PK_S_TILE = 8 * 64 * 8


def pk_off(t, j, K):
    """Byte offset of MXFP8 byte j (a multiple of 4 within a 16-byte unit is fine)
    of row t."""
    kw = K // 8
    nc = kw // 128
    w = j // kw
    jj = j % kw
    u = (jj % 128) // 16
    lane = (u % 4) * 16 + t % 16
    return (((((t // 16) * 8 + w) * nc + jj // 128) * 2 + u // 4) * 64 + lane) * 16 + (
        j % 16
    )


def pk_soff(t, sidx, K):
    """Byte offset of row t's scale of 32-k block sidx."""
    bw = K // 256  # blocks a wave
    w = sidx // bw
    within = sidx % bw
    lane = (within % 4) * 16 + t % 16
    return (((t // 16) * 8 + w) * 64 + lane) * 8 + within // 4


def ld_pk(x, s, t, nc, wave, lane):
    """Tile t's MFMA operands of `wave`: nc chunks of (8 words, scale)."""
    sw = dev.ld(s, ((t * 8 + wave) * 64 + lane) * 2, 2)
    out = []
    for c in range_constexpr(nc):
        off = ((t * 8 + wave) * nc + c) * 2048 + lane * 16
        a = dev.ld(x, off // 4, 4)
        b = dev.ld(x, (off + 1024) // 4, 4)
        out.append(
            (
                fx.Vector.from_elements(
                    [a[e] for e in range(4)] + [b[e] for e in range(4)], fx.Int32
                ),
                (sw[c // 4] >> (8 * (c % 4))) & 0xFF,
            )
        )
    return out


AGW = 1584  # an x8 row's words: 1536 e4m3 words, 48 e8m0 words


def ag_put_x8(o, t, Y, Yf, tag, sym, tid):
    """Row t (16 values a thread) as MXFP8 into this rank's gathered rows at
    `sym + Y`, then (every store acknowledged) its flag at `Yf`."""
    am = fmath.absf(o[0])
    for e in range_constexpr(1, 16):
        am = fx.max(am, fmath.absf(o[e]))
    am = fx.max(am, xshfl(am, 1))
    sb, rs = dev.mx_scale(am)
    wd = [
        fp8_pack4(
            o[4 * m] * rs, o[4 * m + 1] * rs, o[4 * m + 2] * rs, o[4 * m + 3] * rs
        )
        for m in range(4)
    ]
    s0 = sb.to(fx.Int32)
    s2 = xshfl(s0, 2)
    sw = s0 | (s2 << 8) | (xshfl(s0, 4) << 16) | (xshfl(s2, 4) << 24)
    row = Y // 4 + t * AGW  # words
    if tid < 384:
        dev.st(sym, row + tid * 4, fx.Vector.from_elements(wd, fx.Int32), CM_SYS)
        if tid % 8 == 0:
            dev.st(sym, row + 1536 + tid // 8, sw, CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        dev.st(sym, Yf // 4 + t * 2, dev.pair(0, tag), CM_SYS)


ag_put_x8 = traced(ag_put_x8)


def ag_copy_x8_pk(t, src, Y, Yf, tag, x8, xs8, tid, dl, health, rank, site):
    """Row t from its owner's gathered rows (`src`: the owner's peer base) once its
    flag carries `tag`, into the packed x8 / xs8 (K 6144)."""
    _, fl = dev.poll_pairs(src, Yf // 8 + t, 1, tag, dl, health, CM_SYS)
    dev.fail(health, fl, rank, site)
    gpu.barrier()
    row = Y // 4 + t * AGW
    if tid < 384:
        w = dev.ld(src, row + tid * 4, 4, CM_SYS)
        dev.st(x8, pk_off(t, tid * 16, 6144) // 4, w, CM_DEV)
    if (tid >= 384) & (tid < 432):
        js = tid - 384
        w1 = dev.ld(src, row + 1536 + js, 1, CM_SYS)
        for b in range_constexpr(4):
            dev.st8(xs8, pk_soff(t, 4 * js + b, 6144), (w1 >> (8 * b)) & 0xFF, CM_DEV)


ag_copy_x8_pk = traced(ag_copy_x8_pk)


def ag_put_bf(o, t, Y, Yf, tag, sym, tid):
    """Final row t (16 values a thread) as bf16 into this rank's rows at `sym + Y`,
    then its flag at `Yf`."""
    if tid < 384:
        row = Y // 4 + t * layout.ROW_PAIRS + tid * 8
        wds = [dev.bfpair(o[2 * k], o[2 * k + 1]) for k in range(8)]
        dev.st(sym, row, fx.Vector.from_elements(wds[:4], fx.Int32), CM_SYS)
        dev.st(sym, row + 4, fx.Vector.from_elements(wds[4:], fx.Int32), CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        dev.st(sym, Yf // 4 + t * 2, dev.pair(0, tag), CM_SYS)


ag_put_bf = traced(ag_put_bf)


def ag_copy_bf(t, src, Y, Yf, tag, out, tid, dl, health, rank, site):
    """Final row t from its owner once its flag carries `tag`, into out[t]."""
    _, fl = dev.poll_pairs(src, Yf // 8 + t, 1, tag, dl, health, CM_SYS)
    dev.fail(health, fl, rank, site)
    gpu.barrier()
    if tid < 384:
        row = Y // 4 + t * layout.ROW_PAIRS + tid * 8
        a = dev.ld(src, row, 4, CM_SYS)
        b = dev.ld(src, row + 4, 4, CM_SYS)
        dev.st(out, t * 3072 + tid * 8, a)
        dev.st(out, t * 3072 + tid * 8 + 4, b)


ag_copy_bf = traced(ag_copy_bf)


def merge_task_pk(t, hq, na, ap0, a8, as8, ml, tid, lane, maxp):
    """`attention.merge_task` with the MXFP8 output in the packed layout (K 2048)."""
    j = t % layout.QPR
    c_ = tid // 8
    hs = (tid // 2) % 4
    wl = tid % 2
    v = dev.ld(
        ap0,
        fx.max(fx.min(c_, na - 1), I32(0)) * (layout.AP_STRIDE // 4)
        + wl * 128
        + j * 16
        + hq * 4
        + hs,
        1,
        0,
        T.f32,
    )
    v = (c_ < na).select(v, (wl == 0).select(F32(NEG_INF), F32(0.0)))
    fx.ptr_store(v, ml + tid)
    gpu.barrier()
    h4 = tid // 128
    d = tid % 128
    mrow = j * 16 + hq * 4 + h4
    os_ = [
        dev.ldn(
            ap0,
            na * layout.AP_STRIDE,
            c * (layout.AP_STRIDE // 4) + 256 + mrow * 128 + d,
            1,
            0,
            T.f32,
        )
        for c in range(maxp)
    ]
    M = fx.ptr_load(ml + h4 * 2)
    for c in range_constexpr(1, maxp):
        M = fx.max(M, fx.ptr_load(ml + (c * 8 + h4 * 2)))
    lsum = F32(0.0)
    acc = F32(0.0)
    for c in range_constexpr(maxp):
        mc = fx.ptr_load(ml + (c * 8 + h4 * 2))
        w = (mc == F32(NEG_INF)).select(F32(0.0), dev.exp(mc - M))
        lsum = lsum + fx.ptr_load(ml + (c * 8 + h4 * 2 + 1)) * w
        acc = acc + os_[c] * w
    h = hq * 4 + h4
    out = dev.bf16r((lsum > 0.0).select(acc / lsum, F32(0.0)))
    am = butterfly(fmath.absf(out), (16, 8, 4, 2, 1), fx.max)
    sb, rs = dev.mx_scale(am)
    q = fp8_pack4(out * rs, F32(0.0), F32(0.0), F32(0.0)) & 0xFF
    w1 = q | (xshfl(q, 1) << 8)
    w2 = w1 | (xshfl(w1, 2) << 16)
    col = h * 128 + d
    if d % 4 == 0:
        dev.st(a8, pk_off(t, col, layout.QS) // 4, w2, CM_DEV)
    if d % 32 == 0:
        dev.st8(as8, pk_soff(t, col // 32, layout.QS), sb.to(fx.Int32), CM_DEV)


merge_task_pk = traced(merge_task_pk)


def even_cpr(n_reqs):
    """Partitions a request when every request gets the same number: as many as the
    192 G CTAs take, at most 16 (the merge's static bound)."""
    return fx.min(I32(layout.W_MAXP), I32(layout.NG) // fx.max(n_reqs, I32(1)))


def chunk_cpr(r, n_reqs, split):
    """(partition slot of request r's partition 0, partitions of request r)."""
    if const_expr(split == "even"):
        cpr = even_cpr(n_reqs)
        return r * cpr, cpr
    ch = r // CHUNK
    k = fx.min(I32(CHUNK), n_reqs - ch * CHUNK)
    cpr = (k >= 3).select(I32(CH_PARTS) // fx.max(k, I32(1)), I32(6))
    return ch * CH_PARTS + (r % CHUNK) * cpr, cpr


def build_wide_r(
    yw, lay: WideRLayout, wait_ticks=1_000_000_000, timeline=False, split="chunk"
):
    """The `@flyc.jit` launcher of the wide-R drafter kernel for one TP4 rank.

    `yw`: the byte offset of its symmetric regions in the peer buffer. `split`:
    a request's attention partitions as the multi-launch path splits them
    ("chunk": outputs bitwise equal to it for every request of a chunk of 3..8) or
    as many as the G CTAs take ("even": up to 16 a request, shorter partitions).
    """
    if split not in ("chunk", "even"):
        raise ValueError(f"split {split}")
    SPLIT = split
    WF = lay.f_off
    WL = lay.l_off
    LB = lay.layer_bytes
    Y_PO, Y_PD = yw + lay.y_po, yw + lay.y_pd
    Y_FO, Y_FD = yw + lay.y_fo, yw + lay.y_fd
    Y_AGX, Y_AGH, Y_AGF = yw + lay.y_agx, yw + lay.y_agh, yw + lay.y_agf
    Y_AGXF, Y_AGHF, Y_AGFF = yw + lay.y_agxf, yw + lay.y_aghf, yw + lay.y_agff
    TL = timeline

    @fx.struct
    class Smem:
        pool: fx.Array[fx.Int32, POOL_WORDS, 16]

    @flyc.kernel(
        name=kernel_symbol(
            "minimax_m3_dspark_drafter_wide_r",
            r=lay.max_reqs,
            y=yw,
            tl=int(TL),
            s=split,
        ),
        known_block_size=[layout.THREADS, 1, 1],
    )
    def drafter_wide_r(
        x_in: Int64,
        h_out: Int64,
        tab: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_d: Int64,
        s_d: Int64,
        fin_ln: Int64,
        cos_sin: Int64,
        block_table: Int64,
        positions: Int64,
        slots: Int64,
        seq_lens: Int64,
        scratch: Int64,
        sym: Int64,
        pe0: Int64,
        pe1: Int64,
        pe2: Int64,
        pe3: Int64,
        step: Int64,
        health: Int64,
        tl: Int64,
        rank: Int32,
        bt_width: Int32,
        n_rows: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        pool = fx.SharedAllocator().allocate(Smem).peek().pool.ptr
        st_ = uniform(dev.ld(step, 0))
        dl = memrealtime() + wait_ticks
        n_reqs = n_rows // layout.QPR
        nch = (n_reqs + CHUNK - 1) // CHUNK

        def stamp(slot):
            if const_expr(TL):  # noqa: SIM102 (a static condition)
                if tid == 0:
                    dev.st(tl, (bid * TL_SLOTS + slot) * 2, memrealtime().to(fx.Int32))

        def stamp_if(cond, slot):
            if const_expr(TL):  # noqa: SIM102 (a static condition)
                if cond & (tid == 0):
                    dev.st(tl, (bid * TL_SLOTS + slot) * 2, memrealtime().to(fx.Int32))

        def tptr(layer, i):
            v = dev.ld(tab, (layer * 16 + i) * 2, 2)
            return (I64(uniform(v[1])) << 32) | I64(fx.Uint32(uniform(v[0])))

        def flag(name):
            return scratch + I64(WF[name])

        stamp(TL_SLOTS - 2)

        def owner(t):
            o = t % 4
            return (o == 0).select(pe0, (o == 1).select(pe1, (o == 2).select(pe2, pe3)))

        def pull_x8(Y, Yf, tag, lb, x8n, xsn, fname, site):
            """CTA b pulls gathered row b from its owner into local scratch + flag."""
            if bid < n_rows:
                ag_copy_x8_pk(
                    bid,
                    owner(bid),
                    Y,
                    Yf,
                    tag,
                    lb + WL[x8n],
                    lb + WL[xsn],
                    tid,
                    dl,
                    health,
                    rank,
                    site,
                )
                dev.publish(flag(fname), bid, tag, tid)

        def merges(tag, lb, ml):
            """Merge tasks (row t, heads 4 hq ..), task = 4 t + hq, round-robin over
            the CTAs."""
            for task in range(bid, n_rows * 4, layout.BLOCKS):
                task = I32(task)
                t = task // 4
                hq = task % 4
                r = t // layout.QPR
                p0, cpr = chunk_cpr(r, n_reqs, SPLIT)
                dev.wait_flags(
                    flag("apf") + I64(p0 * 8), cpr, tag, tid, dl, health, rank, 6
                )
                merge_task_pk(
                    t,
                    hq,
                    cpr,
                    lb + I64(WL["ap"]) + I64(p0) * layout.AP_STRIDE,
                    lb + WL["a8"],
                    lb + WL["as"],
                    ml,
                    tid,
                    lane,
                    layout.W_MAXP,
                )
                dev.publish(flag("attf"), task, tag, tid)

        def part_info(p):
            """Partition slot p: (active, request, window (p0, p6, lo, hi), first block
            bA, blocks cnt, the all-zero tile's block). Inactive slots get request 0
            with no blocks (their loads stay inside request 0's table row)."""
            if const_expr(SPLIT == "even"):
                cpr = even_cpr(n_reqs)
                act = p < cpr * n_reqs
                r = act.select(p // cpr, I32(0))
                c = p % cpr
            else:
                ch = p // CH_PARTS
                pl = p % CH_PARTS
                k = fx.min(I32(CHUNK), n_reqs - ch * CHUNK)
                cpr = (k >= 3).select(I32(CH_PARTS) // fx.max(k, I32(1)), I32(6))
                act = (pl < cpr * k) & (ch < nch)
                r = act.select(ch * CHUNK + pl // fx.max(cpr, I32(1)), I32(0))
                c = pl % fx.max(cpr, I32(1))
            p0, p6, lo, hi, b0w, b1w = att.window(r, positions, seq_lens)
            nb = b1w - b0w + 1
            q0 = (c * nb) // cpr
            cnt = act.select(((c + 1) * nb) // cpr - q0, I32(0))
            bA = b1w - q0
            nope = b1w + 1  # past hi: an all-zero tile
            return act, r, (p0, p6, lo, hi), bA, cnt, nope

        def part_tiles(layer, info):
            """Load the partition's first two blocks' K / V into the LDS tiles (before
            the layer's QKV: the cache rows they read are not this step's)."""
            act, r, (p0, p6, lo, hi), bA, cnt, nope = info
            cache = (tptr(layer, layout.T_CACHE), I32(tptr(layer, layout.T_NBLK)))
            kw0, vw0 = att.kv_load(
                cache,
                block_table,
                bt_width,
                r,
                (cnt >= 1).select(bA, nope),
                lo,
                hi,
                tid,
            )
            kw1, vw1 = att.kv_load(
                cache,
                block_table,
                bt_width,
                r,
                (cnt >= 2).select(bA - 1, nope),
                lo,
                hi,
                tid,
            )
            return kw0, vw0, kw1, vw1

        def part_run(layer, tag, lb, p, info, pre):
            """The partition after QKV: q / new K / V rows, the online softmax over its
            blocks (the first two already in the tiles), the partial + flag."""
            act, r, (p0, p6, lo, hi), bA, cnt, nope = info
            cache = (tptr(layer, layout.T_CACHE), I32(tptr(layer, layout.T_NBLK)))
            if act:
                dev.wait_flags(flag("qkvf"), layout.NG, tag, tid, dl, health, rank, 5)
                stamp_if(layer == 1, 120)
                att.qkv_rows(r, bA, bA - 1, cnt, pre, lb + WL["qkv"], cache, pool, tid)
                gpu.barrier()
                stamp_if(layer == 1, 121)
                qb = att.q_operand(pool, fx.min(wave, 6), l16, g4)
                m_run = F32(NEG_INF)
                l_run = F32(0.0)
                o_run = [F32(0.0)] * 32
                # blocks bA, bA - 1, .. in tile pairs K0 / V0 (even), K1 / V1 (odd);
                # block it + 2's tiles load into registers during block it, then
                # refill the tile pair block it used
                for it in range(0, cnt, 2):
                    it = I32(it)
                    for h in range_constexpr(2):
                        if it + h < cnt:
                            stamp_if((layer == 1) & (it + h < 16), 64 + 2 * (it + h))
                            kn_, vn_ = att.kv_load(
                                cache,
                                block_table,
                                bt_width,
                                r,
                                (it + h + 2 < cnt).select(bA - it - h - 2, nope),
                                lo,
                                hi,
                                tid,
                            )
                            blk = bA - it - h
                            tk = att.TK0 if h == 0 else att.TK1
                            tv = att.TV0 if h == 0 else att.TV1
                            interior = (
                                (blk * layout.BLOCK > p6 - layout.WINDOW)
                                & (blk * layout.BLOCK >= lo)
                                & ((blk + 1) * layout.BLOCK - 1 <= hi)
                            )
                            edge = (
                                (blk * layout.BLOCK <= p6 - layout.WINDOW)
                                | (blk * layout.BLOCK < lo)
                                | ((blk + 1) * layout.BLOCK - 1 > hi)
                            )
                            if (wave < 7) & interior:
                                m_run, l_run, o_run = att.block_acc(
                                    pool,
                                    qb,
                                    tk,
                                    tv,
                                    blk,
                                    wave,
                                    p0,
                                    lo,
                                    hi,
                                    l16,
                                    g4,
                                    m_run,
                                    l_run,
                                    o_run,
                                    False,
                                )
                            if (wave < 7) & edge:
                                m_run, l_run, o_run = att.block_acc(
                                    pool,
                                    qb,
                                    tk,
                                    tv,
                                    blk,
                                    wave,
                                    p0,
                                    lo,
                                    hi,
                                    l16,
                                    g4,
                                    m_run,
                                    l_run,
                                    o_run,
                                    True,
                                )
                            gpu.barrier()
                            stamp_if((layer == 1) & (it + h < 16), 65 + 2 * (it + h))
                            if it + h + 2 < cnt:
                                att.kv_store(pool, tk, tv, kn_, vn_, tid)
                    gpu.barrier()
                stamp_if(layer == 1, 122)
                if wave < 7:
                    ap = lb + I64(WL["ap"]) + I64(p) * layout.AP_STRIDE
                    att.store_partial(
                        ap, wave, l16, g4, m_run * att.SM_SCALE, l_run, o_run
                    )
                dev.publish(flag("apf"), p, tag, tid)
                stamp_if(layer == 1, 123)

        if bid < layout.NG:
            # ======================================================= G: the GEMVs
            # LDS: red [2 buffers][2 weight tiles][8 waves][16][16] f32 (one
            # barrier a tile), then the stage's outputs [row][cols] f32
            red = fx.recast_iter(fx.Float32, pool) + G_RED
            stg = fx.recast_iter(fx.Float32, pool) + G_STAGE
            mlg = fx.recast_iter(fx.Float32, pool) + 5 * att.TILE_WORDS
            nt = (n_rows + 15) // 16

            def tiles(depth, load, body):
                """Token tiles 0..nt-1, tile t + depth's activations issued before
                tile t's math (newer loads stay in flight across its waits)."""
                if const_expr(depth == 0):
                    for t in range(0, nt):
                        t = I32(t)
                        body(t, red + (t % 2) * 4096, load(t))
                else:
                    pend = [load(I32(d)) for d in range(depth)]
                    for t in range(0, nt):
                        t = I32(t)
                        nxt = load(t + depth)
                        body(t, red + (t % 2) * 4096, pend[0])
                        pend = pend[1:] + [nxt]

            def store_partials(col0, Yp, Yf, tag):
                """The staged [row][32] partials as bf16 to each row's owner rank
                (row % 4), [rank][row // 4][6144]: a row's 4 lanes store its 64
                contiguous bytes, a wave's 16 rows share an owner. Then, once every
                store is acknowledged, flag [rank][bid] in every rank's array."""
                own_rows = (n_rows + 3) // 4
                for h in range_constexpr((4 * 4 * lay.own + 511) // 512):
                    i = tid + 512 * h
                    q = i % 4
                    m = i // 4  # row slot
                    wv = m // 16
                    o = wv % 4  # owner, uniform over the wave
                    k = (wv // 4) * 16 + m % 16
                    row = 4 * k + o
                    if (k < own_rows) & (row < n_rows):
                        v = [
                            fx.ptr_load(stg + (row * 32 + 8 * q + e)) for e in range(8)
                        ]
                        pv = fx.Vector.from_elements(
                            [dev.bfpair(v[2 * e], v[2 * e + 1]) for e in range(4)],
                            fx.Int32,
                        )
                        wd = (
                            Yp // 2
                            + (rank * lay.own + k) * layout.HIDDEN
                            + col0
                            + 8 * q
                        ) // 2
                        if o == 0:
                            dev.st(pe0, wd, pv, CM_SYS)
                        if o == 1:
                            dev.st(pe1, wd, pv, CM_SYS)
                        if o == 2:
                            dev.st(pe2, wd, pv, CM_SYS)
                        if o == 3:
                            dev.st(pe3, wd, pv, CM_SYS)
                rocdl.s_waitcnt(vmcnt=0)
                dev.fence_release()
                gpu.barrier()
                fo = (Yf // 4) + (rank * layout.NG + bid) * 2
                if tid == 0:
                    dev.st(pe0, fo, dev.pair(0, tag), CM_SYS)
                if tid == 1:
                    dev.st(pe1, fo, dev.pair(0, tag), CM_SYS)
                if tid == 2:
                    dev.st(pe2, fo, dev.pair(0, tag), CM_SYS)
                if tid == 3:
                    dev.st(pe3, fo, dev.pair(0, tag), CM_SYS)

            def stage_pairs(t, rb):
                """Tile t's two weight tiles (32 output columns), summed over the
                waves -> stg[row][col]."""
                if tid < 256:
                    k = tid // 16
                    rp = tid % 16
                    tt = rp // 8
                    r0 = (2 * rp) % 16
                    lo = dev.red_sum(rb, tt, r0, k)
                    hi = dev.red_sum(rb, tt, r0 + 1, k)
                    o = (16 * t + k) * 32 + 2 * rp
                    fx.ptr_store(lo, stg + o)
                    fx.ptr_store(hi, stg + (o + 1))

            for layer in range(layout.LAYERS):
                layer = I32(layer)
                tid = dev.opaque(tid, layer)
                lane = tid % 64
                wave = tid // 64
                g4 = lane // 16
                l16 = lane % 16
                rr = tid // 16
                kk = tid % 16
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * LB
                qkv_w = w_qkv + I64(layer) * layout.QKV_W_B
                qkv_s = s_qkv + I64(layer) * layout.QKV_S_B
                o_w = w_o + I64(layer) * layout.O_W_B
                o_s = s_o + I64(layer) * layout.O_S_B
                gu_w = w_gu + I64(layer) * layout.GU_W_B
                gu_s = s_gu + I64(layer) * layout.GU_S_B
                d_w = w_d + I64(layer) * layout.D_W_B
                d_s = s_d + I64(layer) * layout.D_S_B
                # ---- attention partition slot bid: its first K / V tiles now
                info = part_info(bid)
                kv4 = part_tiles(layer, info)
                # ---- QKV: rows 12 bid ..
                wq = [
                    dev.w_chunk(
                        qkv_w,
                        qkv_s,
                        bid * layout.QKV_R + fx.min(l16, layout.QKV_R - 1),
                        6144,
                        wave,
                        g4,
                        c,
                    )
                    for c in range(6)
                ]
                pull_x8(Y_AGX, Y_AGXF, tag, lb, "x8", "xs", "x8f", 10)
                att.kv_store(pool, att.TK0, att.TV0, kv4[0], kv4[1], tid)
                att.kv_store(pool, att.TK1, att.TV1, kv4[2], kv4[3], tid)
                dev.wait_flags(flag("x8f"), n_rows, tag, tid, dl, health, rank, 1)
                stamp(layer * 12 + 0)

                def ld_x8(t, wave=wave, lane=lane, lb=lb):
                    return ld_pk(
                        lb + WL["x8"], lb + WL["xs"], fx.min(t, nt - 1), 6, wave, lane
                    )

                def qkv_body(
                    t, rb, xs, wave=wave, g4=g4, l16=l16, rr=rr, kk=kk, tid=tid, wq=wq
                ):
                    dev.red_store(rb, 0, dev.mma_chain(wq, xs, 6), wave, g4, l16)
                    gpu.barrier()
                    if (tid < 256) & (rr < layout.QKV_R):
                        v = dev.red_sum(rb, 0, rr, kk)
                        fx.ptr_store(v, stg + ((16 * t + kk) * layout.QKV_R + rr))

                tiles(D_QKV, ld_x8, qkv_body)
                pre = att.qkv_pre(
                    info[1],
                    tptr(layer, layout.T_Q_NORM),
                    tptr(layer, layout.T_K_NORM),
                    cos_sin,
                    positions,
                    slots,
                    tid,
                )
                gpu.barrier()
                if tid < n_rows:
                    v = [
                        fx.ptr_load(stg + (tid * layout.QKV_R + e))
                        for e in range(layout.QKV_R)
                    ]
                    for e in range_constexpr(3):
                        dev.st(
                            lb + WL["qkv"],
                            tid * (layout.QKV_N // 2)
                            + bid * (layout.QKV_R // 2)
                            + 2 * e,
                            fx.Vector.from_elements(
                                [
                                    dev.bfpair(v[4 * e], v[4 * e + 1]),
                                    dev.bfpair(v[4 * e + 2], v[4 * e + 3]),
                                ],
                                fx.Int32,
                            ),
                            CM_DEV,
                        )
                dev.publish(flag("qkvf"), bid, tag, tid)
                stamp(layer * 12 + 1)
                part_run(layer, tag, lb, bid, info, pre)
                stamp(layer * 12 + 9)
                wo = [
                    [
                        dev.w_chunk(
                            o_w,
                            o_s,
                            bid * layout.O_R + 16 * tt + l16,
                            2048,
                            wave,
                            g4,
                            c,
                        )
                        for c in range(2)
                    ]
                    for tt in range(2)
                ]
                wu = [
                    dev.w_chunk(
                        gu_w,
                        gu_s,
                        layout.IR + bid * layout.GU_R + l16,
                        6144,
                        wave,
                        g4,
                        c,
                    )
                    for c in range(6)
                ]
                merges(tag, lb, mlg)
                stamp(layer * 12 + 2)
                # ---- o_proj: rows 32 bid .., partials to the rows' owners
                wait_flags_n(flag("attf"), n_rows * 4, tag, tid, dl, health, rank, 2)
                stamp(layer * 12 + 3)

                def ld_a8(t, wave=wave, lane=lane, lb=lb):
                    return ld_pk(
                        lb + WL["a8"], lb + WL["as"], fx.min(t, nt - 1), 2, wave, lane
                    )

                def o_body(t, rb, xs, wave=wave, g4=g4, l16=l16, wo=wo):
                    dev.red_store(rb, 0, dev.mma_chain(wo[0], xs, 2), wave, g4, l16)
                    dev.red_store(rb, 1, dev.mma_chain(wo[1], xs, 2), wave, g4, l16)
                    gpu.barrier()
                    stage_pairs(t, rb)

                tiles(D_O, ld_a8, o_body)
                stamp(layer * 12 + 11)
                gpu.barrier()
                store_partials(bid * layout.O_R, Y_PO, Y_FO, tag)
                stamp(layer * 12 + 4)
                wg = [
                    dev.w_chunk(gu_w, gu_s, bid * layout.GU_R + l16, 6144, wave, g4, c)
                    for c in range(6)
                ]
                # ---- gate / up: rows 16 bid .. of each
                pull_x8(Y_AGH, Y_AGHF, tag, lb, "h8", "hs", "h8f", 11)
                dev.wait_flags(flag("h8f"), n_rows, tag, tid, dl, health, rank, 3)
                stamp(layer * 12 + 5)

                def ld_h8(t, wave=wave, lane=lane, lb=lb):
                    return ld_pk(
                        lb + WL["h8"], lb + WL["hs"], fx.min(t, nt - 1), 6, wave, lane
                    )

                def gu_body(
                    t,
                    rb,
                    xb,
                    wave=wave,
                    g4=g4,
                    l16=l16,
                    rr=rr,
                    kk=kk,
                    tid=tid,
                    wg=wg,
                    wu=wu,
                ):
                    dev.red_store(rb, 0, dev.mma_chain(wg, xb, 6), wave, g4, l16)
                    dev.red_store(rb, 1, dev.mma_chain(wu, xb, 6), wave, g4, l16)
                    gpu.barrier()
                    if tid < 256:
                        g = dev.bf16r(dev.red_sum(rb, 0, rr, kk))
                        u = dev.bf16r(dev.red_sum(rb, 1, rr, kk))
                        g = fx.min(g, F32(7.0))
                        u = fx.min(fx.max(u, F32(-7.0)), F32(7.0))
                        a = g * (1.0 / (1.0 + dev.exp(-(g * 1.702)))) * (u + 1.0)
                        fx.ptr_store(a, stg + ((16 * t + kk) * layout.GU_R + rr))

                tiles(D_GU, ld_h8, gu_body)
                gpu.barrier()
                # act once as MXFP8 (what down's per-CTA quantization of the bf16 act
                # computed): this CTA's 16 columns are half of a 32-column block, so
                # CTAs bid, bid ^ 1 trade their rows' amax first
                mt_ = (lay.mr + 15) // 16 * 16
                ra = fx.min(tid, mt_ - 1)  # rows past n_rows: unused values
                av = [
                    dev.bf16r(fx.ptr_load(stg + (ra * layout.GU_R + e)))
                    for e in range(layout.GU_R)
                ]
                am = fmath.absf(av[0])
                for e in range_constexpr(1, layout.GU_R):
                    am = fx.max(am, fmath.absf(av[e]))
                if tid < n_rows:
                    dev.st(lb + WL["am"], bid * mt_ + tid, am, CM_DEV)
                dev.publish(flag("amf"), bid, tag, tid)
                _, fl = dev.poll_pairs(flag("amf"), bid ^ 1, 1, tag, dl, health)
                dev.fail(health, fl, rank, 13)
                gpu.barrier()
                dev.fence_acquire()
                if tid < n_rows:
                    amp = dev.ld(lb + WL["am"], (bid ^ 1) * mt_ + ra, 1, CM_DEV, T.f32)
                    sb, rs = dev.mx_scale(fx.max(am, amp))
                    wd = [
                        fp8_pack4(
                            av[4 * m] * rs,
                            av[4 * m + 1] * rs,
                            av[4 * m + 2] * rs,
                            av[4 * m + 3] * rs,
                        )
                        for m in range(4)
                    ]
                    dev.st(
                        lb + WL["act"],
                        pk_off(tid, bid * layout.GU_R, layout.IR) // 4,
                        fx.Vector.from_elements(wd, fx.Int32),
                        CM_DEV,
                    )
                    if bid % 2 == 0:
                        dev.st8(
                            lb + WL["acs"],
                            pk_soff(tid, bid // 2, layout.IR),
                            sb.to(fx.Int32),
                            CM_DEV,
                        )
                dev.publish(flag("actf"), bid, tag, tid)
                stamp(layer * 12 + 6)
                wd = [
                    [
                        dev.w_chunk(
                            d_w,
                            d_s,
                            bid * layout.D_R + 16 * tt + l16,
                            3072,
                            wave,
                            g4,
                            c,
                        )
                        for c in range(3)
                    ]
                    for tt in range(2)
                ]
                # ---- down: rows 32 bid .., partials to the rows' owners
                dev.wait_flags(flag("actf"), layout.NG, tag, tid, dl, health, rank, 4)
                stamp(layer * 12 + 7)

                def ld_act(t, wave=wave, lane=lane, lb=lb):
                    return ld_pk(
                        lb + WL["act"], lb + WL["acs"], fx.min(t, nt - 1), 3, wave, lane
                    )

                def d_body(t, rb, xs, wave=wave, g4=g4, l16=l16, wd=wd):
                    dev.red_store(rb, 0, dev.mma_chain(wd[0], xs, 3), wave, g4, l16)
                    dev.red_store(rb, 1, dev.mma_chain(wd[1], xs, 3), wave, g4, l16)
                    gpu.barrier()
                    stage_pairs(t, rb)

                tiles(D_DOWN, ld_act, d_body)
                stamp(layer * 12 + 10)
                gpu.barrier()
                store_partials(bid * layout.D_R, Y_PD, Y_FD, tag)
                stamp(layer * 12 + 8)
        else:
            # ============================== A: row owner + attention partitions
            j = bid - A0
            t_own = 4 * j + rank
            ml = fx.recast_iter(fx.Float32, pool) + 5 * att.TILE_WORDS
            res = pool + RES0
            nred = fx.recast_iter(fx.Float32, pool) + (RES0 + 3072)

            def own_norm(v, w, add, Y, Yf, tag):
                o = dev.norm_row(v, res, w, add, nred, tid, lane, wave)
                ag_put_x8(o, t_own, Y, Yf, tag, sym, tid)

            def post(layer, tag, lb):
                post_w = dev.norm_w(tptr(layer, layout.T_POST_LN), tid)
                if t_own < n_rows:
                    stamp(layer * 12 + 8)
                    wait_flags4(sym, Y_FO, tag, tid, dl, health, rank, 8)
                    stamp(layer * 12 + 10)
                    v = rs_read(sym, lay.own, Y_PO, j, tid)
                    own_norm(v, post_w, True, Y_AGH, Y_AGHF, tag)
                stamp(layer * 12 + 9)
                pull_x8(Y_AGH, Y_AGHF, tag, lb, "h8", "hs", "h8f", 11)

            def attn_merge(layer, tag, lb):
                stamp(layer * 12 + 6)
                merges(tag, lb, ml)
                stamp(layer * 12 + 7)

            # layer 0: the input rows
            tag = st_ * 8 + 1
            in_w = dev.norm_w(tptr(0, layout.T_IN_LN), tid)
            if t_own < n_rows:
                own_norm(dev.row_x(x_in, t_own, tid), in_w, False, Y_AGX, Y_AGXF, tag)
            stamp(4)
            pull_x8(Y_AGX, Y_AGXF, tag, scratch, "x8", "xs", "x8f", 10)
            stamp(5)
            attn_merge(I32(0), tag, scratch)
            post(I32(0), tag, scratch)
            for layer in range(1, layout.LAYERS):
                layer = I32(layer)
                tid = dev.opaque(tid, layer)
                lane = tid % 64
                wave = tid // 64
                g4 = lane // 16
                l16 = lane % 16
                tag = st_ * 8 + layer + 1
                lb = scratch + I64(layer) * LB
                in_w = dev.norm_w(tptr(layer, layout.T_IN_LN), tid)
                if t_own < n_rows:
                    wait_flags4(sym, Y_FD, tag - 1, tid, dl, health, rank, 7)
                    stamp(layer * 12 + 11)
                    v = rs_read(sym, lay.own, Y_PD, j, tid)
                    own_norm(v, in_w, True, Y_AGX, Y_AGXF, tag)
                stamp(layer * 12 + 4)
                pull_x8(Y_AGX, Y_AGXF, tag, lb, "x8", "xs", "x8f", 10)
                stamp(layer * 12 + 5)
                attn_merge(layer, tag, lb)
                post(layer, tag, lb)
            ftag = st_ * 8 + layout.LAYERS + 1
            fin_w = dev.norm_w(fin_ln, tid)
            if t_own < n_rows:
                wait_flags4(
                    sym, Y_FD, st_ * 8 + layout.LAYERS, tid, dl, health, rank, 9
                )
                v = rs_read(sym, lay.own, Y_PD, j, tid)
                o = dev.norm_row(v, res, fin_w, True, nred, tid, lane, wave)
                ag_put_bf(o, t_own, Y_AGF, Y_AGFF, ftag, sym, tid)
        # ---- every CTA: final row b into h_out
        ftag2 = st_ * 8 + layout.LAYERS + 1
        if bid < n_rows:
            ag_copy_bf(
                bid, owner(bid), Y_AGF, Y_AGFF, ftag2, h_out, tid, dl, health, rank, 12
            )
        stamp(TL_SLOTS - 1)

    @flyc.jit
    def launch(
        x_in: Int64,
        h_out: Int64,
        tab: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_gu: Int64,
        s_gu: Int64,
        w_d: Int64,
        s_d: Int64,
        fin_ln: Int64,
        cos_sin: Int64,
        block_table: Int64,
        positions: Int64,
        slots: Int64,
        seq_lens: Int64,
        scratch: Int64,
        sym: Int64,
        pe0: Int64,
        pe1: Int64,
        pe2: Int64,
        pe3: Int64,
        step: Int64,
        health: Int64,
        tl: Int64,
        rank: Int32,
        bt_width: Int32,
        n_rows: Int32,
        stream: fx.Stream = _STREAM,
    ):
        drafter_wide_r(
            x_in,
            h_out,
            tab,
            w_qkv,
            s_qkv,
            w_o,
            s_o,
            w_gu,
            s_gu,
            w_d,
            s_d,
            fin_ln,
            cos_sin,
            block_table,
            positions,
            slots,
            seq_lens,
            scratch,
            sym,
            pe0,
            pe1,
            pe2,
            pe3,
            step,
            health,
            tl,
            rank,
            bt_width,
            n_rows,
        ).launch(grid=(layout.BLOCKS,), block=(layout.THREADS,), stream=stream)

    return launch
