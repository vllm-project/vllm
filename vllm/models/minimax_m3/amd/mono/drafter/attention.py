# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention of the DSpark drafter kernels.

Split-KV partitions with the q / k head norms and RoPE fused in, and merge tasks
(a row, 4 heads) that quantize the output to MXFP8 for o_proj. The semantics are
vLLM's draft attention: non-causal within a request's 7 rows, keys below
seq_len, sliding window 4096, FP8 q.k / sqrt(128), P in e4m3, FP8 PV.

LDS tiles are 128 rows x 128 bytes (4096 words); 16-byte chunks are XOR-swizzled
by row % 8 against bank conflicts:

- TQ: q rows m = 16 j + head (j < 7);
- TK0 / TK1: the K rows (keys) of a block;
- TV0 / TV1: V transposed, a row per dim with its 128 keys in the PV operand's
  order (byte 32 g + 4 i + e holds key 16 i + 4 g + e).

Block math, one wave a query row (its 16 heads), after AITER's gfx950 scheme:
S^T = K Q^T puts keys 16 i + 4 g .. + 3 of a head in one lane, which is the PV
operand at 4 keys a word, so P goes from the QK accumulator to O^T = V^T P^T in
registers. exp2 folds log2(e) / sqrt(128) and vLLM's P x 256 (as +8 in the
exponent): l and o carry the factor 256, which o / l cancels.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.minimax_m3.amd.mono.drafter import device as dev
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.drafter.device import F32, I32, I64
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rsq,
    traced,
    xshfl,
)

TQ, TK0, TV0, TK1, TV1 = 0, 1, 2, 3, 4
TILE_WORDS = 4096
QK_C = 1.4426950408889634 / 11.313708498984761  # log2(e) / sqrt(128)
SM_SCALE = 1.0 / 11.313708498984761
NEG_INF = float("-inf")


def tw(t, r, w):
    """Swizzled word index of word w of row r of tile t."""
    return t * TILE_WORDS + r * 32 + ((w // 4) ^ (r % 8)) * 4 + w % 4


def perm(a, b, sel):
    """v_perm_b32: byte k of the result = byte sel[k] of (a << 32 | b)."""
    return I32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.perm",
            [I32(a).ir_value(), I32(b).ir_value(), I32(sel).ir_value()],
            [],
            [],
        )
    )


def ipos(positions, t):
    return dev.ld(positions, I32(t) * 2)


def window(r, positions, seq_lens):
    """Request r: first / last query position, its window union's first / last key and
    blocks (keys < seq_len, from the first row's window start, <= MAXB blocks)."""
    p0 = ipos(positions, r * layout.QPR)
    p6 = ipos(positions, r * layout.QPR + layout.QPR - 1)
    seq = dev.ld(seq_lens, r)
    lo = fx.max(p0 - (layout.WINDOW - 1), I32(0))
    hi = fx.min(seq - 1, (lo // layout.BLOCK + layout.MAXB) * layout.BLOCK - 1)
    return p0, p6, lo, hi, lo // layout.BLOCK, hi // layout.BLOCK


# ---------------------------------------------------------------- K / V tiles
def kv_load(cache, block_table, bt_width, r, blk, lo, hi, tid):
    """Block blk of request r (keys outside [lo, hi] zero), issued at the layer's start:
    this thread's K words (key tid / 4, bytes 32 (tid % 4) ..: 8 words) and V words
    (keys 4 (tid / 16) .. + 3, dims 8 (tid % 16) ..: 4 x 2 words)."""
    cache, nblk = cache
    # padded requests of a FULL graph replay carry stale metadata: keep every
    # address inside the block table row and the cache
    col = fx.min(fx.max(fx.min(blk, hi // layout.BLOCK), I32(0)), bt_width - 1)
    phys = fx.min(fx.max(dev.ld(block_table, r * bt_width + col), I32(0)), nblk - 1)
    base = cache + I64(phys) * (layout.BLOCK * 256)
    km = tid // 4
    kq = tid % 4
    kp = blk * layout.BLOCK + km
    klive = (kp >= lo) & (kp <= hi)
    k0 = dev.ld(base, km * 64 + kq * 8, 4)
    k1 = dev.ld(base, km * 64 + kq * 8 + 4, 4)
    kw = [klive.select(k0[e], I32(0)) for e in range(4)] + [
        klive.select(k1[e], I32(0)) for e in range(4)
    ]
    vg = tid // 16
    vd = tid % 16
    vw = []
    for j in range_constexpr(4):
        m = vg * 4 + j
        vp = blk * layout.BLOCK + m
        vlive = (vp >= lo) & (vp <= hi)
        v = dev.ld(base, m * 64 + 32 + vd * 2, 2)
        vw.append([vlive.select(v[0], I32(0)), vlive.select(v[1], I32(0))])
    return kw, vw


def kv_store(lds, tk, tv, kw, vw, tid):
    """The thread's K words -> tile ``tk`` (row-major); its V words, transposed in
    registers, -> tile ``tv``."""
    km = tid // 4
    kq = tid % 4
    for h in range_constexpr(2):
        w0 = kq * 8 + h * 4
        fx.ptr_store(
            fx.Vector.from_elements(kw[4 * h : 4 * h + 4], fx.Int32),
            lds + tw(tk, km, w0),
        )
    # keys 4 vg .. 4 vg + 3 = 16 i + 4 g ..: word 8 g + i of each dim row
    vg = tid // 16
    vd = tid % 16
    vwi = (vg % 4) * 8 + vg // 4
    for h in range_constexpr(2):  # dims 8 vd + 4 h .. + 3
        a = [vw[j][h] for j in range(4)]
        t01l = perm(a[1], a[0], 0x05010400)
        t01h = perm(a[1], a[0], 0x07030602)
        t23l = perm(a[3], a[2], 0x05010400)
        t23h = perm(a[3], a[2], 0x07030602)
        outs = [
            perm(t23l, t01l, 0x05040100),
            perm(t23l, t01l, 0x07060302),
            perm(t23h, t01h, 0x05040100),
            perm(t23h, t01h, 0x07060302),
        ]
        for x in range_constexpr(4):
            d = vd * 8 + h * 4 + x
            fx.ptr_store(outs[x], lds + tw(tv, d, vwi))


# ---------------------------------------------------------------- q / new k / new v
# rows 128 slots t = 16-lane group + 32 s (s < 4): t < 112 q rows (j = t / 16, head t %
# 16), 112..118 the new k rows, 119..125 the new v rows. A slot's 128 dims over 16
# lanes, 8 each: lanes 0..7 hold the RoPE pair dims 4 i .. + 3 and 32 + 4 i .. + 3
# (partner in the same lane), lanes 8..15 the pass-through dims 64 + 8 (i - 8) .. + 7:
# no cross-lane RoPE exchange, a 16-lane norm butterfly, v_rsq.
def _slot_rows(r, t):
    isq = t < 112
    jk = fx.min(fx.max(t - 112, I32(0)), I32(13))  # 0..6 k rows, 7..13 v rows
    j = isq.select(t // 16, jk % 7)
    col = isq.select((t % 16) * 128, layout.QS + (jk // 7) * 128)
    return isq, j, col, r * layout.QPR + j


def _dims(tid):
    li = tid % 16
    rope = li < 8
    dA = rope.select(4 * li, 64 + 8 * (li - 8))
    dB = dA + rope.select(I32(32), I32(4))
    return li, rope, dA, dB


def qkv_pre(r, qn, kn, cos_sin, positions, slots, tid):
    """The loads of ``qkv_rows`` that do not depend on this layer's QKV."""
    li, rope, dA, dB = _dims(tid)
    grp = tid // 16
    ib = rope.select(4 * li, I32(0))
    gq = [
        dev.ld(qn, dA // 2),
        dev.ld(qn, dA // 2 + 1),
        dev.ld(qn, dB // 2),
        dev.ld(qn, dB // 2 + 1),
    ]
    gk = [
        dev.ld(kn, dA // 2),
        dev.ld(kn, dA // 2 + 1),
        dev.ld(kn, dB // 2),
        dev.ld(kn, dB // 2 + 1),
    ]
    css, sls, poss = [], [], []
    for s in range_constexpr(4):
        t = grp + 32 * s
        isq, j, col, row = _slot_rows(r, t)
        pos = ipos(positions, row)
        c0 = (pos * layout.ROT + ib) // 2
        s0 = (pos * layout.ROT + layout.ROT // 2 + ib) // 2
        css.append(
            [
                dev.ld(cos_sin, c0),
                dev.ld(cos_sin, c0 + 1),
                dev.ld(cos_sin, s0),
                dev.ld(cos_sin, s0 + 1),
            ]
        )
        sls.append(I64(dev.ld(slots, row * 2)))
        poss.append(pos)
    return gq, gk, css, sls, poss


def _un4(w):
    return [dev.bf_lo(w[0]), dev.bf_hi(w[0]), dev.bf_lo(w[1]), dev.bf_hi(w[1])]


# ---------------------------------------------------------------- one 128-key block
def _mma_unit(a, b, c):
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, 0, 0, 0, 127, 0, 127]
        )
    )


def _lds8w(lds, t, r, w0):
    """Words w0 .. w0 + 3 and w0 + 16 .. w0 + 19 of row r of tile t: an operand."""
    v4 = fx.Vector.make_type(4, fx.Int32)
    x = fx.Vector(fx.ptr_load(lds + tw(t, r, w0), result_type=v4))
    y = fx.Vector(fx.ptr_load(lds + tw(t, r, w0 + 16), result_type=v4))
    return fx.Vector.from_elements(
        [x[e] for e in range(4)] + [y[e] for e in range(4)], fx.Int32
    )


def _lds8c(lds, t, r, w0):
    """Words w0 .. w0 + 7 (two consecutive 16-byte chunks) of row r of tile t."""
    v4 = fx.Vector.make_type(4, fx.Int32)
    x = fx.Vector(fx.ptr_load(lds + tw(t, r, w0), result_type=v4))
    y = fx.Vector(fx.ptr_load(lds + tw(t, r, w0 + 4), result_type=v4))
    return fx.Vector.from_elements(
        [x[e] for e in range(4)] + [y[e] for e in range(4)], fx.Int32
    )


def q_operand(lds, wave, l16, g4):
    """Wave ``wave``'s Q^T operand (query row ``wave``, head ``l16``)."""
    return _lds8w(lds, TQ, wave * 16 + l16, g4 * 4)


def block_math(lds, qb, tk, tv, blk, j, p0, lo, hi, l16, g4, mask):
    """One 128-key block for query row ``j`` (this wave), head ``l16``.

    Returns (m in score units, l, o: 32 f32 of dims 16 n + 4 g + e); l and o are
    scaled by 256. ``mask`` (static): the block is at the window's edge.
    """
    s = []
    for i in range_constexpr(8):
        ka = _lds8w(lds, tk, 16 * i + l16, g4 * 4)
        c = _mma_unit(ka, qb, dev.zero4())
        s += [c[e] for e in range(4)]
    if const_expr(mask):
        qp = p0 + j
        for i in range_constexpr(8):
            for e in range_constexpr(4):
                kp = blk * layout.BLOCK + 16 * i + 4 * g4 + e
                ok = (kp > qp - layout.WINDOW) & (kp >= lo) & (kp <= hi)
                s[4 * i + e] = ok.select(s[4 * i + e], F32(NEG_INF))
    m = s[0]
    for x in range_constexpr(1, 32):
        m = fx.max(m, s[x])
    m = butterfly(m, (16, 32), fx.max)
    msafe = m
    if const_expr(mask):
        msafe = (m == F32(NEG_INF)).select(F32(0.0), m)
    off = msafe * QK_C - 8.0
    p = [hw_exp2(s[x] * QK_C - off) for x in range(32)]
    lsum = p[0]
    for x in range_constexpr(1, 32):
        lsum = lsum + p[x]
    lsum = butterfly(lsum, (16, 32))
    pw = fx.Vector.from_elements(
        [
            fp8_pack4(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3])
            for i in range(8)
        ],
        fx.Int32,
    )
    o = []
    for n in range_constexpr(8):
        va = _lds8c(lds, tv, 16 * n + l16, g4 * 8)
        c = _mma_unit(va, pw, dev.zero4())
        o += [c[e] for e in range(4)]
    return m * SM_SCALE, lsum, o


def store_partial(ap, wave, l16, g4, m, ls, o):
    """(m, l, o) of query rows 16 wave + l16 (< 112) -> the partial at ``ap``:
    m [128] f32 at 0, l at 512 B, o [112][128] at 1 KB."""
    mr = wave * 16 + l16
    if g4 == 0:
        dev.st(ap, mr, m, CM_DEV)
        dev.st(ap, 128 + mr, ls, CM_DEV)
    for n in range_constexpr(8):
        dev.st(
            ap,
            256 + mr * 128 + 16 * n + 4 * g4,
            fx.Vector.from_elements(o[4 * n : 4 * n + 4], fx.Float32),
            CM_DEV,
        )


store_partial = traced(store_partial)


# ---------------------------------------------------------------- merge
def merge_task(t, hq, na, ap0, a8, as8, ml, tid, lane, maxp):
    """Merge row ``t``, heads 4 hq .. 4 hq + 3 (a thread a head and dim).

    The request's partials 0..na-1 at ``ap0`` are merged in order, rounded to bf16
    and quantized to MXFP8 into a8 / as8. The partials' (m, l) of the 4 heads are
    staged through LDS ``ml`` (512 floats). ``maxp``: static bound of ``na``.
    """
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
    # partials past na read 0 (a resource of na partials: no memory access)
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
    out = dev.bf16r(
        (lsum > 0.0).select(acc / lsum, F32(0.0))
    )  # a padded request's rows: no keys
    am = butterfly(fmath.absf(out), (16, 8, 4, 2, 1), fx.max)
    sb, rs = dev.mx_scale(am)
    q = fp8_pack4(out * rs, F32(0.0), F32(0.0), F32(0.0)) & 0xFF
    w1 = q | (xshfl(q, 1) << 8)
    w2 = w1 | (xshfl(w1, 2) << 16)
    col = h * 128 + d
    if d % 4 == 0:
        dev.st(a8, (t * layout.QS + col) // 4, w2, CM_DEV)
    if d % 32 == 0:
        dev.st8(as8, t * (layout.QS // 32) + col // 32, sb.to(fx.Int32), CM_DEV)


merge_task = traced(merge_task)


def block_acc(lds, qb, tk, tv, blk, j, p0, lo, hi, l16, g4, m_run, l_run, o_run, mask):
    """Fold one 128-key block into the running (m, l, o) of an online softmax.

    m is in raw score units, l and o are scaled by 256 at m: m' = max(m, block max),
    l and o are rescaled by exp(m - m'), and P (x 256 at m', e4m3) V accumulates
    into o.
    """
    s = []
    for i in range_constexpr(8):
        c = _mma_unit(_lds8w(lds, tk, 16 * i + l16, g4 * 4), qb, dev.zero4())
        s += [c[e] for e in range(4)]
    if const_expr(mask):
        qp = p0 + j
        for i in range_constexpr(8):
            for e in range_constexpr(4):
                kp = blk * layout.BLOCK + 16 * i + 4 * g4 + e
                ok = (kp > qp - layout.WINDOW) & (kp >= lo) & (kp <= hi)
                s[4 * i + e] = ok.select(s[4 * i + e], F32(NEG_INF))
    mb = s[0]
    for x in range_constexpr(1, 32):
        mb = fx.max(mb, s[x])
    mb = butterfly(mb, (16, 32), fx.max)
    mn = fx.max(m_run, mb)
    msafe = (mn == F32(NEG_INF)).select(F32(0.0), mn)
    alpha = hw_exp2((m_run - msafe) * QK_C)
    off = msafe * QK_C - 8.0
    p = [hw_exp2(s[x] * QK_C - off) for x in range(32)]
    ps = p[0]
    for x in range_constexpr(1, 32):
        ps = ps + p[x]
    l_new = l_run * alpha + butterfly(ps, (16, 32))
    pw = fx.Vector.from_elements(
        [
            fp8_pack4(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3])
            for i in range(8)
        ],
        fx.Int32,
    )
    o = []
    for n in range_constexpr(8):
        acc = fx.Vector.from_elements(
            [o_run[4 * n + e] * alpha for e in range(4)], fx.Float32
        )
        c = _mma_unit(_lds8c(lds, tv, 16 * n + l16, g4 * 8), pw, acc)
        o += [c[e] for e in range(4)]
    return mn, l_new, o


def qkv_rows(r, bA, bB, nfirst, pre, qkv, cache, lds, tid):
    """Request r's q (7 rows x 16 heads: head norm, RoPE, FP8) -> TQ, and its new
    K / V rows -> the cache and, where they fall in the CTA's first blocks ``bA``
    (K0 / V0, nfirst >= 1) or ``bB`` (K1 / V1, nfirst >= 2), the tiles."""
    cache, nblk = cache
    gq, gk, css, sls, poss = pre
    li, rope, dA, dB = _dims(tid)
    grp = tid // 16
    es = []
    for s in range_constexpr(4):
        t = grp + 32 * s
        isq, j, col, row = _slot_rows(r, t)
        base = (row * layout.QKV_N + col) // 2
        es.append(
            [
                dev.ld(qkv, base + dA // 2),
                dev.ld(qkv, base + dA // 2 + 1),
                dev.ld(qkv, base + dB // 2),
                dev.ld(qkv, base + dB // 2 + 1),
            ]
        )
    gqa = [x + 1.0 for x in _un4(gq[0:2])]
    gqb = [x + 1.0 for x in _un4(gq[2:4])]
    lds8 = fx.recast_iter(fx.Int8, lds)
    for s in range_constexpr(4):
        t = grp + 32 * s
        e = es[s]
        a = _un4(e[0:2])
        b = _un4(e[2:4])
        cw = css[s]
        if const_expr(s < 3):
            ga, gb, rot = gqa, gqb, rope
        else:
            isq3 = t < 112
            gw = [isq3.select(gq[x], gk[x]) for x in range(4)]
            ga = [x + 1.0 for x in _un4(gw[0:2])]
            gb = [x + 1.0 for x in _un4(gw[2:4])]
            rot = rope & (t < 119)
        ss = (
            a[0] * a[0]
            + a[1] * a[1]
            + a[2] * a[2]
            + a[3] * a[3]
            + b[0] * b[0]
            + b[1] * b[1]
            + b[2] * b[2]
            + b[3] * b[3]
        )
        ss = butterfly(ss, (8, 4, 2, 1))
        rr = hw_rsq(ss * (1.0 / 128.0) + layout.EPS)
        cs = [
            rot.select(dev.bf_lo(cw[0]), F32(1.0)),
            rot.select(dev.bf_hi(cw[0]), F32(1.0)),
            rot.select(dev.bf_lo(cw[1]), F32(1.0)),
            rot.select(dev.bf_hi(cw[1]), F32(1.0)),
        ]
        sn = [
            rot.select(dev.bf_lo(cw[2]), F32(0.0)),
            rot.select(dev.bf_hi(cw[2]), F32(0.0)),
            rot.select(dev.bf_lo(cw[3]), F32(0.0)),
            rot.select(dev.bf_hi(cw[3]), F32(0.0)),
        ]
        oa, ob = [], []
        for i in range_constexpr(4):
            if const_expr(s < 3):
                na = dev.bf16r(a[i] * rr * ga[i])
                nb = dev.bf16r(b[i] * rr * gb[i])
            else:
                isv = (t >= 119) & (t < 126)
                na = isv.select(a[i], dev.bf16r(a[i] * rr * ga[i]))
                nb = isv.select(b[i], dev.bf16r(b[i] * rr * gb[i]))
            oa.append(dev.bf16r(na * cs[i] - nb * sn[i]))
            ob.append(dev.bf16r(nb * cs[i] + na * sn[i]))
        wa = fp8_pack4(oa[0], oa[1], oa[2], oa[3])
        wb = fp8_pack4(ob[0], ob[1], ob[2], ob[3])
        if const_expr(s < 3):
            fx.ptr_store(wa, lds + tw(TQ, t, dA // 4))
            fx.ptr_store(wb, lds + tw(TQ, t, dB // 4))
        else:
            isq = t < 112
            isk = (t >= 112) & (t < 119)
            isv = (t >= 119) & (t < 126)
            pos = poss[s]
            blk = pos // layout.BLOCK
            inA = (blk == bA) & (nfirst >= 1)
            inB = (blk == bB) & (nfirst >= 2)
            live = (isk | isv) & (inA | inB)
            if isq:
                fx.ptr_store(wa, lds + tw(TQ, t, dA // 4))
                fx.ptr_store(wb, lds + tw(TQ, t, dB // 4))
            kk = pos % layout.BLOCK
            if live & isk:
                tk = inA.select(I32(TK0), I32(TK1))
                fx.ptr_store(wa, lds + tw(tk, kk, dA // 4))
                fx.ptr_store(wb, lds + tw(tk, kk, dB // 4))
            if live & isv:
                tv = inA.select(I32(TV0), I32(TV1))
                vwi = ((kk % 16) // 4) * 8 + kk // 16
                for x in range_constexpr(4):
                    fx.ptr_store(
                        fx.Int32((wa >> (8 * x)) & 0xFF).to(fx.Int8),
                        lds8 + (tw(tv, dA + x, vwi) * 4 + kk % 4),
                    )
                    fx.ptr_store(
                        fx.Int32((wb >> (8 * x)) & 0xFF).to(fx.Int8),
                        lds8 + (tw(tv, dB + x, vwi) * 4 + kk % 4),
                    )
            if live & (sls[s] >= 0) & (sls[s] < I64(nblk) * layout.BLOCK):
                cb = cache + sls[s] * 256 + I64(isv.select(I32(128), I32(0)))
                dev.gst32(cb + I64(dA), wa)
                dev.gst32(cb + I64(dB), wb)


qkv_rows = traced(qkv_rows)
