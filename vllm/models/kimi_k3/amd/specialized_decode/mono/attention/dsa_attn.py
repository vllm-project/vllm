# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""`dsa_attn`: Kimi-K3.1's DSA sparse MLA decode at TP8 + DCP1, the bf16
absorbed query against the fp8_ds_mla cache rows the indexer's top-k names,
split over the top-k: every split's partial (bf16 o, f32 LSE) for M1b's
combine (`mla_back`).

A CTA a (token, split) work item, `tokens x splits` of the `BLOCKS` CTAs (the
rest free for K2's other stages). A split takes `chunk` of the row's TOPK
entries:

    slice               the split's top-k range; with `has_invalid` a scan
                        for the last valid (>= 0) entry trims the tiles
    tiles (TILE rows)   gather the 656 B records (512 E4M3 latents, scales,
                        64 bf16 rope), fp8 QK MFMAs plus the rope dot, the
                        online softmax, PV into the running o
    end                 o / LSE -> `part_o` [splits, S, H, 512] and
                        `part_lse` [splits, S, H]

With `fold` the top-k holds `dsa_front`'s logical token positions: the kernel
maps them through the block table (`bt`, `bsr` rows a block) itself, dividing
by the block size with `fold_word`'s multiply-shift. In K2 (`standalone`
False) waves 0 .. 3 attend and the other waves only match their barriers.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T
from flydsl.expr.typing import Vector as Vec

from vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla_gfx950 import (
    _BLOCK_THREADS,
    _K_DEPTH,
    _KS_OFF,
    _L,
    _LN2,
    _MX_OFF,
    _P_OFF,
    _P_SHIFT,
    _SCALE_CAP_BITS,
    _SINK_OFF,
    _TILE,
    _V_OFF,
    MAX_HEADS,
    RECORD_BYTES,
    _BoundedI32,
    _cat,
    _fmax,
    _global_i32,
    _global_llvm_ptr,
    _guarded,
    _imax,
    _imin,
    _lds_llvm_ptr,
    _lds_load,
    _lds_store,
    _mfma_bf16,
    _mfma_fp8,
    _p_store_f16,
    _pv_mfma,
    _pv_operands,
    _q_prologue,
    _readfirstlane,
    _tr8,
    _xmax,
    _xsum,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import ENTRY, KVL, NH
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import CM_DEV, traced
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    KernelAbi,
    key_tuple,
    source_digest,
)

TOPK = 2048
TILE = _TILE
BS_BITS = 26
BS_POS = 32
LG_POS = BS_POS + BS_BITS

_STREAM = fx.Stream(None)
_SOURCES = source_digest(".")


@dataclass(frozen=True, slots=True)
class DsaAttnBuild:
    """A `dsa_attn` build: `tokens` rows x `splits` splits of `chunk` top-k each."""

    tokens: int = 8
    heads: int = NH
    splits: int = 16
    chunk: int = 128
    tile: int = _TILE
    has_invalid: bool = False
    fold: bool = False
    standalone: bool = True


ABI = KernelAbi(
    (
        "q",
        "stride_q",
        "stride_qh",
        "kv",
        "idx",
        "iptr",
        "topk",
        "topk_st",
        "bt",
        "part_o",
        "part_lse",
        "c_log2",
        "idx_bytes",
        "bsr",
        "bsm",
    )
)

BARRIERS = {"pre": 0, "tile": 3, "end": 1}


def check(key: DsaAttnBuild) -> None:
    assert 1 <= key.heads <= MAX_HEADS and RECORD_BYTES == 656
    assert key.tokens * key.splits <= BLOCKS
    assert key.chunk * key.splits >= TOPK and key.chunk <= key.tile == _TILE


def geometry(key: DsaAttnBuild) -> tuple[int, int]:
    items = key.tokens * key.splits
    return items, BLOCKS - items


def partial_shapes(key: DsaAttnBuild):
    p, s, h = key.splits, key.tokens, key.heads
    return (p, s, h, KVL), (p, s, h)


def lds_bytes(key: DsaAttnBuild) -> int:
    return _P_OFF + 4 * 16 * key.heads * 16


def _bar(bars: dict, phase: str) -> None:
    bars[phase] = bars.get(phase, 0) + 1
    gpu.barrier()


@traced
def _slice(c, key: DsaAttnBuild, v):
    """Work item v's (token, split, top-k source, range, tile count)."""
    P, C = key.splits, key.chunk
    a = c["args"]
    lane = fx.Int32(c["tid"]) % 64
    qi = fx.Int32(v) // P
    sj = fx.Int32(v) % P
    ip_t = GTensor(a["iptr"], dtype=T.i32, shape=(-1,))
    if const_expr(key.fold):
        src = GTensor(a["topk"], dtype=T.i32, shape=(-1,))
        row_beg = qi * a["topk_st"]
    else:
        src = _BoundedI32(a["idx"], a["idx_bytes"])
        row_beg = _readfirstlane(ip_t[qi])
    row_len = _readfirstlane(ip_t[qi + 1]) - _readfirstlane(ip_t[qi])
    r_beg = row_beg + sj * C
    r_len = _imin(row_len - sj * C, fx.Int32(C))
    r_len = (r_len < 0).select(fx.Int32(0), r_len)

    n_scan = ((r_len + 255) // 256) if key.has_invalid else fx.Int32(0)
    lastp = fx.Int32(0)
    for sc_c in range(0, n_scan, 1):
        p0 = fx.Int32(sc_c) * 256 + lane * 4
        vv = Vec(src.load(r_beg + p0, vec_size=4))
        for jj in range_constexpr(4):
            pj = p0 + jj
            lastp = (pj < r_len).select(
                (fx.Int32(vv[jj]) >= 0).select(pj + 1, lastp), lastp
            )
    lastp = lastp if key.has_invalid else r_len
    for sh in (1, 2, 4, 8, 16, 32):
        o_ = fx.Int32(lastp).shuffle_xor(sh, 64)
        lastp = (o_ > lastp).select(o_, lastp)
    nt = _readfirstlane((lastp + (_TILE - 1)) // _TILE)
    return qi, sj, src, r_beg, r_len, nt


@traced
def cta(c, key: DsaAttnBuild, v):
    """Work item v: attend (`_body`), or in K2 the waves past 3 shadow it."""
    if const_expr(key.standalone):
        _body(c, key, v)
    else:
        if _readfirstlane(fx.Int32(c["tid"]) // 64) < 4:
            _body(c, key, v)
        else:
            _shadow(c, key, v)


@traced
def _shadow(c, key: DsaAttnBuild, v):
    """Match `_body`'s barriers without attending."""
    nt = _slice(c, key, v)[-1]
    for _ in range_constexpr(BARRIERS["pre"]):
        rocdl.s_barrier()
    for _tile in range(0, nt, 1):
        for _ in range_constexpr(BARRIERS["tile"]):
            rocdl.s_barrier()
    for _ in range_constexpr(BARRIERS["end"]):
        rocdl.s_barrier()


@traced
def _body(c, key: DsaAttnBuild, v):
    """Work item v: the split's tiles -> its partial o and LSE."""
    H, S = key.heads, key.tokens
    a = c["args"]
    tid = fx.Int32(c["tid"])
    w = _readfirstlane(tid // 64)
    lane = tid % 64
    t = lane % 16
    g = lane // 16
    p_wave_bytes = 16 * H * 16
    bars: dict = {}

    q_t = GTensor(a["q"], dtype=T.i32, shape=(-1,))
    kv_base = buf_base_i64(a["kv"])
    cm = 0 if key.standalone else CM_DEV
    o_t = GTensor(a["part_o"], dtype=T.i32, shape=(-1,), cache_modifier=cm)
    lse_t = GTensor(a["part_lse"], dtype=T.i32, shape=(-1,), cache_modifier=cm)
    base = _readfirstlane(fx.ptrtoint(c["lds"]))
    c_log2 = fx.Int32(a["c_log2"]).bitcast(fx.Float32)
    stride_q, stride_qh = a["stride_q"], a["stride_qh"]
    qi, sj, src, r_beg, r_len, nt = _slice(c, key, v)
    if const_expr(key.fold):
        bt_t = _BoundedI32(a["bt"], a["idx_bytes"])
        nblk = fx.Int32(a["idx_bytes"]) // 4
        bsr = fx.Int32(a["bsr"])
        bsm = fx.Int64(a["bsm"])
        mag = bsm & ((1 << BS_POS) - 1)
        bs = fx.Int32(bsm >> BS_POS) & ((1 << BS_BITS) - 1)
        bs_sh = (bsm >> LG_POS) + 31

    h_ok = t < H
    h_safe = h_ok.select(t, fx.Int32(0))
    q_row = (qi * stride_q + h_safe * stride_qh) // 2
    q8, qs, q8l, qsl = _q_prologue(q_t, q_row, g, h_ok)
    qr = [
        Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4))
        for i in range_constexpr(2)
    ]

    def fsw(tok):
        return ((tok >> 1) & 7) ^ (((tok >> 4) & 1) << 1)

    k_rdb = []
    for s in range_constexpr(2):
        tok = w * 32 + 16 * s + t
        k_rdb.append(
            base + _V_OFF + (g // 2) * 16384 + tok * 128 + ((g % 2) ^ fsw(tok)) * 16
        )
    v_rd = []
    for sub in range_constexpr(2):
        i = t // 2
        tok = (g // 2) * 32 + (i // 4) * 16 + ((g % 2) * 2 + sub) * 4 + i % 4
        v_rd.append(base + _V_OFF + w * 16384 + tok * 128 + fsw(tok) * 16 + (t % 2) * 8)
    dma_c = [
        w * 128
        + ((lane % 8) ^ (4 * (jj & 1) + lane // 16) ^ (2 * ((jj >> 1) & 1))) * 16
        for jj in range_constexpr(4)
    ]
    dma_src = [((8 * jj + lane // 8) & 63) * 4 for jj in range(8)]

    ks_wr = base + _KS_OFF + ((w * 4 + g) * 32 + t) * 4
    ks_rd = base + _KS_OFF + (w * 4 * 32 + 4 * g) * 4
    mx_wr = base + _MX_OFF + (t * 4 + w) * 4
    mx_rd = base + _MX_OFF + t * 16
    sink = base + _SINK_OFF
    h_rd = h_ok.select(t, fx.Int32(H - 1))
    p_wr = h_ok.select(base + _P_OFF + ((w * 4 + g) * H + t) * 16, sink)
    p_rd = (
        base
        + _P_OFF
        + w * p_wave_bytes
        + (((g // 2) * 4 + 2 * (g % 2)) * H + h_rd) * 16
    )

    def poff(cc):
        return h_ok.select(fx.Int32(cc), fx.Int32(0))

    def ld_idx(p):
        ix = fx.Int32(src[r_beg + p])
        if const_expr(key.fold):
            blk = fx.Int32((fx.Int64(ix) * mag) >> bs_sh)
            ok = (ix >= 0) & (blk < nblk)
            row = fx.Int32(bt_t[ok.select(blk, fx.Int32(0))]) * bsr + (ix - blk * bs)
            ix = ok.select(row, fx.Int32(-1))
        return (p < r_len).select(ix, fx.Int32(-1))

    def ld_q_idx(tb):
        return [ld_idx(w * 32 + t + tb + 16 * s) for s in range(2)]

    def ld_d_idx(tb):
        return [ld_idx(lane + 64 * hh + tb) for hh in range(2)]

    def ld_side(ixs):
        out = []
        for s in range_constexpr(2):
            ix = fx.Int32(ixs[s])
            row = kv_base + fx.Int64((ix >= 0).select(ix, fx.Int32(0))) * RECORD_BYTES
            out.append(
                Vec.from_elements(
                    [_global_i32(row + fx.Int64(_L + g * 4), 1)], fx.Int32
                )
            )
            for i in range_constexpr(2):
                out.append(_global_i32(row + fx.Int64(_L + 16 + 64 * i + g * 16), 4))
        return out

    def dma(ixd):
        ixs = [(fx.Int32(ix) >= 0).select(fx.Int32(ix), fx.Int32(0)) for ix in ixd]
        for jj in range_constexpr(16):
            ix = fx.Int32(
                rocdl.ds_bpermute(T.i32, _raw(dma_src[jj % 8]), _raw(ixs[jj // 8]))
            )
            src = _global_llvm_ptr(
                kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(dma_c[jj % 4])
            )
            dst = _lds_llvm_ptr(base + w * 16384 + (_V_OFF + jj * 1024))
            rocdl.global_load_lds(src, dst, 16, 0)

    def nxt(tb, k):
        return (tb < (nt - k) * _TILE).select(tb + k * _TILE, tb)

    t1 = (nt > 1).select(fx.Int32(_TILE), fx.Int32(0))
    ixq0 = ld_q_idx(fx.Int32(0))
    ixd0 = ld_d_idx(fx.Int32(0))
    dma(ixd0)
    side0 = ld_side(ixq0)
    ixq1 = ld_q_idx(t1)
    ixd1 = ld_d_idx(t1)

    neg_inf = fx.Float32(-1.0e30)
    init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
    init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(8)]
    init += [_raw(fx.Int32(x).ir_value()) for x in ixq1 + ixd1]
    init += [_raw(Vec(x).ir_value()) for x in side0]
    init += [_raw(fx.Int32(x).ir_value()) for x in ixq0]
    ns = 10
    for tile, st in range(0, nt, 1, init=init):  # type: ignore[call-overload]
        m_old = fx.Float32(st[0])
        l_old = fx.Float32(st[1])
        o = [Vec(st[2 + i]) for i in range(8)]
        ixq_n = [fx.Int32(st[ns + i]) for i in range(2)]
        ixd_n = [fx.Int32(st[ns + 2 + i]) for i in range(2)]
        side = [Vec(st[ns + 4 + i]) for i in range(6)]
        ixq_c = [fx.Int32(st[ns + 10 + i]) for i in range(2)]
        tb = fx.Int32(tile) * _TILE
        not_last = fx.Int32(tile) < nt - 1

        rocdl.s_waitcnt(vmcnt=0)
        _bar(bars, "tile")
        t2 = nxt(tb, 2)
        ixq_nn = ld_q_idx(t2)

        def kread(jj):
            s, blk = jj // 4, jj % 4
            ka = k_rdb[s] ^ (32 * blk)
            return _cat(
                _lds_load(T.i32, 4, ka, 16), _lds_load(T.i32, 4, ka + 32768, 16), 4
            )

        kl = [kread(jj) for jj in range_constexpr(_K_DEPTH)] + [None] * (8 - _K_DEPTH)
        sacc = [Vec.filled(4, 0.0, fx.Float32), Vec.filled(4, 0.0, fx.Float32)]
        for jj in range_constexpr(8):
            s, blk = jj // 4, jj % 4
            rocdl.sched_barrier(0)
            lt = kl[jj]
            kl[jj] = None
            if const_expr(jj + _K_DEPTH < 8):
                kl[jj + _K_DEPTH] = kread(jj + _K_DEPTH)
            se = (fx.Int32(side[3 * s][0]) >> 23) & 0xFF
            acc = _mfma_fp8(lt, q8[blk], sacc[s], se, qs)
            sacc[s] = _mfma_fp8(lt, q8l[blk], acc, se, qsl)
        for s in range_constexpr(2):
            for i in range_constexpr(2):
                sacc[s] = _mfma_bf16(side[3 * s + 1 + i], qr[i], sacc[s])

        for s in range_constexpr(2):
            sg = fx.Int32(Vec(side[3 * s])[0])
            sg = _imin(_imax(sg, fx.Int32(1)), fx.Int32(_SCALE_CAP_BITS))
            sg = (ixq_c[s] >= 0).select(sg, fx.Int32(0))
            _lds_store(Vec.from_elements([sg], fx.Int32), T.i32, ks_wr + 64 * s, 4)

        svs = [
            [
                _lds_load(T.f32, 4, ks_rd + (wp * 32 + 16 * s) * 4, 16)
                for wp in range_constexpr(4)
            ]
            for s in range_constexpr(2)
        ]
        sl = []
        tmax = neg_inf
        for s in range_constexpr(2):
            for r in range_constexpr(4):
                x = fx.Float32(Vec(sacc[s])[r]) * c_log2
                x = (fx.Float32(svs[s][0][r]) != fx.Float32(0.0)).select(x, neg_inf)
                sl.append(x)
                tmax = _fmax(tmax, x)
        tmax = _xmax(tmax)
        _lds_store(Vec.from_elements([tmax], fx.Float32), T.f32, mx_wr, 4)
        rocdl.s_waitcnt(lgkmcnt=0)
        _bar(bars, "tile")

        mx = _lds_load(T.f32, 4, mx_rd, 16)

        def pv_reads(d0):
            res = []
            for i in range_constexpr(4):
                dc = d0 + i
                parts = []
                for hf in range_constexpr(2):
                    for sub in range_constexpr(2):
                        parts.append(_tr8(v_rd[sub] ^ (16 * dc), hf * 8192))
                res.append(parts)
            return res

        m_new = m_old
        for jj in range_constexpr(4):
            m_new = _fmax(m_new, fx.Float32(mx[jj]))
        alpha = (m_old - m_new).exp2(fastmath="afn")
        m_cl = _fmax(m_new - _P_SHIFT, fx.Float32(-1.0e20))
        pl = [(sl[k] - m_cl).exp2(fastmath="afn") for k in range(8)]
        psum = fx.Float32(0.0)
        for x in pl:
            psum = psum + x
        l_new = l_old * alpha + psum
        o = [
            Vec.from_elements(
                [fx.Float32(Vec(o[i])[jj]) * alpha for jj in range(4)], fx.Float32
            )
            for i in range(8)
        ]

        _p_store_f16(pl, svs, p_wr, poff, p_wave_bytes)
        rocdl.s_waitcnt(lgkmcnt=0)
        _bar(bars, "tile")

        pp = _pv_operands(p_rd, H)

        va0 = pv_reads(0)
        _pv_mfma(0, va0, pp, o)
        va1 = pv_reads(4)
        rocdl.s_waitcnt(lgkmcnt=0)
        _guarded(not_last, lambda _ixd=ixd_n: dma(_ixd))
        _pv_mfma(4, va1, pp, o)
        side_n = ld_side(ixq_n)
        ixd_nn = ld_d_idx(t2)

        fin = yield [m_new, l_new] + o + ixq_nn + ixd_nn + side_n + ixq_n
    assert bars == {k: n for k, n in BARRIERS.items() if n and k != "end"}, bars

    rocdl.s_waitcnt(vmcnt=0)
    l_w = _xsum(fx.Float32(fin[1]))
    _lds_store(Vec.from_elements([l_w], fx.Float32), T.f32, mx_wr, 4)
    rocdl.s_waitcnt(lgkmcnt=0)
    _bar(bars, "end")
    lv = _lds_load(T.f32, 4, mx_rd, 16)
    l_tot = (
        fx.Float32(lv[0]) + fx.Float32(lv[1]) + fx.Float32(lv[2]) + fx.Float32(lv[3])
    )
    has_l = l_tot > fx.Float32(0.0)
    inv_l = has_l.select(fx.Float32(1.0) / l_tot, fx.Float32(0.0))
    pr = sj * S + qi
    o_elem = (pr * H + h_safe) * KVL + w * 128 + 4 * g
    for dc in range_constexpr(8):
        ov = Vec(fin[2 + dc])
        of = Vec.from_elements(
            [fx.Float32(ov[jj]) * inv_l for jj in range(4)], fx.Float32
        )
        val = of.to(fx.BFloat16).bitcast(fx.Int32)

        def _st(_v=val, _off=(o_elem + dc * 16) // 2):
            o_t.store(_off, _v, vec_size=2)

        _guarded(h_ok, _st)

    m_fin = fx.Float32(fin[0])
    lse2 = m_fin + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0)))) - _P_SHIFT
    lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
    lane0 = h_ok.select(w + g, fx.Int32(1)) == 0

    lse_bits = lse.bitcast(fx.Int32)

    def _st_lse(_v=lse_bits, _off=pr * H + t):
        if const_expr(cm):
            bo.buffer_store(_v, lse_t.rsrc, _off, cache_modifier=cm)
        else:
            lse_t.store(_off, _v)

    _guarded(lane0, _st_lse)
    assert bars == {k: n for k, n in BARRIERS.items() if n}, bars


_BUILDS: dict = {}


def build(key: DsaAttnBuild):
    """The compiled launch for `key` (cached)."""
    if key in _BUILDS:
        return _BUILDS[key]
    check(key)
    items, cta0 = geometry(key)
    grid = items if key.standalone else BLOCKS
    threads = _BLOCK_THREADS if key.standalone else 2 * _BLOCK_THREADS
    nbytes = lds_bytes(key)

    npad = 1 if key.standalone else 16384

    @fx.struct
    class Smem:
        pad: fx.Array[fx.Uint8, npad, 16]
        raw: fx.Array[fx.Uint8, nbytes, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_dsa_attn_s{key.tokens}_h{key.heads}_p{key.splits}"
        f"_i{int(key.has_invalid)}"
    )
    name += "_fold" if key.fold else ""
    name += "" if key.standalone else "_stage"

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def dsa_attn(
        q: Int64,
        stride_q: Int32,
        stride_qh: Int32,
        kv: Int64,
        idx: Int64,
        iptr: Int64,
        topk: Int64,
        topk_st: Int32,
        bt: Int64,
        part_o: Int64,
        part_lse: Int64,
        c_log2: Int32,
        idx_bytes: Int32,
        bsr: Int32,
        bsm: Int64,
    ):
        _ = keyed
        lds = fx.SharedAllocator().allocate(Smem).peek()
        c = {
            "tid": fx.thread_idx.x,
            "lds": lds.raw.ptr,
            "args": {
                "q": q,
                "stride_q": stride_q,
                "stride_qh": stride_qh,
                "kv": kv,
                "idx": idx,
                "iptr": iptr,
                "topk": topk,
                "topk_st": topk_st,
                "bt": bt,
                "part_o": part_o,
                "part_lse": part_lse,
                "c_log2": c_log2,
                "idx_bytes": idx_bytes,
                "bsr": bsr,
                "bsm": bsm,
            },
        }
        if const_expr(key.standalone):
            cta(c, key, fx.Int32(fx.block_idx.x))
        else:
            c["args"].update(stride_q=NH * ENTRY, stride_qh=ENTRY, topk_st=TOPK)
            bid = fx.block_idx.x
            if bid >= cta0:
                cta(c, key, bid - cta0)

    @flyc.jit
    def launch(
        q: Int64,
        stride_q: Int32,
        stride_qh: Int32,
        kv: Int64,
        idx: Int64,
        iptr: Int64,
        topk: Int64,
        topk_st: Int32,
        bt: Int64,
        part_o: Int64,
        part_lse: Int64,
        c_log2: Int32,
        idx_bytes: Int32,
        bsr: Int32,
        bsm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        dsa_attn(
            q,
            stride_q,
            stride_qh,
            kv,
            idx,
            iptr,
            topk,
            topk_st,
            bt,
            part_o,
            part_lse,
            c_log2,
            idx_bytes,
            bsr,
            bsm,
        ).launch(grid=(grid,), block=(threads,), stream=stream)

    launch.compile_hints = {"waves_per_eu": 2, "fast_fp_math": False}
    ABI.check(dsa_attn, launch)
    _BUILDS[key] = launch
    return launch


def fold_word(bs: int) -> int:
    """Block size bs -> (multiplier | bs << BS_POS | log2 << LG_POS) for the fold."""
    assert 0 < bs < 1 << BS_BITS, bs
    lg = (bs - 1).bit_length()
    m = -(-(1 << (31 + lg)) // bs)
    assert m < 1 << BS_POS, (bs, m)
    return m | bs << BS_POS | lg << LG_POS


def f32_bits(x: float) -> int:
    return int(torch.tensor(float(x), dtype=torch.float32).view(torch.int32))


def kv_rows(kv_cache, block_size: int) -> tuple[torch.Tensor, int]:
    """The KV cache as flat 656 B rows and the rows a block."""
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    u8 = kv_cache.view(torch.uint8)
    assert u8.size(-1) == RECORD_BYTES and u8.stride(-1) == 1
    return flat_kv_row_view(u8, block_size)


def host_args(
    key: DsaAttnBuild,
    *,
    q,
    kv,
    iptr,
    scale,
    idx=None,
    topk=None,
    bt=None,
    bsr=0,
    bs=0,
):
    """Kernel arguments for `key` from the query, cache and top-k tensors."""
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    assert q.shape[1:] == (key.heads, ENTRY) and q.size(0) >= key.tokens
    assert q.stride(0) % 2 == 0 and q.stride(1) % 2 == 0
    assert kv.dtype == torch.uint8 and kv.shape[-1] == RECORD_BYTES
    assert kv.stride(0) == RECORD_BYTES and kv.data_ptr() % 16 == 0
    assert iptr.dtype == torch.int32 and iptr.numel() >= key.tokens + 1
    out = {
        "q": q.data_ptr(),
        "stride_q": q.stride(0),
        "stride_qh": q.stride(1),
        "kv": kv.data_ptr(),
        "iptr": iptr.data_ptr(),
        "c_log2": f32_bits(scale * 1.4426950408889634),
        "idx": 0,
        "topk": 0,
        "topk_st": 0,
        "bt": 0,
        "bsr": 0,
        "bsm": 0,
    }
    if key.fold:
        assert idx is None and topk.dtype == torch.int32 and bt.dtype == torch.int32
        assert topk.size(-1) == TOPK and topk.stride(-1) == 1
        assert topk.size(0) >= key.tokens and bt.dim() == 1 and bt.stride(0) == 1
        assert 0 < bs <= bsr < 1 << 31
        assert bt.numel() < 1 << 29
        out.update(
            topk=topk.data_ptr(),
            topk_st=topk.stride(0),
            bt=bt.data_ptr(),
            idx_bytes=bt.numel() * 4,
            bsr=bsr,
            bsm=fold_word(bs),
        )
    else:
        assert topk is None and bt is None
        assert idx.dtype == torch.int32 and idx.is_contiguous()
        assert idx.numel() < 1 << 29
        out.update(idx=idx.data_ptr(), idx_bytes=idx.numel() * 4)
    return out


def dsa_attn_launch(key: DsaAttnBuild, *, part_o, part_lse, **inputs):
    """Launch `dsa_attn` for `key` into `part_o` / `part_lse`."""
    so, sl = partial_shapes(key)
    assert part_o.shape == so and part_o.dtype == torch.bfloat16
    assert part_o.is_contiguous()
    assert part_lse.shape == sl and part_lse.dtype == torch.float32
    assert part_lse.is_contiguous()
    a = host_args(key, **inputs)
    a.update(part_o=part_o.data_ptr(), part_lse=part_lse.data_ptr())
    build(key)(*ABI.pack(a), stream=torch.cuda.current_stream())
