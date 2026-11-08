# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
# The CTA body is the one of vllm/models/kimi_k3/amd/ops/
# flydsl_sparse_mla_gfx950.py (the stock FlyDSL sparse MLA, from the ROCm/aiter
# project, MIT license), its helpers imported from there unchanged. Only the
# row selection (a slice of a row's top-k, not the whole row), the argument
# passing (raw addresses) and the outputs (a slice's partial o and LSE) differ.
"""``dsa_attn``: Kimi-K3.1's DSA sparse MLA decode at TP8 + DCP1 (12 heads a
rank, the fp8_ds_mla cache's 656 B records), split over the top-k: every
slice's partial (o normalized, the LSE) for M1b's combine (``mla_back``'s
``splits``), which merges them, then W_UV and the gate (K2).

One launch, a CTA a (row t, slice j) of ``splits`` (P) slices a row: slice j
attends to the row's indices ``idx[iptr[t] + j C : iptr[t] + min((j + 1) C,
n_t)]`` (C = ``chunk``; n_t = ``iptr[t + 1] - iptr[t]``, at most 2048), one
128-token tile at P = 16. The math is the stock kernel's: Q split to fp8 hi +
lo against the raw fp8 latent (E8M0 the record's pow2 scale), the bf16 rope
QK, P = exp2(s - m + 8), the V scale folded into P' (fp16), V^T from LDS
(``ds_read_b64_tr_b8``). So P = 1 is the stock kernel, bit for bit.

The partials: ``part_o`` [P, S, 12, 512] bf16 (f32 with ``part_fp32``),
``part_lse`` [P, S, 12] f32, natural log; an empty slice (j C >= n_t: rows
shorter than 2048) leaves o 0 and LSE -inf.

``standalone`` builds are their own launch (M2); otherwise ``cta`` is a K2
stage (``layer``'s ``dsa`` 2): item v on CTA ``geometry``'s cta0 + v of K2's
512 threads, waves 0 .. 3 the body, waves 4 .. 7 its barriers only
(``_shadow``), the partials stored device-scope for K2's combine on other
CTAs. ``fold`` reads the indexer's logical top-k row and the request's block
table instead of the converter's rows: slot ``bt[tok // bs] * bsr + tok % bs``
(``triton_convert_req_index_to_global_index``'s, its invalid entries 0), the
block size ``bs`` a launch's (the engine sets it after the builds).
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
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import CM_DEV, traced
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    KernelAbi,
    key_tuple,
    source_digest,
)

TOPK = 2048  # the indexer's top-k: a row's indices at most
BS_SHIFT = 26  # the fold's ``bsr`` word: pitch | log2(block size) << BS_SHIFT
TILE = _TILE  # the body's tokens a tile
KVL = 512  # kv_lora_rank
ENTRY = 576  # the query's row a head: the latent's 512, the rope's 64
NH = 12  # heads a rank at TP8

_STREAM = fx.Stream(None)
_SOURCES = source_digest(".")


@dataclass(frozen=True)
class DsaAttnBuild:
    tokens: int = 8  # S, the verify rows (one request: qlen)
    heads: int = NH  # a rank's; the body serves <= MAX_HEADS
    splits: int = 16  # P: slices a row
    chunk: int = 128  # C: indices a slice (P C >= 2048)
    tile: int = _TILE  # the body's tokens a tile (C <= tile: a slice one tile)
    has_invalid: bool = False  # mask indices < 0 (the converter writes none)
    fold: bool = False  # the logical -> physical index fold in-kernel
    standalone: bool = True  # its own launch (256 threads), not a K2 stage
    part_fp32: bool = False  # debug: f32 partial o


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
    )
)

# the body's barriers (``_bar``): before the tile loop, a tile, after it. The
# K2 stage's waves 4 .. 7 (``_shadow``) take exactly these.
BARRIERS = {"pre": 0, "tile": 3, "end": 1}


def check(key: DsaAttnBuild) -> None:
    assert 1 <= key.heads <= MAX_HEADS and RECORD_BYTES == 656
    assert key.tokens * key.splits <= BLOCKS
    assert key.chunk * key.splits >= TOPK and key.chunk <= key.tile == _TILE


def geometry(key: DsaAttnBuild) -> tuple[int, int]:
    """(items, cta0): the (row, slice) items, on CTAs cta0 .. BLOCKS - 1 in K2
    (item v on CTA cta0 + v; a row's P items consecutive, over every XCD)."""
    items = key.tokens * key.splits
    return items, BLOCKS - items


def partial_shapes(key: DsaAttnBuild):
    p, s, h = key.splits, key.tokens, key.heads
    return (p, s, h, KVL), (p, s, h)


def lds_bytes(key: DsaAttnBuild) -> int:
    """KS | MX | SINK | V (4 column blocks x 128 tokens x 128 B) | P."""
    return _P_OFF + 4 * 16 * key.heads * 16


def _bar(bars: dict, phase: str) -> None:
    """The body's barrier, counted at trace time for ``_shadow``."""
    bars[phase] = bars.get(phase, 0) + 1
    gpu.barrier()


@traced
def _slice(c, key: DsaAttnBuild, v):
    """Item v's row qi, slice sj, its indices' first position r_beg in ``src``
    (the converter's rows, or with ``fold`` the logical top-k), r_len of them
    and nt tiles (wave-uniform). The body's and ``_shadow``'s, the same loads."""
    P, C = key.splits, key.chunk
    a = c["args"]
    lane = fx.Int32(c["tid"]) % 64
    qi = fx.Int32(v) // P  # the query row
    sj = fx.Int32(v) % P  # its slice
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

    # ---- tile count. has_invalid: tiles past the last entry >= 0 are
    # skipped (the scan loop runs 0 times otherwise); -1 entries before it
    # are masked as usual.
    n_scan = ((r_len + 255) // 256) if key.has_invalid else fx.Int32(0)
    zero = _raw(fx.Int32(0).ir_value())
    for sc_c, sc_st in range(0, n_scan, 1, init=[zero, zero]):
        lastp = fx.Int32(sc_st[0])
        p0 = fx.Int32(sc_c) * 256 + lane * 4
        vv = Vec(src.load(r_beg + p0, vec_size=4))
        for jj in range_constexpr(4):
            pj = p0 + jj
            lastp = (pj < r_len).select(
                (fx.Int32(vv[jj]) >= 0).select(pj + 1, lastp), lastp
            )
        sc_fin = yield [lastp, fx.Int32(sc_st[1])]
    lastp = fx.Int32(sc_fin[0]) if key.has_invalid else r_len
    for sh in (1, 2, 4, 8, 16, 32):
        o_ = fx.Int32(lastp).shuffle_xor(sh, 64)
        lastp = (o_ > lastp).select(o_, lastp)
    nt = _readfirstlane((lastp + (_TILE - 1)) // _TILE)
    return qi, sj, src, r_beg, r_len, nt


@traced
def cta(c, key: DsaAttnBuild, v):
    """Item v (row ``v // P``, slice ``v % P``): the stock kernel's body over
    the slice on waves 0 .. 3, its partial o / LSE stored. A K2 stage's waves
    4 .. 7 take the body's barriers (``_shadow``); the caller publishes."""
    if const_expr(key.standalone):
        _body(c, key, v)
    else:
        if _readfirstlane(fx.Int32(c["tid"]) // 64) < 4:
            _body(c, key, v)
        else:
            _shadow(c, key, v)


@traced
def _shadow(c, key: DsaAttnBuild, v):
    """Waves 4 .. 7 of a K2 stage: the body's barriers (``BARRIERS``) over the
    item's nt tiles, nothing else. Bare ``s_barrier``s: back-to-back
    ``gpu.barrier``s fold into one (MLIR's canonicalizer), the waves then
    out of step with the body's."""
    nt = _slice(c, key, v)[-1]
    for _ in range_constexpr(BARRIERS["pre"]):
        rocdl.s_barrier()
    zero = _raw(fx.Int32(0).ir_value())
    for _tile, st in range(0, nt, 1, init=[zero]):
        for _ in range_constexpr(BARRIERS["tile"]):
            rocdl.s_barrier()
        _fin = yield [fx.Int32(st[0])]
    for _ in range_constexpr(BARRIERS["end"]):
        rocdl.s_barrier()


@traced
def _body(c, key: DsaAttnBuild, v):
    """The stock kernel's body over item v's slice, waves 0 .. 3."""
    H, S = key.heads, key.tokens
    has_invalid = key.has_invalid
    a = c["args"]
    tid = fx.Int32(c["tid"])
    w = _readfirstlane(tid // 64)
    lane = tid % 64
    t = lane % 16  # QK: token row; S/O: head
    g = lane // 16
    p_wave_bytes = 16 * H * 16  # P bytes per destination wave w'
    bars: dict = {}

    q_t = GTensor(a["q"], dtype=T.i32, shape=(-1,))
    kv_base = buf_base_i64(a["kv"])  # i64: the cache may span more than 4 GiB
    # a K2 stage's partials are read by other CTAs (other XCDs' L2s)
    cm = 0 if key.standalone else CM_DEV
    o_t = GTensor(a["part_o"], dtype=T.i32, shape=(-1,), cache_modifier=cm)
    lse_t = GTensor(a["part_lse"], dtype=T.i32, shape=(-1,), cache_modifier=cm)
    base = _readfirstlane(fx.ptrtoint(c["lds"]))
    c_log2 = fx.Int32(a["c_log2"]).bitcast(fx.Float32)
    stride_q, stride_qh = a["stride_q"], a["stride_qh"]
    qi, sj, src, r_beg, r_len, nt = _slice(c, key, v)
    if const_expr(key.fold):
        # the request's block table row, its idx_bytes / 4 blocks
        bt_t = _BoundedI32(a["bt"], a["idx_bytes"])
        nblk = fx.Int32(a["idx_bytes"]) // 4
        # the pitch, log2 of the block size in bits BS_SHIFT ..
        bsr = fx.Int32(a["bsr"]) & ((1 << BS_SHIFT) - 1)
        bs_log2 = fx.Int32(a["bsr"]) >> BS_SHIFT

    # ---- Q prologue: lane (h = t, g) holds all 128 dims of block g as fp8 hi/lo
    h_ok = t < H
    h_safe = h_ok.select(t, fx.Int32(0))
    q_row = (qi * stride_q + h_safe * stride_qh) // 2  # i32 units (bf16 pairs)
    q8, qs, q8l, qsl = _q_prologue(q_t, q_row, g, h_ok)
    qr = [
        Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4))
        for i in range_constexpr(2)
    ]

    # Latent tile in LDS: 4 column blocks (bytes 128b..+128 of the record),
    # each [128 tokens][8 x 16 B] row-major with the 16-B chunk
    # XOR-swizzled by fsw(tok).
    def fsw(tok):
        return ((tok >> 1) & 7) ^ (((tok >> 4) & 1) << 1)

    k_rdb = []
    for s in range_constexpr(2):
        tok = w * 32 + 16 * s + t
        k_rdb.append(
            base + _V_OFF + (g // 2) * 16384 + tok * 128 + ((g % 2) ^ fsw(tok)) * 16
        )
    # PV tr8 read: lane (t, g) row i = t // 2 ->
    # token 32(g // 2 + 2hf) + 16(i // 4) + 4(2(g % 2) + sub) + i % 4,
    # chunk dc of block w (addr ^ 16dc), byte half t % 2.
    v_rd = []
    for sub in range_constexpr(2):
        i = t // 2
        tok = (g // 2) * 32 + (i // 4) * 16 + ((g % 2) * 2 + sub) * 4 + i % 4
        v_rd.append(base + _V_OFF + w * 16384 + tok * 128 + fsw(tok) * 16 + (t % 2) * 8)
    # DMA: load j fills tokens 8j..8j + 7 of block w;
    # lane -> token 8j + lane // 8, chunk (lane % 8) ^ f.
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
        """Slot of row position p; -1 past the slice end (and for -1 entries)."""
        ix = fx.Int32(src[r_beg + p])
        if const_expr(key.fold):
            # the converter's slot: block table entry, then the row in it;
            # an invalid token (or block) its 0, -1 (masked) with has_invalid
            blk = ix >> bs_log2
            ok = (ix >= 0) & (blk < nblk)
            row = fx.Int32(bt_t[ok.select(blk, fx.Int32(0))]) * bsr + (
                ix - (blk << bs_log2)
            )
            ix = ok.select(row, fx.Int32(-1 if has_invalid else 0))
        return (p < r_len).select(ix, fx.Int32(-1))

    def ld_q_idx(tb):
        return [ld_idx(w * 32 + t + tb + 16 * s) for s in range(2)]

    def ld_d_idx(tb):
        return [ld_idx(lane + 64 * hh + tb) for hh in range(2)]

    def ld_side(ixs):
        """Per-subtile pow2 scale (1 i32) and rope (2 x 4 i32) for the QK rows."""
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
            # The slot id is what gets permuted; its byte offset needs 64 bits.
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

    # Prologue: tile 0 latent via DMA, tile 0 side data, indices of tile 1.
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
    ns = 10  # loop-state slots before the index/side-data ones
    for tile, st in range(0, nt, 1, init=init):
        m_old = fx.Float32(st[0])
        l_old = fx.Float32(st[1])
        o = [Vec(st[2 + i]) for i in range(8)]
        ixq_n = [fx.Int32(st[ns + i]) for i in range(2)]
        ixd_n = [fx.Int32(st[ns + 2 + i]) for i in range(2)]
        side = [Vec(st[ns + 4 + i]) for i in range(6)]
        ixq_c = [fx.Int32(st[ns + 10 + i]) for i in range(2)]
        tb = fx.Int32(tile) * _TILE
        not_last = fx.Int32(tile) < nt - 1

        # This tile's latent has landed (own DMA) and is visible (barrier).
        rocdl.s_waitcnt(vmcnt=0)
        _bar(bars, "tile")
        t2 = nxt(tb, 2)
        ixq_nn = ld_q_idx(t2)

        # ---- QK (A = K rows from LDS, B = Q^T) -> S^T:
        # lane (h = t, g) holds tokens 16s + 4g + r.
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

        # This lane's token scale for slice w' = g, as f32 bits; 0 marks a
        # masked token. A valid token keeps >= 1 (a zero-scale record
        # attends like a zero latent, as in the Gluon path) and <= 2^7
        # (fp16 headroom of P' = P * scale).
        for s in range_constexpr(2):
            sg = fx.Int32(Vec(side[3 * s])[0])
            sg = _imin(_imax(sg, fx.Int32(1)), fx.Int32(_SCALE_CAP_BITS))
            sg = (ixq_c[s] >= 0).select(sg, fx.Int32(0))
            _lds_store(Vec.from_elements([sg], fx.Int32), T.i32, ks_wr + 64 * s, 4)

        # S-layout token scales: svs[s][w'] = 4 f32 for tokens 16s + 4g + r.
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
        # All masked so far: exp2(-1e30 + 1e20) = 0.
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

        # ---- PV for slice w: O^T[d, h] += V^T[d, tok] P'^T[tok, h]
        pp = _pv_operands(p_rd, H)

        va0 = pv_reads(0)
        _pv_mfma(0, va0, pp, o)
        va1 = pv_reads(4)
        # Column block w is read only by this wave's PV: refill it for the
        # next tile.
        rocdl.s_waitcnt(lgkmcnt=0)
        _guarded(not_last, lambda _ixd=ixd_n: dma(_ixd))
        _pv_mfma(4, va1, pp, o)
        side_n = ld_side(ixq_n)
        ixd_nn = ld_d_idx(t2)

        fin = yield [m_new, l_new] + o + ixq_nn + ixd_nn + side_n + ixq_n
    assert bars == {k: n for k, n in BARRIERS.items() if n and k != "end"}, bars

    # ---- epilogue: l over the 4 waves, normalize, store the partial for h < H
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
    # the partial's row: slice-major [P, S, H, 512]
    pr = sj * S + qi
    o_elem = (pr * H + h_safe) * KVL + w * 128 + 4 * g
    for dc in range_constexpr(8):
        ov = Vec(fin[2 + dc])
        of = Vec.from_elements([fx.Float32(ov[jj]) * inv_l for jj in range(4)],
                               fx.Float32)  # fmt: skip
        if const_expr(key.part_fp32):
            val = of.bitcast(fx.Int32)

            def _st(_v=val, _off=o_elem + dc * 16):
                o_t.store(_off, _v, vec_size=4)

        else:
            val = of.to(fx.BFloat16).bitcast(fx.Int32)

            def _st(_v=val, _off=(o_elem + dc * 16) // 2):
                o_t.store(_off, _v, vec_size=2)

        _guarded(h_ok, _st)

    # Natural-log LSE of softmax_scale * q.k over the unmasked keys; -inf when
    # there is none.
    m_fin = fx.Float32(fin[0])
    lse2 = m_fin + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0)))) - _P_SHIFT
    lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
    lane0 = h_ok.select(w + g, fx.Int32(1)) == 0

    lse_bits = lse.bitcast(fx.Int32)

    def _st_lse(_v=lse_bits, _off=pr * H + t):
        if const_expr(cm):
            # GTensor's scalar store drops the cache modifier
            bo.buffer_store(_v, lse_t.rsrc, _off, cache_modifier=cm)
        else:
            lse_t.store(_off, _v)

    _guarded(lane0, _st_lse)
    assert bars == {k: n for k, n in BARRIERS.items() if n}, bars


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: DsaAttnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    check(key)
    items, cta0 = geometry(key)
    grid = items if key.standalone else BLOCKS
    # not standalone: K2's stage alone (its 512 threads, waves 4 .. 7 the
    # shadow), for the tests
    threads = _BLOCK_THREADS if key.standalone else 2 * _BLOCK_THREADS
    nbytes = lds_bytes(key)

    # a K2 stage alone: K2's grid (the items on its top CTAs), its LDS past
    # others', its constant strides
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
    name += "_f32" if key.part_fp32 else ""
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
        ).launch(grid=(grid,), block=(threads,), stream=stream)

    # the stock kernel's: two waves a SIMD, IEEE math
    launch.compile_hints = {"waves_per_eu": 2, "fast_fp_math": False}
    ABI.check(dsa_attn, launch)
    _BUILDS[key] = launch
    return launch


def f32_bits(x: float) -> int:
    return int(torch.tensor(float(x), dtype=torch.float32).view(torch.int32))


def kv_rows(kv_cache, block_size: int) -> torch.Tensor:
    """The fp8_ds_mla cache [blocks, block_size, 656] as flat [rows, 656] u8
    records, block b's at rows b x its pitch .. (``flat_kv_row_view``: other
    groups' pages may sit between two blocks; the indices skip them)."""
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    u8 = kv_cache.view(torch.uint8)
    assert u8.size(-1) == RECORD_BYTES and u8.stride(-1) == 1
    return flat_kv_row_view(u8, block_size)[0]


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
    """ABI's arguments but the partials: ``q`` [S, 12, 576] bf16 (rows and
    heads strided, dims dense), ``kv`` the flat [rows, 656] u8 records
    (``kv_rows``), ``iptr`` [S + 1] int32 (row t's n_t indices from
    ``iptr[t]``), ``scale`` the softmax's. Without ``fold``: ``idx`` int32
    flat physical rows (row t's ``idx[iptr[t]:iptr[t + 1]]``). With it:
    ``topk`` the indexer's logical [>= S, 2048] int32, ``bt`` the request's
    block table row (int32, dense), ``bsr`` the flat rows a block
    (``flat_kv_row_view``'s pitch) and ``bs`` the block size (a power of 2)."""
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
    }
    if key.fold:
        assert idx is None and topk.dtype == torch.int32 and bt.dtype == torch.int32
        assert topk.size(-1) == TOPK and topk.stride(-1) == 1
        assert topk.size(0) >= key.tokens and bt.dim() == 1 and bt.stride(0) == 1
        assert 0 < bs <= bsr < 1 << BS_SHIFT and bs & (bs - 1) == 0
        assert bt.numel() < 1 << 29
        out.update(
            topk=topk.data_ptr(),
            topk_st=topk.stride(0),
            bt=bt.data_ptr(),
            idx_bytes=bt.numel() * 4,
            bsr=bsr | (bs.bit_length() - 1) << BS_SHIFT,
        )
    else:
        assert topk is None and bt is None
        assert idx.dtype == torch.int32 and idx.is_contiguous()
        assert idx.numel() < 1 << 29
        out.update(idx=idx.data_ptr(), idx_bytes=idx.numel() * 4)
    return out


def dsa_attn_launch(key: DsaAttnBuild, *, part_o, part_lse, **inputs):
    """Host launcher: ``host_args``' inputs, the partials of
    ``partial_shapes(key)`` (dense, bf16 / f32 o, f32 LSE)."""
    so, sl = partial_shapes(key)
    po_dt = torch.float32 if key.part_fp32 else torch.bfloat16
    assert part_o.shape == so and part_o.dtype == po_dt and part_o.is_contiguous()
    assert part_lse.shape == sl and part_lse.dtype == torch.float32
    assert part_lse.is_contiguous()
    a = host_args(key, **inputs)
    a.update(part_o=part_o.data_ptr(), part_lse=part_lse.data_ptr())
    build(key)(*ABI.pack(a), stream=torch.cuda.current_stream())
