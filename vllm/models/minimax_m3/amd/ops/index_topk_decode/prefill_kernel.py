# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index top-k for MiniMax-M3 (FlyDSL, gfx950).

One 64-lane workgroup per (row, head) returns the 16 highest-scoring block ids
in AITER's order, walking the row in windows of STRIPS x 64 blocks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import PRELOAD_HINTS

BLK = 128  # tokens per block
TOPK = 16
PREFILL_STRIPS = (4, 8, 16, 20, 26)
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # block index bits (slots * 64 <= 8192)
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _key(bits):
    """Signed key in AITER's order; NaNs land above +inf or below -inf."""
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _fold_nan(k):
    return ((k > KEY_POS_INF) | (k < KEY_NEG_INF)).select(fx.Int32(KEY_NEG_INF), k)


def _ballot(c):
    return fx.Int64(fx.rocdl.ballot(T.i64, c))


def _popc(m):
    """Set bits of a wave mask; two 32-bit counts keep the compare scalar."""
    lo = fx.math.ctpop(m.to(fx.Int32)).to(fx.Int32)
    hi = fx.math.ctpop((m >> 32).to(fx.Int32)).to(fx.Int32)
    return fx.Int32(lo) + fx.Int32(hi)


def _below(mask, base):
    """The set bits of ``mask`` below this lane, plus ``base`` (v_mbcnt)."""
    lo = mask.to(fx.Int32)
    hi = (mask >> 32).to(fx.Int32)
    r = fx.rocdl.mbcnt_lo(T.i32, lo.ir_value(), fx.Int32(base).ir_value())
    return fx.Int32(fx.rocdl.mbcnt_hi(T.i32, hi.ir_value(), r))


def _imax(a, b):
    return (a > b).select(a, b)


def _one(c):
    return c.select(fx.Int32(1), fx.Int32(0))


def _comp(k, i):
    """(key, index) as one i64 whose signed order is AITER's."""
    return (k.to(fx.Int64) << 32) | (i.to(fx.Int64) & 0xFFFFFFFF)


def compile_topk_prefill(strips: int, *, decode: bool = False):
    assert strips == 26 if decode else strips in PREFILL_STRIPS
    STRIPS = strips
    WIN = STRIPS * 64
    RANK_BLOCK = 8
    # Candidates, rank-loop padding, then one discard slot per lane.
    TRASH = CAP + RANK_BLOCK
    CAND = TRASH + 64
    name = f"m3_index_topk_{'decode' if decode else 'prefill'}_indices_w{STRIPS}"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * CAND, 16]
        sel: fx.Array[fx.Int32, 4 * TOPK, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_query_len: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, h = fx.block_idx.x, fx.block_idx.y
        lane = fx.thread_idx.x
        n64 = fx.Int32(n).to(fx.Int64)
        h64 = fx.Int32(h).to(fx.Int64)

        def rsrc(base, nbytes):
            """Buffer at an i64 base; per-row rebasing keeps offsets small."""
            return buffer_ops.create_buffer_resource_from_addr(
                base, num_records_bytes=fx.Int64(nbytes)
            )

        # Reads past the row's width return 0.
        score_row = arg_score + (h64 * i64_s_head + n64 * i64_s_row) * 4
        srs = rsrc(score_row, i32_width.to(fx.Int64) * 4)

        def load_window(srs, wbase):
            return Vec.from_elements(
                [
                    fx.Int32(
                        buffer_ops.buffer_load(
                            srs,
                            wbase + (j * 64) + lane,
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                    for j in range_constexpr(STRIPS)
                ],
                fx.Int32,
            )

        # The first window is loaded before the row's length is known.
        bits = load_window(srs, 0)

        def scalar_at(base, idx64, nbytes):
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc(base + idx64 * 4, nbytes),
                    fx.Int32(0),
                    vec_width=1,
                    is_scalar=True,
                )
            )

        if const_expr(decode):
            req = (fx.Int32(n) // i32_query_len).to(fx.Int64)
            seq = scalar_at(arg_nvp, req, fx.Int64(4))
            kv0 = seq - i32_query_len + fx.Int32(n) % i32_query_len + 1
            kv = (kv0 < 0).select(fx.Int32(0), kv0)
            nblk = (kv + BLK - 1) // BLK
        else:
            nblk = scalar_at(arg_nvp, n64, fx.Int64(4))
        nlive = (nblk < i32_width).select(nblk, i32_width)
        # Later windows read only the row's live blocks.
        srs_live = rsrc(score_row, nlive.to(fx.Int64) * 4)

        def view(ptr, n_dw, width):
            return fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
                fx.make_layout(width, 1),
            )

        pairs = view(smem.cand.ptr, 2 * CAND, 2)
        quads = view(smem.cand.ptr, 2 * CAND, 4)
        sels = view(smem.sel.ptr, 4 * TOPK, 4)
        atom64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def lds_put(atom, v, w, vals):
            r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
            r.store(Vec.from_elements(vals, fx.Int32))
            fx.copy(atom, r, v)

        def lds_get(atom, v, w):
            r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
            fx.copy(atom, v, r)
            x = Vec(r.load())
            return [fx.Int32(x[e]) for e in range_constexpr(w)]

        def put_pair(pos, i, k):
            lds_put(atom64, fx.slice(pairs, (None, pos)), 2, [i, k])

        def wave_sum(c):
            """Sum over the wave of a per-lane count in [0, 63]."""
            t = fx.Int32(0)
            for b in range_constexpr(6):
                t = t + (_popc(_ballot((c & (1 << b)) != 0)) << b)
            return t

        def compact(keys, wbase, base, pred):
            """Append the keys ``pred`` keeps at ``base`` if they fit below CAP."""
            oks = [
                _one(pred(keys[j], wbase + (j * 64) + lane))
                for j in range_constexpr(STRIPS)
            ]
            cnt = oks[0]
            for j in range_constexpr(1, STRIPS):
                cnt = cnt + oks[j]
            pos = fx.Int32(base)
            end = fx.Int32(base)
            for b in range_constexpr(6):  # cnt <= 32
                bm = _ballot((cnt & (1 << b)) != 0)
                pos = pos + (_below(bm, 0) << b)
                end = end + (_popc(bm) << b)
            # On overflow nothing is written; the caller rebuilds the candidates.
            if end <= CAP:
                for j in range_constexpr(STRIPS):
                    put_pair(
                        (oks[j] != 0).select(pos, TRASH + lane),
                        wbase + (j * 64) + lane,
                        keys[j],
                    )
                    pos = pos + oks[j]
            return end

        def exact_cut(keys, wbase):
            """The window's 16th key v and the index cut ie among the keys tied
            with it: (k > v) | (k == v & i >= ie) keeps exactly its top 16."""
            pe = fx.Int32(0)
            b = fx.Int32(31)
            while b >= 0:
                t = (pe | (fx.Int32(1) << b)) ^ INT_MIN
                c = fx.Int32(0)
                for j in range_constexpr(STRIPS):
                    c = c + _one(keys[j] >= t)
                pe = (wave_sum(c) >= TOPK).select(t ^ INT_MIN, pe)
                b = b - 1
            v = pe ^ INT_MIN
            c = fx.Int32(0)
            for j in range_constexpr(STRIPS):
                c = c + _one(keys[j] > v)
            need = TOPK - wave_sum(c)
            ie = fx.Int32(0)
            b = fx.Int32(IDX_BITS - 1)
            while b >= 0:
                t = ie | (fx.Int32(1) << b)
                c = fx.Int32(0)
                for j in range_constexpr(STRIPS):
                    c = c + _one((keys[j] == v) & ((wbase + (j * 64) + lane) >= t))
                ie = (wave_sum(c) >= need).select(t, ie)
                b = b - 1
            return v, ie

        def lane_max_bound(m):
            p = fx.Int32(0)
            for d in range_constexpr(12):
                t = p | _s32(1 << (31 - d))
                count = _popc(_ballot(m >= (t ^ INT_MIN)))
                p = (count >= TOPK).select(t, p)
            return p ^ INT_MIN

        def window(bits, wbase, nrun, tprev):
            """One window of keys merged into the running top 16 (sel)."""
            keys = []
            for j in range_constexpr(STRIPS):
                live = (wbase + (j * 64) + lane) < nlive
                keys.append(live.select(_key(bits[j]), fx.Int32(INT_MIN)))
            m = keys[0]
            for j in range_constexpr(1, STRIPS):
                m = _imax(m, keys[j])
            # Bound: the lane maxima in the first window, the running 16th later.
            tw = fx.Int32(INT_MIN)
            if nrun < TOPK:
                tw = lane_max_bound(m)
            ta = _imax(tw, tprev)
            total = compact(keys, wbase, nrun, lambda k, i: k >= ta)
            # Exact only if 16 keys above -inf bound it and no NaN is above +inf.
            nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
            if (total > CAP) | (ta <= KEY_NEG_INF) | nan_hi:
                wl = nlive - wbase
                if wl <= CAP - nrun:
                    # Every live block of the window is a candidate.
                    put_pair(
                        (lane < wl).select(nrun + lane, TRASH + lane),
                        wbase + lane,
                        _fold_nan(keys[0]),
                    )
                    total = nrun + wl
                else:
                    # NaNs to -inf and re-bound; exact search if still too many.
                    fixed = []
                    for j in range_constexpr(STRIPS):
                        live = (wbase + (j * 64) + lane) < nlive
                        fixed.append(live.select(_fold_nan(keys[j]), fx.Int32(INT_MIN)))
                    mf = fixed[0]
                    for j in range_constexpr(1, STRIPS):
                        mf = _imax(mf, fixed[j])
                    tf = _imax(lane_max_bound(mf), tprev)
                    total = compact(fixed, wbase, nrun, lambda k, i: k >= tf)
                    if total > CAP:
                        v, ie = exact_cut(fixed, wbase)
                        total = compact(
                            fixed,
                            wbase,
                            nrun,
                            lambda k, i: (k > v) | ((k == v) & (i >= ie)),
                        )
            if lane < RANK_BLOCK:
                put_pair(total + lane, fx.Int32(0), fx.Int32(INT_MIN))
            fx.gpu.barrier()
            mi, mk = lds_get(atom64, fx.slice(pairs, (None, lane)), 2)
            mw = _comp(mk, mi)
            rank = fx.Int32(0)
            for g in range_constexpr(CAP // RANK_BLOCK):
                if total > RANK_BLOCK * g:
                    es = [
                        lds_get(
                            atom128, fx.slice(quads, (None, RANK_BLOCK // 2 * g + u)), 4
                        )
                        for u in range_constexpr(RANK_BLOCK // 2)
                    ]
                    # All of a block's reads in flight before the compares.
                    fx.rocdl.sched_barrier(0)
                    for e in es:
                        rank = rank + _one(_comp(e[1], e[0]) > mw)
                        rank = rank + _one(_comp(e[3], e[2]) > mw)
            if (lane < total) & (rank < TOPK):
                lds_put(atom128, fx.slice(sels, (None, rank)), 4, [mi, mk, 0, mi])
            return (total < TOPK).select(total, fx.Int32(TOPK))

        wbase = fx.Int32(0)
        nrun = fx.Int32(0)
        tprev = fx.Int32(INT_MIN + 1)  # dead keys are INT_MIN
        while (wbase == 0) | (wbase < nlive):
            if wbase > 0:
                bits = load_window(srs_live, wbase)
                # The running top 16 back into cand; its 16th key bounds the window.
                fx.gpu.barrier()
                r = lds_get(atom128, fx.slice(sels, (None, lane & (TOPK - 1))), 4)
                if lane < nrun:
                    put_pair(lane, r[0], r[1])
                k15 = fx.Int32(fx.rocdl.readlane(T.i32, r[1], fx.Int32(TOPK - 1)))
                tprev = (nrun >= TOPK).select(k15, fx.Int32(INT_MIN + 1))
                fx.gpu.barrier()
            nrun = window(
                [fx.Int32(bits[j]) for j in range_constexpr(STRIPS)],
                wbase,
                nrun,
                tprev,
            )
            wbase = wbase + WIN

        fx.gpu.barrier()
        r = lds_get(atom128, fx.slice(sels, (None, lane & (TOPK - 1))), 4)
        blk = (lane < nrun).select(r[0], fx.Int32(-1))
        buffer_ops.buffer_store(
            blk,
            rsrc(arg_topk + (h64 * i64_t_head + n64 * i64_t_row) * 4, TOPK * 4),
            lane,
            mask=lane < TOPK,
        )

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_query_len: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
        i32_rows: fx.Int32,
        i32_heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            i64_s_head,
            i64_s_row,
            i32_width,
            arg_nvp,
            i32_query_len,
            arg_topk,
            i64_t_head,
            i64_t_row,
        ).launch(
            grid=(fx.Int64(i32_rows), fx.Int64(i32_heads), 1),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
