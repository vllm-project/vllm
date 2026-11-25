# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/sgl-project/sglang/pull/47371
"""Speculative-decode verify (full or sliding-window attention) on gfx950 in FlyDSL.

Up to 8 verify tokens x GQA 8 (64 query columns) per KV head (for fewer tokens the
unused columns duplicate the last token and their tiles are trimmed), one wave per KV
head, one workgroup per (request, split); BF16 Q, FP8 e4m3 paged KV
`[pages, 16 or 32, kv_heads, 64]`, split-KV FP32 partials. QK and PV run on BF16
MFMAs (K and V upcast exactly from FP8).

Transposed form per wave: S^T = K Q^T (keys x query columns) and O^T = V^T P^T on
16x16x32 BF16 MFMAs.
  * K is the A operand straight from HBM: lane (g, i) loads 16 contiguous bytes (dims
    16g..16g+15) of key i of a 16-key half chunk; dims are permuted consistently in Q
    (k-step ks holds dims 16g + 8ks + 0..7).
  * Softmax runs along keys = MFMA rows: in-thread over the 8 rows of a lane, across the
    four 16-lane groups with two permlane swaps, and only when some score clears the
    running max by 2^8 (lazy rescale, exact: P <= 256).
  * P^T's accumulator registers are the PV B operand once the 32 keys of a chunk are
    relabelled (k-position 8g + i <-> key i < 4 ? 4g + i : 12 + 4g + i); V is written to
    wave-private LDS in that order with contiguous ds_write_b128 and read back as the
    V^T A operand with ds_read_b64_tr_b8 (16-byte chunks XOR-swizzled by row so the
    transposed reads are bank-conflict free).
  * Page ids are loaded one chunk group ahead of use, as vector loads in order with K
    and V.
  * K / V go through buffer loads (32-bit byte offsets), or 64-bit global loads when a
    layer's pool reaches 2^31 bytes.
  * NS register load slots refilled right after use keep ~NS chunks in flight per wave.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.flydsl_kernels import _run_compiled
from vllm.v1.attention.ops.flydsl_kernels.buffer_utils import (
    buffer_view,
    load_vec,
    store_vec,
)

QL = 8
GQA = 8
HD = 64
CHUNK = 32
NS = 2  # load slots per wave
LOG2E = 1.4426950408889634
# Lazy rescale threshold (log2 domain): P = exp2(s - m_used) <= 2^8, exact in FP32
# accumulators.
RESCALE = 8.0
# Finite "minus infinity" for the running max so fully masked chunks give exp2(-inf - m)
# = 0, never NaN.
M_INIT = -1.0e30
LDS_CU = 160 * 1024  # gfx950 LDS per CU


def _raw(v):
    return v if isinstance(v, ir.Value) else v.ir_value()


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _fence():
    # keep the K / V / page-id loads in issue order (their vmcnt waits depend on it)
    # but let ALU, MFMA and LDS work move across chunk boundaries
    fx.rocdl.sched_barrier(0x78F)


def _swap(op_name, x):
    """Partner values across lane groups.

    permlane{32,16}_swap of x with itself -> (vdst, vsrc) pair, i32.
    """
    i32 = ir.IntegerType.get_signless(32)
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(x)
    sw = getattr(_rocdl, op_name)(pair, raw, raw, False, True)
    return llvm.extractvalue(i32, sw, [0]), llvm.extractvalue(i32, sw, [1])


def _allreduce_groups(x_f32, use_max):
    """Reduce an f32 across lanes l, l^16, l^32, l^48 (the four 16-lane groups)."""
    v = _raw(x_f32)
    for op in ("permlane32_swap", "permlane16_swap"):
        bits = arith.bitcast(T.i32, v)
        a, b = _swap(op, bits)
        fa = arith.bitcast(T.f32, a)
        fb = arith.bitcast(T.f32, b)
        v = arith.maximumf(fa, fb) if use_max else arith.addf(fa, fb)
    return v


def _cvt8(lo_word, hi_word):
    """8 FP8 e4m3 bytes (two i32) -> v8bf16 in byte order (exact upcast, scale 1)."""
    one = _f32(1.0)
    parts = []
    for w in (lo_word, hi_word):
        for sel in (False, True):
            pr = _rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), _raw(w), one, sel)
            parts.append(fx.Vector(pr).bitcast(fx.Int32)[0])
    return _raw(fx.Vector.from_elements(parts, fx.Int32).bitcast(fx.BFloat16))


def _global_load16(addr):
    """16 bytes (vector<4 x i32>) at the 64-bit global byte address `addr`."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), _raw(addr)).result
    return llvm.LoadOp(T.vec(4, T.i32), ptr, alignment=16).result


def _sload(view, index):
    """Wave-uniform i32 element of a buffer view (scalar load into an SGPR)."""
    return fx.rocdl.s_buffer_load(fx.get_iter(view), index)


def _v2(x):
    return fx.Vector.from_elements([fx.Float32(x), fx.Float32(x)], fx.Float32)


def _half(vec4, hi):
    return fx.Vector(vec4, shape=4, dtype=fx.Float32).shuffle(
        fx.Vector(vec4, shape=4, dtype=fx.Float32), [2, 3] if hi else [0, 1]
    )


def _max4(vec):
    return arith.maximumf(
        arith.maximumf(_raw(vec[0]), _raw(vec[1])),
        arith.maximumf(_raw(vec[2]), _raw(vec[3])),
    )


@functools.lru_cache(maxsize=64)
def _launcher(
    kvh,
    kv_s0,
    kv_s1,
    kv_s2,
    scale,
    waves_per_eu,
    wg_per_cu,
    addr64,
    balanced,
    ql,
    window,
    page,
):
    # window > 0: sliding-window layer, query position p sees keys (p - window, p].
    # ql query tokens per request: column n = li + 16 nt holds token n // 8, tokens
    # >= ql clamped to ql - 1 (duplicate columns: same loads and mask, identical
    # values). Only the NT = ceil(ql / 2) column tiles holding real tokens are computed.
    assert 1 <= ql <= QL
    NT = (ql + 1) // 2
    # loop-carried state: O accumulators (4 x NT), negm / thr / l (NT each), then the
    # load slots and page ids
    O_END, N_END, T_END, L_END = 4 * NT, 5 * NT, 6 * NT, 7 * NT
    heads = GQA * kvh
    lds_wave = 2 * CHUNK * HD  # two V stages of 32 rows x 64 bytes per wave
    exp_scale = float(scale) * LOG2E
    s0w, s1w = kv_s0 // 4, kv_s1 // 4  # dword strides

    @fx.struct
    class Lds:
        # occupancy pin: V stages padded so at most wg_per_cu workgroups share a CU (the
        # grid is sized for exactly that; when registers allow 2 waves / SIMD the
        # dispatcher otherwise stacks workgroups unevenly)
        v: fx.Array[fx.Int8, max(kvh * lds_wave, LDS_CU // wg_per_cu - 1024), 16]

    @flyc.kernel(name="gqa_verify_fly")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        table: fx.Tensor,
        seqlens: fx.Tensor,
        kd: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        wgmap: fx.Tensor,
        splits: fx.Int32,
        t_stride: fx.Int32,
        q_s0: fx.Int32,
    ):
        work = fx.block_idx.x
        tid = fx.thread_idx.x
        h = tid // 64
        lane = tid % 64
        g = lane // 16
        li = lane % 16
        if fx.const_expr(balanced):
            # length-balanced: workgroup `work` takes chunks [begin, end) of request seq
            # from the per-step map (balance_map: every request owns ceil(units / T)
            # workgroups of <= T chunk groups); idle workgroups get begin == end and
            # write a slot nobody reads. Partials are indexed by workgroup: [work,
            # token, head].
            wm_v = buffer_view(wgmap, fx.Int32)
            seq = _sload(wm_v, work * 4)
            bal_begin = _sload(wm_v, work * 4 + 1)
            bal_end = _sload(wm_v, work * 4 + 2)
            split = work
        else:
            seq = work // splits
            split = work % splits

        q_v = buffer_view(q, fx.BFloat16, 8)
        tb_v = buffer_view(table, fx.Int32)
        if fx.const_expr(addr64):
            k_addr = fx.Int64(fx.ptrtoint(fx.get_iter(k)))
            v_addr = fx.Int64(fx.ptrtoint(fx.get_iter(v)))
        else:
            k_v = buffer_view(k, fx.Int32, 4)
            v_v = buffer_view(v, fx.Int32, 4)
        p_v = buffer_view(part, fx.Float32, 4)
        m_v = buffer_view(maxi, fx.Float32)
        d_v = buffer_view(den, fx.Float32)

        kvlen = _sload(buffer_view(seqlens, fx.Int32), seq)
        kd_bits = _sload(buffer_view(kd, fx.Int32), fx.Int32(0))
        c_scale = arith.mulf(arith.bitcast(T.f32, _raw(kd_bits)), _f32(exp_scale))
        ctx = kvlen - ql
        # clamped at 0 so zero-length (padding) requests stay inside their own row
        last_page = (kvlen - 1) // page
        last_page = (last_page > 0).select(last_page, fx.Int32(0))
        nchunks = (kvlen + CHUNK - 1) // CHUNK
        last_chunk = (nchunks > 0).select(nchunks - 1, fx.Int32(0))
        if fx.const_expr(window):
            # lowest key the first query token sees, and the first chunk every query
            # token sees whole
            low = ctx - (window - 1)
            low = (low > 0).select(low, fx.Int32(0))
            first = low // CHUNK
            whole = (kvlen - window + CHUNK - 1) // CHUNK
        if fx.const_expr(balanced):
            begin = bal_begin
            end = (bal_end < nchunks).select(bal_end, nchunks)
        else:
            # chunks per split rounded up to whole slot groups, so only the last busy
            # split runs dummy chunks
            span = nchunks - first
            cps = ((span + splits - 1) // splits + NS - 1) // NS * NS
            begin = first + split * cps
            end = (begin + cps < nchunks).select(begin + cps, nchunks)
        begin = (begin < end).select(begin, end)
        start = begin
        if fx.const_expr(window):
            # chunk groups below `whole` are masked by the window
            lim = (whole < end).select(whole, end)
            head_groups = (lim - begin + NS - 1) // NS
            head_groups = (head_groups > 0).select(head_groups, fx.Int32(0))
            start = begin + head_groups * NS
        # chunks below `interior` are visible to every query position: no masks
        interior = (ctx + 1) // CHUNK
        interior = (interior < end).select(interior, end)
        interior = (interior > start).select(interior, start)
        row_base = seq * t_stride

        def col_tok(n):
            # query token of column n, clamped to the last real token when ql < 8
            t = n // GQA
            if fx.const_expr(ql < QL):
                t = (t < ql - 1).select(t, fx.Int32(ql - 1))
            return t

        # Q^T fragments (B operand): column n = li + 16 nt -> token n // 8, head h * 8 +
        # n % 8; k-step ks holds dims 16 g + 8 ks + 0..7
        qf = []
        for nt in range_constexpr(NT):
            n = li + 16 * nt
            qrow = seq * ql + col_tok(n)
            qhead = h * GQA + n % GQA
            for ks in range_constexpr(2):
                qf.append(
                    _raw(load_vec(q_v, qrow * q_s0 + qhead * HD + g * 16 + ks * 8, 8))
                )
        c_pair = _v2(c_scale)
        inv_c = arith.divf(_f32(1.0), c_scale)

        lds = fx.SharedAllocator().allocate(Lds).peek()
        vptr = lds.v.ptr
        lds_base = h * lds_wave

        def page_ids(chunk):
            # the pages of the chunk's two 16-key halves (one 32-token page holds
            # both), clamped to the request's last page; vector loads, in order with the
            # K / V loads, so the vmcnt wait for an older chunk covers them
            p0 = chunk * (CHUNK // page)
            p1 = p0 + (CHUNK // page - 1)
            p0 = (p0 < last_page).select(p0, last_page)
            p1 = (p1 < last_page).select(p1, last_page)
            return fx.Int32(tb_v[row_base + p0]), fx.Int32(tb_v[row_base + p1])

        def clamp(chunk):
            return (chunk < last_chunk).select(chunk, last_chunk)

        # per-lane dword offsets inside a page: K row li, dims 16 g..; V load u fills
        # LDS row kp = 16 u + lane // 4 (k-position order) at physical 16-byte chunk
        # lane % 4
        k_off = li * s1w + h * (kv_s2 // 4) + g * 4
        v_keys = []
        v_offs = []
        for u in range_constexpr(2):
            kp = lane // 4 + 16 * u
            gp = kp // 8
            ip = kp % 8
            key = (ip < 4).select(gp * 4 + ip, gp * 4 + ip + 12)
            logical = fx.Int32(arith.xori(_raw(lane % 4), _raw((kp // 4) % 4)))
            v_keys.append(key)
            v_offs.append((key % page) * s1w + h * (kv_s2 // 4) + logical * 4)

        if fx.const_expr(addr64):
            # per-lane 64-bit base addresses; a load adds its page's 64-bit byte offset
            k_lane = k_addr + fx.Int64(k_off * 4)
            v_lane = [v_addr + fx.Int64(v_offs[u] * 4) for u in range_constexpr(2)]

        def issue(pids):
            p0, p1 = pids
            loads = []
            for mt in range_constexpr(2):
                pid = p0 if mt == 0 else p1
                row = (16 * mt % page) * s1w  # second half of a 32-token page
                if fx.const_expr(addr64):
                    loads.append(
                        _global_load16(k_lane + fx.Int64(pid) * kv_s0 + row * 4)
                    )
                else:
                    loads.append(_raw(load_vec(k_v, pid * s0w + k_off + row, 4)))
            for u in range_constexpr(2):
                pid = (v_keys[u] < 16).select(p0, p1)
                if fx.const_expr(addr64):
                    loads.append(_global_load16(v_lane[u] + fx.Int64(pid) * kv_s0))
                else:
                    loads.append(_raw(load_vec(v_v, pid * s0w + v_offs[u], 4)))
            return loads

        def prepare(chunk, loaded, stage, masked):
            # consume a load slot: V rows to wave-private LDS (k-position order,
            # swizzled), K to A fragments
            kraw = loaded[:2]
            vraw = loaded[2:]
            if fx.const_expr(masked):
                # zero V rows outside the request (or below the window) so stale FP8
                # bytes (NaN encodings) cannot reach P.V
                zero = fx.full(4, 0, fx.Int32)
                vraw_new = []
                for u in range_constexpr(2):
                    key = chunk * CHUNK + v_keys[u]
                    keep = key < kvlen
                    if fx.const_expr(window):
                        keep = keep & (key >= low)
                    vraw_new.append(
                        _raw(
                            keep.select(
                                fx.Vector(vraw[u], shape=4, dtype=fx.Int32), zero
                            )
                        )
                    )
                vraw = vraw_new
            vbase = lds_base + stage * (CHUNK * HD)
            for u in range_constexpr(2):
                fx.ptr_store(
                    fx.Vector(vraw[u], shape=4, dtype=fx.Int32),
                    vptr + fx.Int64(vbase + u * 1024 + lane * 16),
                )
            ka = []
            for mt in range_constexpr(2):
                words = fx.Vector(kraw[mt], shape=4, dtype=fx.Int32)
                ka.append(_cvt8(words[0], words[1]))
                ka.append(_cvt8(words[2], words[3]))
            return ka, vbase

        def scores(chunk, ka, masked):
            # S^T = K Q^T: tile (mt, nt) rows = keys 16 mt + 4 g + j, column li + 16 nt
            ninf = fx.Float32(float("-inf"))
            # k-step outer: the 8 tiles' first MFMAs are independent, so the dependent
            # second ones never stall
            accs = [_raw(fx.full(4, 0.0, fx.Float32)) for _ in range(2 * NT)]
            for ks in range_constexpr(2):
                for mt in range_constexpr(2):
                    for nt in range_constexpr(NT):
                        accs[mt * NT + nt] = fx.rocdl.mfma_f32_16x16x32_bf16(
                            T.f32x4,
                            [
                                ka[mt * 2 + ks],
                                qf[nt * 2 + ks],
                                accs[mt * NT + nt],
                                0,
                                0,
                                0,
                            ],
                        )
            s = []
            for mt in range_constexpr(2):
                for nt in range_constexpr(NT):
                    sv = fx.Vector(accs[mt * NT + nt], shape=4, dtype=fx.Float32)
                    if fx.const_expr(masked):
                        qpos = ctx + col_tok(li + 16 * nt)
                        elems = []
                        for j in range_constexpr(4):
                            key = chunk * CHUNK + mt * 16 + g * 4 + j
                            ok = (key <= qpos) & (chunk < end)
                            if fx.const_expr(window):
                                ok = ok & (key + window > qpos)
                            elems.append(ok.select(sv[j], ninf))
                        sv = fx.Vector.from_elements(elems, fx.Float32)
                    s.append(sv)
            return s

        def finish(chunk, ka, vbase, o_acc, negm, thr, l_run, masked):
            # running state per column tile nt: negm = -m_used as an f32 pair (the exp
            # fma operand), thr = (m_used + RESCALE) / c (lazy-rescale trigger in raw
            # score units), l = f32 pair of partial denominators
            s = scores(chunk, ka, masked)
            lmax = []
            need = None
            for nt in range_constexpr(NT):
                mx = arith.maximumf(_max4(s[nt]), _max4(s[NT + nt]))
                lmax.append(mx)
                hit = arith.cmpf(arith.CmpFPredicate.OGT, mx, _raw(thr[nt]))
                need = hit if need is None else arith.ori(need, hit)
            vote = fx.Int64(_rocdl.ballot(T.i64, need))
            if vote != fx.Int64(0):
                negm_new = []
                thr_new = []
                l_new = []
                o_new = list(o_acc)
                for nt in range_constexpr(NT):
                    mx = _allreduce_groups(lmax[nt], True)
                    m_old = arith.subf(
                        _f32(0.0),
                        _raw(fx.Vector(negm[nt], shape=2, dtype=fx.Float32)[0]),
                    )
                    mn = arith.maximumf(m_old, arith.mulf(mx, c_scale))
                    alpha = fx.rocdl.exp2(T.f32, arith.subf(m_old, mn))
                    negm_new.append(_raw(_v2(arith.subf(_f32(0.0), mn))))
                    thr_new.append(arith.mulf(arith.addf(mn, _f32(RESCALE)), inv_c))
                    l_new.append(
                        _raw(
                            fx.Vector(l_run[nt], shape=2, dtype=fx.Float32)
                            * fx.Float32(alpha)
                        )
                    )
                    for mt in range_constexpr(4):
                        o_new[mt * NT + nt] = _raw(
                            fx.Vector(o_acc[mt * NT + nt], shape=4, dtype=fx.Float32)
                            * fx.Float32(alpha)
                        )
                negm = negm_new
                thr = thr_new
                l_run = l_new
                o_acc = o_new
            # P = exp2(s * c - m_used); P^T B fragment of column tile nt = (tile
            # (0, nt), tile (1, nt))
            pf = []
            l_out = []
            for nt in range_constexpr(NT):
                nm = fx.Vector(negm[nt], shape=2, dtype=fx.Float32)
                ps: list = []
                for mt in range_constexpr(2):
                    for hi in range_constexpr(2):
                        z = fx.Vector(
                            fmath.fma(
                                _raw(_half(s[mt * NT + nt], hi)),
                                _raw(c_pair),
                                _raw(nm),
                            ),
                            shape=2,
                            dtype=fx.Float32,
                        )
                        ps.extend(
                            fx.Float32(fx.rocdl.exp2(T.f32, _raw(z[j])))
                            for j in range(2)
                        )
                t = fx.Vector.from_elements(
                    ps[:4], fx.Float32
                ) + fx.Vector.from_elements(ps[4:], fx.Float32)
                l_out.append(
                    _raw(
                        fx.Vector(l_run[nt], shape=2, dtype=fx.Float32)
                        + (_half(t, False) + _half(t, True))
                    )
                )
                pf.append(_raw(fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)))
            # O^T += V^T P^T, V^T fragments read transposed from LDS just in time
            out = [_raw(x) for x in o_acc]
            kp = g * 8 + li // 2
            swz = (kp // 4) % 4
            for mt in range_constexpr(4):
                phys = fx.Int32(arith.xori(_raw(fx.Int32(mt)), _raw(swz)))
                addr = vbase + kp * HD + phys * 16 + (li % 2) * 8
                raw = _rocdl.ds_read_tr8_b64(
                    T.vec(2, T.i32), fx.to_llvm_ptr(vptr + fx.Int64(addr))
                ).result
                pair = fx.Vector(raw, shape=2, dtype=fx.Int32)
                vf = _cvt8(pair[0], pair[1])
                for nt in range_constexpr(NT):
                    out[mt * NT + nt] = fx.rocdl.mfma_f32_16x16x32_bf16(
                        T.f32x4, [vf, pf[nt], out[mt * NT + nt], 0, 0, 0]
                    )
            return out, [_raw(x) for x in negm], [_raw(x) for x in thr], l_out

        def group_step(c0, st, masked):
            # chunks c0 .. c0 + NS - 1 from load slots 0 .. NS - 1; slot j is refilled
            # with chunk c0 + NS + j right after use, with page ids loaded one group
            # earlier (carried)
            o_a, n_a, t_a, l_a = (
                list(st[:O_END]),
                list(st[O_END:N_END]),
                list(st[N_END:T_END]),
                list(st[T_END:L_END]),
            )
            pids = [fx.Int32(x) for x in st[L_END + 4 * NS : L_END + 6 * NS]]
            nxt = []
            for j in range_constexpr(NS):
                nxt.extend(page_ids(clamp(c0 + 2 * NS + j)))
            refill = []
            for j in range_constexpr(NS):
                ka, vb = prepare(
                    c0 + j, list(st[L_END + 4 * j : L_END + 4 + 4 * j]), j % 2, masked
                )
                refill.extend(issue((pids[2 * j], pids[2 * j + 1])))
                _fence()
                o_a, n_a, t_a, l_a = finish(c0 + j, ka, vb, o_a, n_a, t_a, l_a, masked)
                _fence()
            return o_a + n_a + t_a + l_a + refill + [_raw(x) for x in nxt]

        o0 = [_raw(fx.full(4, 0.0, fx.Float32)) for _ in range(4 * NT)]
        n0 = [_raw(_v2(_f32(-M_INIT))) for _ in range(NT)]
        t0 = [arith.mulf(_f32(M_INIT + RESCALE), inv_c) for _ in range(NT)]
        l0 = [_raw(fx.full(2, 0.0, fx.Float32)) for _ in range(NT)]

        # unmasked chunk groups, then the rest masked (chunks past the split end are
        # fully masked dummies); every group prefetches the next one, so the masked loop
        # starts from the interior loop's slots
        ngroups = (interior - start) // NS
        mid = start + ngroups * NS
        tail_groups = (end - mid + NS - 1) // NS
        slots0 = []
        ids0: list = []
        for j in range_constexpr(NS):
            slots0.extend(issue(page_ids(clamp(begin + j))))
        for j in range_constexpr(NS):
            ids0.extend(_raw(x) for x in page_ids(clamp(begin + NS + j)))
        init = o0 + n0 + t0 + l0 + slots0 + ids0
        if fx.const_expr(window):
            for gi, st in range(0, head_groups, 1, init=init):  # type: ignore[call-overload]
                c0 = begin + fx.Int32(arith.index_cast(T.i32, _raw(gi))) * NS
                fin0 = yield group_step(c0, st, True)
            init = list(fin0)
        for gi, st in range(0, ngroups, 1, init=init):  # type: ignore[call-overload]
            c0 = start + fx.Int32(arith.index_cast(T.i32, _raw(gi))) * NS
            fin = yield group_step(c0, st, False)
        for gi, st in range(0, tail_groups, 1, init=list(fin)):  # type: ignore[call-overload]
            c0 = mid + fx.Int32(arith.index_cast(T.i32, _raw(gi))) * NS
            fin2 = yield group_step(c0, st, True)
        o_acc, n_run, l_run = (
            list(fin2[:O_END]),
            list(fin2[O_END:N_END]),
            list(fin2[T_END:L_END]),
        )

        # epilogue: denominators summed over the four lane groups; all four groups hold
        # the same max / denom and store them unmasked (buffer stores here have no OOB
        # check)
        for nt in range_constexpr(NT):
            n = li + 16 * nt
            t = col_tok(n)
            qh = n % GQA
            if fx.const_expr(balanced):
                state = (work * ql + t) * heads + h * GQA + qh
            else:
                state = ((seq * ql + t) * heads + h * GQA + qh) * splits + split
            l2 = fx.Vector(l_run[nt], shape=2, dtype=fx.Float32)
            l_tot = _allreduce_groups(arith.addf(_raw(l2[0]), _raw(l2[1])), False)
            m_fin = arith.subf(
                _f32(0.0), _raw(fx.Vector(n_run[nt], shape=2, dtype=fx.Float32)[0])
            )
            m_v[state] = fx.Float32(m_fin)
            d_v[state] = fx.Float32(l_tot)
            for mt in range_constexpr(4):
                store_vec(
                    p_v,
                    state * HD + mt * 16 + g * 4,
                    fx.Vector(o_acc[mt * NT + nt], shape=4, dtype=fx.Float32),
                    4,
                )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        table: fx.Tensor,
        seqlens: fx.Tensor,
        kd: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        wgmap: fx.Tensor,
        splits: fx.Int32,
        t_stride: fx.Int32,
        q_s0: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            v,
            table,
            seqlens,
            kd,
            part,
            maxi,
            den,
            wgmap,
            splits,
            t_stride,
            q_s0,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(kvh * 64, 1, 1), stream=stream)

    return launch


def _partials(
    q,
    k,
    v,
    seq_lens,
    block_table,
    scale,
    k_scale,
    splits,
    *,
    waves_per_eu,
    wg_per_cu,
    query_len,
    wgmap=None,
    window=0,
):
    """FP32 split-KV partials plus max / denom (log2 domain).

    Sliding window (`wgmap` None): equal splits, partials `[tokens, heads, splits(,
    64)]`. Full attention (`wgmap` from `balance_map`): `splits` is the workgroup
    count G and the partials are `[G, query_len, heads(, 64)]`, indexed by workgroup.
    """
    import torch

    n = seq_lens.numel()
    heads, kvh = q.shape[1], k.shape[2]
    balanced = wgmap is not None
    assert balanced != bool(window)

    # buffer offsets are 32-bit byte offsets: a pool past them takes the 64-bit loads
    def _span(t):  # strided-view span in elements
        return 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))

    addr64 = max(_span(k), _span(v)) * k.element_size() >= 2**31
    st = (splits, query_len, heads) if balanced else (q.shape[0], heads, splits)
    part = torch.empty((*st, HD), dtype=torch.float32, device=q.device)
    maxi = torch.empty(st, dtype=torch.float32, device=q.device)
    den = torch.empty(st, dtype=torch.float32, device=q.device)
    _run_compiled(
        _launcher(
            kvh,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            float(scale),
            waves_per_eu,
            wg_per_cu,
            addr64,
            balanced,
            int(query_len),
            int(window),
            int(k.shape[1]),
        ),
        q,
        k.view(torch.uint8),
        v.view(torch.uint8),
        block_table,
        seq_lens,
        k_scale,
        part,
        maxi,
        den,
        wgmap if balanced else seq_lens,
        int(splits),
        int(block_table.stride(0)),
        int(q.stride(0)),
        int(splits if balanced else n * splits),
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return part, maxi, den


@triton.jit
def _reduce_splits(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    V_SCALE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    o_stride_0,
    o_stride_1,
):
    # log2-domain merge of a row's equal splits
    token, head = tl.program_id(0), tl.program_id(1)
    segments = tl.arange(0, SPLITS)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + segments
    maximum = tl.load(MAXIMUM + index)
    denom = tl.load(DENOM + index)
    global_max = tl.max(tl.where(denom > 0, maximum, float("-inf")), axis=0)
    global_max = tl.where(global_max > float("-inf"), global_max, 0.0)
    weights = tl.where(denom > 0, tl.exp2(maximum - global_max), 0.0)
    total = tl.sum(denom * weights, axis=0)
    partial = tl.load(PARTIAL + index[:, None] * 64 + d[None, :])
    accum = tl.sum(partial * weights[:, None], axis=0)
    inverse = tl.where(total > 0, 1.0 / total, 0.0) * tl.load(V_SCALE)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, accum * inverse)


def pick_window_splits(num_seqs: int, kv_heads: int, num_cus: int) -> int:
    """Splits per request of the sliding-window verify (a power of two). A window
    spans few chunks: one or two requests split it finely, larger batches use fewer,
    longer splits once the grid covers the CUs.
    """
    if num_seqs <= 2:
        return 64 // num_seqs
    if num_seqs * 16 <= num_cus:
        return 16
    return 8 if num_seqs * kv_heads * 2 <= num_cus else 4


def gqa_verify_window(
    q,
    k,
    v,
    out,
    seq_lens,
    block_table,
    scale,
    k_scale,
    v_scale,
    *,
    query_len,
    window,
    num_cus,
):
    """Sliding-window verify with equal splits per request, merged into `out`."""
    splits = pick_window_splits(seq_lens.numel(), k.shape[2], num_cus)
    partial, maximum, denom = _partials(
        q,
        k,
        v,
        seq_lens,
        block_table,
        scale,
        k_scale,
        splits,
        waves_per_eu=0,
        wg_per_cu=2,
        query_len=query_len,
        window=window,
    )
    _reduce_splits[(q.shape[0], q.shape[1])](
        partial,
        maximum,
        denom,
        out,
        v_scale,
        q.shape[1],
        splits,
        out.stride(0),
        out.stride(1),
        num_warps=2,
    )


# -------------------------------------------------------------------------------------
# Length-balanced grid for full attention: a fixed grid of G workgroups (one or two per
# CU) shares the step's total KV chunks evenly instead of giving every request the
# same number of splits. Request r (u_r = ceil(chunks_r / NS) chunk groups) owns w_r =
# ceil(u_r / T) consecutive workgroups, T = the smallest per-workgroup load with sum_r
# w_r <= G (bisection on device); a request never shares a workgroup, so the slowest
# workgroup carries T groups <= ideal + (rounding: at most one partially filled
# workgroup per request). The map and the per-request workgroup offsets are computed on
# device from seq_lens by a small Triton kernel (no host sync; grid fixed = G), the
# partials are indexed by workgroup and the reduce walks each request's variable
# workgroup range.
# -------------------------------------------------------------------------------------


@triton.jit
def _balance_map(
    SEQLENS,
    WGMAP,
    WGSTART,
    n,
    G: tl.constexpr,
    NS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    # grid cdiv(G, BLOCK_G) programs: each recomputes the (tiny) per-request split and
    # fills BLOCK_G map rows
    r = tl.arange(0, BLOCK_N)
    rm = r < n
    seqlen = tl.load(SEQLENS + r, mask=rm, other=0)
    chunks = (seqlen + 31) // 32
    units = (chunks + (NS - 1)) // NS
    total = tl.sum(units, axis=0)
    lo = tl.maximum((total + (G - 1)) // G, 1)
    # T = ceil(total / (G - n)) always fits (sum ceil(u / T) <= total / T + n <= G), so
    # the bisection range is ~total * n / G^2 wide: a handful of iterations. Float
    # quotients are exact here (units < 2^22).
    hi = tl.maximum(
        tl.minimum((total + (G - n - 1)) // (G - n), tl.max(units, axis=0)), lo
    )
    uf = units.to(tl.float32)
    while lo < hi:
        mid = (lo + hi) // 2
        cnt = tl.sum(tl.math.ceil(uf / mid.to(tl.float32)).to(tl.int32), axis=0)
        ok = cnt <= G
        hi = tl.where(ok, mid, hi)
        lo = tl.where(ok, lo, mid + 1)
    w = (units + hi - 1) // hi
    # chunks per workgroup of this request (spread evenly)
    per = ((units + w - 1) // tl.maximum(w, 1)) * NS
    end_incl = tl.cumsum(w, axis=0)
    start = end_incl - w
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(WGSTART + r, start, mask=rm)
        tl.store(WGSTART + n + r, tl.zeros_like(start) + tl.sum(w, axis=0), mask=r == 0)
    p = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    pm = p < G
    own = (p[:, None] >= start[None, :]) & (p[:, None] < end_incl[None, :])
    seq = tl.sum(tl.where(own, r[None, :], 0), axis=1)
    st = tl.sum(tl.where(own, start[None, :], 0), axis=1)
    pr = tl.sum(tl.where(own, per[None, :], 0), axis=1)
    ch = tl.sum(tl.where(own, chunks[None, :], 0), axis=1)
    b = (p - st) * pr
    e = tl.minimum(b + pr, ch)
    e = tl.maximum(e, b)
    tl.store(WGMAP + p * 4, seq, mask=pm)
    tl.store(WGMAP + p * 4 + 1, b, mask=pm)
    tl.store(WGMAP + p * 4 + 2, e, mask=pm)


@triton.jit
def _reduce_balanced(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    V_SCALE,
    WGSTART,
    HEADS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    o_stride_0,
    o_stride_1,
    QLEN: tl.constexpr,
):
    # log2-domain merge, online over the request's variable workgroup range
    token, head = tl.program_id(0), tl.program_id(1)
    seq = token // QLEN
    t = token % QLEN
    s0 = tl.load(WGSTART + seq)
    s1 = tl.load(WGSTART + seq + 1)
    d = tl.arange(0, 64)
    m_run = tl.full((), -1.0e30, tl.float32)
    total = tl.zeros((), tl.float32)
    accum = tl.zeros((64,), tl.float32)
    for s in range(s0, s1, BLOCK_S):
        slot = s + tl.arange(0, BLOCK_S)
        sm = slot < s1
        index = (slot * QLEN + t) * HEADS + head
        maximum = tl.load(MAXIMUM + index, mask=sm, other=-1.0e30)
        denom = tl.load(DENOM + index, mask=sm, other=0.0)
        valid = denom > 0
        m_new = tl.maximum(m_run, tl.max(tl.where(valid, maximum, -1.0e30), axis=0))
        alpha = tl.exp2(m_run - m_new)
        weights = tl.where(valid, tl.exp2(maximum - m_new), 0.0)
        partial = tl.load(
            PARTIAL + index[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0
        )
        accum = accum * alpha + tl.sum(partial * weights[:, None], axis=0)
        total = total * alpha + tl.sum(denom * weights, axis=0)
        m_run = m_new
    inverse = tl.where(total > 0, 1.0 / total, 0.0) * tl.load(V_SCALE)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, accum * inverse)


MAP_BLOCK_G = 16
MAP_WARPS = 4
TWO_WG_PER_CU_MIN_REQUESTS = 24
# with 2 or fewer KV heads (one wave each) a single workgroup per CU leaves SIMDs idle
TWO_WG_PER_CU_MIN_REQUESTS_FEW_HEADS = 3


def pick_balanced(
    num_seqs: int, kv_heads: int, num_cus: int
) -> tuple[int, int, int] | None:
    """(groups, wg_per_cu, waves_per_eu) of the balanced grid for a batch, or None
    when the batch has too many requests for it (each needs its own workgroup).

    Fixed per batch size: CUDA-graph safe. A workgroup runs one wave per KV head. One
    workgroup per CU suits 4 or more KV heads below 24 requests; two per CU
    (waves_per_eu 2: twice the loads in flight) from 24 requests up, and from 3
    requests up with 2 or fewer KV heads (no waves_per_eu hint: the LDS pin already
    allows only one wave per SIMD).
    """
    if kv_heads <= 2:
        two = num_seqs >= TWO_WG_PER_CU_MIN_REQUESTS_FEW_HEADS
        pick = (2, 0) if two else (1, 1)
    else:
        pick = (2, 2) if num_seqs >= TWO_WG_PER_CU_MIN_REQUESTS else (1, 1)
    groups = pick[0] * num_cus
    return (groups, *pick) if num_seqs < groups else None


def balance_map(seq_lens, groups, out=None):
    """Balanced-grid map, computed on device: `wgmap` `[G, 4]` int32 = (request, first
    chunk, end chunk, -) and `wgstart` `[n + 1]` int32. `out` optionally gives
    buffers of at least those sizes to fill.
    """
    import torch

    n = seq_lens.numel()
    assert n < groups
    if out is None:
        wgmap = torch.empty((groups, 4), dtype=torch.int32, device=seq_lens.device)
        wgstart = torch.empty(n + 1, dtype=torch.int32, device=seq_lens.device)
    else:
        wgmap, wgstart = out[0][:groups], out[1][: n + 1]
    _balance_map[(triton.cdiv(groups, MAP_BLOCK_G),)](
        seq_lens,
        wgmap,
        wgstart,
        n,
        groups,
        NS,
        max(16, triton.next_power_of_2(n)),
        MAP_BLOCK_G,
        num_warps=MAP_WARPS,
    )
    return wgmap, wgstart


def gqa_verify_balanced(
    q,
    k,
    v,
    out,
    seq_lens,
    block_table,
    scale,
    k_scale,
    v_scale,
    *,
    query_len,
    grid,
    maps,
):
    """Full-attention verify on the length-balanced grid, merged into `out`.

    `grid` is `pick_balanced`'s (groups, wg_per_cu, waves_per_eu); `maps` the
    `balance_map` of `seq_lens` for those groups.
    """
    groups, wg_per_cu, waves_per_eu = grid
    wgmap, wgstart = maps
    partial, maximum, denom = _partials(
        q,
        k,
        v,
        seq_lens,
        block_table,
        scale,
        k_scale,
        groups,
        waves_per_eu=waves_per_eu,
        wg_per_cu=wg_per_cu,
        query_len=query_len,
        wgmap=wgmap,
    )
    _reduce_balanced[(q.shape[0], q.shape[1])](
        partial,
        maximum,
        denom,
        out,
        v_scale,
        wgstart,
        q.shape[1],
        16,
        out.stride(0),
        out.stride(1),
        query_len,
        num_warps=2,
    )
