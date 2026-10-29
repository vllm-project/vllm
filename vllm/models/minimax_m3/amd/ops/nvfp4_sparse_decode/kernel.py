# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 sparse decode attention for MiniMax-M3 (FlyDSL, gfx950).

One workgroup = one query row x the 16 query heads of one KV head x one split of
the row's top-k blocks. K and V are dequantized in registers to
e4m3_rn_satfinite(e2m1 x block scale), the values AITER's FP8 sparse PA reads,
and widened to bf16; scores and softmax are fp32, P is bf16. Wave w owns keys
[32 w, 32 w + 32) of every block:

  S^T = K Q^T      v_mfma_f32_16x16x32_bf16, A = K, B = Q^T
  O^T += V^T P^T   A = V^T, B = P^T straight from the S^T accumulators

The waves combine through LDS. With ``splits`` > 1 every split stores its
unnormalized partial and the last split of a row to arrive combines them in
split order, so the result does not depend on arrival order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    PRELOAD_HINTS,
    SC1,
    as_f32,
    bits,
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
    exp2,
    global_i32_ptr,
    maxf_nn,
    mfma_16x16x32,
    perm,
    pk_bf16,
    xlane4,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.kernel import (
    BLK,
    HG,
    KPW,
    NDT,
    NKS,
    NW,
    RAW,
    D,
    kv_lane_loader,
)

PART_F = 36  # fp32 per (split, row, lane) partial: O (NDT x 4), max, sum, pad
SIGN4 = 0x80808080 - (1 << 32)
MAG4 = 0x07070707


def _nnan():
    return _ir.Attribute.parse("#arith.fastmath<nnan>")


def _bf16x2_e4m3(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` -> one dword of two bf16."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
            Vec.make_type(2, fx.BFloat16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    ).bitcast(fx.Int32)[0]


def _k_bf16x8(codes, tab_lo, tab_hi):
    """Eight K codes under one block scale -> four dwords of bf16 pairs.

    ``tab_lo`` / ``tab_hi`` hold the e4m3 keys of magnitudes 0-3 / 4-7 for the
    block scale (``nvfp4_e4m3_lut``).
    """
    even = perm(tab_hi, tab_lo, codes & MAG4) ^ ((codes << 4) & SIGN4)
    odd = perm(tab_hi, tab_lo, (codes >> 4) & MAG4) ^ (codes & SIGN4)
    lo = perm(odd, even, 0x05010400)
    hi = perm(odd, even, 0x07030602)
    return [
        _bf16x2_e4m3(lo, False),
        _bf16x2_e4m3(lo, True),
        _bf16x2_e4m3(hi, False),
        _bf16x2_e4m3(hi, True),
    ]


def _v_bf16x8(codes, scale_pairs):
    """One V dword (eight keys of one dim) -> four dwords of bf16 pairs.

    The fp16 products are exact; they are clamped to +-448 and rounded once to
    e4m3.
    """
    big = Vec.filled(2, 448.0, fx.Float16)
    small = Vec.filled(2, -448.0, fx.Float16)
    f8 = [None, None]
    for j in range_constexpr(4):
        v = cvt_fp4x2_f16(codes, j) * scale_pairs[j]
        v = _arith.MinNumFOp(
            fx.as_ir_value(v), fx.as_ir_value(big), fastmath=_nnan()
        ).result
        v = _arith.MaxNumFOp(v, fx.as_ir_value(small), fastmath=_nnan()).result
        old = Vec.filled(2, 0, fx.Int16) if j % 2 == 0 else f8[j // 2]
        f8[j // 2] = Vec(
            fx.rocdl.cvt_scalef32_pk_fp8_f16(
                Vec.make_type(2, fx.Int16),
                fx.as_ir_value(old),
                v,
                fx.as_ir_value(fx.Float32(1.0)),
                j % 2 == 1,
            )
        )
    out = []
    for h in range_constexpr(2):
        word = f8[h].bitcast(fx.Int32)[0]
        out += [_bf16x2_e4m3(word, False), _bf16x2_e4m3(word, True)]
    return out


def _global_atomic_add(addr_i64, value):
    """Relaxed agent-scope atomic add on a global i32; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def compile_nvfp4_sparse_decode(splits: int = 1, topk: int = 16, xcds: int = 0):
    """Build the decode kernel.

    Args:
        splits: workgroups per (row, KV head), each over a contiguous share of
            the row's blocks.
        topk: top-k width (one lane per slot).
        xcds: if > 0, workgroup w runs on XCD w % xcds and a request's rows and
            a row's splits are placed on one XCD.

    """
    assert 1 <= topk <= 64 and splits >= 1
    SPLITS, TOPK, XCDS = splits, topk, xcds
    # Split partials need sc1 unless all splits of a row run on one XCD.
    PART_SCOPE = 0 if XCDS else SC1
    name = f"m3_nvfp4_sparse_decode_k{TOPK}_s{SPLITS}" + (f"_x{XCDS}" if XCDS else "")

    @fx.struct
    class Shared:
        o: fx.Array[fx.Int32, NW * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * 2 * 64]
        lut: fx.Array[fx.Int32, 512]
        ticket: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk stride over KV heads
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_lut: fx.Int64,  # [256] int64 K magnitude tables
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # [grid rows * Hkv, S, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # [grid rows * Hkv] int32 tickets, zero
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        seq_p = global_i32_ptr(arg_seq)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        lut_g = global_i32_ptr(arg_lut)
        smem.lut.ptr[tx] = fx.Int32(lut_g[tx])
        smem.lut.ptr[tx + 256] = fx.Int32(lut_g[tx + 256])

        if const_expr(XCDS > 0):
            # grid.x is padded to XCDS x ql so a request's rows share an XCD.
            xcd = bx % XCDS
            slot = bx // XCDS
            row = ((slot // i32_ql) * XCDS + xcd) * i32_ql + slot % i32_ql
        else:
            row = bx
        row_ok = row < i32_rows
        rowc = row_ok.select(row, fx.Int32(0))
        req = rowc // i32_ql
        pos = fx.Int32(seq_p[req]) - i32_ql + rowc % i32_ql
        # Padding rows (seq_len 0 or past i32_rows) see nothing.
        pos = (row_ok & (pos >= 0)).select(pos, fx.Int32(-1))
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # Lane = top-k slot.
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[kvh * i32_topk_hs + rowc * TOPK + slot_ok.select(lane, fx.Int32(0))]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int64(fx.rocdl.ballot(T.i64, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))
        if const_expr(SPLITS > 1):
            below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below).to(fx.Int32))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            lo = fx.block_idx.z * chunk
            vmask = fx.Int64(
                fx.rocdl.ballot(T.i64, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))

        lim = pos % BLK - w * KPW
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        q_row = (rowc * i32_H + kvh * HG + n) * (D // 2) + g * 16
        qf = [
            fx.Vector(
                buffer_ops.buffer_load(qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32)
            )
            for ks in range_constexpr(NKS)
        ]

        load_block = kv_lane_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, g)

        def dequant_k(raw):
            kf = []
            for st in range_constexpr(2):
                tabs = []
                for grp in range_constexpr(2):
                    sb = raw[8].shrui(fx.Int32(8 * (2 * st + grp))) & fx.Int32(0xFF)
                    tabs.append(
                        (
                            fx.Int32(smem.lut.ptr[sb * 2]),
                            fx.Int32(smem.lut.ptr[sb * 2 + 1]),
                        )
                    )
                kf.append(
                    [
                        Vec.from_elements(
                            _k_bf16x8(raw[st * 4 + ks], *tabs[ks // 2]), fx.Int32
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )
            return kf

        def dequant_v(raw):
            sp = [
                cvt_fp8x2_f16(raw[17 + j // 2], j % 2 == 1) for j in range_constexpr(4)
            ]
            return [
                Vec.from_elements(_v_bf16x8(raw[9 + dt], sp), fx.Int32)
                for dt in range_constexpr(NDT)
            ]

        fx.gpu.barrier()  # LUT in LDS

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        first = (n_v > 0).select(
            fx.Int32(fx.math.cttz(vmask).to(fx.Int32)), fx.Int32(0)
        )

        def slot_at(mask):
            """The lowest slot left in ``mask``, the first one when empty."""
            return (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(kf, vf, cur_blk, live, m_t, l_t, o_t):
            """Fold one block into (max, sum, O); a dead block is fully masked."""
            a = []
            for st in range_constexpr(2):
                acc = zero4
                for ks in range_constexpr(NKS):
                    acc = mfma_16x16x32(kf[st][ks], qf[ks], acc, False)
                a += [acc[e] for e in range_constexpr(4)]
            diag = cur_blk == self_blk
            dead = live == fx.Int32(0)
            a = [
                (dead | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                for i in range_constexpr(8)
            ]
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = maxf_nn(mx, a[i])
            # The running max is kept in scaled units (c > 0).
            m_new = maxf_nn(m_t, xlane4(mx, maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            alpha = exp2(m_t - m_use)
            p = [exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            pf = Vec.from_elements(
                [pk_bf16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                fx.Int32,
            )
            av = Vec.filled(4, alpha, fx.Float32)
            o_t = [
                mfma_16x16x32(vf[dt], pf, o_t[dt] * av, False)
                for dt in range_constexpr(NDT)
            ]
            return m_new, l_t * alpha + ps, o_t

        # Two blocks per iteration, the next pair's loads in flight; an odd last
        # block is masked.
        init = [vmask, neg_inf, fx.Float32(0.0)]
        init += [zero4 for _ in range_constexpr(NDT)]
        init += load_block(page_of(slot_at(vmask)))
        n_st = 3 + NDT
        results = init
        for _, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + 1) // 2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m_t = fx.Float32(st_[1])
            l_t = fx.Float32(st_[2])
            o_t = [Vec(st_[3 + dt]) for dt in range_constexpr(NDT)]
            raw_a = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            blk_a = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem)))
            rem_b = rem & (rem - fx.Int64(1))
            raw_b = load_block(page_of(slot_at(rem_b)))
            fx.rocdl.sched_barrier(0)
            kf_a = dequant_k(raw_a)
            vf_a = dequant_v(raw_a)
            rem_n = rem_b & (rem_b - fx.Int64(1))
            fx.rocdl.sched_barrier(0)
            raw_n = load_block(page_of(slot_at(rem_n)))
            fx.rocdl.sched_barrier(0)
            res = attend(kf_a, vf_a, blk_a, fx.Int32(1), m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            blk_b = fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem_b)))
            live_b = (rem_b != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
            kf_b = dequant_k(raw_b)
            vf_b = dequant_v(raw_b)
            res = attend(kf_b, vf_b, blk_b, live_b, m_t, l_t, o_t)
            m_t, l_t, o_t = res[0], res[1], res[2]
            state = [rem_n, m_t, l_t] + o_t + raw_n
            results = yield [fx.as_ir_value(x) for x in state]

        def store_out(acc, inv):
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out,
                num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2),
            )
            o_row = (row * i32_H + kvh * HG + n) * (D // 2) + g * 16
            for e in range_constexpr(4):
                val = Vec.from_elements(
                    [
                        pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=row < i32_rows)

        # Combine the waves' partials in wave 0.
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        ml = smem.ml.ptr
        ml[(w * 2) * 64 + lane] = fx.Float32(results[1])
        ml[(w * 2 + 1) * 64 + lane] = fx.Float32(results[2])
        for dt in range_constexpr(NDT):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(Vec(results[3 + dt]).bitcast(fx.Int32))
            fx.copy(lds_atom, r, fx.slice(o_lds, (None, (w * NDT + dt) * 64 + lane)))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(o_lds, (None, (ww * NDT + dt) * 64 + lane)),
                        r,
                    )
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(SPLITS > 1):
                tid = row * i32_hkv + kvh
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_part, num_records_bytes=i64_part_bytes
                )
                row_f = (tid * SPLITS + fx.block_idx.z) * 64 + lane
                for dt in range_constexpr(NDT):
                    buffer_ops.buffer_store(
                        acc[dt].bitcast(fx.Int32),
                        pr,
                        row_f * PART_F + dt * 4,
                        cache_modifier=PART_SCOPE,
                    )
                buffer_ops.buffer_store(
                    Vec.from_elements([bits(big_raw), bits(l_tot)], fx.Int32),
                    pr,
                    row_f * PART_F + NDT * 4,
                    cache_modifier=PART_SCOPE,
                )
                # The partial is stored before the ticket is taken.
                fx.rocdl.s_waitcnt(vmcnt=0)
                if lane == 0:
                    smem.ticket.ptr[0] = _global_atomic_add(
                        arg_cnt + fx.Int64(tid) * 4, 1
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                if (
                    fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                    )
                    == SPLITS - 1
                ):
                    rows_f = [
                        (tid * SPLITS + sp) * 64 + lane
                        for sp in range_constexpr(SPLITS)
                    ]
                    mls = [
                        Vec(
                            buffer_ops.buffer_load(
                                pr,
                                rf * PART_F + NDT * 4,
                                vec_width=2,
                                dtype=fx.Int32,
                                cache_modifier=PART_SCOPE,
                            )
                        )
                        for rf in rows_f
                    ]
                    bigs = [as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, SPLITS):
                        top = maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(SPLITS):
                        l_all = l_all + as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(
                        v_scale / l_all, fx.Float32(0.0)
                    )
                    acc_all = [zero4 for _ in range_constexpr(NDT)]
                    for sp in range_constexpr(SPLITS):
                        wv = Vec.filled(4, ws[sp], fx.Float32)
                        for dt in range_constexpr(NDT):
                            part = Vec(
                                buffer_ops.buffer_load(
                                    pr,
                                    rows_f[sp] * PART_F + dt * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            ).bitcast(fx.Float32)
                            acc_all[dt] = acc_all[dt] + part * wv
                    store_out(acc_all, inv_all)
                    if lane == 0:
                        global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)
            else:
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                store_out(acc, inv)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        i32_grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_lut,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            i64_part_bytes,
            i32_page_b,
            f32_c,
        ).launch(
            grid=(i32_grid_x, i32_hkv, SPLITS),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
