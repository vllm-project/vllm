# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 sparse prefill attention for MiniMax-M3 (FlyDSL, gfx950).

A workgroup takes 4 query tokens x the 16 query heads of one KV head and walks
the union of the pages they selected. K and V are the exact e2m1 x e4m3
products in fp16; Q and P are fp16; scores, softmax and accumulation are fp32;
the output is bf16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
    exp2,
    global_i32_ptr,
    maxf_nn,
    mfma_16x16x32,
    pk_bf16,
    pk_f16,
    xlane4,
)

BLK = 128  # tokens per page
D = 128
HG = 16  # query heads per KV head (the MFMA N dimension)
NW = 4
KPW = BLK // NW  # keys per wave
TOK_PER_WG = 4
TOPK = 16
MAXU = TOK_PER_WG * TOPK  # union capacity: one lane per (token, slot)
DATA_B = BLK * D // 2  # e2m1 bytes per head per side
SCALE_B = BLK * D // 16  # e4m3 scale bytes per head per side
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per page
Q_ROW = D * 2 // 16 + 1  # 16 B LDS tiles per Q^T row, padded by one


def _bf16_to_f16(v):
    return Vec(v).bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)


def kv_lane_loader(arg_cache, page_bytes, kvh, hkv, w, n, g):
    """The loader of this lane's NVFP4 K/V dwords of a page.

    S^T tile st row r is key 32 w + 8 (r // 4) + 4 st + r % 4, so lane
    (n, g)'s P^T / V^T key slots are keys 32 w + 8 g + j.
    """
    side_b = hkv * (DATA_B + SCALE_B)
    k_data = kvh * DATA_B
    k_scl = hkv * DATA_B + kvh * SCALE_B
    v_data = side_b + kvh * DATA_B
    v_scl = side_b + hkv * DATA_B + kvh * SCALE_B
    k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
    k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
    k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
    v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
    v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

    def load(page):
        """This lane's RAW NVFP4 dwords of ``page``."""
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_cache + fx.Int64(page) * fx.Int64(page_bytes),
            num_records_bytes=page_bytes,
        )
        raw = []
        for st in range_constexpr(2):
            kd = fx.Vector(
                buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
            )
            raw += [kd[i] for i in range_constexpr(4)]
        raw.append(
            fx.Int32(
                buffer_ops.buffer_load(pr, k_scale_load, vec_width=1, dtype=fx.Int32)
            )
        )
        for h in range_constexpr(2):
            vd = fx.Vector(
                buffer_ops.buffer_load(pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32)
            )
            raw += [vd[i] for i in range_constexpr(4)]
        vs = fx.Vector(
            buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
        )
        raw += [vs[i] for i in range_constexpr(2)]
        return raw

    return load


def compile_nvfp4_prefill_fp16(rounds: int = 2):
    """Build the prefill kernel.

    Args:
        rounds: rounds of the cross-wave combine. 2 halves its LDS so two
            workgroups share a CU; 1 keeps Q^T in registers and combines all
            tokens at once (grids of at most one workgroup per CU).

    """
    assert rounds in (1, 2)
    tpr = TOK_PER_WG // rounds  # tokens per combine round
    wpe = rounds  # waves per SIMD
    name = f"m3_nvfp4_sparse_prefill_fp16matched_tq{TOK_PER_WG}_ksplit"
    if rounds == 1:
        name += "_r1"

    @fx.struct
    class Shared:
        # Q^T during the page loop, then the partial O per (wave, token, d-tile,
        # lane) in 16 B tiles; (max, sum) per (wave, token, lane).
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 * Hkv, 128, 72] uint8
        arg_topk: fx.Int64,  # [Hkv, T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_pos: fx.Int64,  # [T] int32
        arg_out: fx.Int64,  # [T, H, 128] bf16
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,  # topk stride over KV heads
        i32_bt_s: fx.Int32,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        req_p = global_i32_ptr(arg_req)
        pos_p = global_i32_ptr(arg_pos)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        topk_base = kvh * i32_topk_hs

        lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * tpr * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        if const_expr(rounds == 2):
            # Q^T in LDS; thread (token, head, quarter) stages 64 B.
            tq, hq, qq = tx // 64, (tx // 4) % HG, tx % 4
            tok_q = bidx * TOK_PER_WG + tq
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * HG + hq) * (D // 2) + qq * 16
            for i in range_constexpr(4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                q4 = buffer_ops.buffer_load(
                    qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                )
                r.store(_bf16_to_f16(q4))
                q_tile = (tq * HG + hq) * Q_ROW + qq * 4 + i
                fx.copy(lds_atom, r, fx.slice(lds, (None, q_tile)))

        def q_frags(t):
            """Token t's Q^T, one B operand per k-step."""
            if const_expr(rounds == 1):
                return q_regs[t]
            out = []
            for ks in range_constexpr(NKS):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                q_tile = (t * HG + n) * Q_ROW + g * 4 + ks
                fx.copy(lds_atom, fx.slice(lds, (None, q_tile)), r)
                out.append(Vec(r.load()))
            return out

        # Per token: the page of its own block and its causal limit.
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG):
            tok = bidx * TOK_PER_WG + t
            ok = tok < i32_T
            tc = ok.select(tok, fx.Int32(0))
            pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(pos_p[tc])))
            req = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(req_p[tc])))
            diag.append(
                fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, fx.Int32(bt[req * i32_bt_s + pos // BLK])
                    )
                )
            )
            lim.append(pos % BLK - w * KPW)
            if const_expr(rounds == 1):
                q_row = (tc * i32_H + kvh * HG + n) * (D // 2) + g * 16
                q_regs.append(
                    [
                        _bf16_to_f16(
                            buffer_ops.buffer_load(
                                qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32
                            )
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )

        load_page = kv_lane_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, g)

        # Lane (token lane // 16, slot lane % 16): the page it selected, -1 none.
        lane_tok = bidx * TOK_PER_WG + lane // TOPK
        lane_tok_ok = lane_tok < i32_T
        lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
        blk = fx.Int32(topk[topk_base + lane_tok_clamped * TOPK + lane % TOPK])
        lane_blk_ok = lane_tok_ok & (blk >= 0)
        lane_page = fx.Int32(
            bt[
                fx.Int32(req_p[lane_tok_clamped]) * i32_bt_s
                + lane_blk_ok.select(blk, fx.Int32(0))
            ]
        )
        lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))
        # The union's first page is the lowest valid lane's: load it before the
        # dedup runs.
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_page >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest lane left in ``mask``, the first page when empty."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        # Union: the first lane of each distinct page.
        n_dup = fx.Int32(0)
        for j in range_constexpr(MAXU - 1):
            pj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, j))
            n_dup = n_dup + ((lane_page == pj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = fx.Int64(fx.rocdl.ballot(T.i64, (lane_page >= 0) & (n_dup == 0)))
        n_u = fx.Int32(fx.math.ctpop(umask).to(fx.Int32))

        def dequant(raw):
            """K frags kf[st][ks] and V^T frags vf[dt] of a page in fp16."""
            kf = []
            for st in range_constexpr(2):
                ksc = cvt_fp8x2_f16(raw[8], st == 1)
                row = []
                for ks in range_constexpr(NKS):
                    sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
                    row.append(
                        Vec.from_elements(
                            [
                                (cvt_fp4x2_f16(raw[st * 4 + ks], i) * sv).bitcast(
                                    fx.Int32
                                )[0]
                                for i in range_constexpr(4)
                            ],
                            fx.Int32,
                        )
                    )
                kf.append(row)
            vs = []
            for h in range_constexpr(2):
                vs += [
                    cvt_fp8x2_f16(raw[17 + h], False),
                    cvt_fp8x2_f16(raw[17 + h], True),
                ]
            # A V byte holds two consecutive keys: already the operand's fp16 pair.
            vf = [
                Vec.from_elements(
                    [
                        (cvt_fp4x2_f16(raw[9 + dt], i) * vs[i]).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                for dt in range_constexpr(NDT)
            ]
            return kf, vf

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        if const_expr(rounds == 2):
            fx.gpu.barrier()  # Q^T staged
        # Iteration u dequantizes page u and loads page u + 1.
        init = [umask]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG * NDT)]
        init += raw0
        n_st = 1 + 2 * TOK_PER_WG + TOK_PER_WG * NDT
        results = init
        for _, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_u),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m = [fx.Float32(st_[1 + t]) for t in range_constexpr(TOK_PER_WG)]
            l_run = [
                fx.Float32(st_[1 + TOK_PER_WG + t]) for t in range_constexpr(TOK_PER_WG)
            ]
            o = [
                [
                    Vec(st_[1 + 2 * TOK_PER_WG + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            page = page_at(rem)
            rem = rem & (rem - fx.Int64(1))
            raw_next = load_page(page_at(rem))
            kf, vf = dequant(raw)
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            for t in range_constexpr(TOK_PER_WG):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if (hits.shrui(fx.Int64(TOPK * t)) & fx.Int64(0xFFFF)) != fx.Int64(0):
                    qf = q_frags(t)
                    a = []
                    for st in range_constexpr(2):
                        acc = zero4
                        for ks in range_constexpr(NKS):
                            acc = mfma_16x16x32(kf[st][ks], qf[ks], acc, True)
                        a += [acc[e] for e in range_constexpr(4)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
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
                    l_t = l_t * alpha + ps
                    pf = Vec.from_elements(
                        [pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                        fx.Int32,
                    )
                    # alpha is exactly 1 unless some row's max grew.
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [
                        mfma_16x16x32(vf[dt], pf, o_t[dt], True)
                        for dt in range_constexpr(NDT)
                    ]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = (
                [rem] + m + l_run + [x for token_out in o for x in token_out] + raw_next
            )
            results = yield [fx.as_ir_value(x) for x in state]

        # Combine the waves' partials, tpr tokens per round; wave w finalizes
        # token w in round w // tpr.
        ml = smem.ml.ptr
        if const_expr(rounds == 2):
            fx.gpu.barrier()
        for t in range_constexpr(TOK_PER_WG):
            ml[((w * TOK_PER_WG + t) * 2) * 64 + lane] = fx.Float32(results[1 + t])
            ml[((w * TOK_PER_WG + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[1 + TOK_PER_WG + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = 1 + 2 * TOK_PER_WG + (tpr * rnd + tt) * NDT + dt
                    r.store(Vec(results[ri]).bitcast(fx.Int32))
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(lds, (None, ((w * tpr + tt) * NDT + dt) * 64 + lane)),
                    )
            fx.gpu.barrier()
            if w // tpr == rnd:
                t = w
                ms = [
                    fx.Float32(ml[((ww * TOK_PER_WG + t) * 2) * 64 + lane])
                    for ww in range_constexpr(NW)
                ]
                big = ms[0]
                for ww in range_constexpr(1, NW):
                    big = maxf_nn(big, ms[ww])
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [
                    fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)
                ]
                l_tot = fx.Float32(0.0)
                for ww in range_constexpr(NW):
                    l_w = ml[((ww * TOK_PER_WG + t) * 2 + 1) * 64 + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = xlane4(l_tot, lambda a, b: a + b)
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                acc = [zero4 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        o_tile = ((ww * tpr + t - tpr * rnd) * NDT + dt) * 64 + lane
                        fx.copy(lds_atom, fx.slice(lds, (None, o_tile)), r)
                        acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
                # Lane (n, g) holds dims [32 g, 32 g + 32) of head n in order.
                tok = bidx * TOK_PER_WG + t
                words = [
                    [
                        pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ]
                    for e in range_constexpr(4)
                ]
                outr = buffer_ops.create_buffer_resource_from_addr(
                    arg_out,
                    num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2),
                )
                o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + g * 16
                for e in range_constexpr(4):
                    val = Vec.from_elements(words[e], fx.Int32)
                    buffer_ops.buffer_store(val, outr, o_row + e * 4, mask=tok < i32_T)
            fx.gpu.barrier()

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_pos: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_cache,
            arg_topk,
            arg_bt,
            arg_req,
            arg_pos,
            arg_out,
            arg_ks,
            arg_vs,
            i32_T,
            i32_H,
            i32_hkv,
            i32_topk_hs,
            i32_bt_s,
            i32_page_b,
            f32_c,
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=((i32_T + TOK_PER_WG - 1) // TOK_PER_WG, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
