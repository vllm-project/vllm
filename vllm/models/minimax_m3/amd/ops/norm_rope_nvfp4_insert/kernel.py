# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma norm + RoPE of q / k / index-q / index-k and the NVFP4 or MXFP4 K/V +
index-K page inserts in one launch (FlyDSL, gfx950), bit for bit the ROCm
``fused_minimax_m3_qknorm_rope_kv_insert`` followed by the page insert.

Waves of 2 * tp tokens: [q heads | k heads | index-k | index-q heads | v heads x tp];
a norm wave holds 4 dims of a token per lane (the HIP kernel's lane map).

MXFP4 pages keep NVFP4's data layout with E8M0 scales per 32 dims and no global
scale (68 B per token and head): K and index-K scales ``[token][4]``, V scales
``[group][token]``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    PRELOAD_HINTS,
    as_f32,
    bits,
    permlane_swap,
    pk_bf16,
)

NW = 4
D = 128
BLK = 128  # tokens per NVFP4 page
RCP6 = 0x3E2AAAAB  # RN(1 / 6)
F448 = 0x43E00000
F448N = 0xC3E00000 - (1 << 32)


def _bf16x4(v):
    """4 fp32 of a dword pair of bf16 (low half first)."""
    w0, w1 = fx.Int32(v[0]), fx.Int32(v[1])
    return [
        (w0 << 16).bitcast(fx.Float32),
        (w0 & -65536).bitcast(fx.Float32),
        (w1 << 16).bitcast(fx.Float32),
        (w1 & -65536).bitcast(fx.Float32),
    ]


def _norm_rope_out4(xws_t, css_t, w, eps, hi16, first, in_rope, half4):
    """Norm + RoPE of one token's 4 dims in this lane -> 4 fp32."""
    cos, sin = css_t
    e = _bf16x4(xws_t)
    ss = e[0] * e[0] + e[1] * e[1]
    ss = ss + e[2] * e[2]
    ss = ss + e[3] * e[3]
    for m in (16, 8, 4, 2, 1):
        ss = _xsum(ss, m)
    r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
    nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
    out = []
    for i in range_constexpr(4):
        pr = _xor_lane(bits(nrm[i]), half4, hi16).bitcast(fx.Float32)
        ps = pr * sin[i]
        roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
        out.append(in_rope.select(roped, nrm[i]))
    return out


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    """v_mov_b32_dpp; an undef ``old`` lets the move fold into its consumer."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(
        fx.rocdl.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full)
    )


def _swap16(v):
    """permlane16_swap(v, v): (own, lane ^ 16) in rows 0 / 2, swapped in 1 / 3."""
    return permlane_swap(fx.rocdl.permlane16_swap, v, v)


def _xor_lane(v, m, hi16):
    """Lane l ^ m of the 32-lane half; ``hi16`` is this lane's bit 4."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    if m == 4:  # row_shr:4 into banks 1, 3, then row_shl:4 into banks 0, 2
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)  # row_ror:8
    assert m == 16
    a, b = _swap16(v)
    return hi16.select(a, b)


def _xsum(x, m):
    """Adds lane l ^ m's value to ``x``."""
    if m == 16:
        a, b = _swap16(bits(x))
        return a.bitcast(fx.Float32) + b.bitcast(fx.Float32)
    return x + _xor_lane(bits(x), m, None).bitcast(fx.Float32)


def _xmax(x, m):
    return _maxnum(x, _xor_lane(bits(x), m, None).bitcast(fx.Float32))


def _rsq(x):
    """v_rsq_f32; ocml's denormal scaling never applies as x >= eps >= 2^-126."""
    return fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(x)))


def _e4m3_byte(x):
    """RNE e4m3 byte of ``x``."""
    pk = fx.rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        fx.as_ir_value(Vec.filled(2, 0, fx.Int16)),
        fx.as_ir_value(x),
        fx.as_ir_value(x),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return Vec(pk).bitcast(fx.Int32)[0] & 0xFF


def _e4m3_value(byte):
    pk = fx.rocdl.cvt_scalef32_pk_f32_fp8(
        T.vec(2, T.f32),
        fx.as_ir_value(fx.Int32(byte)),
        fx.as_ir_value(fx.Float32(1.0)),
        False,
    )
    return fx.Float32(Vec(pk)[0])


def _iq_fp8(a, b, old, hi):
    """Two fp32 -> e4m3 bytes in the low (``hi``: high) half of ``old``."""
    return fx.Int32(
        fx.rocdl.cvt_pk_fp8_f32(
            T.i32,
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Int32(old)),
            hi,
        )
    )


def _e2m1x2(a, b, old, sel):
    """E2M1 codes of ``a``, ``b`` into byte ``sel`` of ``old``, ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _quant4(xs, gs, inv_gs):
    """E2M1 codes of this lane's 4 values (16 bits) and the e4m3 scale byte of
    their 16-value group (lanes l ^ 0..3)."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    sf = _minnum((m * as_f32(RCP6)) * inv_gs, as_f32(F448))
    byte = _e4m3_byte(sf)
    sv = _e4m3_value(byte)
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte


def _quant4_mx(xs):
    """E2M1 codes of this lane's 4 values (16 bits) and the E8M0 scale byte of
    their 32-value group (lanes l ^ 0..7): 2^ceil(log2(amax / 6)), 2^0 for an
    all-zero group."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(_xmax(m, 1), 2), 4)
    qb = bits(m / fx.Float32(6.0))
    inexact = (qb & 0x7FFFFF) != 0
    e = ((qb >> 23) & 0xFF) - 127 + inexact.select(fx.Int32(1), fx.Int32(0))
    # amax / 6 < 2^126, so only the low end clamps (denormal quotients).
    e = (m == 0.0).select(fx.Int32(0), (e < -126).select(fx.Int32(-126), e))
    inv = as_f32((fx.Int32(127) - e) << 23)
    y = [x * inv for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), e + 127


def _v_row(state, row, r, n_run, p0, quant):
    """Adds row ``r`` of a V run at octet positions p0.. to ``state``: new
    nibbles and scale bytes, with their masks."""
    words, written = state[0:4], state[4]
    sc, sc_mask = state[5:7], state[7:9]
    nib, sf = quant(_bf16x4(row))
    in_run = n_run > r
    p = p0 + r
    sh = p * 4
    words = [
        wd | in_run.select(((nib >> (4 * i)) & 15) << sh, fx.Int32(0))
        for i, wd in enumerate(words)
    ]
    written = written | in_run.select(fx.Int32(15) << sh, fx.Int32(0))
    bsh = (p % 4) * 8
    for k in range_constexpr(2):
        here = in_run & ((p // 4) == k)
        sc[k] = sc[k] | here.select(sf << bsh, fx.Int32(0))
        sc_mask[k] = sc_mask[k] | here.select(fx.Int32(255) << bsh, fx.Int32(0))
    return [*words, written, *sc, *sc_mask]


def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def _k_scale_off(tok, grp):
    """Byte of (token, group) in a head's K scale region."""
    r = tok % 32
    n = (r // 8) * 4 + r % 4
    return ((((tok // 32) * 4 + grp // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + grp % 2


def compile_norm_rope_insert(
    nq: int, nkv: int, niq: int, rotary_dim: int, tp: int, mx: bool = False
):
    """Build the kernel; ``tp`` token pairs per norm / RoPE wave, one per V wave;
    ``mx`` writes MXFP4 pages instead of NVFP4."""
    assert rotary_dim in (8, 16, 32, 64, 128)
    NR = nq + nkv + 1 + niq  # norm + RoPE waves
    S = NR + nkv * tp  # waves per 2 * tp tokens
    ROW = (nq + 2 * nkv + niq + 1) * D
    HALF = rotary_dim // 2
    NG = 4 if mx else 8  # scale groups per token and head
    GL = D // NG // 4  # lanes per scale group
    SIDE = nkv * BLK * (64 + NG)  # bytes of one side of a main page
    TPW = 2 * tp
    fmt = "mxfp4" if mx else "nvfp4"
    name = f"m3_norm_rope_{fmt}_insert_q{nq}_kv{nkv}_iq{niq}_r{rotary_dim}_t{tp}"

    def quant(xs, gs, inv_gs):
        return _quant4_mx(xs) if mx else _quant4(xs, gs, inv_gs)

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_qkv: fx.Int64,  # [n, ROW] bf16
        arg_pos: fx.Int64,  # positions [n] i64
        arg_slot: fx.Int64,  # main slots [n] i64, -1 skips
        arg_islot: fx.Int64,  # index slots [n] i64, -1 skips
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, rotary_dim] bf16
        arg_qw: fx.Int64,  # norm weights [128] bf16: q, k, index-q, index-k
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_q: fx.Int64,  # q_out [n, nq * 128] bf16
        arg_iq: fx.Int64,  # index_q_out [n, niq * 128] e4m3, row stride i32_iq_st
        arg_kv: fx.Int64,  # [blocks, 2 * nkv, 128, 72] u8
        arg_idx: fx.Int64,  # [blocks, 128, 72] u8
        arg_ks: fx.Int64,  # fp32 global scales
        arg_vs: fx.Int64,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_iq_st: fx.Int32,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        s = wid % S
        hl = lane % 32
        hi16 = (hl & 16) != 0
        dim0 = hl * 4
        toks = [(wid // S) * TPW + t * 2 + lane // 32 for t in range_constexpr(tp)]
        lives = [tk < i32_n for tk in toks]

        qkv = buffer_ops.create_buffer_resource_from_addr(
            arg_qkv, num_records_bytes=fx.Int64(i32_n) * (ROW * 2)
        )
        sl_r = buffer_ops.create_buffer_resource_from_addr(
            arg_slot, num_records_bytes=fx.Int64(i32_n) * 8
        )

        if s < NR:
            is_q = s < nq
            is_k = (s >= nq) & (s < nq + nkv)
            is_ik = s == nq + nkv
            is_kq = is_k | is_ik
            # This wave's head column in the qkv row.
            col = (s <= nq + nkv).select(
                is_ik.select(fx.Int32(nq + 2 * nkv + niq), s),
                s - (nq + nkv + 1) + (nq + 2 * nkv),
            )
            w_addr = is_q.select(
                arg_qw, is_k.select(arg_kw, is_ik.select(arg_ikw, arg_iqw))
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                w_addr, num_records_bytes=D * 2
            )
            pos_r = buffer_ops.create_buffer_resource_from_addr(
                arg_pos, num_records_bytes=fx.Int64(i32_n) * 8
            )
            isl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_islot, num_records_bytes=fx.Int64(i32_n) * 8
            )
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (rotary_dim * 2)
            )
            xws, poss, slots = [], [], []
            for t in range_constexpr(tp):
                xws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            qkv,
                            toks[t] * (ROW // 2) + col * (D // 2) + hl * 2,
                            vec_width=2,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                poss.append(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            pos_r,
                            toks[t] * 2,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=lives[t],
                        )
                    )
                )
                kslot = fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_k,
                    )
                )
                islot = fx.Int32(
                    buffer_ops.buffer_load(
                        isl_r,
                        toks[t] * 2,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lives[t] & is_ik,
                    )
                )
                slots.append(is_k.select(kslot, islot))
            w = _bf16x4(
                Vec(buffer_ops.buffer_load(wr, hl * 2, vec_width=2, dtype=fx.Int32))
            )
            first = dim0 < HALF
            in_rope = dim0 < rotary_dim
            ib = first.select(dim0, dim0 - HALF)
            css = []
            for t in range_constexpr(tp):
                cs_off = (poss[t] * rotary_dim + ib) // 2
                css.append(
                    (
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                        _bf16x4(
                            Vec(
                                buffer_ops.buffer_load(
                                    cs_r,
                                    cs_off + HALF // 2,
                                    vec_width=2,
                                    dtype=fx.Int32,
                                    mask=lives[t] & in_rope,
                                )
                            )
                        ),
                    )
                )
            eps = i32_eps.bitcast(fx.Float32)
            qvs = []
            for t in range_constexpr(tp):
                out = _norm_rope_out4(
                    xws[t], css[t], w, eps, hi16, first, in_rope, HALF // 4
                )
                lo, hi = pk_bf16(out[0], out[1]), pk_bf16(out[2], out[3])
                qvs.append(Vec.from_elements([lo, hi], fx.Int32))
            if is_q:
                q_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
                )
                for t in range_constexpr(tp):
                    buffer_ops.buffer_store(
                        qvs[t],
                        q_r,
                        toks[t] * (nq * D // 2) + s * (D // 2) + hl * 2,
                        mask=lives[t],
                    )
            if s > nq + nkv:
                iq_r = buffer_ops.create_buffer_resource_from_addr(
                    arg_iq, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_iq_st)
                )
                iq_col = (s - (nq + nkv + 1)) * D
                for t in range_constexpr(tp):
                    f = [
                        _minnum(_maxnum(x, as_f32(F448N)), as_f32(F448))
                        for x in _bf16x4(qvs[t])
                    ]
                    word = _iq_fp8(
                        f[2], f[3], _iq_fp8(f[0], f[1], fx.Int32(0), False), True
                    )
                    buffer_ops.buffer_store(
                        word,
                        iq_r,
                        toks[t] * i32_iq_st + iq_col + dim0,
                        mask=lives[t],
                        offset_is_bytes=True,
                    )
            if is_kq:
                gs = is_k.select(
                    fx.Float32(_gptr(arg_ks, T.f32, 4)[0]), fx.Float32(1.0)
                )
                inv_gs = fx.Float32(1.0) / gs
                quants = [
                    quant(_bf16x4(qvs[t]), gs, inv_gs) for t in range_constexpr(tp)
                ]
                h = s - nq
                grp = hl // GL
                for t in range_constexpr(tp):
                    slot = slots[t]
                    tp_ = slot & (BLK - 1)
                    page = is_k.select(
                        arg_kv + fx.Int64(slot >> 7) * i64_kv_st,
                        arg_idx + fx.Int64(slot >> 7) * i64_idx_st,
                    )
                    d_off = is_k.select(h * (BLK * 64), fx.Int32(0)) + tp_ * 64 + hl * 2
                    k_sf = tp_ * NG + grp if mx else _k_scale_off(tp_, grp)
                    s_off = is_k.select(
                        nkv * BLK * 64 + h * (BLK * NG) + k_sf,
                        BLK * 64 + tp_ * NG + grp,
                    )
                    d_ptr = _gptr(page + fx.Int64(d_off), T.i16, 2)
                    s_ptr = _gptr(page + fx.Int64(s_off), T.i8, 1)
                    packed = quants[t][0].to(fx.Int16)
                    sf8 = quants[t][1].to(fx.Int8)
                    if lives[t] & (slot >= 0):
                        d_ptr[0] = packed
                        if hl % GL == 0:
                            s_ptr[0] = sf8
        else:
            vh = (s - NR) // tp
            tok = (wid // S) * TPW + ((s - NR) % tp) * 2 + lane // 32
            live = tok < i32_n
            vcol = (nq + nkv + vh) * (D // 2)  # V head column (dwords)
            vs = fx.Float32(_gptr(arg_vs, T.f32, 4)[0])
            near_t = tok - 7 + hl
            has_near = (hl < 16) & (near_t >= 0) & (near_t < i32_n)
            near = has_near.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, near_t * 2, vec_width=1, dtype=fx.Int32, mask=has_near
                    )
                ),
                fx.Int32(-1),
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(
                    sl_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live
                )
            )
            has_prev = live & (tok > 0)
            prev = has_prev.select(
                fx.Int32(
                    buffer_ops.buffer_load(
                        sl_r, tok * 2 - 2, vec_width=1, dtype=fx.Int32, mask=has_prev
                    )
                ),
                fx.Int32(-1),
            )
            # The first token of a run in one 8-token octet writes the run and
            # keeps the octet's other nibbles and scales. Rows it can cover load
            # before its slot is known.
            rows = [
                Vec(
                    buffer_ops.buffer_load(
                        qkv,
                        (tok + r) * (ROW // 2) + vcol + hl * 2,
                        vec_width=2,
                        dtype=fx.Int32,
                        mask=tok + r < i32_n,
                    )
                )
                for r in range_constexpr(8)
            ]
            base = slot - slot % 8
            in_oct = (near >= base) & (near < base + 8)
            votes = fx.Int64(fx.rocdl.ballot(T.i64, in_oct))
            mine = (votes >> fx.Int64((lane // 32) * 32)).to(fx.Int32) & 0xFFFF
            n_run = fx.Int32(fx.math.ctpop(mine))
            run_first = (prev < 0) | ((prev >> 3) != (slot >> 3))
            tp_ = slot & (BLK - 1)
            v_side = arg_kv + fx.Int64(slot >> 7) * i64_kv_st + SIDE
            v_data = v_side + fx.Int64(vh * (BLK * 64) + (tp_ // 8) * (D * 4))
            v_sf = v_side + fx.Int64(nkv * BLK * 64 + vh * (BLK * NG) + tp_ - tp_ % 8)
            dp = _gptr(v_data + fx.Int64(dim0 * 4), T.i32, 4)
            sp = _gptr(v_sf + fx.Int64((hl // GL) * BLK), T.i32, 4)
            go = live & (slot >= 0) & run_first
            inv_vs = fx.Float32(1.0) / vs

            def v_quant(xs):
                return quant(xs, vs, inv_vs)

            # Runs of <= 4 tokens (decode) take the short tile.
            for run_rows, take in (
                (rows[:4], go & (n_run <= 4)),
                (rows, go & (n_run > 4)),
            ):
                if take:
                    old = [fx.Int32(dp[i]) for i in range_constexpr(4)]
                    old_sc = [fx.Int32(sp[i]) for i in range_constexpr(2)]
                    st = [fx.Int32(0)] * 9
                    for r in range_constexpr(len(run_rows)):
                        st = _v_row(st, run_rows[r], r, n_run, tp_ % 8, v_quant)
                    keep = ~st[4]
                    for i in range_constexpr(4):
                        dp[i] = st[i] | (old[i] & keep)
                    sc0 = st[5] | (old_sc[0] & ~st[7])
                    sc1 = st[6] | (old_sc[1] & ~st[8])
                    if hl % GL == 0:
                        sp[0] = sc0
                        sp[1] = sc1

    @flyc.jit
    def launch(
        arg_qkv: fx.Int64,
        arg_q: fx.Int64,
        arg_iq: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_iqw: fx.Int64,
        arg_ikw: fx.Int64,
        arg_cs: fx.Int64,
        arg_pos: fx.Int64,
        arg_slot: fx.Int64,
        arg_islot: fx.Int64,
        arg_kv: fx.Int64,
        arg_idx: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_n: fx.Int32,
        i32_iq_st: fx.Int32,
        i64_kv_st: fx.Int64,
        i64_idx_st: fx.Int64,
        i32_cs_rows: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        waves = (i32_n + (TPW - 1)) // TPW * S
        kernel(
            i32_n,
            i32_cs_rows,
            arg_qkv,
            arg_pos,
            arg_slot,
            arg_islot,
            arg_cs,
            arg_qw,
            arg_kw,
            arg_iqw,
            arg_ikw,
            arg_q,
            arg_iq,
            arg_kv,
            arg_idx,
            arg_ks,
            arg_vs,
            i64_kv_st,
            i64_idx_st,
            i32_iq_st,
            i32_eps,
        ).launch(
            grid=(fx.Int64((waves + (NW - 1)) // NW), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
