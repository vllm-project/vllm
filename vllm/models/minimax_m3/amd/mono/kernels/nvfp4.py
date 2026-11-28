# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono kernels' side of the NVFP4 K/V and index-K pages.

The pages are the ones `ops/norm_rope_nvfp4_insert` writes and the NVFP4 sparse
decode / index scorers read (e2m1 data, one e4m3 scale per 16 elements; K and V
also one fp32 global scale per layer). One kv head a rank, one page a block::

    K/V    [K data 128 x 64 B][K scales 1024 B][V data 8192 B][V scales 1024 B]
    index  [data 128 x 64 B][scales 128 x 8 B]

K data and index data are token-major (low nibble the even dim), K scales in the
sparse decode's lane order (`k_scale_at`), V data a dword per (token octet, dim)
(byte b = tokens 2 b, 2 b + 1 of the octet, nibble 0 the even), V scales
`group * 128 + token`. The writers quantize as the insert kernel's `_quant4`
does, bit for bit: these pages are shared with the stock decode and prefill.
"""

from __future__ import annotations

import struct

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from vllm.models.minimax_m3.amd.mono import config as mono_config
from vllm.models.minimax_m3.amd.mono.config import HEAD_DIM, SPARSE_BLOCK
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    butterfly,
    hw_exp2,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    LOG2E,
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
    mfma_16x16x32,
    pk_f16,
)

GROUP = 16  # elements an e4m3 scale covers
DATA_BYTES = HEAD_DIM // 2  # a token's e2m1 bytes
SCALE_BYTES = HEAD_DIM // GROUP  # a token's scale bytes
SIDE_DATA = SPARSE_BLOCK * DATA_BYTES  # 8192
SIDE_BYTES = SIDE_DATA + SPARSE_BLOCK * SCALE_BYTES  # 9216: one side, or an index page
K_SCALE_AT = SIDE_DATA
V_DATA_AT = SIDE_BYTES
V_SCALE_AT = SIDE_BYTES + SIDE_DATA
KV_PAGE_BYTES = 2 * SIDE_BYTES  # what a split task's buffer resource spans
INDEX_SCALE_AT = SIDE_DATA

_RCP6 = struct.unpack("f", struct.pack("I", 0x3E2AAAAB))[0]  # RN(1 / 6)
E4M3_MAX = 448.0


def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def kv_block(cache, slot_or_block, is_slot=True):
    """Int64 address of the K/V page holding `slot` (or of block `block`)."""
    b = slot_or_block // SPARSE_BLOCK if is_slot else slot_or_block
    return fx.Int64(cache) + fx.Int64(b) * fx.Int64(mono_config.NVFP4_KV_BLOCK_BYTES)


def k_scale_at(t, g):
    """Byte of K scale (token t of the page, group g) in the K scale region."""
    r = t % 32
    n = 4 * (r // 8) + r % 4
    return ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2


@traced
def quant16(vals, gs):
    """(e4m3 scale byte, e2m1 codes in the low 16 bits) of this lane's 4 elements
    of a 16-element group held by 4 consecutive lanes (`vals`: bf16-rounded f32),
    as the insert kernel's `_quant4`: the scale is e4m3(min(amax * RN(1/6) / gs,
    448)) and an element x / (gs * that scale as decoded), RNE e2m1. `gs`: the
    layer's global scale (1.0 for the index cache)."""
    am = fx.math.absf(vals[0])
    for v in vals[1:]:
        am = fx.max(am, fx.math.absf(v))
    am = butterfly(am, (1, 2), fx.max)
    sf = fx.min((am * fx.Float32(_RCP6)) * (fx.Float32(1.0) / gs), fx.Float32(448.0))
    code = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, sf, sf, fx.Int32(0), False)) & 0xFF
    sv = fx.Vector(
        rocdl.cvt_scalef32_pk_f32_fp8(
            T.vec(2, T.f32),
            as_ir_value(code),
            as_ir_value(fx.Float32(1.0)),
            False,
        )
    )[0]
    quant = (sv != 0.0).select(fx.Float32(1.0) / (gs * sv), fx.Float32(0.0))
    packed = fx.Int32(0)
    for pair in range_constexpr(2):
        packed = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                as_ir_value(packed),
                as_ir_value(fx.Float32(vals[2 * pair] * quant)),
                as_ir_value(fx.Float32(vals[2 * pair + 1] * quant)),
                as_ir_value(fx.Float32(1.0)),
                pair,
            )
        )
    return code, packed


# The writers' stores are buffer stores off the cache's base, their byte offsets
# 32-bit like the fp8 path's page offsets (a layer's cache spans < 4 GiB), with
# the caller's cache policy (K1 in the layer kernel: device-coherent, read by K4)
def _store16(r, off, packed, cm):
    bo.buffer_store(
        (packed & 0xFFFF).to(fx.Int16), r, off, cache_modifier=cm, offset_is_bytes=True
    )


def _store8(r, off, byte, cm):
    bo.buffer_store(
        (byte & 0xFF).to(fx.Int8), r, off, cache_modifier=cm, offset_is_bytes=True
    )


@traced
def index_row_store(cache, slot, vals, l32, cm=0):
    """Token `slot`'s index-K row (a half wave, 4 elements a lane, bf16-rounded
    f32) quantized into its index page."""
    code, packed = quant16(vals, fx.Float32(1.0))
    page = (slot // SPARSE_BLOCK) * mono_config.NVFP4_INDEX_BLOCK_BYTES
    t = slot % SPARSE_BLOCK
    r = rsrc(cache)
    _store16(r, page + t * DATA_BYTES + l32 * 2, packed, cm)
    if l32 % 4 == 0:
        _store8(r, page + INDEX_SCALE_AT + t * SCALE_BYTES + l32 // 4, code, cm)


@traced
def k_store(cache, slot, vals, l32, gs, cm=0):
    """`index_row_store` for K into the K/V page, under the layer's k scale `gs`."""
    code, packed = quant16(vals, gs)
    page = (slot // SPARSE_BLOCK) * mono_config.NVFP4_KV_BLOCK_BYTES
    t = slot % SPARSE_BLOCK
    r = rsrc(cache)
    _store16(r, page + t * DATA_BYTES + l32 * 2, packed, cm)
    if l32 % 4 == 0:
        _store8(r, page + K_SCALE_AT + k_scale_at(t, l32 // 4), code, cm)


def _atomic(op, addr, val):
    op(_gptr(addr, T.i32, 4), fx.Int32(val), syncscope="agent")


@traced
def v_store(cache, slot, vals, l32, gs, cm=0):
    """V of token `slot`, under the layer's v scale `gs`: the scale bytes, and each
    nibble merged into its (octet, dim) dword by an atomic clear then set (the
    step's other tokens of the octet write theirs concurrently; agent scope, so
    device-coherent like the other stores)."""
    code, packed = quant16(vals, gs)
    page = kv_block(cache, slot)
    t = slot % SPARSE_BLOCK
    if l32 % 4 == 0:
        _store8(
            rsrc(cache),
            (slot // SPARSE_BLOCK) * mono_config.NVFP4_KV_BLOCK_BYTES
            + V_SCALE_AT
            + (l32 // 4) * SPARSE_BLOCK
            + t,
            code,
            cm,
        )
    shift = ((t % 8) // 2) * 8 + (t % 2) * 4
    for i in range_constexpr(4):
        d = l32 * 4 + i
        dw = page + fx.Int64(V_DATA_AT + ((t // 8) * HEAD_DIM + d) * 4)
        _atomic(fx.llvm.atomic_and, dw, fx.Int32(-1) ^ (fx.Int32(0xF) << shift))
        _atomic(fx.llvm.atomic_or, dw, ((packed >> (4 * i)) & 0xF) << shift)


# ------------------------------------------------------------------ index scores
def e4m3_dwords(vals):
    """Eight f32 -> two dwords of e4m3 (RNE, saturated at 448): the NVFP4
    indexer's rounding of K and of q."""
    hi, lo = fx.Float32(E4M3_MAX), fx.Float32(-E4M3_MAX)
    out = []
    for d in range_constexpr(2):
        packed = fx.Int32(0)
        for h in range_constexpr(2):
            a, b = (fx.max(fx.min(vals[4 * d + 2 * h + i], hi), lo) for i in (0, 1))
            packed = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, packed, bool(h)))
        out.append(packed)
    return out


def e2m1_dword_e4m3x8(word, scale_byte):
    """Eight e2m1 of `word` (low nibble first) times the e4m3 scale in the low byte
    of `scale_byte`, rounded to e4m3 (the stock scorer's keys): two dwords. The
    products are exact in f32 (<= 6 significant bits)."""
    sc = fx.Vector(
        rocdl.cvt_scalef32_pk_f32_fp8(
            T.vec(2, T.f32),
            as_ir_value(fx.Int32(scale_byte) & 0xFF),
            as_ir_value(fx.Float32(1.0)),
            False,
        )
    )[0]
    prod = []
    for sel in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_f32_fp4(
                T.vec(2, T.f32),
                as_ir_value(fx.Int32(word)),
                as_ir_value(fx.Float32(1.0)),
                sel,
            )
        )
        prod += [fx.Float32(pair[0]) * sc, fx.Float32(pair[1]) * sc]
    return e4m3_dwords(prod)


def q_e4m3(frags):
    """Four bf16x8 index-q fragments -> the e4m3 B operand of one 16x16x128 MFMA."""
    words = []
    for f in frags:
        v = fx.Vector(f).to(fx.Float32)
        words += e4m3_dwords([fx.Float32(v[i]) for i in range(8)])
    return fx.Vector.from_elements(words, fx.Int32)


# ------------------------------------------------------------------ split attention
_V = fx.Vector
NEG = -3.4e38
_NEG_BITS = struct.unpack("i", struct.pack("f", NEG))[0]


def _masked(x, left):
    """`x`, or NEG where `left` < 0 (the key is past the context): a bit select by
    the sign of `left`, so a NaN `x` of a stale key is replaced too."""
    dead = fx.Int32(left) >> 31
    bits = x.bitcast(fx.Int32)
    return (bits ^ ((bits ^ fx.Int32(_NEG_BITS)) & dead)).bitcast(fx.Float32)


@traced
def _load_quarter(r_page, w, n, g, kpw, cm):
    """This lane's raw dwords of quarter `w` (32 keys) of a K/V page: K data of its
    two key rows (dims 32 g .. + 31), their K scale dword, V data (dword dt = keys
    8 g .. + 7 of dim 8 n + dt) and the V scale bytes of those keys' group n // 2.
    -> `(kd[2][4], ks, vd[2][4], vs[2])`."""
    kd = []
    for st in range_constexpr(2):
        row = w * kpw + 8 * (n // 4) + 4 * st + n % 4
        kd.append(
            _V(
                bo.buffer_load(
                    r_page,
                    (row * DATA_BYTES + g * 16) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
        )
    ks = fx.Int32(
        bo.buffer_load(
            r_page,
            (K_SCALE_AT + ((w * 4 + g) * 16 + n) * 4) // 4,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )
    vbase = (V_DATA_AT + ((w * 4 + g) * HEAD_DIM + 8 * n) * 4) // 4
    vd = [
        _V(
            bo.buffer_load(
                r_page, vbase + 4 * h, vec_width=4, dtype=T.i32, cache_modifier=cm
            )
        )
        for h in range_constexpr(2)
    ]
    vs = _V(
        bo.buffer_load(
            r_page,
            (V_SCALE_AT + (n // 2) * SPARSE_BLOCK + w * kpw + 8 * g) // 4,
            vec_width=2,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )
    return kd, ks, vd, vs


def _mfma_f16(a, b, c):
    return mfma_16x16x32(a, b, c, True)


@traced
def _dequant_quarter(raw, key_mask):
    """K fragments `kf[st][s]` (A operands of S^T: dims 32 g + 8 s .. + 7) and V^T
    fragments `vf[dt]` in f16, exact products; `key_mask[i]` (0xFFFF per live key of
    V^T dword i) keeps a stale key's NaN out of PV (its P is 0, but 0 x NaN is not)."""
    kd, ks, vd, vs = raw
    kf = []
    for st in range_constexpr(2):
        ksc = cvt_fp8x2_f16(ks, st == 1)  # groups 2 g, 2 g + 1 of row st
        row = []
        for s in range_constexpr(4):
            sv = _V.from_elements([ksc[s // 2], ksc[s // 2]], fx.Float16)
            row.append(
                _V.from_elements(
                    [
                        (cvt_fp4x2_f16(kd[st][s], i) * sv).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
            )
        kf.append(row)
    scales = []
    for h in range_constexpr(2):
        scales += [cvt_fp8x2_f16(vs[h], False), cvt_fp8x2_f16(vs[h], True)]
    words = [vd[h][e] for h in range(2) for e in range(4)]
    vf = []
    for dt in range_constexpr(8):
        vf.append(
            _V.from_elements(
                [
                    (cvt_fp4x2_f16(words[dt], i) * scales[i]).bitcast(fx.Int32)[0]
                    & key_mask[i]
                    for i in range_constexpr(4)
                ],
                fx.Int32,
            )
        )
    return kf, vf


@traced
def split_task(
    tid, lane, wave, tok, t, blk, q, kv_cache, k_gs, v_gs, sm_scale, hst, opart,
    n_ctx, put_o, put_m, put_l, heads, o_k, split_keys, waves, cm=0,
):  # fmt: skip
    """K4's split task (token `tok`, partition `t`: 256 keys x `heads` q heads)
    over NVFP4 K/V pages, f16 MFMA over the exact dequantized products:

      S^T = K Q^T   A = K tile (row r of tile st is key 8 (r // 4) + 4 st + r % 4 of
                    the wave's quarter), B = Q^T; lane (n, g) holds keys 8 g + 4 st
                    + e of head n
      O^T += V^T P^T  A = V^T tile dt (row n is dim 8 n + dt), B = P^T taken from
                    the S^T accumulators as they are

    Scores in the natural-log domain (`sm_scale * k_gs * dot`, as the fp8 split's),
    the partition's normalized output (times `v_gs`), max and sum handed to
    `put_o(e0, ov)`, `put_m(head, m)`, `put_l(head, l)`. `blk`: LDS of the task's
    16 page-16 ids (page-16 id p is block p // 8, keys 16 (p % 8) of it) then its
    context; `hst` / `opart`: the fp8 split's LDS. Every thread of the CTA runs it."""
    l16 = lane % 16
    g4 = lane // 16
    kpw = split_keys // waves  # a wave's keys: one quarter of a block
    ppw = kpw // 16
    c = fx.Float32(sm_scale) * k_gs
    p16 = uniform(fx.ptr_load(blk + wave * ppw))
    r_page = rsrc(kv_block(kv_cache, p16 // 8, False), KV_PAGE_BYTES)
    raw = _load_quarter(r_page, (p16 % 8) // 2, l16, g4, kpw, cm)
    qb = []
    r_q = rsrc(q)
    for s in range_constexpr(4):
        w4 = _V(
            bo.buffer_load(
                r_q,
                (tok * o_k + l16 * HEAD_DIM + 32 * g4 + 8 * s) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        qb.append(
            w4.bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)
        )
    key0 = t * split_keys + wave * kpw
    # this lane's keys of its 8 in the context; a V^T dword (two keys) gets 16 bits
    # a live key (integer arithmetic: compares would cost scalar registers)
    nl = n_ctx - (key0 + 8 * g4)
    key_mask = [
        fx.Int32(
            (
                (fx.Int64(1) << (fx.Int64(fx.min(fx.max(nl - 2 * i, 0), 2)) * 16))
                - fx.Int64(1)
            )
            & fx.Int64(0xFFFFFFFF)
        )
        for i in range(4)
    ]
    kf, vf = _dequant_quarter(raw, key_mask)
    sc = []
    for st in range_constexpr(2):
        acc = _V.filled(4, 0.0, fx.Float32)
        for s in range_constexpr(4):
            acc = _mfma_f16(kf[st][s], qb[s], acc)
        sc.append([_masked(acc[e] * c, nl - 1 - (4 * st + e)) for e in range(4)])
    m = sc[0][0]
    for st in range_constexpr(2):
        for e in range_constexpr(4):
            m = fx.max(m, sc[st][e])
    m = fx.max(m, xshfl(m, 16))
    m = fx.max(m, xshfl(m, 32))
    if g4 == 0:
        fx.ptr_store(m, hst + (wave * heads + l16))
    gpu.barrier()
    mx = fx.ptr_load(hst + l16)
    for w in range_constexpr(1, waves):
        mx = fx.max(mx, fx.ptr_load(hst + (w * heads + l16)))
    p = [[hw_exp2((sc[st][e] - mx) * LOG2E) for e in range(4)] for st in range(2)]
    ls = p[0][0]
    for st in range_constexpr(2):
        for e in range_constexpr(4):
            if (st, e) != (0, 0):
                ls = ls + p[st][e]
    ls = ls + xshfl(ls, 16)
    ls = ls + xshfl(ls, 32)
    if g4 == 0:
        fx.ptr_store(ls, hst + (waves * heads + wave * heads + l16))
    pt = _V.from_elements(
        [pk_f16(p[st][2 * h], p[st][2 * h + 1]) for st in range(2) for h in range(2)],
        fx.Int32,
    )
    # O^T tile dt: lane (n, g) rows 4 g + e = dims 32 g + 8 e + dt of head n
    for e in range_constexpr(4):
        outs = []
        for dt in range_constexpr(8):
            cv = _mfma_f16(vf[dt], pt, _V.filled(4, 0.0, fx.Float32))
            outs.append(cv[e])
        base = opart + (wave * o_k + l16 * HEAD_DIM + 32 * g4 + 8 * e)
        fx.ptr_store(_V.from_elements(outs[:4], fx.Float32), base)
        fx.ptr_store(_V.from_elements(outs[4:], fx.Float32), base + 4)
    if tid < heads:
        fx.ptr_store(mx, hst + (2 * waves * heads + tid))
    gpu.barrier()
    e0 = tid * 4
    head = e0 // HEAD_DIM
    lsum = fx.ptr_load(hst + (waves * heads + head))
    for w in range_constexpr(1, waves):
        lsum = lsum + fx.ptr_load(hst + (waves * heads + w * heads + head))
    inv_l = 1.0 / (lsum > 0.0).select(lsum, fx.Float32(1.0))
    ov = []
    for k in range_constexpr(4):
        acc = fx.ptr_load(opart + (e0 + k))
        for w in range_constexpr(1, waves):
            acc = acc + fx.ptr_load(opart + (w * o_k + e0 + k))
        ov.append((acc * inv_l) * v_gs)
    put_o(e0, ov)
    if tid < heads:
        put_m(tid, fx.ptr_load(hst + (2 * waves * heads + tid)))
        lt = fx.ptr_load(hst + (waves * heads + tid))
        for w in range_constexpr(1, waves):
            lt = lt + fx.ptr_load(hst + (waves * heads + w * heads + tid))
        put_l(tid, lt)
    gpu.barrier()
