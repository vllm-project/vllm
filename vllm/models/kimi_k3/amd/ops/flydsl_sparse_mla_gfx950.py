# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
# mypy: ignore-errors
#
# This file contains code copied from the ROCm/aiter project
# (aiter/ops/flydsl/kernels/sparse_mla_fp8ds_gfx950.py @
# a61e02c26a3915011e85b103574c25d457890825). The original source code was
# licensed under the MIT license. It is carried in-tree so a stock aiter build
# still gets the kernel; only the imports and this header differ.

"""FlyDSL sparse MLA forward over fp8_ds_mla records, gfx950.

Serves the DeepSeek-V3.2 / Kimi-K3 fp8_ds_mla path of ``sparse_mla_fwd``
(kv_lora_rank 512, rope 64, bf16 q, <= 16 heads, CSR index stream). The host
wrapper and its support predicate live in ``flydsl_sparse_mla.py``.

Record (656 B): 512 e4m3 latent | 4 f32 per-128 pow2 scales | 64 bf16 rope.

One 4-wave workgroup per query row (rows issued in reverse order, so the long
causal tail starts first), 128-token tiles over the row's CSR slice
``indices[indptr[q] : indptr[q + 1]]``:

  * QK: wave w computes S for tokens [32w, 32w + 32) of the tile, all heads,
    from the raw fp8 latent: a scaled MFMA 16x16x128 with the record's pow2
    scale as the E8M0 scale_a, and Q quantized per (head, 128) to an fp8 hi
    plus an fp8 lo residual. The rope part is a bf16 MFMA.
  * Softmax: the per-wave tile max is exchanged through LDS;
    P = exp2(s - m + 8).
  * PV: wave w owns output dims [128w, 128w + 128), which is exactly one KV
    scale block, so the V scale folds into P (P' = P * scale_v, fp16) and V
    stays raw fp8 in LDS (ds_read_b64_tr_b8, exact fp8 -> f16, f16 MFMA).
  * Tiles: nt = ceil(len / 128), or with has_invalid ceil(last_valid / 128),
    where a scan of the row finds the last entry >= 0. Positions >= len and,
    with has_invalid, entries < 0 are masked; their latent DMA reads row 0.
    Index reads go through a buffer descriptor sized to ``indices``, so a tail
    tile never reads past the end of the index buffer.
  * KV rows are addressed in 64 bits (global_load ... lds for the latent DMA,
    plain global loads for the scale and rope words), so the cache may span
    more than 4 GiB: vLLM hands the non-DCP path its whole packed block-row
    space, which does. Nothing bounds those reads; see the last precondition.

Preconditions the host cannot check:
  * Every per-128 KV scale is a power of two, as vLLM's fp8_ds_mla writer
    (concat_and_cache_ds_mla) stores it: QK takes the scale's exponent as the
    MFMA E8M0 scale, so a mantissa would be dropped.
  * Every per-128 KV scale is <= 2^7: P' = P * scale has to fit fp16 and
    P <= 2^8. vLLM's writer uses scale = pow2ceil(max(amax, 1e-4) / 448), so
    this holds iff |latent| <= 57344. A larger scale is clamped to 2^7
    in-kernel, which keeps the output finite but makes it inexact.
  * Every consumed index i satisfies 0 <= i < kv rows (as for the Gluon path).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64, ptr_buf_tensor
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

__all__ = [
    "KV_LORA_RANK",
    "MAX_HEADS",
    "QK_ROPE_HEAD_DIM",
    "RECORD_BYTES",
    "build_sparse_mla_fp8ds_module",
]

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
RECORD_BYTES = KV_LORA_RANK + 16 + 2 * QK_ROPE_HEAD_DIM  # 656
MAX_HEADS = 16

_L = KV_LORA_RANK
_TILE = 128
_WAVES = 4
_BLOCK_THREADS = _WAVES * 64
_LN2 = 0.6931471805599453
# fp16 P = exp2(s - m + 8); l carries the same 2^8.
_P_SHIFT = 8.0
# 2^7 as f32 bits: P' = P * scale <= 2^15 stays under the fp16 max.
_SCALE_CAP_BITS = 0x43000000
# QK K-operand reads kept in flight ahead of the MFMA that consumes them.
_K_DEPTH = 2

# LDS map (bytes). The small regions go first so their offsets fit the 16-bit
# ds immediate.
_V_BYTES = 4 * 8 * 4 * 4 * 128  # 4 column blocks x [128 tok][128 B]
_MX_BYTES = 16 * 4 * 4  # [h][w] f32
_KS_BYTES = _WAVES * 4 * 32 * 4  # [wave][w'][32 tok] f32
_SINK_BYTES = 64  # dump slot for lanes h >= H
_KS_OFF = 0
_MX_OFF = _KS_BYTES
_SINK_OFF = _KS_BYTES + _MX_BYTES
_V_OFF = (_KS_BYTES + _MX_BYTES + _SINK_BYTES + 127) // 128 * 128
_P_OFF = _V_OFF + _V_BYTES


def _lds_ptr(elem, addr, align):
    return fx.inttoptr(
        fx.PointerType.get(elem, fx.AddressSpace.Shared, align), fx.Int32(addr)
    )


def _lds_load(elem, n, addr, align):
    return Vec(fx.ptr_load(_lds_ptr(elem, addr, align), result_type=T.vec(n, elem)))


def _lds_store(vec, elem, addr, align):
    fx.ptr_store(Vec(vec).ir_value(), _lds_ptr(elem, addr, align))


def _lds_llvm_ptr(addr):
    return fx.to_llvm_ptr(
        fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(addr))
    )


def _global_llvm_ptr(addr):
    return fx.to_llvm_ptr(
        fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Global), fx.Int64(addr))
    )


def _global_i32(addr, words):
    """``words`` i32 (1 or 4) at the 64-bit global address ``addr``: a cache
    past 4 GiB is out of a buffer descriptor's 32-bit offset reach."""
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4 * words), fx.Int64(addr)
    )
    if words == 1:
        return fx.Int32(fx.generic_load(ptr, dtype=fx.Int32).ir_value())
    return Vec(fx.generic_load(ptr, dtype=fx.Vector[fx.Int32, words]).ir_value())


def _tr8(addr, imm):
    ptr = _lds_llvm_ptr(fx.Int32(addr) + imm)
    return Vec(rocdl.ds_read_tr8_b64_(T.vec(2, T.i32), ptr))


def _cat(a, b, n):
    return Vec(a).shuffle(Vec(b), list(range(2 * n)))


def _mfma_fp8(a, b, c, sa, sb):
    return Vec(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).ir_value()),
                _raw(Vec(b).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
                _raw(fx.Int32(sa)),
                0,
                _raw(fx.Int32(sb)),
            ],
        )
    )


def _mfma_f16(a, b, c):
    return Vec(
        rocdl.mfma_f32_16x16x32_f16(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).bitcast(fx.Float16).ir_value()),
                _raw(Vec(b).bitcast(fx.Float16).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _mfma_bf16(a, b, c):
    return Vec(
        rocdl.mfma_f32_16x16x32_bf16(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).bitcast(fx.BFloat16).ir_value()),
                _raw(Vec(b).bitcast(fx.BFloat16).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _cvt_v8(x):
    """8 raw e4m3 (2 i32 from tr8) -> 8 f16 (4 i32), exact."""
    one = _raw(fx.Float32(1.0).ir_value())
    out = []
    for i in range(2):
        wd = _raw(fx.Int32(Vec(x)[i]).ir_value())
        for hs in (False, True):
            out.append(Vec(rocdl.cvt_scalef32_pk_f16_fp8(T.vec(2, T.f16), wd, one, hs)))
    return _cat(_cat(out[0], out[1], 2), _cat(out[2], out[3], 2), 4).bitcast(fx.Int32)


def _pack_fp8x4(vals):
    """4 f32 -> one i32 of e4m3 (byte i = vals[i])."""
    w = rocdl.cvt_pk_fp8_f32(
        T.i32,
        _raw(fx.Float32(vals[0]).ir_value()),
        _raw(fx.Float32(vals[1]).ir_value()),
        llvm.mlir_poison(T.i32),
        False,
    )
    w = rocdl.cvt_pk_fp8_f32(
        T.i32,
        _raw(fx.Float32(vals[2]).ir_value()),
        _raw(fx.Float32(vals[3]).ir_value()),
        w,
        True,
    )
    return fx.Int32(w)


def _unpack_fp8x4(word):
    wd = _raw(fx.Int32(word).ir_value())
    a = Vec(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, False))
    b = Vec(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, True))
    return [fx.Float32(a[0]), fx.Float32(a[1]), fx.Float32(b[0]), fx.Float32(b[1])]


def _quant(vals, inv):
    """Vals * inv -> fp8 words (hi), and the residual * 32 as fp8 words (lo,
    whose E8M0 scale is the hi one minus 5)."""
    x = [fx.Float32(v) * inv for v in vals]
    hi = [_pack_fp8x4(x[4 * i : 4 * i + 4]) for i in range(len(x) // 4)]
    lo = []
    for i in range(len(hi)):
        dec = _unpack_fp8x4(hi[i])
        lo.append(_pack_fp8x4([(x[4 * i + j] - dec[j]) * 32.0 for j in range(4)]))
    return hi, lo


def _lo_scale(s):
    s = fx.Int32(s) - 5
    return (s < 0).select(fx.Int32(0), s)


def _swap_pair(fn, x):
    st = llvm.StructType.get_literal([T.i32, T.i32])
    v = _raw(fx.Float32(x).bitcast(fx.Int32).ir_value())
    r = fn(st, v, v, False, False)
    a = fx.Int32(llvm.ExtractValueOp(T.i32, r, [0]).res).bitcast(fx.Float32)
    b = fx.Int32(llvm.ExtractValueOp(T.i32, r, [1]).res).bitcast(fx.Float32)
    return fx.Float32(a), fx.Float32(b)


def _fmax(a, b):
    return fx.maxnumf(a, b, fastmath="nnan")


def _xsum(x):
    """Sum over the 4 lane groups (xor 16, 32) via permlane swaps."""
    a, b = _swap_pair(rocdl.permlane16_swap, x)
    a, b = _swap_pair(rocdl.permlane32_swap, a + b)
    return a + b


def _xmax(x):
    """Max over the 4 lane groups (xor 16, 32) via permlane swaps."""
    a, b = _swap_pair(rocdl.permlane16_swap, x)
    a, b = _swap_pair(rocdl.permlane32_swap, _fmax(a, b))
    return _fmax(a, b)


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def _readfirstlane(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _e8m0_for(amax):
    """E8M0 byte s with amax / 2^(s - 127) in [128, 256), and f32 2^(127 - s)."""
    e = (fx.Float32(amax).bitcast(fx.Int32) >> 23) & 0xFF
    s = e - 7
    s = (s < 1).select(fx.Int32(1), s)
    inv = ((254 - s) << 23).bitcast(fx.Float32)
    return s, fx.Float32(inv)


def _p_store_f16(pl, svs, p_wr, poff, p_wave_bytes):
    for wp in range_constexpr(4):
        pv = [
            pl[4 * s + r] * fx.Float32(svs[s][wp][r])
            for s in range(2)
            for r in range(4)
        ]
        hv = Vec.from_elements(pv, fx.Float32).to(fx.Float16).bitcast(fx.Int32)
        _lds_store(hv, T.i32, p_wr + poff(wp * p_wave_bytes), 16)


def _pv_operands(p_rd, num_heads):
    # kk = 2hf + sub; P block (w_src = g // 2 + 2hf, g_src = 2(g % 2) + sub)
    return [
        _lds_load(
            T.i32, 4, p_rd + ((kk // 2) * 8 * num_heads + (kk % 2) * num_heads) * 16, 16
        )
        for kk in range(4)
    ]


def _pv_mfma(d0, v_a, pp, o):
    """O^T[d0 .. d0 + 3] += V^T P'^T for 4 output column chunks, in place."""
    for i in range_constexpr(4):
        for kk in range_constexpr(4):
            o[d0 + i] = _mfma_f16(_cvt_v8(v_a[i][kk]), pp[kk], o[d0 + i])


def _q_prologue(q_t, q_row, g, h_ok):
    """Quantize this lane's q row to fp8 hi/lo MFMA operands.

    MFMA scale group kb (lanes 16kb.. supply its E8M0) is K bytes
    32kb..32kb + 31, so lane d's bytes 0-15 belong to group d // 2 and bytes
    16-31 to group 2 + d // 2. Lane g therefore holds, for MFMA j, chunk
    2j + g % 2 of record block g // 2 (first 16 B) and of block 2 + g // 2
    (second 16 B), and supplies the scale of block g.

    Returns (q8 hi operands, hi E8M0, q8 lo operands, lo E8M0).
    """
    va, vb = [], []
    for j in range(4):
        c = 2 * j + g % 2
        for hb, dst in ((0, va), (2, vb)):
            d0 = (hb + g // 2) * 128 + c * 16  # bf16 dims
            for k in range(2):
                v = Vec(q_t.load(q_row + d0 // 2 + k * 4, vec_size=4))
                dst.append(
                    [fx.Float32(x) for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)]
                )
    va = [h_ok.select(x, fx.Float32(0.0)) for blk in va for x in blk]
    vb = [h_ok.select(x, fx.Float32(0.0)) for blk in vb for x in blk]
    am = []
    for vv in (va, vb):
        a = fx.Float32(0.0)
        for x in vv:
            a = _fmax(a, _fmax(x, fx.Float32(0.0) - x))
        p0, p1 = _swap_pair(rocdl.permlane16_swap, a)
        am.append(_fmax(p0, p1))  # amax of block g // 2 (va) / 2 + g // 2 (vb)
    oth = []
    for a in am:
        p0, p1 = _swap_pair(rocdl.permlane32_swap, a)
        oth.append((g >= 2).select(p0, p1))  # value of lane g ^ 2
    lo_half = g < 2
    blk = [
        lo_half.select(am[0], oth[0]),
        lo_half.select(oth[0], am[0]),
        lo_half.select(am[1], oth[1]),
        lo_half.select(oth[1], am[1]),
    ]
    mine = (g == 0).select(
        blk[0], (g == 1).select(blk[1], (g == 2).select(blk[2], blk[3]))
    )
    _, inva = _e8m0_for(am[0])
    _, invb = _e8m0_for(am[1])
    q8, q8l = [], []
    for j in range(4):
        ha, la = _quant(va[16 * j : 16 * j + 16], inva)
        hb, lb = _quant(vb[16 * j : 16 * j + 16], invb)
        q8.append(Vec.from_elements(ha + hb, fx.Int32))
        q8l.append(Vec.from_elements(la + lb, fx.Int32))
    qs, _ = _e8m0_for(mine)
    return q8, qs, q8l, _lo_scale(qs)


class _BoundedI32(GTensor):
    """GTensor whose buffer descriptor carries the real byte size, so an
    out-of-range read returns 0 instead of touching memory past the end."""

    def __init__(self, memref, nbytes):
        super().__init__(memref, dtype=T.i32, shape=(-1,))
        self.nbytes = nbytes

    def _view(self, elem, unit_elems):
        return ptr_buf_tensor(
            self.base_i64,
            elem,
            unit_elems=unit_elems,
            unit_stride=1,
            num_records_bytes=fx.Int64(self.nbytes),
        )


def _guarded(pred, fn):
    """Run fn() under a runtime predicate (an scf.if around its body)."""

    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


def build_sparse_mla_fp8ds_module(num_heads, has_invalid, return_lse):
    """JIT launcher for one (heads, has_invalid, return_lse) configuration.

    The latent tile lives in LDS as 4 column blocks (bytes 128b..128b + 128 of
    the record), filled directly by global_load ... lds (DMA) from 64-bit
    per-lane row addresses. Wave w streams
    the next tile's column block w right after its own PV tr8 reads; only
    wave w reads block w in PV.
    """
    H = num_heads
    if not 1 <= H <= MAX_HEADS:
        raise ValueError(f"num_heads must be in [1, {MAX_HEADS}], got {H}")
    lds_bytes = _P_OFF + 4 * 16 * H * 16
    p_wave_bytes = 16 * H * 16  # P bytes per destination wave w'
    name = f"sparse_mla_fp8ds_gfx950_h{H}_i{int(has_invalid)}_l{int(return_lse)}_flydsl"

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[_BLOCK_THREADS, 1, 1])
    def sparse_mla_fp8ds_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        IDX: fx.Tensor,
        IPTR: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        stride_q: fx.Int32,
        stride_qh: fx.Int32,
        stride_o: fx.Int32,
        stride_oh: fx.Int32,
        c_log2: fx.Float32,
        n_q: fx.Int32,
        idx_bytes: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        t = lane % 16  # QK: token row; S/O: head
        g = lane // 16
        qi = fx.Int32(n_q - 1) - fx.Int32(fx.block_idx.x)  # long rows first

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)  # i64: the cache may span more than 4 GiB
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        # ---- this row's CSR slice (wave-uniform)
        r_beg = _readfirstlane(ip_t[qi])
        r_end = _readfirstlane(ip_t[qi + 1])
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)

        # ---- Q prologue: lane (h = t, g) holds all 128 dims of block g as fp8 hi/lo
        h_ok = t < H
        h_safe = h_ok.select(t, fx.Int32(0))
        q_row = (qi * stride_q + h_safe * stride_qh) // 2  # i32 units (bf16 pairs)
        q8, qs, q8l, qsl = _q_prologue(q_t, q_row, g, h_ok)
        qr = [
            Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4))
            for i in range_constexpr(2)
        ]

        # ---- tile count. has_invalid: tiles past the last entry >= 0 are
        # skipped (the scan loop runs 0 times otherwise); -1 entries before it
        # are masked as usual.
        n_scan = ((r_len + 255) // 256) if has_invalid else fx.Int32(0)
        zero = _raw(fx.Int32(0).ir_value())
        for sc_c, sc_st in range(0, n_scan, 1, init=[zero, zero]):
            lastp = fx.Int32(sc_st[0])
            p0 = fx.Int32(sc_c) * 256 + lane * 4
            v = Vec(idx_t.load(r_beg + p0, vec_size=4))
            for j in range_constexpr(4):
                pj = p0 + j
                lastp = (pj < r_len).select(
                    (fx.Int32(v[j]) >= 0).select(pj + 1, lastp), lastp
                )
            sc_fin = yield [lastp, fx.Int32(sc_st[1])]
        lastp = fx.Int32(sc_fin[0]) if has_invalid else r_len
        for sh in (1, 2, 4, 8, 16, 32):
            o_ = fx.Int32(lastp).shuffle_xor(sh, 64)
            lastp = (o_ > lastp).select(o_, lastp)
        nt = _readfirstlane((lastp + (_TILE - 1)) // _TILE)

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
            v_rd.append(
                base + _V_OFF + w * 16384 + tok * 128 + fsw(tok) * 16 + (t % 2) * 8
            )
        # DMA: load j fills tokens 8j..8j + 7 of block w;
        # lane -> token 8j + lane // 8, chunk (lane % 8) ^ f.
        dma_c = [
            w * 128
            + ((lane % 8) ^ (4 * (jj & 1) + lane // 16) ^ (2 * ((jj >> 1) & 1))) * 16
            for jj in range_constexpr(4)
        ]
        dma_src = [((8 * j + lane // 8) & 63) * 4 for j in range(8)]

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

        def poff(c):
            return h_ok.select(fx.Int32(c), fx.Int32(0))

        def ld_idx(p):
            """Slot of row position p; -1 past the row end (and for -1 entries)."""
            ix = fx.Int32(idx_t[r_beg + p])
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
                row = (
                    kv_base + fx.Int64((ix >= 0).select(ix, fx.Int32(0))) * RECORD_BYTES
                )
                out.append(
                    Vec.from_elements(
                        [_global_i32(row + fx.Int64(_L + g * 4), 1)], fx.Int32
                    )
                )
                for i in range_constexpr(2):
                    out.append(
                        _global_i32(row + fx.Int64(_L + 16 + 64 * i + g * 16), 4)
                    )
            return out

        def dma(ixd):
            ixs = [(fx.Int32(ix) >= 0).select(fx.Int32(ix), fx.Int32(0)) for ix in ixd]
            for j in range_constexpr(16):
                # The slot id is what gets permuted; its byte offset needs 64 bits.
                ix = fx.Int32(
                    rocdl.ds_bpermute(T.i32, _raw(dma_src[j % 8]), _raw(ixs[j // 8]))
                )
                src = _global_llvm_ptr(
                    kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(dma_c[j % 4])
                )
                dst = _lds_llvm_ptr(base + w * 16384 + (_V_OFF + j * 1024))
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
            gpu.barrier()
            t2 = nxt(tb, 2)
            ixq_nn = ld_q_idx(t2)

            # ---- QK (A = K rows from LDS, B = Q^T) -> S^T:
            # lane (h = t, g) holds tokens 16s + 4g + r.
            def kread(j):
                s, blk = j // 4, j % 4
                ka = k_rdb[s] ^ (32 * blk)
                return _cat(
                    _lds_load(T.i32, 4, ka, 16), _lds_load(T.i32, 4, ka + 32768, 16), 4
                )

            kl = [kread(j) for j in range_constexpr(_K_DEPTH)] + [None] * (8 - _K_DEPTH)
            sacc = [Vec.filled(4, 0.0, fx.Float32), Vec.filled(4, 0.0, fx.Float32)]
            for j in range_constexpr(8):
                s, blk = j // 4, j % 4
                rocdl.sched_barrier(0)
                lt = kl[j]
                kl[j] = None
                if const_expr(j + _K_DEPTH < 8):
                    kl[j + _K_DEPTH] = kread(j + _K_DEPTH)
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
            gpu.barrier()

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
            for j in range_constexpr(4):
                m_new = _fmax(m_new, fx.Float32(mx[j]))
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
                    [fx.Float32(Vec(o[i])[j]) * alpha for j in range(4)], fx.Float32
                )
                for i in range(8)
            ]

            _p_store_f16(pl, svs, p_wr, poff, p_wave_bytes)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

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

        # ---- epilogue: l over the 4 waves, normalize, store bf16 for h < H
        rocdl.s_waitcnt(vmcnt=0)
        l_w = _xsum(fx.Float32(fin[1]))
        _lds_store(Vec.from_elements([l_w], fx.Float32), T.f32, mx_wr, 4)
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        lv = _lds_load(T.f32, 4, mx_rd, 16)
        l_tot = (
            fx.Float32(lv[0])
            + fx.Float32(lv[1])
            + fx.Float32(lv[2])
            + fx.Float32(lv[3])
        )
        has_l = l_tot > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_tot, fx.Float32(0.0))
        o_base = (qi * stride_o + h_safe * stride_oh + w * 128 + 4 * g) // 2
        for dc in range_constexpr(8):
            ov = Vec(fin[2 + dc])
            bf = Vec.from_elements(
                [fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32
            ).to(fx.BFloat16)

            val = bf.bitcast(fx.Int32)

            def _st(_v=val, _off=o_base + dc * 8):
                o_t.store(_off, _v, vec_size=2)

            _guarded(h_ok, _st)

        if const_expr(return_lse):
            # Natural-log LSE of softmax_scale * q.k over the unmasked keys;
            # -inf when there is none.
            m_fin = fx.Float32(fin[0])
            lse2 = (
                m_fin
                + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0))))
                - _P_SHIFT
            )
            lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
            lane0 = h_ok.select(w + g, fx.Int32(1)) == 0

            lse_bits = lse.bitcast(fx.Int32)

            def _st_lse(_v=lse_bits, _off=qi * H + t):
                lse_t.store(_off, _v)

            _guarded(lane0, _st_lse)

    @flyc.jit
    def launch_sparse_mla_fp8ds(
        Q,
        KV,
        IDX,
        IPTR,
        OUT,
        LSE,
        n_q,
        stride_q,
        stride_qh,
        stride_o,
        stride_oh,
        c_log2,
        idx_bytes,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_fp8ds_kernel(
            Q,
            KV,
            IDX,
            IPTR,
            OUT,
            LSE,
            stride_q,
            stride_qh,
            stride_o,
            stride_oh,
            c_log2,
            n_q,
            idx_bytes,
        ).launch(
            grid=(fx.Int64(n_q), 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream
        )

    launch_sparse_mla_fp8ds.compile_hints = {"waves_per_eu": 2, "fast_fp_math": False}
    return launch_sparse_mla_fp8ds
