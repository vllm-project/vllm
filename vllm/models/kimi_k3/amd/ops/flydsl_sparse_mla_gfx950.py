# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
# mypy: ignore-errors
#
# The <= 16-head kernel and the helpers before it are copied from the
# ROCm/aiter project (aiter/ops/flydsl/kernels/sparse_mla_fp8ds_gfx950.py @
# a61e02c26a3915011e85b103574c25d457890825), with 64-bit KV addressing added.
# The original source code was licensed under the MIT license. The many-head
# kernel after it is built on those helpers.

"""FlyDSL sparse MLA forward over fp8_ds_mla records, gfx950.

Serves the DeepSeek-V3.2 / Kimi-K3 fp8_ds_mla path of ``sparse_mla_fwd``
(kv_lora_rank 512, rope 64, bf16 q, CSR index stream) with two kernels: one
for <= 16 heads (``build_sparse_mla_fp8ds_module``) and one for multiples of
16 such as the 96 gathered heads of Kimi-K3.1's DCP branch
(``build_sparse_mla_fp8ds_mh_module``, see its section below). The host
wrapper and its support predicate live in ``flydsl_sparse_mla.py``.

Record (656 B): 512 e4m3 latent | 4 f32 per-128 pow2 scales | 64 bf16 rope.

<= 16 heads: one 4-wave workgroup per query row (rows issued in reverse
order, so the long causal tail starts first), 128-token tiles over the row's
CSR slice ``indices[indptr[q] : indptr[q + 1]]``:

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

The many-head kernel runs the same math (Q split into fp8 hi + lo, scaled fp8
MFMA on the raw latent, bf16 MFMA for rope, fp16 PV on tr8-read raw fp8 with
the V scale folded into P) laid out so that each KV tile is gathered into LDS
once for all of a row's heads, instead of once per 16 heads:

  * Workgroup = (query row, head split). It serves HG groups of 16 heads with
    2 waves per group; NSPLIT = H / (16 HG) workgroups share a row, placed on
    one XCD so they share its L2.
  * The two waves of a group split the 512 latent columns: wave half c owns
    columns [256c, 256c + 256) (record scale blocks 2c, 2c + 1) and rope dims
    [32c, 32c + 32). It holds only its share of Q and accumulates only
    O[:, 256c : 256c + 256].
  * QK: each wave computes partial S over its columns for the tile's 32
    tokens; the pair swaps the partials through LDS, so both hold the full S
    and run the same softmax (no P exchange).
  * Softmax: P = exp2(s - m + 4), with m allowed to lag the running max by up
    to 4 (the O rescale is skipped while it does), so P <= 2^8 as above.
  * PV: the S register layout is the PV B operand when the K order is
    k = 8g + i <-> token 16(i // 4) + 4g + i % 4, which the tr8 V^T reads use.
  * KV tiles of 32 tokens (latent 2 x [32][256 B] XOR-swizzled, rope
    [32][128 B], scales [32][16 B]) are gathered into 2 LDS stages (tile i + 1
    while tile i is computed) by global_load ... lds on 64-bit per-lane
    addresses. The DMAs carry an alias scope that every LDS access is declared
    not to alias, so the compiler does not wait for the in-flight tile before
    each ds_read; the explicit vmcnt wait + barrier at the top of each tile
    orders them.
  * No has_invalid: every index in [indptr[0], indptr[T]) must be a valid
    slot. Positions past a row's end are masked and read row 0; rows run long
    first; KV scales are capped at 2^7; same preconditions otherwise.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64, ptr_buf_tensor
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

__all__ = [
    "KV_LORA_RANK",
    "MAX_HEADS",
    "QK_ROPE_HEAD_DIM",
    "RECORD_BYTES",
    "build_sparse_mla_fp8ds_mh_module",
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


# ---- the many-head kernel (H a multiple of 16, the DCP branch's 96 gathered
# heads)

# P = exp2(s - m + _MH_P_SHIFT) with m allowed to lag the running max by up to
# _MH_TAU (lazy O rescale), so P <= 2^(_MH_TAU + _MH_P_SHIFT) and
# P' = P * scale <= 2^15 fits fp16 for KV scales <= 2^7.
_MH_P_SHIFT = 4.0
_MH_TAU = 4.0
_MH_NEG = -1.0e30
# KV tile and its LDS stage: latent 2 x [_MH_TILE][256 B], rope
# [_MH_TILE][128 B], scales [_MH_TILE][16 B] (one 1 KB DMA job).
_MH_TILE = 32
_MH_NS16 = _MH_TILE // 16  # QK subtiles a tile
_MH_ROPE_OFF = _MH_TILE * 512
_MH_SC_OFF = _MH_ROPE_OFF + _MH_TILE * 128
_MH_STAGE = _MH_SC_OFF + 1024
_MH_XCH_OFF = 2 * _MH_STAGE  # after the 2 stages: the QK partials exchange
# Epilogue: a wave's O tile as bf16 [16 heads][256 dims], row pitch 512 + 16 B.
_MH_OP = 528
# Latent / rope DMA jobs of 1 KB a tile; + 1 scale job.
_MH_NL, _MH_NR = _MH_TILE // 2, _MH_TILE // 8
_MH_NJ = _MH_NL + _MH_NR + 1


def _dma_scope():
    """Alias scope of the KV DMAs (an ArrayAttr; built while the kernel is traced)."""
    dom = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "mla_lds">'
    dma = ir.Attribute.parse(
        f'#llvm.alias_scope<id = distinct[1]<>, domain = {dom}, description = "kv_dma">'
    )
    return ir.ArrayAttr.get([dma])


def _ld(sc, elem, n, addr, align):
    """LDS load of n x elem that does not alias an in-flight KV DMA (scope sc)."""
    ty = T.vec(n, elem) if n > 1 else elem
    v = llvm.load(ty, _lds_llvm_ptr(addr), alignment=align, noalias_scopes=sc)
    return Vec(v) if n > 1 else v


def _st(sc, vec, addr, align):
    llvm.store(
        Vec(vec).ir_value(), _lds_llvm_ptr(addr), alignment=align, noalias_scopes=sc
    )


def _tr8_scoped(sc, addr):
    return Vec(
        rocdl.ds_read_tr8_b64_(T.vec(2, T.i32), _lds_llvm_ptr(addr), noalias_scopes=sc)
    )


def _fsw(tok):
    """16-B chunk swizzle of a 256-B latent half row: conflict-free QK (16
    consecutive tokens) and tr8 reads."""
    return (tok & 15) ^ (((tok >> 4) & 1) << 3)


def _rsw(tok):
    """16-B chunk swizzle of a 128-B rope row."""
    return (tok >> 1) & 7


def _f32(x):
    return fx.Int32(x).bitcast(fx.Float32)


def _amax_bf16(words):
    """The max |x| over bf16 pairs (i32 words), as f32: a packed u16 max of the
    magnitude bits."""
    v2 = T.vec(2, T.i16)
    m = None
    for w in words:
        mw = llvm.bitcast(v2, _raw(fx.Int32(fx.Int32(w) & 0x7FFF7FFF).ir_value()))
        m = mw if m is None else arith.maxui(m, mw)
    mi = fx.Int32(llvm.bitcast(T.i32, m))
    lo16 = mi & 0xFFFF
    hi16 = (mi >> 16) & 0xFFFF
    return ((lo16 > hi16).select(lo16, hi16) << 16).bitcast(fx.Float32)


def _quant_bf16(words, inv):
    """bf16 pairs * inv -> fp8 hi = the value with its mantissa truncated to 3
    bits (exact in e4m3 for |x * inv| >= 2^-6; smaller values are negligible
    next to the block's amax in [128, 256)) and fp8 lo = the remainder * 16
    (E8M0 of hi minus 4; the remainder is below one hi ulp, so * 16 stays
    < 256)."""
    hv, lv = [], []
    for w in words:
        w = fx.Int32(w)
        t = w & fx.Int32(-0x000F0010)
        x = [
            (w << 16).bitcast(fx.Float32),
            (w & fx.Int32(-0x10000)).bitcast(fx.Float32),
        ]
        h = [
            (t << 16).bitcast(fx.Float32),
            (t & fx.Int32(-0x10000)).bitcast(fx.Float32),
        ]
        for k in range(2):
            hv.append(h[k] * inv)
            lv.append((x[k] - h[k]) * (inv * 16.0))
    return (
        [_pack_fp8x4(hv[4 * i : 4 * i + 4]) for i in range(len(hv) // 4)],
        [_pack_fp8x4(lv[4 * i : 4 * i + 4]) for i in range(len(lv) // 4)],
    )


def _barrier():
    """s_barrier without the workgroup fence gpu.barrier adds: that release
    fence would wait (vmcnt) for the in-flight KV DMA. Callers wait lgkmcnt /
    vmcnt themselves; the memory clobber keeps LDS accesses in place."""
    llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


def _order(pin=()):
    """Compiler-only fence: LDS accesses are not moved across it, and the
    values in `pin` (v4f32 MFMA accumulators) are produced before it (no-op asm
    with tied operands). Returns the pinned values."""
    pin = list(pin)
    if not pin:
        llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
        rocdl.sched_barrier(0)
        return []
    irs = [Vec(v).ir_value() for v in pin]
    if len(irs) == 1:
        r = llvm.inline_asm(
            irs[0].type, irs, "", "=v,0,~{memory}", has_side_effects=True
        )
        rocdl.sched_barrier(0)
        return [Vec(r)]
    ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<4xf32>'] * len(irs))})>")
    cons = ",".join(
        ["=v"] * len(irs) + [str(i) for i in range(len(irs))] + ["~{memory}"]
    )
    ret = llvm.inline_asm(ty, irs, "", cons, has_side_effects=True)
    rocdl.sched_barrier(0)
    return [Vec(llvm.extractvalue(irs[i].type, ret, [i])) for i in range(len(irs))]


def build_sparse_mla_fp8ds_mh_module(num_heads, return_lse, hg, waves_per_eu):
    """JIT launcher of the many-head kernel for H = num_heads heads (a multiple
    of 16), no has_invalid: hg 16-head groups per workgroup, waves_per_eu as
    the compile hint."""
    H = num_heads
    if H % 16 or (H // 16) % hg:
        raise ValueError(f"unsupported num_heads {H} with {hg} head groups")
    nsplit = H // 16 // hg
    W = 2 * hg  # waves
    NTH = 64 * W
    LDS = max(_MH_XCH_OFF + hg * 2 * _MH_NS16 * 1024, W * 16 * _MH_OP)
    J = (_MH_NJ + W - 1) // W  # DMA instructions per wave per tile

    name = (
        f"sparse_mla_fp8ds_mh_gfx950_h{H}_g{hg}_w{waves_per_eu}"
        f"_l{int(return_lse)}_flydsl"
    )

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, LDS, 1024]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def sparse_mla_fp8ds_mh_kernel(
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
        sc = _dma_scope()
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        l16 = lane % 16
        g = lane // 16
        grp = w // 2
        half = w % 2
        bid = fx.Int32(fx.block_idx.x)
        if const_expr(nsplit > 1):
            # Workgroups go round-robin over the 8 XCDs: give each XCD a
            # contiguous run of (row, split) so a row's splits share one L2.
            per = fx.Int32(n_q * nsplit) // 8
            sw = (bid % 8) * per + bid // 8
            bid = (bid < per * 8).select(sw, bid)
            sp = bid % nsplit
            qi = bid // nsplit
        else:
            sp = fx.Int32(0)
            qi = bid
        qi = fx.Int32(n_q - 1) - qi  # long rows first
        gi = sp * hg + grp  # 16-head group
        head = gi * 16 + l16

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_t = GTensor(KV, dtype=T.i32, shape=(-1,))
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))
        kv_base = kv_t.base_i64  # i64: the cache may span more than 4 GiB
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        # ---- this row's CSR slice and the indices of tiles 0 and 1 (issued
        # before Q: vmcnt is in order)
        ip_rsrc = ip_t.rsrc
        r_beg = fx.Int32(
            buffer_ops.buffer_load(ip_rsrc, qi, vec_width=1, is_scalar=True)
        )
        r_end = fx.Int32(
            buffer_ops.buffer_load(ip_rsrc, qi + 1, vec_width=1, is_scalar=True)
        )
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)
        idxv = [fx.Int32(idx_t[r_beg + _MH_TILE * t + lane]) for t in range(2)]

        # ---- DMA jobs of this wave (1 KB each; lane L writes LDS bytes
        # 16L .. 16L + 15 of the job):
        #   latent n < _MH_NL: half n // 8, tokens 4(n % 8) + L // 16, swizzled
        #     16-B chunk L % 16;
        #   rope n - _MH_NL: tokens 8(n - _MH_NL) + L // 8, swizzled chunk L % 8;
        #   scale: token L % _MH_TILE.
        # Slots past _MH_NJ redo job n - _MH_NJ (same bytes, harmless).
        def job(k, lane):
            """-> (token, source byte in the record, stage-relative LDS byte) of
            this wave's DMA slot k."""
            n = w + W * k
            n = (n < _MH_NJ).select(n, n - _MH_NJ)
            is_l = n < _MH_NL
            is_r = n < _MH_NL + _MH_NR
            lh = n // (_MH_TILE // 4)
            lt0 = 4 * (n % (_MH_TILE // 4))
            ltok = lt0 + lane // 16
            lsrc = 256 * lh + 16 * ((lane % 16) ^ _fsw(ltok))
            ldst = lh * (_MH_TILE * 256) + lt0 * 256
            rj = n - _MH_NL
            rtok = 8 * rj + lane // 8
            rsrc = 528 + 16 * ((lane % 8) ^ _rsw(rtok))
            rdst = _MH_ROPE_OFF + 1024 * rj
            stok = lane % _MH_TILE
            return (
                is_l.select(ltok, is_r.select(rtok, stok)),
                is_l.select(lsrc, is_r.select(rsrc, fx.Int32(512))),
                _readfirstlane(
                    is_l.select(ldst, is_r.select(rdst, fx.Int32(_MH_SC_OFF)))
                ),
            )

        def dma(idxv, tb, stage, lane=lane):
            """Gather tile rows [tb, tb + _MH_TILE) into stage: idxv lane l holds
            indices[r_beg + tb + l]. Always J DMAs per wave, so the vmcnt waits
            below are static. The row offset is 64-bit: a buffer descriptor's
            32-bit offset cannot reach past 4 GiB, and its wave-uniform base
            cannot follow a per-lane gather."""
            for k in range_constexpr(J):
                tok, src, dst = job(k, lane)
                ix = fx.Int32(
                    rocdl.ds_bpermute(T.i32, _raw(tok * 4), _raw(fx.Int32(idxv)))
                )
                ix = (tb + tok < r_len).select(ix, fx.Int32(0))  # past the end: row 0
                addr = kv_base + fx.Int64(ix) * fx.Int64(RECORD_BYTES) + fx.Int64(src)
                rocdl.global_load_lds(
                    _global_llvm_ptr(addr),
                    _lds_llvm_ptr(base + stage * _MH_STAGE + dst),
                    16,
                    0,
                    alias_scopes=sc,
                )

        dma(idxv[0], fx.Int32(0), fx.Int32(0))
        _order()
        # ---- Q: lane (h = l16, d = g) holds, for scale block 2 half + j, dims
        # 32d .. 32d + 31 as fp8 hi / lo (one E8M0 per head and block), and rope
        # dims 32 half + 8d .. + 7 (bf16).
        q_row = (qi * stride_q + head * stride_qh) // 2  # i32 units
        qv = []
        for j in range_constexpr(2):
            d0 = (256 * half + 128 * j + 32 * g) // 2
            parts = [Vec(q_t.load(q_row + d0 + 4 * k, vec_size=4)) for k in range(4)]
            qv.append([fx.Int32(p[i]) for p in parts for i in range(4)])  # 32 bf16
        qr = Vec(q_t.load(q_row + (KV_LORA_RANK + 32 * half + 8 * g) // 2, vec_size=4))
        _order()

        nt = _readfirstlane((r_len + (_MH_TILE - 1)) // _MH_TILE)

        # ---- quantize Q (waits for the Q loads only; the DMA was issued after them)
        q8, q8l, qs, qsl = [], [], [], []
        for j in range_constexpr(2):
            a = _amax_bf16(qv[j])
            a = _xmax(a)  # amax over the head's 128 dims of block 2 half + j
            s, inv = _e8m0_for(a)
            hi, lo = _quant_bf16(qv[j], inv)
            q8.append(Vec.from_elements(hi, fx.Int32))
            q8l.append(Vec.from_elements(lo, fx.Int32))
            qs.append(s)
            ls = fx.Int32(s) - 4
            qsl.append((ls < 0).select(fx.Int32(0), ls))

        def lane_offsets(lane):
            """Per-lane LDS offsets (stage-relative). Swizzles only touch address
            bits 4-7, so a subtile / chunk / block offset is an XOR on those bits
            plus an add above them (an immediate)."""
            l16 = lane % 16
            g = lane // 16
            hb = half * (_MH_TILE * 256)
            qk0 = (
                hb + l16 * 256 + 16 * ((2 * g) ^ l16)
            )  # + 4096 s, ^ 128 ((s & 1) ^ j) ^ 16 e
            rp0 = (
                _MH_ROPE_OFF + l16 * 128 + 16 * ((4 * half + g) ^ _rsw(l16))
            )  # + 2048 s
            sc0 = _MH_SC_OFF + l16 * 16 + 8 * half  # + 256 s
            ps0 = _MH_SC_OFF + 64 * g + 8 * half  # + 16 (16 s2 + r)
            i8 = l16 // 2
            tok0 = 16 * (i8 // 4) + 4 * g + i8 % 4
            pv0 = hb + tok0 * 256 + 8 * (l16 % 2) + 16 * _fsw(tok0)  # ^ 16 d
            x_wr = (
                base + _MH_XCH_OFF + (grp * 2 + half) * (_MH_NS16 * 1024) + lane * 16
            )  # + 1024 s
            x_rd = (
                base
                + _MH_XCH_OFF
                + (grp * 2 + 1 - half) * (_MH_NS16 * 1024)
                + lane * 16
            )
            return g, qk0, rp0, sc0, ps0, pv0, x_wr, x_rd

        # ---- tile phases (closures over the per-lane offsets)
        def qk_phase(sb, offs):
            """Partial S^T over this wave's columns of the tile in stage sb (lane
            (h = l16, g) holds tokens 16s + 4g + r), written to this wave's
            exchange slot."""
            qk0, rp0, sc0, x_wr = offs[1], offs[2], offs[3], offs[6]
            qkb = sb + qk0

            def qk_read(s, j):
                if const_expr(j == 2):
                    return _ld(sc, T.i32, 4, sb + rp0 + 2048 * s, 16)
                x = 128 * ((s & 1) ^ j)
                return _cat(
                    _ld(sc, T.i32, 4, (qkb ^ x) + 4096 * s, 16),
                    _ld(sc, T.i32, 4, (qkb ^ (x + 16)) + 4096 * s, 16),
                    4,
                )

            # Block j = 0 for every subtile, then j = 1, then rope; operands are
            # read one step ahead.
            steps = [(s_, j_) for j_ in range(3) for s_ in range(_MH_NS16)]
            scp = [_ld(sc, T.i32, 2, sb + sc0 + 256 * s_, 8) for s_ in range(_MH_NS16)]
            ops = [qk_read(*steps[0])]
            qh = q8[0]
            acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range(_MH_NS16)]
            for k in range_constexpr(len(steps)):
                s, j = steps[k]
                qcur = qh
                if const_expr(k + 1 < len(steps)):
                    ops.append(qk_read(*steps[k + 1]))
                    if const_expr(steps[k + 1][0] == 0 and steps[k + 1][1] < 2):
                        qh = q8[steps[k + 1][1]]
                _order()
                if const_expr(j < 2):
                    se = (fx.Int32(scp[s][j]) >> 23) & 0xFF
                    acc[s] = _mfma_fp8(ops[k], qcur, acc[s], se, qs[j])
                    acc[s] = _mfma_fp8(ops[k], q8l[j], acc[s], se, qsl[j])
                else:
                    acc[s] = _mfma_bf16(ops[k], qr, acc[s])
                    _st(sc, acc[s], x_wr + 1024 * s, 16)
                ops[k] = None

        def read_s(offs):
            """Full S (own + partner partial) * log2(e) * softmax_scale."""
            x_wr, x_rd = offs[6], offs[7]
            sl = []
            for s in range_constexpr(_MH_NS16):
                own = _ld(sc, T.f32, 4, x_wr + 1024 * s, 16)
                oth = _ld(sc, T.f32, 4, x_rd + 1024 * s, 16)
                for r in range_constexpr(4):
                    sl.append((fx.Float32(own[r]) + fx.Float32(oth[r])) * c_log2)
            return sl

        def softmax(sl, tb, m_old, l_old, o, gl):
            """Online softmax step over the tile at tb.

            -> (sl masked, m_new, l_old * alpha, m_cl, o rescaled).
            """

            # Only the last tile can hold positions >= r_len (wave-uniform branch).
            @flyc.jit
            def _mask(sl, tb=tb):
                if tb + _MH_TILE > r_len:
                    sl = [
                        (tb + 16 * (k // 4) + 4 * gl + k % 4 < r_len).select(
                            sl[k], fx.Float32(_MH_NEG)
                        )
                        for k in range(len(sl))
                    ]
                return sl

            sl = _mask(sl)
            tmax = fx.Float32(_MH_NEG)
            for x in sl:
                tmax = _fmax(tmax, x)
            tmax = _xmax(tmax)
            # Lazy rescale: move m (and rescale O) only when the tile max exceeds
            # it by more than _MH_TAU.
            m_cand = _fmax(m_old, tmax)
            upd = m_cand > m_old + _MH_TAU
            m_new = upd.select(m_cand, m_old)
            alpha = (m_old - m_new).exp2(fastmath="afn")
            # All masked so far: exp2(-1e30 + 1e20) = 0.
            m_cl = _fmax(m_new - _MH_P_SHIFT, fx.Float32(-1.0e20))
            any_upd = fx.Int64(rocdl.ballot(T.i64, _raw(upd.ir_value()))) != fx.Int64(0)

            @flyc.jit
            def _rescale(o, alpha=alpha, any_upd=any_upd):
                if any_upd:
                    o = [
                        Vec.from_elements(
                            [fx.Float32(Vec(o[i])[k]) * alpha for k in range(4)],
                            fx.Float32,
                        )
                        for i in range(len(o))
                    ]
                return o

            return sl, m_new, l_old * alpha, m_cl, _rescale(o)

        def pv_phase(sb, offs, sl, m_cl, l_new, o):
            """O^T[d, h] += V^T[d, tok] P'^T[tok, h], P' = P * (token's scale of
            the column block, capped at 2^7). Groups of 4 output blocks; group
            q + 1's tr8 (and scale) reads are issued before group q's MFMAs, and
            group q - 1's accumulators are pinned there so the MFMAs stay in
            place. -> (o, l_new)."""
            pvb = sb + offs[5]
            ps0 = offs[4]

            def pv_reads(q):
                rd = [_tr8_scoped(sc, pvb ^ (16 * (4 * q + i))) for i in range(4)]
                psc = None
                if const_expr(q % 2 == 0):
                    # V scale of token 16 (i // 4) + 4g + i % 4, column block
                    # 2 half + q // 2.
                    psc = []
                    for i in range_constexpr(8):
                        tok = 16 * (i // 4) + i % 4
                        sv = _ld(sc, T.i32, 1, sb + ps0 + 4 * (q // 2) + 16 * tok, 4)
                        psc.append(_imin(fx.Int32(sv), fx.Int32(_SCALE_CAP_BITS)))
                return rd, psc

            nxt = pv_reads(0)
            pl = None
            pp = None
            for q in range_constexpr(4):
                cur = nxt
                if const_expr(q + 1 < 4):
                    nxt = pv_reads(q + 1)
                if const_expr(q > 0):
                    o[4 * (q - 1) : 4 * q] = _order(o[4 * (q - 1) : 4 * q])
                else:
                    _order()
                    pl = [(sl[i] - m_cl).exp2(fastmath="afn") for i in range(8)]
                    for x in pl:
                        l_new = l_new + x
                if const_expr(q % 2 == 0):
                    vals = [pl[i] * _f32(cur[1][i]) for i in range(8)]
                    pp = (
                        Vec.from_elements(vals, fx.Float32)
                        .to(fx.Float16)
                        .bitcast(fx.Int32)
                    )
                for i in range_constexpr(4):
                    d = 4 * q + i
                    o[d] = _mfma_f16(_cvt_v8(cur[0][i]), pp, o[d])
            o[12:16] = _order(o[12:16])
            return o, l_new

        offs = lane_offsets(lane)
        init = [_raw(fx.Float32(_MH_NEG).ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(16)]
        init += [_raw(idxv[1].ir_value())]
        for tile, st in range(0, nt, 1, init=init):
            m_old = fx.Float32(st[0])
            l_old = fx.Float32(st[1])
            o = [Vec(st[2 + i]) for i in range(16)]
            idxv_c = fx.Int32(st[18])
            tb = fx.Int32(tile) * _MH_TILE
            stage = fx.Int32(tile) & 1
            sb = base + stage * _MH_STAGE

            # Tile `tile` has landed (own DMAs) and is visible (barrier);
            # everyone is done with tile - 1's stage, which takes tile + 1.
            rocdl.s_waitcnt(vmcnt=0)
            _barrier()
            _guarded(
                fx.Int32(tile) + 1 < nt,
                lambda _v=idxv_c, _tb=tb + _MH_TILE, _s=1 - stage: dma(
                    _v, _tb, _s, lane
                ),
            )
            idxv_n = fx.Int32(idx_t[r_beg + tb + 2 * _MH_TILE + lane])
            qk_phase(sb, offs)
            rocdl.s_waitcnt(lgkmcnt=0)
            _barrier()
            sl = read_s(offs)
            sl, m_new, l_new, m_cl, o = softmax(sl, tb, m_old, l_old, o, offs[0])
            o, l_new = pv_phase(sb, offs, sl, m_cl, l_new, o)
            fin = yield [m_new, l_new] + o + [idxv_n]

        # ---- epilogue
        rocdl.s_waitcnt(vmcnt=0)
        l_tot = _xsum(fx.Float32(fin[1]))
        has_l = l_tot > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_tot, fx.Float32(0.0))
        # O goes through LDS so each store instruction writes whole 512-B row segments.
        _barrier()  # everyone is done with the KV stages and the exchange buffer
        ob = base + w * (16 * _MH_OP)
        for d in range_constexpr(16):
            ov = Vec(fin[2 + d])
            bf = Vec.from_elements(
                [fx.Float32(ov[k]) * inv_l for k in range(4)], fx.Float32
            ).to(fx.BFloat16)
            _st(sc, bf.bitcast(fx.Int32), ob + l16 * _MH_OP + 32 * d + 8 * g, 8)
        rocdl.s_waitcnt(lgkmcnt=0)
        for k in range_constexpr(8):
            e = lane + 64 * k
            row = e // 32
            ch = e % 32
            v = _ld(sc, T.i32, 4, ob + row * _MH_OP + 16 * ch, 16)
            o_t.store(
                (qi * stride_o + (gi * 16 + row) * stride_oh + 256 * half + 8 * ch)
                // 2,
                v,
                vec_size=4,
            )
        if const_expr(return_lse):
            # Natural-log LSE of softmax_scale * q.k over the unmasked keys;
            # -inf when there is none.
            m_fin = fx.Float32(fin[0])
            lse2 = (
                m_fin
                + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0))))
                - _MH_P_SHIFT
            )
            lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
            lse_bits = lse.bitcast(fx.Int32)

            def _st_lse(_v=lse_bits, _off=qi * H + head):
                lse_t.store(_off, _v)

            _guarded((half + g) == 0, _st_lse)

    @flyc.jit
    def launch_sparse_mla_fp8ds_mh(
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
        sparse_mla_fp8ds_mh_kernel(
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
        ).launch(grid=(fx.Int64(n_q * nsplit), 1, 1), block=(NTH, 1, 1), stream=stream)

    launch_sparse_mla_fp8ds_mh.compile_hints = {
        "waves_per_eu": waves_per_eu,
        "fast_fp_math": False,
    }
    return launch_sparse_mla_fp8ds_mh
