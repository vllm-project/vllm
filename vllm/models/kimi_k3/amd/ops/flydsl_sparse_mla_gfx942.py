# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

"""FlyDSL sparse MLA forward over fp8_ds_mla records, gfx942.

Same contract as `flydsl_sparse_mla_gfx950.py` (kv_lora_rank 512, rope 64,
bf16 q, <= 16 heads, CSR index stream); the host wrapper and its support
predicate live in `flydsl_sparse_mla.py`. gfx942 has no scaled MFMA, no
16x16x32 bf16/f16 MFMA, no ds_read_tr8, no permlane swaps and 64 KiB of LDS,
so the tile is laid out differently.

Record (656 B): 512 e4m3fnuz latent | 4 f32 per-128 pow2 scales | 64 bf16 rope.

One 4-wave workgroup per query row (rows issued in reverse order, so the long
causal tail starts first), 64-token tiles over the row's CSR slice
`indices[indptr[q] : indptr[q + 1]]`. Lane l = 16g + t of wave w:

  * Loads: lane (t, g) gathers token 16w + t of the tile straight into VGPRs,
    one tile ahead: latent bytes 64r + 16g (r < 8, 8 x dwordx4), rope bytes
    528 + 16g and 592 + 16g (2 x dwordx4) and the scale of block g. Rows are
    addressed in 64 bits.
  * QK: A = K (m = token, from those VGPRs), B = Q (n = head), so no LDS on
    this side. Q is quantized once per row with a per-head pow2 scale to an
    fp8 hi plus an fp8 lo residual; hi and lo MFMA 16x16x32 fp8 accumulate
    into one accumulator per 128-dim block, and
    S = (rope + sum_b scale_b * acc_b) / q_scale, with the rope part a bf16
    MFMA on q_rope * q_scale (exact: the scale is a power of two).
  * Softmax: the per-wave tile max is exchanged through LDS;
    P = exp2(s - m + 8).
  * PV: wave w owns output dims [128w, 128w + 128), which is exactly one KV
    scale block, so the V scale folds into P (P' = P * scale_v, fp16) and V
    stays raw fp8 in LDS. Lane (t, g) reads 8 dims (128w + 8t ..) of 4 tokens
    with ds_read_b64; fp8 -> f32 -> fp16 packing pairs tokens, which is the
    transpose the f16 MFMA 16x16x16 needs. O^T accumulates as dims
    128w + 32g + [0, 32) of head t.
  * Software pipeline, two barriers per tile. Phase 1: PV of tile i - 1 (its
    V and P' in LDS) and S / max of tile i from QK partials already in
    registers; the max exchange ends it. Phase 2: P' and V of tile i to LDS,
    the index read of tile i + 3, the gather of tile i + 2 into the registers
    tile i's latent just left, QK of tile i + 1 (gathered a tile earlier).
    The compiler sinks those QK MFMAs into the next phase 1, beside PV, which
    measured faster than pinning them in phase 2. The tile loop is unrolled
    by two so every gather lands in the registers its consumer reads: a
    loop-carried copy would wait for the gather.
  * Tiles: nt = ceil(len / 64), or with has_invalid ceil(last_valid / 64),
    where a scan of the row finds the last entry >= 0. Positions >= len and,
    with has_invalid, entries < 0 are masked: they load the row's last valid
    slot (so V is finite and P' = 0 removes it) and their scale is replaced
    by -1, which marks them in S.

LDS map (42752 B, one workgroup per CU):

    [0, 32768)      V      [64 tok][512 B], 16-B chunk c of token p at c ^ (p & 7)
    [32768, 41472)  P'     [4 blocks][16 heads][64 tok] fp16, head pitch 136 B
    [41472, 42496)  scales [wave][block][16 tok] f32 (wave-private)
    [42496, 42752)  max    [16 heads][4 waves] f32 (also the final l sum)

Preconditions the host cannot check:
  * Every per-128 KV scale is a power of two, as vLLM's fp8_ds_mla writer
    (concat_and_cache_ds_mla) stores it, and the latent bytes are fnuz e4m3,
    which is what that writer emits on gfx942.
  * Every per-128 KV scale is <= 2^7: P' = P * scale has to fit fp16 and
    P <= 2^8. vLLM's gfx942 writer uses scale = pow2ceil(max(amax, 1e-4) / 224),
    so this holds iff |latent| <= 28672. A larger scale is clamped to 2^7 in
    P', which keeps the output finite but makes it inexact.
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
_TILE = 64
_WAVES = 4
_BLOCK_THREADS = _WAVES * 64
_LN2 = 0.6931471805599453
# fp16 P = exp2(s - m + 8); l carries the same 2^8.
_P_SHIFT = 8.0
_SCALE_CAP = 128.0
# f32 bits of -1.0: the scale of a masked token.
_MASKED_BITS = -1082130432

_V_OFF = 0
_V_BYTES = _TILE * _L
_PP_PITCH = 2 * _TILE + 8
_PP_BLOCK = MAX_HEADS * _PP_PITCH
_PP_OFF = _V_OFF + _V_BYTES
_PP_BYTES = 4 * _PP_BLOCK
_KS_OFF = _PP_OFF + _PP_BYTES
_KS_BYTES = _WAVES * 4 * 16 * 4
_MX_OFF = _KS_OFF + _KS_BYTES
_MX_BYTES = MAX_HEADS * _WAVES * 4
_LDS_BYTES = _MX_OFF + _MX_BYTES


def _lds_ptr(elem, addr, align):
    return fx.inttoptr(
        fx.PointerType.get(elem, fx.AddressSpace.Shared, align), fx.Int32(addr)
    )


def _lds_load(elem, n, addr, align):
    return Vec(fx.ptr_load(_lds_ptr(elem, addr, align), result_type=T.vec(n, elem)))


def _lds_store(vec, elem, addr, align):
    fx.ptr_store(Vec(vec).ir_value(), _lds_ptr(elem, addr, align))


def _global_i32(addr, words):
    """`words` i32 (1 or 4) at the 64-bit global address `addr`: a cache
    past 4 GiB is out of a buffer descriptor's 32-bit offset reach."""
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4 * words), fx.Int64(addr)
    )
    if words == 1:
        return fx.Int32(fx.generic_load(ptr, dtype=fx.Int32).ir_value())
    return Vec(fx.generic_load(ptr, dtype=fx.Vector[fx.Int32, words]).ir_value())


def _cat(a, b, n):
    return Vec(a).shuffle(Vec(b), list(range(2 * n)))


def _pair(v, k):
    """i32 elements 2k, 2k + 1 of `v` as a 2-vector."""
    return Vec(v).shuffle(Vec(v), [2 * k, 2 * k + 1])


def _i64(v2):
    return fx.Int64(Vec(v2).bitcast(fx.Int64)[0])


def _mfma_fp8(a, b, c):
    """16x16x32 e4m3fnuz MFMA; a, b are 8 fp8 each (i64)."""
    return Vec(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            T.vec(4, T.f32),
            [
                _raw(fx.Int64(a).ir_value()),
                _raw(fx.Int64(b).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _mfma_f16(a, b, c):
    return Vec(
        rocdl.mfma_f32_16x16x16f16(
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
        rocdl.mfma_f32_16x16x16bf16_1k(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).bitcast(fx.Int16).ir_value()),
                _raw(Vec(b).bitcast(fx.Int16).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _pkrtz(a, b):
    """Two f32 -> one i32 of two fp16 (round toward zero)."""
    v = rocdl.cvt_pkrtz(
        T.vec(2, T.f16),
        _raw(fx.Float32(a).ir_value()),
        _raw(fx.Float32(b).ir_value()),
    )
    return fx.Int32(Vec(v).bitcast(fx.Int32)[0])


def _pack_fp8x4(vals):
    """4 f32 -> one i32 of e4m3fnuz (byte i = vals[i])."""
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


def _fmax(a, b):
    return fx.maxnumf(a, b, fastmath="nnan")


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def _readfirstlane(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _xor16(x):
    """Lane l ^ 16 (ds_swizzle bit mode: and 0x1f, xor 0x10)."""
    v = _raw(fx.Float32(x).bitcast(fx.Int32).ir_value())
    r = rocdl.ds_swizzle(T.i32, v, _raw(fx.Int32(0x401F).ir_value()))
    return fx.Int32(r).bitcast(fx.Float32)


def _xor32(x, bp_idx):
    v = _raw(fx.Float32(x).bitcast(fx.Int32).ir_value())
    r = rocdl.ds_bpermute(T.i32, _raw(fx.Int32(bp_idx).ir_value()), v)
    return fx.Int32(r).bitcast(fx.Float32)


def _xmax(x, bp_idx):
    """Max over the 4 lane groups (xor 16, 32)."""
    x = _fmax(x, _xor16(x))
    return _fmax(x, _xor32(x, bp_idx))


def _xsum(x, bp_idx):
    x = x + _xor16(x)
    return x + _xor32(x, bp_idx)


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


def _fenced_barrier():
    """s_barrier fenced for the machine scheduler, which otherwise moves LDS
    and global accesses and their waits across it. Pure ops (MFMAs) still
    move: they hang off no chain."""
    rocdl.sched_barrier(0)
    gpu.barrier()
    rocdl.sched_barrier(0)


def _guarded(pred, fn):
    """Run fn() under a runtime predicate (an scf.if around its body)."""

    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


def _q_prologue(q_t, q_row, g, h_ok):
    """This lane's q latent values and their abs max.

    Lane (h, g) holds, for latent MFMA (r, s), q dims 64r + 16g + 8s + [0, 8)
    (the K bytes of the same lane) as f32, zero for heads >= H. The caller
    reduces the max over the head's lanes, picks the pow2 scale qs
    (amax * qs in [64, 128)) and quantizes to fp8 hi plus an fp8 lo residual.

    Returns (xs[r] = 16 values, s-major; per-lane abs max).
    """
    xs = []
    for r in range(8):
        vals = []
        for k in range(2):
            v = Vec(q_t.load(q_row + 32 * r + 8 * g + 4 * k, vec_size=4))
            vals += [fx.Float32(x) for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)]
        xs.append([h_ok.select(x, fx.Float32(0.0)) for x in vals])
    am = fx.Float32(0.0)
    for row in xs:
        for x in row:
            am = _fmax(am, _fmax(x, fx.Float32(0.0) - x))
    return xs, am


def build_sparse_mla_fp8ds_module(num_heads, has_invalid, return_lse):
    """JIT launcher for one (heads, has_invalid, return_lse) configuration."""
    H = num_heads
    if not 1 <= H <= MAX_HEADS:
        raise ValueError(f"num_heads must be in [1, {MAX_HEADS}], got {H}")
    name = f"sparse_mla_fp8ds_gfx942_h{H}_i{int(has_invalid)}_l{int(return_lse)}_flydsl"

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, _LDS_BYTES, 16]

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
        bp_idx = (lane ^ 32) * 4

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)  # i64: the cache may span more than 4 GiB
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        # V and P' start zeroed: the first iteration's PV (of "tile -1")
        # then adds exact zeros.
        z4 = Vec.filled(4, 0.0, fx.Float32)
        for k in range_constexpr(_V_BYTES // 16 // _BLOCK_THREADS):
            _lds_store(z4, T.f32, base + _V_OFF + (k * _BLOCK_THREADS + tid) * 16, 16)
        for k in range_constexpr(
            (_PP_BYTES // 16 + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        ):
            c = _imin(tid + k * _BLOCK_THREADS, fx.Int32(_PP_BYTES // 16 - 1))
            _lds_store(z4, T.f32, base + _PP_OFF + c * 16, 16)

        # ---- this row's CSR slice (wave-uniform)
        r_beg = _readfirstlane(ip_t[qi])
        r_end = _readfirstlane(ip_t[qi + 1])
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)

        # ---- Q prologue
        h_ok = t < H
        h_safe = h_ok.select(t, fx.Int32(0))
        q_row = (qi * stride_q + h_safe * stride_qh) // 2  # i32 units (bf16 pairs)
        xs, am = _q_prologue(q_t, q_row, g, h_ok)
        am = _xmax(am, bp_idx)
        e = (am.bitcast(fx.Int32) >> 23) & 0xFF
        e = _imin(_imax(e, fx.Int32(67)), fx.Int32(254))
        qs = ((260 - e) << 23).bitcast(fx.Float32)  # amax * qs in [64, 128)
        cq = ((e - 6) << 23).bitcast(fx.Float32) * c_log2  # c_log2 / qs
        qh, ql = [], []
        for r in range_constexpr(8):
            x = [v * qs for v in xs[r]]
            hw = [_pack_fp8x4(x[4 * i : 4 * i + 4]) for i in range(4)]
            lw = []
            for i in range_constexpr(4):
                dec = _unpack_fp8x4(hw[i])
                lw.append(_pack_fp8x4([x[4 * i + j] - dec[j] for j in range(4)]))
            qh.append(
                [
                    _i64(Vec.from_elements(hw[2 * s : 2 * s + 2], fx.Int32))
                    for s in range(2)
                ]
            )
            ql.append(
                [
                    _i64(Vec.from_elements(lw[2 * s : 2 * s + 2], fx.Int32))
                    for s in range(2)
                ]
            )
        qr = []
        for k in range_constexpr(2):
            v = Vec(q_t.load(q_row + _L // 2 + 16 * k + 4 * g, vec_size=4))
            f = [
                h_ok.select(fx.Float32(x) * qs, fx.Float32(0.0))
                for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)
            ]
            bf = Vec.from_elements(f, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            qr += [_pair(bf, 0), _pair(bf, 1)]

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
        lastp = _readfirstlane(lastp)
        nt = _readfirstlane((lastp + (_TILE - 1)) // _TILE)
        # Slot that masked tokens load: the row's last valid entry.
        fb = fx.Int32(idx_t[r_beg + _imax(lastp - 1, fx.Int32(0))])
        fb = _readfirstlane(_imax((lastp > 0).select(fb, fx.Int32(0)), fx.Int32(0)))

        # ---- per-lane LDS addresses
        p_tok = w * 16 + t  # this lane's load token within the tile
        x7 = t & 7
        v_wr0 = base + _V_OFF + p_tok * _L + ((g ^ x7) & 3) * 16
        v_wr = [v_wr0 + (x7 >> 2) * 64, v_wr0 + (1 - (x7 >> 2)) * 64]  # even / odd r
        v_rd = []
        for i in range_constexpr(4):
            sw = (4 * g + i) & 7
            v_rd.append(
                base
                + _V_OFF
                + (4 * g + i) * _L
                + ((8 * w + t // 2) ^ sw) * 16
                + (t % 2) * 8
            )
        pp_wr = base + _PP_OFF + t * _PP_PITCH + (16 * w + 4 * g) * 2
        pp_rd = base + _PP_OFF + w * _PP_BLOCK + t * _PP_PITCH + 8 * g
        ks_wr = base + _KS_OFF + ((w * 4 + g) * 16 + t) * 4
        ks_rd = base + _KS_OFF + (w * 64 + 4 * g) * 4
        mx_wr = base + _MX_OFF + (t * 4 + w) * 4
        mx_rd = base + _MX_OFF + t * 16

        def ld_idx(tb):
            """Raw index entry of this lane's load token in tile tb (0 past the
            buffer end); `valid` applies the row length."""
            return fx.Int32(idx_t[r_beg + tb * _TILE + p_tok])

        def valid(tb, ix):
            # Applied where the index is consumed, not at its load: a select
            # next to the load would make the wave wait for it right there.
            return (tb * _TILE + p_tok < r_len) & (fx.Int32(ix) >= 0)

        def ld_kv(tb, ix):
            """Latent (8), rope (2) words and block-g scale of tile tb's slot ix
            (fb if masked)."""
            ixs = valid(tb, ix).select(fx.Int32(ix), fb)
            row = kv_base + fx.Int64(ixs) * RECORD_BYTES
            rowg = row + fx.Int64(16 * g)
            out = [_global_i32(rowg + fx.Int64(64 * r), 4) for r in range(8)]
            out += [_global_i32(rowg + fx.Int64(_L + 16 + 64 * k), 4) for k in range(2)]
            sc = _global_i32(row + fx.Int64(4 * g) + fx.Int64(_L), 1)
            return out, sc

        def pv(o):
            """O^T[d, h] += V^T[d, tok] P'^T[tok, h] for the tile in LDS."""
            for kb in range_constexpr(4):
                pp = _lds_load(T.i32, 2, pp_rd + 32 * kb, 8)
                f = []
                for i in range_constexpr(4):
                    vr = _lds_load(T.i32, 2, v_rd[i] + 16 * _L * kb, 8)
                    f.append(_unpack_fp8x4(vr[0]) + _unpack_fp8x4(vr[1]))
                for jm in range_constexpr(8):
                    a = Vec.from_elements(
                        [_pkrtz(f[0][jm], f[1][jm]), _pkrtz(f[2][jm], f[3][jm])],
                        fx.Int32,
                    )
                    o[jm] = _mfma_f16(a, pp, o[jm])
            return o

        neg_inf = fx.Float32(-1.0e30)

        def qk(kv):
            """S^T partials of one tile: lane (h = t, g) holds tokens
            16w + 4g + i; one accumulator per 128-dim block, plus rope."""
            accb = [z4, z4, z4, z4]
            for rr in range_constexpr(2):
                for s in range_constexpr(2):
                    for b in range_constexpr(4):
                        r = 2 * b + rr
                        ka = _i64(_pair(kv[r], s))
                        accb[b] = _mfma_fp8(ka, qh[r][s], accb[b])
                        accb[b] = _mfma_fp8(ka, ql[r][s], accb[b])
            accr = z4
            for mm in range_constexpr(4):
                accr = _mfma_bf16(_pair(kv[8 + mm // 2], mm % 2), qr[mm], accr)
            return accb + [accr]

        def ks_store(tl, ix, sc):
            sg = valid(tl, ix).select(sc, fx.Int32(_MASKED_BITS))
            _lds_store(Vec.from_elements([sg], fx.Int32), T.i32, ks_wr, 4)

        def step(tl, m_old, l_old, o, acc, kv, kv1, sc1, ix1, ix2):
            """Tile tl. Phase 1: PV of tile tl - 1 and S / max of tile tl
            (QK partials acc, scales already in LDS). Phase 2: P' and V of tile
            tl to LDS, gather of tile tl + 2 (slot ix2) into the registers kv
            frees, QK of tile tl + 1 (kv1, sc1, ix1)."""
            # ---- phase 1
            o = pv(o)
            ksv = [_lds_load(T.f32, 4, ks_rd + 64 * b, 16) for b in range_constexpr(4)]
            sl = []
            tmax = neg_inf
            for i in range_constexpr(4):
                x = fx.Float32(acc[4][i])
                for b in range_constexpr(4):
                    x = x + fx.Float32(ksv[b][i]) * fx.Float32(acc[b][i])
                x = x * cq
                x = (fx.Float32(ksv[0][i]) >= fx.Float32(0.0)).select(x, neg_inf)
                sl.append(x)
                tmax = _fmax(tmax, x)
            tmax = _xmax(tmax, bp_idx)
            _lds_store(Vec.from_elements([tmax], fx.Float32), T.f32, mx_wr, 4)
            rocdl.s_waitcnt(lgkmcnt=0)
            _fenced_barrier()

            # ---- phase 2
            mx = _lds_load(T.f32, 4, mx_rd, 16)
            for r in range_constexpr(8):
                _lds_store(kv[r], T.i32, v_wr[r % 2] + 128 * (r // 2), 16)
            ix3 = ld_idx(tl + 3)
            # vmcnt retires in order: the index read stays ahead of the gather,
            # so waiting on it later does not also wait on this gather.
            rocdl.sched_barrier(0)
            kv2, sc2 = ld_kv(tl + 2, ix2)
            acc1 = qk(kv1)
            ks_store(tl + 1, ix1, sc1)

            m_new = m_old
            for j in range_constexpr(4):
                m_new = _fmax(m_new, fx.Float32(mx[j]))
            alpha = (m_old - m_new).exp2(fastmath="afn")
            # All masked so far: exp2(-1e30 + 1e20) = 0.
            m_cl = _fmax(m_new - _P_SHIFT, fx.Float32(-1.0e20))
            pl = [(sl[i] - m_cl).exp2(fastmath="afn") for i in range(4)]
            l_new = l_old * alpha + ((pl[0] + pl[1]) + (pl[2] + pl[3]))

            for b in range_constexpr(4):
                ks = [
                    fx.minnumf(fx.Float32(ksv[b][i]), fx.Float32(_SCALE_CAP))
                    for i in range(4)
                ]
                pw = Vec.from_elements(
                    [
                        _pkrtz(pl[0] * ks[0], pl[1] * ks[1]),
                        _pkrtz(pl[2] * ks[2], pl[3] * ks[3]),
                    ],
                    fx.Int32,
                )
                _lds_store(pw, T.i32, pp_wr + b * _PP_BLOCK, 8)
            o = [
                Vec.from_elements(
                    [fx.Float32(Vec(o[jm])[i]) * alpha for i in range(4)], fx.Float32
                )
                for jm in range(8)
            ]
            rocdl.s_waitcnt(lgkmcnt=0)
            _fenced_barrier()
            return m_new, l_new, o, acc1, kv1, kv2, sc2, ix2, ix3

        # Prologue: tiles 0 and 1 gathered, QK of tile 0, indices of tile 2.
        ix0 = ld_idx(fx.Int32(0))
        ix1 = ld_idx(fx.Int32(1))
        ix2 = ld_idx(fx.Int32(2))
        rocdl.sched_barrier(0)
        kv0, sc0 = ld_kv(fx.Int32(0), ix0)
        kv1, sc1 = ld_kv(fx.Int32(1), ix1)
        acc0 = qk(kv0)
        ks_store(fx.Int32(0), ix0, sc0)
        gpu.barrier()

        # Two tiles per iteration, so each gather lands in the registers its
        # consumer reads (no back-edge copy, which would wait on the loads). An
        # odd nt runs one extra tile, fully masked: it adds exact zeros.
        npair = (nt + 1) // 2
        init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(z4.ir_value()) for _ in range(8)]
        init += [_raw(Vec(x).ir_value()) for x in acc0]
        init += [_raw(Vec(x).ir_value()) for x in kv0[:8]]
        init += [_raw(Vec(x).ir_value()) for x in kv1]
        init += [_raw(fx.Int32(x).ir_value()) for x in (sc1, ix1, ix2)]
        for pair, st in range(0, npair, 1, init=init):
            o = [Vec(st[2 + i]) for i in range(8)]
            acc = [Vec(st[10 + i]) for i in range(5)]
            kv_a = [Vec(st[15 + i]) for i in range(8)]
            kv_b = [Vec(st[23 + i]) for i in range(10)]
            sc_b, ix_b, ix_a2 = (fx.Int32(st[33 + i]) for i in range(3))
            ta = fx.Int32(pair) * 2
            m, l_run, o, acc, kv_b, kv_a2, sc_a2, ix_a2, ix_b2 = step(
                ta,
                fx.Float32(st[0]),
                fx.Float32(st[1]),
                o,
                acc,
                kv_a,
                kv_b,
                sc_b,
                ix_b,
                ix_a2,
            )
            m, l_run, o, acc, kv_a2, kv_b2, sc_b2, ix_b2, ix_a3 = step(
                ta + 1, m, l_run, o, acc, kv_b, kv_a2, sc_a2, ix_a2, ix_b2
            )
            fin = yield (
                [m, l_run] + o + acc + kv_a2[:8] + kv_b2 + [sc_b2, ix_b2, ix_a3]
            )

        # ---- last tile's PV, then l over the 4 waves, normalize, store bf16
        o = pv([Vec(fin[2 + i]) for i in range(8)])
        rocdl.s_waitcnt(vmcnt=0)
        l_w = _xsum(fx.Float32(fin[1]), bp_idx)
        _lds_store(Vec.from_elements([l_w], fx.Float32), T.f32, mx_wr, 4)
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        lv = _lds_load(T.f32, 4, mx_rd, 16)
        l_tot = (fx.Float32(lv[0]) + fx.Float32(lv[1])) + (
            fx.Float32(lv[2]) + fx.Float32(lv[3])
        )
        has_l = l_tot > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_tot, fx.Float32(0.0))
        o_base = (qi * stride_o + h_safe * stride_oh + w * 128 + 32 * g) // 2
        for i in range_constexpr(4):
            bf = Vec.from_elements(
                [fx.Float32(Vec(o[jm])[i]) * inv_l for jm in range(8)], fx.Float32
            ).to(fx.BFloat16)
            val = bf.bitcast(fx.Int32)

            def _st(_v=val, _off=o_base + 4 * i):
                o_t.store(_off, _v, vec_size=4)

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

    launch_sparse_mla_fp8ds.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_sparse_mla_fp8ds
