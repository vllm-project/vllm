# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3 prefill MoE stage 2 (a4w4, down projection), token-major output.

    y[tok, slot, :] = h[row, :] @ W2[e]^T            row = sorted row of (tok, slot)

``out_dtype`` "bf16": out[tok*topk + slot, :] = bf16(y * sorted_weights[row]), summed
by the bf16 reduce; "fp8": MXFP8 rows with the routing weights left to the reduce.

One CTA owns an m-tile of 128 sorted rows and sweeps ``NT`` n-tiles of 256 columns:
K = I = 768 is only 3 K-steps, so the A tile stays in AGPRs for the whole sweep and
the (n-tile, k-step) sequence is one depth-3 pipeline. The epilogue runs in the MFMA
shadow and writes token-major 128-B lines; padded rows fall outside the buffer
resource. Consecutive m-tiles (same expert) go to the same XCD.

Layouts (bytes):
  A         [num_m_blocks*BM, I/2]        sorted rows, fp4 (gemm1 OUT_Q)
  A_scale   [num_m_blocks*BM, I/32]       sorted rows, e8m0-shuffled (gemm1 OUT_sc)
  W2        [E, H, I/2]                   aiter shuffle_weight(layout=(16,16))
  W2_sc     [E*H, I/32]                   aiter e8m0_shuffle
  sorted_w  [num_m_blocks*BM]             f32 routing weight per sorted row (bf16 mode)
  OUT       [n_tokens*topk, H]            bf16, or fp8 e4m3 (OCP)
  OUT_sc    [n_tokens*topk, H/32]         e8m0 (fp8 mode; any buffer in bf16 mode)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from .gemm1_prefill import (
    _N_WAVES,
    G2SLoaderAsm,
    Mfma16x16x128Fp4,
    S2RLoaderFp4,
    _as_f32,
    _asm_void,
    _bits,
    _Buf,
    _divmod_nonneg,
    _g2s_thunks,
    _lds_ptr_t,
    _min,
    _permlane16_swap,
    _s2r_thunks,
    _swizzled_col,
    _uniform_i32,
    wait_barrier,
)

_NUM_XCDS = 8


class _MfmaAgprAB(Mfma16x16x128Fp4):
    """Same MFMA, but both matrix fragments come from AGPRs (A pinned there for the
    whole n-sweep, B loaded there by ds_read) and the accumulator lives in VGPRs, so the
    epilogue reads it without v_accvgpr_read copies."""

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        a_op, b_op = (
            b_op,
            a_op,
        )  # feeds (B, A): lane L holds C[row L%16, cols 4*(L//16)..]
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        src2 = "$0" if acc is not None else "0"
        asm = (
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
            f"{opsel} {opsel_hi} cbsz:4 blgp:4"
        )
        ops = [
            fx.as_ir_value(a_op),
            fx.as_ir_value(b_op),
            fx.as_ir_value(sa_v),
            fx.as_ir_value(sb_v),
        ]
        cons = "=v,a,a,v,v"
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


def _lds_ptr(addr_i32):
    return _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(addr_i32))


def _lds_load_i32(addr_i32):
    return fx.Int32(
        _llvm.LoadOp(fx.Int32.ir_type, _lds_ptr(addr_i32), alignment=4).result
    )


def _lds_load_vec(addr_i32, n):
    ty = _ir.VectorType.get([n], _T.i32)
    return _llvm.LoadOp(ty, _lds_ptr(addr_i32), alignment=4 * n).result


def _lds_store_vec(vec, addr_i32, n):
    _llvm.StoreOp(vec, _lds_ptr(addr_i32), alignment=4 * n)


def _bf16x2(a, b):
    """Two f32 -> one i32 of 2 bf16 (RNE, v_cvt_pk_bf16_f32: the same conversion as
    production's arith.truncf)"""
    return Vec.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]


def _cvt_pk_fp8(old, a, b, scale_f32, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32: (a, b) / scale -> 2 x e4m3 into the low (hi=False)
    or high 16 bits of ``old`` (the intrinsic works on <2 x i16>; i32 in/out here)."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    old_v = _llvm.bitcast(v2i16, fx.as_ir_value(old))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            old_v,
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(scale_f32),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(_T.i32, res))


def _e8m0_fp8(amax):
    """floor(log2 amax) - 7, floored at 0: amax / 2^(e-127) lands in [128, 256),
    inside e4m3 (max 448) with one bit of headroom, never saturates."""
    e = (_bits(amax) >> 23) - fx.Int32(7)
    return (e > fx.Int32(0)).select(e, fx.Int32(0))


def _fabs(v):
    return fx.math.absf(v)


def _maxf(a, b):
    """v_max_f32 (arith.maxnumf), not the select-based gemm1 helper"""
    return fx.Float32(fx.arith.maxnumf(fx.as_ir_value(a), fx.as_ir_value(b)))


def _permlane32_swap(d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane32_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )


def _maxf_nn(a, b):
    """v_max_f32 with nnan. Without the flag LLVM canonicalizes both inputs first
    (``v_max x, x, x``, maxnum must quiet sNaNs): 2 extra VALU per max in the
    epilogue."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _xlane_max4_pair(x, y):
    """4-lane max (lanes L, L^16, L^32, L^48) of two values at once, both broadcast to
    every lane, in 3 permlane swaps (2 x 2 done separately, plus 2 copies):
    swap32(x, y) -> [x_lo|y_lo], [x_hi|y_hi]; max -> m = [X2 | Y2];
    swap16(m, m) + max -> M = [X4 X4 | Y4 Y4]; swap32(M, M) -> [X4|X4], [Y4|Y4]."""
    a, b = _permlane32_swap(_bits(x), _bits(y))
    m = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane16_swap(m, m)
    big = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane32_swap(big, big)
    return _as_f32(a), _as_f32(b)


def _undef_i32():
    """Cvt destination whose other half is overwritten anyway: no v_mov 0 for it"""
    return _llvm.mlir_undef(_T.i32)


def _lds_store_i16(v_i32, addr_i32):
    h = _arith.TruncIOp(_ir.IntegerType.get_signless(16), fx.as_ir_value(v_i32)).result
    _llvm.StoreOp(h, _lds_ptr(addr_i32), alignment=2)


def _ld_dword_asm(rsrc, voff_bytes):
    """buffer_load_dword whose completion we account for ourselves: LLVM sees no load,
    so it adds no vmcnt wait (which would drain the whole DMA stream). The result must
    go through ``_wait_pin`` before any use; it stays live until then, so LLVM cannot
    reuse the register while the load is in flight."""
    return fx.Int32(
        _llvm.inline_asm(
            _T.i32,
            [
                fx.as_ir_value(voff_bytes),
                fx.as_ir_value(rsrc),
                _uniform_i32(fx.Int32(0)),
            ],
            "buffer_load_dword $0, $1, $2, $3 offen",
            "=v,v,s,s",
            has_side_effects=True,
        )
    )


def _wait_pin(vals, vmcnt):
    """``s_waitcnt vmcnt(n)`` with every value tied through it: uses of the returned
    values cannot be scheduled before the wait."""
    n = len(vals)
    ty = _ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * n) + ")>")
    cons = ",".join(["=v"] * n + [str(i) for i in range(n)])
    res = _llvm.inline_asm(
        ty,
        [fx.as_ir_value(v) for v in vals],
        f"s_waitcnt vmcnt({vmcnt})",
        cons,
        has_side_effects=True,
    )
    return [fx.Int32(_llvm.extractvalue(_T.i32, res, [i])) for i in range(n)]


def _pin_vec4(v):
    """Route an accumulator through a side-effecting no-op asm (tied VGPR operand): its
    consumers cannot be scheduled before the asm, and the asm keeps its place among the
    MFMA asms, so the reads happen 8+ MFMAs after the one that produced the value (the
    compiler cannot see inline-asm MFMAs -> inserts no hazard wait states)."""
    ty = Vec.make_type(4, fx.Float32)
    return _llvm.inline_asm(
        ty, [fx.as_ir_value(v)], "; pin $0", "=v,0", has_side_effects=True
    )


def _v2i32(a, b):
    return Vec.from_elements([a, b], fx.Int32).ir_value()


class _BScaleGather:
    """Per pipeline step the 8 e8m0 blocks (256 B = 32 W2 rows x 8 K-groups) of
    the n-tile's 256 rows for one K-step, 2 ``buffer_load_dword ... lds`` per
    wave (wave w fetches row-groups 2w, 2w+1 into slot bytes [2w*256, +512))."""

    def __init__(self, rsrc, lane_id, wave_id, region_base_i32):
        self.rsrc = fx.as_ir_value(rsrc)
        self.voff = [
            fx.as_ir_value((wave_id * 2 + q) * fx.Int32(768) + lane_id * fx.Int32(4))
            for q in range(2)
        ]
        wave_u = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(wave_id)))
        self.base_s = _uniform_i32(region_base_i32 + wave_u * fx.Int32(512))

    def gather(self, slot_byte_off, soff_bytes):
        m0 = fx.as_ir_value(fx.Int32(self.base_s) + slot_byte_off)
        soff = _uniform_i32(soff_bytes)
        asm = (
            "s_mov_b32 m0, $0\n"
            "buffer_load_dword $1, $2, $3 offen lds\n"
            "s_add_u32 m0, 256, m0\n"
            "buffer_load_dword $4, $2, $3 offen lds"
        )
        _asm_void(
            [m0, self.voff[0], self.rsrc, soff, self.voff[1]],
            asm,
            "s,v,s,s,v",
            "~{scc}",
        )


# non-temporal row stores (3-8% faster); the 4-B scale stores need L2 write combining
_STORE_CPOL = 0x2
_STORE_CPOL_SC = 0


def compile_moe_gemm2(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    topk: int,
    n_split: int = 2,
    out_dtype: str = "bf16",
    sort_block_m: int = 128,
):
    """fp4 grouped down-projection with token-major output (see the module doc).
    The CTA tile is always 128 sorted rows; ``sort_block_m`` (128 or 256) is the
    block size of the sort that produced the rows: with 256, ``sorted_expert_ids``
    has one entry per 256 rows (index ``tile_i >> 1``) and a 128-row tile that
    starts with a padding sentinel is all padding and is skipped."""
    assert out_dtype in ("bf16", "fp8"), out_dtype
    FP8 = out_dtype == "fp8"
    BM = 128
    assert sort_block_m in (128, 256), sort_block_m
    EID_SHIFT = (sort_block_m // BM).bit_length() - 1  # 0 or 1
    BN = 256
    K = I
    K_BYTES = K // 2
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    assert K % BLOCK_K == 0
    K_ITERS = K // BLOCK_K  # 3
    assert K_ITERS == 3, "the flat pipeline uses LDS set = k-step (3 sets)"
    N_TILES_ALL = H // BN  # 24
    assert N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    assert NT % 2 == 0, "the n loop is unrolled by 2"
    LDS_BLOCK_M = BM // 2  # 64 rows per A half
    LDS_BLOCK_N = BN // 2  # 128 W2 rows per B half
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 2
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    SC_COLS = K // 32  # 24 e8m0 per A row
    SC_BLOCKS_PER_G = SC_COLS // 8  # 3 blocks of 256 B per 32-row group
    OUT_ELEM = 1 if FP8 else 2
    OUT_ROW_BYTES = H * OUT_ELEM
    OUT_SC_COLS = H // 32
    WAVE_COLS = BN // 2  # 128 output columns per wave per n-tile

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BYTES = K_ITERS * 2 * a_lds_size  # 48 KB, whole A tile (prologue only)
    B_SET_BYTES = 2 * b_lds_size  # 32 KB
    # set 2 lives in the A region (A is in registers before B(2) is issued)
    B_SET_OFF = [A_BYTES, A_BYTES + B_SET_BYTES, 0]
    LDS_TILES_BYTES = A_BYTES + 2 * B_SET_BYTES  # 112 KB
    A_SC_BYTES = 4096  # 12 blocks used (4 groups x 3 K-blocks), 4 waves x 1 KB DMA
    B_SC_SLOT = 2048  # 8 blocks
    B_SC_SLOTS = 4
    SC_LDS_BYTES = A_SC_BYTES + B_SC_SLOTS * B_SC_SLOT  # 12 KB
    B_SC_OFF = A_SC_BYTES
    # staging: per wave 32 rows x 128 columns of the output element (+ fp8 scale bytes)
    STG_ROW = WAVE_COLS * OUT_ELEM  # 128 (fp8) / 256 (bf16) B per staged row
    CH = STG_ROW // 16  # 16-B chunks per row: 8 / 16
    STG_DATA = 32 * STG_ROW
    STG_SC = 32 * 4 if FP8 else 0  # 4 e8m0 per row per wave
    STG_WAVE = (STG_DATA + STG_SC + 255) // 256 * 256
    STG_LDS_BYTES = _N_WAVES * STG_WAVE
    ROWS_PER_ST = 64 // CH  # rows covered by one dwordx4 wave store: 8 / 4
    N_ST = 32 // ROWS_PER_ST  # data stores per flushed row half: 4 / 8
    assert LDS_TILES_BYTES + SC_LDS_BYTES + STG_LDS_BYTES <= 160 * 1024

    NB = N_TILES_B
    NG = 2  # gather loads per wave per step
    P_STEP = 2 * NB + NG  # loads per lane per step
    ST = N_ST + (1 if FP8 else 0)  # stores per lane per flush

    B_TILE_BYTES = BN * K_BYTES  # W2 bytes per n-tile
    B_K_STEP = 2 * 1024  # preshuffled: 128 B of K = 2 x (16 rows x 64 B)

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SC_LDS_BYTES, 16]
        stage_lds: fx.Array[fx.Int8, STG_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr
        _sc_ptr = lds.scale_lds.ptr
        _stg_ptr = lds.stage_lds.ptr

        def a_buf(kb, half):
            return _Buf(_base_ptr, (kb * 2 + half) * a_lds_size)

        def b_buf(s, half):
            return _Buf(_base_ptr, B_SET_OFF[s] + half * b_lds_size)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # ---- work item: (m-tile, n-chunk); consecutive m-tiles share an XCD ----
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        num_valid = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        intra, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        work = xcd * per_xcd + intra
        block_valid = (intra < per_xcd) & (work < n_work)
        work_safe = block_valid.select(work, fx.Int32(0))
        tile_i, chunk = _divmod_nonneg(work_safe, n_split)
        m_base = tile_i * BM
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        expert = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                tile_i >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        if const_expr(EID_SHIFT > 0):
            # a 256-sort pads each expert to 256 rows: a 128-row tile whose first
            # row is the sentinel (tok == n_tokens) holds nothing to compute
            ids_pre = buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            first_sid = fx.Int32(
                buffer_ops.buffer_load(
                    ids_pre, m_base, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            block_valid = block_valid & ((first_sid & fx.Int32(0x00FFFFFF)) < n_tokens)
        chunk_n0 = chunk * NT  # first n-tile (global index) of this CTA
        # rotated n-tile sweep: CTA j starts at n-tile 2*j, so the tile it needs next
        # was just read by CTA j+1 and sits in L2 (32768 tokens: 830 -> 669 us); only
        # inside runs of >= 4 same-expert m-tiles
        ROT_STRIDE, ROT_GATE = 2, 3
        _d = fx.Int32(ROT_GATE)
        _lo_ok = tile_i >= _d
        _hi_ok = tile_i + _d < num_m_blocks
        _e_lo = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _lo_ok.select(tile_i - _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        _e_hi = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _hi_ok.select(tile_i + _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        rot_on = (_lo_ok & (_e_lo == expert)) | (_hi_ok & (_e_hi == expert))
        nt_rot = rot_on.select(
            _divmod_nonneg(tile_i * fx.Int32(ROT_STRIDE), NT)[1], fx.Int32(0)
        )

        def _pn(nt):
            """CTA-local logical n-tile (sweep order) -> physical n-tile of the chunk"""
            x = nt + nt_rot
            return (x >= fx.Int32(NT)).select(x - fx.Int32(NT), x)

        if block_valid:
            ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
            )
            sw_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * 4),
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES)
            )
            as_rsrc = buffer_ops.create_buffer_resource(
                A_scale, max_size=False, num_records_bytes=num_m_blocks * (BM * SC_COLS)
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W2, max_size=False, num_records_bytes=E * H * K_BYTES
            )
            bs_rsrc = buffer_ops.create_buffer_resource(
                W2_scale, max_size=False, num_records_bytes=E * H * SC_COLS
            )

            # ---- A: contiguous sorted rows, swizzled 128-B LDS rows ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    offs.append(
                        (m_base + half * LDS_BLOCK_M + row) * fx.Int32(K_BYTES)
                        + _swizzled_col(row, col)
                    )
                return offs

            # B: LDS half hb, row r -> W2 row (r//64)*128 + hb*64 + r%64 of the n-tile
            def _b_offsets(hb):
                offs = []
                for rnd in range_constexpr(NB):
                    r = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    w2row = (r // 64) * 128 + hb * 64 + (r % 64)
                    offs.append(
                        (w2row // 16) * (K_BYTES * 16)
                        + (w2row % 16) * 16
                        + (col // 64) * 1024
                        + ((col % 64) // 16) * 256
                        + (col % 16)
                    )
                return offs

            a0_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(0), N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, _a_offsets(1), N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(0), NB, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, _b_offsets(1), NB, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            # A scales: 4 x 1 KB for the 3 KB of this m-tile
            as_g2s = G2SLoaderAsm(
                as_rsrc,
                [
                    (m_base // 32) * fx.Int32(SC_BLOCKS_PER_G * 256)
                    + wave_id * 1024
                    + lane_id * 16
                ],
                1,
                wave_id,
            )
            as_g2s.set_wave_base(_sc_ptr)
            sc_base_i32 = fx.Int32(fx.ptrtoint(_sc_ptr))
            bsg = _BScaleGather(
                bs_rsrc, lane_id, wave_id, sc_base_i32 + fx.Int32(B_SC_OFF)
            )

            a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
            b_s2r = S2RLoaderFp4(wave_j, NB)
            mfma = _MfmaAgprAB(N_TILES_A, NB)

            expert_rows = expert * fx.Int32(H)
            b_base_bytes = expert_rows * fx.Int32(K_BYTES)
            bs_base_bytes = expert_rows * fx.Int32(SC_COLS)

            def _b_soff(nt, kb):
                """W2 byte offset of (n-tile nt [CTA-local], K-step kb)"""
                return (
                    b_base_bytes
                    + (chunk_n0 + _pn(nt)) * fx.Int32(B_TILE_BYTES)
                    + fx.Int32(kb * B_K_STEP)
                )

            def _g_soff(nt, kb):
                return (
                    bs_base_bytes
                    + (chunk_n0 + _pn(nt)) * fx.Int32(BN * SC_COLS)
                    + fx.Int32(kb * 256)
                )

            def _slot_off(nt, kb):
                """Scale slot of flat step 3*nt + kb"""
                s = (nt * fx.Int32(K_ITERS) + fx.Int32(kb)) & fx.Int32(B_SC_SLOTS - 1)
                return s * fx.Int32(B_SC_SLOT)

            def _gather(nt, kb):
                bsg.gather(_slot_off(nt, kb), _g_soff(nt, kb))

            def _load_b(nt, kb):
                """Both halves of B(nt, kb) into LDS set kb (issue order b0 then b1)"""
                b0_g2s.load(b_buf(kb, 0), _b_soff(nt, kb))
                b1_g2s.load(b_buf(kb, 1), _b_soff(nt, kb))

            # row ids / routing weights: issued ahead of the DMA stream
            def _wave_row(h, r):
                return h * LDS_BLOCK_M + wave_i * 32 + r

            _id_rows = [
                _wave_row(h, lane_id // CH + ROWS_PER_ST * k)
                for h in range(2)
                for k in range(N_ST)
            ] + ([_wave_row(h, lane_id % 32) for h in range(2)] if FP8 else [])
            _raw_ids = [
                _ld_dword_asm(ids_rsrc, (m_base + r) * fx.Int32(4)) for r in _id_rows
            ]
            _raw_w = (
                [
                    _ld_dword_asm(
                        sw_rsrc, (m_base + _wave_row(h, ti * 16 + r16)) * fx.Int32(4)
                    )
                    for h in range(2)
                    for ti in range(N_TILES_A)
                ]
                if not FP8
                else []
            )

            # ---- prologue: A (12) + A scales (1), g(0), B(0), B(1), g(1) ----
            for kb in range_constexpr(K_ITERS):
                a0_g2s.load(a_buf(kb, 0), fx.Int32(kb * BLOCK_K_BYTES))
                a1_g2s.load(a_buf(kb, 1), fx.Int32(kb * BLOCK_K_BYTES))
            as_g2s.load(_Buf(_sc_ptr, 0), fx.Int32(0))
            n0 = fx.Int32(0)
            _gather(n0, 0)
            _load_b(n0, 0)
            _load_b(n0, 1)
            _gather(n0, 1)

            # A + A scales landed (g(0), B(0), B(1), g(1) may fly)
            wait_barrier(NG + 2 * NB + 2 * NB + NG)
            aF = [[a_s2r.load(a_buf(kb, h)) for h in range(2)] for kb in range(K_ITERS)]
            lane_sc = (lane_id // 4) * 16 + (lane_id % 4) * 4
            saA = [
                [
                    _lds_load_i32(
                        sc_base_i32
                        + ((h * 2 + wave_i) * SC_BLOCKS_PER_G + kb) * fx.Int32(256)
                        + lane_sc
                    )
                    for kb in range(K_ITERS)
                ]
                for h in range(2)
            ]
            # every wave's A reads are done before B(2) overwrites the A region
            wait_barrier(NG + 2 * NB + 2 * NB + NG)
            _gather(n0, 2)
            _load_b(n0, 2)

            def _read_bsc_thunks(nt, kb, holder):
                base = sc_base_i32 + fx.Int32(B_SC_OFF) + _slot_off(nt, kb) + lane_sc

                def _rd(idx, hb, sub):
                    gi = wave_j * 4 + hb * 2 + sub
                    holder[idx] = _lds_load_i32(base + fx.Int32(gi * 256))

                return [
                    lambda: _rd(0, 0, 0),
                    lambda: _rd(1, 0, 1),
                    lambda: _rd(2, 1, 0),
                    lambda: _rd(3, 1, 1),
                ]

            # B(0) + g(0) landed (B(1), g(1), g(2), B(2) may fly)
            wait_barrier(2 * NB + NG + NG + 2 * NB)
            b0f = b_s2r.load(b_buf(0, 0), preshuffled=True)
            b1f = b_s2r.load(b_buf(0, 1), preshuffled=True)
            _sc0 = [None] * 4
            for t in _read_bsc_thunks(n0, 0, _sc0):
                t()

            # ---- output rows (token-major) of the rows this lane touches ----
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=n_tokens * (topk * OUT_ROW_BYTES)
            )
            osc_rsrc = buffer_ops.create_buffer_resource(
                OUT_scale,
                max_size=False,
                num_records_bytes=n_tokens * (topk * OUT_SC_COLS),
            )

            _pinned = _wait_pin(_raw_ids + _raw_w, 2 * NB + NG + NG + 2 * NB)
            _pid, _pw = _pinned[: len(_raw_ids)], _pinned[len(_raw_ids) :]

            def _orow(sid):
                tok = sid & fx.Int32(
                    0x00FFFFFF
                )  # padded rows: tok == n_tokens -> OOB -> dropped
                slot = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot

            # data flush: lane L stores staged rows L//CH + ROWS_PER_ST*k, chunk L%CH
            out_off = [
                [
                    _orow(_pid[h * N_ST + k]) * fx.Int32(OUT_ROW_BYTES)
                    for k in range(N_ST)
                ]
                for h in range(2)
            ]
            # conditional expressions: the DSL rewriter drops non-IR values set in an if
            sc_off = (
                [_orow(_pid[2 * N_ST + h]) * fx.Int32(OUT_SC_COLS) for h in range(2)]
                if FP8
                else None
            )
            # bf16: routing weight of the lane's accumulator rows (ti*16 + r16)
            wA = (
                [
                    [_as_f32(_pw[h * N_TILES_A + ti]) for ti in range(N_TILES_A)]
                    for h in range(2)
                ]
                if not FP8
                else None
            )

            stg_base = fx.Int32(fx.ptrtoint(_stg_ptr)) + wave_id * fx.Int32(STG_WAVE)
            stg_sc_base = stg_base + fx.Int32(STG_DATA)

            def _stg_addr(row, chunk, half):
                """Staged (row, 16-B chunk, 8-B half); chunks XOR-swizzled by the row so
                the column-wise writes and the row-wise reads both spread over the
                banks"""
                return (
                    stg_base
                    + row * fx.Int32(STG_ROW)
                    + ((chunk ^ (row % CH)) * 16 + half * 8)
                )

            def _stage_bf16(cq, h, hb, ti, tj):
                """One 16x16 tile: 4 f32 -> bf16(v * w_row), 8 B into the staging
                buffer"""
                cv = Vec(_pin_vec4(cq[mfma.idx(ti, tj)]))
                w = wA[h][ti]  # type: ignore[index]
                v = [fx.Float32(cv[k]) * w for k in range_constexpr(4)]
                row = ti * 16 + r16
                chunk = (
                    hb * 8 + tj * 2 + g4 // 2
                )  # (hb*64 + tj*16 + g4*4) cols * 2 B / 16
                _lds_store_vec(
                    _v2i32(_bf16x2(v[0], v[1]), _bf16x2(v[2], v[3])),
                    _stg_addr(row, chunk, g4 % 2),
                    2,
                )

            def _stage_fp8_pair(cq, hb, ti):
                """The two 16-row x 32-col groups of (ti, hb) -- tiles (0, 1) and (2,
                3) --
                at once. The fp8 epilogue is VALU-issue bound (1 wave / SIMD; the ATT
                showed ~35 sequential instructions per group between two MFMAs), so: one
                amax butterfly serves both groups (3 swaps, not 4), nnan maxes (no
                canonicalizing self-max), undef cvt destinations (no v_mov 0), the two
                e8m0 bytes of a row go out as one 16-bit LDS write. Per group: amax over
                the row's 4 lanes, e8m0, 8 fp8 per lane after a permlane16 swap."""
                v = []
                for p in range_constexpr(2):
                    cv = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p)]))
                    cw = Vec(_pin_vec4(cq[mfma.idx(ti, 2 * p + 1)]))
                    v.append(
                        [fx.Float32(cv[k]) for k in range_constexpr(4)]
                        + [fx.Float32(cw[k]) for k in range_constexpr(4)]
                    )
                am = []
                for p in range_constexpr(2):
                    a = _maxf_nn(_fabs(v[p][0]), _fabs(v[p][1]))
                    for k in range_constexpr(2, 8):
                        a = _maxf_nn(a, _fabs(v[p][k]))
                    am.append(a)
                amax = _xlane_max4_pair(am[0], am[1])
                row = ti * 16 + r16
                e8s = []
                for p in range_constexpr(2):
                    e8 = _e8m0_fp8(amax[p])
                    sf = _as_f32(e8 << 23)
                    da = _cvt_pk_fp8(_undef_i32(), v[p][0], v[p][1], sf, False)
                    da = _cvt_pk_fp8(da, v[p][2], v[p][3], sf, True)
                    db = _cvt_pk_fp8(_undef_i32(), v[p][4], v[p][5], sf, False)
                    db = _cvt_pk_fp8(db, v[p][6], v[p][7], sf, True)
                    da, db = _permlane16_swap(da, db)
                    # lane group g now holds tile (2p + g%2), cols (g//2)*8 .. +8
                    chunk = hb * 4 + 2 * p + g4 % 2
                    _lds_store_vec(_v2i32(da, db), _stg_addr(row, chunk, g4 // 2), 2)
                    e8s.append(e8)
                _lds_store_i16(
                    e8s[0] | (e8s[1] << 8), stg_sc_base + row * fx.Int32(4) + hb * 2
                )

            def _stage_thunks(cq, h, hb):
                ts = []
                for ti in range_constexpr(N_TILES_A):
                    if const_expr(FP8):
                        ts.append(lambda ti=ti: _stage_fp8_pair(cq, hb, ti))
                    else:
                        for tj in range_constexpr(NB):
                            ts.append(
                                lambda ti=ti, tj=tj: _stage_bf16(cq, h, hb, ti, tj)  # type: ignore[misc]
                            )
                return ts

            def _flush_thunks(h, nt, mask):
                """The wave's staged 32 x 128 outputs -> token-major rows, full lines"""
                col_wave = (
                    (chunk_n0 + _pn(nt)) * fx.Int32(BN) + wave_j * WAVE_COLS
                ) * OUT_ELEM
                ts = []
                for k in range_constexpr(N_ST):

                    def _st(k=k):
                        row = lane_id // CH + ROWS_PER_ST * k
                        chunk = lane_id % CH
                        data = _lds_load_vec(_stg_addr(row, chunk, 0), 4)
                        buffer_ops.buffer_store(
                            data,
                            out_rsrc,
                            out_off[h][k] + col_wave + chunk * 16,
                            mask=mask,
                            offset_is_bytes=True,
                            cache_modifier=_STORE_CPOL,
                        )

                    ts.append(_st)
                if const_expr(FP8):

                    def _st_sc():
                        scv = _lds_load_i32(stg_sc_base + (lane_id % 32) * 4)
                        sc_col = (chunk_n0 + _pn(nt)) * fx.Int32(BN // 32) + wave_j * 4
                        buffer_ops.buffer_store(
                            scv,
                            osc_rsrc,
                            sc_off[h] + sc_col,  # type: ignore[index]
                            mask=mask,
                            offset_is_bytes=True,
                            cache_modifier=_STORE_CPOL_SC,
                        )

                    ts.append(_st_sc)  # type: ignore[arg-type]
                return ts

            def _wb2(first, c_first, c_other):
                """``first`` is a Python bool (static) or a wave-uniform DSL bool"""
                if const_expr(isinstance(first, bool)):
                    wait_barrier(c_first if first else c_other)
                elif const_expr(c_first == c_other):
                    wait_barrier(c_first)
                else:
                    if first:
                        wait_barrier(c_first)
                    else:
                        wait_barrier(c_other)

            # vmcnt allowances (loads and stores retire in issue order)
            TOP_K0, SEG_K0 = P_STEP + ST, NG + P_STEP + ST + NB
            TOP_K1, SEG_K1 = ST + P_STEP + ST, NG + ST + P_STEP + ST + NB
            TOP_K2, SEG_K2 = ST + P_STEP, NG + ST + P_STEP + NB
            # first tile: after the prologue [.. B(0), B(1), g(1), g(2), B(2)]
            TOP_K0_F = NG + 2 * NB  # after g(1): g(2), B(2)
            SEG_K0_F = NG + NG + 2 * NB + NB  # after b1(1): g(1), g(2), B(2), b0(3)
            TOP_K1_F = 2 * NB + P_STEP + ST  # after g(2): B(2), step 0
            SEG_K1_F = P_STEP + ST + NB  # after b1(2): step 0, b0(4)
            for c in (
                TOP_K0,
                SEG_K0,
                TOP_K1,
                SEG_K1,
                TOP_K2,
                SEG_K2,
                TOP_K0_F,
                SEG_K0_F,
                TOP_K1_F,
                SEG_K1_F,
            ):
                assert c <= 63, c

            def _one_step(nt, nt_next, kb, first, b0f, b1f, sc, accs, epi):
                """Flat step s = (nt, kb): B(s) in LDS set kb; issues B(s+3) = (nt+1,
                kb) into
                the same set and its scales. ``epi``: {phase: thunk-list factory(accs)}
                for
                the epilogue work hidden in that phase's MFMA shadow."""
                kb1 = (kb + 1) % K_ITERS
                nt1 = nt if (kb + 1) < K_ITERS else nt_next
                bn0, bn1 = b_buf(kb1, 0), b_buf(kb1, 1)
                sbC0, sbC1 = sc
                c00, c01, c10, c11 = accs
                a0f, a1f = aF[kb][0], aF[kb][1]
                zero = kb == 0

                def _late(ph):
                    return epi[ph](accs) if ph in epi else None

                _scn = [None] * 4
                rd_scn = _read_bsc_thunks(nt1, kb1, _scn)
                b_off3 = _b_soff(nt_next, kb)

                if kb == 0:
                    _wb2(first, TOP_K0_F, TOP_K0)
                elif kb == 1:
                    _wb2(first, TOP_K1_F, TOP_K1)
                else:
                    wait_barrier(TOP_K2)
                il = _g2s_thunks(b0_g2s, b_buf(kb, 0), b_off3, NB) + rd_scn[:2]
                mfma.call(
                    a0f,
                    b0f,
                    c00,
                    [saA[0][kb]],
                    sbC0,
                    interleave=il,
                    zero_acc=zero,
                    late=_late(1),
                )
                mfma.call(
                    a0f,
                    b1f,
                    c01,
                    [saA[0][kb]],
                    sbC1,
                    interleave=rd_scn[2:],
                    zero_acc=zero,
                    late=_late(2),
                )

                if kb == 0:
                    _wb2(first, SEG_K0_F, SEG_K0)
                elif kb == 1:
                    _wb2(first, SEG_K1_F, SEG_K1)
                else:
                    wait_barrier(SEG_K2)
                _b0n = [None] * NB
                _b1n = [None] * NB
                # the gather sets m0: it goes after the m0-relative DMAs
                il = _g2s_thunks(b1_g2s, b_buf(kb, 1), b_off3, NB) + [
                    lambda: _gather(nt_next, kb)
                ]
                mfma.call(
                    a1f,
                    b0f,
                    c10,
                    [saA[1][kb]],
                    sbC0,
                    interleave=il,
                    zero_acc=zero,
                    late=_late(3),
                )
                il = _s2r_thunks(b_s2r, bn0, _b0n, NB, True) + _s2r_thunks(
                    b_s2r, bn1, _b1n, NB, True
                )
                mfma.call(
                    a1f,
                    b1f,
                    c11,
                    [saA[1][kb]],
                    sbC1,
                    interleave=il,
                    zero_acc=zero,
                    late=_late(4),
                )
                return _b0n, _b1n, (_scn[:2], _scn[2:])

            _R = fx.as_ir_value

            def _flat_b(frag):
                return [_R(t[0]) for t in frag] + [_R(t[1]) for t in frag]

            def _unflat_b(flat):
                return [[flat[i], flat[NB + i]] for i in range(NB)]

            def _flat_state(b0f, b1f, sc, c11):
                return (
                    _flat_b(b0f)
                    + _flat_b(b1f)
                    + [_R(v) for v in sc[0]]
                    + [_R(v) for v in sc[1]]
                    + [_R(v) for v in c11]
                )

            def _unflat_state(st):
                o = 0
                b0f = _unflat_b(st[o : o + 2 * NB])
                o += 2 * NB
                b1f = _unflat_b(st[o : o + 2 * NB])
                o += 2 * NB
                sc = (list(st[o : o + 2]), list(st[o + 2 : o + 4]))
                o += 4
                c11 = list(st[o : o + N_ACCUMS])
                return b0f, b1f, sc, c11

            def _n_tile(nt, nt_next, first, b0f, b1f, sc, c11_prev, nt_prev, mask_prev):
                """Epilogue placement (P = phase): tile t's c00 staged in (t,2) P2, c01
                in
                (t,2) P3, rows h0 flushed + c10 staged in (t,2) P4, c11 staged in
                (t+1,0)
                P1, rows h1 flushed in (t+1,0) P4. Returns (b0f, b1f, sc, c11)."""
                accs = tuple([None] * N_ACCUMS for _ in range(4))
                epi_by_kb = {
                    0: {
                        1: lambda a: _stage_thunks(c11_prev, 1, 1),
                        4: lambda a: _flush_thunks(1, nt_prev, mask_prev),
                    },
                    1: {},
                    2: {
                        2: lambda a: _stage_thunks(a[0], 0, 0),
                        3: lambda a: _stage_thunks(a[1], 0, 1),
                        4: lambda a: (
                            _flush_thunks(0, nt, None) + _stage_thunks(a[2], 1, 0)
                        ),
                    },
                }
                for kb in range_constexpr(K_ITERS):
                    b0f, b1f, sc = _one_step(
                        nt, nt_next, kb, first, b0f, b1f, sc, accs, epi_by_kb[kb]
                    )
                return b0f, b1f, sc, accs[3]

            zero_v4 = Vec.filled(4, 0.0, fx.Float32).ir_value()  # accumulator zero
            init_state = _flat_state(
                b0f, b1f, (_sc0[:2], _sc0[2:]), [zero_v4] * N_ACCUMS
            )
            for np_, state in range(0, NT // 2, init=init_state):  # type: ignore[call-overload]
                b0f, b1f, sc, c11p = _unflat_state(state)
                np_i = fx.Int32(np_)
                first = np_i == fx.Int32(0)
                nt_e = np_i * fx.Int32(2)
                nt_o = nt_e + fx.Int32(1)
                nt_o_next = _min(
                    nt_e + fx.Int32(2), fx.Int32(NT - 1)
                )  # last pair: redundant reloads
                pend_valid = fx.as_ir_value(np_i > fx.Int32(0))
                b0f, b1f, sc, c11e = _n_tile(
                    nt_e,
                    nt_o,
                    first,
                    b0f,
                    b1f,
                    sc,
                    c11p,
                    nt_e - fx.Int32(1),
                    pend_valid,
                )
                b0f, b1f, sc, c11o = _n_tile(
                    nt_o, nt_o_next, False, b0f, b1f, sc, c11e, nt_e, None
                )
                state = yield _flat_state(b0f, b1f, sc, c11o)

            # the last tile's c11 has no next step to hide in
            b0f, b1f, sc, c11p = _unflat_state(state)
            for t in _stage_thunks(c11p, 1, 1) + _flush_thunks(
                1, fx.Int32(NT - 1), None
            ):
                t()

            # never retire with LDS DMA in flight (the CU reuses the LDS)
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


def gemm2_grid(num_m_blocks: int, n_split: int) -> int:
    """Blocks to launch: every (m-tile, chunk) of the allocation, rounded to the 8 XCDs
    (the kernel drops the fully padded tail tiles at runtime)."""
    return (num_m_blocks * n_split + _NUM_XCDS - 1) // _NUM_XCDS * _NUM_XCDS
