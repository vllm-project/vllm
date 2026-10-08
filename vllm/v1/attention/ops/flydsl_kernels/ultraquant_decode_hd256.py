# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# UltraQuant D=256 decode (FlyDSL) — gfx950 / CDNA4
#
# Production path only: scaled FP4×E4M3 QK MFMA, native V CVT, HW V-transpose,
# Q hoist, strided tile-groups, and in-kernel Walsh-Hadamard for GQA 6/8/16.
#
# AoS slot (D=256, group_size=32 → 8 groups, 272 B):
#   [0:128) K FP4 codes; [128:136) K UE8M0; [136:264) V FP4; [264:272) V UE8M0
# QK MFMA: A=K, B=Q → C[token, query]; PV MFMA: A=V_T, B=P → C[head_dim, query]

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops, vector
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import (
    arith,
    gpu,
    rocdl,
)
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def _vector_insert(value, dest, *, static_position):
    """Insert a DSL scalar using FlyDSL 0.3's raw vector dialect binding."""
    return vector.insert(
        vector.as_ir_value(value),
        vector.as_ir_value(dest),
        dynamic_position=[],
        static_position=static_position,
    )


# === Constants (256-wide-head ultraquant decode profile) =======================
HEAD_SIZE = 256
KV_BLOCK_SIZE = 16  # default; overridable via build(kv_block_size=...)
TILE_SIZE = 16  # MFMA tile = 16 tokens
N_CENTROIDS = 16  # FP4 E2M1 value table (16 entries indexed by nibble)
QUERY_GROUP_SIZE = 16  # default for tests; overridable via build(...)
WARP_SIZE = 64
BLOCK_THREADS = WARP_SIZE
KV_COMPUTE_BLOCK = 256  # 16 K-tiles × 16 tokens

# ultraquant AoS slot layout (per (slot, head), D=256, group_size=32 => 8 groups)
FP8_GROUP_SIZE = 32
N_GROUPS = HEAD_SIZE // FP8_GROUP_SIZE  # 8 UE8M0 scale bytes per head
KEY_CODE_BYTES = HEAD_SIZE // 2  # 128
VAL_CODE_BYTES = HEAD_SIZE // 2  # 128
K_SCALES_OFFSET = KEY_CODE_BYTES  # 128
V_CODES_OFFSET = KEY_CODE_BYTES + N_GROUPS  # 136
V_SCALES_OFFSET = V_CODES_OFFSET + VAL_CODE_BYTES  # 264
SLOT_CONTENT_BYTES = V_SCALES_OFFSET + N_GROUPS  # 272

# MFMA
MFMA_N = 16
PV_N_CHUNKS = HEAD_SIZE // MFMA_N  # 16 for HEAD_SIZE=256

# --- Per-lane data-movement geometry (generalized from the 128 kernel) ------
# The K/V dequant lane assignment splits each token's packed code bytes across
# the 4 ``chunk_in_tok`` sub-lanes (lane % 4). Each lane processes
# ``KEY_CODE_BYTES // 4`` code bytes = 32 B at 256. buffer_load maxes at 16 B
# (dwordx4) so a lane issues ``HALVES``=2 loads; each 16-B half covers exactly
# one UE8M0 group of 32 head-dims.
#   HEAD_SIZE=128: LANE_CODE_BYTES=16, HALVES=1, SUBCHUNK_HDIMS=32
#   HEAD_SIZE=256: LANE_CODE_BYTES=32, HALVES=2, SUBCHUNK_HDIMS=64
LANE_CODE_BYTES = KEY_CODE_BYTES // 4  # 32 code bytes per lane per K-tile
HALVES = LANE_CODE_BYTES // 16  # 2 for HEAD_SIZE=256
HALF_HDIMS = 32  # head-dims per half (= one group)
SUBCHUNK_HDIMS = HEAD_SIZE // 4  # head-dims per chunk_in_tok sub-lane (64)
# Q row-major LDS load geometry (STEP B): each lane loads 8 bf16 (=4 i32),
# so a row spans ``HEAD_SIZE // 8`` column-groups.
Q_GROUPS_PER_ROW = HEAD_SIZE // 8  # 32 for 256
Q_ROW_SHIFT = Q_GROUPS_PER_ROW.bit_length() - 1  # 5 for 256
Q_COL_MASK = Q_GROUPS_PER_ROW - 1  # 31 for 256

# LDS regions
CENTROID_LDS_BYTES = N_CENTROIDS * 4  # 64

# KV LDS rows are padded by 16 B to reduce bank conflicts while preserving
# ds_read/ds_write_b128 alignment.
KV_ROW_PAD_ELEMS = 8
assert KV_ROW_PAD_ELEMS % 8 == 0, "KV row pad must be a multiple of 8 bf16 (16 B)"
KV_ROW_ELEMS = HEAD_SIZE + KV_ROW_PAD_ELEMS  # 264 @ pad 8
KV_ROW_BYTES = KV_ROW_ELEMS * 2  # 528
KFP4_ROW_PAD_I32 = 4
assert KFP4_ROW_PAD_I32 % 4 == 0, "FP4-K row pad must be a multiple of 4 i32 (16 B)"
KFP4_ROW_I32 = HEAD_SIZE // 8 + KFP4_ROW_PAD_I32
KV_TILE_LDS_BYTES_PADDED = TILE_SIZE * KV_ROW_BYTES  # 8448 (V dominates)
SCALE_LDS_BYTES = TILE_SIZE * N_GROUPS * 4  # 16*8*4 = 512 for 256

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, "ir_value") else scalar
    return vec * vector.broadcast(T.f32x4, s)


allocator = None


def build_ultraquant_decode_hd256_module(
    num_seqs: int,
    num_kv_heads: int,
    num_partitions: int,
    padded_slot: int,
    max_blocks_per_seq: int = 512,
    softmax_scale: float | None = None,
    query_group_size: int = QUERY_GROUP_SIZE,
    kv_block_size: int = KV_BLOCK_SIZE,
    tile_groups_per_partition: int = 1,
    fuse_qrot: bool = True,
    stride_q_seq: int | None = None,
    stride_q_head: int | None = None,
):
    """Build the production UltraQuant D=256 decode kernel.

    ``padded_slot`` is the per-(token, head) slot stride in bytes
    (``kv_cache.shape[3]``, >= 272). Cache layout is
    ``[num_blocks, block_size, num_kv_heads, padded_slot]`` uint8 AoS.

    QK is always scaled FP4×E4M3 MFMA; V always uses native CVT + HW
    transpose; tile-groups are always strided. ``fuse_qrot`` selects the
    in-kernel WHT when the query layout is eligible.
    """
    # QG=6 (Qwen3.6 full-attention layers) reuses the 8-row WHT lane map with
    # two inactive rows masked. STEP C may read rows QG..15 from adjacent LDS,
    # but those MFMA rows are discarded by the `mfma_row < QG` output gate.
    assert query_group_size in (6, 8, 16), (
        f"query_group_size must be 6, 8 or 16; got {query_group_size}"
    )
    # head_dim=256 hybrid models (Qwen3.6/3.8) force a larger mamba-aligned KV
    # block; the tiled inner loop (_TILES_PER_BLOCK = kv_block_size // TILE_SIZE)
    # handles any multiple of TILE_SIZE, so accept 16/32/64/128/256.
    assert kv_block_size in (16, 32, 64, 128, 256), (
        f"kv_block_size must be 16/32/64/128/256; got {kv_block_size}"
    )
    assert kv_block_size % TILE_SIZE == 0
    assert int(tile_groups_per_partition) >= 1

    TGPP = int(tile_groups_per_partition)
    # Partition p owns tile-groups p, p+P, p+2P, ... This keeps the graph-fixed
    # grid wide for short sequences; the reducer is agnostic to the mapping.
    NUM_PARTS_C = int(num_partitions)
    MFMA_SCALED_K = 128
    MFMA_ISSUES = HEAD_SIZE // MFMA_SCALED_K  # 2 for HEAD_SIZE=256
    GRPS_PER_ISSUE = MFMA_SCALED_K // FP8_GROUP_SIZE  # 4

    QG = int(query_group_size)
    QG_LOAD_ITERS = (QG * Q_GROUPS_PER_ROW) // WARP_SIZE  # 8@QG16, 4@QG8, 3@QG6

    _BLOCK_THREADS = WARP_SIZE

    # Fuse Q rotation (q_rot = Q @ PiT) into this kernel as an in-register WHT.
    # The 8-row lane map masks its two inactive rows for GQA-6.
    _FUSE_QROT = bool(fuse_qrot)

    # --- Cache policy on the KV code stream ----------------------------------
    # KV codes are read EXACTLY ONCE per decode (81.6 MB at B=4/seq=75k) and
    # never reused, so caching them only evicts Q / block-table / scale lines
    # that ARE reused. gfx9 buffer aux bits: bit0=sc0, bit1=nt, bit4=sc1.
    #   0 = default (what we ship today: no hint at all)
    #   2 = nt        (non-temporal: stream, do not pollute)
    #  18 = nt|sc1    (also bypass the far cache level)
    _KV_AUX = 0

    # --- Occupancy: size the Q LDS to the ACTUAL query group -------------------
    # STEP B only writes rows [0, QG) and STEP C/F only read those rows. For
    # QG=8 the upper 4 KB is allocated-but-never-touched, so shrinking it to
    # QG*HEAD_SIZE*2 is bitwise identical and frees LDS/workgroup (17 KB -> 13 KB
    # at QG=8), lifting waves/CU (~3 -> ~5) since the kernel is LDS-occupancy
    # bound. The launcher pads segm pools to QG=16 regardless, so this is purely
    # an in-kernel LDS footprint reduction.
    _Q_LDS_BYTES = QG * HEAD_SIZE * 2

    _BS = int(kv_block_size)
    _TILES_PER_BLOCK = _BS // TILE_SIZE
    _PADDED_SLOT = int(padded_slot)
    assert _PADDED_SLOT >= SLOT_CONTENT_BYTES, (
        f"padded_slot={_PADDED_SLOT} < {SLOT_CONTENT_BYTES} (ultraquant D=256 content)"
    )
    assert _PADDED_SLOT % 4 == 0, f"padded_slot={_PADDED_SLOT} must be 4-byte aligned"

    arch = get_hip_arch()

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _qk_scale = float(softmax_scale)

    # --- Strides ---
    _Hq = num_kv_heads * QG
    # Defaults describe the pooled [B, Hq, D] q_rot buffer. When the rotation is
    # fused in (STEP B'), Q is read straight from the model's query tensor,
    # which is a SPLIT OF THE FUSED QKV PROJECTION -- so its row stride is
    # (Hq + 2*Hk)*D, not Hq*D. Those strides are constant for a given model and
    # TP rank, so the caller passes them in and they bake in like any other
    # compile-time constant (they are part of the kernel cache key).
    _stride_q_seq = int(stride_q_seq) if stride_q_seq is not None else _Hq * HEAD_SIZE
    _stride_q_head = int(stride_q_head) if stride_q_head is not None else HEAD_SIZE
    _stride_bt_seq = max_blocks_per_seq

    # ultraquant AoS cache: [num_blocks, block_size, num_kv_heads, padded_slot].
    _stride_cache_head = _PADDED_SLOT
    _stride_cache_pos = num_kv_heads * _PADDED_SLOT
    _stride_cache_block = _BS * _stride_cache_pos

    _stride_out_part = QG * HEAD_SIZE
    _stride_out_head = num_partitions * QG * HEAD_SIZE
    _stride_out_seq = num_kv_heads * num_partitions * QG * HEAD_SIZE
    _stride_es_seq = num_kv_heads * num_partitions * QG
    _stride_ml_seq = _stride_es_seq

    # --- LDS layout ---
    # `allocator` is intentionally a BUILD-LOCAL captured by the kernel closure
    # (a freevar), NOT a module global. The @flyc.jit drift guard
    # (_check_globals_drift) snapshots every module global referenced by the JIT
    # launcher's dependency tree (which includes this kernel) and ABORTS if one
    # changes between compiles. When more than one partition variant is
    # JIT-resident at once (e.g. a ragged/adaptive batch mixes num_partitions
    # 256 and 512, or fixed cap=512 yields 294@75k and 512@131k), each build
    # created a fresh SmemAllocator; as a reassigned MODULE GLOBAL that tripped
    # "global 'allocator' changed since first compile" and crashed the worker.
    # Keeping it local + a per-variant symbol name lets the variants coexist.
    # The build-local is handed to the launcher via a module attribute set AFTER
    # the kernel is built (see end of function); that assignment is safe because
    # no @flyc.jit function references the module global anymore.
    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=(f"ultraquant_hd256_smem_p{int(num_partitions)}"),
    )
    centroid_off = 0
    allocator.ptr = CENTROID_LDS_BYTES
    q_off = allocator.ptr
    allocator.ptr += _Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES_PADDED
    scale_off = allocator.ptr
    allocator.ptr += SCALE_LDS_BYTES

    @flyc.kernel
    def ultraquant_decode_hd256_kernel(
        out_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        centroids_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        seq_lens_ptr: fx.Tensor,
    ):
        # ---- IDs ---------------------------------------------------------
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part = gpu.block_idx.z
        lane = tid  # 0..63
        mfma_row = lane & fx.Int32(15)
        mfma_col_grp = lane >> fx.Int32(4)  # 0..3, K-group dim

        # ---- Buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        sl_rsrc = buffer_ops.create_buffer_resource(seq_lens_ptr, max_size=True)
        cent_rsrc = buffer_ops.create_buffer_resource(centroids_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        cent_lds = SmemPtr(base, centroid_off, T.f32, shape=(N_CENTROIDS,))
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(_Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(_Q_LDS_BYTES // 8,)).get()
        # KV / scale views span ALL warps' private regions (W*stride); the
        # per-warp base offset is folded into the element indices below via
        # _kvi*/_sci (a literal 0 for W=1, so the single-warp path is unchanged).
        kv_lds_i32 = SmemPtr(
            base, kv_off, T.i32, shape=(KV_TILE_LDS_BYTES_PADDED // 4,)
        ).get()
        kv_lds_i64 = SmemPtr(
            base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES_PADDED // 8,)
        ).get()
        scale_lds_i32 = SmemPtr(base, scale_off, T.i32, shape=(TILE_SIZE * N_GROUPS,))

        def _kvi32(e):
            return e

        def _kvi64(e):
            return e

        def _sci(e):
            return e

        # ---- Constants ---------------------------------------------------
        c_sq = fx.Int32(_stride_q_seq)
        c_qh = fx.Int32(_stride_q_head)
        c_qg = fx.Int32(QG)
        c_bt = fx.Int32(_stride_bt_seq)
        c_stride_pos = fx.Int32(_stride_cache_pos)
        c_stride_head = fx.Int32(_stride_cache_head)
        c_kscale_off = fx.Int32(K_SCALES_OFFSET)  # 128
        c_vcode_off = fx.Int32(V_CODES_OFFSET)  # 136
        c_vscale_off = fx.Int32(V_SCALES_OFFSET)  # 264
        c_w = fx.Int32(WARP_SIZE)

        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)

        def _ival(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        c_zero_i32 = arith.constant(0, type=T.i32)

        def _f32x8_to_fp8_i64(f):
            # 8 f32 -> 8 E4M3 bytes packed as i64 (MFMA fp8 operand).
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[0], f[1], c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[2], f[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[4], f[5], c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[6], f[7], w1, 1)
            pv = vector.from_elements(T.vec(2, T.i32), [w0, w1])
            return vector.extract(
                vector.bitcast(T.vec(1, T.i64), pv), static_position=[0]
            )

        def _bf16x8_to_fp8_i64(v_bf16):
            f = [
                arith.extf(T.f32, vector.extract(v_bf16, static_position=[i]))
                for i in range(8)
            ]
            return _f32x8_to_fp8_i64(f)

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & fx.Int32(N_CENTROIDS - 1)
        c_val = buffer_ops.buffer_load(cent_rsrc, c_idx_safe, vec_width=1, dtype=T.f32)
        cent_lds.store(c_val, [arith.index_cast(T.index, c_idx_safe)])
        gpu.barrier()

        # ===== STEP B': in-kernel Q rotation (fused WHT) ==================
        # The shipped path needs a SEPARATE prologue kernel to compute
        # q_rot = Q @ PiT. That kernel is ~7 us at B=4 -- essentially ALL fixed
        # per-dispatch cost (measured: 2.1M MACs, and sweeping its grid 4x moves
        # it 0%). PiT is the orthonormal Sylvester Hadamard with a column
        # permutation folded in on the host, so Q @ PiT is a WALSH-HADAMARD
        # TRANSFORM: 8 butterfly stages (2048 add/sub per row) instead of a
        # 256x256 matmul (65536 MACs) -- 32x less work, cheap enough to redo
        # redundantly in every partition and delete the dispatch entirely.
        #
        # Verified offline: butterfly vs matmul differs by <=1.8e-6 in fp32, and
        # after the E4M3 haircut the two agree EXACTLY (0/2048 elements differ),
        # because e4m3 keeps only 3 mantissa bits. So this is bit-exact, not an
        # approximation.
        #
        # Lane map: lane L owns row r = L>>3 and the 32 CONSECUTIVE head-dims
        # d = (L&7)*32 + t. That puts butterfly stages 0..4 (strides 1..16)
        # entirely inside one lane's registers, and leaves only stages 5..7
        # (strides 32/64/128 == chunk XOR 1/2/4) needing a cross-lane exchange.
        if const_expr(_FUSE_QROT):
            _wrow = lane >> fx.Int32(3)  # 0..7  row within the group
            _wchk = lane & fx.Int32(7)  # 0..7  32-dim chunk
            # inv(qperm) is a pure bit permutation of the head-dim index:
            #   dest = (d & 0x8F) | ((d>>2)&0x10) | ((d<<1)&0x60)
            # which splits additively into a compile-time part in t and a
            # per-lane part in the chunk index (verified exhaustively):
            #   ct(t) = (t & 0x0F) | ((t & 0x10) << 1)
            #   rt(c) = ((c&2)<<3) | ((c&1)<<6) | ((c&4)<<5)
            _perm_base = (
                ((_wchk & fx.Int32(2)) << fx.Int32(3))
                | ((_wchk & fx.Int32(1)) << fx.Int32(6))
                | ((_wchk & fx.Int32(4)) << fx.Int32(5))
            )
            _inv_sqrtD = arith.constant(1.0 / math.sqrt(HEAD_SIZE), type=T.f32)
            for _qit in range_constexpr((QG + 7) // 8):
                _qrow = _wrow + fx.Int32(_qit * 8)
                if _qrow < fx.Int32(QG):
                    _qb = (
                        seq * c_sq + (kv_h * c_qg + _qrow) * c_qh + _wchk * fx.Int32(32)
                    )
                    # ---- load this lane's 32 raw-Q head-dims (4 x dwordx4) ----
                    _v = []
                    for _j in range_constexpr(4):
                        _raw = buffer_ops.buffer_load(
                            q_rsrc,
                            (_qb + fx.Int32(_j * 8)) // fx.Int32(2),
                            vec_width=4,
                            dtype=T.i32,
                        )
                        _rb = vector.bitcast(T.vec(8, T.bf16), _raw)
                        for _e in range_constexpr(8):
                            _v.append(
                                arith.extf(
                                    T.f32, vector.extract(_rb, static_position=[_e])
                                )
                            )
                    # ---- stages 0..4: in-register, no LDS, no cross-lane ----
                    for _s in range_constexpr(5):
                        _st = 1 << _s
                        _nx = list(_v)
                        for _g in range_constexpr(32 // (2 * _st)):
                            for _k in range_constexpr(_st):
                                _i0 = _g * 2 * _st + _k
                                _i1 = _i0 + _st
                                _a, _b = _v[_i0], _v[_i1]
                                _nx[_i0] = _a + _b
                                _nx[_i1] = _a - _b
                        _v = _nx
                    # ---- stages 5..7: chunk XOR 1/2/4 via ds_swizzle ----
                    # ds_swizzle_b32 offset (bit15=0, 32-lane groups):
                    #   [4:0]=and_mask [9:5]=or_mask [14:10]=xor_mask
                    # pure XOR k -> and_mask=0x1F, or_mask=0, xor_mask=k. The
                    # partner is always within the same 32-lane group (k<=4).
                    for _s in range_constexpr(3):
                        _k = 1 << _s
                        _swz = arith.constant((_k << 10) | 0x1F, type=T.i32)
                        # High half of the pair computes (partner - mine), the low
                        # half (mine + partner). Fold that into a single sign
                        # computed ONCE per stage, so each element costs one
                        # multiply-add rather than a branch or a per-element select.
                        _chk_bit = _wchk & fx.Int32(_k)
                        _is_hi = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            _chk_bit.ir_value()
                            if hasattr(_chk_bit, "ir_value")
                            else _chk_bit,
                            fx.Int32(0).ir_value(),
                        )
                        _sf = arith.select(
                            _is_hi,
                            arith.constant(-1.0, type=T.f32),
                            arith.constant(1.0, type=T.f32),
                        )
                        _nx = []
                        for _t in range_constexpr(32):
                            _mi = arith.bitcast(T.i32, _v[_t])
                            _pt = arith.bitcast(
                                T.f32, rocdl.ds_swizzle(T.i32, _mi, _swz)
                            )
                            _nx.append(_v[_t] * _sf + _pt)
                        _v = _nx
                    # ---- 1/sqrt(D) normalization + E4M3 haircut -> bf16 ----
                    # cvt_pk_fp8_f32 is the SAME instruction the QK operand build
                    # uses, so this reproduces the host haircut exactly. E4M3 is a
                    # subset of bf16, so the bf16 store below is lossless and every
                    # downstream consumer stays byte-identical.
                    _hc = []
                    for _p in range_constexpr(16):
                        _w = rocdl.cvt_pk_fp8_f32(
                            T.i32,
                            _v[2 * _p] * _inv_sqrtD,
                            _v[2 * _p + 1] * _inv_sqrtD,
                            fx.Int32(0),
                            0,
                        )
                        _u = rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), _w, 0)
                        _hc.append(
                            arith.truncf(
                                T.bf16, vector.extract(_u, static_position=[0])
                            )
                        )
                        _hc.append(
                            arith.truncf(
                                T.bf16, vector.extract(_u, static_position=[1])
                            )
                        )
                    # ---- permuted store ----
                    # ct(t) maps t=0..15 -> 0..15 and t=16..31 -> 32..47, i.e. TWO
                    # contiguous 16-element runs, so the scatter is still 4 aligned
                    # 16-byte vector stores rather than 32 scalar writes.
                    _qrow_elem = _qrow * fx.Int32(HEAD_SIZE)
                    for _half in range_constexpr(2):
                        for _q4 in range_constexpr(2):
                            _elems = [_hc[_half * 16 + _q4 * 8 + _e] for _e in range(8)]
                            _vec = vector.from_elements(T.vec(8, T.bf16), _elems)
                            _dst = (
                                _qrow_elem + _perm_base + fx.Int32(_half * 32 + _q4 * 8)
                            )
                            vector.store(
                                vector.bitcast(T.vec(4, T.i32), _vec),
                                q_lds_i32,
                                [arith.index_cast(T.index, _dst // fx.Int32(2))],
                            )
            gpu.barrier()

        # ===== STEP B: Load pre-rotated q_rot → row-major Q LDS ==========
        for c in range_constexpr(0 if _FUSE_QROT else QG_LOAD_ITERS):
            row_chunk = lane + fx.Int32(c * WARP_SIZE)
            row = row_chunk >> fx.Int32(Q_ROW_SHIFT)  # 0..QG-1
            col_b = row_chunk & fx.Int32(Q_COL_MASK)  # 0..Q_GROUPS_PER_ROW-1
            col_elem = col_b * fx.Int32(8)  # bf16 elem
            q_off_elem = seq * c_sq + (kv_h * c_qg + row) * c_qh + col_elem
            q_v = buffer_ops.buffer_load(
                q_rsrc,
                q_off_elem // fx.Int32(2),
                vec_width=4,
                dtype=T.i32,
            )
            q_lds_byte = row * fx.Int32(HEAD_SIZE * 2) + col_elem * fx.Int32(2)
            vector.store(
                q_v,
                q_lds_i32,
                [arith.index_cast(T.index, q_lds_byte // fx.Int32(4))],
            )
        gpu.barrier()

        # Scaled-MFMA Q operands (loop-invariant): one per MFMA issue.
        q_op_hoisted = []
        for h in range_constexpr(MFMA_ISSUES):
            _q_hoist_words = []
            for j in range_constexpr(4):
                _q_hoist_idx = (
                    mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                    + (fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp) * fx.Int32(8)
                    + fx.Int32(j * 2)
                )
                _q_hoist_v = vector.load_op(
                    T.vec(2, T.i64),
                    q_lds_i64,
                    [arith.index_cast(T.index, _q_hoist_idx)],
                )
                _q_hoist_words.append(
                    _bf16x8_to_fp8_i64(vector.bitcast(T.vec(8, T.bf16), _q_hoist_v))
                )
            q_op_hoisted.append(
                vector.bitcast(
                    T.vec(8, T.i32),
                    vector.from_elements(T.vec(4, T.i64), _q_hoist_words),
                )
            )

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = arith.constant_vector(0.0, T.f32x4)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = buffer_ops.buffer_load(sl_rsrc, seq, vec_width=1, dtype=T.i32)

        # Per-K-tile dequant lane assignment: lane t -> token = t/4,
        # chunk_in_tok = t%4 (each chunk = SUBCHUNK_HDIMS=64 head-dims = 2 groups).
        tok_in_tile = lane >> fx.Int32(2)
        chunk_in_tok = lane & fx.Int32(3)
        # Both UE8M0 group bytes for this lane's chunk (groups 2c, 2c+1) live in
        # the SAME 4-byte scale word at offset (chunk_in_tok>>1)*4 within the
        # 8-byte scale region. Load once; extract per-half below.
        c_scale_word_off = (chunk_in_tok >> fx.Int32(1)) * fx.Int32(4)

        # ===== STEP F: K-tile loop =======================================
        c_kcb = fx.Int32(KV_COMPUTE_BLOCK)
        c_tgpp = fx.Int32(TGPP)
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_nparts = fx.Int32(NUM_PARTS_C)
        total_tgs = (seq_len + c_kcb - c_one_i32) // c_kcb
        avail = total_tgs - part
        in_range = avail > c_zero_i32
        trip_raw = (avail + c_nparts - c_one_i32) // c_nparts
        trip_clamped = (trip_raw > c_tgpp).select(c_tgpp, trip_raw)
        trip_or_zero = in_range.select(trip_clamped, c_zero_i32)
        c_zero_idx = arith.constant(0, index=True)
        c_one_idx = arith.constant(1, index=True)
        trip_idx = arith.index_cast(
            T.index,
            trip_or_zero.ir_value()
            if hasattr(trip_or_zero, "ir_value")
            else trip_or_zero,
        )
        _init_iter = [
            _ival(running_max),
            _ival(running_sum),
            *[_ival(p) for p in acc_pv],
        ]
        _for_op = _scf.ForOp(
            c_zero_idx,
            trip_idx,
            c_one_idx,
            _init_iter,
        )
        _for_ip = ir.InsertionPoint(_for_op.body)
        _for_ip.__enter__()
        try:
            tg_idx = _for_op.induction_variable
            tg_i32 = fx.Int32(arith.index_cast(T.i32, tg_idx))
            partition_start = (part + tg_i32 * c_nparts) * c_kcb
            bt_seq_base = seq * c_bt + (partition_start // fx.Int32(_BS))
            running_max = _for_op.inner_iter_args[0]
            running_sum = _for_op.inner_iter_args[1]
            acc_pv = list(_for_op.inner_iter_args[2:])

            # ==== SOFTWARE PREFETCH =====================================
            # Emit ONLY the HBM buffer_loads for one tile (K codes, V codes,
            # K/V UE8M0 scale words) and return the loaded register values.
            # Emitting tile N+1's loads BEFORE tile N's dequant+MFMA lets the
            # vmem fetches overlap compute: the compiler's s_waitcnt for these
            # values lands at their first use one iteration later, hiding HBM
            # latency (the decode kernel is ~85% HBM-stalled at low batch).
            # This is bit-identical to the pre-prefetch path -- only the
            # *emission order* of independent loads changes, never any
            # arithmetic. All addressing locals (slot_base_byte, blk_rsrc,
            # ...) are consumed only inside this closure; the dequant below
            # needs just the four returned register values.
            def _emit_tile_loads(n_tile):
                # n_tile is a compile-time int on the single-warp path and a
                # runtime fx.Int32 on the multi-warp path (each warp strides the
                # 16 tiles). Compute the block/tile/slot geometry accordingly.
                if const_expr(isinstance(n_tile, int)):
                    block_in_part = fx.Int32(n_tile // _TILES_PER_BLOCK)
                    tile_in_block = n_tile % _TILES_PER_BLOCK
                    tile_start_tok = partition_start + fx.Int32(n_tile * TILE_SIZE)
                    slot = fx.Int32(tile_in_block * TILE_SIZE) + tok_in_tile
                else:
                    block_in_part = n_tile // fx.Int32(_TILES_PER_BLOCK)
                    tile_in_block = n_tile - block_in_part * fx.Int32(_TILES_PER_BLOCK)
                    tile_start_tok = partition_start + n_tile * fx.Int32(TILE_SIZE)
                    slot = tile_in_block * fx.Int32(TILE_SIZE) + tok_in_tile
                tile_in_seq = tile_start_tok < seq_len
                bt_off = bt_seq_base + block_in_part
                bt_off_safe = tile_in_seq.select(bt_off, seq * c_bt)
                phys_block = buffer_ops.buffer_load(
                    bt_rsrc,
                    bt_off_safe,
                    vec_width=1,
                    dtype=T.i32,
                )
                # A buffer descriptor addresses with a 32-bit voffset, so
                # ``phys_block * c_block`` wraps once the cache view exceeds
                # 4 GiB. Fold the block base into the descriptor's 64-bit
                # base pointer and keep only in-block offsets below.
                blk_off_i64 = arith.extui(T.i64, phys_block) * fx.Int64(
                    _stride_cache_block
                )
                blk_rsrc = buffer_ops.create_buffer_resource(
                    kv_cache_ptr,
                    max_size=True,
                    base_byte_offset=blk_off_i64,
                )
                block_base = fx.Int32(0)

                slot_base_byte = block_base + slot * c_stride_pos + kv_h * c_stride_head
                # K codes: LANE_CODE_BYTES (=32) at slot_base + chunk*32, in
                # HALVES x 16-byte buffer_loads.
                k_byte0 = slot_base_byte + chunk_in_tok * fx.Int32(LANE_CODE_BYTES)
                _kpl = []
                for hf in range_constexpr(HALVES):
                    _kpl.append(
                        buffer_ops.buffer_load(
                            blk_rsrc,
                            (k_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=_KV_AUX,
                        )
                    )

                # ---- V codes + K/V scale HBM loads ------------------------
                v_byte0 = (
                    slot_base_byte
                    + c_vcode_off
                    + chunk_in_tok * fx.Int32(LANE_CODE_BYTES)
                )
                _vpl = []
                for hf in range_constexpr(HALVES):
                    _vpl.append(
                        buffer_ops.buffer_load(
                            blk_rsrc,
                            (v_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=_KV_AUX,
                        )
                    )
                # UE8M0 group scale words (one 4-byte word covers this lane's
                # two groups). scale = 2^(byte-127) = bitcast_f32(byte<<23);
                # byte==0 -> +0.0 (zero sentinel).
                _ksw = buffer_ops.buffer_load(
                    blk_rsrc,
                    (slot_base_byte + c_kscale_off + c_scale_word_off) // fx.Int32(4),
                    vec_width=1,
                    dtype=T.i32,
                )
                _vsw = buffer_ops.buffer_load(
                    blk_rsrc,
                    (slot_base_byte + c_vscale_off + c_scale_word_off) // fx.Int32(4),
                    vec_width=1,
                    dtype=T.i32,
                )
                return (_kpl, _vpl, _ksw, _vsw)

            # Depth-1 software prefetch: emit the next tile's HBM loads before
            # consuming the current tile without changing arithmetic order.
            _PF_DEPTH = 1
            # Prime the pipeline with the first _PF_DEPTH tiles' loads.
            _pf_queue = []
            for _pi in range_constexpr(min(_PF_DEPTH, 16)):
                _pf_queue.append(_emit_tile_loads(_pi))
            _tile_range = range_constexpr(16)
            for _jt in _tile_range:
                n_tile = _jt
                _ntile_tok = fx.Int32(n_tile * TILE_SIZE)
                (k_packed_list, v_packed_list, kscale_word, vscale_word) = (
                    _pf_queue.pop(0)
                )
                _pf_nxt = n_tile + _PF_DEPTH
                if const_expr(_pf_nxt < 16):
                    _pf_queue.append(_emit_tile_loads(_pf_nxt))
                # Store raw FP4 codes (natural contiguous order,
                # 16 B = 32 nibbles per UE8M0 group) straight to KV LDS as
                # the scaled-MFMA A operand — no dequant here.
                for hf in range_constexpr(HALVES):
                    grp = chunk_in_tok * fx.Int32(2) + fx.Int32(hf)
                    grp_shift = (grp & fx.Int32(3)) * fx.Int32(8)
                    kscale_byte = (kscale_word >> grp_shift) & fx.Int32(0xFF)
                    k_packed = k_packed_list[hf]
                    vector.store(
                        k_packed,
                        kv_lds_i32,
                        [
                            arith.index_cast(
                                T.index,
                                _kvi32(
                                    tok_in_tile * fx.Int32(KFP4_ROW_I32)
                                    + grp * fx.Int32(4)
                                ),
                            )
                        ],
                    )
                    scale_lds_i32.store(
                        kscale_byte,
                        [
                            arith.index_cast(
                                T.index,
                                _sci(tok_in_tile * fx.Int32(N_GROUPS) + grp),
                            )
                        ],
                    )
                gpu.barrier()
                # QK: MFMA_ISSUES native scaled MFMAs over K=128,
                # chained through the accumulator.
                #   A = K raw FP4 codes (vec4 i32 = 16 B) per (token, group)
                #   B = Q E4M3 (vec8 i32 = 32 B) for the same group
                #   scaleA = per-(token,group) UE8M0 byte at op_sel byte 0
                #   scaleB = 0x7F identity (Q carries no scale)
                BLGP_E2M1 = 4  # A operand type code = fp4 e2m1
                CBSZ_E4M3 = 0  # B operand type code = fp8 e4m3
                IDENT = fx.Int32(0x7F)
                qk_acc = zero_v4
                for h in range_constexpr(MFMA_ISSUES):
                    grp = fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp
                    k_op = vector.load_op(
                        T.vec(4, T.i32),
                        kv_lds_i32,
                        [
                            arith.index_cast(
                                T.index,
                                _kvi32(
                                    mfma_row * fx.Int32(KFP4_ROW_I32)
                                    + grp * fx.Int32(4)
                                ),
                            )
                        ],
                    )
                    q_op = q_op_hoisted[h]
                    scbyte = fx.Int32(
                        scale_lds_i32.load(
                            [
                                arith.index_cast(
                                    T.index,
                                    _sci(mfma_row * fx.Int32(N_GROUPS) + grp),
                                )
                            ]
                        )
                    )
                    kscale = fx.Int32(0x7F7F7F00) | scbyte
                    qk_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32),
                        [
                            k_op,
                            q_op,
                            qk_acc,
                            BLGP_E2M1,
                            CBSZ_E4M3,
                            0,
                            kscale,
                            0,
                            IDENT,
                        ],
                    )

                # Scale + mask out-of-context tokens.
                qk_acc = _vsplat_mul(qk_acc, QK_SCALE)
                for elem in range_constexpr(4):
                    kv_tok = (
                        partition_start
                        + _ntile_tok
                        + mfma_col_grp * fx.Int32(4)
                        + fx.Int32(elem)
                    )
                    in_b = kv_tok < seq_len
                    v = vector.extract(qk_acc, static_position=[elem])
                    qk_acc = _vector_insert(
                        in_b.select(v, NEG_INF),
                        qk_acc,
                        static_position=[elem],
                    )

                # FA2 online softmax: per-query-row reduce.
                local_max = vector.reduction(T.f32, "maxnumf", qk_acc)
                r1 = local_max.shuffle_xor(fx.Int32(16), c_w)
                local_max = local_max.maximumf(r1)
                r2 = local_max.shuffle_xor(fx.Int32(32), c_w)
                tile_max = local_max.maximumf(r2)

                new_max = running_max.maximumf(tile_max)
                max_diff = running_max - new_max
                safe_diff = (running_max > NEG_INF).select(max_diff, ZERO_F)
                scale = (safe_diff * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                running_sum = running_sum * scale
                for h in range_constexpr(PV_N_CHUNKS):
                    acc_pv[h] = _vsplat_mul(acc_pv[h], scale)
                running_max = new_max

                tile_sum = ZERO_F
                for elem in range_constexpr(4):
                    s = vector.extract(qk_acc, static_position=[elem])
                    d = s - new_max
                    d = (new_max > NEG_INF).select(d, NEG_INF)
                    p = (d * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    tile_sum = tile_sum + p
                    qk_acc = _vector_insert(p, qk_acc, static_position=[elem])

                ts1 = tile_sum.shuffle_xor(fx.Int32(16), c_w)
                tile_sum = tile_sum + ts1
                ts2 = tile_sum.shuffle_xor(fx.Int32(32), c_w)
                tile_sum = tile_sum + ts2
                running_sum = running_sum + tile_sum
                # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose)
                # written as ds_write_b128 (vec(2,i64)); read back via the
                # ds_read_tr16_b64 HW transpose in the PV block below.
                v_lds_elem_base = tok_in_tile * fx.Int32(
                    KV_ROW_ELEMS
                ) + chunk_in_tok * fx.Int32(SUBCHUNK_HDIMS)
                for hf in range_constexpr(HALVES):
                    v_packed = v_packed_list[hf]
                    half_hd = fx.Int32(hf * HALF_HDIMS)
                    grp_shift = (
                        (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                    ) * fx.Int32(8)
                    vscale_byte = (vscale_word >> grp_shift) & fx.Int32(0xFF)
                    vscale_f32 = arith.bitcast(
                        T.f32, _ival(vscale_byte << fx.Int32(23))
                    )
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        # Native CDNA4 scaled convert: 2-wide
                        # cvt_scalef32_pk_bf16_fp4. srcSel s picks byte s
                        # of the word (nibbles 2s, 2s+1) → 2 bf16 with the
                        # UE8M0 group scale fused. 4 cvt calls/word.
                        bf16_elems = []
                        for s in range_constexpr(4):
                            v2 = rocdl.cvt_scalef32_pk_bf16_fp4(
                                T.vec(2, T.bf16),
                                _ival(word_i32),
                                _ival(vscale_f32),
                                int(s),
                            )
                            bf16_elems.append(vector.extract(v2, static_position=[0]))
                            bf16_elems.append(vector.extract(v2, static_position=[1]))
                        v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                        v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                        v_lds_i64_idx = (
                            v_lds_elem_base + half_hd + fx.Int32(w * 8)
                        ) // fx.Int32(4)
                        vector.store(
                            v_i64,
                            kv_lds_i64,
                            [arith.index_cast(T.index, _kvi64(v_lds_i64_idx))],
                        )
                # HW V transpose cross-lane LDS fence (see 128 kernel).
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(0xC07F)
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                p_bf16 = arith.trunc_f(T.vec(4, T.bf16), qk_acc)
                p_op = vector.bitcast(T.vec(4, T.i16), p_bf16)
                # HW-transpose PV: V_lds row-major V[token][head_dim].
                token_idx = lane >> fx.Int32(2)
                hd_sub = (lane & fx.Int32(3)) * fx.Int32(4)
                v_lane_byte = (
                    fx.Int32(kv_off)
                    + token_idx * fx.Int32(KV_ROW_BYTES)
                    + hd_sub * fx.Int32(2)
                )
                for h in range_constexpr(PV_N_CHUNKS):
                    v_byte_off = v_lane_byte + fx.Int32(h * 32)
                    v_byte_i64 = fx.Int64(v_byte_off)
                    v_ptr = buffer_ops.create_llvm_ptr(
                        v_byte_i64,
                        address_space=3,
                    )
                    v_op_raw = rocdl.ds_read_tr16_b64(
                        T.vec(4, T.i16),
                        v_ptr,
                    ).result
                    acc_pv[h] = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4, [v_op_raw, p_op, acc_pv[h], 0, 0, 0]
                    )

            _scf.YieldOp(
                [
                    _ival(running_max),
                    _ival(running_sum),
                    *[_ival(p) for p in acc_pv],
                ]
            )
        finally:
            _for_ip.__exit__(None, None, None)
        running_max = _for_op.results[0]
        running_sum = _for_op.results[1]
        acc_pv = list(_for_op.results[2:])

        # ===== STEP G: Output ===========================================
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        c_os = fx.Int32(_stride_out_seq)
        c_oh = fx.Int32(_stride_out_head)
        c_op_ = fx.Int32(_stride_out_part)
        out_base = seq * c_os + kv_h * c_oh + part * c_op_

        valid_row_pred = arith.cmpi(
            arith.CmpIPredicate.ult,
            mfma_row.ir_value() if hasattr(mfma_row, "ir_value") else mfma_row,
            arith.constant(QG, type=T.i32),
        )
        _if = _scf.IfOp(valid_row_pred)
        with ir.InsertionPoint(_if.then_block):
            for h in range_constexpr(PV_N_CHUNKS):
                pv_norm = _vsplat_mul(acc_pv[h], rcp)
                pv_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_norm)
                pv_i32x2 = vector.bitcast(T.vec(2, T.i32), pv_bf16)
                head_dim_start = fx.Int32(h * 16) + mfma_col_grp * fx.Int32(4)
                out_off_elem = (
                    out_base + mfma_row * fx.Int32(HEAD_SIZE) + head_dim_start
                )
                buffer_ops.buffer_store(
                    pv_i32x2,
                    out_rsrc,
                    out_off_elem * fx.Int32(2),
                    offset_is_bytes=True,
                )

            c_npq = fx.Int32(num_partitions * QG)
            ml_off = (
                seq * fx.Int32(_stride_ml_seq) + kv_h * c_npq + part * c_qg + mfma_row
            )
            es_off = (
                seq * fx.Int32(_stride_es_seq) + kv_h * c_npq + part * c_qg + mfma_row
            )
            buffer_ops.buffer_store(running_max, ml_rsrc, ml_off)
            buffer_ops.buffer_store(running_sum, es_rsrc, es_off)
            _scf.YieldOp([])

    # Hand the build-local allocator to the launcher. `_get_kernel` reads
    # `kmod.allocator` immediately after this returns (before the next build can
    # run), so each launcher captures its own variant's allocator. This does NOT
    # reintroduce the drift bug: the kernel above closes over the LOCAL
    # `allocator`, so no @flyc.jit function references module-global `allocator`
    # and it is never snapshotted by _check_globals_drift.
    globals()["allocator"] = allocator
    return ultraquant_decode_hd256_kernel
