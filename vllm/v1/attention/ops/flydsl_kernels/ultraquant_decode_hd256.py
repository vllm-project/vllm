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

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.v1.attention.ops.flydsl_kernels.buffer_utils import (
    buffer_view,
    load_vec,
    store_vec,
)
from vllm.v1.attention.ops.flydsl_kernels.mma_utils import mma

# === Constants (256-wide-head ultraquant decode profile) =======================
HEAD_SIZE = 256
KV_BLOCK_SIZE = 16  # default; overridable via build(kv_block_size=...)
TILE_SIZE = 16  # MFMA tile = 16 tokens
N_CENTROIDS = 16  # FP4 E2M1 value table (16 entries indexed by nibble)
QUERY_GROUP_SIZE = 16  # default for tests; overridable via build(...)
WARP_SIZE = 64
BLOCK_THREADS = WARP_SIZE
KV_COMPUTE_BLOCK = 256  # 16 K-tiles × 16 tokens
N_TILES = KV_COMPUTE_BLOCK // TILE_SIZE

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

# --- Per-lane data-movement geometry ----------------------------------------
# Each token's packed code bytes are split across the 4 ``chunk_in_tok``
# sub-lanes (lane % 4), 32 B per lane. A buffer load moves at most 16 B, so a
# lane issues ``HALVES`` = 2 loads, each covering one 32-head-dim UE8M0 group.
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

    # Cache policy for the read-once KV code stream. gfx9 buffer aux bits:
    # bit0=sc0, bit1=nt, bit4=sc1; 0 = no hint, 2 = nt, 18 = nt|sc1.
    _KV_AUX = 0

    # Q LDS holds only the QG real rows (13 KB instead of 17 KB per workgroup
    # at QG=8), which raises waves/CU for this LDS-occupancy-bound kernel.
    _Q_LDS_BYTES = QG * HEAD_SIZE * 2

    _BS = int(kv_block_size)
    _TILES_PER_BLOCK = _BS // TILE_SIZE
    _PADDED_SLOT = int(padded_slot)
    assert _PADDED_SLOT >= SLOT_CONTENT_BYTES, (
        f"padded_slot={_PADDED_SLOT} < {SLOT_CONTENT_BYTES} (ultraquant D=256 content)"
    )
    assert _PADDED_SLOT % 4 == 0, f"padded_slot={_PADDED_SLOT} must be 4-byte aligned"

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
    # Q and the KV tile share one allocation: for QG < 16 the Q reads of MFMA
    # rows QG..15 run past the Q rows into the KV tile (see the docstring).
    @fx.struct
    class SharedStorage:
        cent: fx.Array[fx.Float32, N_CENTROIDS]
        q_kv: fx.Array[fx.Int32, (_Q_LDS_BYTES + KV_TILE_LDS_BYTES_PADDED) // 4, 16]
        scale: fx.Array[fx.Int32, SCALE_LDS_BYTES // 4, 16]

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
        mfma_row = lane & 15
        mfma_col_grp = lane >> 4  # 0..3, K-group dim

        # ---- Buffer views -----------------------------------------------
        q_buf = buffer_view(query_ptr, fx.Int32, 4)
        bt_buf = buffer_view(block_tables_ptr, fx.Int32)
        sl_buf = buffer_view(seq_lens_ptr, fx.Int32)
        cent_buf = buffer_view(centroids_ptr, fx.Float32)
        out_buf = buffer_view(out_ptr, fx.BFloat16, 4)
        es_buf = buffer_view(exp_sums_ptr, fx.Float32)
        ml_buf = buffer_view(max_logits_ptr, fx.Float32)

        # ---- LDS pointers -----------------------------------------------
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        cent_lds = lds.cent
        scale_lds = lds.scale
        q_lds_i32 = lds.q_kv.ptr
        q_lds_i64 = fx.recast_iter(fx.Int64, q_lds_i32)
        kv_lds_i32 = q_lds_i32 + _Q_LDS_BYTES // 4
        kv_lds_i64 = fx.recast_iter(fx.Int64, kv_lds_i32)
        kv_lds_u8 = fx.recast_iter(fx.Uint8, kv_lds_i32)

        # ---- Constants ---------------------------------------------------
        NEG_INF = fx.Float32(NEG_INF_VAL)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = fx.Float32(LOG2E)
        QK_SCALE = fx.Float32(_qk_scale)
        qk_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float4E2M1FN, fx.Float8E4M3FN)
        )
        pv_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        def _f32x8_to_fp8_i64(f):
            # 8 f32 -> 8 E4M3 bytes packed as i64 (MFMA fp8 operand).
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[0], f[1], 0, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[2], f[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[4], f[5], 0, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[6], f[7], w1, 1)
            packed = fx.Vector.from_elements([w0, w1], dtype=fx.Int32)
            return packed.bitcast(fx.Int64)[0]

        def _bf16x8_to_fp8_i64(v_bf16):
            return _f32x8_to_fp8_i64([v_bf16[i].to(fx.Float32) for i in range(8)])

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & (N_CENTROIDS - 1)
        cent_lds[c_idx_safe] = fx.Float32(cent_buf[c_idx_safe])
        gpu.barrier()

        # ===== STEP B': in-kernel Q rotation (fused WHT) ==================
        # PiT is the orthonormal Sylvester Hadamard with a column permutation
        # folded in on the host, so q_rot = Q @ PiT is a Walsh-Hadamard
        # transform: 8 butterfly stages instead of a 256x256 matmul, cheap
        # enough to redo in every partition instead of a separate kernel.
        # After the E4M3 haircut it matches the matmul bit-for-bit.
        #
        # Lane map: lane L owns row r = L>>3 and the 32 CONSECUTIVE head-dims
        # d = (L&7)*32 + t. That puts butterfly stages 0..4 (strides 1..16)
        # entirely inside one lane's registers, and leaves only stages 5..7
        # (strides 32/64/128 == chunk XOR 1/2/4) needing a cross-lane exchange.
        if const_expr(_FUSE_QROT):
            _wrow = lane >> 3  # 0..7  row within the group
            _wchk = lane & 7  # 0..7  32-dim chunk
            # inv(qperm) is a pure bit permutation of the head-dim index:
            #   dest = (d & 0x8F) | ((d>>2)&0x10) | ((d<<1)&0x60)
            # which splits additively into a compile-time part in t and a
            # per-lane part in the chunk index (verified exhaustively):
            #   ct(t) = (t & 0x0F) | ((t & 0x10) << 1)
            #   rt(c) = ((c&2)<<3) | ((c&1)<<6) | ((c&4)<<5)
            _perm_base = ((_wchk & 2) << 3) | ((_wchk & 1) << 6) | ((_wchk & 4) << 5)
            _inv_sqrtD = fx.Float32(1.0 / math.sqrt(HEAD_SIZE))
            for _qit in range_constexpr((QG + 7) // 8):
                _qrow = _wrow + _qit * 8
                if _qrow < QG:
                    _qb = (
                        seq * _stride_q_seq
                        + (kv_h * QG + _qrow) * _stride_q_head
                        + _wchk * 32
                    )
                    # ---- load this lane's 32 raw-Q head-dims (4 x dwordx4) ----
                    _v = []
                    for _j in range_constexpr(4):
                        _raw = load_vec(q_buf, (_qb + _j * 8) // 2, 4)
                        _rb = _raw.bitcast(fx.BFloat16)
                        for _e in range_constexpr(8):
                            _v.append(_rb[_e].to(fx.Float32))
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
                        _swz = fx.Int32((_k << 10) | 0x1F).ir_value()
                        # High half of the pair computes (partner - mine), the low
                        # half (mine + partner). Fold that into a single sign
                        # computed ONCE per stage, so each element costs one
                        # multiply-add rather than a branch or a per-element select.
                        _sf = ((_wchk & _k) != 0).select(
                            fx.Float32(-1.0), fx.Float32(1.0)
                        )
                        _nx = []
                        for _t in range_constexpr(32):
                            _mi = _v[_t].bitcast(fx.Int32).ir_value()
                            _pt = fx.Int32(rocdl.ds_swizzle(T.i32, _mi, _swz)).bitcast(
                                fx.Float32
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
                            0,
                            0,
                        )
                        _u = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), _w, 0))
                        _hc.append(_u[0].to(fx.BFloat16))
                        _hc.append(_u[1].to(fx.BFloat16))
                    # ---- permuted store ----
                    # ct(t) maps t=0..15 -> 0..15 and t=16..31 -> 32..47, i.e. TWO
                    # contiguous 16-element runs, so the scatter is still 4 aligned
                    # 16-byte vector stores rather than 32 scalar writes.
                    _qrow_elem = _qrow * HEAD_SIZE
                    for _half in range_constexpr(2):
                        for _q4 in range_constexpr(2):
                            _elems = [_hc[_half * 16 + _q4 * 8 + _e] for _e in range(8)]
                            _vec = fx.Vector.from_elements(_elems, dtype=fx.BFloat16)
                            _dst = _qrow_elem + _perm_base + (_half * 32 + _q4 * 8)
                            (q_lds_i32 + _dst // 2).store(_vec.bitcast(fx.Int32))
            gpu.barrier()

        # ===== STEP B: Load pre-rotated q_rot → row-major Q LDS ==========
        for c in range_constexpr(0 if _FUSE_QROT else QG_LOAD_ITERS):
            row_chunk = lane + c * WARP_SIZE
            row = row_chunk >> Q_ROW_SHIFT  # 0..QG-1
            col_b = row_chunk & Q_COL_MASK  # 0..Q_GROUPS_PER_ROW-1
            col_elem = col_b * 8  # bf16 elem
            q_off_elem = (
                seq * _stride_q_seq + (kv_h * QG + row) * _stride_q_head + col_elem
            )
            q_v = load_vec(q_buf, q_off_elem // 2, 4)
            q_lds_byte = row * (HEAD_SIZE * 2) + col_elem * 2
            (q_lds_i32 + q_lds_byte // 4).store(q_v)
        gpu.barrier()

        # Scaled-MFMA Q operands (loop-invariant): one per MFMA issue.
        q_op_hoisted = []
        for h in range_constexpr(MFMA_ISSUES):
            _q_hoist_words = []
            for j in range_constexpr(4):
                _q_hoist_idx = (
                    mfma_row * (HEAD_SIZE * 2 // 8)
                    + (h * GRPS_PER_ISSUE + mfma_col_grp) * 8
                    + j * 2
                )
                _q_hoist_v = fx.Vector((q_lds_i64 + _q_hoist_idx).load(T.vec(2, T.i64)))
                _q_hoist_words.append(
                    _bf16x8_to_fp8_i64(_q_hoist_v.bitcast(fx.BFloat16))
                )
            q_op_hoisted.append(
                fx.Vector.from_elements(_q_hoist_words, dtype=fx.Int64).bitcast(
                    fx.Int32
                )
            )

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = fx.Vector.filled(4, 0.0, fx.Float32)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = fx.Int32(sl_buf[seq])

        # Per-K-tile dequant lane assignment: lane t -> token = t/4,
        # chunk_in_tok = t%4 (each chunk = SUBCHUNK_HDIMS=64 head-dims = 2 groups).
        tok_in_tile = lane >> 2
        chunk_in_tok = lane & 3
        # Both UE8M0 group bytes for this lane's chunk (groups 2c, 2c+1) live in
        # the SAME 4-byte scale word at offset (chunk_in_tok>>1)*4 within the
        # 8-byte scale region. Load once; extract per-half below.
        c_scale_word_off = (chunk_in_tok >> 1) * 4

        # ===== STEP F: K-tile loop =======================================
        total_tgs = (seq_len + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
        avail = total_tgs - part
        in_range = avail > 0
        trip_raw = (avail + NUM_PARTS_C - 1) // NUM_PARTS_C
        trip_clamped = (trip_raw > TGPP).select(TGPP, trip_raw)
        trip_or_zero = in_range.select(trip_clamped, 0)
        loop_init = [running_max, running_sum, *acc_pv]
        results = loop_init
        for tg_idx, state in range(0, trip_or_zero, 1, init=loop_init):  # type: ignore[call-overload]
            tg_i32 = fx.Int32(tg_idx)
            partition_start = (part + tg_i32 * NUM_PARTS_C) * KV_COMPUTE_BLOCK
            bt_seq_base = seq * _stride_bt_seq + (partition_start // _BS)
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            acc_pv = [fx.Vector(v) for v in state[2:]]

            # ==== SOFTWARE PREFETCH =====================================
            # Emit ONLY the HBM buffer_loads for one tile (K codes, V codes,
            # K/V UE8M0 scale words) and return the loaded register values.
            # Emitting tile N+1's loads BEFORE tile N's dequant+MFMA lets the
            # vmem fetches overlap compute: the compiler's s_waitcnt for these
            # values lands at their first use one iteration later, hiding HBM
            # latency (the decode kernel is ~85% HBM-stalled at low batch).
            # This is bit-identical to the pre-prefetch path -- only the
            # *emission order* of independent loads changes, never any
            # arithmetic. All addressing locals (slot_base_byte, blk_buf,
            # ...) are consumed only inside this closure; the dequant below
            # needs just the four returned register values.
            def _emit_tile_loads(n_tile, partition_start, bt_seq_base):
                block_in_part = n_tile // _TILES_PER_BLOCK
                tile_in_block = n_tile % _TILES_PER_BLOCK
                tile_start_tok = partition_start + n_tile * TILE_SIZE
                slot = tok_in_tile + tile_in_block * TILE_SIZE
                tile_in_seq = tile_start_tok < seq_len
                bt_off = bt_seq_base + block_in_part
                bt_off_safe = tile_in_seq.select(bt_off, seq * _stride_bt_seq)
                phys_block = fx.Int32(bt_buf[bt_off_safe])
                # A buffer descriptor addresses with a 32-bit voffset, so
                # ``phys_block * c_block`` wraps once the cache view exceeds
                # 4 GiB. Fold the block base into the descriptor's 64-bit
                # base pointer and keep only in-block offsets below.
                blk_off_i64 = fx.Int64(fx.Uint32(phys_block)) * _stride_cache_block
                blk_buf4 = buffer_view(kv_cache_ptr, fx.Int32, 4, blk_off_i64)
                blk_buf = buffer_view(kv_cache_ptr, fx.Int32, 1, blk_off_i64)

                slot_base_byte = slot * _stride_cache_pos + kv_h * _stride_cache_head
                # K codes: LANE_CODE_BYTES (=32) at slot_base + chunk*32, in
                # HALVES x 16-byte buffer_loads.
                k_byte0 = slot_base_byte + chunk_in_tok * LANE_CODE_BYTES
                _kpl = []
                for hf in range_constexpr(HALVES):
                    _kpl.append(
                        load_vec(blk_buf4, (k_byte0 + hf * 16) // 4, 4, _KV_AUX)
                    )

                # ---- V codes + K/V scale HBM loads ------------------------
                v_byte0 = (
                    slot_base_byte + V_CODES_OFFSET + chunk_in_tok * LANE_CODE_BYTES
                )
                _vpl = []
                for hf in range_constexpr(HALVES):
                    _vpl.append(
                        load_vec(blk_buf4, (v_byte0 + hf * 16) // 4, 4, _KV_AUX)
                    )
                # UE8M0 group scale words (one 4-byte word covers this lane's
                # two groups). scale = 2^(byte-127) = bitcast_f32(byte<<23);
                # byte==0 -> +0.0 (zero sentinel).
                _ksw = fx.Uint32(
                    blk_buf[(slot_base_byte + K_SCALES_OFFSET + c_scale_word_off) // 4]
                )
                _vsw = fx.Uint32(
                    blk_buf[(slot_base_byte + V_SCALES_OFFSET + c_scale_word_off) // 4]
                )
                return (_kpl, _vpl, _ksw, _vsw)

            _PF_DEPTH = 1
            # Prime the pipeline with the first _PF_DEPTH tiles' loads.
            _pf_queue = []
            for _pi in range_constexpr(min(_PF_DEPTH, N_TILES)):
                _pf_queue.append(_emit_tile_loads(_pi, partition_start, bt_seq_base))
            for n_tile in range_constexpr(N_TILES):
                _ntile_tok = n_tile * TILE_SIZE
                (k_packed_list, v_packed_list, kscale_word, vscale_word) = (
                    _pf_queue.pop(0)
                )
                _pf_nxt = n_tile + _PF_DEPTH
                if const_expr(_pf_nxt < N_TILES):
                    _pf_queue.append(
                        _emit_tile_loads(_pf_nxt, partition_start, bt_seq_base)
                    )
                # Store raw FP4 codes (natural contiguous order,
                # 16 B = 32 nibbles per UE8M0 group) straight to KV LDS as
                # the scaled-MFMA A operand — no dequant here.
                for hf in range_constexpr(HALVES):
                    grp = chunk_in_tok * 2 + hf
                    grp_shift = (grp & 3) * 8
                    kscale_byte = (kscale_word >> grp_shift) & 0xFF
                    (kv_lds_i32 + (tok_in_tile * KFP4_ROW_I32 + grp * 4)).store(
                        k_packed_list[hf]
                    )
                    scale_lds[tok_in_tile * N_GROUPS + grp] = kscale_byte
                gpu.barrier()
                # QK: MFMA_ISSUES native scaled MFMAs over K=128,
                # chained through the accumulator.
                #   A = K raw FP4 codes (vec4 i32 = 16 B) per (token, group)
                #   B = Q E4M3 (vec8 i32 = 32 B) for the same group
                #   scaleA = per-(token,group) UE8M0 byte at op_sel byte 0
                #   scaleB = 0x7F identity (Q carries no scale)
                IDENT = fx.Int32(0x7F)
                qk_acc = zero_v4
                for h in range_constexpr(MFMA_ISSUES):
                    grp = mfma_col_grp + h * GRPS_PER_ISSUE
                    k_op = (kv_lds_i32 + (mfma_row * KFP4_ROW_I32 + grp * 4)).load(
                        T.vec(4, T.i32)
                    )
                    q_op = q_op_hoisted[h]
                    scbyte = scale_lds[mfma_row * N_GROUPS + grp]
                    kscale = scbyte | 0x7F7F7F00
                    qk_acc = mma(
                        qk_atom, qk_acc, k_op, q_op, scale_a=kscale, scale_b=IDENT
                    )

                # Scale + mask out-of-context tokens.
                qk_acc = qk_acc * QK_SCALE
                masked = []
                for elem in range_constexpr(4):
                    kv_tok = partition_start + _ntile_tok + mfma_col_grp * 4 + elem
                    masked.append((kv_tok < seq_len).select(qk_acc[elem], NEG_INF))
                qk_acc = fx.Vector.from_elements(masked, dtype=fx.Float32)

                # FA2 online softmax: per-query-row reduce.
                local_max = qk_acc.reduce(fx.ReductionOp.MAX)
                r1 = local_max.shuffle_xor(16, WARP_SIZE)
                local_max = fx.max(local_max, r1)
                r2 = local_max.shuffle_xor(32, WARP_SIZE)
                tile_max = fx.max(local_max, r2)

                new_max = fx.max(running_max, tile_max)
                max_diff = running_max - new_max
                safe_diff = (running_max > NEG_INF).select(max_diff, ZERO_F)
                scale = (safe_diff * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                running_sum = running_sum * scale
                for h in range_constexpr(PV_N_CHUNKS):
                    acc_pv[h] = acc_pv[h] * scale
                running_max = new_max

                tile_sum = ZERO_F
                probs = []
                for elem in range_constexpr(4):
                    d = qk_acc[elem] - new_max
                    d = (new_max > NEG_INF).select(d, NEG_INF)
                    p = (d * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    tile_sum = tile_sum + p
                    probs.append(p)
                qk_acc = fx.Vector.from_elements(probs, dtype=fx.Float32)

                ts1 = tile_sum.shuffle_xor(16, WARP_SIZE)
                tile_sum = tile_sum + ts1
                ts2 = tile_sum.shuffle_xor(32, WARP_SIZE)
                tile_sum = tile_sum + ts2
                running_sum = running_sum + tile_sum
                # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose)
                # written as ds_write_b128 (vec(2,i64)); read back via the
                # ds_read_tr16_b64 HW transpose in the PV block below.
                v_lds_elem_base = (
                    tok_in_tile * KV_ROW_ELEMS + chunk_in_tok * SUBCHUNK_HDIMS
                )
                for hf in range_constexpr(HALVES):
                    v_packed = fx.Vector(v_packed_list[hf])
                    grp_shift = ((chunk_in_tok * 2 + hf) & 3) * 8
                    vscale_byte = (vscale_word >> grp_shift) & 0xFF
                    vscale_f32 = fx.Int32(vscale_byte << 23).bitcast(fx.Float32)
                    for w in range_constexpr(4):
                        # Native CDNA4 scaled convert: 2-wide
                        # cvt_scalef32_pk_bf16_fp4. srcSel s picks byte s
                        # of the word (nibbles 2s, 2s+1) → 2 bf16 with the
                        # UE8M0 group scale fused. 4 cvt calls/word.
                        bf16_elems = []
                        for s in range_constexpr(4):
                            v2 = fx.Vector(
                                rocdl.cvt_scalef32_pk_bf16_fp4(
                                    T.vec(2, T.bf16), v_packed[w], vscale_f32, s
                                )
                            )
                            bf16_elems.append(v2[0])
                            bf16_elems.append(v2[1])
                        v_bf16 = fx.Vector.from_elements(bf16_elems, dtype=fx.BFloat16)
                        v_lds_i64_idx = (v_lds_elem_base + hf * HALF_HDIMS + w * 8) // 4
                        (kv_lds_i64 + v_lds_i64_idx).store(v_bf16.bitcast(fx.Int64))
                # HW V transpose cross-lane LDS fence (see tq_decode.py).
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                p_op = qk_acc.to(fx.BFloat16)
                # HW-transpose PV: V_lds row-major V[token][head_dim].
                token_idx = lane >> 2
                hd_sub = (lane & 3) * 4
                v_lane_byte = token_idx * KV_ROW_BYTES + hd_sub * 2
                for h in range_constexpr(PV_N_CHUNKS):
                    v_ptr = fx.to_llvm_ptr(kv_lds_u8 + (v_lane_byte + h * 32))
                    v_op_raw = rocdl.ds_read_tr16_b64(
                        T.vec(4, T.i16),
                        v_ptr,
                    ).result
                    v_op = fx.Vector(v_op_raw).bitcast(fx.BFloat16)
                    acc_pv[h] = mma(pv_atom, acc_pv[h], v_op, p_op)

            results = yield [running_max, running_sum, *acc_pv]
        running_max = fx.Float32(results[0])
        running_sum = fx.Float32(results[1])
        acc_pv = [fx.Vector(v) for v in results[2:]]

        # ===== STEP G: Output ===========================================
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        out_base = (
            seq * _stride_out_seq + kv_h * _stride_out_head + part * _stride_out_part
        )

        if mfma_row < QG:
            for h in range_constexpr(PV_N_CHUNKS):
                pv_bf16x4 = (acc_pv[h] * rcp).to(fx.BFloat16)
                head_dim_start = mfma_col_grp * 4 + h * 16
                out_off_elem = out_base + mfma_row * HEAD_SIZE + head_dim_start
                store_vec(out_buf, out_off_elem, pv_bf16x4, 4)

            ml_off = (
                seq * _stride_ml_seq
                + kv_h * (num_partitions * QG)
                + part * QG
                + mfma_row
            )
            es_off = (
                seq * _stride_es_seq
                + kv_h * (num_partitions * QG)
                + part * QG
                + mfma_row
            )
            ml_buf[ml_off] = running_max
            es_buf[es_off] = running_sum

    return ultraquant_decode_hd256_kernel
