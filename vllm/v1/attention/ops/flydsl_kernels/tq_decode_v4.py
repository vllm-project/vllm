# SPDX-License-Identifier: Apache-2.0
# TurboQuant decode v4 (FlyDSL) — MI355X / gfx950 / CDNA4
#
# Decode-only paged attention over a TurboQuant SoA KV cache.
# Targets Qwen-class models: HEAD_SIZE=128, GQA group=8, MSE_BITS=4 K, VQB=4 V,
# N_CENTROIDS=16, BLOCK_SIZE=16, partitioned mode.
#
# === Design (v4.1) ===
# * 1 wave (64 lanes) per CTA — simpler than 4-warp; trades parallelism for
#   correctness/debug-ability. v4.4 will fan out to 4 warps.
# * Flash-Attention-2 online softmax: rescale running PV/sum per K-tile,
#   no QK score storage across tiles, no P-LDS round-trip (qk_acc layout
#   matches PV B-operand directly).
# * Centroids loaded once into LDS at kernel start (LDS-resident LUT).
# * K dequant → LDS [token, head_dim] (natural layout)
# * V dequant → LDS [head_dim, token] (TRANSPOSED — required so PV MFMA's
#   A=V operand can be loaded as 4-bf16 contiguous via 1 ds_read_b64).
#
# MFMA layouts used:
#   QK:  mfma(A=K, B=Q, C=qk_acc)    →  C[m=token, n=query],  4 fp32/lane
#                                         = 4 tokens for SAME query → softmax via
#                                           xor_shuffle(16),(32) reduces per-row
#   PV:  mfma(A=V_T, B=P, C=acc_pv)  →  C[m=head_dim, n=query], 4 fp32/lane
#                                         = 4 contiguous head_dim cols for SAME
#                                           query → 1 buffer_store of 8 bytes
#                                           per chunk per lane

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,  # noqa: F401  kept for kernel-DSL API parity
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import Int32, T  # noqa: F401  Int32 via fx.Int32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# === Constants (Qwen-class TQ decode profile) ===============================
HEAD_SIZE = 128
# default; overridable via build_tq_decode_v4_module(kv_block_size=...)
KV_BLOCK_SIZE = 16
# MFMA tile = 16 tokens (do not change without re-deriving MFMA shapes)
TILE_SIZE = 16
N_CENTROIDS = 16
# default for tests; overridable via build_tq_decode_v4_module(query_group_size=...)
QUERY_GROUP_SIZE = 16
WARP_SIZE = 64
NUM_WARPS = 1
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
KV_COMPUTE_BLOCK = 256  # 16 K-tiles × 16 tokens

# TQ SoA layout
NUM_SOA_FIELDS = 3
SOA_K_NORM = 0
SOA_V_SCALE = 1
SOA_V_ZERO = 2
KEY_DATA_BYTES = HEAD_SIZE // 2
VAL_DATA_BYTES = HEAD_SIZE // 2
DATA_BYTES_PER_SLOT = KEY_DATA_BYTES + VAL_DATA_BYTES

# MFMA
MFMA_M = MFMA_N = 16
MFMA_K_BF16_QK = 32  # CDNA4 wide-K (mfma_f32_16x16x32_bf16)
MFMA_K_BF16_PV = 16  # PV's K = token tile = 16
QK_K_CHUNKS = HEAD_SIZE // MFMA_K_BF16_QK  # 4 (down from 8)
PV_N_CHUNKS = HEAD_SIZE // MFMA_N  # 8

# LDS regions
CENTROID_LDS_BYTES = N_CENTROIDS * 4  # 64
Q_LDS_BYTES = QUERY_GROUP_SIZE * HEAD_SIZE * 2  # 4096
KV_TILE_LDS_BYTES = TILE_SIZE * HEAD_SIZE * 2  # 4096

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, "ir_value") else scalar
    return vec * vector.broadcast(T.f32x4, s)


allocator = None


def build_tq_decode_v4_module(
    num_seqs: int,
    num_kv_heads: int,
    num_partitions: int,
    max_blocks_per_seq: int = 512,
    softmax_scale: float | None = None,
    query_group_size: int = QUERY_GROUP_SIZE,
    kv_block_size: int = KV_BLOCK_SIZE,
    use_hw_v_transpose: bool = False,
    tile_groups_per_partition: int = 1,
    use_wht_butterfly: bool = False,
):
    """Build a TQ decode v4 kernel module.

    ``query_group_size`` (= num_query_heads // num_kv_heads) selects the
    GQA factor. Supported values: 8 and 16. For 8 the MFMA's 16-row
    capacity is half-used; lanes 8..15 compute garbage and are gated out
    of all global-memory writes via an OOB-offset trick on the buffer
    resource. Q load iterates ``QG // 4`` chunks (= 2 for QG=8, 4 for
    QG=16) so we never read past the real Q rows in HBM.

    ``kv_block_size`` is the number of tokens per vLLM cache block. Must
    be a multiple of TILE_SIZE (=16). Supported values: 16 and 32. With
    block_size=32, two MFMA K-tiles fit in each block; the kernel walks
    block-table entries every ``kv_block_size // TILE_SIZE`` tiles and
    re-uses the entry for sub-tiles within the same block. SoA metadata
    region grows linearly with block_size.

    ``use_hw_v_transpose`` (CDNA4 / gfx950 only) replaces the 32-element
    scattered ``ds_write_b16`` V-dequant pattern with a row-major
    ``V[token][head_dim]`` LDS layout written as 4 wide ``ds_write_b128``
    per lane, then read back into the MFMA A operand via the hardware
    ``ds_read_tr16_b64`` transpose. This eliminates ~28 ds_write
    instructions per lane per K-tile (32 -> 4) and 8 strided ds_read_b64
    per PV chunk (replaces them with 1 hw-transpose read each). Pattern
    derived from ``flash_attn_func.py`` (USE_HW_TR=True path).

    ``tile_groups_per_partition`` (default 1) controls the FA-2 split-K
    granularity. With value G each partition processes ``G * 16`` K-tiles
    = ``G * KV_COMPUTE_BLOCK`` tokens, with the existing 16-tile loop
    body iterated G times. The launcher uses this to bound
    ``num_partitions`` (e.g. cap at 32) for long context: at 32K /
    block_size=32 / num_partitions=32, G=5 covers 32*5*256 = 40 960
    tokens of worst-case context with grid.z=32 instead of 256.
    Behavior at G=1 is bit-identical to the pre-Option-A kernel.

    ``use_wht_butterfly`` (default False) replaces the STEP B HBM load of
    the externally-rotated ``q_rot`` tensor with an in-register 7-stage
    Walsh-Hadamard butterfly that computes ``H @ q`` directly (H = the
    normalised Hadamard matrix = PiT for TurboQuant).  The launcher must
    pass the raw ``query`` tensor instead of ``q_rot`` when this is True.
    Gate: ``VLLM_TQ_FLYDSL_WHT_BUTTERFLY=1``.
    """
    assert query_group_size in (8, 16), (
        f"query_group_size must be 8 or 16; got {query_group_size}"
    )
    assert kv_block_size in (16, 32), (
        f"kv_block_size must be 16 or 32; got {kv_block_size}"
    )
    assert kv_block_size % TILE_SIZE == 0
    assert int(tile_groups_per_partition) >= 1, (
        f"tile_groups_per_partition must be >= 1; got {tile_groups_per_partition}"
    )
    USE_HW_TR = bool(use_hw_v_transpose)
    TGPP = int(tile_groups_per_partition)
    PARTITION_EXTENT_TOKENS = TGPP * KV_COMPUTE_BLOCK

    QG = int(query_group_size)
    QG_LOAD_ITERS = QG // 4  # 2 for QG=8, 4 for QG=16
    OOB_OFFSET = 0x7FFFFFF0  # noqa: F841  ~2GB byte offset; > any plausible buffer

    _BS = int(kv_block_size)
    _TILES_PER_BLOCK = _BS // TILE_SIZE  # 1 for BS=16, 2 for BS=32

    global allocator
    arch = get_hip_arch()

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE**0.5)
    _qk_scale = float(softmax_scale)

    # --- Strides ---
    _Hq = num_kv_heads * QG
    _stride_q_seq = _Hq * HEAD_SIZE
    _stride_q_head = HEAD_SIZE
    _stride_bt_seq = max_blocks_per_seq

    _data_region_bytes = _BS * num_kv_heads * DATA_BYTES_PER_SLOT
    _meta_region_bytes = num_kv_heads * NUM_SOA_FIELDS * _BS * 2
    _stride_cache_block = _data_region_bytes + _meta_region_bytes
    _meta_region_offset = _data_region_bytes

    _stride_out_part = QG * HEAD_SIZE
    _stride_out_head = num_partitions * QG * HEAD_SIZE
    _stride_out_seq = num_kv_heads * num_partitions * QG * HEAD_SIZE
    _stride_es_seq = num_kv_heads * num_partitions * QG
    _stride_ml_seq = _stride_es_seq

    # --- LDS layout ---
    allocator = SmemAllocator(None, arch=arch, global_sym_name="tq_v4_smem")
    centroid_off = 0
    allocator.ptr = CENTROID_LDS_BYTES
    q_off = allocator.ptr
    allocator.ptr += Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES

    @flyc.kernel
    def tq_decode_v4_kernel(
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
        mfma_row = lane & fx.Int32(15)  # = query_row (B's N) /
        #   token (A's M for QK)
        mfma_col_grp = lane >> fx.Int32(4)  # 0..3, K-group dim

        # ---- Buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_cache_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        sl_rsrc = buffer_ops.create_buffer_resource(seq_lens_ptr, max_size=True)
        cent_rsrc = buffer_ops.create_buffer_resource(centroids_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        cent_lds = SmemPtr(base, centroid_off, T.f32, shape=(N_CENTROIDS,))
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(Q_LDS_BYTES // 8,)).get()
        kv_lds_i32 = SmemPtr(base, kv_off, T.i32, shape=(KV_TILE_LDS_BYTES // 4,)).get()  # noqa: F841
        kv_lds_i64 = SmemPtr(base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES // 8,)).get()
        kv_lds_i16 = SmemPtr(base, kv_off, T.i16, shape=(KV_TILE_LDS_BYTES // 2,)).get()

        # ---- Constants ---------------------------------------------------
        c_sq = fx.Int32(_stride_q_seq)
        c_qh = fx.Int32(_stride_q_head)
        c_qg = fx.Int32(QG)
        c_bt = fx.Int32(_stride_bt_seq)
        c_block = fx.Int32(_stride_cache_block)
        c_meta_off = fx.Int32(_meta_region_offset)
        c_data_per_slot = fx.Int32(DATA_BYTES_PER_SLOT)
        c_kv_heads = fx.Int32(num_kv_heads)
        c_keydata = fx.Int32(KEY_DATA_BYTES)
        c_w = fx.Int32(WARP_SIZE)

        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)

        # Helper: unwrap fx wrapper → raw ir.Value (used in STEP B' and STEP F)
        def _ival(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & fx.Int32(N_CENTROIDS - 1)
        c_val = buffer_ops.buffer_load(cent_rsrc, c_idx_safe, vec_width=1, dtype=T.f32)
        cent_lds.store(c_val, [arith.index_cast(T.index, c_idx_safe)])
        gpu.barrier()

        # ===== STEP B / B': Load Q → row-major LDS [query_row, head_dim] ===
        # Two paths controlled by the use_wht_butterfly compile-time flag:
        #
        # STEP B  (use_wht_butterfly=False, default):
        #   q_rsrc points to the externally-rotated q_rot tensor.  Load 8 bf16
        #   per lane per iter (vec_width=4 i32) in QG_LOAD_ITERS iterations.
        #
        # STEP B' (use_wht_butterfly=True):
        #   q_rsrc points to the RAW query tensor.  For each GQA head h, lane i
        #   loads elements [2*i, 2*i+1], applies a 7-stage Hadamard butterfly
        #   (1 intra-lane + 6 cross-lane shuffle_xor) to compute H @ q in
        #   register, scales by 1/sqrt(D), and writes the rotated pair to Q_LDS.
        #   No external GEMM or HBM q_rot tensor needed.
        if not use_wht_butterfly:
            # --- STEP B: load pre-rotated q_rot → Q_LDS ---
            for c in range_constexpr(QG_LOAD_ITERS):
                row_chunk = lane + fx.Int32(c * WARP_SIZE)  # 0..QG*16-1
                row = row_chunk >> fx.Int32(4)  # 0..QG-1
                col_b = row_chunk & fx.Int32(15)  # 0..15
                col_elem = col_b * fx.Int32(8)  # 0..120 in bf16
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
        else:
            # --- STEP B': in-register FWHT → Q_LDS ---
            # PiT = H (pure normalised Hadamard, symmetric) so H @ q = q @ H.
            # Lane i holds elements [2*i, 2*i+1] of one head at a time.
            # After all QG heads + barrier, Q_LDS is identical to what STEP B
            # would have produced from an externally pre-rotated q_rot tensor.
            _WHT_SCALE = arith.constant(1.0 / (HEAD_SIZE**0.5), type=T.f32)
            _ONE_F32_I32 = arith.constant(0x3F800000, type=T.i32)  # +1.0 bits
            _C16 = arith.constant(16, type=T.i32)
            _C31 = arith.constant(31, type=T.i32)
            _q_lds_smem = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,))

            for h in range_constexpr(QG):
                # -- Load 2 packed bf16 (= 1 i32) for this lane / head --
                _q_elem_off = (
                    seq * c_sq + (kv_h * c_qg + fx.Int32(h)) * c_qh + lane * fx.Int32(2)
                )
                _q_raw = buffer_ops.buffer_load(
                    q_rsrc,
                    _q_elem_off // fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )
                # Unpack i32 → 2 × bf16 → 2 × f32
                _lo_i16 = arith.trunci(T.i16, _q_raw)
                _hi_i16 = arith.trunci(T.i16, arith.shrui(_q_raw, _C16))
                _q_lo = arith.extf(T.f32, arith.bitcast(T.bf16, _lo_i16))
                _q_hi = arith.extf(T.f32, arith.bitcast(T.bf16, _hi_i16))

                # Stage 0: intra-lane butterfly (no shuffle required)
                _a = _q_lo + _q_hi
                _b = _q_lo - _q_hi
                _q_lo = _a
                _q_hi = _b

                # Stages 1-6: cross-lane butterfly (shuffle_xor + branchless ±)
                # Sylvester-Hadamard core [[1,1],[1,-1]] per pair:
                #   low  (lane_bit=0):  y = self + other
                #   high (lane_bit=1):  y = other - self
                # i.e. result = sign * self + other  where sign ∈ {+1, -1}.
                # The sign MUST be applied to `self`, not `other`; applying it
                # to `other` gives high = self - other = -(correct), flipping
                # the transform by a per-coordinate sign vs. the pure Hadamard
                # PiT used to rotate K -> decorrelated Q·K scores -> garbage.
                # sign = bitcast_f32(float_bits(1.0) XOR (lane_bit << 31))
                for _log2m in range_constexpr(6):  # masks 1, 2, 4, 8, 16, 32
                    _mask = 1 << _log2m
                    _other_lo = _q_lo.shuffle_xor(fx.Int32(_mask), c_w)
                    _other_hi = _q_hi.shuffle_xor(fx.Int32(_mask), c_w)
                    # lane_bit = 0 if this lane is "low" in the pair, else 1
                    _lane_bit = _ival((lane >> fx.Int32(_log2m)) & fx.Int32(1))
                    _sign_f32 = arith.bitcast(
                        T.f32,
                        arith.xori(
                            _ONE_F32_I32,
                            arith.shli(_lane_bit, _C31),
                        ),
                    )
                    _q_lo = _sign_f32 * _q_lo + _other_lo
                    _q_hi = _sign_f32 * _q_hi + _other_hi

                # Scale by 1/sqrt(HEAD_SIZE)
                _q_lo = _q_lo * _WHT_SCALE
                _q_hi = _q_hi * _WHT_SCALE

                # Pack 2 × f32 → 2 × bf16 → 1 × i32
                _lo_bf16_out = arith.truncf(T.bf16, _ival(_q_lo))
                _hi_bf16_out = arith.truncf(T.bf16, _ival(_q_hi))
                _lo_i32_out = arith.extui(T.i32, arith.bitcast(T.i16, _lo_bf16_out))
                _hi_i32_out = arith.extui(T.i32, arith.bitcast(T.i16, _hi_bf16_out))
                _packed = arith.ori(
                    _lo_i32_out,
                    arith.shli(_hi_i32_out, _C16),
                )

                # Write to Q_LDS: row h, cols [2*lane, 2*lane+1]
                _lds_i32_idx = fx.Int32(h * HEAD_SIZE // 2) + lane
                _q_lds_smem.store(
                    _packed,
                    [arith.index_cast(T.index, _ival(_lds_i32_idx))],
                )
        gpu.barrier()

        # ===== STEP C: Pre-load Q operands for QK_K_CHUNKS = 4 K-chunks ===
        # Wide-K MFMA: K_per_chunk = 32, K_per_lane = 32/4 = 8 bf16 per chunk.
        # Lane t holds 8 bf16 at row=mfma_row, K-cols
        #   chunk*32 + mfma_col_grp*8 + 0..7
        # Byte addr = mfma_row * HEAD_SIZE*2 + (chunk*32 + col_grp*8)*2
        #          = mfma_row * 256 + chunk*64 + col_grp*16
        # i64 idx  = mfma_row * 32 + chunk*8 + col_grp*2 (each i64 = 4 bf16)
        # Load 16 bytes (= 8 bf16 = 2 i64) per chunk via vec(2, i64).
        q_chunks = []
        for chk in range_constexpr(QK_K_CHUNKS):
            q_idx_i64 = (
                mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                + fx.Int32(chk * 8)
                + mfma_col_grp * fx.Int32(2)
            )
            qv = vector.load_op(
                T.vec(2, T.i64),
                q_lds_i64,
                [arith.index_cast(T.index, q_idx_i64)],
            )
            q_chunks.append(vector.bitcast(T.vec(8, T.bf16), qv))

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = arith.constant_vector(0.0, T.f32x4)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = buffer_ops.buffer_load(sl_rsrc, seq, vec_width=1, dtype=T.i32)
        # ── Option A: bounded num_partitions + internal looping ─────────
        # With TGPP > 1 this CTA owns a contiguous slice of length
        # ``PARTITION_EXTENT_TOKENS`` that is iterated as ``TGPP`` groups
        # of 16 K-tiles each. ``partition_start`` is recomputed per
        # outer ``tg`` iteration below; the FA-2 online-softmax state
        # (running_max, running_sum, acc_pv) accumulates across all
        # ``TGPP * 16`` tiles for this CTA — semantically identical to
        # processing one large 16*TGPP-tile partition.
        partition_base = part * fx.Int32(PARTITION_EXTENT_TOKENS)

        # Per-K-tile dequant lane assignment:
        # 16 tokens × 64 packed bytes per token = 1024 bytes
        # 64 lanes × 16 bytes/lane = 1024 ✓
        # Lane t: token = t / 4, sub-chunk = t % 4 (0..3) → 16 bytes contiguous
        tok_in_tile = lane >> fx.Int32(2)
        chunk_in_tok = lane & fx.Int32(3)
        # ``slot`` (= absolute slot index within the cache block, range
        # 0.._BS-1) is recomputed per tile inside the K-tile loop below
        # so it reflects ``tile_in_block * TILE_SIZE + tok_in_tile``.

        c_meta_u16_per_kvh = fx.Int32(NUM_SOA_FIELDS * _BS)
        c_knorm_off_u16 = fx.Int32(SOA_K_NORM * _BS)
        c_vscale_off_u16 = fx.Int32(SOA_V_SCALE * _BS)
        c_vzero_off_u16 = fx.Int32(SOA_V_ZERO * _BS)

        # ===== STEP F: K-tile loop =======================================
        # With kv_block_size > TILE_SIZE there are _TILES_PER_BLOCK tiles
        # per cache block. Tiles within the same block share a block-table
        # entry but address different slot ranges within the block.
        #
        # ── Per-tile block-table OOB redirect ────────────────────────────
        # The K-tile loop is unrolled 16 times so every CTA issues 16
        # block-table reads regardless of (partition, seq_len). When a
        # partition extends past ``seq_len`` (e.g. num_partitions=2 for
        # a 256-token sequence — partition 1 covers tokens 256..511, all
        # masked out) the per-tile bt offset can land beyond the bt
        # allocation. With ``max_size=True`` the descriptor advertises
        # 4 GB so the HW returns whatever pre-existing HBM bytes live at
        # that offset; the resulting garbage ``phys_block`` is multiplied
        # by ``stride_cache_block`` and the subsequent kv_cache read
        # jumps into pages that may be unmapped (→ "Memory access fault
        # by GPU node") or may decode as NaN bytes (→ NaN segm_out).
        # Allocator-pattern dependent, but reproduces deterministically
        # at large B (e.g. B=64 seq=256 in the test sweep produces 8 192
        # NaN entries even on canonical QG=8).
        #
        # Fix: when the tile starts at or past ``seq_len`` (so its
        # qk_acc is going to be killed by the per-token
        # ``kv_tok < seq_len`` mask anyway), redirect the bt read to
        # ``bt[seq, 0]`` — always in bounds. The redundant phys_block
        # decode + kv_cache read is wasted work, but correctness is
        # preserved without changing the kernel's iteration count.
        # ===== Option A': scf.ForOp w/ iter_args (HIP-style) =========
        # Runtime-adaptive outer loop: trip count derives from
        # actual seq_len, NOT TGPP_max. At cudagraph-capture warmup
        # (seq_len=1) only 1 iteration runs; at 32K production
        # decode all TGPP iterations run. Kernel binary stays small
        # (single body, looped at runtime) — capture time matches
        # the legacy 16-tile-only kernel rather than scaling 4x.
        #
        # FA-2 state (running_max, running_sum, acc_pv[8]) threads
        # through scf iter_args; the body reads them at top, runs
        # the unchanged 16-tile inner body, and yields the new
        # state at bottom. After the loop, results are pulled out
        # of for_op.results back into the local Python names so the
        # downstream STEP G (output) is unmodified.
        c_kcb = fx.Int32(KV_COMPUTE_BLOCK)
        c_tgpp = fx.Int32(TGPP)
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        # remaining = seq_len - partition_base   (signed, may be <=0)
        remaining = seq_len - partition_base
        in_range = remaining > c_zero_i32
        # trip_raw = ceil(remaining / KV_COMPUTE_BLOCK)
        # (when in_range is false, the divisor branch produces
        # garbage that the select below overrides with zero, so we
        # don't pre-clamp)
        trip_raw = (remaining + c_kcb - c_one_i32) // c_kcb
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
        # Initial iter_args list: FA-2 accumulator state.
        # Order: running_max, running_sum, acc_pv[0..PV_N_CHUNKS-1].
        # NOTE: scf.ForOp requires ir.Value (with .type). The DSL
        # constants ZERO_F / ONE_F are fx.Float32 (Numeric) wrappers,
        # not raw ir.Value, so we unwrap via .ir_value() before
        # passing.  _ival() is defined once near the top of the kernel body.
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
            partition_start = partition_base + tg_i32 * c_kcb
            bt_seq_base = seq * c_bt + (partition_start // fx.Int32(_BS))
            running_max = _for_op.inner_iter_args[0]
            running_sum = _for_op.inner_iter_args[1]
            acc_pv = list(_for_op.inner_iter_args[2:])
            for n_tile in range_constexpr(16):
                block_in_part = n_tile // _TILES_PER_BLOCK
                tile_in_block = n_tile % _TILES_PER_BLOCK
                tile_start_tok = partition_start + fx.Int32(n_tile * TILE_SIZE)
                tile_in_seq = tile_start_tok < seq_len
                bt_off = bt_seq_base + fx.Int32(block_in_part)
                bt_off_safe = tile_in_seq.select(bt_off, seq * c_bt)
                phys_block = buffer_ops.buffer_load(
                    bt_rsrc,
                    bt_off_safe,
                    vec_width=1,
                    dtype=T.i32,
                )
                block_base = phys_block * c_block
                data_region = block_base
                meta_region = block_base + c_meta_off

                # ``slot`` is the absolute slot index within the cache block
                # (range 0.._BS-1). For BS=16 it equals tok_in_tile; for BS=32
                # it equals tile_in_block * 16 + tok_in_tile. Used both for
                # the K/V data region and SoA metadata region addressing.
                slot = fx.Int32(tile_in_block * TILE_SIZE) + tok_in_tile
                data_bases_byte = (
                    data_region
                    + slot * (c_kv_heads * c_data_per_slot)
                    + kv_h * c_data_per_slot
                )
                k_byte = data_bases_byte + chunk_in_tok * fx.Int32(16)
                k_packed = buffer_ops.buffer_load(
                    kv_rsrc,
                    k_byte // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                )

                # ---- HOISTED: issue V data + meta HBM loads early ---------
                # These are async; their s_waitcnt is pushed by the compiler
                # past the K dequant + QK MFMA + softmax block, hiding most
                # of the V HBM latency behind compute.
                v_byte = data_bases_byte + c_keydata + chunk_in_tok * fx.Int32(16)
                v_packed = buffer_ops.buffer_load(
                    kv_rsrc,
                    v_byte // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                )
                vscale_u16 = (
                    meta_region // fx.Int32(2)
                    + kv_h * c_meta_u16_per_kvh
                    + c_vscale_off_u16
                    + slot
                )
                vzero_u16 = (
                    meta_region // fx.Int32(2)
                    + kv_h * c_meta_u16_per_kvh
                    + c_vzero_off_u16
                    + slot
                )
                vscale_raw = buffer_ops.buffer_load(
                    kv_rsrc,
                    vscale_u16,
                    vec_width=1,
                    dtype=T.i16,
                )
                vzero_raw = buffer_ops.buffer_load(
                    kv_rsrc,
                    vzero_u16,
                    vec_width=1,
                    dtype=T.i16,
                )

                knorm_u16 = (
                    meta_region // fx.Int32(2)
                    + kv_h * c_meta_u16_per_kvh
                    + c_knorm_off_u16
                    + slot
                )
                knorm_raw = buffer_ops.buffer_load(
                    kv_rsrc,
                    knorm_u16,
                    vec_width=1,
                    dtype=T.i16,
                )
                knorm_f16 = arith.bitcast(T.f16, knorm_raw)
                knorm_f32 = arith.extf(T.f32, knorm_f16)

                # K dequant → LDS [token, head_dim] (natural)
                # Lane writes 32 bf16 (= 4×8) for token=tok_in_tile,
                # head_dims chunk_in_tok*32..+31 (4 sub-chunks of 8).
                tok_kreg = tok_in_tile * fx.Int32(HEAD_SIZE * 2 // 8)
                chunk_kreg = tok_kreg + chunk_in_tok * fx.Int32(8)
                for w in range_constexpr(4):
                    word_i32 = vector.extract(k_packed, static_position=[w])
                    bf16_elems = []
                    for n in range_constexpr(8):
                        nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                        nibble_idx = arith.index_cast(T.index, nibble)
                        cent_f32 = cent_lds.load([nibble_idx])
                        elem_bf16 = arith.trunc_f(T.bf16, cent_f32 * knorm_f32)
                        bf16_elems.append(elem_bf16)
                    v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                    v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                    vector.store(
                        v_i64,
                        kv_lds_i64,
                        [arith.index_cast(T.index, chunk_kreg + fx.Int32(w * 2))],
                    )
                gpu.barrier()

                # QK MFMA (CDNA4 wide-K): A=K[token, head_dim], B=Q (= Q^T).
                # K read: lane t = K[token=mfma_row, head_dim=chunk*32+col_grp*8..+7]
                # Same i64×2 indexing as Q.
                qk_acc = zero_v4
                for chk in range_constexpr(QK_K_CHUNKS):
                    k_idx_i64 = (
                        mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                        + fx.Int32(chk * 8)
                        + mfma_col_grp * fx.Int32(2)
                    )
                    kv_load = vector.load_op(
                        T.vec(2, T.i64),
                        kv_lds_i64,
                        [arith.index_cast(T.index, k_idx_i64)],
                    )
                    k_op = vector.bitcast(T.vec(8, T.bf16), kv_load)
                    qk_acc = rocdl.mfma_f32_16x16x32_bf16(
                        T.f32x4, [k_op, q_chunks[chk], qk_acc, 0, 0, 0]
                    )

                # qk_acc layout: lane t holds C[token=(t/16)*4..+3, query=t%16]
                # 4 fp32/lane = 4 different tokens at SAME query_row.

                # Scale + mask out-of-context tokens.
                qk_acc = _vsplat_mul(qk_acc, QK_SCALE)
                for elem in range_constexpr(4):
                    kv_tok = (
                        partition_start
                        + fx.Int32(n_tile * TILE_SIZE)
                        + mfma_col_grp * fx.Int32(4)
                        + fx.Int32(elem)
                    )
                    in_b = kv_tok < seq_len
                    v = vector.extract(qk_acc, static_position=[elem])
                    qk_acc = vector.insert(
                        in_b.select(v, NEG_INF),
                        qk_acc,
                        static_position=[elem],
                        dynamic_position=[],
                    )

                # FA2 online softmax: per-query-row reduce.
                # Per-row max: max over 4 fp32 in lane, then xor-shuffle 16, 32
                # (across the 4 col_grps that share same mfma_row).
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

                # Compute probs: p = exp((qk - new_max) * LOG2E)
                tile_sum = ZERO_F
                for elem in range_constexpr(4):
                    s = vector.extract(qk_acc, static_position=[elem])
                    d = s - new_max
                    d = (new_max > NEG_INF).select(d, NEG_INF)
                    p = (d * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    tile_sum = tile_sum + p
                    qk_acc = vector.insert(
                        p, qk_acc, static_position=[elem], dynamic_position=[]
                    )

                ts1 = tile_sum.shuffle_xor(fx.Int32(16), c_w)
                tile_sum = tile_sum + ts1
                ts2 = tile_sum.shuffle_xor(fx.Int32(32), c_w)
                tile_sum = tile_sum + ts2
                running_sum = running_sum + tile_sum

                # ---- V dequant → LDS (V/scale/zero raw loads were hoisted) -
                # v_packed, vscale_raw, vzero_raw are already in flight from
                # the top of this iteration; only the f32 conversion stays
                # here so the actual consume site is still close to the use.
                vscale_f32 = arith.extf(T.f32, arith.bitcast(T.f16, vscale_raw))
                vzero_f32 = arith.extf(T.f32, arith.bitcast(T.f16, vzero_raw))

                if USE_HW_TR:
                    # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose).
                    # Each lane writes 32 contiguous bf16 (one token, head_dims
                    # chunk_in_tok*32..+31) as 4× ds_write_b128 = 4× vec(2,i64).
                    # Replaces 32× ds_write_b16 of the legacy transposed path.
                    v_lds_elem_base = tok_in_tile * fx.Int32(
                        HEAD_SIZE
                    ) + chunk_in_tok * fx.Int32(32)
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        bf16_elems = []
                        for n in range_constexpr(8):
                            nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                            nibble_f32 = arith.sitofp(T.f32, nibble)
                            elem_f32 = nibble_f32 * vscale_f32 + vzero_f32
                            elem_bf16 = arith.trunc_f(T.bf16, elem_f32)
                            bf16_elems.append(elem_bf16)
                        v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                        v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                        v_lds_i64_idx = (v_lds_elem_base + fx.Int32(w * 8)) // fx.Int32(
                            4
                        )
                        vector.store(
                            v_i64,
                            kv_lds_i64,
                            [arith.index_cast(T.index, v_lds_i64_idx)],
                        )
                    # ── HW V transpose: cross-lane LDS sync ─────────────────────
                    # ds_read_tr16_b64 below introduces a cross-lane LDS read:
                    # consumer lane t reads bytes written by source lanes
                    # 0,4,8,12 (etc.). The compiler's automatic waitcnt insertion
                    # is per-lane and has no awareness of the HW transpose's
                    # cross-lane forwarding, so it can sink the next iteration's
                    # ds_read ahead of these ds_write_b128 ops in the schedule.
                    #
                    # Two-part fence (mirrors mla_fwd_decode_m16x8_fp8_fp8.py):
                    #   1. sched_barrier(0): MachineScheduler reorder barrier
                    #      (mask=0 → no instruction class may cross). Pure hint,
                    #      generates no runtime instruction. Use the simple
                    #      sched_barrier (NOT sched_group_barrier, which is an
                    #      IGLP marker that requires partner barriers and
                    #      corrupts the schedule when used in isolation).
                    #   2. s_waitcnt lgkmcnt=0: runtime drain of the LDS write
                    #      queue. vmcnt/expcnt left at no-wait so we don't
                    #      stall on speculatively-issued HBM loads (which can
                    #      include OOB-but-masked buffer_load reads from the
                    #      next K-tile's hoisted V prefetch).
                    #
                    # The SW path below does NOT need this because each lane
                    # only reads cells it itself wrote (no cross-lane dep), and
                    # the per-lane same-address waitcnt the compiler emits is
                    # correct.
                    #
                    # Empirical: targets the -3.3pp GSM8K regression on
                    # Qwen3-32B (padded num_partitions=64, ~16 K-tiles each)
                    # while leaving Qwen2.5-72B (16 partitions) unchanged.
                    rocdl.sched_barrier(0)
                    # encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=0)
                    #   = 0xF | (7<<4) | (0<<8) | (3<<14) = 0xC07F
                    rocdl.s_waitcnt(0xC07F)
                else:
                    # Legacy transposed V_LDS path: 32 ds_write_b16 per lane.
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        for n in range_constexpr(8):
                            nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                            nibble_f32 = arith.sitofp(T.f32, nibble)
                            elem_f32 = nibble_f32 * vscale_f32 + vzero_f32
                            elem_bf16 = arith.trunc_f(T.bf16, elem_f32)
                            elem_i16 = arith.bitcast(T.i16, elem_bf16)
                            head_dim = chunk_in_tok * fx.Int32(32) + fx.Int32(w * 8 + n)
                            v_idx_i16 = head_dim * fx.Int32(TILE_SIZE) + tok_in_tile
                            v_vec = vector.from_elements(T.vec(1, T.i16), [elem_i16])
                            vector.store(
                                v_vec,
                                kv_lds_i16,
                                [arith.index_cast(T.index, v_idx_i16)],
                            )
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                # P operand B layout matches qk_acc: lane t holds 4 bf16 at
                # K=token=(t/16)*4..+3, N=query=t%16. Just trunc_f to bf16.
                p_bf16 = arith.trunc_f(T.vec(4, T.bf16), qk_acc)
                p_op = vector.bitcast(T.vec(4, T.i16), p_bf16)

                if USE_HW_TR:
                    # HW-transpose PV path: V_lds is row-major V[token][head_dim].
                    # MFMA A operand layout: lane t holds 4 bf16 at
                    #   M=head_dim=mfma_row + h*16, K=token=(t/16)*4..+3
                    # ds_read_tr16_b64 (4-element 16-bit transpose, per-16-lane block):
                    #   result[lane=t, elem=e] = Input[source_lane=e*4 + (t%16)//4,
                    #                                   col=t%4]
                    # Per-lane address: token_idx = lane // 4 (covers 0..15 across
                    # all four 16-lane MFMA blocks), hd_sub = (lane % 4)*4 selects
                    # the 4-element column window inside the h*16 chunk.
                    # Total LDS byte offset = kv_off + token_idx*HEAD_SIZE*2
                    #                        + (h*16 + hd_sub)*2
                    token_idx = lane >> fx.Int32(2)
                    hd_sub = (lane & fx.Int32(3)) * fx.Int32(4)
                    v_lane_byte = (
                        fx.Int32(kv_off)
                        + token_idx * fx.Int32(HEAD_SIZE * 2)
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
                else:
                    # Legacy transposed-LDS PV: 1 ds_read_b64 per chunk.
                    # Byte addr = (mfma_row + h*16) * TILE_SIZE * 2 + (col_grp*4) * 2
                    #          = (mfma_row + h*16) * 32 + col_grp*8
                    # i64 idx  = (mfma_row + h*16) * 4 + col_grp
                    for h in range_constexpr(PV_N_CHUNKS):
                        v_idx_i64 = (mfma_row + fx.Int32(h * 16)) * fx.Int32(
                            4
                        ) + mfma_col_grp
                        kv_load = vector.load_op(
                            T.vec(1, T.i64),
                            kv_lds_i64,
                            [arith.index_cast(T.index, v_idx_i64)],
                        )
                        v_op = vector.bitcast(T.vec(4, T.i16), kv_load)
                        acc_pv[h] = rocdl.mfma_f32_16x16x16bf16_1k(
                            T.f32x4, [v_op, p_op, acc_pv[h], 0, 0, 0]
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
        # Pull final accumulator values out of the for_op results.
        running_max = _for_op.results[0]
        running_sum = _for_op.results[1]
        acc_pv = list(_for_op.results[2:])

        # ===== STEP G: Output ===========================================
        # acc_pv[h] layout: lane t holds 4 fp32 at M=head_dim=(t/16)*4..+3 + h*16,
        #                                       N=query_row=t%16.
        # 4 fp32/lane = 4 contiguous head_dim values for ONE query_row.
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        c_os = fx.Int32(_stride_out_seq)
        c_oh = fx.Int32(_stride_out_head)
        c_op_ = fx.Int32(_stride_out_part)
        out_base = seq * c_os + kv_h * c_oh + part * c_op_

        # Gate all global stores by mfma_row < QG. For QG=16 the predicate is
        # tautological (mfma_row ∈ [0,15]); for QG=8 lanes 8..15 (whose MFMA
        # outputs are computational waste) skip the stores entirely.
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

    return tq_decode_v4_kernel
