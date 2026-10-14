# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TurboQuant decode GQA-6 sibling (FlyDSL) — MI355X / gfx950 / CDNA4
#
# Decode-only paged attention over a TurboQuant SoA KV cache.
# Targets MiniMax-M2.5 class models: HEAD_SIZE=128, GQA group=6, MSE_BITS=4 K,
# VQB=4 V, N_CENTROIDS=16, BLOCK_SIZE=16, partitioned mode.
#
# === Sibling-file rationale ===
# Structurally identical to tq_decode.py except:
#   * QUERY_GROUP_SIZE default = 6 (was 8)
#   * Build-time assert allows query_group_size = 6 only (Qwen sizes go to
#     the canonical tq_decode kernel — NOT this file)
#   * QG_LOAD_ITERS = (QG + 3) // 4 (round-up; 2 iters for QG=6 loads
#     8 LDS rows of which only 6 carry real Q; rows 6..7 are zero-padded
#     by buffer_load OOB protection — gated out by the same mfma_row < QG
#     predicate the canonical kernel already uses for QG=8)
#
# All cross-lane operations (xor_shuffle softmax reductions, ds_read_tr16_b64
# HW V transpose) are mfma_row-isolated and thus already GQA-agnostic. The
# kernel is functionally a direct copy of tq_decode.py with the QG knob
# turned to 6 — kept as a sibling per FlyDSL convention to preserve the
# canonical kernel's invariants for its production callers (Qwen2.5-72B,
# Qwen3-32B).
#
# === Design ===
# * 1 wave (64 lanes) per CTA — simpler than 4-warp; trades parallelism for
#   correctness/debug-ability. A later revision will fan out to 4 warps.
# * Flash-Attention-2 online softmax: rescale running PV/sum per K-tile,
#   no QK score storage across tiles, no P-LDS round-trip (qk_acc layout
#   matches PV B-operand directly).
# * Centroids loaded once into LDS at kernel start (LDS-resident LUT).
# * K dequant → LDS [token, head_dim] (natural layout)
# * V dequant → LDS [token, head_dim]; the PV MFMA's A=V operand is read
#   back through the ds_read_tr16_b64 hardware transpose.
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
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from vllm.v1.attention.ops.flydsl_kernels.buffer_utils import (
    buffer_view,
    load_vec,
    store_vec,
)
from vllm.v1.attention.ops.flydsl_kernels.mma_utils import mma

# === Constants (MiniMax-M2.5-class TQ decode profile) =======================
HEAD_SIZE = 128
# default; overridable via build_tq_decode_gqa6_module(kv_block_size=...)
KV_BLOCK_SIZE = 16
# MFMA tile = 16 tokens (do not change without re-deriving MFMA shapes)
TILE_SIZE = 16
N_CENTROIDS = 16
QUERY_GROUP_SIZE = 6  # GQA-6 default for MiniMax-M2.5
WARP_SIZE = 64
NUM_WARPS = 1
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
KV_COMPUTE_BLOCK = 256  # 16 K-tiles × 16 tokens
N_TILES = KV_COMPUTE_BLOCK // TILE_SIZE

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
QK_K_CHUNKS = HEAD_SIZE // MFMA_K_BF16_QK  # 4
PV_N_CHUNKS = HEAD_SIZE // MFMA_N  # 8

# LDS regions
# Q region uses the full 16-row footprint (4096 B) rather than a tight
# QG=6 sizing, keeping the LDS layout identical to the canonical kernel.
CENTROID_LDS_BYTES = N_CENTROIDS * 4  # 64
Q_LDS_BYTES = 16 * HEAD_SIZE * 2  # 4096
KV_TILE_LDS_BYTES = TILE_SIZE * HEAD_SIZE * 2  # 4096

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")

allocator = None


def build_tq_decode_gqa6_module(
    num_seqs: int,
    num_kv_heads: int,
    num_partitions: int,
    max_blocks_per_seq: int = 512,
    softmax_scale: float | None = None,
    query_group_size: int = QUERY_GROUP_SIZE,
    kv_block_size: int = KV_BLOCK_SIZE,
    tile_groups_per_partition: int = 1,
):
    """Build a TQ decode GQA-6 kernel module (MiniMax-M2.5).

    ``query_group_size`` (= num_query_heads // num_kv_heads) is fixed at 6
    for this sibling. The MFMA's 16-row capacity is 6/16 = 37.5% used;
    lanes 6..15 compute garbage and are gated out of all global-memory
    writes via the same ``mfma_row < QG`` predicate the canonical kernel
    uses for QG=8. Q load iterates ``(QG + 3) // 4`` = 2 chunks (loads
    8 LDS rows of which rows 6..7 are zero-padded by the buffer-resource
    OOB protection — they're never consumed because the gating lanes
    don't write).

    For GQA-8 or GQA-16 (Qwen models), use the canonical
    ``build_tq_decode_module`` from ``tq_decode.py`` instead.

    ``kv_block_size`` is the number of tokens per vLLM cache block. Must
    be a multiple of TILE_SIZE (=16). Supported values: 16 and 32.
    """
    assert query_group_size == 6, (
        f"build_tq_decode_gqa6_module is GQA-6 only; got "
        f"query_group_size={query_group_size}. Use build_tq_decode_module "
        f"from tq_decode.py for GQA-8 or GQA-16."
    )
    assert kv_block_size in (16, 32), (
        f"kv_block_size must be 16 or 32; got {kv_block_size}"
    )
    assert kv_block_size % TILE_SIZE == 0
    assert int(tile_groups_per_partition) >= 1, (
        f"tile_groups_per_partition must be >= 1; got {tile_groups_per_partition}"
    )
    TGPP = int(tile_groups_per_partition)
    PARTITION_EXTENT_TOKENS = TGPP * KV_COMPUTE_BLOCK

    QG = int(query_group_size)
    # Round-up: 2 iters for QG=6 → loads 8 LDS rows (rows 6..7 are
    # zero-padded by buffer_load OOB protection; never consumed because
    # mfma_row >= QG lanes are gated out of all global stores below).
    QG_LOAD_ITERS = (QG + 3) // 4

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

    _stride_data_slot = num_kv_heads * DATA_BYTES_PER_SLOT
    _data_region_bytes = _BS * _stride_data_slot
    _meta_region_bytes = num_kv_heads * NUM_SOA_FIELDS * _BS * 2
    _stride_cache_block = _data_region_bytes + _meta_region_bytes
    _meta_region_offset = _data_region_bytes

    _stride_out_part = QG * HEAD_SIZE
    _stride_out_head = num_partitions * QG * HEAD_SIZE
    _stride_out_seq = num_kv_heads * num_partitions * QG * HEAD_SIZE
    _stride_es_seq = num_kv_heads * num_partitions * QG
    _stride_ml_seq = _stride_es_seq

    # --- LDS layout ---
    # Legacy SmemAllocator kept to match tq_decode.py (see the note there).
    allocator = SmemAllocator(None, arch=arch, global_sym_name="tq_gqa6_smem")
    centroid_off = 0
    allocator.ptr = CENTROID_LDS_BYTES
    q_off = allocator.ptr
    allocator.ptr += Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES

    @flyc.kernel
    def tq_decode_gqa6_kernel(
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
        mfma_row = lane & 15  # = query_row (B's N) /
        #   token (A's M for QK)
        mfma_col_grp = lane >> 4  # 0..3, K-group dim

        # ---- Buffer views -----------------------------------------------
        q_buf = buffer_view(query_ptr, fx.Int32, 4)
        kv_buf4 = buffer_view(kv_cache_ptr, fx.Int32, 4)
        kv_buf_i16 = buffer_view(kv_cache_ptr, fx.Int16)
        bt_buf = buffer_view(block_tables_ptr, fx.Int32)
        sl_buf = buffer_view(seq_lens_ptr, fx.Int32)
        cent_buf = buffer_view(centroids_ptr, fx.Float32)
        out_buf = buffer_view(out_ptr, fx.BFloat16, 4)
        es_buf = buffer_view(exp_sums_ptr, fx.Float32)
        ml_buf = buffer_view(max_logits_ptr, fx.Float32)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        cent_lds = SmemPtr(base, centroid_off, T.f32, shape=(N_CENTROIDS,))
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(Q_LDS_BYTES // 8,)).get()
        kv_lds_i64 = SmemPtr(base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES // 8,)).get()
        # ---- Constants ---------------------------------------------------
        NEG_INF = fx.Float32(NEG_INF_VAL)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = fx.Float32(LOG2E)
        QK_SCALE = fx.Float32(_qk_scale)
        qk_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        pv_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & (N_CENTROIDS - 1)
        c_val = fx.Float32(cent_buf[c_idx_safe])
        cent_lds.store(c_val, [fx.Index(c_idx_safe)])
        gpu.barrier()

        # ===== STEP B: Load Q → row-major LDS [query_row, head_dim] ======
        # Q layout in HBM: [N, Hq, D] bf16, slice = QG rows × D=128 cols × 2B.
        # QG=6 → 1536 B real, 96 chunks; QG_LOAD_ITERS=2 issues 128 chunks
        # (8 rows × 16 col_b) so lanes whose row ∈ {6, 7} address heads that
        # do NOT exist in the Q tensor (Hq = num_kv_heads * 6).
        #
        # ── Per-lane Q-load OOB redirect ─────────────────────────────────
        # Hq = num_kv_heads * QG, so head index = kv_h * QG + row. For QG=6
        # the upper rows alias to (kv_h+1)'s real heads (still in-bounds
        # for most kv_h), but for the *last* (seq, kv_h) the read tips
        # past the Q tensor entirely. With ``max_size=True`` the buffer
        # descriptor advertises 4 GB so the HW does no OOB clamping; the
        # padding read then touches unmapped pages and the kernel dies
        # with "Memory access fault by GPU node" the moment the
        # surrounding allocator state leaves the next page unmapped
        # (allocator-pattern-dependent → nondeterministic fault).
        #
        # Fix: redirect OOB lanes (``row >= QG``) to load Q[0,0,col_elem]
        # — a byte offset that is always inside the actual Q tensor.
        # The redundant load is wasted but harmless; the resulting LDS
        # write to row ∈ {6, 7} is overwritten/ignored downstream because
        # every global store is gated by the output-time
        # ``mfma_row < QG`` predicate (the same gate canonical QG=8 uses
        # for its garbage rows 8..15).
        #
        # NB: A ``mask=`` parameter on ``buffer_load`` substitutes the
        # masked-out offset with 0x7FFFFFFF *bytes*; with ``max_size=True``
        # this is still inside the 4 GB descriptor and so the HW happily
        # dereferences ``q_base + 2 GB`` → another unmapped page → fault.
        # The address-redirect below is the only safe option short of
        # threading the true Q size through as a runtime kernel arg.
        #
        # The canonical Qwen kernel does not need this redirect because
        # for QG ∈ {8, 16} the (kv_h, row) iteration covers exactly the
        # real heads; QG=6 is the only case where
        # ``QG_LOAD_ITERS * 4 != QG``.
        # STEP B: load pre-rotated q_rot → Q_LDS. QG=6 over-reads
        # (QG_LOAD_ITERS * 4 != QG), so rows >= QG redirect to a safe
        # in-bounds offset to avoid faulting past the real Q rows.
        for c in range_constexpr(QG_LOAD_ITERS):
            row_chunk = lane + c * WARP_SIZE  # 0..127
            row = row_chunk >> 4  # 0..7
            col_b = row_chunk & 15  # 0..15
            col_elem = col_b * 8  # 0..120 in bf16
            q_off_real = (
                seq * _stride_q_seq + (kv_h * QG + row) * _stride_q_head + col_elem
            )
            q_off_elem = (row < QG).select(q_off_real, col_elem)
            q_v = load_vec(q_buf, q_off_elem // 2, 4)
            q_lds_byte = row * (HEAD_SIZE * 2) + col_elem * 2
            fx.Vector(q_v).store(q_lds_i32, [fx.Index(q_lds_byte // 4)])
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
            q_idx_i64 = mfma_row * (HEAD_SIZE * 2 // 8) + chk * 8 + mfma_col_grp * 2
            qv = fx.Vector.load(T.vec(2, T.i64), q_lds_i64, [fx.Index(q_idx_i64)])
            q_chunks.append(qv.bitcast(fx.BFloat16))

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = fx.Vector.filled(4, 0.0, fx.Float32)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = fx.Int32(sl_buf[seq])
        # ── Bounded num_partitions + internal looping ───────────────────
        # Mirrors the canonical tq_decode kernel: each CTA owns
        # ``PARTITION_EXTENT_TOKENS = TGPP * KV_COMPUTE_BLOCK`` contiguous
        # tokens of the seq, processed as TGPP groups of 16 K-tiles each.
        # FA-2 online-softmax state accumulates across all TGPP * 16 tiles.
        # With TGPP=1 this is bit-identical to the legacy 16-tile body.
        partition_base = part * PARTITION_EXTENT_TOKENS

        # Per-K-tile dequant lane assignment:
        # 16 tokens × 64 packed bytes per token = 1024 bytes
        # 64 lanes × 16 bytes/lane = 1024 ✓
        # Lane t: token = t / 4, sub-chunk = t % 4 (0..3) → 16 bytes contiguous
        tok_in_tile = lane >> 2
        chunk_in_tok = lane & 3
        # ``slot`` (= absolute slot index within the cache block, range
        # 0.._BS-1) is recomputed per tile inside the K-tile loop below
        # so it reflects ``tile_in_block * TILE_SIZE + tok_in_tile``.

        meta_u16_per_kvh = NUM_SOA_FIELDS * _BS

        # ===== STEP F: K-tile loop =======================================
        # With kv_block_size > TILE_SIZE there are _TILES_PER_BLOCK tiles
        # per cache block. Tiles within the same block share a block-table
        # entry but address different slot ranges within the block.
        #
        # ── Per-tile block-table OOB redirect ────────────────────────────
        # The K-tile loop is unrolled 16 times so every CTA issues 16
        # block-table reads regardless of (partition, seq_len). When a
        # partition extends past ``seq_len`` (e.g. num_partitions=2 for a
        # 256-token sequence — partition 1 covers tokens 256..511, all
        # masked out) the per-tile bt offset can land beyond the bt
        # allocation. With ``max_size=True`` the descriptor is 4 GB so
        # the HW returns whatever pre-existing HBM bytes live at that
        # offset; the resulting garbage ``phys_block`` is multiplied by
        # ``stride_cache_block`` and the subsequent kv_cache read jumps
        # into unmapped pages, killing the whole launch with a
        # "Memory access fault by GPU node" exactly as the OOB Q load
        # used to (same allocator-pattern-dependent fault; B=4 makes
        # it deterministic by adding more colliding CTAs).
        #
        # Fix: when the tile starts at or past ``seq_len`` (so its
        # qk_acc is going to be killed by the per-token
        # ``kv_tok < seq_len`` mask anyway), redirect the bt read to
        # ``bt[seq, 0]`` — always in bounds. The redundant phys_block
        # decode + kv_cache read is wasted work, but correctness is
        # preserved without changing the kernel's iteration count.
        #
        # The canonical Qwen kernel reads the same tiles and is also
        # exposed to this issue in principle; in practice it surfaces on
        # GQA-6 first because the surrounding allocator state differs
        # (extra Q-tail allocation) and shifts the page maps so the bt
        # overflow lands on unmapped pages. Applying the same redirect to
        # the canonical kernel is a possible follow-up.
        # ===== Runtime-adaptive scf.ForOp w/ iter_args =========
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
        # remaining = seq_len - partition_base   (signed, may be <=0)
        remaining = seq_len - partition_base
        in_range = remaining > 0
        # trip_raw = ceil(remaining / KV_COMPUTE_BLOCK)
        # (when in_range is false, the divisor branch produces
        # garbage that the select below overrides with zero, so we
        # don't pre-clamp)
        trip_raw = (remaining + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
        trip_clamped = (trip_raw > TGPP).select(TGPP, trip_raw)
        trip_or_zero = in_range.select(trip_clamped, 0)
        # Loop-carried FA-2 state: running_max, running_sum, acc_pv[...].
        loop_init = [running_max, running_sum, *acc_pv]
        results = loop_init
        for tg_idx, state in range(0, trip_or_zero, 1, init=loop_init):  # type: ignore[call-overload]
            tg_i32 = fx.Int32(tg_idx)
            partition_start = partition_base + tg_i32 * KV_COMPUTE_BLOCK
            bt_seq_base = seq * _stride_bt_seq + (partition_start // _BS)
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            acc_pv = [fx.Vector(v) for v in state[2:]]
            for n_tile in range_constexpr(N_TILES):
                block_in_part = n_tile // _TILES_PER_BLOCK
                tile_in_block = n_tile % _TILES_PER_BLOCK
                tile_start_tok = partition_start + n_tile * TILE_SIZE
                tile_in_seq = tile_start_tok < seq_len
                bt_off = bt_seq_base + block_in_part
                bt_off_safe = tile_in_seq.select(bt_off, seq * _stride_bt_seq)
                phys_block = fx.Int32(bt_buf[bt_off_safe])
                block_base = phys_block * _stride_cache_block
                data_region = block_base
                meta_region = block_base + _meta_region_offset

                # ``slot`` is the absolute slot index within the cache block
                # (range 0.._BS-1). For BS=16 it equals tok_in_tile; for BS=32
                # it equals tile_in_block * 16 + tok_in_tile. Used both for
                # the K/V data region and SoA metadata region addressing.
                slot = tok_in_tile + tile_in_block * TILE_SIZE
                data_bases_byte = (
                    data_region + slot * _stride_data_slot + kv_h * DATA_BYTES_PER_SLOT
                )
                k_byte = data_bases_byte + chunk_in_tok * 16
                k_packed = load_vec(kv_buf4, k_byte // 4, 4)

                # ---- HOISTED: issue V data + meta HBM loads early ---------
                # These are async; their s_waitcnt is pushed by the compiler
                # past the K dequant + QK MFMA + softmax block, hiding most
                # of the V HBM latency behind compute.
                v_byte = data_bases_byte + KEY_DATA_BYTES + chunk_in_tok * 16
                v_packed = load_vec(kv_buf4, v_byte // 4, 4)
                vscale_u16 = (
                    meta_region // 2
                    + kv_h * meta_u16_per_kvh
                    + SOA_V_SCALE * _BS
                    + slot
                )
                vzero_u16 = (
                    meta_region // 2 + kv_h * meta_u16_per_kvh + SOA_V_ZERO * _BS + slot
                )
                vscale_raw = kv_buf_i16[vscale_u16]
                vzero_raw = kv_buf_i16[vzero_u16]

                knorm_u16 = (
                    meta_region // 2 + kv_h * meta_u16_per_kvh + SOA_K_NORM * _BS + slot
                )
                knorm_raw = kv_buf_i16[knorm_u16]
                knorm_f32 = fx.Int16(knorm_raw).bitcast(fx.Float16).to(fx.Float32)

                # K dequant → LDS [token, head_dim] (natural)
                # Lane writes 32 bf16 (= 4×8) for token=tok_in_tile,
                # head_dims chunk_in_tok*32..+31 (4 sub-chunks of 8).
                tok_kreg = tok_in_tile * (HEAD_SIZE * 2 // 8)
                chunk_kreg = tok_kreg + chunk_in_tok * 8
                k_words = fx.Vector(k_packed)
                for w in range_constexpr(4):
                    bf16_elems = []
                    for n in range_constexpr(8):
                        nibble = (k_words[w] >> (n * 4)) & 0xF
                        cent_f32 = fx.Float32(cent_lds.load([fx.Index(nibble)]))
                        bf16_elems.append((cent_f32 * knorm_f32).to(fx.BFloat16))
                    v_bf16 = fx.Vector.from_elements(bf16_elems, dtype=fx.BFloat16)
                    v_bf16.bitcast(fx.Int64).store(
                        kv_lds_i64, [fx.Index(chunk_kreg + w * 2)]
                    )
                gpu.barrier()

                # QK MFMA (CDNA4 wide-K): A=K[token, head_dim], B=Q (= Q^T).
                # K read: lane t = K[token=mfma_row, head_dim=chunk*32+col_grp*8..+7]
                # Same i64×2 indexing as Q.
                qk_acc = zero_v4
                for chk in range_constexpr(QK_K_CHUNKS):
                    k_idx_i64 = (
                        mfma_row * (HEAD_SIZE * 2 // 8) + chk * 8 + mfma_col_grp * 2
                    )
                    kv_load = fx.Vector.load(
                        T.vec(2, T.i64), kv_lds_i64, [fx.Index(k_idx_i64)]
                    )
                    k_op = kv_load.bitcast(fx.BFloat16)
                    qk_acc = mma(qk_atom, qk_acc, k_op, q_chunks[chk])

                # qk_acc layout: lane t holds C[token=(t/16)*4..+3, query=t%16]
                # 4 fp32/lane = 4 different tokens at SAME query_row.

                # Scale + mask out-of-context tokens.
                qk_acc = qk_acc * QK_SCALE
                masked = []
                for elem in range_constexpr(4):
                    kv_tok = (
                        partition_start + n_tile * TILE_SIZE + mfma_col_grp * 4 + elem
                    )
                    masked.append((kv_tok < seq_len).select(qk_acc[elem], NEG_INF))
                qk_acc = fx.Vector.from_elements(masked, dtype=fx.Float32)

                # FA2 online softmax: per-query-row reduce.
                # Per-row max: max over 4 fp32 in lane, then xor-shuffle 16, 32
                # (across the 4 col_grps that share same mfma_row).
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

                # Compute probs: p = exp((qk - new_max) * LOG2E)
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

                # ---- V dequant → LDS (V/scale/zero raw loads were hoisted) -
                # v_packed, vscale_raw, vzero_raw are already in flight from
                # the top of this iteration; only the f32 conversion stays
                # here so the actual consume site is still close to the use.
                vscale_f32 = fx.Int16(vscale_raw).bitcast(fx.Float16).to(fx.Float32)
                vzero_f32 = fx.Int16(vzero_raw).bitcast(fx.Float16).to(fx.Float32)

                # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose).
                # Each lane writes 32 contiguous bf16 (one token, head_dims
                # chunk_in_tok*32..+31) as 4× ds_write_b128 = 4× vec(2,i64).
                v_lds_elem_base = tok_in_tile * HEAD_SIZE + chunk_in_tok * 32
                for w in range_constexpr(4):
                    bf16_elems = []
                    for n in range_constexpr(8):
                        nibble = (v_packed[w] >> (n * 4)) & 0xF
                        elem_f32 = nibble.to(fx.Float32) * vscale_f32 + vzero_f32
                        bf16_elems.append(elem_f32.to(fx.BFloat16))
                    v_bf16 = fx.Vector.from_elements(bf16_elems, dtype=fx.BFloat16)
                    v_lds_i64_idx = (v_lds_elem_base + w * 8) // 4
                    v_bf16.bitcast(fx.Int64).store(
                        kv_lds_i64, [fx.Index(v_lds_i64_idx)]
                    )
                # ── HW V transpose: cross-lane LDS sync ─────────────────────
                # ds_read_tr16_b64 below introduces a cross-lane LDS read:
                # consumer lane t reads bytes written by source lanes
                # 0,4,8,12 (etc.). The compiler's automatic waitcnt insertion
                # is per-lane and has no awareness of the HW transpose's
                # cross-lane forwarding, so it can sink the next iteration's
                # ds_read ahead of these ds_write_b128 ops in the schedule.
                #
                # Two-part fence:
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
                # Without this fence the reordered ds_read produces
                # wrong results on high-partition-count configs (large
                # padded num_partitions); low-partition configs are
                # unaffected.
                rocdl.sched_barrier(0)
                # encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=0)
                #   = 0xF | (7<<4) | (0<<8) | (3<<14) = 0xC07F
                rocdl.s_waitcnt(0xC07F)
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                # P operand B layout matches qk_acc: lane t holds 4 bf16 at
                # K=token=(t/16)*4..+3, N=query=t%16. Just trunc_f to bf16.
                p_op = qk_acc.to(fx.BFloat16)

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
                token_idx = lane >> 2
                hd_sub = (lane & 3) * 4
                lds_u8_ptr = fx.PointerType.get(
                    fx.Uint8.ir_type, fx.AddressSpace.Shared, 1
                )
                v_lane_byte = kv_off + token_idx * (HEAD_SIZE * 2) + hd_sub * 2
                for h in range_constexpr(PV_N_CHUNKS):
                    v_ptr = fx.to_llvm_ptr(
                        fx.inttoptr(lds_u8_ptr, fx.Int64(v_lane_byte + h * 32))
                    )
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
        # acc_pv[h] layout: lane t holds 4 fp32 at M=head_dim=(t/16)*4..+3 + h*16,
        #                                       N=query_row=t%16.
        # 4 fp32/lane = 4 contiguous head_dim values for ONE query_row.
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        out_base = (
            seq * _stride_out_seq + kv_h * _stride_out_head + part * _stride_out_part
        )

        # Gate all global stores by mfma_row < QG. For QG=16 the predicate is
        # tautological (mfma_row ∈ [0,15]); for QG=8 lanes 8..15 (whose MFMA
        # outputs are computational waste) skip the stores entirely.
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

    return tq_decode_gqa6_kernel
