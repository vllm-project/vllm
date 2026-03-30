# SPDX-License-Identifier: Apache-2.0
"""Triton fused TurboQuant decode attention — Checkpoint 5.

Three decode paths (tried in order):
  1. Pre-dequant + GQA SDPA: Bulk dequant K+V to fp16, then cuBLAS GEMM +
     FlashAttention SDPA with enable_gqa=True. ~2x faster than fused WPH.
  2. CUDA warp-per-head (WPH): Fused score+value+softmax per warp.
  3. Triton stage1+stage2: Split-KV tiled fallback.

Supports both FP8 (E4M3) and uniform quantized (2/4-bit) values.
"""

import math
import os
import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------

@triton.jit
def _tq_decode_stage1(
    # Precomputed query projections
    Q_rot_ptr,         # [B, Hq, D] float32
    Q_proj_ptr,        # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,      # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,   # [B, max_num_blocks] int32
    Seq_lens_ptr,      # [B] int32
    # TQ parameters
    Centroids_ptr,     # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,         # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb, stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block, stride_cache_pos, stride_cache_head,  # KV cache
    stride_bt_b,       # block_table stride per batch
    stride_mid_b, stride_mid_h, stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,       # KV cache block_size (pages)
    PADDED_SLOT: tl.constexpr,      # padded slot bytes
    MAX_NUM_BLOCKS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,    # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,         # 2
    MSE_BYTES: tl.constexpr,        # ceil(D * mse_bits / 8)
    QJL_BYTES: tl.constexpr,        # ceil(D / 8)
    KPS: tl.constexpr,              # key_packed_size
    VQB: tl.constexpr,              # value_quant_bits (2, 4, or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,   # ceil(D * vqb / 8) or D for FP8
    N_CENTROIDS: tl.constexpr,      # 4 for 2-bit MSE
    # Score constants
    CORRECTION: tl.constexpr,       # sqrt(pi/2) / D
    ATTN_SCALE: tl.constexpr,       # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,          # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,         # tokens per tile (16)
):
    bid = tl.program_id(0)   # batch index
    hid = tl.program_id(1)   # q_head index
    sid = tl.program_id(2)   # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets — used for all vectorized operations
    d_offs = tl.arange(0, BLOCK_D)       # [BLOCK_D]
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)    # [BLOCK_KV]

    # Load query vectors: q_rot and q_proj — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0)
    q_proj = tl.load(Q_proj_ptr + q_base + d_offs, mask=d_mask, other=0.0)

    # Load centroids (small — 4 for 2-bit MSE)
    c0 = tl.load(Centroids_ptr)
    c1 = tl.load(Centroids_ptr + 1)
    c2 = tl.load(Centroids_ptr + 2)
    c3 = tl.load(Centroids_ptr + 3)

    # Precompute centroid lookup: q_rot * centroid[k] for k=0..3
    qc0 = q_rot * c0  # [BLOCK_D]
    qc1 = q_rot * c1
    qc2 = q_rot * c2
    qc3 = q_rot * c3

    # Precompute byte/bit index vectors for gather loads
    mse_byte_idx = d_offs // 4            # [BLOCK_D]
    mse_bit_shift = (d_offs % 4) * 2     # [BLOCK_D]
    qjl_byte_idx = d_offs // 8            # [BLOCK_D]
    qjl_bit_shift = d_offs % 8           # [BLOCK_D]

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Precompute block table base for this batch
    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range                       # [BLOCK_KV]
        kv_mask = kv_offs < split_end                      # [BLOCK_KV]

        # Page table lookups for BLOCK_KV tokens
        page_idx = kv_offs // BLOCK_SIZE                   # [BLOCK_KV]
        page_off = kv_offs % BLOCK_SIZE                    # [BLOCK_KV]
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask, other=0,
        )

        # Slot bases for all BLOCK_KV tokens: [BLOCK_KV]
        slot_bases = (block_nums * stride_cache_block
                     + page_off * stride_cache_pos
                     + kv_head * stride_cache_head)

        # ============================================================
        # MSE 2-BIT UNPACK + CENTROID LOOKUP (batched)
        # ============================================================
        # Load MSE bytes: [BLOCK_KV, BLOCK_D]
        mse_addrs = slot_bases[:, None] + mse_byte_idx[None, :]
        mse_raw = tl.load(
            KV_cache_ptr + mse_addrs,
            mask=kv_mask[:, None] & d_mask[None, :], other=0,
        ).to(tl.int32)
        mse_idx = (mse_raw >> mse_bit_shift[None, :]) & 0x3  # [BLOCK_KV, BLOCK_D]

        # Centroid lookup: q_rot * centroid[mse_idx] for each token
        qc_vals = tl.where(
            mse_idx == 0, qc0[None, :],
            tl.where(mse_idx == 1, qc1[None, :],
                     tl.where(mse_idx == 2, qc2[None, :], qc3[None, :])))
        # term1[k] = sum_d(q_rot[d] * centroid[mse_idx[k,d]]) for each token k
        term1 = tl.sum(tl.where(d_mask[None, :], qc_vals, 0.0), axis=1)  # [BLOCK_KV]

        # ============================================================
        # QJL 1-BIT SIGN UNPACK + DOT PRODUCT (batched)
        # ============================================================
        sign_addrs = slot_bases[:, None] + MSE_BYTES + qjl_byte_idx[None, :]
        sign_raw = tl.load(
            KV_cache_ptr + sign_addrs,
            mask=kv_mask[:, None] & d_mask[None, :], other=0,
        ).to(tl.int32)
        sign_bits = (sign_raw >> qjl_bit_shift[None, :]) & 1
        signs = sign_bits.to(tl.float32) * 2.0 - 1.0      # [BLOCK_KV, BLOCK_D]

        # term2[k] = sum_d(q_proj[d] * signs[k,d])
        term2 = tl.sum(
            tl.where(d_mask[None, :], q_proj[None, :] * signs, 0.0),
            axis=1,
        )  # [BLOCK_KV]

        # ============================================================
        # LOAD NORMS for BLOCK_KV tokens (fp16 → fp32)
        # ============================================================
        norm_bases = slot_bases + MSE_BYTES + QJL_BYTES     # [BLOCK_KV]

        n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
        vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        g_lo = tl.load(KV_cache_ptr + norm_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
        g_hi = tl.load(KV_cache_ptr + norm_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
        res_norms = (g_lo | (g_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        scores = vec_norms * (term1 + CORRECTION * res_norms * term2) * ATTN_SCALE
        scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)                       # [BLOCK_KV]

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS                        # [BLOCK_KV]

        if VQB == 8:
            # FP8 E4M3
            val_addrs = val_bases[:, None] + d_offs[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            )
            values = val_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        elif VQB == 4:
            vb_idx = d_offs // 2                            # [BLOCK_D]
            vb_shift = (d_offs % 2) * 4                     # [BLOCK_D]
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            # Per-token scale and zero: load from after val data
            sc_bases = val_bases + VAL_DATA_BYTES           # [BLOCK_KV]
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 2
            vb_idx = d_offs // 4
            vb_shift = (d_offs % 4) * 2
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0x3).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        # acc += sum_k(p[k] * values[k, :])  — batched across BLOCK_KV tokens
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)  # [BLOCK_D]
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result: mid_o[bid, hid, sid, :D] = acc/l_prev
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    # Store log-sum-exp: mid_o[bid, hid, sid, D] = m_prev + log(l_prev)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+QJL+norms) and V to fp16
# ---------------------------------------------------------------------------

@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,          # [B, Hk, max_seq, D] float16
    V_out_ptr,          # [B, Hk, max_seq, D] float16
    stride_ko_b, stride_ko_h, stride_ko_s,
    stride_vo_b, stride_vo_h, stride_vo_s,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    CORRECTION: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Full dequant: reconstruct K (MSE centroids * norm) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx)
    slot_base = (block_num * stride_cache_block
                + page_off * stride_cache_pos
                + hid * stride_cache_head)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    c0 = tl.load(Centroids_ptr)
    c1 = tl.load(Centroids_ptr + 1)
    c2 = tl.load(Centroids_ptr + 2)
    c3 = tl.load(Centroids_ptr + 3)

    mse_byte_idx = d_offs // 4
    mse_bit_shift = (d_offs % 4) * 2
    mse_raw = tl.load(KV_cache_ptr + slot_base + mse_byte_idx,
                      mask=d_mask, other=0).to(tl.int32)
    mse_idx = (mse_raw >> mse_bit_shift) & 0x3
    k_mse = tl.where(mse_idx == 0, c0,
             tl.where(mse_idx == 1, c1,
             tl.where(mse_idx == 2, c2, c3)))

    qjl_byte_idx = d_offs // 8
    qjl_bit_shift = d_offs % 8
    sign_raw = tl.load(KV_cache_ptr + slot_base + MSE_BYTES + qjl_byte_idx,
                       mask=d_mask, other=0).to(tl.int32)
    sign_bits = (sign_raw >> qjl_bit_shift) & 1
    signs = sign_bits.to(tl.float32) * 2.0 - 1.0

    norm_base = slot_base + MSE_BYTES + QJL_BYTES
    n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
    n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
    vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    g_lo = tl.load(KV_cache_ptr + norm_base + 2).to(tl.uint16)
    g_hi = tl.load(KV_cache_ptr + norm_base + 3).to(tl.uint16)
    res_norm = (g_lo | (g_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    k_recon = vec_norm * (k_mse + CORRECTION * res_norm * signs)
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx,
                          mask=d_mask, other=0).to(tl.int32)
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 8:
        val_raw = tl.load(KV_cache_ptr + val_base + d_offs,
                          mask=d_mask, other=0)
        v_vals = val_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)


# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}
_fused_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        if value_quant_bits == 8:
            val_data_bytes = D  # FP8: 1 byte per element
        else:
            val_data_bytes = math.ceil(D * value_quant_bits / 8)
        _no_qjl = os.environ.get("TQ_NO_QJL", "1") == "1"
        corr = 0.0 if _no_qjl else math.sqrt(math.pi / 2) / D
        cfg = {
            'mse_bytes': math.ceil(D * mse_bits / 8),
            'qjl_bytes': math.ceil(D / 8),
            'val_data_bytes': val_data_bytes,
            'value_fp8': value_quant_bits == 8,
            'correction': corr,
            'BLOCK_D': triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def _get_fused_pi_s(Pi, S):
    """Get cached fused [Pi; S]^T matrix, keyed by tensor data pointers."""
    key = (Pi.data_ptr(), S.data_ptr())
    Pi_S = _fused_cache.get(key)
    if Pi_S is None:
        Pi_S = torch.cat([Pi, S], dim=0).T.contiguous()  # [D, 2D]
        _fused_cache[key] = Pi_S
    return Pi_S


# ---------------------------------------------------------------------------
# CUDA warp-per-head kernel (faster alternative to Triton stage 1)
# ---------------------------------------------------------------------------

_wph_module = None
_wph_available = None
_wph_has_smem = False
# Model-specific D and GS for minimal template instantiations.
# Set by first call based on actual model config; triggers recompile
# with -DTQ_HEAD_DIM=D -DTQ_KV_GROUP_SIZE=GS for just 2 kernels.
_wph_compiled_d = None
_wph_compiled_gs = None

def _get_wph_module(head_dim=None, kv_group_size=None):
    """Lazy-compile and cache the warp-per-head CUDA kernel.

    When head_dim and kv_group_size are provided, compiles with
    -DTQ_HEAD_DIM=<D> -DTQ_KV_GROUP_SIZE=<GS> to emit only 2 template
    instantiations (original + smem) instead of 20, keeping binary
    small enough for CUDA graph capture.
    """
    global _wph_module, _wph_available, _wph_has_smem
    global _wph_compiled_d, _wph_compiled_gs
    if _wph_available is not None:
        return _wph_module if _wph_available else None
    try:
        cu_path = os.path.join(os.path.dirname(__file__),
                               'tq_decode_warp_per_head.cu')
        with open(cu_path) as f:
            cuda_src = f.read()

        cpp_src = """
void tq_decode_wph_launch(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t value_fp8,
    double correction, double attn_scale,
    double sparse_v_threshold);

void tq_decode_wph_smem_launch(
    torch::Tensor q_rot, torch::Tensor q_proj,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t slot_bytes,
    int64_t value_fp8,
    double correction, double attn_scale,
    double sparse_v_threshold);

void tq_full_dequant_kv_launch(
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor k_out, torch::Tensor v_out,
    int64_t alloc_seq_len, int64_t head_dim,
    int64_t num_kv_heads, int64_t block_size,
    int64_t mse_bytes, int64_t qjl_bytes,
    int64_t kps, int64_t val_data_bytes,
    double correction);

void tq_masked_softmax_launch(
    torch::Tensor scores, torch::Tensor seq_lens,
    int64_t alloc_seq_len, int64_t num_q_heads);
"""
        extra_cflags = ["-O3", "--use_fast_math"]

        # When model-specific dims are known, restrict instantiations
        if head_dim is not None and kv_group_size is not None:
            extra_cflags.append(f"-DTQ_HEAD_DIM={head_dim}")
            extra_cflags.append(f"-DTQ_KV_GROUP_SIZE={kv_group_size}")
            _wph_compiled_d = head_dim
            _wph_compiled_gs = kv_group_size
            logger.info("TQ WPH: compiling for D=%d GS=%d (2 kernels)",
                        head_dim, kv_group_size)

        from torch.utils.cpp_extension import load_inline
        _wph_module = load_inline(
            name="tq_decode_wph",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["tq_decode_wph_launch", "tq_decode_wph_smem_launch",
                       "tq_full_dequant_kv_launch", "tq_masked_softmax_launch"],
            verbose=False,
            extra_cuda_cflags=extra_cflags,
        )
        _wph_available = True
        _wph_has_smem = hasattr(_wph_module, 'tq_decode_wph_smem_launch')
        logger.info("TQ WPH CUDA kernel compiled (smem=%s)", _wph_has_smem)
        return _wph_module
    except Exception as e:
        logger.warning("TQ WPH CUDA kernel failed to compile, "
                       "falling back to Triton: %s", e)
        _wph_available = False
        return None


# ---------------------------------------------------------------------------
# Pre-dequant + manual GQA decode path (~2-4x faster than fused WPH)
# CUDA-graph compatible: CUDA dequant kernel (fixed grid, reads seq_lens
# from GPU), cuBLAS GEMMs, CUDA masked softmax. All fixed-pointer ops.
# ---------------------------------------------------------------------------

_predequant_available = None
_pi_t_cache: dict = {}  # Pi.data_ptr() → Pi.T contiguous
_dequant_buf_cache: dict = {}  # device → (k, v, scores, max_batch)
_predequant_max_batch = 0  # largest B that fits in memory


def _check_predequant_available():
    """Check if CUDA dequant + manual GQA path is available."""
    global _predequant_available
    if _predequant_available is True:
        return True
    # Don't trigger compilation — check the already-compiled module.
    # The module gets compiled with dequant+softmax by _get_wph_module().
    # NOTE: Don't cache False — _wph_module may not be compiled yet on
    # first call (warmup calls predequant check before WPH path triggers
    # compilation).
    if _wph_module is not None and hasattr(_wph_module, 'tq_full_dequant_kv_launch'):
        _predequant_available = True
        logger.info("Pre-dequant manual GQA path available (CUDA-graph compatible)")
        return True
    return False


def _get_pi_t(Pi):
    """Cached Pi.T contiguous."""
    key = Pi.data_ptr()
    pit = _pi_t_cache.get(key)
    if pit is None:
        pit = Pi.T.contiguous()
        _pi_t_cache[key] = pit
    return pit


def _predequant_gqa_sdpa_decode(
    query: torch.Tensor,        # [B, Hq, D]
    kv_cache: torch.Tensor,     # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,     # [B] int32
    Pi: torch.Tensor,           # [D, D] float32
    centroids: torch.Tensor,    # [n_centroids] float32
    scale: float,
    cfg: dict,
    key_packed_size: int,
    value_quant_bits: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Pre-dequant K+V to fp16 → q@Pi^T → manual GQA attention.

    CUDA-graph compatible: CUDA dequant (fixed grid), cuBLAS GEMMs (fixed
    shapes), CUDA masked softmax. Single pre-allocated buffer set shared
    across batch sizes via views.
    """
    global _predequant_max_batch
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    device = query.device
    group_size = Hq // Hk

    wph_mod = _wph_module  # already compiled by main dispatch path
    if wph_mod is None:
        raise RuntimeError("WPH module required for predequant path")

    # Fixed alloc_seq_len = max_model_len (derived from block_table)
    max_num_blocks = block_table.shape[1]
    alloc_seq_len = max_num_blocks * block_size

    # One-time buffer allocation: find max batch that fits in memory.
    # Cannot allocate during CUDA graph capture — skip if not ready.
    cached = _dequant_buf_cache.get(device)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Predequant buffers not allocated yet "
                               "(cannot allocate during graph capture)")
        # Compute per-batch memory: K + V (fp16) + scores (fp32) +
        # q_rot (fp16) + output (fp16) + q_float (fp32) + k_float (fp32)
        per_batch_bytes = (
            2 * Hk * alloc_seq_len * D * 2  # K + V, fp16
            + Hq * alloc_seq_len * 4         # scores, fp32
            + Hq * D * 2                     # q_rot, fp16
            + Hq * D * 2                     # output, fp16
            + Hq * D * 4                     # q_float, fp32
        )
        free_mem = torch.cuda.mem_get_info(device)[0]
        # Use up to 15% of free memory for predequant buffers
        max_bytes = int(free_mem * 0.15)
        _predequant_max_batch = max(1, max_bytes // per_batch_bytes)
        # Cap at reasonable value (512 = max CUDA graph capture batch)
        _predequant_max_batch = min(_predequant_max_batch, 512)

        if _predequant_max_batch < 1:
            raise RuntimeError(
                f"Pre-dequant needs {per_batch_bytes / 1e9:.2f} GiB/batch "
                f"but only {free_mem / 1e9:.2f} GiB free")

        alloc_B = _predequant_max_batch
        k_fp16 = torch.empty(alloc_B, Hk, alloc_seq_len, D,
                              dtype=torch.float16, device=device)
        v_fp16 = torch.empty(alloc_B, Hk, alloc_seq_len, D,
                              dtype=torch.float16, device=device)
        scores_buf = torch.empty(alloc_B * Hq, alloc_seq_len,
                                  dtype=torch.float32, device=device)
        q_rot_buf = torch.empty(alloc_B, Hq, D,
                                 dtype=torch.float16, device=device)
        q_float_buf = torch.empty(alloc_B * Hq, D,
                                   dtype=torch.float32, device=device)
        output_buf = torch.empty(alloc_B, Hq, D,
                                  dtype=torch.float16, device=device)
        _dequant_buf_cache[device] = (k_fp16, v_fp16, scores_buf,
                                       q_rot_buf, q_float_buf, output_buf)
        logger.info("Pre-dequant buffers allocated: max_batch=%d "
                    "(%.2f GiB)", alloc_B,
                    (alloc_B * per_batch_bytes) / 1e9)
        cached = _dequant_buf_cache[device]

    # Check batch fits in pre-allocated buffers
    if B > _predequant_max_batch:
        raise RuntimeError(
            f"Batch {B} > predequant max_batch {_predequant_max_batch}")

    (k_fp16_full, v_fp16_full, scores_buf_full,
     q_rot_full, q_float_full, output_full) = cached
    # Views into pre-allocated buffers (same base pointers for CUDA graphs)
    k_fp16 = k_fp16_full[:B]
    v_fp16 = v_fp16_full[:B]
    scores_buf = scores_buf_full[:B * Hq]
    q_rot = q_rot_full[:B]
    q_float = q_float_full[:B * Hq]
    output = output_full[:B]

    # Phase 2: CUDA dequant (fixed grid, reads seq_lens from GPU)
    wph_mod.tq_full_dequant_kv_launch(
        kv_cache, block_table, seq_lens, centroids,
        k_fp16, v_fp16,
        alloc_seq_len, D, Hk, block_size,
        cfg['mse_bytes'], cfg['qjl_bytes'],
        key_packed_size, cfg['val_data_bytes'],
        cfg['correction'],
    )

    # Phase 3: q @ Pi^T rotation (cuBLAS GEMM, pre-allocated output)
    Pi_T = _get_pi_t(Pi)
    # query.float() → q_float (pre-allocated)
    q_float.copy_(query.float().reshape(B * Hq, D))
    # q_float @ Pi_T → q_rot (via fp32 intermediate then fp16)
    torch.mm(q_float, Pi_T, out=q_float)  # reuse q_float as output
    q_rot.copy_(q_float.reshape(B, Hq, D).to(torch.float16))

    # Phase 4: Manual GQA attention
    # scores = q_grouped @ K^T * scale
    q_grouped = q_rot.float().reshape(B, Hk, group_size, D)
    torch.matmul(
        q_grouped, k_fp16.float().transpose(-2, -1),
        out=scores_buf.reshape(B, Hk, group_size, alloc_seq_len),
    )
    scores_buf.mul_(scale)

    # Phase 5: Masked softmax (CUDA kernel)
    wph_mod.tq_masked_softmax_launch(
        scores_buf, seq_lens, alloc_seq_len, Hq,
    )

    # Phase 6: output = weights @ V (cuBLAS GEMM)
    weights = scores_buf.reshape(B, Hk, group_size, alloc_seq_len)
    output.copy_(torch.matmul(weights, v_fp16.float()).to(torch.float16).reshape(B, Hq, D))

    return output


def triton_tq_decode_attention(
    query: torch.Tensor,        # [B, Hq, D] — original query
    kv_cache: torch.Tensor,     # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,     # [B] int32
    Pi: torch.Tensor,           # [D, D] float32
    S: torch.Tensor,            # [D, D] float32
    centroids: torch.Tensor,    # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,
    num_kv_splits: int = 128,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """Launch fused TQ decode attention.

    Tries three paths in order:
      1. Pre-dequant + GQA SDPA (~2x faster, requires enable_gqa support)
      2. CUDA warp-per-head kernel
      3. Triton stage1+stage2 fallback

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    max_num_blocks = block_table.shape[1]
    n_centroids = centroids.shape[0]
    kv_group_size = Hq // Hk
    device = query.device

    # Get cached layout constants
    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Path 1: Pre-dequant + manual GQA attention — DISABLED.
    # O(alloc_seq_len) dequant per step is too expensive with CUDA graphs
    # (fixed grid = max_model_len). Only beneficial for short sequences.
    # TODO: Re-enable if we find a way to use actual seq_len for grid size.

    # Path 2+3: Fused WPH or Triton (need q_rot, q_proj)
    _no_qjl = cfg['correction'] == 0.0
    if _no_qjl:
        # No QJL: only need q_rot = q @ Pi.T (D×D instead of D×2D)
        q_float = query.float()
        q_rot = (q_float @ _get_pi_t(Pi)).contiguous()
        q_proj = torch.zeros_like(q_rot)
    else:
        Pi_S = _get_fused_pi_s(Pi, S)
        q_float = query.float()
        q_both = (q_float @ Pi_S).contiguous()  # [B, Hq, 2D]
        q_rot = q_both[:, :, :D]
        q_proj = q_both[:, :, D:]

    # Occupancy-aware NUM_KV_SPLITS: increase splits for low batch sizes
    # to ensure enough grid blocks to fill the GPU.
    # Grid = (B, Hk, NUM_KV_SPLITS). At B=1, Hk=2, default 64 splits gives
    # only 128 blocks for 96 SMs (1.3 blocks/SM). Doubling to 128 splits
    # gives 256 blocks and ~20% improvement at long sequences.
    # For B≥2, 64 splits already gives ≥256 blocks — no change needed.
    MIN_TOKENS_PER_SPLIT = 128
    max_seq = max_seq_len if max_seq_len > 0 else num_kv_splits * MIN_TOKENS_PER_SPLIT
    effective = max(1, max_seq // MIN_TOKENS_PER_SPLIT)
    NUM_KV_SPLITS = 1
    while NUM_KV_SPLITS * 2 <= min(effective, num_kv_splits):
        NUM_KV_SPLITS *= 2

    # Increase splits if grid is too small for GPU occupancy.
    # Target: B * Hk * splits >= 192 (96 SMs × 2 blocks/SM).
    SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    TARGET_GRID = SM_COUNT * 2
    grid_blocks = B * Hk * NUM_KV_SPLITS
    if grid_blocks < TARGET_GRID:
        needed = math.ceil(TARGET_GRID / (B * Hk))
        ns = NUM_KV_SPLITS
        while ns < needed and ns < 128:
            ns *= 2
        # Cap: at least 16 tokens per split for meaningful work
        max_allowed = max(1, max_seq // 16)
        ns = min(ns, max_allowed, 128)
        # Ensure power of 2 and at least base
        final = NUM_KV_SPLITS
        while final * 2 <= ns:
            final *= 2
        NUM_KV_SPLITS = final

    mid_o = torch.empty(
        B, Hq, NUM_KV_SPLITS, D + 1,
        dtype=torch.float32, device=device,
    )

    # Sparse V dequant: skip V load when attention weight < threshold.
    # Default 1e-6 — lossless quality, ~15% ITL win at 16K+ context.
    sparse_v_threshold = float(os.environ.get("TQ_SPARSE_V_THRESHOLD", "1e-6"))

    # Path 2: CUDA warp-per-head kernel (pass D/GS for minimal compilation)
    wph_mod = _get_wph_module(head_dim=D, kv_group_size=kv_group_size)
    use_wph = (wph_mod is not None
               and D in (128, 256)
               and kv_group_size in (1, 2, 4, 8, 16)
               and 32 * kv_group_size <= 1024)

    if use_wph:
        # Compute unpadded slot size for smem kernel
        # Key: mse_bytes + qjl_bytes + 4 (norms). Value: val_data_bytes + 4 (scale+zero) or just val_data_bytes for FP8.
        val_overhead = 0 if cfg['value_fp8'] else 4
        slot_bytes = (cfg['mse_bytes'] + cfg['qjl_bytes']
                      + 4 + cfg['val_data_bytes'] + val_overhead)
        # Use smem kernel (5b: 1.3-1.9x faster) when available
        _vfp8 = 1 if cfg['value_fp8'] else 0
        if _wph_has_smem:
            wph_mod.tq_decode_wph_smem_launch(
                q_rot.contiguous(), q_proj.contiguous(),
                kv_cache.contiguous(), block_table, seq_lens,
                centroids, mid_o,
                NUM_KV_SPLITS, D, Hk, kv_group_size, block_size,
                cfg['mse_bytes'], cfg['qjl_bytes'],
                key_packed_size, cfg['val_data_bytes'],
                slot_bytes,
                _vfp8,
                cfg['correction'], scale,
                sparse_v_threshold,
            )
        else:
            wph_mod.tq_decode_wph_launch(
                q_rot.contiguous(), q_proj.contiguous(),
                kv_cache.contiguous(), block_table, seq_lens,
                centroids, mid_o,
                NUM_KV_SPLITS, D, Hk, kv_group_size, block_size,
                cfg['mse_bytes'], cfg['qjl_bytes'],
                key_packed_size, cfg['val_data_bytes'],
                _vfp8,
                cfg['correction'], scale,
                sparse_v_threshold,
            )
    else:
        # Path 3: Triton stage 1
        BLOCK_KV = 4
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot, q_proj,
            kv_cache,
            block_table, seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            NUM_Q_HEADS=Hq,
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            PADDED_SLOT=padded_slot,
            MAX_NUM_BLOCKS=max_num_blocks,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg['mse_bytes'],
            QJL_BYTES=cfg['qjl_bytes'],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg['val_data_bytes'],
            N_CENTROIDS=n_centroids,
            CORRECTION=cfg['correction'],
            ATTN_SCALE=scale,
            BLOCK_D=cfg['BLOCK_D'],
            BLOCK_KV=BLOCK_KV,
            num_warps=4,
            num_stages=2,
        )

    # Stage 2: Reduce across KV splits
    output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        output.stride(0), output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg['BLOCK_D'],
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output.to(query.dtype)
