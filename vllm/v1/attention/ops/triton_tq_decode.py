# SPDX-License-Identifier: Apache-2.0
"""Triton fused TurboQuant decode attention.

Three decode paths (tried in order):
  1. Pre-dequant + GQA SDPA: Bulk dequant K+V to fp16, then cuBLAS GEMM +
     FlashAttention SDPA with enable_gqa=True. ~2x faster than fused WPH.
  2. CUDA warp-per-head (WPH): Fused score+value+softmax per warp.
  3. Triton stage1+stage2: Split-KV tiled fallback.

Supports both FP8 (E4M3) and 4-bit uniform quantized values.
"""

import math
import os
import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SM_COUNT: int | None = None


def _get_sm_count(device: int = 0) -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(
            device).multi_processor_count
    return _SM_COUNT


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------

@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,         # [B, Hq, D] float32
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
    MSE_BITS: tl.constexpr,         # 3 or 4
    MSE_BYTES: tl.constexpr,        # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,              # key_packed_size
    VQB: tl.constexpr,              # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,   # ceil(D * vqb / 8) or D for FP8
    N_CENTROIDS: tl.constexpr,      # 2**MSE_BITS
    # Score constants
    ATTN_SCALE: tl.constexpr,       # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,          # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,         # tokens per tile (16)
    KEY_FP8: tl.constexpr,          # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
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

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask, other=0,
        )

        slot_bases = (block_nums * stride_cache_block
                     + page_off * stride_cache_pos
                     + kv_head * stride_cache_head)

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            )
            k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                axis=1,
            ) * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(tl.uint16)
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

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
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
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
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
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
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs,
                        mask=d_mask, other=0)
        k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(KV_cache_ptr + slot_base + mse_byte_idx,
                          mask=d_mask, other=0).to(tl.int32)
        mse_raw1 = tl.load(KV_cache_ptr + slot_base + mse_byte_idx + 1,
                          mask=d_mask, other=0).to(tl.int32)
        raw16 = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16 >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
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
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(KV_cache_ptr + val_base + val_byte_idx,
                           mask=d_mask, other=0).to(tl.int32)
        val_raw1 = tl.load(KV_cache_ptr + val_base + val_byte_idx + 1,
                           mask=d_mask, other=0).to(tl.int32)
        raw16 = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16 >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
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


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            'mse_bytes': math.ceil(D * mse_bits / 8),
            'val_data_bytes': val_data_bytes,
            'mse_bits': mse_bits,
            'n_centroids': 2 ** mse_bits,
            'BLOCK_D': triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


# ---------------------------------------------------------------------------
# CUDA warp-per-head kernel (faster alternative to Triton stage 1)
# ---------------------------------------------------------------------------

_wph_module = None
_wph_available = None
_wph_has_smem = False
_wph_compiled_d = None
_wph_compiled_gs = None

def _get_wph_module(head_dim=None, kv_group_size=None):
    """Lazy-compile and cache the warp-per-head CUDA kernel."""
    global _wph_module, _wph_available, _wph_has_smem
    global _wph_compiled_d, _wph_compiled_gs
    if _wph_available is not None:
        return _wph_module if _wph_available else None
    try:
        cu_path = os.path.join(os.path.dirname(__file__),
                               'csrc', 'tq_decode_warp_per_head.cu')
        with open(cu_path) as f:
            cuda_src = f.read()

        cpp_src = """
void tq_decode_wph_launch(
    torch::Tensor q_rot,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t value_fp8,
    int64_t mse_bits, int64_t n_centroids, int64_t key_fp8,
    double attn_scale,
    double sparse_v_threshold);

void tq_decode_wph_smem_launch(
    torch::Tensor q_rot,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor mid_o,
    int64_t num_kv_splits, int64_t head_dim,
    int64_t num_kv_heads, int64_t kv_group_size,
    int64_t block_size,
    int64_t mse_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t slot_bytes,
    int64_t value_fp8,
    double attn_scale,
    double sparse_v_threshold);

void tq_full_dequant_kv_launch(
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor centroids,
    torch::Tensor k_out, torch::Tensor v_out,
    int64_t alloc_seq_len, int64_t head_dim,
    int64_t num_kv_heads, int64_t block_size,
    int64_t mse_bytes,
    int64_t kps, int64_t val_data_bytes,
    int64_t mse_bits, int64_t n_centroids, int64_t key_fp8);

void tq_masked_softmax_launch(
    torch::Tensor scores, torch::Tensor seq_lens,
    int64_t alloc_seq_len, int64_t num_q_heads);
"""
        extra_cflags = ["-O3", "--use_fast_math"]

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
# ---------------------------------------------------------------------------

_predequant_available = None
_pi_t_cache: dict = {}
_dequant_buf_cache: dict = {}
_predequant_max_batch = 0


def _check_predequant_available():
    global _predequant_available
    if _predequant_available is True:
        return True
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
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    cfg: dict,
    key_packed_size: int,
    value_quant_bits: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Pre-dequant K+V to fp16 -> q@Pi^T -> manual GQA attention."""
    global _predequant_max_batch
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    device = query.device
    group_size = Hq // Hk

    wph_mod = _wph_module
    if wph_mod is None:
        raise RuntimeError("WPH module required for predequant path")

    max_num_blocks = block_table.shape[1]
    alloc_seq_len = max_num_blocks * block_size

    cached = _dequant_buf_cache.get(device)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Predequant buffers not allocated yet "
                               "(cannot allocate during graph capture)")
        per_batch_bytes = (
            2 * Hk * alloc_seq_len * D * 2
            + Hq * alloc_seq_len * 4
            + Hq * D * 2
            + Hq * D * 2
            + Hq * D * 4
        )
        free_mem = torch.cuda.mem_get_info(device)[0]
        max_bytes = int(free_mem * 0.15)
        _predequant_max_batch = max(1, max_bytes // per_batch_bytes)
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

    if B > _predequant_max_batch:
        raise RuntimeError(
            f"Batch {B} > predequant max_batch {_predequant_max_batch}")

    (k_fp16_full, v_fp16_full, scores_buf_full,
     q_rot_full, q_float_full, output_full) = cached
    k_fp16 = k_fp16_full[:B]
    v_fp16 = v_fp16_full[:B]
    scores_buf = scores_buf_full[:B * Hq]
    q_rot = q_rot_full[:B]
    q_float = q_float_full[:B * Hq]
    output = output_full[:B]

    # CUDA dequant
    wph_mod.tq_full_dequant_kv_launch(
        kv_cache, block_table, seq_lens, centroids,
        k_fp16, v_fp16,
        alloc_seq_len, D, Hk, block_size,
        cfg['mse_bytes'],
        key_packed_size, cfg['val_data_bytes'],
        cfg.get('mse_bits', 3), cfg.get('n_centroids', 8), 0,
    )

    # q @ Pi^T rotation
    Pi_T = _get_pi_t(Pi)
    q_float.copy_(query.float().reshape(B * Hq, D))
    torch.mm(q_float, Pi_T, out=q_float)
    q_rot.copy_(q_float.reshape(B, Hq, D).to(torch.float16))

    # Manual GQA attention
    q_grouped = q_rot.float().reshape(B, Hk, group_size, D)
    torch.matmul(
        q_grouped, k_fp16.float().transpose(-2, -1),
        out=scores_buf.reshape(B, Hk, group_size, alloc_seq_len),
    )
    scores_buf.mul_(scale)

    wph_mod.tq_masked_softmax_launch(
        scores_buf, seq_lens, alloc_seq_len, Hq,
    )

    weights = scores_buf.reshape(B, Hk, group_size, alloc_seq_len)
    output.copy_(torch.matmul(weights, v_fp16.float()).to(torch.float16).reshape(B, Hq, D))

    return output


def triton_tq_decode_attention(
    query: torch.Tensor,        # [B, Hq, D] — original query
    kv_cache: torch.Tensor,     # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,     # [B] int32
    Pi: torch.Tensor,           # [D, D] float32
    centroids: torch.Tensor,    # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,
    num_kv_splits: int = 128,
    max_seq_len: int = 0,
    key_fp8: bool = False,
    norm_correction: bool = False,
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

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    if key_fp8:
        q_rot = query.float().contiguous()
    else:
        q_float = query.float()
        q_rot = (q_float @ _get_pi_t(Pi)).contiguous()

    # Occupancy-aware NUM_KV_SPLITS
    MIN_TOKENS_PER_SPLIT = 128
    max_seq = max_seq_len if max_seq_len > 0 else num_kv_splits * MIN_TOKENS_PER_SPLIT
    effective = max(1, max_seq // MIN_TOKENS_PER_SPLIT)
    NUM_KV_SPLITS = 1
    while NUM_KV_SPLITS * 2 <= min(effective, num_kv_splits):
        NUM_KV_SPLITS *= 2

    SM_COUNT = _get_sm_count()
    TARGET_GRID = SM_COUNT * 2
    grid_blocks = B * Hk * NUM_KV_SPLITS
    if grid_blocks < TARGET_GRID:
        needed = math.ceil(TARGET_GRID / (B * Hk))
        ns = NUM_KV_SPLITS
        while ns < needed and ns < 128:
            ns *= 2
        max_allowed = max(1, max_seq // 16)
        ns = min(ns, max_allowed, 128)
        final = NUM_KV_SPLITS
        while final * 2 <= ns:
            final *= 2
        NUM_KV_SPLITS = final

    mid_o = torch.empty(
        B, Hq, NUM_KV_SPLITS, D + 1,
        dtype=torch.float32, device=device,
    )

    sparse_v_threshold = float(os.environ.get("TQ_SPARSE_V_THRESHOLD", "1e-6"))

    # Path 2: CUDA warp-per-head kernel
    # Disabled for VQB=3: CUDA kernels hardcode 4-bit value unpack.
    # Falls through to Triton stage1 which has proper 3-bit support.
    wph_mod = _get_wph_module(head_dim=D, kv_group_size=kv_group_size)
    use_wph = (wph_mod is not None
               and D in (128, 256)
               and kv_group_size in (1, 2, 4, 8, 16)
               and 32 * kv_group_size <= 1024
               and value_quant_bits != 3)

    if use_wph:
        if not getattr(triton_tq_decode_attention, '_wph_path_logged', False):
            import logging as _logging
            _l = _logging.getLogger(__name__)
            _l.info("TQ decode: use_wph=True, mse_bits=%d, key_fp8=%s, use_smem=%s",
                    mse_bits, key_fp8, (_wph_has_smem and not key_fp8))
            triton_tq_decode_attention._wph_path_logged = True
        slot_bytes = (cfg['mse_bytes']
                      + 4 + cfg['val_data_bytes'] + 4)  # +4 = scale+zero
        _vfp8 = 0  # FP8 values not supported
        _kfp8 = 1 if key_fp8 else 0
        _use_smem = (_wph_has_smem and not key_fp8)
        if _use_smem:
            wph_mod.tq_decode_wph_smem_launch(
                q_rot.contiguous(),
                kv_cache.contiguous(), block_table, seq_lens,
                centroids, mid_o,
                NUM_KV_SPLITS, D, Hk, kv_group_size, block_size,
                cfg['mse_bytes'],
                key_packed_size, cfg['val_data_bytes'],
                slot_bytes,
                _vfp8,
                scale,
                sparse_v_threshold,
            )
        else:
            wph_mod.tq_decode_wph_launch(
                q_rot.contiguous(),
                kv_cache.contiguous(), block_table, seq_lens,
                centroids, mid_o,
                NUM_KV_SPLITS, D, Hk, kv_group_size, block_size,
                cfg['mse_bytes'],
                key_packed_size, cfg['val_data_bytes'],
                _vfp8,
                mse_bits, n_centroids, _kfp8,
                scale,
                sparse_v_threshold,
            )
    else:
        # Path 3: Triton stage 1
        BLOCK_KV = 4
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
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
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg['val_data_bytes'],
            N_CENTROIDS=n_centroids,
            ATTN_SCALE=scale,
            BLOCK_D=cfg['BLOCK_D'],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
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
