// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention v2 INT8 per-token-head kernel for AMD RDNA3
// (gfx1100). Extends v2 (4 waves, BLOCK_M=64) for INT8 KV cache with
// per-token-head dequantization.
//
// K/V cache is stored as int8 with per-(token,head) float32 scales.
// Dequantization: val_fp16 = int8_val * scale (uses native v_cvt_f16_i16).
// Scale loaded once per K_TILE (16 slots) and broadcast across HEAD_SIZE.
//
// Layout:
//   K cache (int8): [num_blocks, num_kv_heads, head_size/X, block_size, X]
//     where X = 16 (16 int8 values per vec load = same 16-byte alignment)
//   V cache (int8): [num_blocks, num_kv_heads, head_size, block_size]
//   k_scale_cache (fp32): [num_blocks, block_size, num_kv_heads]
//   v_scale_cache (fp32): [num_blocks, block_size, num_kv_heads]

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_rdna3.cuh"

namespace vllm {
namespace prefill_attn_rdna3_v2_int8 {

#if defined(USE_ROCM)

using vllm::prefill_attn_rdna3::bf16_t;
using vllm::prefill_attn_rdna3::bitcast_elem;
using vllm::prefill_attn_rdna3::to_f;
using vllm::prefill_attn_rdna3::to_T;
using vllm::prefill_attn_rdna3::v16bf16;
using vllm::prefill_attn_rdna3::v16fp16;
using vllm::prefill_attn_rdna3::v16i8;
using vllm::prefill_attn_rdna3::v8fp32;
using vllm::prefill_attn_rdna3::v8i32;
using vllm::prefill_attn_rdna3::wmma_mma;
using vllm::prefill_attn_rdna3::wmma_mma_ii8;
using vllm::prefill_attn_rdna3::WmmaNative;

constexpr int K_TILE = 16;
constexpr int M_PER_WAVE = 16;
// THREADS = HEAD_SIZE, NUM_WAVES = HEAD_SIZE/32, BLOCK_M = NUM_WAVES*16.
// These are derived per-template inside the kernel/functions.

__device__ __forceinline__ float wave16_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  return v;
}

__device__ __forceinline__ float wave16_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  return v;
}

// ---------------------------------------------------------------------------
// INT8 K cache load + dequant to LDS (fp16/bf16).
// K cache layout (per-token-head 4D): [blocks, slots, heads, dim]
// dim is contiguous (stride=1). Each vec load reads 16 contiguous int8
// values along dim for a given (block, slot, head).
//
// Strategy: K_TILE(16) slots × D_HIGH(8) chunks of 16 int8 = 128 work items.
// 128 threads → 1 vec load per thread (perfect mapping).
// Distribution: slot = tid / 8, d_chunk = tid % 8.
//
// After dequant (int8→fp16), store to K_lds in the v2 WMMA-friendly layout:
// K_lds_raw[d_high][k_idx][X_FP16] where X_FP16 = 8 (= 16 bytes per entry).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_paged_int8_coop(
    T* __restrict__ K_lds_raw, const int8_t* __restrict__ k_cache_i8,
    const float* __restrict__ k_scale_cache,
    const int* __restrict__ block_table, int seq_idx, int kv_head_idx,
    int start_n, int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_head, int64_t stride_kc_dhi,
    int64_t stride_kc_slot, int64_t stride_ks_blk, int64_t stride_ks_slot,
    int64_t stride_ks_head,
    float* __restrict__ scale_lds,  // [K_TILE] scales in LDS (fp32)
    int tid) {
  constexpr int X_INT8 = 16;
  constexpr int D_CHUNKS =
      HEAD_SIZE / X_INT8;  // chunks per head (8 for 128, 16 for 256)
  constexpr int X_FP16 = 16 / sizeof(T);  // = 8

  // HEAD_SIZE threads: slot = tid / D_CHUNKS, d_chunk = tid % D_CHUNKS
  const int my_k_idx = tid / D_CHUNKS;
  const int my_dh = tid % D_CHUNKS;

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < seq_ctx_len;
  const int log_block = abs_k / block_size;
  const int slot = abs_k - log_block * block_size;
  const int p_block =
      valid_k ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;

  // Load scale (1 per slot, thread with d_chunk==0 loads it)
  if (my_dh == 0) {
    float sc =
        valid_k
            ? k_scale_cache[p_block * stride_ks_blk + slot * stride_ks_slot +
                            kv_head_idx * stride_ks_head]
            : 0.0f;
    scale_lds[my_k_idx] = sc;
  }

  // Load 16 contiguous int8 from K[p_block, slot, kv_head, d_base..d_base+16]
  // 4D layout: offset = block*stride_block + slot*stride_slot +
  // head*stride_head + d stride_kc_dhi repurposed as "16" (chunk stride = 16
  // int8 per chunk) stride_kc_slot repurposed as actual slot stride
  const int d_base = my_dh * X_INT8;
  const int8_t* src = k_cache_i8 + (int64_t)p_block * stride_kc_block +
                      (int64_t)slot * stride_kc_slot +
                      (int64_t)kv_head_idx * stride_kc_head +
                      (int64_t)d_base;  // dim is contiguous

  int8_t k_i8[X_INT8];
  if (valid_k) {
    *(int4*)k_i8 = *(const int4*)src;
  } else {
  #pragma unroll
    for (int i = 0; i < X_INT8; ++i) k_i8[i] = 0;
  }

  // Dequant int8 → fp16 (scale applied post-matmul in attn_step)
  T dequant[X_INT8];
  #pragma unroll
  for (int i = 0; i < X_INT8; ++i) {
    dequant[i] = to_T<T>((float)k_i8[i]);
  }

  // Store to K_lds in v2 layout: K_lds_raw[d_high][k_idx][X_FP16]
  // 16 fp16 values → 2 groups of 8 fp16 (2 d_high entries)
  *(int4*)&K_lds_raw[(my_dh * 2 + 0) * (K_TILE * X_FP16) + my_k_idx * X_FP16] =
      *(int4*)&dequant[0];
  *(int4*)&K_lds_raw[(my_dh * 2 + 1) * (K_TILE * X_FP16) + my_k_idx * X_FP16] =
      *(int4*)&dequant[8];
}

// ---------------------------------------------------------------------------
// INT8 V cache load + dequant to LDS.
// V cache layout (per-token-head 4D): [blocks, slots, heads, dim]
// dim is contiguous (stride=1). Slots are strided (stride = heads × dim).
//
// Strategy: 128 threads cooperatively load K_TILE=16 slots × HEAD_SIZE=128
// = 2048 int8 values = 2048 bytes. Each thread loads one 16-byte vec
// (= 16 int8 values along the dim axis of one slot).
// Distribution: slot = tid / 8, d_chunk = tid % 8.
// 16 slots × 8 chunks = 128 thread assignments.
//
// V_lds target layout: [HEAD_SIZE][K_TILE] (dim outer, slot inner) to match
// the WMMA fragment access pattern in attn_step.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_paged_int8_coop(
    T* __restrict__ V_lds, const int8_t* __restrict__ v_cache_i8,
    const float* __restrict__ v_scale_cache,
    const int* __restrict__ block_table, int seq_idx, int kv_head_idx,
    int start_n, int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_vc_block, int64_t stride_vc_head, int64_t stride_vc_d,
    int64_t stride_vc_slot, int64_t stride_vs_blk, int64_t stride_vs_slot,
    int64_t stride_vs_head,
    float* __restrict__ v_scale_lds,  // [K_TILE] v_scales in LDS (fp32)
    int tid) {
  constexpr int D_CHUNKS = HEAD_SIZE / 16;  // chunks per head

  const int valid_k_count = max(0, min(K_TILE, seq_ctx_len - start_n));

  // Load V scales for this tile (16 slots). Thread 0..15 each load one.
  // Per-slot block lookup to handle tiles that cross a block boundary.
  if (tid < K_TILE) {
    const int abs_k = start_n + tid;
    const bool valid_s = tid < valid_k_count;
    const int log_blk = abs_k / block_size;
    const int slot = abs_k - log_blk * block_size;
    const int p_blk =
        valid_s ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    float sc =
        valid_s ? v_scale_cache[p_blk * stride_vs_blk + slot * stride_vs_slot +
                                kv_head_idx * stride_vs_head]
                : 0.0f;
    v_scale_lds[tid] = sc;
  }

  // HEAD_SIZE threads: slot = tid / D_CHUNKS, d_chunk = tid % D_CHUNKS
  const int my_slot_offset = tid / D_CHUNKS;
  const int my_d_chunk = tid % D_CHUNKS;
  const int d_base = my_d_chunk * 16;
  const int abs_k = start_n + my_slot_offset;
  const bool valid = my_slot_offset < valid_k_count;
  const int log_block = abs_k / block_size;
  const int abs_slot = abs_k - log_block * block_size;
  const int p_block =
      valid ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;

  // Load 16 contiguous int8 from V[p_block, abs_slot, kv_head,
  // d_base..d_base+16]
  int8_t v_i8[16];
  if (valid) {
    const int8_t* src = v_cache_i8 + (int64_t)p_block * stride_vc_block +
                        (int64_t)abs_slot * stride_vc_slot +
                        (int64_t)kv_head_idx * stride_vc_head +
                        (int64_t)d_base * stride_vc_d;
    *(int4*)v_i8 = *(const int4*)src;
  } else {
  #pragma unroll
    for (int i = 0; i < 16; ++i) v_i8[i] = 0;
  }

  // Dequant int8 → fp16 and transpose-store to V_lds[d][k]
  // V_lds layout: [HEAD_SIZE][K_TILE]. We write 16 elements scattered:
  // for each of the 16 d values, write to V_lds[(d_base+i) * K_TILE + my_slot]
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    V_lds[(d_base + i) * K_TILE + my_slot_offset] = to_T<T>((float)v_i8[i]);
  }
}

// ---------------------------------------------------------------------------
// attn_step for INT8 per-token-head. Same as v2 but:
// - After Q@K WMMA: multiply S by k_scales (per-token scale from LDS)
// - Before P@V WMMA: multiply P by v_scales (per-token scale from LDS)
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step_wave_int8(
    const T* __restrict__ K_lds_raw, const T* __restrict__ V_lds,
    T* __restrict__ P_lds_wave,
    const float* __restrict__ k_scale_lds,  // [K_TILE] k_scales (fp32)
    const float* __restrict__ v_scale_lds,  // [K_TILE] v_scales (fp32)
    typename WmmaNative<T>::v16 (&q_frags)[HEAD_SIZE / 16],
    v8fp32 (&out_acc)[HEAD_SIZE / 16], float (&m_state)[8], float (&l_state)[8],
    int wave_q_tile_start, int start_n, int valid_q_count, int valid_k_count,
    float sm_scale, int lane, int lane_lo, int lane_hi) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int FRAGS = HEAD_SIZE / 16;

  // ---- Q @ K (8 WMMAs into s_acc) ----
  v8fp32 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 b_frag;
    int4 lo =
        *(const int4*)&K_lds_raw[(dh * 2 + 0) * (K_TILE * X) + lane_lo * X];
    int4 hi =
        *(const int4*)&K_lds_raw[(dh * 2 + 1) * (K_TILE * X) + lane_lo * X];
    __builtin_memcpy(&b_frag, &lo, 16);
    __builtin_memcpy(((char*)&b_frag) + 16, &hi, 16);
    s_acc = wmma_mma(q_frags[dh], b_frag, s_acc);
  }

  // ---- Apply k_scale per token + softmax scale + mask ----
  // k_scale_lds[lane_lo] contains the scale for the k-token at position
  // lane_lo. S[m][n] = Q[m] @ K[n] * k_scale[n] * sm_scale
  const float k_sc = k_scale_lds[lane_lo];
  const int abs_k = start_n + lane_lo;
  const bool k_in_seg = (lane_lo < valid_k_count);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const bool m_in_q = (m_row < valid_q_count);
    bool keep = m_in_q && k_in_seg;
    if constexpr (CAUSAL_MASK) {
      const int abs_q = wave_q_tile_start + m_row;
      keep = keep && (abs_k <= abs_q);
    }
    // Fuse k_scale into sm_scale: one multiply instead of two
    s_acc[i] = keep ? (s_acc[i] * sm_scale * k_sc) : -INFINITY;
  }

  // ---- Online softmax ----
  float m_ij[8], m_new[8], alpha[8], p_ij[8], l_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_ij[i] = wave16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.0f : __expf(m_state[i] - m_new[i]);
    p_ij[i] = (m_new[i] == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new[i]);
    l_ij[i] = wave16_sum(p_ij[i]);
    l_state[i] = l_state[i] * alpha[i] + l_ij[i];
    m_state[i] = m_new[i];
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      out_acc[dh][i] *= alpha[i];
    }
  }

  // ---- Fuse v_scale into P before transpose ----
  // P[m][k] *= v_scale[k] (multiply each column of P by the v_scale of that
  // token). lane_lo corresponds to the k position.
  const float v_sc = v_scale_lds[lane_lo];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    p_ij[i] *= v_sc;
  }

  // ---- Transpose P → p_frag via WAVE-LOCAL P_lds ----
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    P_lds_wave[m_row * K_TILE + lane_lo] = to_T<T>(p_ij[i]);
  }

  V16 p_frag;
  int4 p_lo = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 0];
  int4 p_hi = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 8];
  __builtin_memcpy(&p_frag, &p_lo, 16);
  __builtin_memcpy(((char*)&p_frag) + 16, &p_hi, 16);

  // ---- P @ V (8 WMMAs, V already dequantized in LDS) ----
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 v_frag;
    int4 v_lo = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 0];
    int4 v_hi = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 8];
    __builtin_memcpy(&v_frag, &v_lo, 16);
    __builtin_memcpy(((char*)&v_frag) + 16, &v_hi, 16);
    out_acc[dh] = wmma_mma(p_frag, v_frag, out_acc[dh]);
  }
}

// ---------------------------------------------------------------------------
// INT8-WMMA-QK path (Phase 1, the O(N^2) cached-prefix cost).
//
// Instead of dequantizing int8 K->fp16 and running fp16 WMMA, this loads K as
// raw int8 into LDS (no dequant, half the LDS), quantizes Q to int8 per-row,
// and runs Q.K^T through the native wmma_i32_16x16x16_iu8 (i32 accumulate),
// dequantizing the score by qscale[row]*kscale[token]. P.V stays fp16 (V
// dequant in loader). Measured ~1.5x over the dequant-fp16 path at long ctx on
// gfx1100 (the int8 attention O(N^2) coefficient drops ~1.8x->~1.0x vs fp16
// Triton). Q is quantized lane-locally (each lane holds its full row, no
// cross-lane reduce); per-row qscale is broadcast via LDS.
// ---------------------------------------------------------------------------

// Load a K tile as raw int8 into LDS: 16 contiguous int8 per (d_chunk, k_idx).
template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_int8_raw(
    int8_t* __restrict__ K_lds_i8, const int8_t* __restrict__ k_cache,
    const float* __restrict__ k_scale_cache,
    const int* __restrict__ block_table, int seq_idx, int kv_head_idx,
    int start_n, int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_head, int64_t stride_kc_slot,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    float* __restrict__ scale_lds, int tid) {
  constexpr int X_INT8 = 16;
  constexpr int D_CHUNKS = HEAD_SIZE / X_INT8;
  const int my_k_idx = tid / D_CHUNKS;
  const int my_dh = tid % D_CHUNKS;
  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < seq_ctx_len;
  const int log_block = abs_k / block_size;
  const int slot = abs_k - log_block * block_size;
  const int p_block =
      valid_k ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;
  if (my_dh == 0) {
    scale_lds[my_k_idx] =
        valid_k
            ? k_scale_cache[p_block * stride_ks_blk + slot * stride_ks_slot +
                            kv_head_idx * stride_ks_head]
            : 0.0f;
  }
  const int d_base = my_dh * X_INT8;
  const int8_t* src = k_cache + (int64_t)p_block * stride_kc_block +
                      (int64_t)slot * stride_kc_slot +
                      (int64_t)kv_head_idx * stride_kc_head + (int64_t)d_base;
  int4 v;
  if (valid_k)
    v = *(const int4*)src;
  else {
    v.x = v.y = v.z = v.w = 0;
  }
  *(int4*)&K_lds_i8[my_dh * (K_TILE * 16) + my_k_idx * 16] = v;
}

// Phase-1 attn step: int8 WMMA Q.K^T + dequant, then online softmax (with lazy
// rescale) and fp16 P.V. Mirrors attn_step_wave_int8 after the QK section.
template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void attn_step_int8qk(
    const int8_t* __restrict__ K_lds_i8, const T* __restrict__ V_lds,
    T* __restrict__ P_lds_wave, const float* __restrict__ k_scale_lds,
    const float* __restrict__ v_scale_lds, const float* __restrict__ qscale_lds,
    v16i8 (&q_i8)[HEAD_SIZE / 16], v8fp32 (&out_acc)[HEAD_SIZE / 16],
    float (&m_state)[8], float (&l_state)[8], int valid_q_count,
    int valid_k_count, float sm_scale, int lane_lo, int lane_hi) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int FRAGS = HEAD_SIZE / 16;

  // ---- Q @ K via int8 WMMA (i32 accumulate) ----
  v8i32 s_i = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    v16i8 kf = *(const v16i8*)&K_lds_i8[dh * (K_TILE * 16) + lane_lo * 16];
    s_i = wmma_mma_ii8(q_i8[dh], kf, s_i);
  }

  // ---- dequant: s = i32 * qscale[row] * kscale[token] * sm ----
  const float k_sc = k_scale_lds[lane_lo];
  const bool k_in = (lane_lo < valid_k_count);
  float s_acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const bool m_in = (m_row < valid_q_count);
    const float q_sc = qscale_lds[m_row];
    s_acc[i] =
        (m_in && k_in) ? ((float)s_i[i] * sm_scale * k_sc * q_sc) : -INFINITY;
  }

  // ---- online softmax (lazy rescale of out_acc) ----
  float m_ij[8], m_new[8], alpha[8], p_ij[8], l_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_ij[i] = wave16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.0f : __expf(m_state[i] - m_new[i]);
    p_ij[i] = (m_new[i] == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new[i]);
    l_ij[i] = wave16_sum(p_ij[i]);
    l_state[i] = l_state[i] * alpha[i] + l_ij[i];
    m_state[i] = m_new[i];
  }
  // alpha == 1 (uniform across wave) for almost every tile once the global max
  // is seen; skip the spill-touching out_acc rescale then. Divergence-free.
  bool need_rescale = false;
  #pragma unroll
  for (int i = 0; i < 8; ++i) need_rescale |= (alpha[i] != 1.0f);
  if (need_rescale) {
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int i = 0; i < 8; ++i) out_acc[dh][i] *= alpha[i];
  }

  const float v_sc = v_scale_lds[lane_lo];
  #pragma unroll
  for (int i = 0; i < 8; ++i) p_ij[i] *= v_sc;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    P_lds_wave[m_row * K_TILE + lane_lo] = to_T<T>(p_ij[i]);
  }
  V16 p_frag;
  int4 p_lo = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 0];
  int4 p_hi = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 8];
  __builtin_memcpy(&p_frag, &p_lo, 16);
  __builtin_memcpy(((char*)&p_frag) + 16, &p_hi, 16);
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 v_frag;
    int4 v_lo = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 0];
    int4 v_hi = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 8];
    __builtin_memcpy(&v_frag, &v_lo, 16);
    __builtin_memcpy(((char*)&v_frag) + 16, &v_hi, 16);
    out_acc[dh] = wmma_mma(p_frag, v_frag, out_acc[dh]);
  }
}

// ---------------------------------------------------------------------------
// Main INT8 per-token-head kernel
// ---------------------------------------------------------------------------

// __launch_bounds__(HEAD_SIZE): the block launches exactly HEAD_SIZE threads
// (THREADS = HEAD_SIZE). Without this the compiler assumes the 1024-thread
// default and budgets VGPRs ultra-conservatively (capped at 192 -> ~600 B/
// thread scratch spill at HS=256). Declaring the real 256-thread bound lets it
// use up to 256 VGPRs, cutting the spill and ~1.27x on long-context prefill.
template <typename T, int HEAD_SIZE>
__global__ void __launch_bounds__(HEAD_SIZE) paged_prefill_attn_kernel_v2_int8(
    T* __restrict__ out, const T* __restrict__ q,
    const T* __restrict__ k_chunk,            // current chunk K (fp16/bf16)
    const T* __restrict__ v_chunk,            // current chunk V (fp16/bf16)
    const int8_t* __restrict__ k_cache,       // paged K cache (int8)
    const int8_t* __restrict__ v_cache,       // paged V cache (int8)
    const float* __restrict__ k_scale_cache,  // [blocks, slots, kv_heads]
    const float* __restrict__ v_scale_cache,  // [blocks, slots, kv_heads]
    const int* __restrict__ block_table, const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens, const int num_query_heads,
    const int num_kv_heads, const int block_size, const int max_blocks_per_seq,
    const float sm_scale, const bool causal,
    // Strides for Q (fp16)
    const int64_t stride_q_token, const int64_t stride_q_head,
    // Strides for K/V chunk (fp16, current tokens)
    const int64_t stride_kc_token, const int64_t stride_kc_head,
    const int64_t stride_vc_token, const int64_t stride_vc_head,
    // Strides for K cache (int8, paged, 5D)
    const int64_t stride_kcache_block, const int64_t stride_kcache_head,
    const int64_t stride_kcache_dhi, const int64_t stride_kcache_slot,
    // Strides for V cache (int8, paged, 4D)
    const int64_t stride_vcache_block, const int64_t stride_vcache_head,
    const int64_t stride_vcache_d, const int64_t stride_vcache_slot,
    // Strides for scale caches
    const int64_t stride_ks_blk, const int64_t stride_ks_slot,
    const int64_t stride_ks_head, const int64_t stride_vs_blk,
    const int64_t stride_vs_slot, const int64_t stride_vs_head,
    // Output strides
    const int64_t stride_o_token, const int64_t stride_o_head) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int FRAGS = HEAD_SIZE / 16;
  constexpr int X_FP16 = 16 / sizeof(T);  // = 8
  constexpr int THREADS = HEAD_SIZE;
  constexpr int NUM_WAVES = THREADS / 32;
  constexpr int BLOCK_M = NUM_WAVES * M_PER_WAVE;

  const int seq_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int q_start_token = cu_seqlens_q[seq_idx];
  const int q_end_token = cu_seqlens_q[seq_idx + 1];
  const int query_len = q_end_token - q_start_token;
  const int seq_len = seq_lens[seq_idx];
  int ctx_len = seq_len - query_len;
  // Defensive clamp on the cached-prefix length. ctx_len drives the phase-1
  // loop and the paged block_table lookups; a garbage seq_len (e.g. an async
  // H2D copy not yet landed on ROCm, or a stale persistent buffer) would make
  // this loop run for billions of iterations and index block_table past
  // max_blocks_per_seq (page fault). Bound it to the per-seq KV capacity; for a
  // valid seq_len this is a no-op.
  const int max_ctx = max_blocks_per_seq * block_size;
  if (ctx_len < 0) ctx_len = 0;
  if (ctx_len > max_ctx) ctx_len = max_ctx;

  const int q_tile_start = q_tile_idx * BLOCK_M;
  if (q_tile_start >= query_len) return;

  const int wave_q_offset = wave_id * M_PER_WAVE;
  const int wave_q_tile_start = q_tile_start + wave_q_offset;
  const int my_q_pos = wave_q_tile_start + lane_lo;
  const bool valid_q = my_q_pos < query_len;
  const int valid_q_count_for_wave =
      max(0, min(M_PER_WAVE, query_len - wave_q_tile_start));

  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  // ---- Shared LDS ----
  __shared__ int8_t K_lds_i8[FRAGS * K_TILE * 16];      // phase1 raw int8 K
  __shared__ T K_lds_raw[FRAGS * 2 * K_TILE * X_FP16];  // phase2 fp16 chunk K
  __shared__ T V_lds[HEAD_SIZE * K_TILE];
  __shared__ T P_lds[NUM_WAVES][M_PER_WAVE * K_TILE];
  __shared__ float k_scale_lds[K_TILE];
  __shared__ float v_scale_lds[K_TILE];
  __shared__ float qscale_lds[NUM_WAVES][M_PER_WAVE];  // per-row Q scale
  T* P_lds_wave = &P_lds[wave_id][0];

  // Per-wave online-softmax state.
  float m_state[8], l_state[8];
  v8fp32 out_acc[FRAGS];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_state[i] = -INFINITY;
    l_state[i] = 0.0f;
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh)
    out_acc[dh] = (v8fp32){0, 0, 0, 0, 0, 0, 0, 0};

  // ---- Load Q (fp16/bf16, temp scope), quantize to int8 per row ----
  // Each lane holds its whole query row, so the max is lane-local (no reduce).
  // q_frags dies at the end of this scope; only q_i8 survives into phase 1,
  // freeing VGPRs. Phase 2 reloads Q in fp16.
  v16i8 q_i8[FRAGS];
  {
    V16 q_frags[FRAGS];
    if (valid_q) {
      const T* q_row = q +
                       (int64_t)(q_start_token + my_q_pos) * stride_q_token +
                       (int64_t)head_idx * stride_q_head;
  #pragma unroll
      for (int dh = 0; dh < FRAGS; ++dh)
        __builtin_memcpy(&q_frags[dh], q_row + dh * 16, sizeof(V16));
    } else {
  #pragma unroll
      for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
        for (int k = 0; k < 16; ++k) q_frags[dh][k] = (E)0;
    }
    float qmax = 1e-8f;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int k = 0; k < 16; ++k)
        qmax = fmaxf(qmax, fabsf(to_f<T>((T)q_frags[dh][k])));
    const float qinv = 127.0f / qmax;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int k = 0; k < 16; ++k) {
        int v = (int)lrintf(to_f<T>((T)q_frags[dh][k]) * qinv);
        v = max(-127, min(127, v));
        q_i8[dh][k] = (int8_t)v;
      }
    if (lane_hi == 0) qscale_lds[wave_id][lane_lo] = qmax * (1.0f / 127.0f);
  }
  __syncthreads();

  // ---- PHASE 1: Cached prefix (INT8 paged cache, int8 WMMA QK, no causal)
  // ----
  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_int8_raw<T, HEAD_SIZE>(
        K_lds_i8, k_cache, k_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq, stride_kcache_block,
        stride_kcache_head, stride_kcache_slot, stride_ks_blk, stride_ks_slot,
        stride_ks_head, k_scale_lds, tid);
    load_v_tile_paged_int8_coop<T, HEAD_SIZE>(
        V_lds, v_cache, v_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq, stride_vcache_block,
        stride_vcache_head, stride_vcache_d, stride_vcache_slot, stride_vs_blk,
        stride_vs_slot, stride_vs_head, v_scale_lds, tid);
    __syncthreads();

    const int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step_int8qk<T, HEAD_SIZE, X_FP16>(
        K_lds_i8, V_lds, P_lds_wave, k_scale_lds, v_scale_lds,
        &qscale_lds[wave_id][0], q_i8, out_acc, m_state, l_state,
        valid_q_count_for_wave, valid_k_count, sm_scale, lane_lo, lane_hi);
    __syncthreads();
  }

  // ---- PHASE 2: Current chunk (fp16, causal) ----
  // Current chunk tokens are NOT yet in the int8 cache — they're in fp16.
  // Reuse the fp16 v2 loaders and the original attn_step (no int8 scales).
  // Import from the v2 namespace.
  const int valid_q_count_for_block =
      max(0, min(BLOCK_M, query_len - q_tile_start));
  const int causal_k_upper =
      causal ? (q_tile_start + valid_q_count_for_block) : query_len;
  const int phase2_k_end = min(query_len, causal_k_upper);

  // Reload Q in fp16 for phase 2 (it was only kept as int8 during phase 1).
  V16 q_frags[FRAGS];
  if (valid_q) {
    const T* q_row = q + (int64_t)(q_start_token + my_q_pos) * stride_q_token +
                     (int64_t)head_idx * stride_q_head;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
      __builtin_memcpy(&q_frags[dh], q_row + dh * 16, sizeof(V16));
  } else {
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int k = 0; k < 16; ++k) q_frags[dh][k] = (E)0;
  }

  for (int start_n = 0; start_n < phase2_k_end; start_n += K_TILE) {
    // Phase 2 uses fp16 K/V from k_chunk/v_chunk (same as v2 kernel).
    // For now, inline a simplified chunk loader.
    // K chunk: load cooperatively into K_lds_raw (same format as fp16 v2)
    {
      constexpr int X = X_FP16;
      constexpr int D_CHUNKS = HEAD_SIZE / 16;  // threads per slot
      const int my_k_idx = tid / D_CHUNKS;
      const int my_dh_base = (tid % D_CHUNKS) * 2;
      const int abs_k = start_n + my_k_idx;
      const bool valid_k = abs_k < query_len;
      const T* row =
          valid_k
              ? (k_chunk + (int64_t)(q_start_token + abs_k) * stride_kc_token +
                 (int64_t)kv_head_idx * stride_kc_head)
              : nullptr;
  #pragma unroll
      for (int dh = 0; dh < 2; ++dh) {
        const int d_high = my_dh_base + dh;
        int4 vec;
        if (valid_k) {
          vec = *(const int4*)(row + d_high * X);
        } else {
          vec.x = vec.y = vec.z = vec.w = 0;
        }
        *(int4*)&K_lds_raw[d_high * (K_TILE * X) + my_k_idx * X] = vec;
      }
    }
    // V chunk
    {
      constexpr int TPS = HEAD_SIZE / K_TILE;  // threads per slot
  #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int my_k = tid / TPS;
        const int my_dc = (tid % TPS) + p * TPS;
        const int d_base = my_dc * 8;
        const int abs_k = start_n + my_k;
        const bool valid = abs_k < query_len;
        int4 vec;
        if (valid) {
          const T* src =
              v_chunk + (int64_t)(q_start_token + abs_k) * stride_vc_token +
              (int64_t)kv_head_idx * stride_vc_head + (int64_t)d_base;
          vec = *(const int4*)src;
        } else {
          vec.x = vec.y = vec.z = vec.w = 0;
        }
        T tmp[8];
        __builtin_memcpy(tmp, &vec, 16);
  #pragma unroll
        for (int e = 0; e < 8; ++e) {
          V_lds[(d_base + e) * K_TILE + my_k] = tmp[e];
        }
      }
    }
    // Scales = 1.0 for chunk (not quantized yet)
    if (tid < K_TILE) {
      k_scale_lds[tid] = 1.0f;
      v_scale_lds[tid] = 1.0f;
    }
    __syncthreads();

    const int valid_k_count = min(K_TILE, query_len - start_n);
    attn_step_wave_int8<T, HEAD_SIZE, X_FP16, /*CAUSAL_MASK=*/true>(
        K_lds_raw, V_lds, P_lds_wave, k_scale_lds, v_scale_lds, q_frags,
        out_acc, m_state, l_state, wave_q_tile_start, start_n,
        valid_q_count_for_wave, valid_k_count, sm_scale, lane, lane_lo,
        lane_hi);
    __syncthreads();
  }

  // ---- Epilogue: divide by L, write output ----
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const int abs_m_row = wave_q_offset + m_row;
    const int abs_q_pos = q_tile_start + abs_m_row;
    if (abs_q_pos >= query_len) continue;
    const float l_inv = 1.0f / (l_state[i] + 1e-10f);
    T* out_row = out + (int64_t)(q_start_token + abs_q_pos) * stride_o_token +
                 (int64_t)head_idx * stride_o_head;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      const int out_col = dh * 16 + lane_lo;
      out_row[out_col] = to_T<T>(out_acc[dh][i] * l_inv);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
void launch_paged_prefill_attn_v2_int8(
    T* out, const T* q, const T* k_chunk, const T* v_chunk,
    const int8_t* k_cache, const int8_t* v_cache, const float* k_scale_cache,
    const float* v_scale_cache, const int* block_table, const int* cu_seqlens_q,
    const int* seq_lens, int num_seqs, int num_query_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq, int max_query_len, float sm_scale,
    bool causal, int64_t stride_q_token, int64_t stride_q_head,
    int64_t stride_kc_token, int64_t stride_kc_head, int64_t stride_vc_token,
    int64_t stride_vc_head, int64_t stride_kcache_block,
    int64_t stride_kcache_head, int64_t stride_kcache_dhi,
    int64_t stride_kcache_slot, int64_t stride_vcache_block,
    int64_t stride_vcache_head, int64_t stride_vcache_d,
    int64_t stride_vcache_slot, int64_t stride_ks_blk, int64_t stride_ks_slot,
    int64_t stride_ks_head, int64_t stride_vs_blk, int64_t stride_vs_slot,
    int64_t stride_vs_head, int64_t stride_o_token, int64_t stride_o_head,
    cudaStream_t stream) {
  constexpr int THREADS = HEAD_SIZE;
  constexpr int NUM_WAVES = THREADS / 32;
  constexpr int BLOCK_M = NUM_WAVES * M_PER_WAVE;
  const int q_blocks = (max_query_len + BLOCK_M - 1) / BLOCK_M;
  dim3 block(THREADS);
  dim3 grid(num_seqs, num_query_heads, q_blocks);
  paged_prefill_attn_kernel_v2_int8<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache, k_scale_cache, v_scale_cache,
      block_table, cu_seqlens_q, seq_lens, num_query_heads, num_kv_heads,
      block_size, max_blocks_per_seq, sm_scale, causal, stride_q_token,
      stride_q_head, stride_kc_token, stride_kc_head, stride_vc_token,
      stride_vc_head, stride_kcache_block, stride_kcache_head,
      stride_kcache_dhi, stride_kcache_slot, stride_vcache_block,
      stride_vcache_head, stride_vcache_d, stride_vcache_slot, stride_ks_blk,
      stride_ks_slot, stride_ks_head, stride_vs_blk, stride_vs_slot,
      stride_vs_head, stride_o_token, stride_o_head);
}

// Explicit instantiations
template void launch_paged_prefill_attn_v2_int8<half, 128>(
    half*, const half*, const half*, const half*, const int8_t*, const int8_t*,
    const float*, const float*, const int*, const int*, const int*, int, int,
    int, int, int, int, float, bool, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, cudaStream_t);

template void launch_paged_prefill_attn_v2_int8<bf16_t, 128>(
    bf16_t*, const bf16_t*, const bf16_t*, const bf16_t*, const int8_t*,
    const int8_t*, const float*, const float*, const int*, const int*,
    const int*, int, int, int, int, int, int, float, bool, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, cudaStream_t);

#endif  // USE_ROCM

}  // namespace prefill_attn_rdna3_v2_int8
}  // namespace vllm

// ---------------------------------------------------------------------------
// Torch-callable entry point (registered in torch_bindings.cpp)
// ---------------------------------------------------------------------------

#if defined(USE_ROCM)
void paged_prefill_attn_rdna3_int8(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  using namespace vllm::prefill_attn_rdna3_v2_int8;

  const int num_seqs = seq_lens.size(0);
  const int num_query_heads = q.size(1);
  const int num_kv_heads = k_scale_cache.size(2);  // [blocks, slots, heads]
  // k_cache: [blocks, slots, heads, dim] — block_size = k_cache.size(1)
  const int block_size = k_cache.size(1);
  const int max_blocks_per_seq = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "paged_prefill_attn_rdna3_int8: only fp16/bf16 supported");
  TORCH_CHECK(k_cache.dtype() == at::kChar, "k_cache must be int8");

  // Derive head_size from cache layout: padded_hs - scale_pad(4)
  const int head_size = q.size(2);

  // Macro to reduce boilerplate for head_size dispatch
  #define LAUNCH_INT8(T, HS)                                                  \
    launch_paged_prefill_attn_v2_int8<T, HS>(                                 \
        (T*)out.data_ptr(), (const T*)q.data_ptr(),                           \
        (const T*)k_chunk.data_ptr(), (const T*)v_chunk.data_ptr(),           \
        (const int8_t*)k_cache.data_ptr(), (const int8_t*)v_cache.data_ptr(), \
        (const float*)k_scale_cache.data_ptr(),                               \
        (const float*)v_scale_cache.data_ptr(),                               \
        (const int*)block_table.data_ptr(),                                   \
        (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(), \
        num_seqs, num_query_heads, num_kv_heads, block_size,                  \
        max_blocks_per_seq, (int)max_query_len, (float)sm_scale, causal,      \
        q.stride(0), q.stride(1), k_chunk.stride(0), k_chunk.stride(1),       \
        v_chunk.stride(0), v_chunk.stride(1), k_cache.stride(0),              \
        k_cache.stride(2), (int64_t)1, k_cache.stride(1), v_cache.stride(0),  \
        v_cache.stride(2), (int64_t)1, v_cache.stride(1),                     \
        k_scale_cache.stride(0), k_scale_cache.stride(1),                     \
        k_scale_cache.stride(2), v_scale_cache.stride(0),                     \
        v_scale_cache.stride(1), v_scale_cache.stride(2), out.stride(0),      \
        out.stride(1), stream)

  if (q.dtype() == at::kHalf) {
    using T = half;
    switch (head_size) {
      case 64:
        LAUNCH_INT8(T, 64);
        break;
      case 128:
        LAUNCH_INT8(T, 128);
        break;
      case 256:
        LAUNCH_INT8(T, 256);
        break;
      default:
        TORCH_CHECK(false,
                    "paged_prefill_attn_rdna3_int8: unsupported head_size=",
                    head_size, " (supported: 64, 128, 256)");
    }
  } else {
    using T = vllm::prefill_attn_rdna3::bf16_t;
    switch (head_size) {
      case 64:
        LAUNCH_INT8(T, 64);
        break;
      case 128:
        LAUNCH_INT8(T, 128);
        break;
      case 256:
        LAUNCH_INT8(T, 256);
        break;
      default:
        TORCH_CHECK(false,
                    "paged_prefill_attn_rdna3_int8: unsupported head_size=",
                    head_size, " (supported: 64, 128, 256)");
    }
  }
  #undef LAUNCH_INT8
}
#else
void paged_prefill_attn_rdna3_int8(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  TORCH_CHECK(false, "paged_prefill_attn_rdna3_int8 requires ROCm");
}
#endif
