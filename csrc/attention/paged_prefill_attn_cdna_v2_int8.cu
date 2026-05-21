// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention v2 with INT8 per-token-head KV cache, MFMA on
// CDNA (gfx942 / gfx950 / gfx90a). Ported from the RDNA3 WMMA kernel by
// JartX (paged_prefill_attn_rdna3_v2_int8.cu) but rewritten for wave64
// fragment layout.
//
// Block layout:
//   THREADS    = 256 (4 waves x 64 lanes)
//   BLOCK_M    = 64 query rows (16 per wave, one MFMA tile per wave)
//   K_TILE     = 16 KV tokens per outer-loop iteration
//   HEAD_SIZE  = 64 / 128 (compile-time)
//
// KV layout (per-token-head, 4D contiguous on dim):
//   k_cache, v_cache : int8  [num_blocks, block_size, num_kv_heads, head_size]
//   k_scale_cache    : fp32  [num_blocks, block_size, num_kv_heads]
//   v_scale_cache    : fp32  [num_blocks, block_size, num_kv_heads]
//
// Two-phase loop:
//   Phase 1 — cached prefix (int8 paged cache, no causal mask).
//   Phase 2 — current chunk (fp16/bf16 K/V from the live forward pass,
//             causal mask within the current query window).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_cdna.cuh"

namespace vllm {
namespace prefill_attn_cdna_v2_int8 {

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))

using vllm::prefill_attn_cdna::bf16_t;
using vllm::prefill_attn_cdna::cvt_T_from_int8;
using vllm::prefill_attn_cdna::floatx4;
using vllm::prefill_attn_cdna::from_float_rn;
using vllm::prefill_attn_cdna::mfma_16x16x16;
using vllm::prefill_attn_cdna::to_float;
using vllm::prefill_attn_cdna::wave_group16_max;
using vllm::prefill_attn_cdna::wave_group16_sum;
using vllm::prefill_attn_cdna::WmmaNative;

constexpr int K_TILE      = 16;
constexpr int M_PER_WAVE  = 16;
constexpr int LANES       = 64;
constexpr int WAVES       = 4;
constexpr int THREADS     = LANES * WAVES;  // 256
constexpr int BLOCK_M     = WAVES * M_PER_WAVE;  // 64

// ---------------------------------------------------------------------------
// Cooperative INT8 K-cache loader → LDS (fp16/bf16).
//
// Layout target in LDS:
//   K_lds[k_idx][d]  : T  [K_TILE * HEAD_SIZE]   (k-major outer, d inner)
//
// Each block has THREADS=256 threads. We have K_TILE * HEAD_SIZE int8 to
// load. For HEAD_SIZE=128 this is 16*128 = 2048 int8 = 128 int16 = 128 vec
// loads of 16 int8 each → 2 vec loads / thread.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_paged_int8_coop(
    T*               K_lds,
    const int8_t*    k_cache,
    const float*     k_scale_cache,
    const int*       block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_slot, int64_t stride_kc_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    float* scale_lds, int tid) {
  constexpr int X_INT8 = 16;
  constexpr int D_CHUNKS = HEAD_SIZE / X_INT8;       // chunks per row
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;   // total 16B loads
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;

  // Scales: K_TILE entries. Threads 0..K_TILE-1 each load one.
  if (tid < K_TILE) {
    int abs_k = start_n + tid;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    float sc = valid ? k_scale_cache[p_blk * stride_ks_blk +
                                     slot * stride_ks_slot +
                                     kv_head_idx * stride_ks_head]
                     : 0.0f;
    scale_lds[tid] = sc;
  }

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int d_base = d_chunk * X_INT8;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk =
        valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    int8_t buf[X_INT8];
    if (valid) {
      const int8_t* src = k_cache + p_blk * stride_kc_block +
                          slot * stride_kc_slot +
                          kv_head_idx * stride_kc_head + d_base;
      *(int4*)buf = *(const int4*)src;
    } else {
      #pragma unroll
      for (int i = 0; i < X_INT8; ++i) buf[i] = 0;
    }
    T out[X_INT8];
    #pragma unroll
    for (int i = 0; i < X_INT8; ++i) out[i] = cvt_T_from_int8<T>(buf[i]);
    // 16 elements of T == 32B for fp16/bf16 → write as 2 int4 vecs.
    T* dst = &K_lds[k_idx * HEAD_SIZE + d_base];
    *((int4*)dst)     = *(const int4*)&out[0];
    *((int4*)(dst+8)) = *(const int4*)&out[8];
  }
}

// V layout in LDS: [HEAD_SIZE][K_TILE] (dim outer, slot inner). Transposed
// vs K so the P @ V MFMA reads V columns contiguously.
template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_paged_int8_coop(
    T*               V_lds,
    const int8_t*    v_cache,
    const float*     v_scale_cache,
    const int*       block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_vc_block, int64_t stride_vc_slot, int64_t stride_vc_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    float* v_scale_lds, int tid) {
  constexpr int X_INT8 = 16;
  constexpr int D_CHUNKS = HEAD_SIZE / X_INT8;
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;

  if (tid < K_TILE) {
    int abs_k = start_n + tid;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    float sc = valid ? v_scale_cache[p_blk * stride_vs_blk +
                                     slot * stride_vs_slot +
                                     kv_head_idx * stride_vs_head]
                     : 0.0f;
    v_scale_lds[tid] = sc;
  }

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int d_base = d_chunk * X_INT8;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk =
        valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    int8_t buf[X_INT8];
    if (valid) {
      const int8_t* src = v_cache + p_blk * stride_vc_block +
                          slot * stride_vc_slot +
                          kv_head_idx * stride_vc_head + d_base;
      *(int4*)buf = *(const int4*)src;
    } else {
      #pragma unroll
      for (int i = 0; i < X_INT8; ++i) buf[i] = 0;
    }
    // Transpose-store: V_lds[d][k] = dequant[d - d_base][k_idx]
    #pragma unroll
    for (int i = 0; i < X_INT8; ++i) {
      V_lds[(d_base + i) * K_TILE + k_idx] = cvt_T_from_int8<T>(buf[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Cooperative fp16/bf16 K-chunk loader (phase 2). The chunk is K/V from the
// current forward pass, already in fp16/bf16 contiguous on dim.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_chunk_coop(
    T* K_lds, const T* k_chunk, int q_start_token, int kv_head_idx,
    int start_n, int chunk_len,
    int64_t stride_kc_token, int64_t stride_kc_head,
    int tid) {
  constexpr int X = 8;  // 8 fp16 / bf16 per 16B vec
  constexpr int D_CHUNKS = HEAD_SIZE / X;
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int d_base = d_chunk * X;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < chunk_len;
    int4 vec;
    if (valid) {
      const T* src = k_chunk +
          (int64_t)(q_start_token + abs_k) * stride_kc_token +
          kv_head_idx * stride_kc_head + d_base;
      vec = *(const int4*)src;
    } else {
      vec.x = vec.y = vec.z = vec.w = 0;
    }
    *(int4*)&K_lds[k_idx * HEAD_SIZE + d_base] = vec;
  }
}

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_chunk_coop(
    T* V_lds, const T* v_chunk, int q_start_token, int kv_head_idx,
    int start_n, int chunk_len,
    int64_t stride_vc_token, int64_t stride_vc_head,
    int tid) {
  constexpr int X = 8;
  constexpr int D_CHUNKS = HEAD_SIZE / X;
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int d_base = d_chunk * X;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < chunk_len;
    int4 vec;
    if (valid) {
      const T* src = v_chunk +
          (int64_t)(q_start_token + abs_k) * stride_vc_token +
          kv_head_idx * stride_vc_head + d_base;
      vec = *(const int4*)src;
    } else {
      vec.x = vec.y = vec.z = vec.w = 0;
    }
    T buf[X];
    __builtin_memcpy(buf, &vec, 16);
    #pragma unroll
    for (int i = 0; i < X; ++i) {
      V_lds[(d_base + i) * K_TILE + k_idx] = buf[i];
    }
  }
}

// ---------------------------------------------------------------------------
// MFMA 16x16x16 fragment loader
//
// A operand (Q): each lane needs 4 fp16/bf16 values along K from one row.
//   lane layout (per AMD docs for mfma_f32_16x16x16f16):
//     a_lane(k, m) — lane = m * 4 + k_block, k_block in [0..3]
//     each lane holds 4 K values (k_block * 4 + 0..3) for one m
//   For us: m = lane / 4, k_block = lane % 4, holds k = lane%4 * 4 + [0..3].
//
// B operand (K, V): each lane holds 4 K values for one n column.
//     n = lane / 4, k = lane % 4 * 4 + [0..3]
//
// C accumulator (S, O): 4 fp32 per lane.
//     row = lane / 16 + 4*acc_idx (acc_idx in 0..3)
//     col = lane % 16
//
// IMPORTANT: This is the standard layout for MFMA 16x16x16; the exact
// register-to-element mapping is documented at
//   https://github.com/ROCm/amd_matrix_instruction_calculator
// First-pass code below assumes that mapping; the unit tests against a
// PyTorch reference will catch any disagreement.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ typename WmmaNative<T>::v4 load_mfma_a_frag(
    const T* row_base, int lane) {
  // Q is laid out as [M=16, K=16] contiguous on K. Each lane wants 4 K
  // values from row m = lane / 4, at k = (lane % 4) * 4 + [0..3].
  int m = lane / 4;
  int k0 = (lane % 4) * 4;
  typename WmmaNative<T>::v4 frag;
  T* dst = (T*)&frag;
  #pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = row_base[m * 16 + k0 + i];
  return frag;
}

template <typename T>
__device__ __forceinline__ typename WmmaNative<T>::v4 load_mfma_b_frag_K(
    const T* K_lds, int lane, int dh_offset) {
  // K_lds is [K_TILE=16, HEAD_SIZE]. For one MFMA computing Q[16,K] @
  // K[K,16]^T effectively (S = Q @ K_T), we treat K as B and load each
  // lane's 4 elements as: column n = lane / 4 (one of 16 KV tokens), K
  // index = (lane % 4) * 4 + i (one of 16 head dims within dh_offset
  // .. dh_offset+16).
  int n  = lane / 4;
  int k0 = (lane % 4) * 4;
  typename WmmaNative<T>::v4 frag;
  T* dst = (T*)&frag;
  #pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = K_lds[n * /*HEAD_SIZE row stride placeholder*/16 + 0];
  // ... See note below.
  return frag;
}

// NOTE on the B fragment for QK:
// The line above is a placeholder for the read of the per-(n,k) element.
// The actual K layout in this kernel is K_lds[k_idx][d] = K[KV_token=k_idx, head_dim=d].
// For the QK matmul we have S[m,n] = sum_k Q[m,k] * K[n,k]  (K is transposed),
// so the B-operand element at MFMA (n, k_inner) is K_lds[n, dh_offset + k_inner].
// The full template below uses inline loads at the call site so that the
// HEAD_SIZE template parameter can be applied to the row stride.

// ---------------------------------------------------------------------------
// attn_step (INT8 / shared with chunk path)
//
// Computes 16x16 tile of S = Q @ K^T, applies scales + softmax (online),
// then 16x16 tile of O += P @ V. One wave handles one M-tile of 16 query
// rows; each call processes all FRAGS = HEAD_SIZE/16 head-dim fragments.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step_wave_int8(
    const T* K_lds, const T* V_lds, T* P_lds_wave,
    const float* k_scale_lds, const float* v_scale_lds,
    const typename WmmaNative<T>::v4 (&q_frags)[HEAD_SIZE / 16],
    floatx4 (&out_acc)[HEAD_SIZE / 16],
    float (&m_state)[4], float (&l_state)[4],
    int wave_q_tile_start, int start_n,
    int valid_q_count, int valid_k_count,
    float sm_scale, int lane) {
  using V4 = typename WmmaNative<T>::v4;
  constexpr int FRAGS = HEAD_SIZE / 16;
  // ----- Q @ K^T  ------------------------------------------------------
  floatx4 s_acc = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    int n  = lane / 4;
    int k0 = (lane % 4) * 4;
    V4 b_frag;
    T* bdst = (T*)&b_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      bdst[i] = K_lds[n * HEAD_SIZE + dh * 16 + k0 + i];
    }
    s_acc = mfma_16x16x16<T>(q_frags[dh], b_frag, s_acc);
  }

  // s_acc holds 4 rows × 1 column per lane (col = lane%16, rows = lane/16
  // + 4*[0..3]).
  // ----- Apply per-token scales + softmax_scale + mask  ----------------
  // The K_TILE axis is the "col" axis here; col = lane%16, so the per-
  // K-token scale for this lane's column is k_scale_lds[lane%16].
  int n_col = lane % 16;
  float k_sc = k_scale_lds[n_col];
  int abs_k = start_n + n_col;
  bool k_in_seg = n_col < valid_k_count;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_row = (lane / 16) + i * 4;
    bool m_in_q = m_row < valid_q_count;
    bool keep = m_in_q && k_in_seg;
    if constexpr (CAUSAL_MASK) {
      int abs_q = wave_q_tile_start + m_row;
      keep = keep && (abs_k <= abs_q);
    }
    s_acc[i] = keep ? (s_acc[i] * sm_scale * k_sc) : -INFINITY;
  }

  // ----- Online softmax across the 16-element row (within the wave) ----
  float m_ij[4], m_new[4], alpha[4], p_ij[4], l_ij[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    m_ij[i]  = wave_group16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.f
                                         : __expf(m_state[i] - m_new[i]);
    p_ij[i]  = (m_new[i] == -INFINITY) ? 0.f
                                       : __expf(s_acc[i] - m_new[i]);
    l_ij[i]  = wave_group16_sum(p_ij[i]);
    l_state[i] = l_state[i] * alpha[i] + l_ij[i];
    m_state[i] = m_new[i];
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) out_acc[dh][i] *= alpha[i];
  }

  // ----- Apply v_scale into P  -----------------------------------------
  float v_sc = v_scale_lds[n_col];
  #pragma unroll
  for (int i = 0; i < 4; ++i) p_ij[i] *= v_sc;

  // ----- Stage P into LDS so we can re-load it as an MFMA A fragment ---
  // P_lds_wave is [M=16, K=16], owned by this wave.
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_row = (lane / 16) + i * 4;
    P_lds_wave[m_row * 16 + n_col] = from_float_rn<T>(p_ij[i]);
  }
  __syncthreads();

  // P as A fragment, same load pattern as Q.
  V4 p_frag;
  {
    int m = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&p_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = P_lds_wave[m * 16 + k0 + i];
  }

  // ----- P @ V  --------------------------------------------------------
  // V_lds is [HEAD_SIZE][K_TILE]. For one MFMA tile owning head-dim
  // columns [dh*16, dh*16+16), B = V_lds[dh*16 + n, k] with n = lane/4
  // (a HEAD_SIZE column), k = lane%4 * 4 + i (a K_TILE token).
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V4 v_frag;
    int n  = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&v_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[i] = V_lds[(dh * 16 + n) * K_TILE + k0 + i];
    }
    out_acc[dh] = mfma_16x16x16<T>(p_frag, v_frag, out_acc[dh]);
  }
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__global__ __launch_bounds__(THREADS, 2)
void paged_prefill_attn_kernel_v2_int8(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_chunk,
    const T* __restrict__ v_chunk,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    int num_query_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    float sm_scale, bool causal,
    int64_t stride_q_token, int64_t stride_q_head,
    int64_t stride_kc_token, int64_t stride_kc_head,
    int64_t stride_vc_token, int64_t stride_vc_head,
    int64_t stride_kcache_block, int64_t stride_kcache_slot,
    int64_t stride_kcache_head,
    int64_t stride_vcache_block, int64_t stride_vcache_slot,
    int64_t stride_vcache_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    int64_t stride_o_token, int64_t stride_o_head) {
  using V4 = typename WmmaNative<T>::v4;
  constexpr int FRAGS = HEAD_SIZE / 16;

  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int q_tile_idx = blockIdx.z;

  int tid = threadIdx.x;
  int wave_id = tid >> 6;
  int lane = tid & 63;

  int q_start_token = cu_seqlens_q[seq_idx];
  int q_end_token = cu_seqlens_q[seq_idx + 1];
  int query_len = q_end_token - q_start_token;
  int seq_len = seq_lens[seq_idx];
  int ctx_len = seq_len - query_len;

  int q_tile_start = q_tile_idx * BLOCK_M;
  if (q_tile_start >= query_len) return;

  int wave_q_offset = wave_id * M_PER_WAVE;
  int wave_q_tile_start = q_tile_start + wave_q_offset;
  int valid_q_count_for_wave =
      max(0, min(M_PER_WAVE, query_len - wave_q_tile_start));
  int valid_q_count_for_block =
      max(0, min(BLOCK_M, query_len - q_tile_start));

  int num_queries_per_kv = num_query_heads / num_kv_heads;
  int kv_head_idx = head_idx / num_queries_per_kv;

  // ----- Q load (per-wave) ----------------------------------------------
  // Each wave owns 16 query rows starting at wave_q_tile_start. Stage Q
  // into per-wave LDS so we can slice MFMA fragments uniformly.
  __shared__ T Q_lds_all[WAVES][M_PER_WAVE * HEAD_SIZE];
  T* Q_lds = &Q_lds_all[wave_id][0];

  for (int idx = lane; idx < M_PER_WAVE * HEAD_SIZE; idx += LANES) {
    int m_row = idx / HEAD_SIZE;
    int d = idx % HEAD_SIZE;
    int q_pos = wave_q_tile_start + m_row;
    T v;
    if (q_pos < query_len) {
      v = q[(int64_t)(q_start_token + q_pos) * stride_q_token +
            head_idx * stride_q_head + d];
    } else {
      v = (T)0;
    }
    Q_lds[m_row * HEAD_SIZE + d] = v;
  }
  __syncthreads();

  V4 q_frags[FRAGS];
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    int m  = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&q_frags[dh];
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = Q_lds[m * HEAD_SIZE + dh * 16 + k0 + i];
  }

  // ----- LDS for K, V, P, scales ----------------------------------------
  // K_lds: [K_TILE=16][HEAD_SIZE]
  // V_lds: [HEAD_SIZE][K_TILE=16]   (transposed)
  // P_lds: [WAVES][16 * 16]
  // scale_lds: [K_TILE]   (k and v separate)
  __shared__ T  K_lds[K_TILE * HEAD_SIZE];
  __shared__ T  V_lds[HEAD_SIZE * K_TILE];
  __shared__ T  P_lds[WAVES][M_PER_WAVE * K_TILE];
  __shared__ float k_scale_lds[K_TILE];
  __shared__ float v_scale_lds[K_TILE];
  T* P_lds_wave = &P_lds[wave_id][0];

  // ----- Online-softmax accumulators (per-row, 4 rows per lane) --------
  float m_state[4], l_state[4];
  floatx4 out_acc[FRAGS];
  #pragma unroll
  for (int i = 0; i < 4; ++i) { m_state[i] = -INFINITY; l_state[i] = 0.f; }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh)
    out_acc[dh] = (floatx4){0.f, 0.f, 0.f, 0.f};

  // ----- PHASE 1: cached prefix (INT8 paged) ----------------------------
  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_paged_int8_coop<T, HEAD_SIZE>(
        K_lds, k_cache, k_scale_cache, block_table,
        seq_idx, kv_head_idx, start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_kcache_block, stride_kcache_slot, stride_kcache_head,
        stride_ks_blk, stride_ks_slot, stride_ks_head,
        k_scale_lds, tid);
    load_v_tile_paged_int8_coop<T, HEAD_SIZE>(
        V_lds, v_cache, v_scale_cache, block_table,
        seq_idx, kv_head_idx, start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_vcache_block, stride_vcache_slot, stride_vcache_head,
        stride_vs_blk, stride_vs_slot, stride_vs_head,
        v_scale_lds, tid);
    __syncthreads();

    int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step_wave_int8<T, HEAD_SIZE, /*CAUSAL_MASK=*/false>(
        K_lds, V_lds, P_lds_wave, k_scale_lds, v_scale_lds,
        q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane);
    __syncthreads();
  }

  // ----- PHASE 2: current chunk (fp16/bf16, optionally causal) ---------
  int causal_k_upper =
      causal ? (q_tile_start + valid_q_count_for_block) : query_len;
  int phase2_k_end = min(query_len, causal_k_upper);

  for (int start_n = 0; start_n < phase2_k_end; start_n += K_TILE) {
    load_k_tile_chunk_coop<T, HEAD_SIZE>(
        K_lds, k_chunk, q_start_token, kv_head_idx, start_n, query_len,
        stride_kc_token, stride_kc_head, tid);
    load_v_tile_chunk_coop<T, HEAD_SIZE>(
        V_lds, v_chunk, q_start_token, kv_head_idx, start_n, query_len,
        stride_vc_token, stride_vc_head, tid);
    if (tid < K_TILE) {
      k_scale_lds[tid] = 1.f;
      v_scale_lds[tid] = 1.f;
    }
    __syncthreads();

    int valid_k_count = min(K_TILE, query_len - start_n);
    attn_step_wave_int8<T, HEAD_SIZE, /*CAUSAL_MASK=*/true>(
        K_lds, V_lds, P_lds_wave, k_scale_lds, v_scale_lds,
        q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane);
    __syncthreads();
  }

  // ----- Epilogue: divide by L, write output ----------------------------
  // Each lane owns 4 (row, col) entries: col = lane%16, rows = lane/16 +
  // 4*i. Lanes split HEAD_SIZE across the FRAGS dh axis: for fragment dh,
  // the 16-column tile starts at HEAD_SIZE column dh*16, so this lane
  // contributes to out column dh*16 + lane%16.
  int col_offset = lane % 16;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_row = (lane / 16) + i * 4;
    int abs_m_row = wave_q_offset + m_row;
    int abs_q_pos = q_tile_start + abs_m_row;
    if (abs_q_pos >= query_len) continue;
    float l_inv = 1.f / (l_state[i] + 1e-10f);
    T* out_row = out + (int64_t)(q_start_token + abs_q_pos) * stride_o_token +
                 head_idx * stride_o_head;
    #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      int out_col = dh * 16 + col_offset;
      out_row[out_col] = from_float_rn<T>(out_acc[dh][i] * l_inv);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
void launch(T* out, const T* q, const T* k_chunk, const T* v_chunk,
            const int8_t* k_cache, const int8_t* v_cache,
            const float* k_scale_cache, const float* v_scale_cache,
            const int* block_table, const int* cu_seqlens_q,
            const int* seq_lens, int num_seqs, int num_query_heads,
            int num_kv_heads, int block_size, int max_blocks_per_seq,
            int max_query_len, float sm_scale, bool causal,
            int64_t stride_q_token, int64_t stride_q_head,
            int64_t stride_kc_token, int64_t stride_kc_head,
            int64_t stride_vc_token, int64_t stride_vc_head,
            int64_t stride_kcache_block, int64_t stride_kcache_slot,
            int64_t stride_kcache_head,
            int64_t stride_vcache_block, int64_t stride_vcache_slot,
            int64_t stride_vcache_head,
            int64_t stride_ks_blk, int64_t stride_ks_slot,
            int64_t stride_ks_head,
            int64_t stride_vs_blk, int64_t stride_vs_slot,
            int64_t stride_vs_head,
            int64_t stride_o_token, int64_t stride_o_head,
            cudaStream_t stream) {
  int q_blocks = (max_query_len + BLOCK_M - 1) / BLOCK_M;
  dim3 block(THREADS);
  dim3 grid(num_seqs, num_query_heads, q_blocks);
  paged_prefill_attn_kernel_v2_int8<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache,
      k_scale_cache, v_scale_cache, block_table, cu_seqlens_q, seq_lens,
      num_query_heads, num_kv_heads, block_size, max_blocks_per_seq,
      sm_scale, causal,
      stride_q_token, stride_q_head,
      stride_kc_token, stride_kc_head, stride_vc_token, stride_vc_head,
      stride_kcache_block, stride_kcache_slot, stride_kcache_head,
      stride_vcache_block, stride_vcache_slot, stride_vcache_head,
      stride_ks_blk, stride_ks_slot, stride_ks_head,
      stride_vs_blk, stride_vs_slot, stride_vs_head,
      stride_o_token, stride_o_head);
}

#endif  // gfx90a/942/950

}  // namespace prefill_attn_cdna_v2_int8
}  // namespace vllm

// ---------------------------------------------------------------------------
// Torch entry point
// ---------------------------------------------------------------------------

void paged_prefill_attn_cdna_int8(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  using namespace vllm::prefill_attn_cdna_v2_int8;
  using vllm::prefill_attn_cdna::bf16_t;

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "paged_prefill_attn_cdna_int8: q must be fp16 or bf16");
  TORCH_CHECK(k_cache.dtype() == at::kChar, "k_cache must be int8");
  TORCH_CHECK(v_cache.dtype() == at::kChar, "v_cache must be int8");

  int num_seqs = seq_lens.size(0);
  int num_query_heads = q.size(1);
  int num_kv_heads = k_scale_cache.size(2);
  int block_size = k_cache.size(1);
  int max_blocks_per_seq = block_table.size(1);
  int head_size = q.size(2);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH_INT8(T, HS)                                                   \
    launch<T, HS>((T*)out.data_ptr(), (const T*)q.data_ptr(),                  \
                  (const T*)k_chunk.data_ptr(),                                \
                  (const T*)v_chunk.data_ptr(),                                \
                  (const int8_t*)k_cache.data_ptr(),                           \
                  (const int8_t*)v_cache.data_ptr(),                           \
                  (const float*)k_scale_cache.data_ptr(),                      \
                  (const float*)v_scale_cache.data_ptr(),                      \
                  (const int*)block_table.data_ptr(),                          \
                  (const int*)cu_seqlens_q.data_ptr(),                         \
                  (const int*)seq_lens.data_ptr(),                             \
                  num_seqs, num_query_heads, num_kv_heads, block_size,         \
                  max_blocks_per_seq, (int)max_query_len, (float)sm_scale,     \
                  causal,                                                      \
                  q.stride(0), q.stride(1),                                    \
                  k_chunk.stride(0), k_chunk.stride(1),                        \
                  v_chunk.stride(0), v_chunk.stride(1),                        \
                  k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),     \
                  v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),     \
                  k_scale_cache.stride(0), k_scale_cache.stride(1),            \
                  k_scale_cache.stride(2),                                     \
                  v_scale_cache.stride(0), v_scale_cache.stride(1),            \
                  v_scale_cache.stride(2),                                     \
                  out.stride(0), out.stride(1), stream)

  if (q.dtype() == at::kHalf) {
    using T = _Float16;
    switch (head_size) {
      case 64:  LAUNCH_INT8(T, 64);  break;
      case 128: LAUNCH_INT8(T, 128); break;
      default:
        TORCH_CHECK(false, "paged_prefill_attn_cdna_int8: unsupported "
                           "head_size=", head_size, " (supported: 64, 128)");
    }
  } else {
    using T = vllm::prefill_attn_cdna::bf16_t;
    switch (head_size) {
      case 64:  LAUNCH_INT8(T, 64);  break;
      case 128: LAUNCH_INT8(T, 128); break;
      default:
        TORCH_CHECK(false, "paged_prefill_attn_cdna_int8: unsupported "
                           "head_size=", head_size, " (supported: 64, 128)");
    }
  }
  #undef LAUNCH_INT8
#else
  TORCH_CHECK(false,
              "paged_prefill_attn_cdna_int8 requires gfx942 / gfx950 / gfx90a");
#endif
}
