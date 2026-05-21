// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention with INT4 per-token-head KV cache on CDNA.
// Sibling of paged_prefill_attn_cdna_v2_int8.cu.
//
// KV layout (per-token-head, packed nibbles):
//   k_cache, v_cache : uint8 [num_blocks, block_size, num_kv_heads, head_size/2]
//     Two int4 values per byte. Low nibble = element 2i; high nibble = 2i+1.
//   k_scale_cache    : fp32  [num_blocks, block_size, num_kv_heads]
//     Bit-packed: low 4 bits = unsigned zero-point in [0..15]
//                  upper 28 bits = scale (fp32 with low 4 bits zeroed)
//
// Dequant at load: int4_value - zp → signed int in [-15, 15]. The signed
// value is then cast to fp16/bf16 and stored in LDS. The MFMA inner loop
// is identical to the INT8 path (no per-iteration zp correction).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_cdna.cuh"

namespace vllm {
namespace prefill_attn_cdna_v2_int4 {

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))

using vllm::prefill_attn_cdna::bf16_t;
using vllm::prefill_attn_cdna::cvt_T_from_int8;
using vllm::prefill_attn_cdna::floatx4;
using vllm::prefill_attn_cdna::from_float_rn;
using vllm::prefill_attn_cdna::mfma_16x16x16;
using vllm::prefill_attn_cdna::wave_group16_max;
using vllm::prefill_attn_cdna::wave_group16_sum;
using vllm::prefill_attn_cdna::WmmaNative;

constexpr int K_TILE     = 16;
constexpr int M_PER_WAVE = 16;
constexpr int LANES      = 64;
constexpr int WAVES      = 4;
constexpr int THREADS    = LANES * WAVES;
constexpr int BLOCK_M    = WAVES * M_PER_WAVE;

// Unpack steganographed scale word.
__device__ __forceinline__ void unpack_scale_zp(uint32_t packed, float& scale,
                                                int& zp) {
  zp = (int)(packed & 0xFu);
  uint32_t scale_bits = packed & 0xFFFFFFF0u;
  float s;
  __builtin_memcpy(&s, &scale_bits, 4);
  scale = s;
}

// Unpack 2 nibbles from a byte; subtract zp; cast to T. Returns 2 values.
template <typename T>
__device__ __forceinline__ void unpack_nibble_pair(uint8_t byte, int zp,
                                                   T& lo_out, T& hi_out) {
  int lo = (int)(byte & 0xFu) - zp;
  int hi = (int)((byte >> 4) & 0xFu) - zp;
  lo_out = cvt_T_from_int8<T>((int8_t)lo);
  hi_out = cvt_T_from_int8<T>((int8_t)hi);
}

// ---------------------------------------------------------------------------
// INT4 K-cache loader -> LDS (fp16/bf16), with zp subtraction at load time.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_paged_int4_coop(
    T* K_lds, const uint8_t* k_cache, const float* k_scale_cache,
    const int* block_table, int seq_idx, int kv_head_idx, int start_n,
    int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_slot, int64_t stride_kc_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    float* scale_lds,  // [K_TILE] dequantised fp32 scale (zp already applied)
    int tid) {
  constexpr int BYTES_PER_ROW = HEAD_SIZE / 2;
  constexpr int X_BYTES = 8;                              // 16 nibbles / vec
  constexpr int D_CHUNKS = BYTES_PER_ROW / X_BYTES;       // chunks per row
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;

  // Per-slot scale+zp.
  int my_zp = 0;
  float my_scale = 0.f;
  if (tid < K_TILE) {
    int abs_k = start_n + tid;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    uint32_t packed = 0;
    if (valid) {
      const float* p = k_scale_cache + p_blk * stride_ks_blk +
                       slot * stride_ks_slot + kv_head_idx * stride_ks_head;
      __builtin_memcpy(&packed, p, 4);
    }
    unpack_scale_zp(packed, my_scale, my_zp);
    scale_lds[tid] = my_scale;
  }
  __syncthreads();

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int byte_base = d_chunk * X_BYTES;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;

    // Re-fetch the per-slot zp (cheap; could also stage to LDS).
    int zp_for_this_k = 0;
    if (valid) {
      uint32_t packed = 0;
      const float* sp = k_scale_cache + p_blk * stride_ks_blk +
                        slot * stride_ks_slot + kv_head_idx * stride_ks_head;
      __builtin_memcpy(&packed, sp, 4);
      zp_for_this_k = (int)(packed & 0xFu);
    }

    uint8_t bytes[X_BYTES];
    if (valid) {
      const uint8_t* src = k_cache + p_blk * stride_kc_block +
                           slot * stride_kc_slot +
                           kv_head_idx * stride_kc_head + byte_base;
      *(int2*)bytes = *(const int2*)src;  // 8 bytes = int2
    } else {
      #pragma unroll
      for (int i = 0; i < X_BYTES; ++i) bytes[i] = 0;
    }

    T out[X_BYTES * 2];
    #pragma unroll
    for (int i = 0; i < X_BYTES; ++i) {
      unpack_nibble_pair<T>(bytes[i], zp_for_this_k, out[2 * i],
                            out[2 * i + 1]);
    }
    // Write 16 elements of T (= 32B for fp16/bf16) as 2 int4 vec stores.
    T* dst = &K_lds[k_idx * HEAD_SIZE + byte_base * 2];
    *((int4*)dst) = *(const int4*)&out[0];
    *((int4*)(dst + 8)) = *(const int4*)&out[8];
  }
}

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_paged_int4_coop(
    T* V_lds, const uint8_t* v_cache, const float* v_scale_cache,
    const int* block_table, int seq_idx, int kv_head_idx, int start_n,
    int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_vc_block, int64_t stride_vc_slot, int64_t stride_vc_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    float* v_scale_lds, int tid) {
  constexpr int BYTES_PER_ROW = HEAD_SIZE / 2;
  constexpr int X_BYTES = 8;
  constexpr int D_CHUNKS = BYTES_PER_ROW / X_BYTES;
  constexpr int VECS_PER_TILE = K_TILE * D_CHUNKS;
  constexpr int VECS_PER_THREAD = (VECS_PER_TILE + THREADS - 1) / THREADS;

  if (tid < K_TILE) {
    int abs_k = start_n + tid;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;
    uint32_t packed = 0;
    if (valid) {
      const float* p = v_scale_cache + p_blk * stride_vs_blk +
                       slot * stride_vs_slot + kv_head_idx * stride_vs_head;
      __builtin_memcpy(&packed, p, 4);
    }
    float s;
    int zp;
    unpack_scale_zp(packed, s, zp);
    v_scale_lds[tid] = s;
  }
  __syncthreads();

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    int idx = v * THREADS + tid;
    if (idx >= VECS_PER_TILE) break;
    int k_idx = idx / D_CHUNKS;
    int d_chunk = idx % D_CHUNKS;
    int byte_base = d_chunk * X_BYTES;
    int abs_k = start_n + k_idx;
    bool valid = abs_k < seq_ctx_len;
    int log_blk = abs_k / block_size;
    int slot = abs_k - log_blk * block_size;
    int p_blk = valid ? block_table[seq_idx * max_blocks_per_seq + log_blk] : 0;

    int zp_for_this_k = 0;
    if (valid) {
      uint32_t packed = 0;
      const float* sp = v_scale_cache + p_blk * stride_vs_blk +
                        slot * stride_vs_slot + kv_head_idx * stride_vs_head;
      __builtin_memcpy(&packed, sp, 4);
      zp_for_this_k = (int)(packed & 0xFu);
    }

    uint8_t bytes[X_BYTES];
    if (valid) {
      const uint8_t* src = v_cache + p_blk * stride_vc_block +
                           slot * stride_vc_slot +
                           kv_head_idx * stride_vc_head + byte_base;
      *(int2*)bytes = *(const int2*)src;
    } else {
      #pragma unroll
      for (int i = 0; i < X_BYTES; ++i) bytes[i] = 0;
    }

    // Transpose-store into V_lds[d][k_idx].
    #pragma unroll
    for (int i = 0; i < X_BYTES; ++i) {
      T lo, hi;
      unpack_nibble_pair<T>(bytes[i], zp_for_this_k, lo, hi);
      V_lds[(byte_base * 2 + 2 * i + 0) * K_TILE + k_idx] = lo;
      V_lds[(byte_base * 2 + 2 * i + 1) * K_TILE + k_idx] = hi;
    }
  }
}

// ---------------------------------------------------------------------------
// fp16/bf16 chunk loaders (identical to INT8 kernel — duplicated here to
// keep this TU self-contained).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_chunk_coop(
    T* K_lds, const T* k_chunk, int q_start_token, int kv_head_idx,
    int start_n, int chunk_len, int64_t stride_kc_token,
    int64_t stride_kc_head, int tid) {
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
    int start_n, int chunk_len, int64_t stride_vc_token,
    int64_t stride_vc_head, int tid) {
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
// attn_step — identical to INT8 (after centered-int8 load, no kernel-side
// nibble logic remains).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step_wave_int4(
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
  floatx4 s_acc = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    int n  = lane / 4;
    int k0 = (lane % 4) * 4;
    V4 b_frag;
    T* bdst = (T*)&b_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      bdst[i] = K_lds[n * HEAD_SIZE + dh * 16 + k0 + i];
    s_acc = mfma_16x16x16<T>(q_frags[dh], b_frag, s_acc);
  }
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
  float m_ij[4], m_new[4], alpha[4], p_ij[4], l_ij[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    m_ij[i] = wave_group16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.f
                                         : __expf(m_state[i] - m_new[i]);
    p_ij[i] = (m_new[i] == -INFINITY) ? 0.f
                                      : __expf(s_acc[i] - m_new[i]);
    l_ij[i] = wave_group16_sum(p_ij[i]);
    l_state[i] = l_state[i] * alpha[i] + l_ij[i];
    m_state[i] = m_new[i];
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh)
    #pragma unroll
    for (int i = 0; i < 4; ++i) out_acc[dh][i] *= alpha[i];

  float v_sc = v_scale_lds[n_col];
  #pragma unroll
  for (int i = 0; i < 4; ++i) p_ij[i] *= v_sc;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_row = (lane / 16) + i * 4;
    P_lds_wave[m_row * 16 + n_col] = from_float_rn<T>(p_ij[i]);
  }
  __syncthreads();

  V4 p_frag;
  {
    int m = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&p_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = P_lds_wave[m * 16 + k0 + i];
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V4 v_frag;
    int n  = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&v_frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = V_lds[(dh * 16 + n) * K_TILE + k0 + i];
    out_acc[dh] = mfma_16x16x16<T>(p_frag, v_frag, out_acc[dh]);
  }
}

// ---------------------------------------------------------------------------
// Main INT4 kernel
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__global__ __launch_bounds__(THREADS, 2)
void paged_prefill_attn_kernel_v2_int4(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_chunk,
    const T* __restrict__ v_chunk,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
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
    int m = lane / 4;
    int k0 = (lane % 4) * 4;
    T* dst = (T*)&q_frags[dh];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = Q_lds[m * HEAD_SIZE + dh * 16 + k0 + i];
  }

  __shared__ T  K_lds[K_TILE * HEAD_SIZE];
  __shared__ T  V_lds[HEAD_SIZE * K_TILE];
  __shared__ T  P_lds[WAVES][M_PER_WAVE * K_TILE];
  __shared__ float k_scale_lds[K_TILE];
  __shared__ float v_scale_lds[K_TILE];
  T* P_lds_wave = &P_lds[wave_id][0];

  float m_state[4], l_state[4];
  floatx4 out_acc[FRAGS];
  #pragma unroll
  for (int i = 0; i < 4; ++i) { m_state[i] = -INFINITY; l_state[i] = 0.f; }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh)
    out_acc[dh] = (floatx4){0.f, 0.f, 0.f, 0.f};

  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_paged_int4_coop<T, HEAD_SIZE>(
        K_lds, k_cache, k_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_kcache_block, stride_kcache_slot, stride_kcache_head,
        stride_ks_blk, stride_ks_slot, stride_ks_head, k_scale_lds, tid);
    load_v_tile_paged_int4_coop<T, HEAD_SIZE>(
        V_lds, v_cache, v_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_vcache_block, stride_vcache_slot, stride_vcache_head,
        stride_vs_blk, stride_vs_slot, stride_vs_head, v_scale_lds, tid);
    __syncthreads();
    int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step_wave_int4<T, HEAD_SIZE, /*CAUSAL_MASK=*/false>(
        K_lds, V_lds, P_lds_wave, k_scale_lds, v_scale_lds,
        q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane);
    __syncthreads();
  }

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
    attn_step_wave_int4<T, HEAD_SIZE, /*CAUSAL_MASK=*/true>(
        K_lds, V_lds, P_lds_wave, k_scale_lds, v_scale_lds,
        q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane);
    __syncthreads();
  }

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

template <typename T, int HEAD_SIZE>
void launch(T* out, const T* q, const T* k_chunk, const T* v_chunk,
            const uint8_t* k_cache, const uint8_t* v_cache,
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
  paged_prefill_attn_kernel_v2_int4<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache, k_scale_cache, v_scale_cache,
      block_table, cu_seqlens_q, seq_lens, num_query_heads, num_kv_heads,
      block_size, max_blocks_per_seq, sm_scale, causal,
      stride_q_token, stride_q_head,
      stride_kc_token, stride_kc_head, stride_vc_token, stride_vc_head,
      stride_kcache_block, stride_kcache_slot, stride_kcache_head,
      stride_vcache_block, stride_vcache_slot, stride_vcache_head,
      stride_ks_blk, stride_ks_slot, stride_ks_head,
      stride_vs_blk, stride_vs_slot, stride_vs_head,
      stride_o_token, stride_o_head);
}

#endif  // gfx90a/942/950

}  // namespace prefill_attn_cdna_v2_int4
}  // namespace vllm

// ---------------------------------------------------------------------------
// Torch entry point
// ---------------------------------------------------------------------------

void paged_prefill_attn_cdna_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  using namespace vllm::prefill_attn_cdna_v2_int4;
  using vllm::prefill_attn_cdna::bf16_t;

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "paged_prefill_attn_cdna_int4: q must be fp16 or bf16");
  TORCH_CHECK(k_cache.dtype() == at::kByte, "k_cache must be uint8 (packed)");
  TORCH_CHECK(v_cache.dtype() == at::kByte, "v_cache must be uint8 (packed)");

  int num_seqs = seq_lens.size(0);
  int num_query_heads = q.size(1);
  int num_kv_heads = k_scale_cache.size(2);
  int block_size = k_cache.size(1);
  int max_blocks_per_seq = block_table.size(1);
  int head_size = q.size(2);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH_INT4(T, HS)                                                   \
    launch<T, HS>((T*)out.data_ptr(), (const T*)q.data_ptr(),                  \
                  (const T*)k_chunk.data_ptr(),                                \
                  (const T*)v_chunk.data_ptr(),                                \
                  (const uint8_t*)k_cache.data_ptr(),                          \
                  (const uint8_t*)v_cache.data_ptr(),                          \
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
      case 64:  LAUNCH_INT4(T, 64);  break;
      case 128: LAUNCH_INT4(T, 128); break;
      default:
        TORCH_CHECK(false, "paged_prefill_attn_cdna_int4: unsupported "
                           "head_size=", head_size, " (supported: 64, 128)");
    }
  } else {
    using T = vllm::prefill_attn_cdna::bf16_t;
    switch (head_size) {
      case 64:  LAUNCH_INT4(T, 64);  break;
      case 128: LAUNCH_INT4(T, 128); break;
      default:
        TORCH_CHECK(false, "paged_prefill_attn_cdna_int4: unsupported "
                           "head_size=", head_size, " (supported: 64, 128)");
    }
  }
  #undef LAUNCH_INT4
#else
  TORCH_CHECK(false,
              "paged_prefill_attn_cdna_int4 requires gfx942 / gfx950 / gfx90a");
#endif
}
