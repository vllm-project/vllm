// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention v2 kernel for AMD RDNA3 (gfx1100). 4 waves per
// block, BLOCK_M = 64 (each wave owns 16 query rows). K/V tiles in LDS
// are SHARED across the 4 waves — one cooperative load per K-iter feeds
// all 64 query rows. Cuts global K/V reads by 4× vs v1's
// 1-wave-per-block design, which is the dominant cost in chunk-heavy
// prefill (e.g., the initial big prompt).
//
// v1 (paged_prefill_attn_rdna3.cu): 1 wave/block, BLOCK_M=16. Wins for
//   short qlen, loses for long qlen due to repeated K reload.
// v2 (this file): 4 waves/block, BLOCK_M=64. Wins for long qlen.
//
// The shared op `paged_prefill_attn_rdna3` (in v1.cu) dispatches between
// v1 and v2 based on max_query_len at the C++ level. v1 launcher used
// when max_query_len < 64 (where v2's wider tile would waste 3 of 4
// waves). Single Python entry, branch-free under torch.compile.
//
// Same constraints as v1: fp16 / bf16, head_size == 128, no FP8 KV /
// alibi / SW / sinks / FP8 output / softcap, causal, paged K cache
// 5-D layout, paged V cache 4-D layout, block_size % 16 == 0.
//
// Lesson 4 (W4A16 WMMA): keep this in its OWN translation unit. Hipcc
// scopes some optimizer decisions (register pressure / SGPR heuristics)
// at the TU level — putting v1 and v2 in the same file in the W4A16
// kernels miscompiled the M=1 path even when the WMMA template was
// never instantiated.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_rdna3.cuh"

namespace vllm {
namespace prefill_attn_rdna3_v2 {

#if defined(USE_ROCM)

using vllm::prefill_attn_rdna3::bf16_t;
using vllm::prefill_attn_rdna3::WmmaNative;
using vllm::prefill_attn_rdna3::wmma_mma;
using vllm::prefill_attn_rdna3::bitcast_elem;
using vllm::prefill_attn_rdna3::to_T;
using vllm::prefill_attn_rdna3::v16fp16;
using vllm::prefill_attn_rdna3::v16bf16;
using vllm::prefill_attn_rdna3::v8fp32;

constexpr int K_TILE = 16;
constexpr int NUM_WAVES = 4;
constexpr int M_PER_WAVE = 16;
constexpr int BLOCK_M_V2 = NUM_WAVES * M_PER_WAVE;  // = 64
constexpr int THREADS = NUM_WAVES * 32;             // = 128

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
// Cooperative K cache load. 128 threads = 4 waves cooperate to populate
// K_lds_raw [d_high (HEAD_SIZE/X)][k (K_TILE)][x (X)] = 16 × 16 × 8 fp16
// = 4 KB. Each thread does 2 vec_16B loads (vs 8 in the 1-wave path).
//
// Distribution: 256 vec loads / 128 threads = 2 per thread.
// thread t handles k_idx = t / 16 (8 k_idxs covered by 16 threads each via
// d_high distribution... hmm let me redo).
//
// Cleaner: thread t handles ONE (k_idx, d_high) pair per vec load × 2 vec
// loads. 128 threads × 2 = 256 = 16 k_idx × 16 d_high.
// Map: thread t = (k_idx_lane, d_high_pair). 16 k_idxs × 8 d_high_pairs
// = 128 thread slots. Each thread does 2 d_high vec loads (the pair).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void load_k_tile_paged_coop(
    T* __restrict__ K_lds_raw,
    const T* __restrict__ k_cache,
    const int* __restrict__ block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_head,
    int64_t stride_kc_dhi, int64_t stride_kc_slot,
    int tid) {
  // 128 threads × 2 vec loads each = 256 work items = K_TILE × (HEAD_SIZE/X).
  // Map: thread t = (k_idx, d_high_base). 16 k_idxs × 8 d_high pairs = 128.
  const int my_k_idx = tid >> 3;       // tid / 8 → 0..15
  const int my_dh_base = (tid & 7) * 2; // (tid % 8) * 2 → 0,2,4,...,14

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < seq_ctx_len;
  const int log_block = abs_k / block_size;
  const int slot = abs_k - log_block * block_size;
  const int p_block =
      valid_k ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;

#pragma unroll
  for (int dh = 0; dh < 2; ++dh) {
    const int d_high = my_dh_base + dh;
    const T* src = k_cache + (int64_t)p_block * stride_kc_block +
                   (int64_t)kv_head_idx * stride_kc_head +
                   (int64_t)d_high * stride_kc_dhi +
                   (int64_t)slot * stride_kc_slot;
    int4 vec;
    if (valid_k) {
      vec = *(const int4*)src;
    } else {
      vec.x = vec.y = vec.z = vec.w = 0;
    }
    *(int4*)&K_lds_raw[d_high * (K_TILE * X) + my_k_idx * X] = vec;
  }
}

// Cooperative K chunk load (current-chunk K from linear K). 128 threads
// = 4 waves cooperate. Same pattern as v2 paged loader.
template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void load_k_tile_chunk_coop(
    T* __restrict__ K_lds_raw,
    const T* __restrict__ k_chunk,
    int chunk_token_base, int kv_head_idx, int start_n, int chunk_len,
    int64_t stride_kc_token, int64_t stride_kc_head,
    int tid) {
  const int my_k_idx = tid >> 3;
  const int my_dh_base = (tid & 7) * 2;

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < chunk_len;
  const T* row =
      valid_k ? (k_chunk + (int64_t)(chunk_token_base + abs_k) * stride_kc_token +
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

// Cooperative V cache load. V_lds [HEAD_SIZE][K_TILE] = 128 × 16 = 2048 fp16
// = 4 KB. 256 vec loads / 128 threads = 2 per thread.
// Map: thread t handles (d, k_chunk) where d = t / 2, k_chunk = t & 1
// (low half k=0..7 or high half k=8..15). Per thread: 1 vec load + 1 vec
// write. With HEAD_SIZE = 128, that's 64 d values × 2 k-chunks = 128 = THREADS.
// HEAD_SIZE / 2 d values per thread doesn't work cleanly... let me redo.
//
// Total vec loads needed: HEAD_SIZE × 2 (low + high K_TILE half) = 256.
// 128 threads × 2 vec loads each = 256. Map: thread t = (d_idx, kc).
// We use t / 2 = d_idx (covers d 0..63 across 128 threads if we wrap)...
// Actually: 256 / 128 = 2 vec per thread. Simplest: thread t handles d = t,
// loading both k chunks (2 vec loads per thread). HEAD_SIZE = 128 = THREADS,
// so this maps perfectly.
template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_paged_coop(
    T* __restrict__ V_lds,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_vc_block, int64_t stride_vc_head,
    int64_t stride_vc_d, int64_t stride_vc_slot,
    int tid) {
  static_assert(HEAD_SIZE == THREADS,
                "v2 V load assumes HEAD_SIZE == THREADS (128)");

  const int log_block = start_n / block_size;
  const int slot_base = start_n - log_block * block_size;
  const int p_block = block_table[seq_idx * max_blocks_per_seq + log_block];
  const int valid_k_count = max(0, min(K_TILE, seq_ctx_len - start_n));

  const int d = tid;  // tid 0..127 == d 0..127
  const T* src = v_cache + (int64_t)p_block * stride_vc_block +
                 (int64_t)kv_head_idx * stride_vc_head +
                 (int64_t)d * stride_vc_d +
                 (int64_t)slot_base * stride_vc_slot;
  int4 vec_lo, vec_hi;
  if (valid_k_count >= K_TILE) {
    vec_lo = *(const int4*)src;
    vec_hi = *(const int4*)(src + 8);
  } else {
    T tmp[K_TILE];
#pragma unroll
    for (int k = 0; k < K_TILE; ++k) {
      tmp[k] = (k < valid_k_count) ? src[k] : (T)0;
    }
    __builtin_memcpy(&vec_lo, tmp, 16);
    __builtin_memcpy(&vec_hi, tmp + 8, 16);
  }
  *(int4*)&V_lds[d * K_TILE + 0] = vec_lo;
  *(int4*)&V_lds[d * K_TILE + 8] = vec_hi;
}

// V chunk: V_chunk[token][kv_head][d]. For each (k_idx, d) we read
// V_chunk[chunk_token_base + start_n + k_idx][kv_head][d]. Different
// k_idxs are different tokens (different rows in V_chunk).
//
// Per-thread: handles one (k_idx, d_chunk_of_8) pair = 1 vec load.
// Total work: K_TILE × (HEAD_SIZE / 8) = 16 × 16 = 256. 128 threads × 2 vecs.
template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_chunk_coop(
    T* __restrict__ V_lds,
    const T* __restrict__ v_chunk,
    int chunk_token_base, int kv_head_idx, int start_n, int chunk_len,
    int64_t stride_vc_token, int64_t stride_vc_head,
    int tid) {
  // Map: thread t = (k_idx, d_chunk). 16 k_idxs × 16 d_chunks = 256. Each
  // thread does 2 such pairs.
  // tid 0..127 → first work item (k_idx_a, d_chunk_a)
  //              second work item (k_idx_b, d_chunk_b)
  // Simple: thread t handles k_idx = t / 8, d_chunk = (t & 7) for first
  // and d_chunk = (t & 7) + 8 for second.
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int my_k = tid >> 3;
    const int my_dc = (tid & 7) + p * 8;  // d chunk in 0..15
    const int d_base = my_dc * 8;          // 8 fp16 per chunk

    const int abs_k = start_n + my_k;
    const bool valid = abs_k < chunk_len;
    int4 vec;
    if (valid) {
      const T* src = v_chunk +
                     (int64_t)(chunk_token_base + abs_k) * stride_vc_token +
                     (int64_t)kv_head_idx * stride_vc_head +
                     (int64_t)d_base;
      vec = *(const int4*)src;
    } else {
      vec.x = vec.y = vec.z = vec.w = 0;
    }
    // Write 8 fp16 to V_lds[d_base..d_base+8][my_k] — scattered along d.
    T tmp[8];
    __builtin_memcpy(tmp, &vec, 16);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      V_lds[(d_base + e) * K_TILE + my_k] = tmp[e];
    }
  }
}

// ---------------------------------------------------------------------------
// Per-wave attn_step. Same as v1's attn_step but operates on this wave's
// 16 query rows. K_lds_raw / V_lds are SHARED across all 4 waves; P_lds
// is wave-local (each wave gets its own [16][K_TILE] slice = 512 B per
// wave = 2 KB total).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step_wave(
    const T* __restrict__ K_lds_raw,
    const T* __restrict__ V_lds,
    T* __restrict__ P_lds_wave,                       // [BLOCK_M_PER_WAVE = 16][K_TILE]
    typename WmmaNative<T>::v16 (&q_frags)[HEAD_SIZE / 16],
    v8fp32 (&out_acc)[HEAD_SIZE / 16],
    float (&m_state)[8],
    float (&l_state)[8],
    int wave_q_tile_start, int start_n, int valid_q_count, int valid_k_count,
    float sm_scale, int lane, int lane_lo, int lane_hi) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int FRAGS = HEAD_SIZE / 16;

  // ---- Q @ K (8 WMMAs into s_acc) ----
  v8fp32 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 b_frag;
    int4 lo = *(const int4*)&K_lds_raw[(dh * 2 + 0) * (K_TILE * X) +
                                       lane_lo * X];
    int4 hi = *(const int4*)&K_lds_raw[(dh * 2 + 1) * (K_TILE * X) +
                                       lane_lo * X];
    __builtin_memcpy(&b_frag, &lo, 16);
    __builtin_memcpy(((char*)&b_frag) + 16, &hi, 16);
    s_acc = wmma_mma(q_frags[dh], b_frag, s_acc);
  }

  // ---- Scale + mask ----
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
    s_acc[i] = keep ? (s_acc[i] * sm_scale) : -INFINITY;
  }

  // ---- Online softmax (parallel across 8 rows, ILP-friendly) ----
  float m_ij[8], m_new[8], alpha[8], p_ij[8], l_ij[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_ij[i] = wave16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.0f
                                          : __expf(m_state[i] - m_new[i]);
    p_ij[i] =
        (m_new[i] == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new[i]);
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

  // ---- Transpose P → p_frag via WAVE-LOCAL P_lds ----
  // Single wave context; cross-lane writes to P_lds_wave land before
  // dependent reads (compiler inserts s_waitcnt lgkmcnt(0)).
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

  // ---- P @ V (8 WMMAs accumulating into out_acc) ----
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
// Main v2 kernel. 4 waves per block, each owns 16 query rows.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__global__ void paged_prefill_attn_kernel_v2(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_chunk,
    const T* __restrict__ v_chunk,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    const int num_query_heads, const int num_kv_heads,
    const int block_size, const int max_blocks_per_seq, const int x,
    const float sm_scale, const bool causal,
    const int64_t stride_q_token, const int64_t stride_q_head,
    const int64_t stride_kc_token, const int64_t stride_kc_head,
    const int64_t stride_vc_token, const int64_t stride_vc_head,
    const int64_t stride_kcache_block, const int64_t stride_kcache_head,
    const int64_t stride_kcache_dhi, const int64_t stride_kcache_slot,
    const int64_t stride_vcache_block, const int64_t stride_vcache_head,
    const int64_t stride_vcache_d, const int64_t stride_vcache_slot,
    const int64_t stride_o_token, const int64_t stride_o_head) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int FRAGS = HEAD_SIZE / 16;
  constexpr int X = 16 / sizeof(T);

  const int seq_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int tid = threadIdx.x;       // 0..127
  const int wave_id = tid >> 5;       // 0..3
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  // Per-seq metadata.
  const int q_start_token = cu_seqlens_q[seq_idx];
  const int q_end_token = cu_seqlens_q[seq_idx + 1];
  const int query_len = q_end_token - q_start_token;
  const int seq_len = seq_lens[seq_idx];
  const int ctx_len = seq_len - query_len;

  const int q_tile_start = q_tile_idx * BLOCK_M_V2;
  if (q_tile_start >= query_len) return;

  // This wave's row range within the BLOCK_M_V2-tile.
  const int wave_q_offset = wave_id * M_PER_WAVE;
  const int wave_q_tile_start = q_tile_start + wave_q_offset;
  const int my_m_row = lane_lo;  // == row within wave's 16-row band
  const int my_q_pos = wave_q_tile_start + my_m_row;
  const bool valid_q = my_q_pos < query_len;
  const int valid_q_count_for_wave =
      max(0, min(M_PER_WAVE, query_len - wave_q_tile_start));

  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  // ---- Load Q (per-wave) ----
  V16 q_frags[FRAGS];
  if (valid_q) {
    const T* q_row = q + (int64_t)(q_start_token + my_q_pos) * stride_q_token +
                     (int64_t)head_idx * stride_q_head;
#pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      __builtin_memcpy(&q_frags[dh], q_row + dh * 16, sizeof(V16));
    }
  } else {
#pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
#pragma unroll
      for (int k = 0; k < 16; ++k) q_frags[dh][k] = (E)0;
    }
  }

  // Per-wave online-softmax state.
  float m_state[8];
  float l_state[8];
  v8fp32 out_acc[FRAGS];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_state[i] = -INFINITY;
    l_state[i] = 0.0f;
  }
#pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    out_acc[dh] = (v8fp32){0, 0, 0, 0, 0, 0, 0, 0};
  }

  // ---- Shared LDS workspaces ----
  // K and V tiles SHARED across the 4 waves (cooperative load).
  __shared__ T K_lds_raw[FRAGS * 2 * K_TILE * X];      // 4 KB
  __shared__ T V_lds[HEAD_SIZE * K_TILE];              // 4 KB
  // P_lds is wave-local (4 × 512 B = 2 KB).
  __shared__ T P_lds[NUM_WAVES][M_PER_WAVE * K_TILE];
  T* P_lds_wave = &P_lds[wave_id][0];

  // ---- PHASE 1: Cached prefix (no causal) ----
  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_paged_coop<T, HEAD_SIZE, X>(
        K_lds_raw, k_cache, block_table, seq_idx, kv_head_idx, start_n,
        ctx_len, block_size, max_blocks_per_seq, stride_kcache_block,
        stride_kcache_head, stride_kcache_dhi, stride_kcache_slot, tid);
    load_v_tile_paged_coop<T, HEAD_SIZE>(
        V_lds, v_cache, block_table, seq_idx, kv_head_idx, start_n, ctx_len,
        block_size, max_blocks_per_seq, stride_vcache_block, stride_vcache_head,
        stride_vcache_d, stride_vcache_slot, tid);
    __syncthreads();  // K/V loads visible to all waves before WMMA

    const int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step_wave<T, HEAD_SIZE, X, /*CAUSAL_MASK=*/false>(
        K_lds_raw, V_lds, P_lds_wave, q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane, lane_lo, lane_hi);
    __syncthreads();  // ensure all waves done with K/V before next load
  }

  // ---- PHASE 2: Current chunk (causal) ----
  // Causal upper bound for K direction: any K beyond the wave's last
  // valid query position contributes nothing. Different waves in the
  // same block may reach different K bounds; we take the MAX so all
  // waves stay synchronized at __syncthreads.
  // valid_q_count_for_wave depends on wave; max query position covered
  // by ANY wave in the block: q_tile_start + valid_q_count_for_block.
  const int valid_q_count_for_block =
      max(0, min(BLOCK_M_V2, query_len - q_tile_start));
  const int causal_k_upper =
      causal ? (q_tile_start + valid_q_count_for_block) : query_len;
  const int phase2_k_end = min(query_len, causal_k_upper);
  for (int start_n = 0; start_n < phase2_k_end; start_n += K_TILE) {
    load_k_tile_chunk_coop<T, HEAD_SIZE, X>(
        K_lds_raw, k_chunk, q_start_token, kv_head_idx, start_n, query_len,
        stride_kc_token, stride_kc_head, tid);
    load_v_tile_chunk_coop<T, HEAD_SIZE>(
        V_lds, v_chunk, q_start_token, kv_head_idx, start_n, query_len,
        stride_vc_token, stride_vc_head, tid);
    __syncthreads();

    const int valid_k_count = min(K_TILE, query_len - start_n);
    attn_step_wave<T, HEAD_SIZE, X, /*CAUSAL_MASK=*/true>(
        K_lds_raw, V_lds, P_lds_wave, q_frags, out_acc, m_state, l_state,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane, lane_lo, lane_hi);
    __syncthreads();
  }

  // ---- Epilogue: divide by L, write output (per wave) ----
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;       // within wave (0..15)
    const int abs_m_row = wave_q_offset + m_row;  // within block (0..63)
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

template <typename T, int HEAD_SIZE>
void launch_paged_prefill_attn_v2(
    T* out, const T* q, const T* k_chunk, const T* v_chunk, const T* k_cache,
    const T* v_cache, const int* block_table, const int* cu_seqlens_q,
    const int* seq_lens, int num_seqs, int num_query_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq, int max_query_len, float sm_scale,
    bool causal, int64_t stride_q_token, int64_t stride_q_head,
    int64_t stride_kc_token, int64_t stride_kc_head, int64_t stride_vc_token,
    int64_t stride_vc_head, int64_t stride_kcache_block,
    int64_t stride_kcache_head, int64_t stride_kcache_dhi,
    int64_t stride_kcache_slot, int64_t stride_vcache_block,
    int64_t stride_vcache_head, int64_t stride_vcache_d,
    int64_t stride_vcache_slot, int64_t stride_o_token, int64_t stride_o_head,
    cudaStream_t stream) {
  constexpr int X = 16 / sizeof(T);
  const int q_blocks = (max_query_len + BLOCK_M_V2 - 1) / BLOCK_M_V2;
  dim3 block(THREADS);  // 128 = 4 waves
  dim3 grid(num_seqs, num_query_heads, q_blocks);
  paged_prefill_attn_kernel_v2<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache, block_table, cu_seqlens_q,
      seq_lens, num_query_heads, num_kv_heads, block_size, max_blocks_per_seq,
      X, sm_scale, causal, stride_q_token, stride_q_head, stride_kc_token,
      stride_kc_head, stride_vc_token, stride_vc_head, stride_kcache_block,
      stride_kcache_head, stride_kcache_dhi, stride_kcache_slot,
      stride_vcache_block, stride_vcache_head, stride_vcache_d,
      stride_vcache_slot, stride_o_token, stride_o_head);
}

// Explicit instantiations exposed to the v1 TU's dispatcher.
template void launch_paged_prefill_attn_v2<half, 128>(
    half*, const half*, const half*, const half*, const half*, const half*,
    const int*, const int*, const int*, int, int, int, int, int, int, float,
    bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, cudaStream_t);

template void launch_paged_prefill_attn_v2<bf16_t, 128>(
    bf16_t*, const bf16_t*, const bf16_t*, const bf16_t*, const bf16_t*,
    const bf16_t*, const int*, const int*, const int*, int, int, int, int, int,
    int, float, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, cudaStream_t);

#endif  // USE_ROCM

}  // namespace prefill_attn_rdna3_v2
}  // namespace vllm
