// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention kernel for AMD RDNA3 (gfx1100 / RX 7900 XTX).
// Replaces the Triton ``context_attention_fwd`` (in
// vllm/v1/attention/ops/prefix_prefill.py) for ROCM_ATTN's
// ``max_query_len > 1`` path: 16M × 16K WMMA tiles with 1 wave per
// block, paged K/V via ``block_table``, two-phase iteration (cached
// prefix without causal mask, then current chunk with causal mask),
// and online-softmax merge.
//
// v1 (this file): 1 wave per block, BLOCK_M = 16. Mirrors the W4A16
// WMMA v1 kernel structure (csrc/quantization/gptq/q_gemm_rdna3_wmma.cu).
// v2 (separate TU per Lesson 4): 2 waves per block + double-buffered
// LDS, target M >= 32 — added later.
//
// Hardware notes:
//   * v_wmma_f32_16x16x16_{f16,bf16}_w32 — 16-cycle nominal throughput.
//   * Wave32 doubled-frag layout: lanes 0..15 and 16..31 hold IDENTICAL
//     A/B fragments. Output C uses lane_hi to interleave M rows
//     (lanes 0..15 → even rows, lanes 16..31 → odd rows).
//   * No __syncthreads() needed in single-wave kernels — compiler
//     emits s_waitcnt lgkmcnt(0) for dependent ds ops within the wave.
//
// Layout assumptions (mirroring the existing ROCM_ATTN paths):
//   * K cache: (num_blocks, num_kv_heads, HEAD_SIZE/x, block_size, x)
//     with x = 16 / sizeof(T). Innermost ``x`` is contiguous = 16 B
//     vec load.
//   * V cache: (num_blocks, num_kv_heads, HEAD_SIZE, block_size).
//     Slot is innermost = ``stride_v_cache_slot = 1`` so K_TILE
//     consecutive slots within one block are 16 fp16 = 32 B.
//   * Q / current-chunk K, V: (num_actual_tokens, num_*_heads, HEAD_SIZE)
//     contiguous along HEAD_SIZE.
//   * Output: same layout as Q.
//
// Constraint: ``block_size % K_TILE == 0`` (always true for K_TILE=16
// and the standard block sizes in production: 16, 32, 64, 544, 784,
// 1056). Enforced at the launcher; if violated, caller must fall back
// to the Triton path.
//
// FP8 KV cache, alibi, sliding window, sinks, FP8 output, and
// non-causal attention are NOT supported in v1 — caller falls through
// to ``context_attention_fwd`` for any of those.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_rdna3.cuh"

namespace vllm {
namespace prefill_attn_rdna3 {

#if defined(USE_ROCM)

constexpr int BLOCK_M = 16;
constexpr int K_TILE = 16;

// ---------------------------------------------------------------------------
// Wave-level reduction helpers (lane-half scoped: 0..15 vs 16..31).
// The c_acc layout means row m is held by lanes (lane_hi == m & 1) at
// slot i = m / 2. Reductions across the 16 columns of S are therefore
// 4-step shfl_xor reductions WITHIN a 16-lane half.
// ---------------------------------------------------------------------------

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
// K cache load: 5D (num_blocks, num_kv_heads, HEAD_SIZE/x, block_size, x).
// Loads one K_TILE = 16 keys for one (kv_head, q_block_start) into
// ``K_lds_raw`` with layout [d_high (HEAD_SIZE/x)][k (K_TILE)][x] —
// directly mirroring K_cache so each WMMA b_frag load is a single
// 16 B ds_read.
//
// Distribution across the 32-lane wave: one lane handles k_idx = lane / 2
// for d_high range [(lane & 1) * (D_HIGH / 2), (lane & 1) * (D_HIGH / 2) +
// D_HIGH / 2). With HEAD_SIZE = 128 and x = 8, D_HIGH = 16 so each lane
// covers 8 d_highs × 1 k_idx = 8 vec loads.
//
// Boundary: caller asserts block_size % K_TILE == 0, so the K_TILE keys
// share a single physical block. start_n / block_size resolves the block
// for ALL keys in the tile.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void load_k_tile_paged(
    T* __restrict__ K_lds_raw, const T* __restrict__ k_cache,
    const int* __restrict__ block_table, int seq_idx, int kv_head_idx,
    int start_n, int seq_ctx_len, int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_head, int64_t stride_kc_dhi,
    int64_t stride_kc_slot, int lane) {
  constexpr int D_HIGH = HEAD_SIZE / X;
  constexpr int VEC_BYTES = X * sizeof(T);  // 16 for fp16/bf16

  const int my_k_idx = lane >> 1;
  const int my_dh_base = (lane & 1) * (D_HIGH / 2);

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < seq_ctx_len;
  const int log_block = abs_k / block_size;
  const int slot = abs_k - log_block * block_size;
  const int p_block =
      valid_k ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;

  #pragma unroll
  for (int dh = 0; dh < D_HIGH / 2; ++dh) {
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
    static_assert(VEC_BYTES == 16, "vec load assumes 16-byte alignment");
  }
}

// ---------------------------------------------------------------------------
// K chunk load: linear (num_actual_tokens, num_kv_heads, HEAD_SIZE).
// Same destination layout as paged. The chunk K does NOT use the 5D
// vec-friendly layout — it's just (token, head, dim) — so we use
// per-fp16 scalar loads here. Slow but only used during the second
// (current-chunk) phase, which is bounded by query_len << ctx_len in
// the typical long-prompt prefill case where this kernel matters.
//
// TODO(perf): if chunk K is contiguous along HEAD_SIZE, we could vec
// load 16B at a time per (token, head) and shuffle in LDS. v2.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void load_k_tile_chunk(
    T* __restrict__ K_lds_raw, const T* __restrict__ k_chunk,
    int chunk_token_base, int kv_head_idx, int start_n, int chunk_len,
    int64_t stride_kc_token, int64_t stride_kc_head, int lane) {
  constexpr int D_HIGH = HEAD_SIZE / X;

  const int my_k_idx = lane >> 1;
  const int my_dh_base = (lane & 1) * (D_HIGH / 2);

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < chunk_len;
  const T* row =
      valid_k
          ? (k_chunk + (int64_t)(chunk_token_base + abs_k) * stride_kc_token +
             (int64_t)kv_head_idx * stride_kc_head)
          : nullptr;

  #pragma unroll
  for (int dh = 0; dh < D_HIGH / 2; ++dh) {
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

// ---------------------------------------------------------------------------
// V cache load: 4D (num_blocks, num_kv_heads, HEAD_SIZE, block_size).
// Slot is innermost so K_TILE consecutive slots for one (block, head, d)
// are contiguous = 32 B = 2 × 16 B vec loads. Destination V_lds is
// shape (HEAD_SIZE, K_TILE) which matches the source's (d, slot)
// layout — vec write is also contiguous. b_frag access reads
// V_lds[d][k_axis_slot] which is contiguous in the inner dim.
//
// Distribution: lane t handles d range [t * 4, t * 4 + 4) (assuming
// HEAD_SIZE = 128 and 32 lanes). Per d: 2 vec loads (low + high half
// of K_TILE = 16). Per lane: 4 d × 2 vec = 8 vec loads.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X>
__device__ __forceinline__ void load_v_tile_paged(
    T* __restrict__ V_lds,  // shape (HEAD_SIZE, K_TILE)
    const T* __restrict__ v_cache, const int* __restrict__ block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len, int block_size,
    int max_blocks_per_seq, int64_t stride_vc_block, int64_t stride_vc_head,
    int64_t stride_vc_d, int64_t stride_vc_slot, int lane) {
  static_assert(HEAD_SIZE % 32 == 0,
                "v1 V load assumes HEAD_SIZE divisible by wave size 32");
  constexpr int D_PER_LANE = HEAD_SIZE / 32;

  // Tile is fully within one block (caller invariant: block_size % K_TILE ==
  // 0).
  const int log_block = start_n / block_size;
  const int slot_base = start_n - log_block * block_size;
  const int p_block = block_table[seq_idx * max_blocks_per_seq + log_block];
  // Bound the # of valid keys in this tile (last tile may be partial).
  const int valid_k_count = max(0, min(K_TILE, seq_ctx_len - start_n));

  const int my_d_base = lane * D_PER_LANE;
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = my_d_base + dd;
    const T* src = v_cache + (int64_t)p_block * stride_vc_block +
                   (int64_t)kv_head_idx * stride_vc_head +
                   (int64_t)d * stride_vc_d +
                   (int64_t)slot_base * stride_vc_slot;
    int4 vec_lo, vec_hi;
    if (valid_k_count >= K_TILE) {
      vec_lo = *(const int4*)src;
      vec_hi = *(const int4*)(src + 8);
    } else {
      // Partial tile: load element-wise with bounds check; zeros for the rest.
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
}

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_chunk(
    T* __restrict__ V_lds, const T* __restrict__ v_chunk, int chunk_token_base,
    int kv_head_idx, int start_n, int chunk_len, int64_t stride_vc_token,
    int64_t stride_vc_head, int lane) {
  static_assert(HEAD_SIZE % 32 == 0, "");
  constexpr int D_PER_LANE = HEAD_SIZE / 32;

  // V_lds[d][k_idx]. For each (d, k_idx) pair we need V[token = chunk_base +
  // start_n + k_idx][kv_head][d]. Per lane: 4 d values × K_TILE k_idxs = 64
  // fp16 written, with each fp16 from a different token row — strided global
  // access. Slow path kept simple.
  const int my_d_base = lane * D_PER_LANE;
  #pragma unroll
  for (int k_idx = 0; k_idx < K_TILE; ++k_idx) {
    const int abs_k = start_n + k_idx;
    const bool valid = abs_k < chunk_len;
    const T* row =
        valid
            ? (v_chunk + (int64_t)(chunk_token_base + abs_k) * stride_vc_token +
               (int64_t)kv_head_idx * stride_vc_head)
            : nullptr;
  #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = my_d_base + dd;
      V_lds[d * K_TILE + k_idx] = valid ? row[d] : (T)0;
    }
  }
}

// ---------------------------------------------------------------------------
// Online-softmax + WMMA accumulation step. Shared between the prefix
// (no-causal) and chunk (causal) phases. Caller writes the K-tile + V-tile
// to ``K_lds_raw`` / ``V_lds`` first, then calls this. The mask logic is
// templated on ``CAUSAL_MASK``: if true, scores at k_pos > q_pos are
// masked; the ``q_pos_offset`` parameter shifts the q_pos basis (== 0
// for chunk phase where both q and k positions are local to the chunk;
// some other value if you wanted to mask within the prefix, which v1
// does not).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step(
    const T* __restrict__ K_lds_raw, const T* __restrict__ V_lds,
    T* __restrict__ P_lds,  // shape [BLOCK_M=16][K_TILE=16]
    typename WmmaNative<T>::v16 (&q_frags)[HEAD_SIZE / 16],
    v8fp32 (&out_acc)[HEAD_SIZE / 16], float (&m_state)[8], float (&l_state)[8],
    int q_tile_start, int start_n, int valid_q_count, int valid_k_count,
    float sm_scale, int lane, int lane_lo, int lane_hi) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int FRAGS = HEAD_SIZE / 16;

  // ---- Q @ K (8 WMMAs, accumulating into s_acc) ----
  v8fp32 s_acc = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 b_frag;
    // K_lds_raw layout: [d_high (HEAD_SIZE/X)][k (K_TILE)][x (X)]
    // For dh covering D values [dh * 16, (dh + 1) * 16), the two d_highs are
    // dh * 2 and dh * 2 + 1 (since each d_high spans X = 8 fp16/bf16).
    int4 lo =
        *(const int4*)&K_lds_raw[(dh * 2 + 0) * (K_TILE * X) + lane_lo * X];
    int4 hi =
        *(const int4*)&K_lds_raw[(dh * 2 + 1) * (K_TILE * X) + lane_lo * X];
    __builtin_memcpy(&b_frag, &lo, 16);
    __builtin_memcpy(((char*)&b_frag) + 16, &hi, 16);
    s_acc = wmma_mma(q_frags[dh], b_frag, s_acc);
  }

  // ---- Apply scale + mask ----
  // c_acc[i] is at row m = 2*i + lane_hi, column n = lane_lo (within the
  // BLOCK_M × K_TILE score tile). Convert to absolute positions for masking.
  const int abs_k = start_n + lane_lo;
  const bool k_in_seg = (lane_lo < valid_k_count);

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const bool m_in_q = (m_row < valid_q_count);
    bool keep = m_in_q && k_in_seg;
    if constexpr (CAUSAL_MASK) {
      const int abs_q = q_tile_start + m_row;
      keep = keep && (abs_k <= abs_q);
    }
    s_acc[i] = keep ? (s_acc[i] * sm_scale) : -INFINITY;
  }

  // ---- Online softmax ----
  // Per-row max across the K_TILE (16 lanes within the same lane_hi half).
  float m_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_ij[i] = wave16_max(s_acc[i]);
  }

  float m_new[8];
  float alpha[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.0f : __expf(m_state[i] - m_new[i]);
  }

  // p_ij = exp(s - m_new). Same c_acc layout.
  float p_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    p_ij[i] = (m_new[i] == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new[i]);
  }

  // l_ij = sum(p) per row across the K_TILE.
  float l_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    l_ij[i] = wave16_sum(p_ij[i]);
  }

  // Update m_state, l_state, scale acc.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
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

  // ---- Transpose P (c_acc layout) → p_frag (a_frag layout) via LDS ----
  // c_acc: lane_lo = N (k_axis), slot i covers row 2*i + lane_hi.
  // a_frag: lane_lo = M (m_row), slot k covers column k.
  // Cross-lane gather → LDS round-trip.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    P_lds[m_row * K_TILE + lane_lo] = to_T<T>(p_ij[i]);
  }
  // Single-wave: no __syncthreads needed; compiler emits s_waitcnt
  // lgkmcnt(0) before subsequent ds_reads. (Lesson 9 from W4A16 WMMA v1.)

  V16 p_frag;
  // p_frag[k] = P[lane_lo (m)][k (slot)]. Read 16 fp16/bf16 at
  // P_lds[lane_lo][:].
  int4 p_lo = *(const int4*)&P_lds[lane_lo * K_TILE + 0];
  int4 p_hi = *(const int4*)&P_lds[lane_lo * K_TILE + 8];
  __builtin_memcpy(&p_frag, &p_lo, 16);
  __builtin_memcpy(((char*)&p_frag) + 16, &p_hi, 16);

  // ---- P @ V (8 WMMAs accumulating into out_acc per d-fragment) ----
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 v_frag;
    // V_lds[d (HEAD_SIZE)][k (K_TILE)]. Lane lane_lo handles d_out =
    // dh * 16 + lane_lo. For v_frag access, slot=k_axis, lane=N=d_out.
    // Read V_lds[dh * 16 + lane_lo][0..15] = 32 B contiguous.
    int4 v_lo = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 0];
    int4 v_hi = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 8];
    __builtin_memcpy(&v_frag, &v_lo, 16);
    __builtin_memcpy(((char*)&v_frag) + 16, &v_hi, 16);
    out_acc[dh] = wmma_mma(p_frag, v_frag, out_acc[dh]);
  }
}

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__global__ void paged_prefill_attn_kernel(
    T* __restrict__ out, const T* __restrict__ q, const T* __restrict__ k_chunk,
    const T* __restrict__ v_chunk, const T* __restrict__ k_cache,
    const T* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q, const int* __restrict__ seq_lens,
    const int num_query_heads, const int num_kv_heads, const int block_size,
    const int max_blocks_per_seq, const int x, const float sm_scale,
    const bool causal, const int64_t stride_q_token,
    const int64_t stride_q_head, const int64_t stride_kc_token,
    const int64_t stride_kc_head, const int64_t stride_vc_token,
    const int64_t stride_vc_head, const int64_t stride_kcache_block,
    const int64_t stride_kcache_head, const int64_t stride_kcache_dhi,
    const int64_t stride_kcache_slot, const int64_t stride_vcache_block,
    const int64_t stride_vcache_head, const int64_t stride_vcache_d,
    const int64_t stride_vcache_slot, const int64_t stride_o_token,
    const int64_t stride_o_head) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int FRAGS = HEAD_SIZE / 16;
  constexpr int X = 16 / sizeof(T);  // 8 for fp16/bf16

  const int seq_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int lane = threadIdx.x;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  // Per-seq metadata.
  const int q_start_token = cu_seqlens_q[seq_idx];
  const int q_end_token = cu_seqlens_q[seq_idx + 1];
  const int query_len = q_end_token - q_start_token;
  const int seq_len = seq_lens[seq_idx];
  const int ctx_len = seq_len - query_len;

  const int q_tile_start = q_tile_idx * BLOCK_M;
  if (q_tile_start >= query_len) return;

  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  // Per-lane query position (lane_lo == m row in the tile).
  const int my_m_row = lane_lo;
  const int my_q_pos = q_tile_start + my_m_row;
  const bool valid_q = my_q_pos < query_len;
  const int valid_q_count = min(BLOCK_M, query_len - q_tile_start);

  // ---- Load Q (8 a_fragments per lane) ----
  V16 q_frags[FRAGS];
  if (valid_q) {
    const T* q_row = q + (int64_t)(q_start_token + my_q_pos) * stride_q_token +
                     (int64_t)head_idx * stride_q_head;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      static_assert(sizeof(V16) == 32, "V16 must be 32 bytes (16 × 2)");
      __builtin_memcpy(&q_frags[dh], q_row + dh * 16, sizeof(V16));
    }
  } else {
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
  #pragma unroll
      for (int k = 0; k < 16; ++k) q_frags[dh][k] = (E)0;
    }
  }

  // ---- Init online softmax state ----
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

  // ---- LDS workspaces ----
  __shared__ T
      K_lds_raw[FRAGS * 2 * K_TILE * X];  // == HEAD_SIZE * K_TILE / 2 ... wait
  // Actually: K_lds layout is [d_high (HEAD_SIZE/X)][k (K_TILE)][x (X)].
  // Total size: (HEAD_SIZE / X) * K_TILE * X = HEAD_SIZE * K_TILE = 2048 fp16 =
  // 4 KB.
  __shared__ T V_lds[HEAD_SIZE * K_TILE];  // 4 KB + 256 B pad
  __shared__ T P_lds[BLOCK_M * K_TILE];    // 512 B + 32 B pad

  static_assert(sizeof(K_lds_raw) == HEAD_SIZE * K_TILE * sizeof(T),
                "K_lds size mismatch");

  // ---- PHASE 1: Cached prefix (no causal mask) ----
  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_paged<T, HEAD_SIZE, X>(
        K_lds_raw, k_cache, block_table, seq_idx, kv_head_idx, start_n, ctx_len,
        block_size, max_blocks_per_seq, stride_kcache_block, stride_kcache_head,
        stride_kcache_dhi, stride_kcache_slot, lane);
    load_v_tile_paged<T, HEAD_SIZE, X>(
        V_lds, v_cache, block_table, seq_idx, kv_head_idx, start_n, ctx_len,
        block_size, max_blocks_per_seq, stride_vcache_block, stride_vcache_head,
        stride_vcache_d, stride_vcache_slot, lane);
    const int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step<T, HEAD_SIZE, X, /*CAUSAL_MASK=*/false>(
        K_lds_raw, V_lds, P_lds, q_frags, out_acc, m_state, l_state,
        q_tile_start, start_n, valid_q_count, valid_k_count, sm_scale, lane,
        lane_lo, lane_hi);
  }

  // ---- PHASE 2: Current chunk (causal mask) ----
  // The causal upper bound for the K direction within this Q-tile:
  // any K beyond q_tile_start + valid_q_count - 1 contributes nothing
  // (would be masked anyway). Trim the loop to save iterations.
  const int causal_k_upper =
      causal ? (q_tile_start + valid_q_count) : query_len;
  const int phase2_k_end = min(query_len, causal_k_upper);
  for (int start_n = 0; start_n < phase2_k_end; start_n += K_TILE) {
    load_k_tile_chunk<T, HEAD_SIZE, X>(K_lds_raw, k_chunk, q_start_token,
                                       kv_head_idx, start_n, query_len,
                                       stride_kc_token, stride_kc_head, lane);
    load_v_tile_chunk<T, HEAD_SIZE>(V_lds, v_chunk, q_start_token, kv_head_idx,
                                    start_n, query_len, stride_vc_token,
                                    stride_vc_head, lane);
    const int valid_k_count = min(K_TILE, query_len - start_n);
    attn_step<T, HEAD_SIZE, X, /*CAUSAL_MASK=*/true>(
        K_lds_raw, V_lds, P_lds, q_frags, out_acc, m_state, l_state,
        q_tile_start, start_n, valid_q_count, valid_k_count, sm_scale, lane,
        lane_lo, lane_hi);
  }

  // ---- Epilogue: divide by L, write output ----
  // Each lane holds the c_acc values for 8 m_rows (rows 2*i + lane_hi for
  // i = 0..7) at column n = lane_lo within each of the FRAGS output
  // d-fragments. For HEAD_SIZE = 128 and 16-lane spread, lanes 0..15
  // (lane_hi == 0) cover EVEN rows {0, 2, ..., 14} and lanes 16..31
  // cover ODD rows {1, 3, ..., 15}. Per (m_row, n) cell there is
  // EXACTLY ONE lane — no atomic / no shared race.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const int abs_q_pos = q_tile_start + m_row;
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
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
void launch_paged_prefill_attn(
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
  const int q_blocks = (max_query_len + BLOCK_M - 1) / BLOCK_M;
  dim3 block(32);
  dim3 grid(num_seqs, num_query_heads, q_blocks);
  paged_prefill_attn_kernel<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache, block_table, cu_seqlens_q,
      seq_lens, num_query_heads, num_kv_heads, block_size, max_blocks_per_seq,
      X, sm_scale, causal, stride_q_token, stride_q_head, stride_kc_token,
      stride_kc_head, stride_vc_token, stride_vc_head, stride_kcache_block,
      stride_kcache_head, stride_kcache_dhi, stride_kcache_slot,
      stride_vcache_block, stride_vcache_head, stride_vcache_d,
      stride_vcache_slot, stride_o_token, stride_o_head);
}

#endif  // USE_ROCM

}  // namespace prefill_attn_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   out         [num_actual_tokens, num_query_heads, head_size]  fp16/bf16
//   (in-place) q           [num_actual_tokens, num_query_heads, head_size]
//   fp16/bf16 k_chunk     [num_actual_tokens, num_kv_heads, head_size]
//   fp16/bf16 v_chunk     [num_actual_tokens, num_kv_heads, head_size]
//   fp16/bf16 k_cache     [num_blocks, num_kv_heads, head_size//x, block_size,
//   x] v_cache     [num_blocks, num_kv_heads, head_size, block_size]
//   block_table [num_seqs, max_blocks_per_seq] int32
//   cu_seqlens_q[num_seqs + 1] int32
//   seq_lens    [num_seqs] int32
//   sm_scale    float
//   causal      bool
//
// Constraints (caller responsibility — falls through to Triton if violated):
//   * dtype ∈ {fp16, bf16}
//   * head_size == 128 (v1)
//   * block_size % K_TILE (== 16) == 0
//   * No FP8 KV cache, alibi, sliding window, sinks, FP8 output.

// Forward decl from paged_prefill_attn_rdna3_v2.cu (Lesson 4: separate TU
// to keep hipcc's TU-level optimizer decisions isolated). Cross-TU call
// resolved at link time, no codegen interaction.
namespace vllm {
namespace prefill_attn_rdna3_v2 {
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
    cudaStream_t stream);
}  // namespace prefill_attn_rdna3_v2
}  // namespace vllm

void paged_prefill_attn_rdna3(torch::Tensor& out, torch::Tensor q,
                              torch::Tensor k_chunk, torch::Tensor v_chunk,
                              torch::Tensor k_cache, torch::Tensor v_cache,
                              torch::Tensor block_table,
                              torch::Tensor cu_seqlens_q,
                              torch::Tensor seq_lens, int64_t max_query_len,
                              double sm_scale, bool causal) {
  TORCH_CHECK(out.is_cuda() && q.is_cuda() && k_chunk.is_cuda() &&
                  v_chunk.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() &&
                  block_table.is_cuda() && cu_seqlens_q.is_cuda() &&
                  seq_lens.is_cuda(),
              "all tensors must be CUDA/HIP");
  TORCH_CHECK(q.dim() == 3 && k_chunk.dim() == 3 && v_chunk.dim() == 3 &&
                  out.dim() == 3,
              "q/k/v/out must be 3D");
  TORCH_CHECK(k_cache.dim() == 5, "k_cache must be 5D");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(
      q.scalar_type() == torch::kHalf || q.scalar_type() == torch::kBFloat16,
      "q must be fp16 or bf16");
  TORCH_CHECK(out.scalar_type() == q.scalar_type(),
              "out dtype must match q dtype");
  TORCH_CHECK(k_chunk.scalar_type() == q.scalar_type(),
              "k_chunk dtype must match q dtype");
  TORCH_CHECK(v_chunk.scalar_type() == q.scalar_type(),
              "v_chunk dtype must match q dtype");

  const int num_query_heads = q.size(1);
  const int head_size = q.size(2);
  const int num_kv_heads = k_chunk.size(1);
  const int num_seqs = (int)seq_lens.size(0);
  const int block_size = (int)v_cache.size(3);
  const int max_blocks_per_seq = (int)block_table.size(1);

  TORCH_CHECK(head_size == 128,
              "v1 RDNA3 prefill kernel only supports head_size == 128 "
              "(got ",
              head_size, ")");
  TORCH_CHECK(block_size % 16 == 0,
              "v1 RDNA3 prefill kernel requires block_size % 16 == 0 "
              "(got ",
              block_size, ")");
  TORCH_CHECK(num_query_heads % num_kv_heads == 0,
              "num_query_heads must be a multiple of num_kv_heads");

  // ``max_query_len`` is supplied by the caller (already a Python int in
  // ``RocmAttentionMetadata``), so we avoid the GPU->CPU sync that would
  // be required to derive it from ``cu_seqlens_q`` here.  Doing the sync
  // would stall the stream every prefill call (~once per layer per step),
  // which kills CUDA-graph / kernel overlap on hot serving paths.
  TORCH_CHECK(max_query_len > 0, "max_query_len must be > 0 (got ",
              max_query_len, ")");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  // Dispatch v1 (1 wave/block, BLOCK_M=16) vs v2 (4 waves/block, BLOCK_M=64).
  // v2 wins for max_query_len >= 64 because the 4× cooperative K/V load
  // amortizes the same K-tile across 4× output rows. Below 64, v2 wastes
  // 3 of its 4 waves on out-of-range rows; v1's smaller BLOCK_M=16 is
  // strictly better there.
  const bool use_v2 = (max_query_len >= 64);

#if defined(USE_ROCM)
  if (q.scalar_type() == torch::kHalf) {
    if (use_v2) {
      vllm::prefill_attn_rdna3_v2::launch_paged_prefill_attn_v2<half, 128>(
          (half*)out.data_ptr(), (const half*)q.data_ptr(),
          (const half*)k_chunk.data_ptr(), (const half*)v_chunk.data_ptr(),
          (const half*)k_cache.data_ptr(), (const half*)v_cache.data_ptr(),
          (const int*)block_table.data_ptr(),
          (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(),
          num_seqs, num_query_heads, num_kv_heads, block_size,
          max_blocks_per_seq, max_query_len, (float)sm_scale, causal,
          q.stride(0), q.stride(1), k_chunk.stride(0), k_chunk.stride(1),
          v_chunk.stride(0), v_chunk.stride(1), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
          v_cache.stride(3), out.stride(0), out.stride(1), stream);
    } else {
      vllm::prefill_attn_rdna3::launch_paged_prefill_attn<half, 128>(
          (half*)out.data_ptr(), (const half*)q.data_ptr(),
          (const half*)k_chunk.data_ptr(), (const half*)v_chunk.data_ptr(),
          (const half*)k_cache.data_ptr(), (const half*)v_cache.data_ptr(),
          (const int*)block_table.data_ptr(),
          (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(),
          num_seqs, num_query_heads, num_kv_heads, block_size,
          max_blocks_per_seq, max_query_len, (float)sm_scale, causal,
          q.stride(0), q.stride(1), k_chunk.stride(0), k_chunk.stride(1),
          v_chunk.stride(0), v_chunk.stride(1), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
          v_cache.stride(3), out.stride(0), out.stride(1), stream);
    }
  } else {
    using bf = vllm::prefill_attn_rdna3::bf16_t;
    if (use_v2) {
      vllm::prefill_attn_rdna3_v2::launch_paged_prefill_attn_v2<bf, 128>(
          (bf*)out.data_ptr(), (const bf*)q.data_ptr(),
          (const bf*)k_chunk.data_ptr(), (const bf*)v_chunk.data_ptr(),
          (const bf*)k_cache.data_ptr(), (const bf*)v_cache.data_ptr(),
          (const int*)block_table.data_ptr(),
          (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(),
          num_seqs, num_query_heads, num_kv_heads, block_size,
          max_blocks_per_seq, max_query_len, (float)sm_scale, causal,
          q.stride(0), q.stride(1), k_chunk.stride(0), k_chunk.stride(1),
          v_chunk.stride(0), v_chunk.stride(1), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
          v_cache.stride(3), out.stride(0), out.stride(1), stream);
    } else {
      vllm::prefill_attn_rdna3::launch_paged_prefill_attn<bf, 128>(
          (bf*)out.data_ptr(), (const bf*)q.data_ptr(),
          (const bf*)k_chunk.data_ptr(), (const bf*)v_chunk.data_ptr(),
          (const bf*)k_cache.data_ptr(), (const bf*)v_cache.data_ptr(),
          (const int*)block_table.data_ptr(),
          (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(),
          num_seqs, num_query_heads, num_kv_heads, block_size,
          max_blocks_per_seq, max_query_len, (float)sm_scale, causal,
          q.stride(0), q.stride(1), k_chunk.stride(0), k_chunk.stride(1),
          v_chunk.stride(0), v_chunk.stride(1), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
          v_cache.stride(3), out.stride(0), out.stride(1), stream);
    }
  }
#else
  TORCH_CHECK(false,
              "paged_prefill_attn_rdna3 is only available on ROCm (gfx11)");
#endif
}
