// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-token-head INT8 paged decode attention for CDNA (gfx942/950/90a).
// One CTA per (sequence, query-head). 64 threads = 1 wave. Each thread
// owns HEAD_SIZE/64 output dims and iterates over the KV axis with
// stride 64 (FlashAttention-decode pattern, single-CTA variant).
//
// Online softmax is computed per-thread; a wave-wide merge folds the 64
// partial (m, l) states into a global (m, l) and rescales each thread's
// owned-output-dim slice before the final divide.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_cdna.cuh"

namespace vllm {
namespace pth_decode_int8_cdna {

#if defined(USE_ROCM)

using vllm::prefill_attn_cdna::bf16_t;
using vllm::prefill_attn_cdna::from_float_rn;
using vllm::prefill_attn_cdna::to_float;

constexpr int THREADS = 64;  // one wave64

// Wave-wide max / sum across 64 lanes (5 butterfly stages).
__device__ __forceinline__ float wave64_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  v = fmaxf(v, __shfl_xor(v, 16));
  v = fmaxf(v, __shfl_xor(v, 32));
  return v;
}
__device__ __forceinline__ float wave64_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  v += __shfl_xor(v, 16);
  v += __shfl_xor(v, 32);
  return v;
}

template <typename T, int HEAD_SIZE>
__global__ __launch_bounds__(THREADS, 1)
void pth_decode_int8_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int num_query_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq, float sm_scale,
    int64_t stride_q_seq, int64_t stride_q_head,
    int64_t stride_kc_block, int64_t stride_kc_slot, int64_t stride_kc_head,
    int64_t stride_vc_block, int64_t stride_vc_slot, int64_t stride_vc_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    int64_t stride_o_seq, int64_t stride_o_head) {
  // FlashAttention-decode pattern: the wave (64 lanes) is parallel along the
  // K (sequence) axis. Lane t handles k = t, t+THREADS, t+2*THREADS, … < L.
  // Each lane keeps its own (m_t, l_t, o_t[0..HEAD_SIZE-1]) partial. After
  // the K-loop a wave-wide online-softmax merge produces (m_g, l_g) and the
  // per-lane o_t is rescaled by alpha_t = exp(m_t - m_g). The HEAD_SIZE
  // owned-dims of the final output are then materialised by transposing the
  // 64 × HEAD_SIZE partials through LDS and summing the lane dimension.
  constexpr int OWN_D = HEAD_SIZE / THREADS;
  static_assert(HEAD_SIZE % THREADS == 0,
                "HEAD_SIZE must be a multiple of 64");
  // +1 float of LDS padding per row to break the 32-bank periodicity on the
  // column-major reads in the output reduction.
  constexpr int O_LDS_STRIDE = HEAD_SIZE + 1;

  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tid = threadIdx.x;
  int num_queries_per_kv = num_query_heads / num_kv_heads;
  int kv_head_idx = head_idx / num_queries_per_kv;
  int seq_len = seq_lens[seq_idx];

  // ----- Stage Q in LDS (broadcast read by every lane in the K-loop). ----
  // Two static LDS allocations:
  //   Q_lds  — HEAD_SIZE elements of T (~128–256 B), broadcast Q to all lanes.
  //   O_lds  — 64 × (HEAD_SIZE+1) floats (~16 KB / 33 KB), per-lane output
  //            partial used for the column-sum reduction at the end.
  __shared__ T Q_lds[HEAD_SIZE];
  __shared__ float O_lds[THREADS * O_LDS_STRIDE];
  for (int d = tid; d < HEAD_SIZE; d += THREADS) {
    Q_lds[d] = q[(int64_t)seq_idx * stride_q_seq +
                  head_idx * stride_q_head + d];
  }
  __syncthreads();

  // ----- Per-lane online-softmax state ----------------------------------
  // o_local holds the full HEAD_SIZE per-lane partial of sum_{k in R_t} of
  // exp(s_k - m_t) * V[k]. Sized at compile time so the array stays in
  // VGPRs (~64-128 floats/lane).
  float m_local = -INFINITY;
  float l_local = 0.f;
  float o_local[HEAD_SIZE];
  #pragma unroll
  for (int d = 0; d < HEAD_SIZE; ++d) o_local[d] = 0.f;

  // ----- K-loop: each lane owns a strided K subset -----------------------
  // Lane t processes k = t, t + THREADS, ... Lanes whose range is empty
  // (seq_len <= t) keep m_local = -INF, l_local = 0, o_local = 0 — they fold
  // cleanly through the wave-merge below (alpha_t = 0).
  for (int k = tid; k < seq_len; k += THREADS) {
    int log_blk = k / block_size;
    int slot = k - log_blk * block_size;
    int p_blk = block_table[seq_idx * max_blocks_per_seq + log_blk];
    const int8_t* k_row = k_cache + (int64_t)p_blk * stride_kc_block +
                          slot * stride_kc_slot +
                          kv_head_idx * stride_kc_head;
    const int8_t* v_row = v_cache + (int64_t)p_blk * stride_vc_block +
                          slot * stride_vc_slot +
                          kv_head_idx * stride_vc_head;
    float k_sc = k_scale_cache[p_blk * stride_ks_blk +
                                slot * stride_ks_slot +
                                kv_head_idx * stride_ks_head];
    float v_sc = v_scale_cache[p_blk * stride_vs_blk +
                                slot * stride_vs_slot +
                                kv_head_idx * stride_vs_head];

    // Q · K dot, 16 bytes per chunk (keeps K bytes off the live register
    // set across the softmax update).
    float s = 0.f;
    #pragma unroll
    for (int b = 0; b < HEAD_SIZE; b += 16) {
      int8_t bytes[16];
      *(int4*)bytes = *(const int4*)(k_row + b);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        s += to_float<T>(Q_lds[b + i]) * (float)bytes[i];
      }
    }
    s = s * sm_scale * k_sc;

    // Online softmax — per-lane (each lane has its own running (m_t, l_t)).
    float m_new = fmaxf(m_local, s);
    float alpha = (m_local == -INFINITY) ? 0.f : __expf(m_local - m_new);
    float pij = (m_new == -INFINITY) ? 0.f : __expf(s - m_new);
    l_local = l_local * alpha + pij;
    m_local = m_new;
    float p_v = pij * v_sc;

    // V · P accumulation across ALL HEAD_SIZE output dims (V row loaded in
    // 16-byte chunks for the same reason as K above).
    #pragma unroll
    for (int b = 0; b < HEAD_SIZE; b += 16) {
      int8_t bytes[16];
      *(int4*)bytes = *(const int4*)(v_row + b);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        o_local[b + i] = o_local[b + i] * alpha + p_v * (float)bytes[i];
      }
    }
  }

  // ----- Wave-wide online-softmax merge ---------------------------------
  // Each lane t has its own (m_t, l_t). Compute the global m_g and l_g and
  // rescale this lane's o partial to the global m-base so a simple sum
  // across lanes yields the correct unnormalised output.
  float m_global = wave64_max(m_local);
  float alpha_lane = (m_local == -INFINITY) ? 0.f
                                            : __expf(m_local - m_global);
  float l_global = wave64_sum(alpha_lane * l_local);
  float inv_l = 1.f / (l_global + 1e-10f);
  #pragma unroll
  for (int d = 0; d < HEAD_SIZE; ++d) {
    o_local[d] *= alpha_lane;
  }

  // ----- Output reduction via LDS transpose -----------------------------
  // Lane t writes its full o_local row at rows[t] of an LDS [64 × stride]
  // tile. After a barrier each lane sums down the column for its OWN_D
  // owned output dims and writes the normalised result. No barrier is
  // needed before the writes because Q_lds and O_lds are distinct LDS
  // allocations and the K-loop reads of Q_lds have all retired by the
  // post-loop convergence point.
  #pragma unroll
  for (int d = 0; d < HEAD_SIZE; ++d) {
    O_lds[tid * O_LDS_STRIDE + d] = o_local[d];
  }
  __syncthreads();

  T* out_row = out + (int64_t)seq_idx * stride_o_seq +
               head_idx * stride_o_head;
  #pragma unroll
  for (int i = 0; i < OWN_D; ++i) {
    int d = tid * OWN_D + i;
    float sum_d = 0.f;
    #pragma unroll
    for (int t = 0; t < THREADS; ++t) {
      sum_d += O_lds[t * O_LDS_STRIDE + d];
    }
    out_row[d] = from_float_rn<T>(sum_d * inv_l);
  }
}

template <typename T, int HEAD_SIZE>
void launch(T* out, const T* q, const int8_t* k_cache, const int8_t* v_cache,
            const float* k_scale_cache, const float* v_scale_cache,
            const int* block_table, const int* seq_lens, int num_seqs,
            int num_query_heads, int num_kv_heads, int block_size,
            int max_blocks_per_seq, float sm_scale,
            int64_t stride_q_seq, int64_t stride_q_head,
            int64_t stride_kc_block, int64_t stride_kc_slot,
            int64_t stride_kc_head,
            int64_t stride_vc_block, int64_t stride_vc_slot,
            int64_t stride_vc_head,
            int64_t stride_ks_blk, int64_t stride_ks_slot,
            int64_t stride_ks_head,
            int64_t stride_vs_blk, int64_t stride_vs_slot,
            int64_t stride_vs_head,
            int64_t stride_o_seq, int64_t stride_o_head,
            cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid(num_seqs, num_query_heads);
  pth_decode_int8_kernel<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_cache, v_cache, k_scale_cache, v_scale_cache,
      block_table, seq_lens, num_query_heads, num_kv_heads, block_size,
      max_blocks_per_seq, sm_scale,
      stride_q_seq, stride_q_head,
      stride_kc_block, stride_kc_slot, stride_kc_head,
      stride_vc_block, stride_vc_slot, stride_vc_head,
      stride_ks_blk, stride_ks_slot, stride_ks_head,
      stride_vs_blk, stride_vs_slot, stride_vs_head,
      stride_o_seq, stride_o_head);
}

#endif  // gfx90a/942/950

}  // namespace pth_decode_int8_cdna
}  // namespace vllm

void pth_decode_int8_cdna(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, double sm_scale) {
#if defined(USE_ROCM)
  using namespace vllm::pth_decode_int8_cdna;
  using vllm::prefill_attn_cdna::bf16_t;

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16);
  TORCH_CHECK(k_cache.dtype() == at::kChar && v_cache.dtype() == at::kChar);

  int num_seqs = q.size(0);
  int num_query_heads = q.size(1);
  int head_size = q.size(2);
  int num_kv_heads = k_scale_cache.size(2);
  int block_size = k_cache.size(1);
  int max_blocks_per_seq = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH(T, HS)                                                        \
    launch<T, HS>((T*)out.data_ptr(), (const T*)q.data_ptr(),                  \
                  (const int8_t*)k_cache.data_ptr(),                           \
                  (const int8_t*)v_cache.data_ptr(),                           \
                  (const float*)k_scale_cache.data_ptr(),                      \
                  (const float*)v_scale_cache.data_ptr(),                      \
                  (const int*)block_table.data_ptr(),                          \
                  (const int*)seq_lens.data_ptr(),                             \
                  num_seqs, num_query_heads, num_kv_heads, block_size,         \
                  max_blocks_per_seq, (float)sm_scale,                         \
                  q.stride(0), q.stride(1),                                    \
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
      case 64:  LAUNCH(T, 64);  break;
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false, "unsupported head_size=", head_size);
    }
  } else {
    using T = vllm::prefill_attn_cdna::bf16_t;
    switch (head_size) {
      case 64:  LAUNCH(T, 64);  break;
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false, "unsupported head_size=", head_size);
    }
  }
  #undef LAUNCH
#else
  TORCH_CHECK(false, "pth_decode_int8_cdna requires gfx942/950/90a");
#endif
}
