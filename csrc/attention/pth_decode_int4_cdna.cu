// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-token-head INT4 paged decode attention for CDNA. Mirrors
// pth_decode_int8_cdna.cu but uses packed-nibble K/V cache with
// steganographed (scale, zp) fp32.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_cdna.cuh"

namespace vllm {
namespace pth_decode_int4_cdna {

#if defined(USE_ROCM)

using vllm::prefill_attn_cdna::bf16_t;
using vllm::prefill_attn_cdna::from_float_rn;
using vllm::prefill_attn_cdna::to_float;

constexpr int THREADS = 64;

__device__ __forceinline__ void unpack_scale_zp(uint32_t packed, float& scale,
                                                int& zp) {
  zp = (int)(packed & 0xFu);
  uint32_t scale_bits = packed & 0xFFFFFFF0u;
  __builtin_memcpy(&scale, &scale_bits, 4);
}

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
void pth_decode_int4_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
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
  // K-parallel FlashAttention-decode for INT4 packed KV. Same wave-merge as
  // the INT8 sibling; the only structural difference is the packed-nibble
  // dequant in the Q·K and V·P inner loops.
  constexpr int BYTES_PER_ROW = HEAD_SIZE / 2;
  constexpr int OWN_D = HEAD_SIZE / THREADS;       // owned output dims/lane
  static_assert(BYTES_PER_ROW >= THREADS,
                "INT4 decode requires HEAD_SIZE >= 128 (so >=64 bytes/row)");
  static_assert(HEAD_SIZE % THREADS == 0,
                "HEAD_SIZE must be a multiple of 64");
  constexpr int O_LDS_STRIDE = HEAD_SIZE + 1;      // +1 to break bank periodicity

  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tid = threadIdx.x;
  int num_queries_per_kv = num_query_heads / num_kv_heads;
  int kv_head_idx = head_idx / num_queries_per_kv;
  int seq_len = seq_lens[seq_idx];

  __shared__ T Q_lds[HEAD_SIZE];
  __shared__ float O_lds[THREADS * O_LDS_STRIDE];
  for (int d = tid; d < HEAD_SIZE; d += THREADS) {
    Q_lds[d] = q[(int64_t)seq_idx * stride_q_seq +
                  head_idx * stride_q_head + d];
  }
  __syncthreads();

  // Each lane keeps the full HEAD_SIZE per-lane output partial.
  float m_local = -INFINITY;
  float l_local = 0.f;
  float o_local[HEAD_SIZE];
  #pragma unroll
  for (int d = 0; d < HEAD_SIZE; ++d) o_local[d] = 0.f;

  // K-loop: lane t handles k = t, t+THREADS, ... < seq_len.
  for (int k = tid; k < seq_len; k += THREADS) {
    int log_blk = k / block_size;
    int slot = k - log_blk * block_size;
    int p_blk = block_table[seq_idx * max_blocks_per_seq + log_blk];
    const uint8_t* k_row = k_cache + (int64_t)p_blk * stride_kc_block +
                           slot * stride_kc_slot +
                           kv_head_idx * stride_kc_head;
    const uint8_t* v_row = v_cache + (int64_t)p_blk * stride_vc_block +
                           slot * stride_vc_slot +
                           kv_head_idx * stride_vc_head;
    uint32_t k_packed = 0, v_packed = 0;
    __builtin_memcpy(&k_packed,
                     k_scale_cache + p_blk * stride_ks_blk +
                         slot * stride_ks_slot + kv_head_idx * stride_ks_head,
                     4);
    __builtin_memcpy(&v_packed,
                     v_scale_cache + p_blk * stride_vs_blk +
                         slot * stride_vs_slot + kv_head_idx * stride_vs_head,
                     4);
    float k_sc, v_sc;
    int k_zp, v_zp;
    unpack_scale_zp(k_packed, k_sc, k_zp);
    unpack_scale_zp(v_packed, v_sc, v_zp);

    // Q · K dot, 8 bytes (16 dims) per chunk.
    float s = 0.f;
    #pragma unroll
    for (int b = 0; b < BYTES_PER_ROW; b += 8) {
      uint8_t bytes[8];
      *(int2*)bytes = *(const int2*)(k_row + b);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        int lo = (int)(bytes[i] & 0xFu) - k_zp;
        int hi = (int)((bytes[i] >> 4) & 0xFu) - k_zp;
        s += to_float<T>(Q_lds[b * 2 + 2 * i + 0]) * (float)lo;
        s += to_float<T>(Q_lds[b * 2 + 2 * i + 1]) * (float)hi;
      }
    }
    s = s * sm_scale * k_sc;

    // Per-lane online softmax.
    float m_new = fmaxf(m_local, s);
    float alpha = (m_local == -INFINITY) ? 0.f : __expf(m_local - m_new);
    float pij = (m_new == -INFINITY) ? 0.f : __expf(s - m_new);
    l_local = l_local * alpha + pij;
    m_local = m_new;
    float p_v = pij * v_sc;

    // V · P over ALL HEAD_SIZE output dims (8 bytes = 16 dims per chunk).
    #pragma unroll
    for (int b = 0; b < BYTES_PER_ROW; b += 8) {
      uint8_t bytes[8];
      *(int2*)bytes = *(const int2*)(v_row + b);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        int lo = (int)(bytes[i] & 0xFu) - v_zp;
        int hi = (int)((bytes[i] >> 4) & 0xFu) - v_zp;
        int d_lo = b * 2 + 2 * i + 0;
        int d_hi = b * 2 + 2 * i + 1;
        o_local[d_lo] = o_local[d_lo] * alpha + p_v * (float)lo;
        o_local[d_hi] = o_local[d_hi] * alpha + p_v * (float)hi;
      }
    }
  }

  // Wave-wide online-softmax merge (same identity as INT8 sibling).
  float m_global = wave64_max(m_local);
  float alpha_lane = (m_local == -INFINITY) ? 0.f
                                            : __expf(m_local - m_global);
  float l_global = wave64_sum(alpha_lane * l_local);
  float inv_l = 1.f / (l_global + 1e-10f);
  #pragma unroll
  for (int d = 0; d < HEAD_SIZE; ++d) {
    o_local[d] *= alpha_lane;
  }

  // LDS transpose: lane-rows → dim-columns, then column-sum + write. No
  // pre-write barrier: Q_lds and O_lds are distinct LDS allocations, and
  // the K-loop's Q_lds reads have all retired at the loop convergence
  // point. Only the post-write barrier is required (lane d reads other
  // lanes' rows in the column-sum).
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
void launch(T* out, const T* q, const uint8_t* k_cache, const uint8_t* v_cache,
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
  pth_decode_int4_kernel<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
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

}  // namespace pth_decode_int4_cdna
}  // namespace vllm

void pth_decode_int4_cdna(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor block_table,
    torch::Tensor seq_lens, double sm_scale) {
#if defined(USE_ROCM)
  using namespace vllm::pth_decode_int4_cdna;
  using vllm::prefill_attn_cdna::bf16_t;

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16);
  TORCH_CHECK(k_cache.dtype() == at::kByte && v_cache.dtype() == at::kByte);

  int num_seqs = q.size(0);
  int num_query_heads = q.size(1);
  int head_size = q.size(2);
  int num_kv_heads = k_scale_cache.size(2);
  int block_size = k_cache.size(1);
  int max_blocks_per_seq = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH(T, HS)                                                        \
    launch<T, HS>((T*)out.data_ptr(), (const T*)q.data_ptr(),                  \
                  (const uint8_t*)k_cache.data_ptr(),                          \
                  (const uint8_t*)v_cache.data_ptr(),                          \
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
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false,
                "pth_decode_int4_cdna: unsupported head_size=", head_size,
                " (only 128 supported; INT4 decode needs >=64 bytes/row)");
    }
  } else {
    using T = vllm::prefill_attn_cdna::bf16_t;
    switch (head_size) {
      case 128: LAUNCH(T, 128); break;
      default: TORCH_CHECK(false,
                "pth_decode_int4_cdna: unsupported head_size=", head_size,
                " (only 128 supported; INT4 decode needs >=64 bytes/row)");
    }
  }
  #undef LAUNCH
#else
  TORCH_CHECK(false, "pth_decode_int4_cdna requires gfx942/950/90a");
#endif
}
