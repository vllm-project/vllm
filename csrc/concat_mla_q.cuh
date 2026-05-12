#ifndef CONCAT_MLA_Q_CUH_
#define CONCAT_MLA_Q_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda_vec_utils.cuh"

namespace vllm {

// Concatenates ql_nope [num_tokens, num_heads, NOPE_DIM] and
// q_pe [num_tokens, num_heads, 64]
// into q_out [num_tokens, num_heads, NOPE_DIM+64].
// Currently instantiated only for NOPE_DIM=512.
// Rope dim is hardcoded to 64 (DeepSeek V3.2 MLA)
template <typename DType, int NOPE_DIM>
__global__ void ConcatMLAQKernel(
    DType* __restrict__ q_out, const DType* __restrict__ ql_nope,
    const DType* __restrict__ q_pe, const int num_tokens, const int num_heads,
    const int64_t out_stride_0, const int64_t out_stride_1,
    const int64_t nope_stride_0, const int64_t nope_stride_1,
    const int64_t pe_stride_0, const int64_t pe_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= num_tokens * num_heads) return;

  const int token_id = flat_warp_id / num_heads;
  const int head_id = flat_warp_id % num_heads;
  const int lane_id = threadIdx.x & 31;

  constexpr bool use_256b = VLLM_256B_PTX_ENABLED;
  constexpr int nope_vec_loads =
      NOPE_DIM * sizeof(DType) / (VecTraits<use_256b>::ARCH_MAX_VEC_SIZE * 32);

  const DType* nope_src =
      ql_nope + token_id * nope_stride_0 + head_id * nope_stride_1;
  DType* nope_dst = q_out + token_id * out_stride_0 + head_id * out_stride_1;

#pragma unroll
  for (int i = 0; i < nope_vec_loads; i++) {
    const int offset = i * 32 + lane_id;
    if constexpr (use_256b) {
      st256_cs(reinterpret_cast<u32x8_t*>(nope_dst) + offset,
               ld256_cs(reinterpret_cast<const u32x8_t*>(nope_src) + offset));
    } else {
      st128_cs(reinterpret_cast<int4*>(nope_dst) + offset,
               ld128_cs(reinterpret_cast<const int4*>(nope_src) + offset));
    }
  }

  const int* rope_src = reinterpret_cast<const int*>(
      q_pe + token_id * pe_stride_0 + head_id * pe_stride_1);
  int* rope_dst = reinterpret_cast<int*>(q_out + token_id * out_stride_0 +
                                         head_id * out_stride_1 + NOPE_DIM);

  st32_cs(rope_dst + lane_id, ld32_cs(rope_src + lane_id));
}

}  // namespace vllm

#endif  // CONCAT_MLA_Q_CUH_
