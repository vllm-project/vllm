#pragma once

// NVFP4 SiLU+Mul quantization kernel for expert-parallel workloads
// Ported from tlrmchlsmth/flashinfer:nvfp4-silu-mul-quant-opt
// Original: csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh
//
// BF16 input -> SiLU+Mul -> e2m1 (4-bit) output with FP8 scale factors
// Uses the fused cvt_silu_mul_fp16_to_fp4 path to avoid intermediate bf16

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nvfp4_utils.cuh"

namespace nvfp4 {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) &&              \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9) ||     \
     (__CUDACC_VER_MAJOR__ >= 13))
constexpr int CVT_FP16_TO_FP4_ELTS_PER_THREAD = 16;
#else
constexpr int CVT_FP16_TO_FP4_ELTS_PER_THREAD = 8;
#endif

constexpr int CVT_FP4_SF_VEC_SIZE = 16;

template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(128, 8)
#endif
    cvt_fp16_to_fp4_expert(int32_t numRows, int32_t numCols, Type const* in,
                           float const* SFScale, uint32_t* out, uint32_t* SFout,
                           int32_t* mask, int n_experts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVecT = PackedVec<Type, CVT_FP16_TO_FP4_ELTS_PER_THREAD>;
  using PackedFp4OutT =
      std::conditional_t<CVT_FP16_TO_FP4_ELTS_PER_THREAD == 16, uint64_t,
                         uint32_t>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP16_TO_FP4_ELTS_PER_THREAD);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;
  int remainder = (gridDim.x * blockDim.x) % n_experts;
  int expert_idx;
  int tid_in_expert;
  int actual_stride;
  if (remainder > 0) {
    int bound = remainder * (stride + 1);
    if (tid < bound) {
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  int m = numRows / n_experts;
  int padded_m = (m + (128 - 1)) / 128 * 128;

  int colsPerRow = numCols / CVT_FP16_TO_FP4_ELTS_PER_THREAD;
  bool use_mask = mask != nullptr;
  // silu_and_mul: input has 2x columns (gate + up)
  int actualColsPerRow = colsPerRow * 2;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];
  constexpr int factor = CVT_FP4_SF_VEC_SIZE * 4;
  int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
  int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
  uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;
  int mask_limit = use_mask ? mask[expert_idx] : m;

  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow;
       globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx - rowIdx * colsPerRow;
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    if (rowIdx_in_expert >= mask_limit) {
      break;
    }

    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    int64_t outOffset = rowIdx * colsPerRow + colIdx;

    auto sf_out =
        get_sf_out_offset(rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    PackedVecT gate_vec, up_vec;
    loadPackedVec(gate_vec, reinterpret_cast<PackedVecT const*>(in) + inOffset);
    loadPackedVec(up_vec, reinterpret_cast<PackedVecT const*>(in) + inOffset +
                              colsPerRow);
    reinterpret_cast<PackedFp4OutT*>(out)[outOffset] =
        cvt_silu_mul_fp16_to_fp4<Type, CVT_FP4_SF_VEC_SIZE,
                                 CVT_FP16_TO_FP4_ELTS_PER_THREAD, UE8M0_SF>(
            gate_vec, up_vec, SFScaleVal, sf_out);
  }
#endif
}

}  // namespace nvfp4
