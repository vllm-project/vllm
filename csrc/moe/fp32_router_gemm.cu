// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// FP32 x FP32 -> FP32 router GEMM for H=3072, E=256, M<=32.
// Adapted from dsv3_router_gemm_float_out.cu.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "dsv3_router_gemm_utils.h"

// VPT=4: uint4 = 128 bits = 4 x float32
template <int kBlockSize, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ __launch_bounds__(128, 1) void fp32_router_gemm_kernel(
    float* out, float const* mat_a, float const* mat_b) {
  constexpr int VPT = 4;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;  // 512
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;  // 6
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;  // 4

  int const n_idx  = blockIdx.x;
  int const tid    = threadIdx.x;
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  float const* b_col = mat_b + n_idx * kHiddenDim;

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;");
#endif

  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    float4 b_vec = *reinterpret_cast<float4 const*>(b_col + k_base);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      float4 a_vec = *reinterpret_cast<float4 const*>(
          mat_a + m_idx * kHiddenDim + k_base);
      acc[m_idx] += a_vec.x * b_vec.x + a_vec.y * b_vec.y
                 + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
    }
  }

  // Warp-level butterfly reduction
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (laneId == 0) sm_reduction[m][warpId] = sum;
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) final_sum += sm_reduction[m][w];
      out[m * kNumExperts + n_idx] = final_sum;
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, float const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  constexpr int kBlockSize = 128;
  cudaLaunchConfig_t config;
  config.gridDim          = kNumExperts;
  config.blockDim         = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream           = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs    = attrs;
  cudaLaunchKernelEx(
      &config,
      fp32_router_gemm_kernel<kBlockSize, kNumTokens, kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
}

// Explicit instantiations: M=1..32, E=256, H=3072
#define INSTANTIATE(M) \
  template void invokeFp32RouterGemm<M, 256, 3072>( \
      float*, float const*, float const*, cudaStream_t);

INSTANTIATE(1)  INSTANTIATE(2)  INSTANTIATE(3)  INSTANTIATE(4)
INSTANTIATE(5)  INSTANTIATE(6)  INSTANTIATE(7)  INSTANTIATE(8)
INSTANTIATE(9)  INSTANTIATE(10) INSTANTIATE(11) INSTANTIATE(12)
INSTANTIATE(13) INSTANTIATE(14) INSTANTIATE(15) INSTANTIATE(16)
INSTANTIATE(17) INSTANTIATE(18) INSTANTIATE(19) INSTANTIATE(20)
INSTANTIATE(21) INSTANTIATE(22) INSTANTIATE(23) INSTANTIATE(24)
INSTANTIATE(25) INSTANTIATE(26) INSTANTIATE(27) INSTANTIATE(28)
INSTANTIATE(29) INSTANTIATE(30) INSTANTIATE(31) INSTANTIATE(32)

#undef INSTANTIATE
