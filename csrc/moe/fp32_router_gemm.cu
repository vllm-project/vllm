// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM: activation(T) x weight(fp32) -> fp32, H=3072, E=256, M<=32.
// Supports bf16 or fp32 activation; weight is always fp32.
// Adapted from dsv3_router_gemm_float_out.cu.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsv3_router_gemm_utils.h"

// ---------------------------------------------------------------------------
// Load helpers
// ---------------------------------------------------------------------------

// Load VPT fp32 values from the weight matrix (always fp32).
//   VPT=4 when activation is fp32 (one float4 load)
//   VPT=8 when activation is bf16 (two float4 loads)
template <int VPT>
__device__ __forceinline__ void load_weight(float const* ptr, float* dst);

template <>
__device__ __forceinline__ void load_weight<4>(float const* ptr, float* dst) {
  float4 v = *reinterpret_cast<float4 const*>(ptr);
  dst[0] = v.x;
  dst[1] = v.y;
  dst[2] = v.z;
  dst[3] = v.w;
}

template <>
__device__ __forceinline__ void load_weight<8>(float const* ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = v0.x;
  dst[1] = v0.y;
  dst[2] = v0.z;
  dst[3] = v0.w;
  dst[4] = v1.x;
  dst[5] = v1.y;
  dst[6] = v1.z;
  dst[7] = v1.w;
}

// Load VPT activation values and convert to fp32.
template <typename T, int VPT>
__device__ __forceinline__ void load_activation(T const* ptr, float* dst);

// fp32 activation: one float4 load, no conversion needed.
template <>
__device__ __forceinline__ void load_activation<float, 4>(float const* ptr,
                                                          float* dst) {
  float4 v = *reinterpret_cast<float4 const*>(ptr);
  dst[0] = v.x;
  dst[1] = v.y;
  dst[2] = v.z;
  dst[3] = v.w;
}

// bf16 activation: one uint4 load (8 × bf16) + element-wise conversion.
template <>
__device__ __forceinline__ void load_activation<__nv_bfloat16, 8>(
    __nv_bfloat16 const* ptr, float* dst) {
  uint4 v = *reinterpret_cast<uint4 const*>(ptr);
  __nv_bfloat16 const* bf16_ptr = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
  for (int i = 0; i < 8; i++) dst[i] = __bfloat162float(bf16_ptr[i]);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// InputT : type of activation (float or __nv_bfloat16)
// Weight is always fp32; output is always fp32.
// VPT = 16 / sizeof(InputT):  4 for fp32, 8 for bf16
template <typename InputT, int kBlockSize, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void fp32_router_gemm_kernel(
    float* out, InputT const* mat_a, float const* mat_b) {
  constexpr int VPT = 16 / sizeof(InputT);
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
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

    float b_float[VPT];
    load_weight<VPT>(b_col + k_base, b_float);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      float a_float[VPT];
      load_activation<InputT, VPT>(mat_a + m_idx * kHiddenDim + k_base,
                                   a_float);
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
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

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  constexpr int kBlockSize = 128;
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config,
                     fp32_router_gemm_kernel<InputT, kBlockSize, kNumTokens,
                                             kNumExperts, kHiddenDim>,
                     output, mat_a, mat_b);
}

// ---------------------------------------------------------------------------
// Explicit instantiations: M=1..32, E=256, H=3072, for both input types
// ---------------------------------------------------------------------------

#define INSTANTIATE(T, M)                              \
  template void invokeFp32RouterGemm<T, M, 256, 3072>( \
      float*, T const*, float const*, cudaStream_t);

#define INSTANTIATE_ALL(T)                                                     \
  INSTANTIATE(T, 1)                                                            \
  INSTANTIATE(T, 2)                                                            \
  INSTANTIATE(T, 3) INSTANTIATE(T, 4) INSTANTIATE(T, 5) INSTANTIATE(T, 6)      \
      INSTANTIATE(T, 7) INSTANTIATE(T, 8) INSTANTIATE(T, 9) INSTANTIATE(T, 10) \
          INSTANTIATE(T, 11) INSTANTIATE(T, 12) INSTANTIATE(T, 13)             \
              INSTANTIATE(T, 14) INSTANTIATE(T, 15) INSTANTIATE(T, 16)         \
                  INSTANTIATE(T, 17) INSTANTIATE(T, 18) INSTANTIATE(T, 19)     \
                      INSTANTIATE(T, 20) INSTANTIATE(T, 21) INSTANTIATE(T, 22) \
                          INSTANTIATE(T, 23) INSTANTIATE(T, 24)                \
                              INSTANTIATE(T, 25) INSTANTIATE(T, 26)            \
                                  INSTANTIATE(T, 27) INSTANTIATE(T, 28)        \
                                      INSTANTIATE(T, 29) INSTANTIATE(T, 30)    \
                                          INSTANTIATE(T, 31)                   \
                                              INSTANTIATE(T, 32)

INSTANTIATE_ALL(float)
INSTANTIATE_ALL(__nv_bfloat16)

#undef INSTANTIATE_ALL
#undef INSTANTIATE
