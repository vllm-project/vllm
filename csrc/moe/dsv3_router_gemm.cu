/*
 * Adapted from SGLang's sgl-kernel implementation, which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace vllm {

// Constants for supported configurations
static constexpr int DEFAULT_NUM_EXPERTS = 256;
static constexpr int KIMI_K2_NUM_EXPERTS = 384;
static constexpr int DEFAULT_HIDDEN_DIM = 7168;

// Helper function to get SM version
inline int getSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  int sm_major = 0;
  int sm_minor = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
  return sm_major * 10 + sm_minor;
}

// Helper function to check if PDL is enabled via environment variable
inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      const char* env = std::getenv("TRTLLM_ENABLE_PDL");
      enablePDL = env && env[0] == '1' && env[1] == '\0';
    }
  });
  return enablePDL;
}

// Convert 8 bfloat16 values from a uint4 to float array
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec,
                                                     float* dst) {
  __nv_bfloat16* bf16_ptr =
      reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));

#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

// Router GEMM kernel with float32 output
template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel_float_output(
    float* out, T const* mat_a, T const* mat_b) {
  // Each block handles one expert column
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  // Initialize accumulators for all M rows
  float acc[kNumTokens] = {};

  // Shared memory for warp-level reduction
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  // B matrix is in column-major order
  T const* b_col = mat_b + n_idx * kHiddenDim;

  // Pre-compute k_base values
  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Process the GEMM in chunks
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    // Load B matrix values using vector load
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

    // Convert B values to float
    float b_float[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_float);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      uint4 a_vec = *reinterpret_cast<uint4 const*>(
          mat_a + (m_idx * kHiddenDim) + k_base);

      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);

#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  // Warp-level reduction
  int const warpId = tid / 32;
  int const laneId = tid % 32;

  float warp_result[kNumTokens];
#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    warp_result[m_idx] = acc[m_idx];
  }

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = warp_result[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  // Final reduction across warps
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }
      out[m * kNumExperts + n_idx] = final_sum;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Router GEMM kernel with bfloat16 output
template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel_bf16_output(
    __nv_bfloat16* out, T const* mat_a, T const* mat_b) {
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  T const* b_col = mat_b + n_idx * kHiddenDim;

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

    float b_float[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_float);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      uint4 a_vec = *reinterpret_cast<uint4 const*>(
          mat_a + (m_idx * kHiddenDim) + k_base);

      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);

#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  int const warpId = tid / 32;
  int const laneId = tid % 32;

  float warp_result[kNumTokens];
#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    warp_result[m_idx] = acc[m_idx];
  }

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = warp_result[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }
      out[m * kNumExperts + n_idx] = __float2bfloat16(final_sum);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Launcher functions
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,
                                 cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
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
  cudaLaunchKernelEx(
      &config,
      router_gemm_kernel_float_output<T, kBlockSize, VPT, kNumTokens,
                                      kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
}

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a,
                                T const* mat_b, cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
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
  cudaLaunchKernelEx(
      &config,
      router_gemm_kernel_bf16_output<T, kBlockSize, VPT, kNumTokens,
                                     kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
}

// Template unrollers for runtime dispatch
template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller {
  static void unroll_float_output(int num_tokens, float* output,
                                  __nv_bfloat16 const* input,
                                  __nv_bfloat16 const* weights,
                                  cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeRouterGemmFloatOutput<__nv_bfloat16, kBegin, kNumExperts,
                                  kHiddenDim>(output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts,
                   kHiddenDim>::unroll_float_output(num_tokens, output, input,
                                                    weights, stream);
    }
  }

  static void unroll_bf16_output(int num_tokens, __nv_bfloat16* output,
                                 __nv_bfloat16 const* input,
                                 __nv_bfloat16 const* weights,
                                 cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeRouterGemmBf16Output<__nv_bfloat16, kBegin, kNumExperts,
                                 kHiddenDim>(output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts,
                   kHiddenDim>::unroll_bf16_output(num_tokens, output, input,
                                                   weights, stream);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim> {
  static void unroll_float_output(int num_tokens, float* output,
                                  __nv_bfloat16 const* input,
                                  __nv_bfloat16 const* weights,
                                  cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeRouterGemmFloatOutput<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(
          output, input, weights, stream);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }

  static void unroll_bf16_output(int num_tokens, __nv_bfloat16* output,
                                 __nv_bfloat16 const* input,
                                 __nv_bfloat16 const* weights,
                                 cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeRouterGemmBf16Output<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(
          output, input, weights, stream);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }
};

// Explicit template instantiations for float output - 256 experts
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 1, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 2, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 3, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 4, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 5, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 6, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 7, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 8, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 9, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 10, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 11, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 12, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 13, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 14, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 16, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// Explicit template instantiations for float output - 384 experts
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 1, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 2, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 3, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 4, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 5, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 6, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 7, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 8, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 9, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 10, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 11, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 12, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 13, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 14, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmFloatOutput<__nv_bfloat16, 16, 384, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// Explicit template instantiations for bf16 output - 256 experts
template void invokeRouterGemmBf16Output<__nv_bfloat16, 1, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 2, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 3, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 4, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 5, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 6, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 7, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 8, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 9, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 10, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 11, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 12, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 13, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 14, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 15, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 16, 256, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// Explicit template instantiations for bf16 output - 384 experts
template void invokeRouterGemmBf16Output<__nv_bfloat16, 1, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 2, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 3, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 4, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 5, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 6, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 7, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 8, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 9, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 10, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 11, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 12, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 13, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 14, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 15, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
template void invokeRouterGemmBf16Output<__nv_bfloat16, 16, 384, 7168>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

}  // namespace vllm

// Public API
void dsv3_router_gemm(torch::Tensor& output,       // [num_tokens, num_experts]
                      const torch::Tensor& mat_a,  // [num_tokens, hidden_dim]
                      const torch::Tensor& mat_b   // [num_experts, hidden_dim]
) {
  TORCH_CHECK(output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2);

  const int num_tokens = mat_a.size(0);
  const int num_experts = mat_b.size(0);
  const int hidden_dim = mat_a.size(1);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(1),
              "mat_a and mat_b must have the same hidden_dim");
  TORCH_CHECK(hidden_dim == vllm::DEFAULT_HIDDEN_DIM,
              "Expected hidden_dim=", vllm::DEFAULT_HIDDEN_DIM,
              ", but got hidden_dim=", hidden_dim);
  TORCH_CHECK(num_experts == vllm::DEFAULT_NUM_EXPERTS ||
                  num_experts == vllm::KIMI_K2_NUM_EXPERTS,
              "Expected num_experts=", vllm::DEFAULT_NUM_EXPERTS,
              " or num_experts=", vllm::KIMI_K2_NUM_EXPERTS,
              ", but got num_experts=", num_experts);
  TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
              "num_tokens must be between 1 and 16 for router_gemm");
  TORCH_CHECK(mat_a.dtype() == torch::kBFloat16, "mat_a must be bf16");
  TORCH_CHECK(mat_b.dtype() == torch::kBFloat16, "mat_b must be bf16");
  TORCH_CHECK(
      output.dtype() == torch::kFloat32 || output.dtype() == torch::kBFloat16,
      "output must be float32 or bf16");

  const int sm = vllm::getSMVersion();
  TORCH_CHECK(sm >= 90, "dsv3_router_gemm requires SM >= 90 (Hopper or newer)");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (output.dtype() == torch::kFloat32) {
    if (num_experts == vllm::DEFAULT_NUM_EXPERTS) {
      vllm::LoopUnroller<1, 16, vllm::DEFAULT_NUM_EXPERTS,
                         vllm::DEFAULT_HIDDEN_DIM>::
          unroll_float_output(
              num_tokens, reinterpret_cast<float*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    } else if (num_experts == vllm::KIMI_K2_NUM_EXPERTS) {
      vllm::LoopUnroller<1, 16, vllm::KIMI_K2_NUM_EXPERTS,
                         vllm::DEFAULT_HIDDEN_DIM>::
          unroll_float_output(
              num_tokens, reinterpret_cast<float*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
  } else if (output.dtype() == torch::kBFloat16) {
    if (num_experts == vllm::DEFAULT_NUM_EXPERTS) {
      vllm::LoopUnroller<1, 16, vllm::DEFAULT_NUM_EXPERTS,
                         vllm::DEFAULT_HIDDEN_DIM>::
          unroll_bf16_output(
              num_tokens,
              reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    } else if (num_experts == vllm::KIMI_K2_NUM_EXPERTS) {
      vllm::LoopUnroller<1, 16, vllm::KIMI_K2_NUM_EXPERTS,
                         vllm::DEFAULT_HIDDEN_DIM>::
          unroll_bf16_output(
              num_tokens,
              reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
              reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream);
    }
  }
}
