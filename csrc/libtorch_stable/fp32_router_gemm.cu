// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM: activation(T) x weight(fp32) -> fp32, M<=32, for the
// supported (E, H) pairs listed at the bottom of this file.
// Supports bf16 or fp32 activation; weight is always fp32.
// Adapted from dsv3_router_gemm_float_out.cu.
// (E=256, H=6144) bf16 uses a B300-tuned wide-block geometry; see
// invokeFp32RouterGemm.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

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
// Each block computes kEPB expert columns; wider blocks / kEPB > 1 are
// selected per (shape, M) in invokeFp32RouterGemm (B300-tuned, see below).
// kTGroups > 1 splits the tokens across groups of kBlockSize threads within
// the block: all groups scan the same weight K-slices (group 0 misses to
// DRAM, later groups hit L1) so weight traffic stays 1x, while per-thread
// accumulator registers drop by kTGroups (at M=16 the 32 fp32 accumulators
// push the kernel to 128 regs/thread and 1 block/SM).
template <typename InputT, int kBlockSize, int kNumTokens, int kEPB,
          int kNumExperts, int kHiddenDim, int kTGroups = 1>
__global__ __launch_bounds__(
    kBlockSize* kTGroups, 1) void fp32_router_gemm_kernel(float* out,
                                                          InputT const* mat_a,
                                                          float const* mat_b) {
  constexpr int VPT = 16 / sizeof(InputT);
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;
  static_assert(kHiddenDim % k_elems_per_k_iteration == 0);
  static_assert(kNumTokens % kTGroups == 0);
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;  // per token group
  constexpr int kMG = kNumTokens / kTGroups;         // tokens per group

  int const e_base = blockIdx.x * kEPB;
  int const tid = threadIdx.x % kBlockSize;
  int const m0 = (threadIdx.x / kBlockSize) * kMG;
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

  float acc[kMG][kEPB] = {};
  __shared__ float sm_reduction[kNumTokens][kEPB][kNumWarps];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  // Fire the PDL trigger right after our own wait instead of at kernel end:
  // a gridsync-ing consumer is unaffected (its wait always targets full grid
  // completion), while a consumer that reads none of our outputs (e.g. the
  // NVFP4 activation quant, which reads the same hidden_states) can launch
  // now and fully overlap this kernel's body.
  cudaTriggerProgrammaticLaunchCompletion();
#endif

#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = ki * k_elems_per_k_iteration + tid * VPT;

    float b_float[kEPB][VPT];
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      load_weight<VPT>(mat_b + (e_base + e) * kHiddenDim + k_base, b_float[e]);
    }

#pragma unroll
    for (int m_idx = 0; m_idx < kMG; m_idx++) {
      float a_float[VPT];
      load_activation<InputT, VPT>(
          mat_a + (size_t)(m0 + m_idx) * kHiddenDim + k_base, a_float);
#pragma unroll
      for (int e = 0; e < kEPB; e++) {
#pragma unroll
        for (int k = 0; k < VPT; k++) {
          acc[m_idx][e] += a_float[k] * b_float[e][k];
        }
      }
    }
  }

  // Warp-level butterfly reduction
#pragma unroll
  for (int m = 0; m < kMG; m++) {
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float sum = acc[m][e];
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      sum += __shfl_xor_sync(0xffffffff, sum, 8);
      sum += __shfl_xor_sync(0xffffffff, sum, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      if (laneId == 0) sm_reduction[m0 + m][e][warpId] = sum;
    }
  }

  __syncthreads();

  // Parallel finalize: one thread per (m, e) output.
  for (int idx = threadIdx.x; idx < kNumTokens * kEPB;
       idx += kBlockSize * kTGroups) {
    int const m = idx / kEPB;
    int const e = idx % kEPB;
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; w++) final_sum += sm_reduction[m][e][w];
    out[m * kNumExperts + e_base + e] = final_sum;
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename InputT, int kBlockSize, int kEPB, int kNumTokens,
          int kNumExperts, int kHiddenDim, int kTGroups = 1>
static void launchFp32RouterGemm(float* output, InputT const* mat_a,
                                 float const* mat_b, cudaStream_t stream) {
  static_assert(kNumExperts % kEPB == 0);
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts / kEPB;
  config.blockDim = kBlockSize * kTGroups;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(
      &config,
      fp32_router_gemm_kernel<InputT, kBlockSize, kNumTokens, kEPB, kNumExperts,
                              kHiddenDim, kTGroups>,
      output, mat_a, mat_b);
}

static bool isBlackwellFamily() {
  static int sm = []() {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor;
  }();
  return sm >= 100;
}

template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  // Geometry tuned on B300 per supported shape, bf16 activation, under a
  // production-fidelity harness (CUDA-graph replay, per-layer cold weights).
  // GLM-5.2 (E=256, H=6144):
  //   M <= 4        : BS=768, EPB=1 (2.7us vs cast+cuBLAS 8.1us at M=1)
  //   M in [5, 15]
  //   or odd        : BS=384, EPB=2 (crossover vs BS=768 measured in (4, 8))
  //   M >= 16, even : BS=192, EPB=2, 2 token groups (M=16 4.79us vs 5.04,
  //                   M=24 5.71 vs 6.38, M=32 6.79 vs 7.72; M=12 loses at
  //                   0.97x, so the boundary is 16).
  // Only enabled on the Blackwell family where it was validated; Hopper and
  // other shapes / fp32 activation keep the legacy geometry.
  if constexpr (std::is_same_v<InputT, __nv_bfloat16> && kNumExperts == 256 &&
                kHiddenDim == 6144) {
    if (!isBlackwellFamily()) {
      launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
      return;
    }
    if constexpr (kNumTokens <= 4) {
      launchFp32RouterGemm<InputT, 768, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    } else if constexpr (kNumTokens >= 16 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 2, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, 384, 2, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    }
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       kNumExperts == 128 && kHiddenDim == 6144) {
    // MiniMax-M3. Legacy 128/1 only fills 128 blocks and pays the same
    // accumulator register cliffs; B300 sweep:
    //   even M in [6, 10] : BS=384, EPB=1, 2 token groups (1.26-1.43x)
    //   even M >= 12      : BS=192, EPB=1, 2 token groups (1.59-1.66x at
    //                       M >= 18; re-measured on B300+B200: 192 also wins
    //                       M=12/14 by 5-11%% on both, ties 384 at 16)
    //   M <= 5 / odd      : BS=384, EPB=1 (1.03-1.19x)
    if (!isBlackwellFamily()) {
      launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
      return;
    }
    if constexpr (kNumTokens >= 12 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 1, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else if constexpr (kNumTokens >= 6 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 384, 1, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, 384, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    }
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       kNumExperts == 256 && kHiddenDim == 3072) {
    // MiniMax-M2/M2.5. The 3.1MB weight is latency-floor bound at small M
    // (legacy already optimal); token groups win only at even M >= 8
    // (1.05-1.17x). EPB crossover measured between 12 and 16.
    if (!isBlackwellFamily()) {
      launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
      return;
    }
    if constexpr (kNumTokens >= 14 && kNumTokens % 2 == 0) {
      // M=14 originally measured 0.91x and stayed on legacy; two fresh
      // sweeps (B300 dev1 + B200) both put 192/2/tg2 ahead by 3.5-4%%.
      launchFp32RouterGemm<InputT, 192, 2, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else if constexpr (kNumTokens >= 8 && kNumTokens <= 12 &&
                         kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 1, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(
          output, mat_a, mat_b, stream);
    }
  } else {
    launchFp32RouterGemm<InputT, 128, 1, kNumTokens, kNumExperts, kHiddenDim>(
        output, mat_a, mat_b, stream);
  }
}

// ---------------------------------------------------------------------------
// Explicit instantiations: M=1..32, for both input types, for the supported
// (E, H) pairs:  (256, 3072) [MiniMax-M2/M2.5],  (128, 6144) [MiniMax-M3]
// and  (256, 6144) [GLM-5.2].
// ---------------------------------------------------------------------------

#define INSTANTIATE(T, M, E, H)                                    \
  template void invokeFp32RouterGemm<T, M, E, H>(float*, T const*, \
                                                 float const*, cudaStream_t);

#define INSTANTIATE_ALL(T, E, H) \
  INSTANTIATE(T, 1, E, H)        \
  INSTANTIATE(T, 2, E, H)        \
  INSTANTIATE(T, 3, E, H)        \
  INSTANTIATE(T, 4, E, H)        \
  INSTANTIATE(T, 5, E, H)        \
  INSTANTIATE(T, 6, E, H)        \
  INSTANTIATE(T, 7, E, H)        \
  INSTANTIATE(T, 8, E, H)        \
  INSTANTIATE(T, 9, E, H)        \
  INSTANTIATE(T, 10, E, H)       \
  INSTANTIATE(T, 11, E, H)       \
  INSTANTIATE(T, 12, E, H)       \
  INSTANTIATE(T, 13, E, H)       \
  INSTANTIATE(T, 14, E, H)       \
  INSTANTIATE(T, 15, E, H)       \
  INSTANTIATE(T, 16, E, H)       \
  INSTANTIATE(T, 17, E, H)       \
  INSTANTIATE(T, 18, E, H)       \
  INSTANTIATE(T, 19, E, H)       \
  INSTANTIATE(T, 20, E, H)       \
  INSTANTIATE(T, 21, E, H)       \
  INSTANTIATE(T, 22, E, H)       \
  INSTANTIATE(T, 23, E, H)       \
  INSTANTIATE(T, 24, E, H)       \
  INSTANTIATE(T, 25, E, H)       \
  INSTANTIATE(T, 26, E, H)       \
  INSTANTIATE(T, 27, E, H)       \
  INSTANTIATE(T, 28, E, H)       \
  INSTANTIATE(T, 29, E, H)       \
  INSTANTIATE(T, 30, E, H)       \
  INSTANTIATE(T, 31, E, H)       \
  INSTANTIATE(T, 32, E, H)

INSTANTIATE_ALL(float, 256, 3072)
INSTANTIATE_ALL(__nv_bfloat16, 256, 3072)
INSTANTIATE_ALL(float, 128, 6144)
INSTANTIATE_ALL(__nv_bfloat16, 128, 6144)
INSTANTIATE_ALL(float, 256, 6144)
INSTANTIATE_ALL(__nv_bfloat16, 256, 6144)

#undef INSTANTIATE_ALL
#undef INSTANTIATE
