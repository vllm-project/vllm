/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define CHECK_TYPE(x, st)                                              \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);    \
  CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL

  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
// On ROCm versions before 7.0, __syncwarp isn't defined. The below
// implementation is copy/pasted from the implementation in ROCm 7.0
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
  #endif
#else
  #define FINAL_MASK 0xffffffff
#endif

namespace tensorrt_llm::common {
template <typename T, int num>
struct packed_as;
// Specialization for packed_as used in this kernel.
template <>
struct packed_as<uint, 1> {
  using type = uint;
};

template <>
struct packed_as<uint, 2> {
  using type = uint2;
};

template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace tensorrt_llm::common

namespace tensorrt_llm::kernels {

// cp.async helpers: async copy from global to shared memory (SM80+).
// On older architectures these fall back to synchronous loads.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
__device__ __forceinline__ void rope_cp_async4(void* smem_ptr,
                                               const void* glob_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(glob_ptr));
}
__device__ __forceinline__ void rope_cp_async_ca(void* smem_ptr,
                                                 const void* glob_ptr,
                                                 int size_bytes) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if (size_bytes == 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else if (size_bytes == 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  }
}
__device__ __forceinline__ void rope_cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int n>
__device__ __forceinline__ void rope_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(n));
}
#else
__device__ __forceinline__ void rope_cp_async4(void* smem_ptr,
                                               const void* glob_ptr) {
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
}
__device__ __forceinline__ void rope_cp_async_ca(void* smem_ptr,
                                                 const void* glob_ptr,
                                                 int size_bytes) {
  if (size_bytes == 4) {
    *reinterpret_cast<uint32_t*>(smem_ptr) =
        *reinterpret_cast<const uint32_t*>(glob_ptr);
  } else if (size_bytes == 8) {
    *reinterpret_cast<uint64_t*>(smem_ptr) =
        *reinterpret_cast<const uint64_t*>(glob_ptr);
  } else {
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(glob_ptr);
  }
}
__device__ __forceinline__ void rope_cp_async_fence() {}
template <int n>
__device__ __forceinline__ void rope_cp_async_wait() {}
#endif

// NOTE(zhuhaoran): This kernel is adapted from TensorRT-LLM implementation,
// with added support for passing the cos_sin_cache as an input.
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu

// Perform per-head QK Norm and RoPE in a single kernel.
// scalar_t_in: data type of QKV and RMSNorm weights
// scalar_t_cache: data type of cos/sin cache
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void fusedQKNormRopeKernel(
    void* qkv_void,                  // Combined QKV tensor
    int const num_heads_q,           // Number of query heads
    int const num_heads_k,           // Number of key heads
    int const num_heads_v,           // Number of value heads
    float const eps,                 // Epsilon for RMS normalization
    void const* q_weight_void,       // RMSNorm weights for query
    void const* k_weight_void,       // RMSNorm weights for key
    void const* cos_sin_cache_void,  // Pre-computed cos/sin cache
    int64_t const* position_ids,     // Position IDs for RoPE
    int const num_tokens,            // Number of tokens
    int const rotary_dim             // Dimension for RoPE
) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input QKV data type is not supported for this CUDA "
                  "architecture or toolkit version.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache data type is not supported for this CUDA architecture "
                  "or toolkit version.");
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp
    // processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;

    // Determine which token and head type (Q or K) this warp processes
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens) return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64 (each warp processes one "
                  "head, and each thread gets even number of "
                  "elements)");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0,
                  "numSizeBytes must be a multiple of 4");
    constexpr int vecSize =
        elemSizeBytes /
        4;  // Use packed_as<uint, vecSize> to perform loading/saving.
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    int offsetWarp;  // Offset for the warp
    if (isQ) {
      // Q segment: token offset + head offset within Q segment
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      // K segment: token offset + entire Q segment + head offset within K
      // segment
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                   headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
    {
      vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        // Interpret the generic vector chunk as the specific packed type
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        // Convert to float2 for computation
        float2 vals = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x;
        sumOfSquares += vals.y * vals.y;

        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    // Reduce sum across warp using the utility function
    sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);

    // Compute RMS normalization factor
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Normalize elements
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim = laneId * numElemsPerThread + i;
      float weight = isQ ? Converter::convert(q_weight[dim])
                         : Converter::convert(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

    // Apply RoPE to normalized elements
    float elements2[numElemsPerThread];  // Additional buffer required for RoPE.

    int64_t pos_id = position_ids[tokenIdx];

    // Calculate cache pointer for this position - similar to
    // pos_encoding_kernels.cu
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / numElemsPerThread;  // rotary range
    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // Perform interleaving. Use pre-computed cos/sin values.
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          // Global dimension index in the head
          int const dim_idx = laneId * numElemsPerThread + idx0;

          float const val0 = elements[idx0];
          float const val1 = elements[idx1];

          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
        // Get the data from the other half of the warp. Use pre-computed
        // cos/sin values.
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);

          if (laneId < pairOffset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = laneId * numElemsPerThread + i;

          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          float cos_val = CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float sin_val = CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
      }
    }
    // Store.
    {
      vec_T vec;
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        // Convert from float2 back to the specific packed type
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        // Place it into the generic vector
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// Multi-token-head kernel: one warp processes HEADS_PER_WARP token-heads for
// the same token, sharing cos/sin from shared memory via cp.async.
// When HEADS_PER_WARP > 1 the warp reuses the loaded cos/sin across all heads,
// hiding global-memory latency and improving occupancy for large batches.
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave, int HEADS_PER_WARP>
__global__ void fusedQKNormRopeKernelNTokenHeads(
    void* qkv_void, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, float const eps, void const* q_weight_void,
    void const* k_weight_void, void const* cos_sin_cache_void,
    int64_t const* position_ids, int const num_tokens, int const rotary_dim) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input QKV data type is not supported for this CUDA "
                  "architecture or toolkit version.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache data type is not supported for this CUDA architecture "
                  "or toolkit version.");
    using T_cache = typename CacheConverter::hip_type;

    extern __shared__ char smem_storage[];
    // Shared memory layout:
    //   [0, cos_sin_bytes)           : cos/sin for each warp  (warpsPerBlock *
    //   rotary_dim * sizeof(T_cache)) [cos_sin_bytes, ...)         : QKV tiles
    //   per warp     (warpsPerBlock * HEADS_PER_WARP * 32 * elemSizeBytes)
    T_cache* const smem = reinterpret_cast<T_cache*>(smem_storage);

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const total_qk_heads = num_heads_q + num_heads_k;
    int const num_heads = num_heads_q + num_heads_k + num_heads_v;
    int const head_chunks_per_token =
        (total_qk_heads + HEADS_PER_WARP - 1) / HEADS_PER_WARP;

    int const warp_global = blockIdx.x * warpsPerBlock + warpId;
    int const tokenIdx = warp_global / head_chunks_per_token;
    int const headChunk = warp_global % head_chunks_per_token;
    int const first_head = headChunk * HEADS_PER_WARP;
    int const num_heads_this_warp =
        (first_head + HEADS_PER_WARP <= total_qk_heads)
            ? HEADS_PER_WARP
            : (total_qk_heads - first_head);

    if (tokenIdx >= num_tokens) return;

    static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
    constexpr int numElemsPerThread = head_dim / 32;
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0,
                  "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    int const cos_sin_bytes =
        warpsPerBlock * rotary_dim * static_cast<int>(sizeof(T_cache));
    int const qkv_tile_bytes = 32 * elemSizeBytes;
    char* const this_warp_head_smem =
        smem_storage + cos_sin_bytes +
        warpId * (HEADS_PER_WARP * qkv_tile_bytes);

    // === Group 0: async load all heads' QKV into smem (issued first). ===
    for (int k = 0; k < num_heads_this_warp; ++k) {
      int const localHeadIdx = first_head + k;
      bool const isQ = localHeadIdx < num_heads_q;
      int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
      int offWarp;
      if (isQ) {
        offWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
      } else {
        offWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                  headIdx * head_dim;
      }
      int const offThread = offWarp + laneId * numElemsPerThread;
      char* smem_dst =
          this_warp_head_smem + k * qkv_tile_bytes + laneId * elemSizeBytes;
      rope_cp_async_ca(smem_dst, reinterpret_cast<const char*>(&qkv[offThread]),
                       elemSizeBytes);
    }
    rope_cp_async_fence();  // commit group 0 (QKV)

    // === Group 1: async load cos/sin into smem (issued second). ===
    int64_t const pos_id = position_ids[tokenIdx];
    T_cache const* const cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const copy_bytes = rotary_dim * static_cast<int>(sizeof(T_cache));
    int const num_copies = (copy_bytes + 15) / 16;
    for (int copyId = laneId; copyId < num_copies; copyId += 32) {
      char* smem_ptr =
          reinterpret_cast<char*>(&smem[warpId * rotary_dim]) + copyId * 16;
      const char* glob_ptr =
          reinterpret_cast<const char*>(cache_ptr) + copyId * 16;
      rope_cp_async4(smem_ptr, glob_ptr);
    }
    rope_cp_async_fence();  // commit group 1 (cos/sin)

    // wait<1>: allow at most 1 pending group (group 1) → group 0 (QKV) is done.
    rope_cp_async_wait<1>();

    float elements[numElemsPerThread];
    float elements2[numElemsPerThread];
    int const rotary_lanes = rotary_dim / numElemsPerThread;
    int const embed_dim = rotary_dim / 2;
    T_cache const* const cos_smem = &smem[warpId * rotary_dim];
    T_cache const* const sin_smem = &smem[warpId * rotary_dim + embed_dim];

    // Preload weights into registers once, reused across all heads.
    float q_w[numElemsPerThread];
    float k_w[numElemsPerThread];
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int const dim = laneId * numElemsPerThread + i;
      q_w[i] = Converter::convert(q_weight[dim]);
      k_w[i] = Converter::convert(k_weight[dim]);
    }

    for (int k = 0; k < num_heads_this_warp; ++k) {
      int const localHeadIdx = first_head + k;
      bool const isQ = localHeadIdx < num_heads_q;
      int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

      int offsetWarp;
      if (isQ) {
        offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
      } else {
        offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                     headIdx * head_dim;
      }
      int const offsetThread = offsetWarp + laneId * numElemsPerThread;

      // === Part 1: QK Norm (read from smem; group 0 already done). ===
      float sumOfSquares = 0.0f;
      {
        char const* smem_src =
            this_warp_head_smem + k * qkv_tile_bytes + laneId * elemSizeBytes;
        vec_T vec = *reinterpret_cast<vec_T const*>(smem_src);
        constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
        for (int i = 0; i < num_packed_elems; i++) {
          T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
          float2 vals = Converter::convert(packed_val);
          sumOfSquares += vals.x * vals.x;
          sumOfSquares += vals.y * vals.y;
          elements[2 * i] = vals.x;
          elements[2 * i + 1] = vals.y;
        }
      }

      sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);
      float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        elements[i] *= rms_rcp * (isQ ? q_w[i] : k_w[i]);
      }

      // On first head: wait for group 1 (cos/sin) before RoPE.
      if (k == 0) rope_cp_async_wait<0>();

      // === Part 2: RoPE using cos/sin from shared memory. ===
      if (laneId < rotary_lanes) {
        if constexpr (interleave) {
#pragma unroll
          for (int i = 0; i < numElemsPerThread / 2; ++i) {
            int const idx0 = 2 * i;
            int const idx1 = 2 * i + 1;
            int const dim_idx = laneId * numElemsPerThread + idx0;
            float const val0 = elements[idx0];
            float const val1 = elements[idx1];
            int const half_dim = dim_idx / 2;
            float const cos_val = CacheConverter::convert(cos_smem[half_dim]);
            float const sin_val = CacheConverter::convert(sin_smem[half_dim]);
            elements[idx0] = val0 * cos_val - val1 * sin_val;
            elements[idx1] = val0 * sin_val + val1 * cos_val;
          }
        } else {
          __syncwarp();
          int const pairOffset = (rotary_dim / 2) / numElemsPerThread;
#pragma unroll
          for (int i = 0; i < numElemsPerThread; i++) {
            elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);
            if (laneId < pairOffset) elements2[i] = -elements2[i];
            int dim_idx = laneId * numElemsPerThread + i;
            dim_idx = (dim_idx * 2) % rotary_dim;
            int const half_dim = dim_idx / 2;
            float const cos_val = CacheConverter::convert(cos_smem[half_dim]);
            float const sin_val = CacheConverter::convert(sin_smem[half_dim]);
            elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
          }
          __syncwarp();
        }
      }

      // Store.
      {
        vec_T vec;
        constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
        for (int i = 0; i < num_packed_elems; i++) {
          T2_in packed_val = Converter::convert(
              make_float2(elements[2 * i], elements[2 * i + 1]));
          *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
        }
        *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
      }
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// Borrowed from
// https://github.com/flashinfer-ai/flashinfer/blob/8125d079a43e9a0ba463a4ed1b639cefd084cec9/include/flashinfer/pos_enc.cuh#L568
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormRope(void* qkv, int const num_tokens,
                           int const num_heads_q, int const num_heads_k,
                           int const num_heads_v, int const head_dim,
                           int const rotary_dim, float const eps,
                           void const* q_weight, void const* k_weight,
                           void const* cos_sin_cache, bool const interleave,
                           int64_t const* position_ids, cudaStream_t stream) {
  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;
  int const gridSize = common::divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);
  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head dimension for fusedQKNormRope: ", head_dim);
  }
}

// Launch: one warp processes token_heads_per_warp token-heads (1, 2, 4, or 8).
// When token_heads_per_warp == 1, delegates to the 1-head baseline above.
template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormRopeNTokenHeads(
    void* qkv, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim,
    int const rotary_dim, float const eps, void const* q_weight,
    void const* k_weight, void const* cos_sin_cache, bool const interleave,
    int64_t const* position_ids, int const token_heads_per_warp,
    cudaStream_t stream) {
  TORCH_CHECK(token_heads_per_warp == 1 || token_heads_per_warp == 2 ||
                  token_heads_per_warp == 4 || token_heads_per_warp == 8,
              "token_heads_per_warp must be 1, 2, 4, or 8, got ",
              token_heads_per_warp);

  // token_heads_per_warp == 1: delegate to the 1-head baseline kernel.
  if (token_heads_per_warp == 1) {
    launchFusedQKNormRope<scalar_t_in, scalar_t_cache>(
        qkv, num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim,
        rotary_dim, eps, q_weight, k_weight, cos_sin_cache, interleave,
        position_ids, stream);
    return;
  }

  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  // Grid: one warp per (token, head_chunk); same token → reuse cos/sin in smem.
  int const head_chunks_per_token =
      (totalQKHeads + token_heads_per_warp - 1) / token_heads_per_warp;
  int const total_warps = num_tokens * head_chunks_per_token;
  int const gridSize = common::divUp(total_warps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);
  // Cache element size: float=4, bfloat16=2 (host-safe; kernel uses same
  // layout).
  size_t const cache_elem_size =
      std::is_same_v<scalar_t_cache, float> ? sizeof(float) : 2u;
  // QKV smem: token_heads_per_warp tiles per warp, each tile 32*(head_dim/32*2)
  // = 2*head_dim bytes.
  size_t const qkv_smem_per_warp = static_cast<size_t>(token_heads_per_warp) *
                                   2u * static_cast<size_t>(head_dim);
  size_t const smem_bytes =
      warpsPerBlock * static_cast<size_t>(rotary_dim) * cache_elem_size +
      warpsPerBlock * qkv_smem_per_warp;

#define LAUNCH_N_TOKEN_HEADS(N)                                              \
  do {                                                                       \
    switch (head_dim) {                                                      \
      case 64:                                                               \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                        \
          fusedQKNormRopeKernelNTokenHeads<scalar_t_in, scalar_t_cache, 64,  \
                                           INTERLEAVE, (N)>                  \
              <<<gridDim, blockDim, smem_bytes, stream>>>(                   \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight, \
                  k_weight, cos_sin_cache, position_ids, num_tokens,         \
                  rotary_dim);                                               \
        });                                                                  \
        break;                                                               \
      case 128:                                                              \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                        \
          fusedQKNormRopeKernelNTokenHeads<scalar_t_in, scalar_t_cache, 128, \
                                           INTERLEAVE, (N)>                  \
              <<<gridDim, blockDim, smem_bytes, stream>>>(                   \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight, \
                  k_weight, cos_sin_cache, position_ids, num_tokens,         \
                  rotary_dim);                                               \
        });                                                                  \
        break;                                                               \
      case 256:                                                              \
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                        \
          fusedQKNormRopeKernelNTokenHeads<scalar_t_in, scalar_t_cache, 256, \
                                           INTERLEAVE, (N)>                  \
              <<<gridDim, blockDim, smem_bytes, stream>>>(                   \
                  qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight, \
                  k_weight, cos_sin_cache, position_ids, num_tokens,         \
                  rotary_dim);                                               \
        });                                                                  \
        break;                                                               \
      default:                                                               \
        TORCH_CHECK(false, "Unsupported head dimension: ", head_dim);        \
    }                                                                        \
  } while (0)

  if (token_heads_per_warp == 2) {
    LAUNCH_N_TOKEN_HEADS(2);
  } else if (token_heads_per_warp == 4) {
    LAUNCH_N_TOKEN_HEADS(4);
  } else if (token_heads_per_warp == 8) {
    LAUNCH_N_TOKEN_HEADS(8);
  }
#undef LAUNCH_N_TOKEN_HEADS
}
}  // namespace tensorrt_llm::kernels

void fused_qk_norm_rope(
    torch::Tensor& qkv,       // Combined QKV tensor [num_tokens,
                              // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,      // Number of query heads
    int64_t num_heads_k,      // Number of key heads
    int64_t num_heads_v,      // Number of value heads
    int64_t head_dim,         // Dimension per head
    double eps,               // Epsilon for RMS normalization
    torch::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    torch::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    torch::Tensor& cos_sin_cache,  // Cos/sin cache [max_position, head_dim]
    bool is_neox,                  // Whether RoPE is applied in Neox style
    torch::Tensor& position_ids,   // Position IDs for RoPE [num_tokens]
    int64_t forced_token_heads_per_warp  // -1 = auto-select, >0 = forced value
) {
  // Input validation
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(position_ids, torch::kInt64);

  TORCH_CHECK(qkv.dim() == 2,
              "QKV tensor must be 2D: [num_tokens, "
              "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "Cos/sin cache must be 2D: [max_position, head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim,
              "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim,
              "Key weights size must match head dimension");

  TORCH_CHECK(cos_sin_cache.size(1) % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(cos_sin_cache.size(1) <= head_dim,
              "rotary_dim must be less than or equal to head_dim");

  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type(),
              "qkv, q_weight and k_weight must have the same dtype");

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  // Select token_heads_per_warp: forced value if >0, else auto-select.
  // Auto thresholds are calibrated on SM 9.0 (H100). On other architectures,
  // fall back to token_heads_per_warp=1 (base kernel) until profiled.
  int token_heads_per_warp;
  if (forced_token_heads_per_warp > 0) {
    token_heads_per_warp = static_cast<int>(forced_token_heads_per_warp);
  } else {
    token_heads_per_warp = 1;
    auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    int sm_version = dev_prop->major * 10 + dev_prop->minor;
    int64_t total_qk_units = num_tokens * (num_heads_q + num_heads_k);
    if (sm_version == 90) {
      if (head_dim >= 256) {
        // head_dim=256: NTokenHeads uses 56 regs/thread vs 36 for base kernel,
        // limiting occupancy to ~12% vs ~43% at small token counts (100 tokens,
        // H100, num_heads_q=32, num_heads_k=4). Thresholds are 2x looser than
        // head_dim=128; hpw capped at 4 (hpw=8 offers no benefit over hpw=4 at
        // head_dim=256 and is much worse at small token counts).
        // (benchmark: bench_headdim256.py, H100, num_heads_q=32, num_heads_k=4)
        // hpw=2 wins at ~100 tokens (total_qk_units ~3600), hpw=4 wins at ~200
        // tokens (~7200). Use 4096/8192 as round thresholds.
        if (total_qk_units < 4096LL) {
          token_heads_per_warp = 1;
        } else if (total_qk_units < 8192LL) {
          token_heads_per_warp = 2;
        } else {
          token_heads_per_warp = 4;
        }
      } else {
        // head_dim=128: calibrated by bench_token_heads_crossover.py on H100,
        // num_heads_q=32, num_heads_k=8.
        if (total_qk_units < 10240LL) {
          token_heads_per_warp = 1;
        } else if (total_qk_units < 40960LL) {
          token_heads_per_warp = 4;
        } else {
          token_heads_per_warp = 8;
        }
      }
    }
  }

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
          using cache_scalar_t = scalar_t;
          tensorrt_llm::kernels::launchFusedQKNormRopeNTokenHeads<
              qkv_scalar_t, cache_scalar_t>(
              qkv.data_ptr(), static_cast<int>(num_tokens),
              static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
              static_cast<int>(num_heads_v), static_cast<int>(head_dim),
              static_cast<int>(cos_sin_cache.size(1)), static_cast<float>(eps),
              q_weight.data_ptr(), k_weight.data_ptr(),
              cos_sin_cache.data_ptr(), !is_neox,
              reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
              token_heads_per_warp, stream);
        });
  });
}