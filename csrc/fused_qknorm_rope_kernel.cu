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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#define CHECK_TYPE(x, st)                                              \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st) \
  CHECK_TH_CUDA(x);        \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)

#define FINAL_MASK 0xffffffff

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
    val += __shfl_xor_sync(FINAL_MASK, val, mask,
                           32);  //__shfl_sync bf16 return float when sm < 80
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace tensorrt_llm::common

namespace tensorrt_llm::kernels {
// NOTE(zhuhaoran): This kernel is adapted from TensorRT-LLM implementation,
// with added support for passing the cos_sin_cache as an input.
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu

// Perform per-head QK Norm and RoPE in a single kernel.
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,     // Combined QKV tensor [num_tokens,
                            // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const num_heads_q,  // Number of query heads
    int const num_heads_k,  // Number of key heads
    int const num_heads_v,  // Number of value heads
    float const eps,        // Epsilon for RMS normalization
    __nv_bfloat16 const* q_weight,       // RMSNorm weights for query
    __nv_bfloat16 const* k_weight,       // RMSNorm weights for key
    __nv_bfloat16 const* cos_sin_cache,  // Pre-computed cos/sin cache
    int64_t const* position_ids,         // Position IDs for RoPE
    int const num_tokens                 // Number of tokens
) {
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
  static_assert(elemSizeBytes % 4 == 0, "numSizeBytes must be a multiple of 4");
  constexpr int vecSize =
      elemSizeBytes /
      4;  // Use packed_as<uint, vecSize> to perform loading/saving.
  using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

  int offsetWarp;  // Offset for the warp
  if (isQ) {
    // Q segment: token offset + head offset within Q segment
    offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
  } else {
    // K segment: token offset + entire Q segment + head offset within K segment
    offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                 headIdx * head_dim;
  }
  int offsetThread = offsetWarp + laneId * numElemsPerThread;

  // Sum of squares for RMSNorm
  float sumOfSquares = 0.0f;

  // Load.
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(
          reinterpret_cast<uint*>(&vec) + i));
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
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    float weight =
        isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }

  // Apply RoPE to normalized elements
  float elements2[numElemsPerThread];  // Additional buffer required for RoPE.

  int64_t pos_id = position_ids[tokenIdx];

  // Calculate cache pointer for this position - similar to
  // pos_encoding_kernels.cu
  __nv_bfloat16 const* cache_ptr = cos_sin_cache + pos_id * head_dim;
  int const embed_dim = head_dim / 2;
  __nv_bfloat16 const* cos_ptr = cache_ptr;
  __nv_bfloat16 const* sin_ptr = cache_ptr + embed_dim;

  if constexpr (interleave) {
    // Perform interleaving. Use pre-computed cos/sin values.
    for (int i = 0; i < numElemsPerThread / 2; ++i) {
      int const idx0 = 2 * i;
      int const idx1 = 2 * i + 1;

      float const val0 = elements[idx0];
      float const val1 = elements[idx1];

      int const dim_idx = laneId * numElemsPerThread + idx0;
      int const half_dim = dim_idx / 2;
      float const cos_val = __bfloat162float(VLLM_LDG(cos_ptr + half_dim));
      float const sin_val = __bfloat162float(VLLM_LDG(sin_ptr + half_dim));

      float const rotated_val0 = val0 * cos_val - val1 * sin_val;
      float const rotated_val1 = val0 * sin_val + val1 * cos_val;
  
      elements[idx0] = rotated_val0;
      elements[idx1] = rotated_val1;
    }
  } else {
    // Before data exchange with in warp, we need to sync.
    __syncwarp();
    // Get the data from the other half of the warp. Use pre-computed cos/sin
    // values.
    for (int i = 0; i < numElemsPerThread; i++) {
      elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], 16);
      if (laneId < 16) {
        elements2[i] = -elements2[i];
      }

      int dim_idx = laneId * numElemsPerThread + i;
      dim_idx = (dim_idx * 2) % head_dim;
      int half_dim = dim_idx / 2;
      // Use pre-computed cos/sin from cache with optimized memory access
      float cos_val = __bfloat162float(VLLM_LDG(cos_ptr + half_dim));
      float sin_val = __bfloat162float(VLLM_LDG(sin_ptr + half_dim));

      elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
    }
    // __shfl_xor_sync does not provide memfence. Need to sync again.
    __syncwarp();
  }

  // Store.
  {
    vec_T vec;
    for (int i = 0; i < vecSize; i++) {
      __nv_bfloat162 vals = __float22bfloat162_rn(
          make_float2(elements[2 * i], elements[2 * i + 1]));
      reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) =
          vals;
    }
    vec_T* outputPtr = reinterpret_cast<vec_T*>(&qkv[offsetThread]);
    *outputPtr = vec;
  }
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

void launchFusedQKNormRope(void* qkv, int const num_tokens,
                           int const num_heads_q, int const num_heads_k,
                           int const num_heads_v, int const head_dim,
                           float const eps, void const* q_weight,
                           void const* k_weight,
                           __nv_bfloat16 const* cos_sin_cache,
                           bool const interleave, int64_t const* position_ids,
                           cudaStream_t stream) {
  constexpr int blockSize = 256;

  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;

  int const gridSize = common::divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  // Head dimensions should be a multiple of 64
  // Add more cases as needed
  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<64, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
            num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), cos_sin_cache,
            position_ids, num_tokens);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                num_heads_v, eps,
                reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                reinterpret_cast<__nv_bfloat16 const*>(k_weight), cos_sin_cache,
                position_ids, num_tokens);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                num_heads_v, eps,
                reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                reinterpret_cast<__nv_bfloat16 const*>(k_weight), cos_sin_cache,
                position_ids, num_tokens);
      });
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head dimension for fusedQKNormRope: ", head_dim);
  }
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
    torch::Tensor& position_ids    // Position IDs for RoPE [num_tokens]
) {
  // Input validation
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
  TORCH_CHECK(cos_sin_cache.size(1) == head_dim,
              "Cos/sin cache dimension must match head_dim");

  CHECK_INPUT(qkv, torch::kBFloat16);
  CHECK_INPUT(position_ids, torch::kInt64);
  CHECK_INPUT(q_weight, torch::kBFloat16);
  CHECK_INPUT(k_weight, torch::kBFloat16);
  CHECK_INPUT(cos_sin_cache, torch::kBFloat16);

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  tensorrt_llm::kernels::launchFusedQKNormRope(
      reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
      static_cast<int>(num_tokens), static_cast<int>(num_heads_q),
      static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
      static_cast<int>(head_dim), static_cast<float>(eps),
      reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
      !is_neox,  // interleave
      reinterpret_cast<int64_t const*>(position_ids.data_ptr()), stream);
}
