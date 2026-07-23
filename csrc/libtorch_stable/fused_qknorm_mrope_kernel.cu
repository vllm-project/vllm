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

// Fused per-head QK RMSNorm + multimodal RoPE (mRoPE) for Qwen3-VL-class models.
//
// This is the mRoPE analogue of fused_qknorm_rope_kernel.cu. The RMSNorm and
// RoPE-rotation math is identical; the only difference is where cos/sin come
// from. Standard RoPE uses a single position id per token, so cos/sin for
// half-dimension d live at cos_sin_cache[pos_id, d]. mRoPE instead has three
// position streams (time/height/width) in position_ids[3, num_tokens], and the
// half-dimension d selects which stream to use via the mrope_section [t, h, w]
// boundaries (with t + h + w == rotary_dim / 2). We therefore look up
// cos_sin_cache[position_ids[section(d), token], d] per half-dimension, with no
// pre-gather of the cache.
//
// v1 scope: base (one warp per token-head) kernel only, non-interleaved mRoPE
// section layout (mrope_interleaved == false, the Qwen3-VL default), head dims
// 64/128/256, both neox and gpt-j (interleaved) rotation styles. The
// cp.async NTokenHeads throughput variant present in the plain-RoPE kernel is a
// planned follow-up; the compile-fusion pass only fires for the configurations
// supported here.

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "../cuda_compat.h"
#include "type_convert.cuh"
#include "dispatch_utils.h"

#define CHECK_TYPE(x, st)                                                  \
  STD_TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
                  ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) \
  STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);    \
  CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL

  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
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

// Select the mRoPE section (0=time, 1=height, 2=width) that owns half-dimension
// `half_dim` for the non-interleaved layout, then return the token's position id
// for that section. position_ids is laid out row-major as [3, num_tokens].
__device__ __forceinline__ int64_t mropePositionForHalfDim(
    int64_t const* position_ids, int const num_tokens, int const tokenIdx,
    int const half_dim, int const mrope_section_t, int const mrope_section_h) {
  int section;
  if (half_dim < mrope_section_t) {
    section = 0;
  } else if (half_dim < mrope_section_t + mrope_section_h) {
    section = 1;
  } else {
    section = 2;
  }
  return position_ids[section * num_tokens + tokenIdx];
}

// Perform per-head QK Norm and mRoPE in a single kernel.
// scalar_t_in: data type of QKV and RMSNorm weights
// scalar_t_cache: data type of cos/sin cache
// head_dim: the dimension of each head
// interleave: interleave = !is_neox (rotary rotation style).
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void fusedQKNormMRopeKernel(
    void* qkv_void,                  // Combined QKV tensor
    int const num_heads_q,           // Number of query heads
    int const num_heads_k,           // Number of key heads
    int const num_heads_v,           // Number of value heads
    float const eps,                 // Epsilon for RMS normalization
    void const* q_weight_void,       // RMSNorm weights for query
    void const* k_weight_void,       // RMSNorm weights for key
    void const* cos_sin_cache_void,  // Pre-computed cos/sin cache
    int64_t const* position_ids,     // Position IDs for mRoPE [3, num_tokens]
    int const num_tokens,            // Number of tokens
    int const rotary_dim,            // Dimension for RoPE
    int const mrope_section_t,       // mRoPE time-section size (half-dims)
    int const mrope_section_h        // mRoPE height-section size (half-dims)
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

    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const total_qk_heads = num_heads_q + num_heads_k;

    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

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
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    int offsetWarp;
    if (isQ) {
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
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
      int dim = laneId * numElemsPerThread + i;
      float weight = isQ ? Converter::convert(q_weight[dim])
                         : Converter::convert(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

    // Apply mRoPE to normalized elements
    float elements2[numElemsPerThread];  // Additional buffer required for RoPE.

    int const embed_dim = rotary_dim / 2;
    int const rotary_lanes = rotary_dim / numElemsPerThread;
    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // gpt-j / interleaved rotation style.
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          int const dim_idx = laneId * numElemsPerThread + idx0;

          float const val0 = elements[idx0];
          float const val1 = elements[idx1];

          int const half_dim = dim_idx / 2;
          int64_t const pos_id = mropePositionForHalfDim(
              position_ids, num_tokens, tokenIdx, half_dim, mrope_section_t,
              mrope_section_h);
          T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cache_ptr + half_dim));
          float const sin_val = CacheConverter::convert(
              VLLM_LDG(cache_ptr + embed_dim + half_dim));

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // neox rotation style: pair element i with the element rotary_dim/2 away
        // in the head, exchanged across the warp.
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);

          if (laneId < pairOffset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = laneId * numElemsPerThread + i;

          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          int64_t const pos_id = mropePositionForHalfDim(
              position_ids, num_tokens, tokenIdx, half_dim, mrope_section_t,
              mrope_section_h);
          T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
          float cos_val =
              CacheConverter::convert(VLLM_LDG(cache_ptr + half_dim));
          float sin_val =
              CacheConverter::convert(VLLM_LDG(cache_ptr + embed_dim + half_dim));

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

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormMRope(void* qkv, int const num_tokens,
                            int const num_heads_q, int const num_heads_k,
                            int const num_heads_v, int const head_dim,
                            int const rotary_dim, float const eps,
                            void const* q_weight, void const* k_weight,
                            void const* cos_sin_cache, bool const interleave,
                            int64_t const* position_ids,
                            int const mrope_section_t, int const mrope_section_h,
                            cudaStream_t stream) {
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
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim,
                mrope_section_t, mrope_section_h);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim,
                mrope_section_t, mrope_section_h);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormMRopeKernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim,
                mrope_section_t, mrope_section_h);
      });
      break;
    default:
      STD_TORCH_CHECK(
          false, "Unsupported head dimension for fusedQKNormMRope: ", head_dim);
  }
}

}  // namespace tensorrt_llm::kernels

void fused_qk_norm_mrope(
    torch::stable::Tensor&
        qkv,              // Combined QKV tensor [num_tokens,
                          // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,  // Number of query heads
    int64_t num_heads_k,  // Number of key heads
    int64_t num_heads_v,  // Number of value heads
    int64_t head_dim,     // Dimension per head
    double eps,           // Epsilon for RMS normalization
    torch::stable::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    torch::stable::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    torch::stable::Tensor& cos_sin_cache,  // Cos/sin cache [max_position,
                                           // rotary_dim]
    bool is_neox,  // Whether RoPE is applied in Neox style
    torch::stable::Tensor&
        position_ids,          // mRoPE position IDs [3, num_tokens] (t/h/w)
    int64_t mrope_section_t,   // mRoPE time-section size (in half-dims)
    int64_t mrope_section_h    // mRoPE height-section size (in half-dims)
) {
  // Input validation
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(position_ids, torch::headeronly::ScalarType::Long);

  STD_TORCH_CHECK(qkv.dim() == 2,
                  "QKV tensor must be 2D: [num_tokens, "
                  "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  STD_TORCH_CHECK(position_ids.dim() == 2 && position_ids.size(0) == 3,
                  "mRoPE position IDs must be 2D: [3, num_tokens]");
  STD_TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  STD_TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2,
                  "Cos/sin cache must be 2D: [max_position, rotary_dim]");
  STD_TORCH_CHECK(q_weight.size(0) == head_dim,
                  "Query weights size must match head dimension");
  STD_TORCH_CHECK(k_weight.size(0) == head_dim,
                  "Key weights size must match head dimension");

  STD_TORCH_CHECK(cos_sin_cache.size(1) % 2 == 0, "rotary_dim must be even");
  STD_TORCH_CHECK(cos_sin_cache.size(1) <= head_dim,
                  "rotary_dim must be less than or equal to head_dim");

  STD_TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                      qkv.scalar_type() == k_weight.scalar_type(),
                  "qkv, q_weight and k_weight must have the same dtype");

  int64_t num_tokens = qkv.size(0);
  STD_TORCH_CHECK(position_ids.size(1) == num_tokens,
                  "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  STD_TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  int64_t const embed_dim = cos_sin_cache.size(1) / 2;
  STD_TORCH_CHECK(
      mrope_section_t >= 0 && mrope_section_h >= 0 &&
          mrope_section_t + mrope_section_h <= embed_dim,
      "mrope_section_t + mrope_section_h must be <= rotary_dim/2");

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  auto stream = get_current_cuda_stream(qkv.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qk_norm_mrope_kernel", [&] {
        using qkv_scalar_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "fused_qk_norm_mrope_kernel", [&] {
              using cache_scalar_t = scalar_t;
              tensorrt_llm::kernels::launchFusedQKNormMRope<qkv_scalar_t,
                                                            cache_scalar_t>(
                  qkv.data_ptr(), static_cast<int>(num_tokens),
                  static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
                  static_cast<int>(num_heads_v), static_cast<int>(head_dim),
                  static_cast<int>(cos_sin_cache.size(1)),
                  static_cast<float>(eps), q_weight.data_ptr(),
                  k_weight.data_ptr(), cos_sin_cache.data_ptr(), !is_neox,
                  reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
                  static_cast<int>(mrope_section_t),
                  static_cast<int>(mrope_section_h), stream);
            });
      });
}
