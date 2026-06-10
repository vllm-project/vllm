// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NVFP4 KV cache store kernel.
// Quantizes bf16 key/value to packed FP4 + FP8 block scales and writes them
// into the paged KV cache.
//
// Per page layout: [K_data | K_scale | V_data | V_scale]
// Both data and scale regions are contiguous per head, enabling direct
// TMA descriptor use.
//
// Reuses device functions from nvfp4_utils.cuh:
//   - cvt_warp_fp16_to_fp4()  for bf16 → fp4 quantization + block scale
//   - pack_fp4()              for packing float pairs to fp4
//   - reciprocal_approximate_ftz() for fast reciprocal

#define NVFP4_ENABLE_ELTS16 1
#include "libtorch_stable/quantization/fp4/nvfp4_utils.cuh"

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"

#include <cmath>
#include <string>

namespace vllm {

enum class NVFP4KVScaleSearch {
  DEFAULT,
  FOUR_OVER_SIX,
  FOUR_OVER_SIX_K_ONLY,
};

// Compute swizzled scale offset for SM100 trtllm-gen MHA kernel.
// The swizzle pattern for HND layout is:
//   [T//4, 4, 4, S//4] → permute(0, 2, 3, 1) → reshape to [T, S]
// where T = block_size (page_size), S = scale_dim = head_size // 16.
//
// For a linear (t, s) position, the swizzled position is:
//   swizzled_t = (t / 4) * 4 + (s / (S / 4))
//   swizzled_s = (s % (S / 4)) * 4 + (t % 4)
__device__ __forceinline__ int swizzle_scale_offset(int t, int s,
                                                    int scale_dim) {
  int s_group = scale_dim / 4;
  int swizzled_t = (t / 4) * 4 + (s / s_group);
  int swizzled_s = (s % s_group) * 4 + (t % 4);
  return swizzled_t * scale_dim + swizzled_s;
}

__device__ __forceinline__ float round_to_nearest_e2m1(float x) {
  const float ax = fabsf(x);
  float q;
  if (ax < 0.25f) {
    q = 0.0f;
  } else if (ax < 0.75f) {
    q = 0.5f;
  } else if (ax < 1.25f) {
    q = 1.0f;
  } else if (ax < 1.75f) {
    q = 1.5f;
  } else if (ax < 2.5f) {
    q = 2.0f;
  } else if (ax < 3.5f) {
    q = 3.0f;
  } else if (ax < 5.0f) {
    q = 4.0f;
  } else {
    q = 6.0f;
  }
  return copysignf(q, x);
}

__device__ __forceinline__ void nvfp4_candidate_scale(
    float SFScaleVal, float vecMax, float denominator, uint8_t* fp8_sf,
    float* outputScale) {
  float SFValue =
      SFScaleVal * (vecMax * reciprocal_approximate_ftz(denominator));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  reinterpret_cast<__nv_fp8_e4m3&>(*fp8_sf) = tmp;
  SFValue = float(tmp);
  *outputScale =
      SFValue != 0.0f ? reciprocal_approximate_ftz(
                            SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
}

template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ float nvfp4_reconstruction_error(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float outputScale) {
  float error = 0.0f;
  const float dequantScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    float2 vals = cast_to_float2(vec.elts[i]);
    if (outputScale == 0.0f) {
      error += vals.x * vals.x + vals.y * vals.y;
    } else {
      float qx = round_to_nearest_e2m1(vals.x * outputScale);
      float qy = round_to_nearest_e2m1(vals.y * outputScale);
      float dx = qx * dequantScale - vals.x;
      float dy = qy * dequantScale - vals.y;
      error += dx * dx + dy * dy;
    }
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    error += __shfl_xor_sync(0xffffffffu, error, 1);
  }
  return error;
}

template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ fp4_packed_t cvt_warp_fp16_to_fp4_4over6(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float SFScaleVal, uint8_t* SFout) {
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  const float vecMax = float(__hmax(localMax.x, localMax.y));

  uint8_t sf6;
  uint8_t sf4;
  float outputScale6;
  float outputScale4;
  nvfp4_candidate_scale(SFScaleVal, vecMax, 6.0f, &sf6, &outputScale6);
  nvfp4_candidate_scale(SFScaleVal, vecMax, 4.0f, &sf4, &outputScale4);

  const float err6 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale6);
  const float err4 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale4);
  const bool use4 = err4 < err6;
  const float outputScale = use4 ? outputScale4 : outputScale6;

  if (SFout) *SFout = use4 ? sf4 : sf6;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return pack_fp4(fp2Vals);
}

// Kernel: quantize bf16 key/value to NVFP4 and store in paged KV cache.
//
// Takes separate data and scale cache pointers for K and V.
// Within each KV side, data and scale are separate contiguous regions.
//
// Threading: one CUDA block per token, threads process heads and
// groups of 16 elements within each head.
template <typename scalar_t, NVFP4KVScaleSearch SCALE_SEARCH>
__global__ void reshape_and_cache_nvfp4_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_data_cache,  // data region for K
    uint8_t* __restrict__ value_data_cache,    // data region for V
    uint8_t* __restrict__ key_scale_cache,     // scale region for K
    uint8_t* __restrict__ value_scale_cache,   // scale region for V
    const int64_t* __restrict__ slot_mapping,  // [num_actual_tokens]
    const float* __restrict__ k_scale_ptr,     // pointer to checkpoint k_scale
    const float* __restrict__ v_scale_ptr,     // pointer to checkpoint v_scale
    const int64_t key_stride,                  // key.stride(0) in elements
    const int64_t value_stride,                // value.stride(0) in elements
    const int num_heads, const int head_size, const int block_size,
    const int64_t data_block_stride,         // data cache stride for dim 0
    const int64_t data_head_stride,          // data cache stride for heads
    const int64_t data_block_offset_stride,  // data cache stride for tokens
    const int64_t scale_block_stride,        // scale cache stride for dim 0
    const int64_t scale_head_stride,         // scale cache stride for heads
    const int64_t scale_block_offset_stride  // scale cache stride for tokens
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int ELTS = CVT_FP4_ELTS_PER_THREAD;  // 16 or 8
  static constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / ELTS;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  const int scale_dim = head_size / 16;
  const int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;

  const int total_groups = num_heads * groups_per_head;
  const int tid = threadIdx.x;
  const int num_thread_groups = blockDim.x / THREADS_PER_SF;
  const int tg_id = tid / THREADS_PER_SF;
  const int tg_lane = tid % THREADS_PER_SF;

  // Process both K (kv=0) and V (kv=1)
#pragma unroll
  for (int kv = 0; kv < 2; kv++) {
    const scalar_t* __restrict__ src = (kv == 0) ? key : value;
    const float global_scale = 1.0f / ((kv == 0) ? *k_scale_ptr : *v_scale_ptr);
    const int64_t src_stride = (kv == 0) ? key_stride : value_stride;
    uint8_t* __restrict__ data_cache =
        (kv == 0) ? key_data_cache : value_data_cache;
    uint8_t* __restrict__ sc_cache =
        (kv == 0) ? key_scale_cache : value_scale_cache;

    // Source pointer for this token (use actual stride, not assumed contiguous)
    const CudaType* __restrict__ token_src =
        reinterpret_cast<const CudaType*>(src) + token_idx * src_stride;

    // Destination bases in data and scale caches for this token's block
    uint8_t* __restrict__ data_block =
        data_cache + block_idx * data_block_stride;
    uint8_t* __restrict__ scale_block =
        sc_cache + block_idx * scale_block_stride;

    for (int g = tg_id; g < total_groups; g += num_thread_groups) {
      const int head = g / groups_per_head;
      const int group_in_head = g % groups_per_head;

      // Load 16 (or 8) bf16 elements from source
      PVec in_vec;
      const CudaType* __restrict__ src_ptr =
          token_src + head * head_size + group_in_head * CVT_FP4_SF_VEC_SIZE +
          tg_lane * ELTS;

#pragma unroll
      for (int i = 0; i < ELTS / 2; i++) {
        in_vec.elts[i] = reinterpret_cast<
            const typename PackedTypeConverter<CudaType>::Type*>(src_ptr)[i];
      }

      // Quantize: produces packed fp4 and writes scale factor.
      uint8_t sf_val;
      uint8_t* sf_out_ptr = (tg_lane == 0) ? &sf_val : nullptr;

      fp4_packed_t packed;
      if constexpr (SCALE_SEARCH == NVFP4KVScaleSearch::FOUR_OVER_SIX) {
        packed = cvt_warp_fp16_to_fp4_4over6<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      } else if constexpr (SCALE_SEARCH ==
                           NVFP4KVScaleSearch::FOUR_OVER_SIX_K_ONLY) {
        if (kv == 0) {
          packed = cvt_warp_fp16_to_fp4_4over6<CudaType, THREADS_PER_SF>(
              in_vec, global_scale, sf_out_ptr);
        } else {
          packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
              in_vec, global_scale, sf_out_ptr);
        }
      } else {
        packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      }

      // Write packed FP4 data to data cache
      uint8_t* __restrict__ data_dst = data_block + head * data_head_stride +
                                       block_offset * data_block_offset_stride;

#if CVT_FP4_PACK16
      {
        // 16 elements → 8 bytes (u32x2)
        int data_byte_offset = group_in_head * 8;
        reinterpret_cast<uint64_t*>(data_dst + data_byte_offset)[0] =
            (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
      }
#else
      {
        // 8 elements → 4 bytes (uint32_t)
        int data_byte_offset =
            group_in_head * CVT_FP4_SF_VEC_SIZE / 2 + tg_lane * ELTS / 2;
        reinterpret_cast<uint32_t*>(data_dst + data_byte_offset)[0] = packed;
      }
#endif

      // Write block scale to scale cache.
      // K (kv==0): linear layout (no swizzle).
      // V (kv==1): swizzled layout for SM100 trtllm-gen MHA kernel.
      if (sf_out_ptr != nullptr) {
        int scale_idx = group_in_head;
        uint8_t* __restrict__ scale_dst;
        if (kv == 0) {
          scale_dst = scale_block + head * scale_head_stride +
                      block_offset * scale_block_offset_stride + scale_idx;
        } else {
          int swizzled_offset =
              swizzle_scale_offset(block_offset, scale_idx, scale_dim);
          int swizzled_t = swizzled_offset / scale_dim;
          int swizzled_s = swizzled_offset % scale_dim;
          scale_dst = scale_block + head * scale_head_stride +
                      swizzled_t * scale_block_offset_stride + swizzled_s;
        }
        *scale_dst = sf_val;
      }
    }
  }
}

}  // namespace vllm

// Non-template entry point callable from cache_kernels.cu.
// Receives key_cache/value_cache as kv_cache[:, 0] and kv_cache[:, 1].
// Each KV side contains both data and scale:
//   page = [K_data | K_scale | V_data | V_scale]
void reshape_and_cache_nvfp4_dispatch(torch::stable::Tensor& key,
                                      torch::stable::Tensor& value,
                                      torch::stable::Tensor& key_cache,
                                      torch::stable::Tensor& value_cache,
                                      torch::stable::Tensor& slot_mapping,
                                      torch::stable::Tensor& k_scale,
                                      torch::stable::Tensor& v_scale,
                                      const std::string& kv_cache_dtype) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int data_dim = head_size / 2;
  int scale_dim = head_size / 16;
  int full_dim = data_dim + scale_dim;

  // key_cache is kv_cache[:, 0] with shape
  // [num_blocks, block_size, num_heads, full_dim] in logical order.
  // Strides encode the physical layout (HND or NHD).
  STD_TORCH_CHECK(key_cache.dim() == 4, "key_cache must be 4D");
  STD_TORCH_CHECK(key_cache.size(3) == full_dim,
                  "key_cache last dim must be data_dim + scale_dim, got ",
                  key_cache.size(3), " expected ", full_dim);

  int block_size = key_cache.size(1);

  STD_TORCH_CHECK(head_size % 16 == 0,
                  "head_size must be divisible by 16 for NVFP4 KV cache");
  STD_TORCH_CHECK(block_size % 4 == 0,
                  "block_size must be divisible by 4 for NVFP4 KV cache "
                  "swizzle");

  // Detect physical layout from strides (based on full_dim).
  // HND: head stride > block_offset stride.
  bool is_hnd = key_cache.stride(2) > key_cache.stride(1);

  int64_t data_block_stride = key_cache.stride(0);  // page_bytes
  int64_t data_head_stride, data_block_offset_stride;
  if (is_hnd) {
    data_head_stride = (int64_t)block_size * data_dim;
    data_block_offset_stride = data_dim;
  } else {
    data_head_stride = data_dim;
    data_block_offset_stride = (int64_t)num_heads * data_dim;
  }

  // Page layout: [K_data | K_scale | V_data | V_scale]
  // Scale follows data within each KV side.
  int64_t data_per_kv = (int64_t)num_heads * block_size * data_dim;

  uint8_t* key_scale_ptr = key_cache.mutable_data_ptr<uint8_t>() + data_per_kv;
  uint8_t* value_scale_ptr =
      value_cache.mutable_data_ptr<uint8_t>() + data_per_kv;

  // Scale strides: same page stride, inner strides from layout.
  int64_t scale_block_stride = data_block_stride;
  int64_t scale_head_stride, scale_block_offset_stride;
  if (is_hnd) {
    scale_head_stride = (int64_t)block_size * scale_dim;
    scale_block_offset_stride = scale_dim;
  } else {
    scale_head_stride = scale_dim;
    scale_block_offset_stride = (int64_t)num_heads * scale_dim;
  }

  const float* k_scale_ptr = k_scale.const_data_ptr<float>();
  const float* v_scale_ptr = v_scale.const_data_ptr<float>();

  int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;
  int total_groups = num_heads * groups_per_head;
  constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  int num_threads = std::min(total_groups * THREADS_PER_SF, 512);
  num_threads = ((num_threads + 31) / 32) * 32;

  dim3 grid(num_tokens);
  dim3 block(num_threads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      key.scalar_type(), "reshape_and_cache_nvfp4", [&] {
        if (kv_cache_dtype == "nvfp4") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::DEFAULT>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else if (kv_cache_dtype == "nvfp4_4over6") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::FOUR_OVER_SIX>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else if (kv_cache_dtype == "nvfp4_4over6_k_only") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::FOUR_OVER_SIX_K_ONLY>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else {
          STD_TORCH_CHECK(false, "Unsupported NVFP4 KV quantization mode: ",
                          kv_cache_dtype);
        }
      });
}
