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
// The MLA writer and reader (concat_and_cache_mla_nvfp4,
// gather_and_dequant_cache_mla_nvfp4) use a page-segmented layout instead;
// see concat_and_cache_mla_nvfp4_kernel.
//
// Reuses device functions from nvfp4_utils.cuh:
//   - cvt_warp_fp16_to_fp4()  for bf16 → fp4 quantization + block scale
//   - pack_fp4()              for packing float pairs to fp4
//   - reciprocal_approximate_ftz() for fast reciprocal

#define NVFP4_ENABLE_ELTS16 1
#include "quantization/fp4/nvfp4_utils.cuh"

#include "dispatch_utils.h"
#include "gather_page_task.cuh"
#include "torch_utils.h"

#include <cmath>
#include <string>

namespace vllm {

// The cache layout is identical; this only selects the store-time scale search.
enum class NVFP4KVScaleSearch {
  DEFAULT,
  FOUR_OVER_SIX,
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
  // Match cvt.rn.satfinite.e2m1x2.f32, including ties-to-even boundaries.
  if (ax <= 0.25f) {
    q = 0.0f;
  } else if (ax < 0.75f) {
    q = 0.5f;
  } else if (ax <= 1.25f) {
    q = 1.0f;
  } else if (ax < 1.75f) {
    q = 1.5f;
  } else if (ax <= 2.5f) {
    q = 2.0f;
  } else if (ax < 3.5f) {
    q = 3.0f;
  } else if (ax <= 5.0f) {
    q = 4.0f;
  } else {
    q = 6.0f;
  }
  return copysignf(q, x);
}

__device__ __forceinline__ void nvfp4_candidate_scale(float SFScaleVal,
                                                      float vecMax,
                                                      float denominator,
                                                      uint8_t* fp8_sf,
                                                      float* outputScale) {
  float SFValue =
      SFScaleVal * (vecMax * reciprocal_approximate_ftz(denominator));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  reinterpret_cast<__nv_fp8_e4m3&>(*fp8_sf) = tmp;
  SFValue = float(tmp);
  *outputScale = SFValue != 0.0f
                     ? reciprocal_approximate_ftz(
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

// Kernel: quantize bf16/half MLA latent KV to NVFP4 + FP8-E4M3 and store in
// the page-segmented MLA NVFP4 KV cache layout.
//
// Per page (block_size tokens), with nope_dim = kv_lora_rank (512) and
// pe_dim (64), all offsets in bytes relative to the page base:
//   token t data record at t * token_data_bytes, where
//     token_data_bytes = nope_dim / 2 + pe_dim (= 320):
//       [nope_dim/2 bytes packed FP4 of quantized kv_c |
//        pe_dim bytes FP8-E4M3 of k_pe]
//   token t scale factors at block_size * token_data_bytes + t * sf_dim,
//     where sf_dim = nope_dim / 16 (= 32):
//       sf_dim E4M3 scale factors for the NoPE groups, linear (sf s at
//       byte +s, no swizzle).
//
// Quantization contract (cache stores x / k_scale):
//   NoPE: NVFP4 quantization of kv_c with global scale 1/k_scale.
//   RoPE: direct E4M3 cast of k_pe / k_scale, no scale factors.
//
// Threading: one CUDA block per token (like concat_and_cache_mla).
// Groups of THREADS_PER_SF threads each quantize one 16-element NoPE
// group; all threads then stride over the RoPE elements.
template <typename scalar_t>
__global__ void concat_and_cache_mla_nvfp4_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    uint8_t* __restrict__ kv_cache,     // [num_blocks, block_size, full_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const float* __restrict__ scale_ptr,       // pointer to checkpoint k_scale
    const int64_t kv_c_stride,                 // kv_c.stride(0) in elements
    const int64_t k_pe_stride,                 // k_pe.stride(0) in elements
    const int64_t block_stride,                // kv_cache.stride(0) in bytes
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size                       //
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int ELTS = CVT_FP4_ELTS_PER_THREAD;  // 16 or 8
  static constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / ELTS;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  // Page-segmented layout offsets (bytes).
  const int token_data_bytes = kv_lora_rank / 2 + pe_dim;  // 320
  const int sf_dim = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;   // 32

  uint8_t* __restrict__ page_base = kv_cache + block_idx * block_stride;
  uint8_t* __restrict__ data_dst =
      page_base + static_cast<int64_t>(block_offset) * token_data_bytes;
  uint8_t* __restrict__ rope_dst = data_dst + kv_lora_rank / 2;
  uint8_t* __restrict__ sf_dst =
      page_base + static_cast<int64_t>(block_size) * token_data_bytes +
      static_cast<int64_t>(block_offset) * sf_dim;

  const float global_scale = 1.0f / *scale_ptr;

  const CudaType* __restrict__ nope_src =
      reinterpret_cast<const CudaType*>(kv_c) + token_idx * kv_c_stride;
  const CudaType* __restrict__ rope_src =
      reinterpret_cast<const CudaType*>(k_pe) + token_idx * k_pe_stride;

  const int tid = threadIdx.x;
  const int num_thread_groups = blockDim.x / THREADS_PER_SF;
  const int tg_id = tid / THREADS_PER_SF;
  const int tg_lane = tid % THREADS_PER_SF;

  const int total_groups = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;  // 32

  for (int g = tg_id; g < total_groups; g += num_thread_groups) {
    // Load 16 (or 8) bf16/half elements from kv_c
    PVec in_vec;
    const CudaType* __restrict__ src_ptr =
        nope_src + g * CVT_FP4_SF_VEC_SIZE + tg_lane * ELTS;

#pragma unroll
    for (int i = 0; i < ELTS / 2; i++) {
      in_vec.elts[i] =
          reinterpret_cast<const typename PackedTypeConverter<CudaType>::Type*>(
              src_ptr)[i];
    }

    // Quantize: produces packed fp4 and writes the block scale factor.
    uint8_t sf_val;
    uint8_t* sf_out_ptr = (tg_lane == 0) ? &sf_val : nullptr;

    fp4_packed_t packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
        in_vec, global_scale, sf_out_ptr);

    // Write packed FP4 data.
#if CVT_FP4_PACK16
    {
      // 16 elements -> 8 bytes (u32x2)
      reinterpret_cast<uint64_t*>(data_dst + g * 8)[0] =
          (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
    }
#else
    {
      // 8 elements -> 4 bytes (uint32_t)
      int data_byte_offset = g * CVT_FP4_SF_VEC_SIZE / 2 + tg_lane * ELTS / 2;
      reinterpret_cast<uint32_t*>(data_dst + data_byte_offset)[0] = packed;
    }
#endif

    if (sf_out_ptr != nullptr) {
      sf_dst[g] = sf_val;
    }
  }

  for (int i = tid; i < pe_dim; i += blockDim.x) {
    float v = static_cast<float>(rope_src[i]) * global_scale;
    __nv_fp8_e4m3 q(v);
    rope_dst[i] = *reinterpret_cast<uint8_t*>(&q);
  }
}

// Decode one E2M1 (FP4) code to float.
__device__ __forceinline__ float e2m1_to_float(uint8_t code) {
  static constexpr float kE2m1Lut[16] = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kE2m1Lut[code & 0xF];
}

// Kernel: gather context tokens from the page-segmented MLA NVFP4 KV cache
// (see concat_and_cache_mla_nvfp4_kernel above) and dequantize them into a
// dense bf16/half [num_tokens, kv_lora_rank + pe_dim] workspace.
// Used for chunked-prefill / prefix-cache context reads; inverts the writer's
// scaling (cache stores x / k_scale):
//   NoPE: x = fp4 * float(sf_e4m3) * k_scale
//   RoPE: x = float(e4m3) * k_scale
//
// Threading: one CUDA block per logical cache page, scheduled like
// gather_and_maybe_dequant_cache_page in cache_kernels.cu. Threads stride over
// the page's 16-byte output chunks of 8 elements, each decoded from one uint32
// of packed FP4 NoPE or one uint2 of E4M3 RoPE.
template <typename scalar_t>
__global__ void gather_and_dequant_cache_mla_nvfp4_kernel(
    const uint8_t* __restrict__ src_cache,    // [num_blocks, block_size,
                                              //  full_dim] bytes
    scalar_t* __restrict__ dst,               // [num_tokens, nope+pe]
    const int32_t* __restrict__ block_table,  // [batch, max_blocks]
    const int32_t* __restrict__ cu_seq_lens,  // [batch+1]
    const int32_t num_reqs, const int32_t num_tokens,
    const float* __restrict__ scale_ptr,     // checkpoint k_scale
    const int32_t* __restrict__ seq_starts,  // [batch] or nullptr
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t dst_entry_stride, const int kv_lora_rank, const int pe_dim,
    const int block_size) {
  __shared__ GatherPageTask page;
  __shared__ int32_t physical_block;
  __shared__ bool copy_task;

  if (threadIdx.x == 0) {
    copy_task =
        map_gather_page_task<true>(blockIdx.x, cu_seq_lens, num_reqs,
                                   num_tokens, block_size, seq_starts, page) &&
        page.logical_block < block_table_stride;
    if (copy_task) {
      physical_block =
          block_table[page.req_id * block_table_stride + page.logical_block];
    }
  }
  __syncthreads();
  if (!copy_task) {
    return;
  }

  const int token_data_bytes = kv_lora_rank / 2 + pe_dim;  // 320
  const int sf_dim = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;   // 32
  const int nope_chunks = kv_lora_rank / 8;
  const int row_chunks = nope_chunks + pe_dim / 8;
  const float k_scale = *scale_ptr;
  const uint8_t* __restrict__ page_base =
      src_cache + physical_block * cache_block_stride;
  const uint8_t* __restrict__ sf_base =
      page_base + static_cast<int64_t>(block_size) * token_data_bytes;

  const int page_chunks =
      (page.page_token_end - page.page_token_begin) * row_chunks;
  for (int flat_idx = threadIdx.x; flat_idx < page_chunks;
       flat_idx += blockDim.x) {
    const int token_offset = flat_idx / row_chunks;
    const int chunk = flat_idx - token_offset * row_chunks;
    const int slot = page.page_token_begin + token_offset;
    const uint8_t* __restrict__ data =
        page_base + static_cast<int64_t>(slot) * token_data_bytes;
    alignas(16) scalar_t out[8];
    if (chunk < nope_chunks) {
      // Element j sits at bits [4j+3:4j]; chunks 2s and 2s+1 share SF s.
      const uint32_t packed = reinterpret_cast<const uint32_t*>(data)[chunk];
      const __nv_fp8_e4m3* __restrict__ sf =
          reinterpret_cast<const __nv_fp8_e4m3*>(
              sf_base + static_cast<int64_t>(slot) * sf_dim);
      const float sf_scale = float(sf[chunk >> 1]) * k_scale;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        out[j] = static_cast<scalar_t>(
            e2m1_to_float((packed >> (4 * j)) & 0xF) * sf_scale);
      }
    } else {
      const uint2 packed = reinterpret_cast<const uint2*>(
          data + kv_lora_rank / 2)[chunk - nope_chunks];
      const __nv_fp8_e4m3* rope =
          reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        out[j] = static_cast<scalar_t>(float(rope[j]) * k_scale);
      }
    }
    // The launcher checks that dst rows are 16-byte aligned.
    scalar_t* __restrict__ dst_row =
        dst + static_cast<int64_t>(page.output_token_begin + token_offset) *
                  dst_entry_stride;
    reinterpret_cast<uint4*>(dst_row)[chunk] =
        *reinterpret_cast<const uint4*>(out);
  }
}

}  // namespace vllm

// Non-template entry point callable from cache_kernels.cu.
// Receives key_cache/value_cache as kv_cache[:, 0] and kv_cache[:, 1].
// Each KV side contains both data and scale:
//   page = [K_data | K_scale | V_data | V_scale]
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype) {
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
        } else {
          STD_TORCH_CHECK(false,
                          "Unsupported NVFP4 KV cache dtype: ", kv_cache_dtype);
        }
      });
}

// Non-template entry point callable from cache_kernels.cu.
void concat_and_cache_mla_nvfp4_dispatch(
    torch::stable::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::stable::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::stable::Tensor& kv_cache,      // [num_blocks, block_size, full_dim]
    torch::stable::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    torch::stable::Tensor& scale) {       // [1], float, k_scale
  // NOTE: slot_mapping.size(0) is the number of actual tokens; kv_c/k_pe
  // may be padded for CUDA graphs (see concat_and_cache_mla).
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);

  STD_TORCH_CHECK(kv_cache.dim() == 3, "kv_cache must be 3D");
  int block_size = kv_cache.size(1);

  STD_TORCH_CHECK(kv_lora_rank % CVT_FP4_SF_VEC_SIZE == 0,
                  "kv_lora_rank must be divisible by 16 for NVFP4 MLA KV "
                  "cache, got ",
                  kv_lora_rank);

  int token_data_bytes = kv_lora_rank / 2 + pe_dim;
  int sf_dim = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;
  int full_dim = token_data_bytes + sf_dim;

  STD_TORCH_CHECK(kv_cache.size(2) == full_dim,
                  "kv_cache last dim must be kv_lora_rank/2 + pe_dim + "
                  "kv_lora_rank/16, got ",
                  kv_cache.size(2), " expected ", full_dim);
  STD_TORCH_CHECK(kv_cache.stride(2) == 1 && kv_cache.stride(1) == full_dim,
                  "kv_cache pages must be contiguous for NVFP4 MLA KV cache");

  int64_t kv_c_stride = kv_c.stride(0);
  int64_t k_pe_stride = k_pe.stride(0);
  int64_t block_stride = kv_cache.stride(0);  // page bytes

  constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  int total_groups = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;
  int num_threads = std::max(total_groups * THREADS_PER_SF, pe_dim);
  num_threads = std::min(num_threads, 512);
  num_threads = ((num_threads + 31) / 32) * 32;

  dim3 grid(num_tokens);
  dim3 block(num_threads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_c.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      kv_c.scalar_type(), "concat_and_cache_mla_nvfp4", [&] {
        vllm::concat_and_cache_mla_nvfp4_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(kv_c.const_data_ptr<scalar_t>(),
                                         k_pe.const_data_ptr<scalar_t>(),
                                         kv_cache.mutable_data_ptr<uint8_t>(),
                                         slot_mapping.const_data_ptr<int64_t>(),
                                         scale.const_data_ptr<float>(),
                                         kv_c_stride, k_pe_stride, block_stride,
                                         kv_lora_rank, pe_dim, block_size);
      });
}

// Non-template entry point callable from cache_kernels.cu.
void gather_and_dequant_cache_mla_nvfp4_dispatch(
    const torch::stable::Tensor& src_cache,    // [num_blocks, block_size,
                                               //  full_dim] uint8
    const torch::stable::Tensor& dst,          // [num_tokens, nope+pe]
    const torch::stable::Tensor& block_table,  // [batch, max_blocks] int32
    const torch::stable::Tensor& cu_seq_lens,  // [batch+1] int32
    int64_t num_tokens, int64_t kv_lora_rank,
    const torch::stable::Tensor& scale,  // [] float, k_scale
    const int32_t* seq_starts_ptr) {     // [batch] int32 or nullptr
  int block_size = src_cache.size(1);
  int dst_dim = dst.size(-1);
  int pe_dim = dst_dim - static_cast<int>(kv_lora_rank);

  STD_TORCH_CHECK(pe_dim > 0 && pe_dim % 8 == 0,
                  "dst last dim minus kv_lora_rank must be a positive "
                  "multiple of 8, got ",
                  dst_dim, " - ", kv_lora_rank);
  STD_TORCH_CHECK(kv_lora_rank % CVT_FP4_SF_VEC_SIZE == 0,
                  "kv_lora_rank must be divisible by 16, got ", kv_lora_rank);

  int token_data_bytes = kv_lora_rank / 2 + pe_dim;
  int sf_dim = kv_lora_rank / CVT_FP4_SF_VEC_SIZE;
  int full_dim = token_data_bytes + sf_dim;

  STD_TORCH_CHECK(src_cache.size(2) == full_dim,
                  "src_cache last dim must be kv_lora_rank/2 + pe_dim + "
                  "kv_lora_rank/16, got ",
                  src_cache.size(2), " expected ", full_dim);
  STD_TORCH_CHECK(src_cache.stride(2) == 1 && src_cache.stride(1) == full_dim,
                  "src_cache pages must be contiguous for NVFP4 MLA KV cache");

  // RoPE chunks are 8-byte loads; every output chunk is a 16-byte store.
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(src_cache.const_data_ptr()) % 8 == 0,
      "src_cache must be 8-byte aligned");
  int64_t dst_entry_stride = dst.stride(0);
  STD_TORCH_CHECK(
      dst.stride(-1) == 1 && (dst_entry_stride * 2) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(dst.const_data_ptr()) % 16 == 0,
      "dst rows must be contiguous and 16-byte aligned");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);  // page bytes
  const int32_t num_reqs = cu_seq_lens.size(0) - 1;

  // One CUDA block per logical page; partial first/last pages add up to two
  // per request (see gather_and_maybe_dequant_cache).
  const int threads = num_tokens >= (1 << 20) ? 128 : 256;
  dim3 grid(div_ceil(static_cast<int32_t>(num_tokens), block_size) +
            2 * num_reqs);
  dim3 block(threads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dst.scalar_type(), "gather_and_dequant_cache_mla_nvfp4", [&] {
        vllm::gather_and_dequant_cache_mla_nvfp4_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                src_cache.const_data_ptr<uint8_t>(),
                dst.mutable_data_ptr<scalar_t>(),
                block_table.const_data_ptr<int32_t>(),
                cu_seq_lens.const_data_ptr<int32_t>(), num_reqs,
                static_cast<int32_t>(num_tokens), scale.const_data_ptr<float>(),
                seq_starts_ptr, block_table_stride, cache_block_stride,
                dst_entry_stride, static_cast<int>(kv_lora_rank), pe_dim,
                block_size);
      });
}
