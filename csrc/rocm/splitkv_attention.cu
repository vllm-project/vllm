// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstring>

#include "../attention/dtype_fp8.cuh"
#include "../cuda_compat.h"
#include "../quantization/w8a8/fp8/amd/quant_utils.cuh"

namespace {

constexpr int HEAD_SIZE = 256;
constexpr int LOGICAL_TILE_SIZE = 32;
[[maybe_unused]] constexpr int WMMA_LOGICAL_TILE_SIZE = 16;

bool current_device_is_gfx11() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return false;
  }
  return std::strncmp(properties.gcnArchName, "gfx11", 5) == 0;
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __hip_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

template <typename scalar_t, bool FP8_KV>
__device__ __forceinline__ float load_kv(const void* cache, int64_t offset,
                                         float scale) {
  if constexpr (FP8_KV) {
    return vllm::fp8::scaled_convert<
        float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3>(
        static_cast<const uint8_t*>(cache)[offset], scale);
  } else {
    return to_float(static_cast<const scalar_t*>(cache)[offset]);
  }
}

#if defined(__GFX11__)
using splitkv_bit16x8 =
    __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using splitkv_bit16x16 =
    __attribute__((__vector_size__(16 * sizeof(uint16_t)))) uint16_t;
using splitkv_bit8x16 =
    __attribute__((__vector_size__(16 * sizeof(uint8_t)))) uint8_t;
using splitkv_floatx8 =
    __attribute__((__vector_size__(8 * sizeof(float)))) float;

union splitkv_wmma_fragment {
  splitkv_bit16x16 full;
  splitkv_bit16x8 half[2];
  uint32_t words[8];
};

union splitkv_packed_fp8 {
  splitkv_bit8x16 bytes;
  uint16_t pairs[8];
};

template <typename scalar_t>
__device__ __forceinline__ splitkv_floatx8 splitkv_wmma(
    splitkv_bit16x16 a, splitkv_bit16x16 b, splitkv_floatx8 c) {
  if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  } else {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  }
}

__device__ __forceinline__ splitkv_floatx8 splitkv_wmma_fp16(
    splitkv_bit16x16 a, splitkv_bit16x16 b, splitkv_floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
#endif

// gfx11 wave32 WMMA specialization for the production 24:4 Qwen GQA shape.
// One wave evaluates a 16-query-head x 16-token score tile. Only the first
// GQA_RATIO rows are retained; padding the remaining WMMA rows is still much
// cheaper than the scalar head_dim=256 dot products. Physical pages remain a
// runtime addressing concern: each of the 16 logical tokens is translated to
// (physical page, offset) independently before the matrix fragment is loaded.
template <typename scalar_t, bool FP8_KV, int GQA_RATIO>
__global__ __launch_bounds__(HEAD_SIZE, 2)
void paged_attention_splitkv_stage1_wmma_kernel(
    float* __restrict__ partial_out, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, const scalar_t* __restrict__ query,
    const void* __restrict__ key_cache, const void* __restrict__ value_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, int num_query_heads,
    int physical_page_size, int num_splits, int64_t query_stride_0,
    int64_t query_stride_1, int64_t key_stride_0, int64_t key_stride_1,
    int64_t key_stride_2, int64_t key_stride_3, int64_t key_stride_4,
    int64_t value_stride_0, int64_t value_stride_1,
    int64_t value_stride_2, int64_t value_stride_3,
    int64_t block_table_stride, int key_vec_size, float softmax_scale) {
#if defined(__GFX11__)
  const int seq_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int dim = threadIdx.x;

  if (query_start_loc != nullptr &&
      query_start_loc[seq_idx + 1] - query_start_loc[seq_idx] != 1) {
    return;
  }

  const int query_idx =
      query_start_loc == nullptr ? seq_idx : query_start_loc[seq_idx];
  const int seq_len = seq_lens[seq_idx];
  const int split_len =
      ((seq_len + num_splits - 1) / num_splits +
       WMMA_LOGICAL_TILE_SIZE - 1) /
      WMMA_LOGICAL_TILE_SIZE * WMMA_LOGICAL_TILE_SIZE;
  const int split_start = split_idx * split_len;
  const int split_end = min(split_start + split_len, seq_len);
  const int query_head_start = kv_head_idx * GQA_RATIO;
  const float key_scale_value = FP8_KV ? *k_scale : 1.0f;
  const float value_scale_value = FP8_KV ? *v_scale : 1.0f;

  if (split_start >= split_end) {
#pragma unroll
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      const int64_t partial_idx =
          (static_cast<int64_t>(seq_idx) * num_query_heads +
           query_head_start + gqa_idx) *
              num_splits +
          split_idx;
      partial_out[partial_idx * HEAD_SIZE + dim] = 0.0f;
      if (dim == 0) {
        partial_max[partial_idx] = -INFINITY;
        partial_sum[partial_idx] = 0.0f;
      }
    }
    return;
  }

  __shared__ float scores[GQA_RATIO][WMMA_LOGICAL_TILE_SIZE];
  __shared__ float weights[GQA_RATIO][WMMA_LOGICAL_TILE_SIZE];
  __shared__ float running_max[GQA_RATIO];
  __shared__ float running_sum[GQA_RATIO];
  __shared__ float previous_scale[GQA_RATIO];
  using value_t = std::conditional_t<FP8_KV, float, scalar_t>;
  __shared__ value_t value_tile[HEAD_SIZE][WMMA_LOGICAL_TILE_SIZE];
  constexpr int FP8_QUERY_STAGE_SIZE = FP8_KV ? GQA_RATIO * HEAD_SIZE : 1;
  constexpr int FP8_KEY_STAGE_SIZE =
      FP8_KV ? WMMA_LOGICAL_TILE_SIZE * HEAD_SIZE : 1;
  __shared__ _Float16 fp8_query_tile[FP8_QUERY_STAGE_SIZE];
  __shared__ _Float16 fp8_key_tile[FP8_KEY_STAGE_SIZE];
  __shared__ int physical_pages[WMMA_LOGICAL_TILE_SIZE];
  __shared__ int page_offsets[WMMA_LOGICAL_TILE_SIZE];

  float output_acc[GQA_RATIO] = {};
  if (dim < GQA_RATIO) {
    running_max[dim] = -INFINITY;
    running_sum[dim] = 0.0f;
  }
  if constexpr (FP8_KV) {
#pragma unroll
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      fp8_query_tile[gqa_idx * HEAD_SIZE + dim] =
          static_cast<_Float16>(to_float(
              query[static_cast<int64_t>(query_idx) * query_stride_0 +
                    static_cast<int64_t>(query_head_start + gqa_idx) *
                        query_stride_1 +
                    dim]));
    }
  }
  __syncthreads();

  for (int tile_start = split_start; tile_start < split_end;
       tile_start += WMMA_LOGICAL_TILE_SIZE) {
    const int tile_tokens =
        min(WMMA_LOGICAL_TILE_SIZE, split_end - tile_start);
    if (dim < WMMA_LOGICAL_TILE_SIZE) {
      const int token_idx = tile_start + min(dim, tile_tokens - 1);
      const int logical_page_idx = token_idx / physical_page_size;
      page_offsets[dim] = token_idx % physical_page_size;
      physical_pages[dim] =
          block_tables[static_cast<int64_t>(seq_idx) * block_table_stride +
                       logical_page_idx];
    }
    __syncthreads();

    if constexpr (FP8_KV) {
      const int token_lane = dim % WMMA_LOGICAL_TILE_SIZE;
      const int dim_group = dim / WMMA_LOGICAL_TILE_SIZE;
      if (token_lane < tile_tokens) {
        const int physical_page_idx = physical_pages[token_lane];
        const int page_offset = page_offsets[token_lane];
        const int k_base = dim_group * WMMA_LOGICAL_TILE_SIZE;
        const int64_t key_base =
            static_cast<int64_t>(physical_page_idx) * key_stride_0 +
            static_cast<int64_t>(kv_head_idx) * key_stride_1 +
            static_cast<int64_t>(k_base / key_vec_size) * key_stride_2 +
            static_cast<int64_t>(page_offset) * key_stride_3;
        splitkv_packed_fp8 packed;
        packed.bytes = *reinterpret_cast<const splitkv_bit8x16*>(
            static_cast<const uint8_t*>(key_cache) + key_base);
        uint32_t* destination = reinterpret_cast<uint32_t*>(
            fp8_key_tile + token_lane * HEAD_SIZE + k_base);
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
          destination[pair] =
              vllm::fp8::scaled_convert<
                  uint32_t, uint16_t,
                  vllm::Fp8KVCacheDataType::kFp8E4M3>(packed.pairs[pair],
                                                      key_scale_value);
        }
      }
      __syncthreads();
    }

    if (dim < 32) {
      const int lane = dim;
      const int lane_lo = lane & 15;
      const int lane_hi = lane >> 4;
      splitkv_floatx8 accum = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
      for (int k_base = 0; k_base < HEAD_SIZE; k_base += 16) {
        splitkv_wmma_fragment q_fragment;
        splitkv_wmma_fragment k_fragment;

        if (lane_lo < GQA_RATIO) {
          if constexpr (FP8_KV) {
            q_fragment.full =
                *reinterpret_cast<const splitkv_bit16x16*>(
                    fp8_query_tile + lane_lo * HEAD_SIZE + k_base);
          } else {
            const scalar_t* q_ptr =
                query + static_cast<int64_t>(query_idx) * query_stride_0 +
                static_cast<int64_t>(query_head_start + lane_lo) *
                    query_stride_1 +
                k_base;
            q_fragment.full =
                *reinterpret_cast<const splitkv_bit16x16*>(q_ptr);
          }
        } else {
          q_fragment.full = {};
        }

        if (lane_lo < tile_tokens) {
          if constexpr (FP8_KV) {
            k_fragment.full =
                *reinterpret_cast<const splitkv_bit16x16*>(
                    fp8_key_tile + lane_lo * HEAD_SIZE + k_base);
          } else {
            const int physical_page_idx = physical_pages[lane_lo];
            const int page_offset = page_offsets[lane_lo];
            const int64_t key_base =
                static_cast<int64_t>(physical_page_idx) * key_stride_0 +
                static_cast<int64_t>(kv_head_idx) * key_stride_1 +
                static_cast<int64_t>(k_base / key_vec_size) * key_stride_2 +
                static_cast<int64_t>(page_offset) * key_stride_3;
            k_fragment.half[0] =
                *(reinterpret_cast<const splitkv_bit16x8*>(
                    static_cast<const scalar_t*>(key_cache) + key_base));
            const int64_t second_half =
                static_cast<int64_t>(8 / key_vec_size) * key_stride_2 +
                static_cast<int64_t>(8 % key_vec_size) * key_stride_4;
            k_fragment.half[1] =
                *(reinterpret_cast<const splitkv_bit16x8*>(
                      static_cast<const scalar_t*>(key_cache) + key_base +
                      second_half));
          }
        } else {
          k_fragment.full = {};
        }
        if constexpr (FP8_KV) {
          accum = splitkv_wmma_fp16(q_fragment.full, k_fragment.full, accum);
        } else {
          accum = splitkv_wmma<scalar_t>(q_fragment.full, k_fragment.full,
                                         accum);
        }
      }

#pragma unroll
      for (int slot = 0; slot < 8; ++slot) {
        const int gqa_idx = 2 * slot + lane_hi;
        if (gqa_idx < GQA_RATIO && lane_lo < tile_tokens) {
          scores[gqa_idx][lane_lo] = accum[slot] * softmax_scale;
        }
      }
    }
    __syncthreads();

    if (dim < GQA_RATIO) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int token = 0; token < WMMA_LOGICAL_TILE_SIZE; ++token) {
        if (token < tile_tokens) {
          tile_max = fmaxf(tile_max, scores[dim][token]);
        }
      }
      const float next_max = fmaxf(running_max[dim], tile_max);
      const float alpha = expf(running_max[dim] - next_max);
      float next_sum = running_sum[dim] * alpha;
#pragma unroll
      for (int token = 0; token < WMMA_LOGICAL_TILE_SIZE; ++token) {
        if (token < tile_tokens) {
          const float weight = expf(scores[dim][token] - next_max);
          weights[dim][token] = weight;
          next_sum += weight;
        }
      }
      previous_scale[dim] = alpha;
      running_max[dim] = next_max;
      running_sum[dim] = next_sum;
    }
    __syncthreads();

    if constexpr (FP8_KV) {
      const bool contiguous_tile =
          tile_tokens == WMMA_LOGICAL_TILE_SIZE && value_stride_3 == 1 &&
          physical_pages[0] ==
              physical_pages[WMMA_LOGICAL_TILE_SIZE - 1] &&
          page_offsets[WMMA_LOGICAL_TILE_SIZE - 1] ==
              page_offsets[0] + WMMA_LOGICAL_TILE_SIZE - 1;
      if (contiguous_tile) {
        const int64_t value_offset =
            static_cast<int64_t>(physical_pages[0]) * value_stride_0 +
            static_cast<int64_t>(kv_head_idx) * value_stride_1 +
            static_cast<int64_t>(dim) * value_stride_2 +
            static_cast<int64_t>(page_offsets[0]);
        splitkv_packed_fp8 packed;
        packed.bytes = *reinterpret_cast<const splitkv_bit8x16*>(
            static_cast<const uint8_t*>(value_cache) + value_offset);
        float2* destination =
            reinterpret_cast<float2*>(&value_tile[dim][0]);
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
          destination[pair] =
              vllm::fp8::scaled_convert<
                  float2, uint16_t,
                  vllm::Fp8KVCacheDataType::kFp8E4M3>(packed.pairs[pair],
                                                      value_scale_value);
        }
      } else {
#pragma unroll
        for (int token = 0; token < WMMA_LOGICAL_TILE_SIZE; ++token) {
          if (token < tile_tokens) {
            const int64_t value_offset =
                static_cast<int64_t>(physical_pages[token]) * value_stride_0 +
                static_cast<int64_t>(kv_head_idx) * value_stride_1 +
                static_cast<int64_t>(dim) * value_stride_2 +
                static_cast<int64_t>(page_offsets[token]) * value_stride_3;
            value_tile[dim][token] = load_kv<scalar_t, true>(
                value_cache, value_offset, value_scale_value);
          }
        }
      }
    } else {
      const int token_lane = dim % WMMA_LOGICAL_TILE_SIZE;
      const int dim_group = dim / WMMA_LOGICAL_TILE_SIZE;
      if (token_lane < tile_tokens) {
        const int physical_page_idx = physical_pages[token_lane];
        const int page_offset = page_offsets[token_lane];
        const int head_start = dim_group * WMMA_LOGICAL_TILE_SIZE;
#pragma unroll
        for (int head_offset = head_start;
             head_offset < head_start + WMMA_LOGICAL_TILE_SIZE;
             ++head_offset) {
          const int64_t value_offset =
              static_cast<int64_t>(physical_page_idx) * value_stride_0 +
              static_cast<int64_t>(kv_head_idx) * value_stride_1 +
              static_cast<int64_t>(head_offset) * value_stride_2 +
              static_cast<int64_t>(page_offset) * value_stride_3;
          value_tile[head_offset][token_lane] =
              static_cast<const scalar_t*>(value_cache)[value_offset];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      output_acc[gqa_idx] *= previous_scale[gqa_idx];
    }
#pragma unroll
    for (int token = 0; token < WMMA_LOGICAL_TILE_SIZE; ++token) {
      if (token < tile_tokens) {
        float value;
        if constexpr (FP8_KV) {
          value = value_tile[dim][token];
        } else {
          value = to_float(value_tile[dim][token]);
        }
#pragma unroll
        for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
          output_acc[gqa_idx] += value * weights[gqa_idx][token];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
    const int64_t partial_idx =
        (static_cast<int64_t>(seq_idx) * num_query_heads + query_head_start +
         gqa_idx) *
            num_splits +
        split_idx;
    partial_out[partial_idx * HEAD_SIZE + dim] = output_acc[gqa_idx];
    if (dim == 0) {
      partial_max[partial_idx] = running_max[gqa_idx];
      partial_sum[partial_idx] = running_sum[gqa_idx];
    }
  }
#endif
}

template <typename scalar_t, bool FP8_KV, int GQA_RATIO>
__global__ void paged_attention_splitkv_stage1_kernel(
    float* __restrict__ partial_out, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, const scalar_t* __restrict__ query,
    const void* __restrict__ key_cache, const void* __restrict__ value_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, int num_query_heads, int num_kv_heads,
    int physical_page_size, int num_splits, int64_t query_stride_0,
    int64_t query_stride_1, int64_t key_stride_0, int64_t key_stride_1,
    int64_t key_stride_2, int64_t key_stride_3, int64_t key_stride_4,
    int64_t value_stride_0, int64_t value_stride_1,
    int64_t value_stride_2, int64_t value_stride_3,
    int64_t block_table_stride, int key_vec_size, float softmax_scale) {
  const int seq_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int dim = threadIdx.x;

  if (query_start_loc != nullptr &&
      query_start_loc[seq_idx + 1] - query_start_loc[seq_idx] != 1) {
    return;
  }

  const int query_idx =
      query_start_loc == nullptr ? seq_idx : query_start_loc[seq_idx];
  const int seq_len = seq_lens[seq_idx];
  const int split_len =
      ((seq_len + num_splits - 1) / num_splits + LOGICAL_TILE_SIZE - 1) /
      LOGICAL_TILE_SIZE * LOGICAL_TILE_SIZE;
  const int split_start = split_idx * split_len;
  const int split_end = min(split_start + split_len, seq_len);
  const int query_head_start = kv_head_idx * GQA_RATIO;

  if (split_start >= split_end) {
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      const int64_t partial_idx =
          (static_cast<int64_t>(seq_idx) * num_query_heads +
           query_head_start + gqa_idx) *
              num_splits +
          split_idx;
      partial_out[partial_idx * HEAD_SIZE + dim] = 0.0f;
      if (dim == 0) {
        partial_max[partial_idx] = -INFINITY;
        partial_sum[partial_idx] = 0.0f;
      }
    }
    return;
  }

  const float key_scale_value = FP8_KV ? *k_scale : 1.0f;
  const float value_scale_value = FP8_KV ? *v_scale : 1.0f;
  __shared__ float score_partial[GQA_RATIO][HEAD_SIZE / LOGICAL_TILE_SIZE]
                                [LOGICAL_TILE_SIZE];
  __shared__ float weights[GQA_RATIO][LOGICAL_TILE_SIZE];
  __shared__ float running_max[GQA_RATIO];
  __shared__ float running_sum[GQA_RATIO];
  __shared__ float previous_scale[GQA_RATIO];
  __shared__ float value_tile[HEAD_SIZE][LOGICAL_TILE_SIZE];
  __shared__ int physical_pages[LOGICAL_TILE_SIZE];
  __shared__ int page_offsets[LOGICAL_TILE_SIZE];
  float output_acc[GQA_RATIO];
  if (dim < GQA_RATIO) {
    running_max[dim] = -INFINITY;
    running_sum[dim] = 0.0f;
  }
#pragma unroll
  for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
    output_acc[gqa_idx] = 0.0f;
  }
  __syncthreads();

  for (int tile_start = split_start; tile_start < split_end;
       tile_start += LOGICAL_TILE_SIZE) {
    const int tile_tokens = min(LOGICAL_TILE_SIZE, split_end - tile_start);
    const int token_lane = dim % LOGICAL_TILE_SIZE;
    const int dim_group = dim / LOGICAL_TILE_SIZE;

    // A logical tile can straddle a physical page. Translate every token
    // independently; the physical page itself is never staged in LDS.
    if (dim < LOGICAL_TILE_SIZE && dim < tile_tokens) {
      const int token_idx = tile_start + dim;
      const int logical_page_idx = token_idx / physical_page_size;
      page_offsets[dim] = token_idx % physical_page_size;
      physical_pages[dim] =
          block_tables[static_cast<int64_t>(seq_idx) * block_table_stride +
                       logical_page_idx];
    }
    __syncthreads();

    float qk[GQA_RATIO] = {};
    if (token_lane < tile_tokens) {
      const int physical_page_idx = physical_pages[token_lane];
      const int page_offset = page_offsets[token_lane];
      const int head_start = dim_group * LOGICAL_TILE_SIZE;
      for (int head_offset = head_start;
           head_offset < head_start + LOGICAL_TILE_SIZE; ++head_offset) {
        const int64_t key_offset =
            static_cast<int64_t>(physical_page_idx) * key_stride_0 +
            static_cast<int64_t>(kv_head_idx) * key_stride_1 +
            static_cast<int64_t>(head_offset / key_vec_size) * key_stride_2 +
            static_cast<int64_t>(page_offset) * key_stride_3 +
            static_cast<int64_t>(head_offset % key_vec_size) * key_stride_4;
        const float key = load_kv<scalar_t, FP8_KV>(
            key_cache, key_offset, key_scale_value);
#pragma unroll
        for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
          const int query_head_idx = query_head_start + gqa_idx;
          const float query_value = to_float(
              query[static_cast<int64_t>(query_idx) * query_stride_0 +
                    static_cast<int64_t>(query_head_idx) * query_stride_1 +
                    head_offset]);
          qk[gqa_idx] += query_value * key;
        }
      }
    }
#pragma unroll
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      score_partial[gqa_idx][dim_group][token_lane] = qk[gqa_idx];
    }
    __syncthreads();

    if (dim_group == 0) {
#pragma unroll
      for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
        float score = 0.0f;
#pragma unroll
        for (int group = 0; group < HEAD_SIZE / LOGICAL_TILE_SIZE; ++group) {
          score += score_partial[gqa_idx][group][token_lane];
        }
        score_partial[gqa_idx][0][token_lane] = score * softmax_scale;
      }
    }
    __syncthreads();

    if (dim < GQA_RATIO) {
      float tile_max = -INFINITY;
      for (int token = 0; token < tile_tokens; ++token) {
        tile_max = fmaxf(tile_max, score_partial[dim][0][token]);
      }
      const float next_max = fmaxf(running_max[dim], tile_max);
      const float alpha = expf(running_max[dim] - next_max);
      float next_sum = running_sum[dim] * alpha;
      for (int token = 0; token < tile_tokens; ++token) {
        const float weight =
            expf(score_partial[dim][0][token] - next_max);
        weights[dim][token] = weight;
        next_sum += weight;
      }
      previous_scale[dim] = alpha;
      running_max[dim] = next_max;
      running_sum[dim] = next_sum;
    }
    __syncthreads();

    if (token_lane < tile_tokens) {
      const int physical_page_idx = physical_pages[token_lane];
      const int page_offset = page_offsets[token_lane];
      const int head_start = dim_group * LOGICAL_TILE_SIZE;
      for (int head_offset = head_start;
           head_offset < head_start + LOGICAL_TILE_SIZE; ++head_offset) {
        const int64_t value_offset =
            static_cast<int64_t>(physical_page_idx) * value_stride_0 +
            static_cast<int64_t>(kv_head_idx) * value_stride_1 +
            static_cast<int64_t>(head_offset) * value_stride_2 +
            static_cast<int64_t>(page_offset) * value_stride_3;
        value_tile[head_offset][token_lane] = load_kv<scalar_t, FP8_KV>(
            value_cache, value_offset, value_scale_value);
      }
    }
    __syncthreads();

#pragma unroll
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
      output_acc[gqa_idx] *= previous_scale[gqa_idx];
    }
    for (int token = 0; token < tile_tokens; ++token) {
      const float value = value_tile[dim][token];
#pragma unroll
      for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
        output_acc[gqa_idx] += value * weights[gqa_idx][token];
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int gqa_idx = 0; gqa_idx < GQA_RATIO; ++gqa_idx) {
    const int64_t partial_idx =
        (static_cast<int64_t>(seq_idx) * num_query_heads + query_head_start +
         gqa_idx) *
            num_splits +
        split_idx;
    partial_out[partial_idx * HEAD_SIZE + dim] = output_acc[gqa_idx];
    if (dim == 0) {
      partial_max[partial_idx] = running_max[gqa_idx];
      partial_sum[partial_idx] = running_sum[gqa_idx];
    }
  }
}

template <typename scalar_t>
__global__ void paged_attention_splitkv_reduce_kernel(
    scalar_t* __restrict__ output, const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, int num_query_heads,
    int num_splits, int64_t output_stride_0, int64_t output_stride_1) {
  const int seq_idx = blockIdx.x;
  const int query_head_idx = blockIdx.y;
  const int dim = threadIdx.x;
  if (query_start_loc != nullptr &&
      query_start_loc[seq_idx + 1] - query_start_loc[seq_idx] != 1) {
    return;
  }
  const int query_idx =
      query_start_loc == nullptr ? seq_idx : query_start_loc[seq_idx];
  const int64_t partial_base =
      (static_cast<int64_t>(seq_idx) * num_query_heads + query_head_idx) *
      num_splits;

  float global_max = -INFINITY;
  for (int split_idx = 0; split_idx < num_splits; ++split_idx) {
    global_max = fmaxf(global_max, partial_max[partial_base + split_idx]);
  }

  float global_sum = 0.0f;
  float output_acc = 0.0f;
  for (int split_idx = 0; split_idx < num_splits; ++split_idx) {
    const int64_t partial_idx = partial_base + split_idx;
    const float split_weight = expf(partial_max[partial_idx] - global_max);
    global_sum += partial_sum[partial_idx] * split_weight;
    output_acc += partial_out[partial_idx * HEAD_SIZE + dim] * split_weight;
  }
  output[static_cast<int64_t>(query_idx) * output_stride_0 +
         static_cast<int64_t>(query_head_idx) * output_stride_1 + dim] =
      from_float<scalar_t>(output_acc / global_sum);
}

template <typename scalar_t, bool FP8_KV>
void launch_splitkv(
    torch::Tensor& output, torch::Tensor& partial_out,
    torch::Tensor& partial_max, torch::Tensor& partial_sum,
    torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float softmax_scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc,
    int physical_page_size, torch::Tensor& k_scale, torch::Tensor& v_scale) {
  const int num_seqs = block_tables.size(0);
  const int num_query_heads = query.size(1);
  const int gqa_ratio = num_query_heads / num_kv_heads;
  const int num_splits = partial_max.size(2);
  const int* query_start_ptr =
      query_start_loc ? query_start_loc->data_ptr<int>() : nullptr;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (gqa_ratio == 4) {
    paged_attention_splitkv_stage1_kernel<scalar_t, FP8_KV, 4>
        <<<dim3(num_seqs, num_kv_heads, num_splits), dim3(HEAD_SIZE), 0,
           stream>>>(
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            key_cache.data_ptr(), value_cache.data_ptr(),
            block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
            query_start_ptr, k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(), num_query_heads, num_kv_heads,
            physical_page_size, num_splits, query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),
            value_cache.stride(1), value_cache.stride(2),
            value_cache.stride(3), block_tables.stride(0), key_cache.size(4),
            softmax_scale);
  } else if (gqa_ratio == 6) {
    bool launched_wmma = false;
    if constexpr (!FP8_KV ||
                  std::is_same_v<scalar_t, __hip_bfloat16>) {
      if (current_device_is_gfx11() && key_cache.stride(4) == 1) {
        paged_attention_splitkv_stage1_wmma_kernel<scalar_t, FP8_KV, 6>
            <<<dim3(num_seqs, num_kv_heads, num_splits), dim3(HEAD_SIZE), 0,
               stream>>>(
                partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr(), value_cache.data_ptr(),
                block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
                query_start_ptr, k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(), num_query_heads,
                physical_page_size, num_splits, query.stride(0),
                query.stride(1), key_cache.stride(0), key_cache.stride(1),
                key_cache.stride(2), key_cache.stride(3), key_cache.stride(4),
                value_cache.stride(0), value_cache.stride(1),
                value_cache.stride(2), value_cache.stride(3),
                block_tables.stride(0), key_cache.size(4), softmax_scale);
        launched_wmma = true;
      }
    }
    if (!launched_wmma) {
      paged_attention_splitkv_stage1_kernel<scalar_t, FP8_KV, 6>
          <<<dim3(num_seqs, num_kv_heads, num_splits), dim3(HEAD_SIZE), 0,
             stream>>>(
              partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
              partial_sum.data_ptr<float>(),
              reinterpret_cast<const scalar_t*>(query.data_ptr()),
              key_cache.data_ptr(), value_cache.data_ptr(),
              block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
              query_start_ptr, k_scale.data_ptr<float>(),
              v_scale.data_ptr<float>(), num_query_heads, num_kv_heads,
              physical_page_size, num_splits, query.stride(0), query.stride(1),
              key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
              key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),
              value_cache.stride(1), value_cache.stride(2),
              value_cache.stride(3), block_tables.stride(0), key_cache.size(4),
              softmax_scale);
    }
  } else {
    TORCH_CHECK(false, "native split-KV requires GQA ratio 4 or 6");
  }
  paged_attention_splitkv_reduce_kernel<scalar_t>
      <<<dim3(num_seqs, num_query_heads), dim3(HEAD_SIZE), 0, stream>>>(
          reinterpret_cast<scalar_t*>(output.data_ptr()),
          partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
          partial_sum.data_ptr<float>(), seq_lens.data_ptr<int>(),
          query_start_ptr, num_query_heads, num_splits, output.stride(0),
          output.stride(1));
}

}  // namespace

void paged_attention_splitkv(
    torch::Tensor& output, torch::Tensor& partial_out,
    torch::Tensor& partial_max, torch::Tensor& partial_sum,
    torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double softmax_scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc,
    int64_t physical_page_size, const std::string& kv_cache_dtype,
    torch::Tensor& k_scale, torch::Tensor& v_scale) {
  TORCH_CHECK(query.is_cuda(), "query must be on a GPU");
  TORCH_CHECK(query.size(2) == HEAD_SIZE,
              "native split-KV requires head_dim=256");
  TORCH_CHECK(query.size(1) % num_kv_heads == 0, "invalid GQA ratio");
  TORCH_CHECK(physical_page_size == key_cache.size(3),
              "physical page size does not match key cache");
  TORCH_CHECK(partial_out.scalar_type() == at::kFloat &&
                  partial_max.scalar_type() == at::kFloat &&
                  partial_sum.scalar_type() == at::kFloat,
              "split-KV partial tensors must be float32");
  TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "fp8" ||
                  kv_cache_dtype == "fp8_e4m3",
              "native split-KV supports auto and fp8_e4m3 KV cache");
  const bool fp8_kv = kv_cache_dtype != "auto";
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  if (query.scalar_type() == at::kBFloat16) {
    if (fp8_kv) {
      launch_splitkv<__hip_bfloat16, true>(
          output, partial_out, partial_max, partial_sum, query, key_cache,
          value_cache, num_kv_heads, softmax_scale, block_tables, seq_lens,
          query_start_loc, physical_page_size, k_scale, v_scale);
    } else {
      launch_splitkv<__hip_bfloat16, false>(
          output, partial_out, partial_max, partial_sum, query, key_cache,
          value_cache, num_kv_heads, softmax_scale, block_tables, seq_lens,
          query_start_loc, physical_page_size, k_scale, v_scale);
    }
  } else if (query.scalar_type() == at::kHalf) {
    if (fp8_kv) {
      launch_splitkv<_Float16, true>(
          output, partial_out, partial_max, partial_sum, query, key_cache,
          value_cache, num_kv_heads, softmax_scale, block_tables, seq_lens,
          query_start_loc, physical_page_size, k_scale, v_scale);
    } else {
      launch_splitkv<_Float16, false>(
          output, partial_out, partial_max, partial_sum, query, key_cache,
          value_cache, num_kv_heads, softmax_scale, block_tables, seq_lens,
          query_start_loc, physical_page_size, k_scale, v_scale);
    }
  } else {
    TORCH_CHECK(false, "native split-KV requires BF16 or FP16 query");
  }
}
