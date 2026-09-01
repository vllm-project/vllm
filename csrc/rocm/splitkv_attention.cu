// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

#include "../attention/dtype_fp8.cuh"
#include "../cuda_compat.h"
#include "../quantization/w8a8/fp8/amd/quant_utils.cuh"

namespace {

constexpr int HEAD_SIZE = 256;
constexpr int LOGICAL_TILE_SIZE = 32;

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
