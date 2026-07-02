// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "libtorch_stable/torch_utils.h"

namespace {

constexpr int64_t kModeMaterializedWarpStats = 16;
constexpr int kWarpStatsReduceThreads = 1024;

inline void check_cublas(cublasStatus_t status, const char* label) {
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                  std::string("diffusion_gemma_flashdenoise: ") + label +
                      " failed with cuBLAS status " +
                      std::to_string(static_cast<int>(status)));
}

__device__ __forceinline__ bool better_pair(float candidate_value,
                                            int candidate_index,
                                            float current_value,
                                            int current_index) {
  return candidate_value > current_value ||
         (candidate_value == current_value && candidate_index < current_index);
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float deterministic_uniform01(
    uint64_t seed, uint64_t offset, int row, int64_t vocab_idx) {
  uint64_t key = seed;
  key ^= offset * 0x9E3779B97F4A7C15ULL;
  key ^= static_cast<uint64_t>(row) * 0xBF58476D1CE4E5B9ULL;
  key ^= static_cast<uint64_t>(vocab_idx) * 0x94D049BB133111EBULL;
  const uint64_t z = splitmix64(key);
  const uint32_t mantissa = static_cast<uint32_t>((z >> 40) & 0xFFFFFFULL);
  return (static_cast<float>(mantissa) + 0.5f) * 0x1.0p-24f;
}

__device__ __forceinline__ float apply_final_logit_softcap(float value,
                                                           float softcap) {
  return softcap > 0.0f ? softcap * tanhf(value / softcap) : value;
}

__device__ __forceinline__ void warp_reduce_best_pair(float& value,
                                                      int& index) {
  constexpr unsigned kMask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(kMask, value, offset);
    const int other_index = __shfl_down_sync(kMask, index, offset);
    if (better_pair(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  constexpr unsigned kMask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kMask, value, offset);
  }
  return value;
}

__global__ void logits_to_stats_probs_warp_kernel(
    const float* __restrict__ logits, __nv_bfloat16* __restrict__ probs,
    const float* __restrict__ logit_scale, float final_logit_softcapping,
    float* __restrict__ entropy, float* __restrict__ sample_values,
    int64_t* __restrict__ sample_indices, float* __restrict__ clean_values,
    int64_t* __restrict__ clean_indices, int rows, int vocab_size,
    uint64_t rng_seed, uint64_t rng_offset, int rng_row_offset) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  if (row >= rows) {
    return;
  }

  __shared__ float shared_values[32];
  __shared__ int shared_indices[32];
  __shared__ float shared_sums[32];
  __shared__ float shared_weighted[32];

  const float* row_logits = logits + static_cast<int64_t>(row) * vocab_size;
  const float row_scale = logit_scale == nullptr ? 1.0f : logit_scale[row];
  __nv_bfloat16* row_probs =
      probs + static_cast<int64_t>(row) * vocab_size;

  float local_max = -INFINITY;
  int local_max_idx = 0;
  for (int v = tid; v < vocab_size; v += blockDim.x) {
    const float value =
        apply_final_logit_softcap(row_logits[v], final_logit_softcapping) *
        row_scale;
    if (better_pair(value, v, local_max, local_max_idx)) {
      local_max = value;
      local_max_idx = v;
    }
  }
  warp_reduce_best_pair(local_max, local_max_idx);
  if (lane == 0) {
    shared_values[warp_id] = local_max;
    shared_indices[warp_id] = local_max_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane < num_warps ? shared_values[lane] : -INFINITY;
    int block_idx = lane < num_warps ? shared_indices[lane] : 0;
    warp_reduce_best_pair(block_max, block_idx);
    if (lane == 0) {
      shared_values[0] = block_max;
      shared_indices[0] = block_idx;
    }
  }
  __syncthreads();

  const float max_logit = shared_values[0];
  const int max_idx = shared_indices[0];

  float local_sum_exp = 0.0f;
  float local_weighted_logits = 0.0f;
  float local_noisy_max = -INFINITY;
  int local_noisy_idx = 0;
  for (int v = tid; v < vocab_size; v += blockDim.x) {
    const float value =
        apply_final_logit_softcap(row_logits[v], final_logit_softcapping) *
        row_scale;
    const float exp_value = expf(value - max_logit);
    local_sum_exp += exp_value;
    local_weighted_logits += exp_value * value;

    const float u =
        deterministic_uniform01(rng_seed, rng_offset, row + rng_row_offset, v);
    const float noisy_value = value + (-logf(-logf(u)));
    if (better_pair(noisy_value, v, local_noisy_max, local_noisy_idx)) {
      local_noisy_max = noisy_value;
      local_noisy_idx = v;
    }
  }

  local_sum_exp = warp_reduce_sum(local_sum_exp);
  local_weighted_logits = warp_reduce_sum(local_weighted_logits);
  warp_reduce_best_pair(local_noisy_max, local_noisy_idx);
  if (lane == 0) {
    shared_sums[warp_id] = local_sum_exp;
    shared_weighted[warp_id] = local_weighted_logits;
    shared_values[warp_id] = local_noisy_max;
    shared_indices[warp_id] = local_noisy_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < num_warps ? shared_sums[lane] : 0.0f;
    float block_weighted = lane < num_warps ? shared_weighted[lane] : 0.0f;
    float block_noisy = lane < num_warps ? shared_values[lane] : -INFINITY;
    int block_noisy_idx = lane < num_warps ? shared_indices[lane] : 0;
    block_sum = warp_reduce_sum(block_sum);
    block_weighted = warp_reduce_sum(block_weighted);
    warp_reduce_best_pair(block_noisy, block_noisy_idx);
    if (lane == 0) {
      shared_sums[0] = block_sum;
      shared_weighted[0] = block_weighted;
      shared_values[0] = block_noisy;
      shared_indices[0] = block_noisy_idx;
    }
  }
  __syncthreads();

  const float sum_exp = shared_sums[0];
  const float inv_sum_exp = 1.0f / sum_exp;
  if (tid == 0) {
    entropy[row] =
        logf(sum_exp) + max_logit - shared_weighted[0] * inv_sum_exp;
    sample_values[row] = shared_values[0];
    sample_indices[row] = static_cast<int64_t>(shared_indices[0]);
    clean_values[row] = max_logit;
    clean_indices[row] = static_cast<int64_t>(max_idx);
  }

  for (int v = tid; v < vocab_size; v += blockDim.x) {
    const float value =
        apply_final_logit_softcap(row_logits[v], final_logit_softcapping) *
        row_scale;
    const float prob = expf(value - max_logit) * inv_sum_exp;
    row_probs[v] = __float2bfloat16(prob);
  }
}

__global__ void logits_to_local_state_exp_warp_kernel(
    const float* __restrict__ logits, __nv_bfloat16* __restrict__ exp_weights,
    const float* __restrict__ logit_scale, bool logit_scale_is_scalar,
    float final_logit_softcapping, float* __restrict__ local_max,
    float* __restrict__ local_sum_exp,
    float* __restrict__ local_weighted_logits,
    float* __restrict__ sample_values, int64_t* __restrict__ sample_indices,
    float* __restrict__ clean_values, int64_t* __restrict__ clean_indices,
    int rows, int vocab_size, int64_t vocab_start_index, uint64_t rng_seed,
    uint64_t rng_offset) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  if (row >= rows) {
    return;
  }

  __shared__ float shared_values[32];
  __shared__ int shared_indices[32];
  __shared__ float shared_sums[32];
  __shared__ float shared_weighted[32];

  const float* row_logits = logits + static_cast<int64_t>(row) * vocab_size;
  __nv_bfloat16* row_exp =
      exp_weights + static_cast<int64_t>(row) * vocab_size;
  const float row_scale = logit_scale_is_scalar ? logit_scale[0]
                                                : logit_scale[row];

  float local_clean_max = -INFINITY;
  int local_clean_idx = 0;
  for (int v = tid; v < vocab_size; v += blockDim.x) {
    const float value =
        apply_final_logit_softcap(row_logits[v], final_logit_softcapping) *
        row_scale;
    if (better_pair(value, v, local_clean_max, local_clean_idx)) {
      local_clean_max = value;
      local_clean_idx = v;
    }
  }
  warp_reduce_best_pair(local_clean_max, local_clean_idx);
  if (lane == 0) {
    shared_values[warp_id] = local_clean_max;
    shared_indices[warp_id] = local_clean_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane < num_warps ? shared_values[lane] : -INFINITY;
    int block_idx = lane < num_warps ? shared_indices[lane] : 0;
    warp_reduce_best_pair(block_max, block_idx);
    if (lane == 0) {
      shared_values[0] = block_max;
      shared_indices[0] = block_idx;
    }
  }
  __syncthreads();

  const float max_logit = shared_values[0];
  const int max_idx = shared_indices[0];

  float row_sum_exp = 0.0f;
  float row_weighted_logits = 0.0f;
  float local_noisy_max = -INFINITY;
  int local_noisy_idx = 0;
  for (int v = tid; v < vocab_size; v += blockDim.x) {
    const float value =
        apply_final_logit_softcap(row_logits[v], final_logit_softcapping) *
        row_scale;
    const float exp_value = expf(value - max_logit);
    row_exp[v] = __float2bfloat16(exp_value);
    row_sum_exp += exp_value;
    row_weighted_logits += exp_value * value;

    // MVP sampling uses deterministic SplitMix64 keyed by row and global token.
    const int64_t global_idx = vocab_start_index + static_cast<int64_t>(v);
    const float u = deterministic_uniform01(rng_seed, rng_offset, row,
                                            global_idx);
    const float noisy_value = value + (-logf(-logf(u)));
    if (better_pair(noisy_value, v, local_noisy_max, local_noisy_idx)) {
      local_noisy_max = noisy_value;
      local_noisy_idx = v;
    }
  }

  row_sum_exp = warp_reduce_sum(row_sum_exp);
  row_weighted_logits = warp_reduce_sum(row_weighted_logits);
  warp_reduce_best_pair(local_noisy_max, local_noisy_idx);
  if (lane == 0) {
    shared_sums[warp_id] = row_sum_exp;
    shared_weighted[warp_id] = row_weighted_logits;
    shared_values[warp_id] = local_noisy_max;
    shared_indices[warp_id] = local_noisy_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < num_warps ? shared_sums[lane] : 0.0f;
    float block_weighted = lane < num_warps ? shared_weighted[lane] : 0.0f;
    float block_noisy = lane < num_warps ? shared_values[lane] : -INFINITY;
    int block_noisy_idx = lane < num_warps ? shared_indices[lane] : 0;
    block_sum = warp_reduce_sum(block_sum);
    block_weighted = warp_reduce_sum(block_weighted);
    warp_reduce_best_pair(block_noisy, block_noisy_idx);
    if (lane == 0) {
      shared_sums[0] = block_sum;
      shared_weighted[0] = block_weighted;
      shared_values[0] = block_noisy;
      shared_indices[0] = block_noisy_idx;
    }
  }
  __syncthreads();

  if (tid == 0) {
    local_max[row] = max_logit;
    local_sum_exp[row] = shared_sums[0];
    local_weighted_logits[row] = shared_weighted[0];
    sample_values[row] = shared_values[0];
    sample_indices[row] =
        vocab_start_index + static_cast<int64_t>(shared_indices[0]);
    clean_values[row] = max_logit;
    clean_indices[row] = vocab_start_index + static_cast<int64_t>(max_idx);
  }
}

__global__ void pack_local_state_kernel(
    float* __restrict__ packed, const float* __restrict__ local_max,
    const float* __restrict__ global_max,
    const float* __restrict__ local_sum_exp,
    const float* __restrict__ local_weighted_logits,
    const float* __restrict__ local_soft_part, int rows, int hidden_size) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float merge_scale;
  if (threadIdx.x == 0) {
    merge_scale = expf(local_max[row] - global_max[row]);
  }
  __syncthreads();

  const int columns = hidden_size + 2;
  float* row_packed = packed + static_cast<int64_t>(row) * columns;
  const float* row_soft =
      local_soft_part + static_cast<int64_t>(row) * hidden_size;
  for (int col = threadIdx.x; col < columns; col += blockDim.x) {
    float value;
    if (col == 0) {
      value = local_sum_exp[row];
    } else if (col == 1) {
      value = local_weighted_logits[row];
    } else {
      value = row_soft[col - 2];
    }
    row_packed[col] = value * merge_scale;
  }
}

void validate_flashdenoise_tensors(
    const char* label, torch::stable::Tensor& entropy,
    torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices,
    torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices,
    torch::stable::Tensor& soft_embed,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight,
    const torch::stable::Tensor* logit_scale) {
  STD_TORCH_CHECK(hidden.dim() == 2 && lm_head_weight.dim() == 2,
                  label, ": hidden and lm_head_weight must be rank-2");
  STD_TORCH_CHECK(entropy.dim() == 1 && sample_values.dim() == 1 &&
                      sample_indices.dim() == 1 && clean_values.dim() == 1 &&
                      clean_indices.dim() == 1 && soft_embed.dim() == 2,
                  label, ": output shapes must be [rows] scalars and "
                         "soft_embed=[rows, hidden]");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->dim() == 1,
                    label, ": logit_scale must be rank-1");
  }

  STD_TORCH_CHECK(hidden.is_cuda() && lm_head_weight.is_cuda() &&
                      entropy.is_cuda() && sample_values.is_cuda() &&
                      sample_indices.is_cuda() && clean_values.is_cuda() &&
                      clean_indices.is_cuda() && soft_embed.is_cuda(),
                  label, ": all tensors must be CUDA tensors");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->is_cuda(),
                    label, ": logit_scale must be a CUDA tensor");
  }

  STD_TORCH_CHECK(hidden.is_contiguous() && lm_head_weight.is_contiguous() &&
                      entropy.is_contiguous() && sample_values.is_contiguous() &&
                      sample_indices.is_contiguous() &&
                      clean_values.is_contiguous() &&
                      clean_indices.is_contiguous() && soft_embed.is_contiguous(),
                  label, ": all tensors must be contiguous");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->is_contiguous(),
                    label, ": logit_scale must be contiguous");
  }

  const int64_t rows = hidden.size(0);
  const int64_t hidden_size = hidden.size(1);
  const int64_t vocab_size = lm_head_weight.size(0);
  STD_TORCH_CHECK(lm_head_weight.size(1) == hidden_size,
                  label, ": hidden size mismatch");
  STD_TORCH_CHECK(entropy.size(0) == rows && sample_values.size(0) == rows &&
                      sample_indices.size(0) == rows &&
                      clean_values.size(0) == rows &&
                      clean_indices.size(0) == rows &&
                      soft_embed.size(0) == rows &&
                      soft_embed.size(1) == hidden_size,
                  label, ": output shapes do not match hidden");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->size(0) == rows,
                    label, ": logit_scale must have one value per row");
  }

  STD_TORCH_CHECK(vocab_size > 0 && hidden_size > 0,
                  label, ": vocab and hidden sizes must be positive");
  STD_TORCH_CHECK(vocab_size <= std::numeric_limits<int>::max() &&
                      hidden_size <= std::numeric_limits<int>::max() &&
                      rows <= std::numeric_limits<int>::max(),
                  label, ": shapes exceed CUDA kernel int32 limits");
  STD_TORCH_CHECK(vocab_size % 8 == 0 && hidden_size % 8 == 0,
                  label, ": vocab and hidden sizes must be divisible by 8");
  STD_TORCH_CHECK(rows <= 1024 && vocab_size <= 262144 && hidden_size <= 4096,
                  label, ": mode16 supports rows<=1024, vocab<=262144, "
                         "hidden<=4096");

  STD_TORCH_CHECK(
      hidden.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          lm_head_weight.scalar_type() ==
              torch::headeronly::ScalarType::BFloat16,
      label, ": hidden and lm_head_weight must be bf16");
  STD_TORCH_CHECK(entropy.scalar_type() == torch::headeronly::ScalarType::Float &&
                      sample_values.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      clean_values.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      soft_embed.scalar_type() ==
                          torch::headeronly::ScalarType::Float,
                  label, ": entropy, sample_values, clean_values, and "
                         "soft_embed must be fp32");
  STD_TORCH_CHECK(
      sample_indices.scalar_type() == torch::headeronly::ScalarType::Long &&
          clean_indices.scalar_type() == torch::headeronly::ScalarType::Long,
      label, ": sample_indices and clean_indices must be int64");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->scalar_type() ==
                        torch::headeronly::ScalarType::Float,
                    label, ": logit_scale must be fp32");
  }

  const int32_t device_index = hidden.get_device_index();
  STD_TORCH_CHECK(lm_head_weight.get_device_index() == device_index &&
                      entropy.get_device_index() == device_index &&
                      sample_values.get_device_index() == device_index &&
                      sample_indices.get_device_index() == device_index &&
                      clean_values.get_device_index() == device_index &&
                      clean_indices.get_device_index() == device_index &&
                      soft_embed.get_device_index() == device_index,
                  label, ": all tensors must be on the same CUDA device");
  if (logit_scale != nullptr) {
    STD_TORCH_CHECK(logit_scale->get_device_index() == device_index,
                    label, ": logit_scale must be on the same CUDA device");
  }
}

void validate_local_state_tensors(
    const char* label, torch::stable::Tensor& local_max,
    torch::stable::Tensor& local_sum_exp,
    torch::stable::Tensor& local_weighted_logits,
    torch::stable::Tensor& local_soft_part,
    torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices,
    torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight,
    torch::stable::Tensor const& logit_scale) {
  STD_TORCH_CHECK(hidden.dim() == 2 && lm_head_weight.dim() == 2,
                  label, ": hidden and lm_head_weight must be rank-2");
  STD_TORCH_CHECK(local_max.dim() == 1 && local_sum_exp.dim() == 1 &&
                      local_weighted_logits.dim() == 1 &&
                      local_soft_part.dim() == 2 && clean_values.dim() == 1 &&
                      clean_indices.dim() == 1 && sample_values.dim() == 1 &&
                      sample_indices.dim() == 1,
                  label, ": output shapes must be [rows] state values and "
                         "local_soft_part=[rows, hidden]");
  STD_TORCH_CHECK(logit_scale.dim() == 0 || logit_scale.dim() == 1, label,
                  ": logit_scale must be scalar or rank-1");

  STD_TORCH_CHECK(hidden.is_cuda() && lm_head_weight.is_cuda() &&
                      logit_scale.is_cuda() && local_max.is_cuda() &&
                      local_sum_exp.is_cuda() &&
                      local_weighted_logits.is_cuda() &&
                      local_soft_part.is_cuda() && clean_values.is_cuda() &&
                      clean_indices.is_cuda() && sample_values.is_cuda() &&
                      sample_indices.is_cuda(),
                  label, ": all tensors must be CUDA tensors");

  STD_TORCH_CHECK(hidden.is_contiguous() && lm_head_weight.is_contiguous() &&
                      logit_scale.is_contiguous() && local_max.is_contiguous() &&
                      local_sum_exp.is_contiguous() &&
                      local_weighted_logits.is_contiguous() &&
                      local_soft_part.is_contiguous() &&
                      clean_values.is_contiguous() &&
                      clean_indices.is_contiguous() &&
                      sample_values.is_contiguous() &&
                      sample_indices.is_contiguous(),
                  label, ": all tensors must be contiguous");

  const int64_t rows = hidden.size(0);
  const int64_t hidden_size = hidden.size(1);
  const int64_t vocab_size = lm_head_weight.size(0);
  STD_TORCH_CHECK(lm_head_weight.size(1) == hidden_size,
                  label, ": hidden size mismatch");
  STD_TORCH_CHECK(local_max.size(0) == rows && local_sum_exp.size(0) == rows &&
                      local_weighted_logits.size(0) == rows &&
                      local_soft_part.size(0) == rows &&
                      local_soft_part.size(1) == hidden_size &&
                      clean_values.size(0) == rows &&
                      clean_indices.size(0) == rows &&
                      sample_values.size(0) == rows &&
                      sample_indices.size(0) == rows,
                  label, ": output shapes do not match hidden");
  if (logit_scale.dim() == 1) {
    STD_TORCH_CHECK(logit_scale.size(0) == 1 || logit_scale.size(0) == rows,
                    label,
                    ": logit_scale must have one value or one value per row");
  }

  STD_TORCH_CHECK(vocab_size > 0 && hidden_size > 0,
                  label, ": vocab and hidden sizes must be positive");
  STD_TORCH_CHECK(vocab_size <= std::numeric_limits<int>::max() &&
                      hidden_size <= std::numeric_limits<int>::max() &&
                      rows <= std::numeric_limits<int>::max(),
                  label, ": shapes exceed CUDA kernel int32 limits");
  STD_TORCH_CHECK(vocab_size % 8 == 0 && hidden_size % 8 == 0,
                  label, ": vocab and hidden sizes must be divisible by 8");
  STD_TORCH_CHECK(rows <= 32768 && vocab_size <= 262144 && hidden_size <= 8192,
                  label, ": local-state MVP supports rows<=32768, "
                         "vocab<=262144, hidden<=8192");

  STD_TORCH_CHECK(
      hidden.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          lm_head_weight.scalar_type() ==
              torch::headeronly::ScalarType::BFloat16,
      label, ": hidden and lm_head_weight must be bf16");
  STD_TORCH_CHECK(local_max.scalar_type() == torch::headeronly::ScalarType::Float &&
                      local_sum_exp.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      local_weighted_logits.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      local_soft_part.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      clean_values.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      sample_values.scalar_type() ==
                          torch::headeronly::ScalarType::Float,
                  label, ": floating state outputs must be fp32");
  STD_TORCH_CHECK(clean_indices.scalar_type() ==
                          torch::headeronly::ScalarType::Long &&
                      sample_indices.scalar_type() ==
                          torch::headeronly::ScalarType::Long,
                  label, ": indices must be int64");
  STD_TORCH_CHECK(logit_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float,
                  label, ": logit_scale must be fp32");

  const int32_t device_index = hidden.get_device_index();
  STD_TORCH_CHECK(lm_head_weight.get_device_index() == device_index &&
                      logit_scale.get_device_index() == device_index &&
                      local_max.get_device_index() == device_index &&
                      local_sum_exp.get_device_index() == device_index &&
                      local_weighted_logits.get_device_index() == device_index &&
                      local_soft_part.get_device_index() == device_index &&
                      clean_values.get_device_index() == device_index &&
                      clean_indices.get_device_index() == device_index &&
                      sample_values.get_device_index() == device_index &&
                      sample_indices.get_device_index() == device_index,
                  label, ": all tensors must be on the same CUDA device");
}

void validate_pack_local_state_tensors(
    const char* label, torch::stable::Tensor& packed,
    torch::stable::Tensor const& local_max,
    torch::stable::Tensor const& global_max,
    torch::stable::Tensor const& local_sum_exp,
    torch::stable::Tensor const& local_weighted_logits,
    torch::stable::Tensor const& local_soft_part) {
  STD_TORCH_CHECK(packed.dim() == 2 && local_soft_part.dim() == 2,
                  label, ": packed and local_soft_part must be rank-2");
  STD_TORCH_CHECK(local_max.dim() == 1 && global_max.dim() == 1 &&
                      local_sum_exp.dim() == 1 &&
                      local_weighted_logits.dim() == 1,
                  label, ": scalar state tensors must be rank-1");
  STD_TORCH_CHECK(packed.is_cuda() && local_max.is_cuda() &&
                      global_max.is_cuda() && local_sum_exp.is_cuda() &&
                      local_weighted_logits.is_cuda() &&
                      local_soft_part.is_cuda(),
                  label, ": all tensors must be CUDA tensors");
  STD_TORCH_CHECK(packed.is_contiguous() && local_max.is_contiguous() &&
                      global_max.is_contiguous() &&
                      local_sum_exp.is_contiguous() &&
                      local_weighted_logits.is_contiguous() &&
                      local_soft_part.is_contiguous(),
                  label, ": all tensors must be contiguous");

  const int64_t rows = local_soft_part.size(0);
  const int64_t hidden_size = local_soft_part.size(1);
  STD_TORCH_CHECK(local_max.size(0) == rows && global_max.size(0) == rows &&
                      local_sum_exp.size(0) == rows &&
                      local_weighted_logits.size(0) == rows &&
                      packed.size(0) == rows &&
                      packed.size(1) == hidden_size + 2,
                  label, ": packed must be [rows, hidden+2] and scalar "
                         "state tensors must be [rows]");
  STD_TORCH_CHECK(hidden_size >= 0 &&
                      rows <= std::numeric_limits<int>::max() &&
                      hidden_size <= std::numeric_limits<int>::max() - 2,
                  label, ": shapes exceed CUDA kernel limits");

  STD_TORCH_CHECK(packed.scalar_type() == torch::headeronly::ScalarType::Float &&
                      local_max.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      global_max.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      local_sum_exp.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      local_weighted_logits.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      local_soft_part.scalar_type() ==
                          torch::headeronly::ScalarType::Float,
                  label, ": all tensors must be fp32");

  const int32_t device_index = local_soft_part.get_device_index();
  STD_TORCH_CHECK(packed.get_device_index() == device_index &&
                      local_max.get_device_index() == device_index &&
                      global_max.get_device_index() == device_index &&
                      local_sum_exp.get_device_index() == device_index &&
                      local_weighted_logits.get_device_index() == device_index,
                  label, ": all tensors must be on the same CUDA device");
}

void run_mode16(
    torch::stable::Tensor& entropy, torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices, torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices, torch::stable::Tensor& soft_embed,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight, const float* logit_scale,
    float normalizer, float final_logit_softcapping, int64_t rng_seed,
    int64_t rng_offset, int64_t rng_row_offset) {
  const int64_t rows = hidden.size(0);
  if (rows == 0) {
    return;
  }

  const int64_t vocab_size = lm_head_weight.size(0);
  const int64_t hidden_size = hidden.size(1);
  const int32_t device_index = hidden.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  const auto device = hidden.device();
  auto logits = torch::stable::empty({rows, vocab_size},
                                     torch::headeronly::ScalarType::Float,
                                     std::nullopt, device);
  auto probs = torch::stable::empty({rows, vocab_size},
                                    torch::headeronly::ScalarType::BFloat16,
                                    std::nullopt, device);

  auto* logits_ptr = logits.mutable_data_ptr<float>();
  auto* probs_ptr =
      reinterpret_cast<__nv_bfloat16*>(probs.mutable_data_ptr());
  const auto* hidden_ptr =
      reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr());
  const auto* weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(lm_head_weight.const_data_ptr());
  auto* soft_embed_ptr = soft_embed.mutable_data_ptr<float>();

  cublasHandle_t handle = get_current_cuda_blas_handle();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

  const float one = 1.0f;
  const float zero = 0.0f;

  check_cublas(
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   static_cast<int>(vocab_size), static_cast<int>(rows),
                   static_cast<int>(hidden_size), &one, weight_ptr,
                   CUDA_R_16BF, static_cast<int>(hidden_size), hidden_ptr,
                   CUDA_R_16BF, static_cast<int>(hidden_size), &zero,
                   logits_ptr, CUDA_R_32F, static_cast<int>(vocab_size),
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "LM-head GEMM");

  logits_to_stats_probs_warp_kernel<<<static_cast<unsigned int>(rows),
                                      kWarpStatsReduceThreads, 0, stream>>>(
      logits_ptr, probs_ptr, logit_scale, final_logit_softcapping,
      entropy.mutable_data_ptr<float>(), sample_values.mutable_data_ptr<float>(),
      sample_indices.mutable_data_ptr<int64_t>(),
      clean_values.mutable_data_ptr<float>(),
      clean_indices.mutable_data_ptr<int64_t>(), static_cast<int>(rows),
      static_cast<int>(vocab_size), static_cast<uint64_t>(rng_seed),
      static_cast<uint64_t>(rng_offset), static_cast<int>(rng_row_offset));
  const cudaError_t status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess,
                  "diffusion_gemma_flashdenoise: logits-to-probs kernel "
                  "launch failed: " +
                      std::string(cudaGetErrorString(status)));

  check_cublas(
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   static_cast<int>(hidden_size), static_cast<int>(rows),
                   static_cast<int>(vocab_size), &normalizer, weight_ptr,
                   CUDA_R_16BF, static_cast<int>(hidden_size), probs_ptr,
                   CUDA_R_16BF, static_cast<int>(vocab_size), &zero,
                   soft_embed_ptr, CUDA_R_32F, static_cast<int>(hidden_size),
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "soft-embed GEMM");
}

void run_local_state_scaled(
    torch::stable::Tensor& local_max,
    torch::stable::Tensor& local_sum_exp,
    torch::stable::Tensor& local_weighted_logits,
    torch::stable::Tensor& local_soft_part,
    torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices,
    torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight,
    torch::stable::Tensor const& logit_scale, int64_t vocab_start_index,
    float final_logit_softcapping, int64_t rng_seed, int64_t rng_offset) {
  const int64_t rows = hidden.size(0);
  if (rows == 0) {
    return;
  }

  const int64_t vocab_size = lm_head_weight.size(0);
  const int64_t hidden_size = hidden.size(1);
  STD_TORCH_CHECK(vocab_start_index >= 0 &&
                      vocab_start_index <=
                          std::numeric_limits<int64_t>::max() - vocab_size,
                  "diffusion_gemma_flashdenoise_local_state_scaled: invalid "
                  "vocab_start_index");
  const int32_t device_index = hidden.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  const auto device = hidden.device();
  auto logits = torch::stable::empty({rows, vocab_size},
                                     torch::headeronly::ScalarType::Float,
                                     std::nullopt, device);
  auto exp_weights = torch::stable::empty(
      {rows, vocab_size}, torch::headeronly::ScalarType::BFloat16, std::nullopt,
      device);

  auto* logits_ptr = logits.mutable_data_ptr<float>();
  auto* exp_weights_ptr =
      reinterpret_cast<__nv_bfloat16*>(exp_weights.mutable_data_ptr());
  const auto* hidden_ptr =
      reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr());
  const auto* weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(lm_head_weight.const_data_ptr());
  auto* local_soft_part_ptr = local_soft_part.mutable_data_ptr<float>();

  cublasHandle_t handle = get_current_cuda_blas_handle();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

  const float one = 1.0f;
  const float zero = 0.0f;

  check_cublas(
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   static_cast<int>(vocab_size), static_cast<int>(rows),
                   static_cast<int>(hidden_size), &one, weight_ptr,
                   CUDA_R_16BF, static_cast<int>(hidden_size), hidden_ptr,
                   CUDA_R_16BF, static_cast<int>(hidden_size), &zero,
                   logits_ptr, CUDA_R_32F, static_cast<int>(vocab_size),
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "local-state LM-head GEMM");

  logits_to_local_state_exp_warp_kernel<<<static_cast<unsigned int>(rows),
                                          kWarpStatsReduceThreads, 0, stream>>>(
      logits_ptr, exp_weights_ptr, logit_scale.const_data_ptr<float>(),
      logit_scale.dim() == 0 || logit_scale.size(0) == 1,
      final_logit_softcapping, local_max.mutable_data_ptr<float>(),
      local_sum_exp.mutable_data_ptr<float>(),
      local_weighted_logits.mutable_data_ptr<float>(),
      sample_values.mutable_data_ptr<float>(),
      sample_indices.mutable_data_ptr<int64_t>(),
      clean_values.mutable_data_ptr<float>(),
      clean_indices.mutable_data_ptr<int64_t>(), static_cast<int>(rows),
      static_cast<int>(vocab_size), vocab_start_index,
      static_cast<uint64_t>(rng_seed), static_cast<uint64_t>(rng_offset));
  cudaError_t status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess,
                  "diffusion_gemma_flashdenoise_local_state_scaled: "
                  "logits-to-local-state kernel launch failed: " +
                      std::string(cudaGetErrorString(status)));

  check_cublas(
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   static_cast<int>(hidden_size), static_cast<int>(rows),
                   static_cast<int>(vocab_size), &one, weight_ptr, CUDA_R_16BF,
                   static_cast<int>(hidden_size), exp_weights_ptr, CUDA_R_16BF,
                   static_cast<int>(vocab_size), &zero, local_soft_part_ptr,
                   CUDA_R_32F, static_cast<int>(hidden_size),
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "local-state soft-part GEMM");
}

void run_pack_local_state(torch::stable::Tensor& packed,
                          torch::stable::Tensor const& local_max,
                          torch::stable::Tensor const& global_max,
                          torch::stable::Tensor const& local_sum_exp,
                          torch::stable::Tensor const& local_weighted_logits,
                          torch::stable::Tensor const& local_soft_part) {
  const int64_t rows = local_soft_part.size(0);
  if (rows == 0) {
    return;
  }

  const int64_t hidden_size = local_soft_part.size(1);
  const int32_t device_index = local_soft_part.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  pack_local_state_kernel<<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
      packed.mutable_data_ptr<float>(), local_max.const_data_ptr<float>(),
      global_max.const_data_ptr<float>(), local_sum_exp.const_data_ptr<float>(),
      local_weighted_logits.const_data_ptr<float>(),
      local_soft_part.const_data_ptr<float>(), static_cast<int>(rows),
      static_cast<int>(hidden_size));
  const cudaError_t status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess,
                  "diffusion_gemma_flashdenoise_pack_local_state: kernel "
                  "launch failed: " +
                      std::string(cudaGetErrorString(status)));
}

}  // namespace

void diffusion_gemma_flashdenoise(
    torch::stable::Tensor& entropy, torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices, torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices, torch::stable::Tensor& soft_embed,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight, double normalizer,
    int64_t mode_flags, int64_t rng_seed, int64_t rng_offset) {
  validate_flashdenoise_tensors("diffusion_gemma_flashdenoise", entropy,
                                sample_values, sample_indices, clean_values,
                                clean_indices, soft_embed, hidden,
                                lm_head_weight, nullptr);
  STD_TORCH_CHECK(mode_flags == kModeMaterializedWarpStats,
                  "diffusion_gemma_flashdenoise: this build only supports "
                  "mode_flags=16");
  run_mode16(entropy, sample_values, sample_indices, clean_values,
             clean_indices, soft_embed, hidden, lm_head_weight, nullptr,
             static_cast<float>(normalizer), 0.0f, rng_seed, rng_offset, 0);
}

void diffusion_gemma_flashdenoise_scaled(
    torch::stable::Tensor& entropy, torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices, torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices, torch::stable::Tensor& soft_embed,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight,
    torch::stable::Tensor const& logit_scale, double normalizer,
    double final_logit_softcapping, int64_t mode_flags, int64_t rng_seed,
    int64_t rng_offset, int64_t rng_row_offset) {
  validate_flashdenoise_tensors("diffusion_gemma_flashdenoise_scaled", entropy,
                                sample_values, sample_indices, clean_values,
                                clean_indices, soft_embed, hidden,
                                lm_head_weight, &logit_scale);
  STD_TORCH_CHECK(mode_flags == kModeMaterializedWarpStats,
                  "diffusion_gemma_flashdenoise_scaled: this build only "
                  "supports mode_flags=16");
  run_mode16(entropy, sample_values, sample_indices, clean_values,
             clean_indices, soft_embed, hidden, lm_head_weight,
             logit_scale.const_data_ptr<float>(), static_cast<float>(normalizer),
             static_cast<float>(final_logit_softcapping), rng_seed, rng_offset,
             rng_row_offset);
}

void diffusion_gemma_flashdenoise_local_state_scaled(
    torch::stable::Tensor& local_max,
    torch::stable::Tensor& local_sum_exp,
    torch::stable::Tensor& local_weighted_logits,
    torch::stable::Tensor& local_soft_part,
    torch::stable::Tensor& clean_values,
    torch::stable::Tensor& clean_indices,
    torch::stable::Tensor& sample_values,
    torch::stable::Tensor& sample_indices,
    torch::stable::Tensor const& hidden,
    torch::stable::Tensor const& lm_head_weight,
    torch::stable::Tensor const& logit_scale, int64_t vocab_start_index,
    double final_logit_softcapping, int64_t rng_seed, int64_t rng_offset) {
  validate_local_state_tensors(
      "diffusion_gemma_flashdenoise_local_state_scaled", local_max,
      local_sum_exp, local_weighted_logits, local_soft_part, clean_values,
      clean_indices, sample_values, sample_indices, hidden, lm_head_weight,
      logit_scale);
  run_local_state_scaled(local_max, local_sum_exp, local_weighted_logits,
                         local_soft_part, clean_values, clean_indices,
                         sample_values, sample_indices, hidden, lm_head_weight,
                         logit_scale, vocab_start_index,
                         static_cast<float>(final_logit_softcapping), rng_seed,
                         rng_offset);
}

void diffusion_gemma_flashdenoise_pack_local_state(
    torch::stable::Tensor& packed, torch::stable::Tensor const& local_max,
    torch::stable::Tensor const& global_max,
    torch::stable::Tensor const& local_sum_exp,
    torch::stable::Tensor const& local_weighted_logits,
    torch::stable::Tensor const& local_soft_part) {
  validate_pack_local_state_tensors(
      "diffusion_gemma_flashdenoise_pack_local_state", packed, local_max,
      global_max, local_sum_exp, local_weighted_logits, local_soft_part);
  run_pack_local_state(packed, local_max, global_max, local_sum_exp,
                       local_weighted_logits, local_soft_part);
}
