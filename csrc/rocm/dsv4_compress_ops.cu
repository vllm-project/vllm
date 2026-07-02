// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/**
 * Torch op wrappers for the fused DeepSeek-V4 compressor kernels (gfx950 only).
 *
 * The device kernels + C-ABI launchers live in
 * csrc/rocm/dsv4_{csa,hca,indexer}_compress.cu. These host-only wrappers adapt
 * the model's tensor-level call (see vllm/models/deepseek_v4/compressor.py) to
 * the launchers. Registered into _rocm_C from torch_bindings.cpp under
 * VLLM_ROCM_GFX950.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rocm/ops.h"

#ifdef VLLM_ROCM_GFX950

extern "C" void launch_csa_compress(
    int num_tokens,
    const void* state_cache, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    void* stream);

extern "C" void launch_hca_compress_plan(
    int num_tokens, int nw_select,
    const void* state_cache, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    int32_t* plan, int32_t* counter, int plan_capacity, void* stream);

extern "C" void launch_indexer_compress(
    int num_tokens, int use_fp4_cache,
    const void* state_cache, int64_t state_stride0, int64_t state_stride1,
    const float* ape_raw, int64_t ape_stride,
    const int32_t* token_to_req_indices, const int64_t* positions,
    const int64_t* slot_mapping, const int32_t* block_table,
    int64_t block_table_stride, int32_t block_size,
    const void* rms_norm_weight, bool rms_norm_weight_is_bf16,
    float rms_norm_eps,
    const float* cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* kv_cache, const int64_t* kv_slot_mapping,
    int32_t kv_cache_block_size, int64_t kv_block_stride, int32_t scale_dim,
    void* stream);

namespace {
bool is_supported_rms_weight_dtype(const torch::Tensor& w) {
  return w.dtype() == torch::kBFloat16 || w.dtype() == torch::kFloat32;
}
}  // namespace

void dsv4_csa_compress(torch::Tensor state_cache, int64_t num_actual,
                       torch::Tensor ape, torch::Tensor token_to_req_indices,
                       torch::Tensor positions, torch::Tensor slot_mapping,
                       torch::Tensor block_table, int64_t block_size,
                       torch::Tensor rms_norm_weight, double rms_norm_eps,
                       torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                       torch::Tensor kv_slot_mapping,
                       int64_t kv_cache_block_size, int64_t scale_dim) {
  TORCH_CHECK(state_cache.dtype() == torch::kBFloat16, "state_cache must be bf16");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape must be fp32");
  TORCH_CHECK(is_supported_rms_weight_dtype(rms_norm_weight),
              "rms_norm_weight must be bf16 or fp32");
  TORCH_CHECK(ape.size(0) == 4, "CSA expects RAW ape [4, 2*head_dim]");
  if (num_actual == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(state_cache));
  bool rms_norm_weight_is_bf16 = rms_norm_weight.dtype() == torch::kBFloat16;
  launch_csa_compress(
      static_cast<int>(num_actual), state_cache.data_ptr(),
      state_cache.stride(0), state_cache.stride(1), ape.data_ptr<float>(),
      ape.stride(0), token_to_req_indices.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
      block_table.data_ptr<int32_t>(), block_table.stride(0),
      static_cast<int32_t>(block_size), rms_norm_weight.data_ptr(),
      rms_norm_weight_is_bf16, static_cast<float>(rms_norm_eps),
      cos_sin_cache.data_ptr<float>(),
      cos_sin_cache.stride(0), kv_cache.data_ptr<uint8_t>(),
      kv_slot_mapping.data_ptr<int64_t>(),
      static_cast<int32_t>(kv_cache_block_size), kv_cache.stride(0),
      static_cast<int32_t>(scale_dim),
      at::cuda::getCurrentCUDAStream().stream());
}

void dsv4_hca_compress(torch::Tensor state_cache, int64_t num_actual,
                       torch::Tensor ape, torch::Tensor token_to_req_indices,
                       torch::Tensor positions, torch::Tensor slot_mapping,
                       torch::Tensor block_table, int64_t block_size,
                       torch::Tensor rms_norm_weight, double rms_norm_eps,
                       torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                       torch::Tensor kv_slot_mapping,
                       int64_t kv_cache_block_size, int64_t scale_dim,
                       torch::Tensor plan_scratch,
                       torch::Tensor counter_scratch) {
  TORCH_CHECK(state_cache.dtype() == torch::kBFloat16, "state_cache must be bf16");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape must be fp32");
  TORCH_CHECK(is_supported_rms_weight_dtype(rms_norm_weight),
              "rms_norm_weight must be bf16 or fp32");
  TORCH_CHECK(plan_scratch.dtype() == torch::kInt32,
              "plan_scratch must be int32");
  TORCH_CHECK(counter_scratch.dtype() == torch::kInt32,
              "counter_scratch must be int32");
  TORCH_CHECK(ape.size(0) == 128, "HCA expects RAW ape [128, head_dim]");
  if (num_actual == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(state_cache));
  bool rms_norm_weight_is_bf16 = rms_norm_weight.dtype() == torch::kBFloat16;

  // Compact-plan scratch: safe upper bound on boundary count.
  int64_t num_reqs = block_table.size(0);
  int plan_capacity = static_cast<int>(num_actual / 128 + num_reqs + 2);
  TORCH_CHECK(plan_scratch.device() == state_cache.device(),
              "plan_scratch must be on the same device as state_cache");
  TORCH_CHECK(counter_scratch.device() == state_cache.device(),
              "counter_scratch must be on the same device as state_cache");
  TORCH_CHECK(plan_scratch.numel() >= plan_capacity,
              "plan_scratch is too small for HCA compact plan");
  TORCH_CHECK(counter_scratch.numel() >= 1,
              "counter_scratch must have at least one element");

  launch_hca_compress_plan(
      static_cast<int>(num_actual), /*nw_select=*/0, state_cache.data_ptr(),
      state_cache.stride(0), state_cache.stride(1), ape.data_ptr<float>(),
      ape.stride(0), token_to_req_indices.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
      block_table.data_ptr<int32_t>(), block_table.stride(0),
      static_cast<int32_t>(block_size), rms_norm_weight.data_ptr(),
      rms_norm_weight_is_bf16, static_cast<float>(rms_norm_eps),
      cos_sin_cache.data_ptr<float>(),
      cos_sin_cache.stride(0), kv_cache.data_ptr<uint8_t>(),
      kv_slot_mapping.data_ptr<int64_t>(),
      static_cast<int32_t>(kv_cache_block_size), kv_cache.stride(0),
      static_cast<int32_t>(scale_dim), plan_scratch.data_ptr<int32_t>(),
      counter_scratch.data_ptr<int32_t>(), plan_capacity,
      at::cuda::getCurrentCUDAStream().stream());
}

void dsv4_indexer_compress(torch::Tensor state_cache, int64_t num_actual,
                           torch::Tensor ape,
                           torch::Tensor token_to_req_indices,
                           torch::Tensor positions, torch::Tensor slot_mapping,
                           torch::Tensor block_table, int64_t block_size,
                           torch::Tensor rms_norm_weight, double rms_norm_eps,
                           torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                           torch::Tensor kv_slot_mapping,
                           int64_t kv_cache_block_size, int64_t scale_dim,
                           bool use_fp4_cache) {
  TORCH_CHECK(state_cache.dtype() == torch::kBFloat16, "state_cache must be bf16");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape must be fp32");
  TORCH_CHECK(is_supported_rms_weight_dtype(rms_norm_weight),
              "rms_norm_weight must be bf16 or fp32");
  TORCH_CHECK(ape.size(0) == 4 && ape.size(1) == 256,
              "indexer expects RAW ape [4, 256]");
  if (num_actual == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(state_cache));
  bool rms_norm_weight_is_bf16 = rms_norm_weight.dtype() == torch::kBFloat16;
  launch_indexer_compress(
      static_cast<int>(num_actual), use_fp4_cache ? 1 : 0, state_cache.data_ptr(),
      state_cache.stride(0), state_cache.stride(1), ape.data_ptr<float>(),
      ape.stride(0), token_to_req_indices.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
      block_table.data_ptr<int32_t>(), block_table.stride(0),
      static_cast<int32_t>(block_size), rms_norm_weight.data_ptr(),
      rms_norm_weight_is_bf16, static_cast<float>(rms_norm_eps),
      cos_sin_cache.data_ptr<float>(),
      cos_sin_cache.stride(0), kv_cache.data_ptr<uint8_t>(),
      kv_slot_mapping.data_ptr<int64_t>(),
      static_cast<int32_t>(kv_cache_block_size), kv_cache.stride(0),
      static_cast<int32_t>(scale_dim),
      at::cuda::getCurrentCUDAStream().stream());
}

#endif  // VLLM_ROCM_GFX950
