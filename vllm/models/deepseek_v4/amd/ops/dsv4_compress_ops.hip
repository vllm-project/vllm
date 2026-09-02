// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <limits>

#include "rocm/ops.h"

#ifdef VLLM_ROCM_GFX942

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

extern "C" void launch_hca_compress_nw(
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
  TORCH_CHECK(num_actual >= 0 &&
                  num_actual <= std::numeric_limits<int32_t>::max(),
              "num_actual must fit in a non-negative int32");
  TORCH_CHECK(state_cache.dtype() == torch::kBFloat16, "state_cache must be bf16");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape must be fp32");
  TORCH_CHECK(is_supported_rms_weight_dtype(rms_norm_weight),
              "rms_norm_weight must be bf16 or fp32");
  TORCH_CHECK(ape.dim() == 2 && ape.size(0) == 4 && ape.size(1) == 1024,
              "CSA expects contiguous RAW ape [4, 1024]");
  TORCH_CHECK(ape.stride(1) == 1, "CSA ape innermost dimension must be contiguous");
  TORCH_CHECK(block_size == 4, "CSA state-cache block size must be 4");
  TORCH_CHECK(scale_dim == 8, "CSA packed-cache scale dimension must be 8");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(2) == 2048 &&
                  state_cache.stride(2) == 1,
              "CSA state_cache must have contiguous rows of width 2048");
  TORCH_CHECK(rms_norm_weight.numel() == 512,
              "CSA rms_norm_weight must contain 512 elements");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
              "CSA cos_sin_cache must be fp32 [max_position, 64]");
  TORCH_CHECK(kv_cache.dtype() == torch::kUInt8 && kv_cache.dim() == 3,
              "CSA kv_cache must use the 3D uint8 fp8_ds_mla layout");
  TORCH_CHECK(token_to_req_indices.dtype() == torch::kInt32 &&
                  positions.dtype() == torch::kInt64 &&
                  slot_mapping.dtype() == torch::kInt64 &&
                  block_table.dtype() == torch::kInt32 &&
                  kv_slot_mapping.dtype() == torch::kInt64,
              "CSA metadata tensors have unexpected dtypes");
  TORCH_CHECK(token_to_req_indices.numel() >= num_actual &&
                  positions.numel() >= num_actual &&
                  slot_mapping.numel() >= num_actual &&
                  kv_slot_mapping.numel() >= num_actual,
              "CSA per-token metadata is shorter than num_actual");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1,
              "CSA block_table must be a row-contiguous 2D tensor");
  TORCH_CHECK(rms_norm_weight.is_contiguous(),
              "CSA rms_norm_weight must be contiguous");
  TORCH_CHECK(cos_sin_cache.stride(1) == 1,
              "CSA cos_sin_cache rows must be contiguous");
  TORCH_CHECK(kv_cache.size(1) == kv_cache_block_size &&
                  kv_cache.size(2) >= 584 && kv_cache.stride(2) == 1,
              "CSA kv_cache has an invalid block size or packed-row layout");
  for (const torch::Tensor* tensor :
       {&ape, &token_to_req_indices, &positions, &slot_mapping, &block_table,
        &rms_norm_weight, &cos_sin_cache, &kv_cache, &kv_slot_mapping}) {
    TORCH_CHECK(tensor->device() == state_cache.device(),
                "CSA inputs must be on the state_cache device");
  }
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
  TORCH_CHECK(num_actual >= 0 &&
                  num_actual <= std::numeric_limits<int32_t>::max(),
              "num_actual must fit in a non-negative int32");
  TORCH_CHECK(state_cache.dtype() == torch::kBFloat16, "state_cache must be bf16");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape must be fp32");
  TORCH_CHECK(is_supported_rms_weight_dtype(rms_norm_weight),
              "rms_norm_weight must be bf16 or fp32");
  TORCH_CHECK(plan_scratch.dtype() == torch::kInt32,
              "plan_scratch must be int32");
  TORCH_CHECK(counter_scratch.dtype() == torch::kInt32,
              "counter_scratch must be int32");
  TORCH_CHECK(ape.dim() == 2 && ape.size(0) == 128 && ape.size(1) == 512,
              "HCA expects contiguous RAW ape [128, 512]");
  TORCH_CHECK(ape.stride(1) == 1, "HCA ape innermost dimension must be contiguous");
  TORCH_CHECK(block_size == 8, "HCA state-cache block size must be 8");
  TORCH_CHECK(scale_dim == 8, "HCA packed-cache scale dimension must be 8");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(2) == 1024 &&
                  state_cache.stride(2) == 1,
              "HCA state_cache must have contiguous rows of width 1024");
  TORCH_CHECK(rms_norm_weight.numel() == 512,
              "HCA rms_norm_weight must contain 512 elements");
  TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
              "HCA cos_sin_cache must be fp32 [max_position, 64]");
  TORCH_CHECK(kv_cache.dtype() == torch::kUInt8 && kv_cache.dim() == 3,
              "HCA kv_cache must use the 3D uint8 fp8_ds_mla layout");
  TORCH_CHECK(token_to_req_indices.dtype() == torch::kInt32 &&
                  positions.dtype() == torch::kInt64 &&
                  slot_mapping.dtype() == torch::kInt64 &&
                  block_table.dtype() == torch::kInt32 &&
                  kv_slot_mapping.dtype() == torch::kInt64,
              "HCA metadata tensors have unexpected dtypes");
  TORCH_CHECK(token_to_req_indices.numel() >= num_actual &&
                  positions.numel() >= num_actual &&
                  slot_mapping.numel() >= num_actual &&
                  kv_slot_mapping.numel() >= num_actual,
              "HCA per-token metadata is shorter than num_actual");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1,
              "HCA block_table must be a row-contiguous 2D tensor");
  TORCH_CHECK(rms_norm_weight.is_contiguous(),
              "HCA rms_norm_weight must be contiguous");
  TORCH_CHECK(cos_sin_cache.stride(1) == 1,
              "HCA cos_sin_cache rows must be contiguous");
  TORCH_CHECK(kv_cache.size(1) == kv_cache_block_size &&
                  kv_cache.size(2) >= 584 && kv_cache.stride(2) == 1,
              "HCA kv_cache has an invalid block size or packed-row layout");
  for (const torch::Tensor* tensor :
       {&ape, &token_to_req_indices, &positions, &slot_mapping, &block_table,
        &rms_norm_weight, &cos_sin_cache, &kv_cache, &kv_slot_mapping,
        &plan_scratch, &counter_scratch}) {
    TORCH_CHECK(tensor->device() == state_cache.device(),
                "HCA inputs must be on the state_cache device");
  }
  if (num_actual == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(state_cache));
  bool rms_norm_weight_is_bf16 = rms_norm_weight.dtype() == torch::kBFloat16;

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

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (num_actual <= 2048) {
    launch_hca_compress_nw(
        static_cast<int>(num_actual), /*nw_select=*/8, state_cache.data_ptr(),
        state_cache.stride(0), state_cache.stride(1), ape.data_ptr<float>(),
        ape.stride(0), token_to_req_indices.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
        block_table.data_ptr<int32_t>(), block_table.stride(0),
        static_cast<int32_t>(block_size), rms_norm_weight.data_ptr(),
        rms_norm_weight_is_bf16, static_cast<float>(rms_norm_eps),
        cos_sin_cache.data_ptr<float>(), cos_sin_cache.stride(0),
        kv_cache.data_ptr<uint8_t>(), kv_slot_mapping.data_ptr<int64_t>(),
        static_cast<int32_t>(kv_cache_block_size), kv_cache.stride(0),
        static_cast<int32_t>(scale_dim), stream);
    return;
  }

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
      stream);
}

#endif  // VLLM_ROCM_GFX942
