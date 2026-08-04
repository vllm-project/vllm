// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include "kv_transfer_types.h"

void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size = 0,
    const int head_size = 0, const int skip_prefix_n_tokens = 0);

// Max chunks per fused launch; the by-value pack must fit the 4 KB
// kernel-arg buffer.
constexpr int MAX_FUSED_TRANSFER_CHUNKS = 16;

// Fused transfer: one launch, up to MAX_FUSED_TRANSFER_CHUNKS same-geometry
// chunks; params ride in the kernel-arg buffer (no device upload).
void multi_layer_kv_transfer_fused_ptr(
    const std::vector<uintptr_t>& key_values,
    const std::vector<uintptr_t>& slot_mappings, const std::vector<int>& n_toks,
    uintptr_t page_buffer_ptrs, const int num_layers,
    const int layout_num_tokens, const int num_origin_elements,
    const int element_size, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size = 0,
    const int head_size = 0);

// collapses to multi_layer_kv_transfer for MLA
void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const EngineKVFormat engine_kv_format,
                              const bool token_major = false);

void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const TransferDirection direction,
                                  const bool token_major = false);

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);
