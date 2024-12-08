#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

void kv_store_copy_incomplete_blocks(
    torch::Tensor& src, torch::Tensor& dst, const int64_t layer_id,
    const torch::Tensor& incomplete_block_mapping);
void kv_store_copy_blocks2CPU(torch::Tensor& src, torch::Tensor& dst,
                              const int64_t layer_id,
                              const torch::Tensor& block_mapping);

void kv_store_copy_blocks2GPU(
    torch::Tensor& src, std::vector<torch::Tensor> const& dst,
    const int64_t num_layers, const torch::Tensor& block_mapping,
    const torch::Tensor& block_offsets, const torch::Tensor& req_ids,
    std::vector<long> const& events, const bool is_batch_layer);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype, const double k_scale,
                       const double v_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             const double k_scale, const double v_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);
