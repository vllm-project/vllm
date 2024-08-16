#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor const& src, torch::Tensor& dst,
                 torch::Tensor const& block_mapping);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 torch::Tensor const& block_mapping);

void reshape_and_cache(torch::Tensor const& key, torch::Tensor const& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor const& slot_mapping,
                       std::string const& kv_cache_dtype, const double k_scale,
                       double const v_scale);

void reshape_and_cache_flash(torch::Tensor const& key,
                             torch::Tensor const& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor const& slot_mapping,
                             std::string const& kv_cache_dtype,
                             double const k_scale, double const v_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor const& src_cache,
                 double const scale, std::string const& kv_cache_dtype);
