#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::vector<int64_t>& src_block_numbers,
  const std::vector<int64_t>& dst_block_numbers);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::vector<int64_t>& src_block_numbers,
  const std::vector<int64_t>& dst_block_numbers);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);
