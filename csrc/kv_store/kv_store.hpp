#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <thread>

class KVStore {
 public:
  KVStore() = default;
  void CopyIncompleteBlocks(torch::Tensor& src, torch::Tensor& dst,
      const int64_t layer_id,
      const torch::Tensor& incomplete_block_mapping);

  void CopyBlocks2CPU(torch::Tensor& src, torch::Tensor& dst,
      const int64_t layer_id,
      const torch::Tensor& block_mapping);

  void CopyBlocks2GPUBatch(torch::Tensor& src,
      std::vector<torch::Tensor> const& kv_caches,
      const int64_t num_layers,
      const torch::Tensor& block_mapping,
      const torch::Tensor& block_offsets,
      const int64_t num_requests,
      std::vector<long> const& events);

  void CopyLayerBlocks2GPU(
      torch::Tensor& src,
      std::vector<torch::Tensor> const& kv_caches,
      const int64_t num_layer,
      const torch::Tensor& block_mapping,
      const torch::Tensor& block_offsets,
      const torch::Tensor& req_ids,
      const std::vector<long>& events);

  ~KVStore() {
    if (_copy_thread.joinable()) {
      _copy_thread.join();
    }
  }

 private:
  std::thread _copy_thread;

};
