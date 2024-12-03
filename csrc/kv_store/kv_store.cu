#include "kv_store.hpp"



#define CHECK_CUDA(x) {\
  cudaError_t err = (x);\
  if (err != cudaSuccess) {\
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
    std::abort();\
  }\
}

// src layout: [2, num_blocks, block_size, num_kv_heads, head_size]
// dst layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
void KVStore::CopyIncompleteBlocks(torch::Tensor& src, torch::Tensor& dst,
    const int64_t layer_id,
    const torch::Tensor& incomplete_block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else {
    TORCH_CHECK(false, "only support copy from GPU to CPU");
  }
  TORCH_CHECK(incomplete_block_mapping.device().is_cpu(),
          "block_mapping must be on CPU");

  const int64_t slot_size_in_bytes = src.element_size() * src[0][0][0].numel();
  const at::cuda::OptionalCUDAGuard device_guard(src_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_items = incomplete_block_mapping.size(0);
  for (size_t i = 0; i < num_items; i++) {
    int64_t src_block    = incomplete_block_mapping[i][0].item<int64_t>();
    int64_t start_offset = incomplete_block_mapping[i][1].item<int64_t>();
    int64_t end_offset   = incomplete_block_mapping[i][2].item<int64_t>();
    int64_t dst_block    = incomplete_block_mapping[i][3].item<int64_t>();
    int64_t copy_nbytes = (end_offset - start_offset ) * slot_size_in_bytes;
    char* src_ptr = reinterpret_cast<char*>(src[0][src_block].data_ptr());
    char* dst_ptr = reinterpret_cast<char*>(
        dst[dst_block][0][layer_id].data_ptr());
    start_offset *= slot_size_in_bytes;
    CHECK_CUDA(cudaMemcpyAsync(dst_ptr + start_offset, src_ptr + start_offset,
                    copy_nbytes, memcpy_type, stream));
    src_ptr = reinterpret_cast<char*>(src[1][src_block].data_ptr());
    dst_ptr = reinterpret_cast<char*>(dst[dst_block][1][layer_id].data_ptr());
    CHECK_CUDA(cudaMemcpyAsync(dst_ptr + start_offset, src_ptr + start_offset,
                    copy_nbytes, memcpy_type, stream));
  }
}

// src layout: [2, num_blocks, block_size, num_kv_heads, head_size]
// dst layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
void KVStore::CopyBlocks2CPU(torch::Tensor& src, torch::Tensor& dst,
                            const int64_t layer_id,
                            const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else {
    TORCH_CHECK(false, "only support copy from GPU to CPU");
  }
  TORCH_CHECK(block_mapping.device().is_cpu(),
          "block_mapping must be on CPU");
  const int64_t src_block_numel = src[0][0].numel();
  const int64_t dst_block_numel = dst[0][0][0].numel();
  TORCH_CHECK(src_block_numel == dst_block_numel,
          "src and dst must have the same number of elements");
  const int64_t block_size_in_bytes = src.element_size() * src_block_numel;
  const at::cuda::OptionalCUDAGuard device_guard(src_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    // key
    char* src_ptr = reinterpret_cast<char*>(
        src[0][src_block_number].data_ptr());
    char* dst_ptr = reinterpret_cast<char*>(
        dst[dst_block_number][0][layer_id].data_ptr());
    CHECK_CUDA(cudaMemcpyAsync(
          dst_ptr, src_ptr, block_size_in_bytes, memcpy_type, stream));
    // value
    src_ptr = reinterpret_cast<char*>(
        src[1][src_block_number].data_ptr());
    dst_ptr = reinterpret_cast<char*>(
        dst[dst_block_number][1][layer_id].data_ptr());
    CHECK_CUDA(cudaMemcpyAsync(
          dst_ptr, src_ptr, block_size_in_bytes, memcpy_type, stream));
  }
}


namespace vllm {

// Grid: (num_layers, num_blocks)
template <typename scalar_t>
__global__ void kv_store_copy_blocks_kernel(
    scalar_t *src,
    int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
    const int64_t *block_mapping, const int64_t *block_offsets,
    const int request_idx,
    const int64_t numel_per_block) {
  int64_t layer_idx = blockIdx.x;
  int64_t pair_idx = blockIdx.y;
  int64_t num_layer = gridDim.x;
  scalar_t *key_cache =
    reinterpret_cast<scalar_t *>(key_cache_ptrs[layer_idx]);
  scalar_t *value_cache =
    reinterpret_cast<scalar_t *>(value_cache_ptrs[layer_idx]);
  int64_t block_mapping_idx = block_offsets[request_idx] + pair_idx;
  int64_t dst_block_number = block_mapping[2 * block_mapping_idx + 1];
  scalar_t *key_block = key_cache + dst_block_number * numel_per_block;
  scalar_t *src_key_block = (src + pair_idx * 2 * num_layer * numel_per_block
                            + layer_idx * numel_per_block);
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    key_block[i] = src_key_block[i];
  }
  scalar_t *value_block = value_cache + dst_block_number * numel_per_block;
  scalar_t *src_value_block = (src
                            + (pair_idx * 2 + 1) * num_layer * numel_per_block
                            + layer_idx * numel_per_block);
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    value_block[i] = src_value_block[i];
  }
}

} // namespace vllm

std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i != vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}


// src layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
// kv_caches layout: [laysers, [2, num_blocks, block_size, num_kv_heads, head_size]]
void KVStore::CopyBlocks2GPUBatch(torch::Tensor& src,
                            std::vector<torch::Tensor> const& kv_caches,
                            const int64_t num_layers,
                            const torch::Tensor& block_mapping,
                            const torch::Tensor& block_offsets,
                            const int64_t num_requests,
                            std::vector<long> const& events) {
  torch::Device src_device = src.device();
  torch::Device dst_device = kv_caches[0].device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "only support copy from CPU to GPU");
  }
  TORCH_CHECK(block_mapping.device().is_cpu(),
          "block_mapping must be on CPU");
  TORCH_CHECK(block_offsets.device().is_cpu(),
          "block_offsets must be on CPU");
  const at::cuda::OptionalCUDAGuard device_guard(dst_device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto block_mapping_gpu = block_mapping.to(dst_device,
                                            block_mapping.scalar_type(),
                                            /*non_blocking=*/true);
  auto block_offsets_gpu = block_offsets.to(dst_device,
                                            block_offsets.scalar_type(),
                                            /*non_blocking=*/true);
  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(kv_caches[layer_idx][0].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(kv_caches[layer_idx][1].data_ptr());
  }
  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(dst_device, /*non_blocking=*/true);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(dst_device, /*non_blocking=*/true);

  for (size_t i = 0; i < num_requests; i++) {
    const int64_t start_idx = block_offsets[i].item<int64_t>();
    const int64_t end_idx = block_offsets[i + 1].item<int64_t>();
    const int64_t num_blocks = end_idx - start_idx;
    auto options = torch::TensorOptions()
                      .dtype(kv_caches[0].dtype())
                      .device(dst_device);
    std::vector<int64_t> shape = src.sizes().vec();
    shape[0] = num_blocks;
    // XXX: may cause out of memory in VLLM framework
    torch::Tensor trans_buffer = torch::empty(shape, options);
    for (size_t j = 0; j < num_blocks; j++) {
      int64_t idx = (start_idx + j);
      int64_t src_block_number = block_mapping[idx][0].item<int64_t>();
      char* src_ptr = reinterpret_cast<char*>(src[src_block_number].data_ptr());
      char* dst_ptr = reinterpret_cast<char*>(trans_buffer[j].data_ptr());
      int64_t trans_nbytes = src[0].element_size() * src[0].numel();
      CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, trans_nbytes,
            memcpy_type, stream));
    }
    const int numel_per_block = src[0][0][0].numel();
    const dim3 grid(num_layers, num_blocks);
    const dim3 block(std::min(1024, numel_per_block));
    VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
        src.scalar_type(), "kv_store_copy_blocks_kernel", ([&] {
          vllm::kv_store_copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
              trans_buffer.data_ptr<scalar_t>(),
              key_cache_ptrs_tensor.data_ptr<int64_t>(),
              value_cache_ptrs_tensor.data_ptr<int64_t>(),
              block_mapping_gpu.data_ptr<int64_t>(),
              block_offsets_gpu.data_ptr<int64_t>(),
              i,
              numel_per_block);
        }));
    CHECK_CUDA(cudaEventRecord(reinterpret_cast<cudaEvent_t>(events[i]), stream));
  }
}

namespace vllm {

// src layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
// key layout: [num_blocks, block_size, num_kv_heads, head_size]
// value layout: [num_blocks, block_size, num_kv_heads, head_size]
// Grid: (num_blocks)
template <typename scalar_t>
__global__ void kv_store_copy_blocks_kernel(
    scalar_t* src,
    scalar_t* key_cache,
    scalar_t* value_cache,
    const int64_t* block_mapping,
    const int64_t* block_offsets,
    const int request_idx,
    const int64_t layer_id,
    const int64_t num_layer,
    const int64_t numel_per_block) {

  int pair_idx = blockIdx.x;
  int64_t block_mapping_idx = block_offsets[request_idx] + pair_idx;
  int64_t src_block_number = block_mapping[2 * block_mapping_idx];
  int64_t dst_block_number = block_mapping[2 * block_mapping_idx + 1];
  scalar_t* src_key_block = src
    + src_block_number * 2 * num_layer * numel_per_block
    + layer_id * numel_per_block;
  scalar_t* dst_key_block = key_cache
    + dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    dst_key_block[i] = src_key_block[i];
  }
  scalar_t* src_value_block = src
    + (src_block_number * 2 + 1) * num_layer * numel_per_block
    + layer_id * numel_per_block;
  scalar_t* dst_value_block = value_cache
    + dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    dst_value_block[i] = src_value_block[i];
  }
}

}; // namespace vllm

namespace {

// src layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
// kv_caches layout: [laysers, [2, num_blocks, block_size, num_kv_heads, head_size]]
void CopyLayerBlocks2GPUKernelFunc(
    const torch::Tensor& src,
    std::vector<torch::Tensor> const& kv_caches,
    const int64_t num_layer,
    const torch::Tensor& block_mapping,
    const torch::Tensor& block_offsets,
    const torch::Tensor& req_ids,
    const std::vector<long>& events,
    const at::cuda::CUDAStream& stream) { // is the current stream
  size_t num_requests = req_ids.size(0);
  const int numel_per_block = src[0][0][0].numel();
  const int64_t block_nbytes = numel_per_block * src.element_size();
  torch::Device src_device = src.device();
  torch::Device dst_device = kv_caches[0].device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "only support copy from CPU to GPU");
  }
  TORCH_CHECK(block_mapping.device().is_cpu(),
      "block_mapping must be on CPU");
  TORCH_CHECK(block_offsets.device().is_cpu(),
      "block_offsets must be on CPU");
  TORCH_CHECK(req_ids.device().is_cpu(),
      "req_ids must be on CPU");
  const at::cuda::OptionalCUDAGuard device_guard(dst_device);
  auto block_mapping_gpu = block_mapping.to(dst_device,
                                            block_mapping.scalar_type(),
                                            /*non_blocking=*/true);
  auto block_offsets_gpu = block_offsets.to(dst_device,
                                            block_offsets.scalar_type(),
                                            /*non_blocking=*/true);
  for (size_t i = 0; i < num_requests; i++) {
    const int64_t req_id = req_ids[i].item<int64_t>();
    const int64_t start_idx = block_offsets[i].item<int64_t>();
    const int64_t end_idx = block_offsets[i + 1].item<int64_t>();
    const int64_t num_blocks = end_idx - start_idx;

    for (int64_t layer_id = 0; layer_id < num_layer; layer_id++) {
      if (num_blocks >= 2) { // if blocks are too many, use kernel
        const dim3 grid(num_blocks);
        const dim3 block(std::min(1024, numel_per_block));
        VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
            src.scalar_type(), "kv_store_copy_blocks_kernel", ([&] {
              vllm::kv_store_copy_blocks_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                  src.data_ptr<scalar_t>(),
                  kv_caches[layer_id][0].data_ptr<scalar_t>(),
                  kv_caches[layer_id][1].data_ptr<scalar_t>(),
                  block_mapping_gpu.data_ptr<int64_t>(),
                  block_offsets_gpu.data_ptr<int64_t>(),
                  i,
                  layer_id,
                  num_layer,
                  numel_per_block);
            }));
      }
      else {
        for (size_t j = 0; j < num_blocks; j++) {
          int64_t idx = (start_idx + j);
          int64_t src_block_number = block_mapping[idx][0].item<int64_t>();
          int64_t dst_block_number = block_mapping[idx][1].item<int64_t>();
          char* src_ptr = reinterpret_cast<char*>(
              src[src_block_number][0][layer_id].data_ptr());
          char* dst_ptr = reinterpret_cast<char*>(
              kv_caches[layer_id][0][dst_block_number].data_ptr());
          CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr,
                block_nbytes, memcpy_type, stream));
          src_ptr = reinterpret_cast<char*>(
              src[src_block_number][1][layer_id].data_ptr());
          dst_ptr = reinterpret_cast<char*>(
              kv_caches[layer_id][1][dst_block_number].data_ptr());
          CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr,
                block_nbytes, memcpy_type, stream));
        }
      }
      CHECK_CUDA(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(events[i * num_layer + layer_id]),
            stream));
    }
  }
}

// src layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
// kv_caches layout: [laysers, [2, num_blocks, block_size, num_kv_heads, head_size]]
void CopyLayerBlocks2GPUThreadFunc(
    const torch::Tensor& src,
    std::vector<torch::Tensor> const& kv_caches,
    const int64_t num_layer,
    const torch::Tensor& block_mapping,
    const torch::Tensor& block_offsets,
    const torch::Tensor& req_ids,
    const std::vector<long>& events,
    const at::cuda::CUDAStream& stream) {
  size_t num_requests = req_ids.size(0);
  const int64_t block_nbytes =
    kv_caches[0][0][0].numel() * kv_caches[0].element_size();
  torch::Device src_device = src.device();
  torch::Device dst_device = kv_caches[0].device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "only support copy from CPU to GPU");
  }
  TORCH_CHECK(block_mapping.device().is_cpu(),
      "block_mapping must be on CPU");
  TORCH_CHECK(block_offsets.device().is_cpu(),
      "block_offsets must be on CPU");
  TORCH_CHECK(req_ids.device().is_cpu(),
      "req_ids must be on CPU");
  const at::cuda::OptionalCUDAGuard device_guard(dst_device);
  for (size_t i = 0; i < num_requests; i++) {
    const int64_t req_id = req_ids[i].item<int64_t>();
    const int64_t start_idx = block_offsets[i].item<int64_t>();
    const int64_t end_idx = block_offsets[i + 1].item<int64_t>();
    const int64_t num_blocks = end_idx - start_idx;
    for (int64_t layer_id = 0; layer_id < num_layer; layer_id++) {
      for (size_t j = 0; j < num_blocks; j++) {
        int64_t idx = (start_idx + j);
        int64_t src_block_number = block_mapping[idx][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[idx][1].item<int64_t>();
        char* src_ptr = reinterpret_cast<char*>(
            src[src_block_number][0][layer_id].data_ptr());
        char* dst_ptr = reinterpret_cast<char*>(
            kv_caches[layer_id][0][dst_block_number].data_ptr());
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr,
              block_nbytes, memcpy_type, stream));
        src_ptr = reinterpret_cast<char*>(
            src[src_block_number][1][layer_id].data_ptr());
        dst_ptr = reinterpret_cast<char*>(
            kv_caches[layer_id][1][dst_block_number].data_ptr());
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr,
              block_nbytes, memcpy_type, stream));
      }
      CHECK_CUDA(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(events[i * num_layer + layer_id]),
            stream));
    }
  }
}

}; // namespace


// src layout: [num_blocks, 2, num_layer, block_size, num_kv_heads*head_size]
// kv_caches layout: [laysers, [2, num_blocks, block_size, num_kv_heads, head_size]]
void KVStore::CopyLayerBlocks2GPU(
    torch::Tensor& src,
    std::vector<torch::Tensor> const& kv_caches,
    const int64_t num_layer,
    const torch::Tensor& block_mapping,
    const torch::Tensor& block_offsets,
    const torch::Tensor& req_ids,
    const std::vector<long>& events) {
  if (block_mapping.size(0) == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  /*
  if (_copy_thread.joinable()) {
    _copy_thread.join();
  }
  _copy_thread = std::thread(CopyLayerBlocks2GPUThreadFunc,
      src, kv_caches, num_layer,
      block_mapping.clone(),
      block_offsets.clone(),
      req_ids.clone(),
      events, stream);
  */
  CopyLayerBlocks2GPUKernelFunc(src, kv_caches, num_layer,
      block_mapping, block_offsets, req_ids, events, stream);
}

