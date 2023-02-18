#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cassert>
#include <map>

void copy_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    assert(src_device.index() == dst_device.index());
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    assert(false);
  }

  void *src_ptr = src.data_ptr();
  void *dst_ptr = dst.data_ptr();

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, block_size, head_size]
  const int* __restrict__ slot_mapping, // [num_tokens]
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_idx = token_idx * n + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                            + head_idx * (head_size / x) * block_size * x
                            + x_idx * block_size * x
                            + block_offset * x
                            + x_offset;
    const int tgt_value_idx = block_idx * num_heads * block_size * head_size
                              + head_idx * block_size * head_size
                              + block_offset * head_size
                              + head_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_idx]);
  }
}

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping) {
  int num_tokens = key.size(0);
  int head_num = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  dim3 grid(num_tokens);
  dim3 block(std::min(head_num * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        head_num,
        head_size,
        block_size,
        x);
    });
}
