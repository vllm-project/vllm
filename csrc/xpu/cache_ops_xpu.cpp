// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// clang-format on
#include "xpu_types.h"

#include <torch/extension.h>
#include "utils.h"

template <typename scalar_t>
void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key, // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache, // [num_blocks, num_heads, head_size/x,
                                      // block_size, x]
    scalar_t* __restrict__ value_cache, // [num_blocks, num_heads, head_size,
                                        // block_size]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const sycl::nd_item<3>& item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range(2)) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

template <typename scalar_t>
void call_reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(num_heads * head_size, 512));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          reshape_and_cache_kernel<sycl_t>(
              (const sycl_t* __restrict__)key,
              (const sycl_t* __restrict__)value,
              (sycl_t* __restrict__)key_cache,
              (sycl_t* __restrict__)value_cache,
              slot_mapping,
              key_stride,
              value_stride,
              num_heads,
              head_size,
              block_size,
              x,
              item_ct1);
        });
  });
}

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const float kv_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "call_reshape_and_cache_kernel", [&] {
        call_reshape_and_cache_kernel<scalar_t>(
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens,
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size,
            x);
      });
}

template <typename scalar_t>
void copy_blocks_kernel(
    int64_t* key_cache_ptrs,
    int64_t* value_cache_ptrs,
    const int64_t* __restrict__ block_mapping,
    const int numel_per_block,
    const sycl::nd_item<3>& item_ct1) {
  const int layer_idx = item_ct1.get_group(2);
  const int pair_idx = item_ct1.get_group(1);

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = item_ct1.get_local_id(2); i < numel_per_block;
       i += item_ct1.get_local_range(2)) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = item_ct1.get_local_id(2); i < numel_per_block;
       i += item_ct1.get_local_range(2)) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

template <typename scalar_t>
void call_copy_blocks_kernel(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_xpu());
  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }
  // Create block mapping array.
  std::vector<int64_t> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    for (int64_t dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int64_t* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;
  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);
  torch::Tensor block_mapping_tensor =
      torch::from_blob(block_mapping_array, {2 * num_pairs}, torch::kInt64)
          .to(cache_device);
  auto k_ptr = key_cache_ptrs_tensor.data_ptr<int64_t>();
  auto v_ptr = value_cache_ptrs_tensor.data_ptr<int64_t>();
  auto b_ptr = block_mapping_tensor.data_ptr<int64_t>();
  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();

  sycl::range<3> grid(1, num_pairs, num_layers);
  sycl::range<3> block(1, 1, std::min(1024, numel_per_block));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          copy_blocks_kernel<sycl_t>(
              k_ptr, v_ptr, b_ptr, numel_per_block, item_ct1);
        });
  });
}

void copy_blocks(
    std::vector<torch::Tensor>& key_caches,
    std::vector<torch::Tensor>& value_caches,
    const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key_caches[0].scalar_type(), "call_copy_blocks_kernel", [&] {
        call_copy_blocks_kernel<scalar_t>(
            key_caches, value_caches, block_mapping);
      });
}

void swap_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::map<int64_t, int64_t>& block_mapping) {
  char* src_ptr = (char*)src.data_ptr();
  char* dst_ptr = (char*)dst.data_ptr();

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  auto& queue = vllm::xpu::vllmGetQueue();

  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    queue.memcpy(
        dst_ptr + dst_offset, src_ptr + src_offset, block_size_in_bytes);
  }
  queue.wait();
}

template <typename scalar_t>
void gather_cached_kv_kernel(
    scalar_t* __restrict__ key, // [num_tokens, [stride], num_heads, head_size]
    scalar_t* __restrict__ value, // [num_tokens, [stride], num_heads,
                                  // head_size]
    const scalar_t* __restrict__ key_cache, // [num_blocks, num_heads,
                                            // head_size/x, block_size, x]
    const scalar_t* __restrict__ value_cache, // [num_blocks, num_heads,
                                              // head_size, block_size]
    const int* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const sycl::nd_item<3>& item_ct1) {
  const int token_idx = item_ct1.get_group(2);
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int num_tokens = num_heads * head_size;
  for (int i = item_ct1.get_local_id(2); i < num_tokens;
       i += item_ct1.get_local_range(2)) {
    const int tgt_key_idx = token_idx * key_stride + i;
    const int tgt_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx =
        head_offset / x; // the offset of the [head_size/x] dimension
    const int x_offset = head_offset % x;

    const int src_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int src_value_idx = block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;

    key[tgt_key_idx] = VLLM_LDG(&key_cache[src_key_idx]);
    value[tgt_value_idx] = VLLM_LDG(&value_cache[src_value_idx]);
  }
}

template <typename scalar_t>
void gather_cached_kv_kernel_optimized(
    scalar_t* __restrict__ key, // [num_tokens, [stride], num_heads, head_size]
    scalar_t* __restrict__ value, // [num_tokens, [stride], num_heads,
                                  // head_size]
    const scalar_t* __restrict__ key_cache, // [num_blocks, num_heads,
                                            // head_size/x, block_size, x]
    const scalar_t* __restrict__ value_cache, // [num_blocks, num_heads,
                                              // head_size, block_size]
    const int* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const sycl::nd_item<3>& item_ct1) {
  const int token_idx = item_ct1.get_group(2);
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int dim = num_heads * head_size;
  assert(dim % 4 == 0); // this is true for known use cases
  const int unroll_factor = 4;
  const int unrolled_dim = dim / unroll_factor;

  for (int i = item_ct1.get_local_id(2); i < unrolled_dim;
       i += item_ct1.get_local_range(2)) {
    int tgt_key_indices[unroll_factor];
    int tgt_value_indices[unroll_factor];
    int src_key_indices[unroll_factor];
    int src_value_indices[unroll_factor];
    scalar_t keys_to_store[unroll_factor];
    scalar_t values_to_store[unroll_factor];

#pragma unroll
    for (int j = 0; j < unroll_factor; ++j) {
      int index = i + j * unrolled_dim;

      const int tgt_key_idx = token_idx * key_stride + index;
      const int tgt_value_idx = token_idx * value_stride + index;

      const int head_idx = index / head_size;
      const int head_offset = index % head_size;
      const int x_idx = head_offset / x;
      const int x_offset = head_offset % x;

      const int src_key_idx =
          block_idx * num_heads * (head_size / x) * block_size * x +
          head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
          block_offset * x + x_offset;
      const int src_value_idx = block_idx * num_heads * head_size * block_size +
          head_idx * head_size * block_size + head_offset * block_size +
          block_offset;

      tgt_key_indices[j] = tgt_key_idx;
      tgt_value_indices[j] = tgt_value_idx;
      src_key_indices[j] = src_key_idx;
      src_value_indices[j] = src_value_idx;

      keys_to_store[j] = VLLM_LDG(&key_cache[src_key_idx]);
      values_to_store[j] = VLLM_LDG(&value_cache[src_value_idx]);
    }

#pragma unroll
    for (int j = 0; j < unroll_factor; ++j) {
      key[tgt_key_indices[j]] = keys_to_store[j];
      value[tgt_value_indices[j]] = values_to_store[j];
    }
  }
}

template <typename scalar_t>
void call_gather_cached_kv_kernel_optimized(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  auto key_ptr = key.data_ptr<scalar_t>();
  auto value_ptr = value.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
  auto slot_mapping_ptr = slot_mapping.data_ptr<int>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(num_heads * head_size, 512));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          gather_cached_kv_kernel_optimized<sycl_t>(
              (sycl_t* __restrict__)key_ptr,
              (sycl_t* __restrict__)value_ptr,
              (const sycl_t* __restrict__)key_cache_ptr,
              (const sycl_t* __restrict__)value_cache_ptr,
              slot_mapping_ptr,
              key_stride,
              value_stride,
              num_heads,
              head_size,
              block_size,
              x,
              item_ct1);
        });
  });
}

void gather_cached_kv(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping) {
  VLLM_XPU_DISPATCH_FLOATING_TYPES(
      key_cache[0].scalar_type(),
      "call_gather_cached_kv_kernel_optimized",
      [&] {
        call_gather_cached_kv_kernel_optimized<scalar_t>(
            key, value, key_cache, value_cache, slot_mapping);
      });
}
