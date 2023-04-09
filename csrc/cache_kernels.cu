#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

void swap_blocks(
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

namespace cacheflow {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int src_block_number = block_mapping[2 * pair_idx];
  int dst_block_number = block_mapping[2 * pair_idx + 1];

  const int src_block_offset = src_block_number * numel_per_block;
  const int dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

} // namespace cacheflow

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }
  // Create block mapping array.
  std::vector<int> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int src_block_number = pair.first;
    for (int dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt).to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
      cacheflow::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int>(),
        numel_per_block);
    }));
}

namespace cacheflow {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping, // [num_tokens]
  const int key_stride,
  const int value_stride,
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
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                            + head_idx * (head_size / x) * block_size * x
                            + x_idx * block_size * x
                            + block_offset * x
                            + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size
                              + head_idx * head_size * block_size
                              + head_offset * block_size
                              + block_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

// Grid: (num_blocks, num_heads).
template<typename scalar_t>
__global__ void gather_cached_kv_kernel(
  scalar_t* __restrict__ out,             // [cu_seqlens_k[-1], 3(QKV), num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ cu_seqlens_k,   // aka 'cu_seqlens_k' in '_flash_attn_forward', or 'context_lens' in cacheflow
  const int num_seqs,
  const int max_num_blocks_per_seq,
  const int head_size,
  const int block_size) {
    // Each CUDA gird is mapped to (num_blocks, num_heads).
    const int block_idx = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int head_idx = blockIdx.y;
    const int num_heads = gridDim.y;
    // Each CUDA block is responsible for (head_size, block_size).
    const int thread_idx = threadIdx.x;
    const int num_threads = blockDim.x;
    // in the original attention kernel, each thread group fetch x elements at a time.
    constexpr int x = 16 / sizeof(scalar_t);

    // the index of the sequence this thread is working on.
    int seq_idx;
    // the index of the block in the sequence this thread is working on.
    int local_block_idx;
    // calculate the sequence index and block index in the sequence.
    int num_total_blocks = 0;
#pragma unroll
    for (int i = 0; i < num_seqs; ++i) {
      int context_len = cu_seqlens_k[i + 1] - cu_seqlens_k[i];
      int num_blocks = (context_len + block_size - 1) / block_size;
      num_total_blocks += num_blocks;
      if (num_total_blocks > block_idx) {
        seq_idx = i;
        local_block_idx = block_idx - (num_total_blocks - num_blocks);
        break;
      }
    }
    // const int context_len = cu_seqlens_k[seq_idx];
    // const int num_blocks = (context_len + block_size - 1) / block_size;
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    const int physical_block_number = block_table[local_block_idx];

    // number of chunks handled by a CUDA block.
    const int n_chunks = (head_size * block_size + (num_threads - 1)) / num_threads;
    const int physical_cache_offset = (physical_block_number * num_heads + head_idx) * head_size * block_size;

    // The common output pointer base used by both key and value:
    scalar_t* common_out = out + (block_idx * block_size) * 3 * num_heads * head_size
                               + head_idx * head_size;
    // key is the second tensor in QKV, so qkv_offset = 1
    scalar_t* key_out = common_out + 1 * num_heads * head_size;
    // value is the third tensor in QKV, so qkv_offset = 2
    scalar_t* value_out = common_out + 2 * num_heads * head_size;

    // process key in chunks
#pragma unroll
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      const int offset = chunk_idx * num_threads + thread_idx;
      if (offset >= head_size * block_size) {
        break;
      }
      // calculate offsets in [head_size/x, block_size, x]
      const int head_offset = offset / x / block_size;
      const int block_offset = offset / x % block_size;
      const int x_offset = offset % x;

      const scalar_t* k_ptr = k_cache + physical_cache_offset + offset;
      scalar_t* out_ptr = key_out + block_offset * 3 * num_heads * head_size 
                                  + head_offset * x + x_offset;
      *out_ptr = __ldg(k_ptr);
    }

    // process value in chunks
#pragma unroll
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      const int offset = chunk_idx * num_threads + thread_idx;
      if (offset >= head_size * block_size) {
        break;
      }
      // calculate offsets in [head_size, block_size]
      const int head_offset = offset / block_size;
      const int block_offset = offset % block_size;

      const scalar_t* v_ptr = v_cache + physical_cache_offset + offset;
      scalar_t* out_ptr = value_out + block_offset * 3 * num_heads * head_size + head_offset;
      *out_ptr = __ldg(v_ptr);
    }
}


// Grid: (num_blocks, block_size).
template<typename scalar_t>
__global__ void gather_cached_kv_kernel_2(
  scalar_t* __restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
  scalar_t* __restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
  const scalar_t* __restrict__ key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int num_tokens = num_heads * head_size;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
      const int tgt_key_idx = token_idx * key_stride + i;
      const int tgt_value_idx = token_idx * value_stride + i;
  
      const int head_idx = i / head_size;
      const int head_offset = i % head_size;
      const int x_idx = head_offset / x;  // the offset of the [head_size/x] dimension
      const int x_offset = head_offset % x;
  
      const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                              + head_idx * (head_size / x) * block_size * x
                              + x_idx * block_size * x
                              + block_offset * x
                              + x_offset;
      const int src_value_idx = block_idx * num_heads * head_size * block_size
                                + head_idx * head_size * block_size
                                + head_offset * block_size
                                + block_offset;

      key[tgt_key_idx] = __ldg(&key_cache[src_key_idx]);
      value[tgt_value_idx] = __ldg(&value_cache[src_value_idx]);
    }
}

} // namespace cacheflow

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      cacheflow::reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}


void gather_cached_kv(
  torch::Tensor& key,           // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [in]  [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [in]  [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [in]  [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    key.scalar_type(),
    "gather_cached_kv_kernel_2",
    [&] {
      cacheflow::gather_cached_kv_kernel_2<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}

/*
// same group of threads will be working on the same block
void gather_cached_kv(
  torch::Tensor& qkv_out,         // [cu_seqlens_k[-1], 3(QKV), num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& cu_seqlens_k,    // aka 'cu_seqlens_k' in '_flash_attn_forward', or 'context_lens' in cacheflow
  torch::Tensor& seqlens_k,       // CPU version of 'cu_seqlens_k'
  torch::Tensor& block_tables) {  // [num_seqs, max_num_blocks_per_seq]
    const int num_seqs = cu_seqlens_k.size(0) - 1;
    const int num_heads = value_cache.size(1);
    const int head_size = value_cache.size(2);
    const int block_size = value_cache.size(3);
    // const int x = key_cache.size(4);
    const int max_num_blocks_per_seq = block_tables.size(1);
    const int* context_lens_ptr = cu_seqlens_k.data_ptr<int>();
    const int* cpu_context_lens_ptr = seqlens_k.data_ptr<int>();

    // calculate the total amount of blocks
    int num_total_blocks = 0;
    for (int i = 0; i < num_seqs; ++i) {
      int context_len = cpu_context_lens_ptr[i + 1] - cpu_context_lens_ptr[i];
      int num_blocks = (context_len + block_size - 1) / block_size;
      num_total_blocks += num_blocks;
    }

    constexpr int NUM_THREADS = 256;
    dim3 grid(num_total_blocks, num_heads);
    dim3 block(NUM_THREADS);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      key_cache.scalar_type(),
      "gather_cached_kv_kernel",
      [&] {
        cacheflow::gather_cached_kv_kernel<scalar_t><<<grid, block, 0, stream>>>(
          qkv_out.data_ptr<scalar_t>(),
          key_cache.data_ptr<scalar_t>(),
          value_cache.data_ptr<scalar_t>(),
          block_tables.data_ptr<int>(),
          cu_seqlens_k.data_ptr<int>(),
          num_seqs,
          max_num_blocks_per_seq,
          head_size,
          block_size);
      });
}
*/


// // same group of threads will be working on the same block
// void gather_cached_kv(
//   torch::Tensor& qkv_out,         // [cu_seqlens_k[-1], 3(QKV), num_heads, head_size]
//   torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
//   torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
//   torch::Tensor& cu_seqlens_k,    // aka 'cu_seqlens_k' in '_flash_attn_forward', or 'context_lens' in cacheflow
//   torch::Tensor& seqlens_k,       // CPU version of 'cu_seqlens_k'
//   torch::Tensor& block_tables) {  // [num_seqs, max_num_blocks_per_seq]
//     const int num_seqs = cu_seqlens_k.size(0) - 1;
//     const int num_heads = value_cache.size(1);
//     const int head_size = value_cache.size(2);
//     const int block_size = value_cache.size(3);
//     // const int x = key_cache.size(4);
//     const int max_num_blocks_per_seq = block_tables.size(1);
//     const int* context_lens_ptr = cu_seqlens_k.data_ptr<int>();
//     const int* cpu_context_lens_ptr = seqlens_k.data_ptr<int>();

//     // calculate the total amount of blocks
//     int num_total_blocks = 0;
//     for (int i = 0; i < num_seqs; ++i) {
//       int context_len = cpu_context_lens_ptr[i + 1] - cpu_context_lens_ptr[i];
//       int num_blocks = (context_len + block_size - 1) / block_size;
//       num_total_blocks += num_blocks;
//     }

//     dim3 grid(num_total_blocks, block_size);
//     dim3 block(std::min(num_heads * head_size, 512));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//       key_cache.scalar_type(),
//       "gather_cached_kv_kernel_2",
//       [&] {
//         cacheflow::gather_cached_kv_kernel_2<scalar_t><<<grid, block, 0, stream>>>(
//           qkv_out.data_ptr<scalar_t>(),
//           key_cache.data_ptr<scalar_t>(),
//           value_cache.data_ptr<scalar_t>(),
//           block_tables.data_ptr<int>(),
//           cu_seqlens_k.data_ptr<int>(),
//           num_seqs,
//           max_num_blocks_per_seq,
//           num_heads,
//           head_size);
//       });
// }
