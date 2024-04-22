#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#if defined(ENABLE_FP8_E5M2)
#include "quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#elif defined(ENABLE_FP8_E4M3)
#include "quantization/fp8/amd_detail/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 __nv_bfloat16;
#endif

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char *src_ptr = static_cast<char*>(src.data_ptr());
  char *dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const at::cuda::OptionalCUDAGuard device_guard(src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
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

namespace vllm {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

} // namespace vllm

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
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt64).to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
    key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
      vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int64_t>(),
        numel_per_block);
    }));
}

namespace vllm {

template<typename scalar_t, typename cache_t, bool is_fp8_kv_cache>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
  cache_t* __restrict__ value_cache,          // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x,
  const float kv_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (is_fp8_kv_cache) {
#if defined(ENABLE_FP8_E5M2)
      key_cache[tgt_key_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_key);
      value_cache[tgt_value_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_value);
#elif defined(ENABLE_FP8_E4M3)
      key_cache[tgt_key_idx] = fp8_e4m3::scaled_vec_conversion<uint8_t, scalar_t>(tgt_key, kv_scale);
      value_cache[tgt_value_idx] = fp8_e4m3::scaled_vec_conversion<uint8_t, scalar_t>(tgt_value, kv_scale);
#else
      assert(false);
#endif
    } else {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    }
  }
}

template<typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ k_cache,             // [num_blocks, block_size, num_heads, head_size]
  scalar_t* __restrict__ v_cache,             // [num_blocks, block_size, num_heads, head_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int block_stride,
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_value_idx = block_idx * block_stride
                              + block_offset * num_heads * head_size
                              + head_idx * head_size
                              + head_offset;
    k_cache[tgt_value_idx] = key[src_key_idx];
    v_cache[tgt_value_idx] = value[src_value_idx];
  }
}

} // namespace vllm

#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, IS_FP8_KV_CACHE)                                     \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, IS_FP8_KV_CACHE><<<grid, block, 0, stream>>>(      \
    reinterpret_cast<KV_T*>(key.data_ptr()),                                                       \
    reinterpret_cast<KV_T*>(value.data_ptr()),                                                     \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                              \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                            \
    slot_mapping.data_ptr<int64_t>(),                                                              \
    key_stride,                                                                                    \
    value_stride,                                                                                  \
    num_heads,                                                                                     \
    head_size,                                                                                     \
    block_size,                                                                                    \
    x,                                                                                             \
    kv_scale);

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping,  // [num_tokens]
  const std::string& kv_cache_dtype,
  const float kv_scale)
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kv_cache_dtype == "auto") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE(float, float, false);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint16_t, false);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, __nv_bfloat16, false);
    }
  } else if (kv_cache_dtype == "fp8") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE(float, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, uint8_t, true);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}

void reshape_and_cache_flash(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& k_cache,       // [num_blocks, block_size, num_heads, head_size]
  torch::Tensor& v_cache,       // [num_blocks, block_size, num_heads, head_size]
  torch::Tensor& slot_mapping,  // [num_tokens]
  const std::string& kv_cache_dtype)
{
  if (kv_cache_dtype != "auto") {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = k_cache.size(1);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int block_stride = k_cache.stride(0);
  TORCH_CHECK(k_cache.stride(0) == v_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_flash",
    [&] {
      vllm::reshape_and_cache_flash_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        k_cache.data_ptr<scalar_t>(),
        v_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        block_stride,
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size);
    });
}

namespace vllm {

template<typename Tout, typename Tin>
__global__ void convert_fp8_kernel(
  const Tin* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
#if defined(ENABLE_FP8_E5M2)
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, Tin>(src_cache[idx]);
#elif defined(ENABLE_FP8_E4M3)
    dst_cache[idx] = fp8_e4m3::vec_conversion<Tout, Tin>(src_cache[idx]);
#else
    assert(false);
#endif
  }
}

} // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin)                                 \
  vllm::convert_fp8_kernel<Tout, Tin><<<grid, block, 0, stream>>>(  \
    reinterpret_cast<Tin*>(src_cache.data_ptr()),                   \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                  \
    block_stride);

void convert_fp8(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache)
{
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(
    src_device.index() == dst_device.index(),
    "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (src_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8(uint8_t, float);
  } else if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8(uint8_t, uint16_t);
  } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8(uint8_t, __nv_bfloat16);
  } else if (dst_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8(float, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8(uint16_t, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8(__nv_bfloat16, uint8_t);
  }
}
