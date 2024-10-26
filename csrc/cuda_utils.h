#pragma once

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __forceinline__ __host__ __device__
  #define DEVICE_INLINE __forceinline__ __device__
  #define HOST_INLINE __forceinline__ __host__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

#include <torch/extension.h>
#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}
