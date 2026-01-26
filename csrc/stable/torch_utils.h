#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>

// Utility to get the current CUDA stream for a given device using stable APIs.
// Returns a cudaStream_t for use in kernel launches.
inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

// Utility to get the storage size in bytes for a tensor.
// Equivalent to t.storage().nbytes() in the unstable API.
inline int64_t get_storage_nbytes(const torch::stable::Tensor& t) {
  int64_t storage_size = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(t.get(), &storage_size));
  return storage_size;
}

// Stable ABI equivalent of TORCH_CHECK_EQ - checks that two values are equal.
// Automatically generates an error message showing both values if they differ.
#define STD_TORCH_CHECK_EQ(a, b) \
  STD_TORCH_CHECK((a) == (b), #a " (", (a), ") != " #b " (", (b), ")")

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
// Note: This currently throws a runtime_error. When PyTorch adds
// NotImplementedError support to the headeronly API, this should be updated.
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)
