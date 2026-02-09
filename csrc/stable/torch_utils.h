#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
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

// Stable ABI equivalent of TORCH_CHECK_EQ - checks that two values are equal.
// Automatically generates an error message showing both values if they differ.
#define STD_TORCH_CHECK_EQ(a, b) \
  STD_TORCH_CHECK((a) == (b), #a " (", (a), ") != " #b " (", (b), ")")
