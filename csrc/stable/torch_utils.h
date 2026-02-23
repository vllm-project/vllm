#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Device properties cache for stable ABI compatibility.
// Uses raw CUDA/HIP APIs instead of ATen functions.
// Thread-safe: each device's properties are queried exactly once.
inline std::deque<std::once_flag> device_prop_flags;
inline std::vector<cudaDeviceProp> device_prop_cache;
inline std::once_flag device_prop_vectors_init_flag;

inline void init_device_prop_vectors() {
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceCount failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_prop_flags.resize(device_count);
  device_prop_cache.resize(device_count);
}

inline void init_device_prop(int device_index) {
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_prop_cache[device_index] = prop;
}

inline cudaDeviceProp* get_device_prop() {
  std::call_once(device_prop_vectors_init_flag, init_device_prop_vectors);
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(
        false, "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
  }
  std::call_once(device_prop_flags[device_index], init_device_prop,
                 device_index);
  return &device_prop_cache[device_index];
}

// Utility to get the current CUDA stream for a given device using stable APIs.
// Returns a cudaStream_t for use in kernel launches.
inline cudaStream_t get_current_cuda_stream(int32_t device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}
