#pragma once

#include "../../vectorization.cuh"
#include "../../utils.cuh"

#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <cmath>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#ifndef USE_ROCM
  #include "stable/quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include "stable/quantization/w8a8/fp8/amd/quant_utils.cuh"
#endif

// Device properties cache for stable ABI compatibility
// Uses raw CUDA/HIP APIs instead of ATen functions
// Using inline ensures a single instance across all translation units
inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;

inline void do_init_vectors() {
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceCount failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void initDeviceVectors() {
  std::call_once(vectors_init_flag, do_init_vectors);
}

inline void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

// Get device properties using raw CUDA/HIP APIs (stable ABI compatible)
inline cudaDeviceProp* get_device_prop() {
  initDeviceVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(
        false, "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
  }

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

// Determines the preferred FP8 type for the current platform.
// Returns true for OCP format (Float8_e4m3fn), false for FNUZ format
// (Float8_e4m3fnuz). On CUDA this always returns true. On ROCm it checks
// device properties to determine the format.
inline bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto* dprops = get_device_prop();
  std::string device_arch = dprops->gcnArchName;
  // gfx94x devices use FNUZ format, others use OCP format
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
#ifndef USE_ROCM
  // Use hardware cvt instruction for fp8 on nvidia
  // Currently only support fp8_type = c10::Float8_e4m3fn
  return fp8::vec_conversion<fp8_type, float>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

}  // namespace vllm
