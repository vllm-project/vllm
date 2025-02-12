#pragma once

#include "cutlass/cutlass.h"
#include <climits>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {TORCH_CHECK(status == cutlass::Status::kSuccess, \
               cutlassGetStatusString(status))}

inline uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& file, int line, const std::string& message)
      : std::runtime_error("CUDA Error at " + file + ":" +
                           std::to_string(line) + " - " + message) {}
};

template <typename T>
void check(T result, const char* func, const char* file, int line) {
  if (result != cudaSuccess) {
    throw CudaException(
        file, line,
        std::string("[TensorRT-LLM][ERROR] CUDA runtime error in ") + func +
            ": " + cudaGetErrorString(static_cast<cudaError_t>(result)));
  }
}

template <typename T>
void checkEx(T result, std::initializer_list<T> const& validReturns,
             char const* const func, char const* const file, int const line) {
  if (std::all_of(std::begin(validReturns), std::end(validReturns),
                  [&result](T const& t) { return t != result; })) {
    throw TllmException(
        file, line,
        fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func,
               _cudaGetErrorEnum(result)));
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)
#define sync_check_cuda_error() \
  tensorrt_llm::common::syncAndCheck(__FILE__, __LINE__)

inline int getMaxSharedMemoryPerBlockOptin() {
  int device_id;
  int max_shared_memory_per_block;
  check_cuda_error(cudaGetDevice(&device_id));
  check_cuda_error(cudaDeviceGetAttribute(
      &max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id));
  return max_shared_memory_per_block;
}

inline int getSMVersion() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  check_cuda_error(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  check_cuda_error(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}