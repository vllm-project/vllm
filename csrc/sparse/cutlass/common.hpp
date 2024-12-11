#pragma once

#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <iostream>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                        \
  {                                                  \
    TORCH_CHECK(status == cutlass::Status::kSuccess, \
                cutlassGetStatusString(status))      \
  }

inline uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        device);
  return max_shared_mem_per_block_opt_in;
}

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id;
  cudaEvent_t _start;
  cudaEvent_t _stop;

  /// Constructor
  GpuTimer() : _stream_id(0) {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void start(cudaStream_t stream_id = 0) {
    _stream_id = stream_id;
    CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void stop() { CUDA_CHECK(cudaEventRecord(_stop, _stream_id)); }

  /// Return the elapsed time (in milliseconds)
  float elapsed_millis() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
};
