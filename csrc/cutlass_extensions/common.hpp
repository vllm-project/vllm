#pragma once

#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <iostream>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                        \
  {                                                               \
    cudaError_t error = status;                                   \
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error)); \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();
