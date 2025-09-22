#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>

// maximum blocks per SM cap
#ifndef VLLM_LAUNCH_BLOCKS_CAP
  #define VLLM_LAUNCH_BLOCKS_CAP 4
#endif

// compile-time estimate of max threads per SM for launch bounds.
#ifndef VLLM_MAX_THREADS_PER_SM
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    #define VLLM_MAX_THREADS_PER_SM 1536
  #else
    #define VLLM_MAX_THREADS_PER_SM 2048
  #endif
#endif

// compute the number of blocks per SM to request in __launch_bounds__
#define VLLM_BLOCKS_DIV(VAL) (VLLM_MAX_THREADS_PER_SM / (VAL))
#define VLLM_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0)                       \
       ? 1                            \
       : (((VAL) < VLLM_LAUNCH_BLOCKS_CAP) ? (VAL) : VLLM_LAUNCH_BLOCKS_CAP))
#define VLLM_BLOCKS_PER_SM(BLOCK_THREADS) \
  VLLM_CLAMP_BLOCKS_PER_SM(VLLM_BLOCKS_DIV(BLOCK_THREADS))

// runtime-time helper to compute blocks/SM
static inline int vllm_runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = VLLM_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return VLLM_CLAMP_BLOCKS_PER_SM(blocks);
}
