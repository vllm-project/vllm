#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>

// maximum blocks per SM cap
#ifndef VLLM_LAUNCH_BLOCKS_CAP
  #define VLLM_LAUNCH_BLOCKS_CAP 4
#endif

// Compile-time estimate of max threads per SM for launch bounds.
// Families: 1024, 1536, 2048 threads/SM.
#ifndef VLLM_MAX_THREADS_PER_SM
  #ifdef __CUDA_ARCH__

    /* 1024 thr/SM: Turing (sm_75) */
    #if (__CUDA_ARCH__ == 750)
      #define VLLM_MAX_THREADS_PER_SM 1024

    /* 1536 thr/SM: Ampere GA10x (sm_86/87), Ada (sm_89),
        GB20x consumer (sm_120/121), Thor (sm_101 or sm_110) */
    #elif (__CUDA_ARCH__ == 860) || (__CUDA_ARCH__ == 870) || \
        (__CUDA_ARCH__ == 890) || (__CUDA_ARCH__ == 1010) ||  \
        (__CUDA_ARCH__ == 1100) || (__CUDA_ARCH__ == 1200) || \
        (__CUDA_ARCH__ == 1210)
      #define VLLM_MAX_THREADS_PER_SM 1536

    /* 2048 thr/SM: Volta (sm_70/72), Ampere GA100 (sm_80),
        Hopper (sm_90), Blackwell (sm_100/103) */
    #elif (__CUDA_ARCH__ == 700) || (__CUDA_ARCH__ == 720) || \
        (__CUDA_ARCH__ == 800) || (__CUDA_ARCH__ == 900) ||   \
        (__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1030)
      #define VLLM_MAX_THREADS_PER_SM 2048

    /* Fallback: use 2048 for unknown future CCs */
    #else
      #define VLLM_MAX_THREADS_PER_SM 2048
    #endif

  #else
  /* Host pass (no __CUDA_ARCH__): neutral default */
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
