#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>

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

// Launch config for elementwise kernels that run one block per token and stride
// it over the hidden dim: such a grid is only num_tokens blocks wide, so at
// decode most of the device sits idle. Splitting a token's work over `chunks`
// blocks in gridDim.y scales the launch with the device instead. Once
// num_tokens alone fills it this returns chunks == 1, the previous config.
// Each element is still handled by one thread, so results are unchanged.
struct VllmElementwiseLaunch {
  int threads;
  int chunks;
};

static inline VllmElementwiseLaunch vllm_elementwise_launch_config(
    int64_t num_tokens, int work_per_token, int sm_count,
    int max_threads = 1024) {
  if (num_tokens >= sm_count) {
    return {std::min(work_per_token, max_threads), 1};
  }
  // Smaller blocks, so a token's work spreads over more of them.
  const int threads =
      std::clamp(std::min(work_per_token, max_threads), 64, 256);
  // Oversubscribe the device, capped so that no block comes up empty.
  const int max_chunks = std::max(1, (work_per_token + threads - 1) / threads);
  const int wanted = static_cast<int>(
      (static_cast<int64_t>(sm_count) * VLLM_LAUNCH_BLOCKS_CAP + num_tokens -
       1) /
      num_tokens);
  return {threads, std::min(max_chunks, std::max(1, wanted))};
}
