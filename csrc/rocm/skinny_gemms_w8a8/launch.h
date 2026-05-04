#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Per-N launchers for the wvSplitK_w8a8 kernel. Each instantiate_n{K}.cu TU
// defines launch_w8a8_n{K}() for a single N value, which keeps the template
// instantiation footprint per TU small and lets make parallelize across the
// shards. The dispatcher in the parent .cu picks (yt, ur) via the heuristic
// and then calls into the appropriate shard.

#define DECLARE_LAUNCH_W8A8(N)                                              \
  template <typename scalar_t>                                              \
  void launch_w8a8_n##N(                                                    \
      dim3 grid, cudaStream_t stream, int K, int M, int Bx, int By,         \
      const int8_t* B, const void* A_raw, const scalar_t* w_scale,          \
      const float* a_scale, const scalar_t* BIAS, scalar_t* C, int CuCount, \
      int quant_mode, int thrds, int ytile, int unrl, int achunk);

DECLARE_LAUNCH_W8A8(1)
DECLARE_LAUNCH_W8A8(2)
DECLARE_LAUNCH_W8A8(3)
DECLARE_LAUNCH_W8A8(4)
DECLARE_LAUNCH_W8A8(5)

#undef DECLARE_LAUNCH_W8A8

#ifdef VLLM_SKINNY_GEMM_SWEEP

  // Sweep variants take the full (yt, ur, ac, wv) tuple. Gated behind the
  // sweep build flag so the production build never pays for the wider
  // template space.
  #define DECLARE_LAUNCH_W8A8_SWEEP(N)                                        \
    template <typename scalar_t>                                              \
    void launch_w8a8_n##N##_sweep(                                            \
        dim3 grid, cudaStream_t stream, int K, int M, int Bx, int By,         \
        const int8_t* B, const void* A_raw, const scalar_t* w_scale,          \
        const float* a_scale, const scalar_t* BIAS, scalar_t* C, int CuCount, \
        int quant_mode, int thrds, int ytile, int wvprgrp, int achunk,        \
        int unrl);

DECLARE_LAUNCH_W8A8_SWEEP(1)
DECLARE_LAUNCH_W8A8_SWEEP(2)
DECLARE_LAUNCH_W8A8_SWEEP(3)
DECLARE_LAUNCH_W8A8_SWEEP(4)
DECLARE_LAUNCH_W8A8_SWEEP(5)

  #undef DECLARE_LAUNCH_W8A8_SWEEP

#endif  // VLLM_SKINNY_GEMM_SWEEP
