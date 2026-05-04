// Per-N kernel instantiation shard for wvSplitK_w8a8 (N=5).
// One TU per N value keeps the template instantiation footprint per file
// small and lets make parallelize across the shards. See dispatch.cuh.

#include "dispatch.cuh"
#include "launch.h"

template <typename scalar_t>
void launch_w8a8_n5(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                    int By, const int8_t* B, const void* A_raw,
                    const scalar_t* w_scale, const float* a_scale,
                    const scalar_t* BIAS, scalar_t* C, int CuCount,
                    int quant_mode, int thrds, int ytile, int unrl,
                    int achunk) {
  dispatch_w8a8<scalar_t, 5>(grid, stream, K, M, Bx, By, B, A_raw, w_scale,
                             a_scale, BIAS, C, CuCount, quant_mode, thrds,
                             ytile, unrl, achunk);
}

template void launch_w8a8_n5<half>(dim3, cudaStream_t, int, int, int, int,
                                   const int8_t*, const void*, const half*,
                                   const float*, const half*, half*, int, int,
                                   int, int, int, int);
template void launch_w8a8_n5<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, const int8_t*, const void*,
    const __hip_bfloat16*, const float*, const __hip_bfloat16*, __hip_bfloat16*,
    int, int, int, int, int, int);

#ifdef VLLM_SKINNY_GEMM_SWEEP

template <typename scalar_t>
void launch_w8a8_n5_sweep(dim3 grid, cudaStream_t stream, int K, int M, int Bx,
                          int By, const int8_t* B, const void* A_raw,
                          const scalar_t* w_scale, const float* a_scale,
                          const scalar_t* BIAS, scalar_t* C, int CuCount,
                          int quant_mode, int thrds, int ytile, int wvprgrp,
                          int achunk, int unrl) {
  dispatch_w8a8_sweep<scalar_t, 5>(
      grid, stream, K, M, Bx, By, B, A_raw, w_scale, a_scale, BIAS, C, CuCount,
      quant_mode, thrds, ytile, wvprgrp, achunk, unrl);
}

template void launch_w8a8_n5_sweep<half>(dim3, cudaStream_t, int, int, int, int,
                                         const int8_t*, const void*,
                                         const half*, const float*, const half*,
                                         half*, int, int, int, int, int, int,
                                         int);
template void launch_w8a8_n5_sweep<__hip_bfloat16>(
    dim3, cudaStream_t, int, int, int, int, const int8_t*, const void*,
    const __hip_bfloat16*, const float*, const __hip_bfloat16*, __hip_bfloat16*,
    int, int, int, int, int, int, int);

#endif  // VLLM_SKINNY_GEMM_SWEEP
