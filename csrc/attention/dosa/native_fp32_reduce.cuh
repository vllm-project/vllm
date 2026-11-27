#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace native_fp32 {
// Input dots already include native block-scaled MMA Q/K scales. Weights
// remain original FP32 (including any Q scale folded in by the caller).
// Four increasing-head chains, shared by dense/gathered consumers.
struct Reduction {
  float2 even = make_float2(0.0f, 0.0f);
  float2 odd = make_float2(0.0f, 0.0f);

  __device__ __forceinline__ void update4(float4 x, float4 w) {
    const auto a =
        __fadd2_rn(make_float2(x.x, x.y), make_float2(fabsf(x.x), fabsf(x.y)));
    const auto b =
        __fadd2_rn(make_float2(x.z, x.w), make_float2(fabsf(x.z), fabsf(x.w)));
    even = __ffma2_rn(a, make_float2(w.x, w.y), even);
    odd = __ffma2_rn(b, make_float2(w.z, w.w), odd);
  }
  __device__ __forceinline__ float finish() const {
    const auto pair = __fadd2_rn(even, odd);
    return __fmul_rn(__fadd_rn(pair.x, pair.y), 0.5f);
  }
};
// H is a multiple of four. Chunks must visit heads in increasing order;
// retain Reduction across 16/8/4 chunks and call finish exactly once.
__device__ __forceinline__ void update(Reduction& state, const float* dots,
                                       const float* weights, int heads) {
  for (int h = 0; h < heads; h += 4)
    state.update4(make_float4(dots[h], dots[h + 1], dots[h + 2], dots[h + 3]),
                  make_float4(weights[h], weights[h + 1], weights[h + 2],
                              weights[h + 3]));
}
}  // namespace native_fp32
