// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace vllm {
__global__ void silu_mul_fp8_sm90_kernel(const __nv_bfloat16* x,
                                         __nv_fp8_e4m3* q, float* scales, int m,
                                         int k, int stride, bool native) {
  int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  int ng = k / 128;
  if (group >= m * ng) return;
  int row = group / ng, col = group % ng;
  float vals[4], mx = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int off = row * (2 * k) + col * 128 + lane + i * 32;
    float g = __bfloat162float(x[off]);
    float u = __bfloat162float(x[off + k]);
    if (!native) u = __fadd_rn(u, 0.f);
    float silu = g / (1.f + expf(-g));
    if (!native) silu = __bfloat162float(__float2bfloat16_rn(silu));
    vals[i] = __bfloat162float(__float2bfloat16_rn(silu * u));
    mx = fmaxf(mx, fabsf(vals[i]));
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, delta));
  float inv = mx != 0.f ? 448.f / mx : 1.f;
  if (lane == 0) scales[col * stride + row] = 1.f / inv;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    q[row * k + col * 128 + lane + i * 32] = __nv_fp8_e4m3(vals[i] * inv);
}
}  // namespace vllm
