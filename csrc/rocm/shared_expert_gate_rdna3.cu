// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Gated shared expert of Qwen2-MoE style blocks in one launch (gfx11):
//
//   out[m, :] *= fp16(sigmoid(fp16(x[m, :] . w)))
//
// Unfused, the one-row gate GEMM, the sigmoid and the broadcast multiply are
// three small launches per MoE layer of a decode step. One block per token:
// the block reduces the gate logit, then scales its row of out in place. The
// logit and the gate are rounded to fp16 like the unfused path.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace shared_expert_gate_rdna3 {

constexpr int kThreads = 256;
constexpr int kVec = 8;

struct alignas(16) half8 {
  half v[kVec];
};

__global__ void __launch_bounds__(kThreads)
    gate_kernel(half* __restrict__ out, const half* __restrict__ x,
                const half* __restrict__ w, int k, int n, int64_t sx,
                int64_t so) {
  __shared__ float part[kThreads / 32];
  __shared__ half gate;
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const half* xr = x + m * sx;

  float acc = 0.f;
  for (int c = t; c < k / kVec; c += kThreads) {
    const half8 a = *reinterpret_cast<const half8*>(xr + c * kVec);
    const half8 b = *reinterpret_cast<const half8*>(w + c * kVec);
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      acc += __half2float(a.v[i]) * __half2float(b.v[i]);
  }
#pragma unroll
  for (int s = 16; s >= 1; s >>= 1) acc += __shfl_xor(acc, s, 32);
  if ((t & 31) == 0) part[t >> 5] = acc;
  __syncthreads();
  if (t == 0) {
    float g = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) g += part[i];
    const float gl = __half2float(__float2half_rn(g));
    gate = __float2half_rn(1.f / (1.f + expf(-gl)));
  }
  __syncthreads();

  const float s = __half2float(gate);
  half* orow = out + m * so;
  for (int c = t; c < n / kVec; c += kThreads) {
    half8 o = *reinterpret_cast<half8*>(orow + c * kVec);
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      o.v[i] = __float2half_rn(__half2float(o.v[i]) * s);
    *reinterpret_cast<half8*>(orow + c * kVec) = o;
  }
}

}  // namespace shared_expert_gate_rdna3
}  // namespace vllm

void shared_expert_gate_rdna3(torch::Tensor out, torch::Tensor x,
                              torch::Tensor weight) {
  using namespace vllm::shared_expert_gate_rdna3;
  TORCH_CHECK(out.dim() == 2 && x.dim() == 2 && out.size(0) == x.size(0),
              "shared_expert_gate_rdna3: out [M, N], x [M, K]");
  TORCH_CHECK(out.scalar_type() == torch::kHalf &&
                  x.scalar_type() == torch::kHalf &&
                  weight.scalar_type() == torch::kHalf,
              "shared_expert_gate_rdna3: fp16 only");
  const int k = x.size(1), n = out.size(1);
  TORCH_CHECK(weight.numel() == k && weight.stride(-1) == 1,
              "shared_expert_gate_rdna3: weight must be one row of K");
  TORCH_CHECK(k % kVec == 0 && n % kVec == 0 && x.stride(1) == 1 &&
                  out.stride(1) == 1 && x.stride(0) % kVec == 0 &&
                  out.stride(0) % kVec == 0,
              "shared_expert_gate_rdna3: rows of 16-byte aligned fp16");
  for (const auto* p : {&out, &x, &weight})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(p->data_ptr()) % 16 == 0,
                "shared_expert_gate_rdna3: 16-byte aligned pointers");
  const int m = x.size(0);
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  gate_kernel<<<m, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (half*)out.data_ptr(), (const half*)x.data_ptr(),
      (const half*)weight.data_ptr(), k, n, x.stride(0), out.stride(0));
}
