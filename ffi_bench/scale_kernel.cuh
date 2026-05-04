// Trivial CUDA kernel shared by all three binding variants. Computing
// `out[i] = in[i] * factor` is essentially free; with a 1-element tensor the
// per-call latency is dominated by host-side dispatch + kernel launch overhead.
#pragma once
#include <cuda_runtime.h>

template <typename T>
__global__ void scale_kernel(T* __restrict__ out, const T* __restrict__ in,
                             T factor, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * factor;
}

inline void launch_scale_f32(float* out, const float* in, float factor,
                             int64_t n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  scale_kernel<float><<<blocks, threads, 0, stream>>>(out, in, factor, n);
}
