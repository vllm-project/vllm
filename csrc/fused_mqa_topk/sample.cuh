// Gate sample of the persistent kernel: each math warp histograms its row's
// sampled scores (fp16 bins >> 5, 2048 bins padded to 33 words per 32) and
// finish_sample_histogram picks the gate from them.
#pragma once
#include <cutlass/arch/barrier.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
constexpr unsigned SampleQ = 8;
struct alignas(16) SampleShared {
  unsigned hist[SampleQ][2112];
};
__device__ __forceinline__ void sample_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(256, 2).sync();
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ unsigned sample_bin(float x) {
  unsigned h = __half_as_ushort(__float2half_rd(x));
  return ((h & 0x8000u) ? ((~h) & 65535u) : (h ^ 0x8000u)) >> 5;
}
__device__ __forceinline__ float sample_floor(unsigned b) {
  unsigned o = b << 5, h = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(h));
}
