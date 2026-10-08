#pragma once
#include <cutlass/arch/barrier.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
constexpr unsigned SampleQ = 8, SampleN = 8192;
// target: candidate count the gate aims for (0 = legacy 4096/6144 by visible
// length).
struct SampleBuffers {
  float* ring;
  float* retained;
  int* indices;
  int* output;
  unsigned target;
};
struct alignas(16) SampleShared {
  float scores[SampleQ][256];
  unsigned hist[SampleQ][2112], counts[SampleQ], rank, cut;
  __device__ void init(unsigned tid) {
    for (unsigned j = tid; j < SampleQ * 2112; j += 384)
      reinterpret_cast<unsigned*>(hist)[j] = 0;
    if (tid < SampleQ) counts[tid] = 0;
  }
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
__device__ void finish_sample(SampleShared& s, SampleBuffers b, unsigned local,
                              unsigned row, unsigned visible, unsigned tid) {
  const unsigned target =
      b.target ? b.target : (visible <= 65536 ? 4096u : 6144u);
  if (visible <= target) {
    if (tid == 0) b.ring[row] = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  unsigned lane = tid % 32;
  auto* hist = s.hist[local];
  unsigned count = s.counts[local];
  if (count == 0) {
    if (tid == 0) b.ring[row] = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  unsigned sample_rank =
      max(1u, min(count, (count * target + visible - 1) / visible));
  __syncwarp();
  if (tid < 32) {
    unsigned sum = 0;
    for (unsigned j = 0; j < 64; ++j) {
      unsigned bin = 2047 - lane * 64 - j;
      sum += hist[bin + bin / 32];
    }
    unsigned inclusive = sum;
    for (unsigned d = 1; d < 32; d *= 2) {
      unsigned v = __shfl_up_sync(0xffffffffu, inclusive, d);
      if (lane >= d) inclusive += v;
    }
    unsigned before = inclusive - sum, rank = sample_rank,
             mask =
                 __ballot_sync(0xffffffffu, before < rank && inclusive >= rank),
             winner = __ffs(mask) - 1, chosen = 0, above = before;
    if (lane == winner) {
      for (unsigned j = 0; j < 64; ++j) {
        unsigned bin = 2047 - lane * 64 - j;
        if (above + hist[bin + bin / 32] >= rank) {
          chosen = bin;
          break;
        }
        above += hist[bin + bin / 32];
      }
    }
    chosen = __shfl_sync(0xffffffffu, chosen, winner);
    if (lane == 0) b.ring[row] = sample_floor(chosen);
  }
  __syncwarp();
}

__device__ void accumulate_sample(SampleShared& s, unsigned tid) {
  unsigned local = tid / 32, lane = tid % 32, count = 0;
#pragma unroll
  for (unsigned j = lane; j < 256; j += 32) {
    float v = s.scores[local][j];
    bool valid = isfinite(v);
    count += valid;
    unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      unsigned bin = sample_bin(v), peers = __match_any_sync(mask, bin);
      if (lane == unsigned(__ffs(peers) - 1))
        atomicAdd(s.hist[local] + bin + bin / 32, __popc(peers));
    }
  }
  count = __reduce_add_sync(0xffffffffu, count);
  if (lane == 0) s.counts[local] += count;
}
