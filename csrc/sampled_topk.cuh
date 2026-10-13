// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Sampled top-k for long FP32 sparse-indexer rows, inspired by DeepSelect:
// https://github.com/deepseek-ai/DeepSelect
// A sample estimates a cutoff, then survivors are compacted in shared memory
// for exact FP32 selection. Too few or too many candidates trigger an exact
// full-row fallback.

#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

#include "persistent_topk.cuh"

namespace vllm::sampled_topk {

constexpr int kThreads = 1024;
// Conservative crossover bounds from the B300 batch/length sweep.
template <int K>
constexpr int kMinSampledLength = K == 512 ? 98304 : 65536;
// Target half the buffer to leave room for sampling error.
constexpr int kSample = 4096;
constexpr int kCapacity = 8192;

struct Storage {
  uint2 candidates[kCapacity];
  int histogram[2048];
  typename cub::BlockScan<int, kThreads>::TempStorage scan;
  int count;
  int remaining;
  int bin;
  uint32_t prefix;
};

__device__ __forceinline__ uint32_t ordered(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint32_t coarse(uint32_t key) {
  const uint32_t bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
  return topk_histogram_4096::extract_coarse_bin_N<11>(__uint_as_float(bits));
}

// Select the bin containing the rank and update the rank within that bin.
template <int Bins>
__device__ void select_bin(Storage& s) {
  constexpr int kItems = (Bins + kThreads - 1) / kThreads;
  int sums[kItems];
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int bin = Bins - 1 - (threadIdx.x * kItems + j);
    sums[j] = bin >= 0 ? s.histogram[bin] : 0;
  }
  cub::BlockScan<int, kThreads>(s.scan).InclusiveSum(sums, sums);
  const int remaining = s.remaining;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int bin = Bins - 1 - (threadIdx.x * kItems + j);
    if (bin >= 0 && sums[j] >= remaining &&
        sums[j] - s.histogram[bin] < remaining) {
      s.bin = bin;
      s.remaining = remaining - (sums[j] - s.histogram[bin]);
    }
  }
  __syncthreads();
}

// Find a coarse bin, then refine its full FP32 keys one byte at a time.
// Read shared candidates normally, or the full row for exact fallback.
template <bool Buffered>
__device__ uint32_t threshold(Storage& s, const float* row, int count,
                              int rank) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    s.remaining = rank;
    s.prefix = 0;
  }
  for (int i = tid; i < 2048; i += kThreads) s.histogram[i] = 0;
  __syncthreads();
  for (int i = tid; i < count; i += kThreads) {
    const uint32_t key = Buffered ? s.candidates[i].y : ordered(row[i]);
    atomicAdd(&s.histogram[coarse(key)], 1);
  }
  __syncthreads();
  select_bin<2048>(s);
  const int coarse_bin = s.bin;
  uint32_t mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) s.histogram[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s.prefix;
    for (int i = tid; i < count; i += kThreads) {
      const uint32_t key = Buffered ? s.candidates[i].y : ordered(row[i]);
      if (coarse(key) == coarse_bin && (key & mask) == prefix) {
        atomicAdd(&s.histogram[(key >> shift) & 255], 1);
      }
    }
    __syncthreads();
    select_bin<256>(s);
    if (tid == 0) s.prefix = prefix | (static_cast<uint32_t>(s.bin) << shift);
    mask |= 255u << shift;
    __syncthreads();
  }
  return s.prefix;
}

__device__ __forceinline__ int reserve(bool hit, int* counter) {
  const uint32_t mask = __ballot_sync(0xffffffffu, hit);
  if (mask == 0) return 0;
  const int lane = threadIdx.x & 31;
  int base = 0;
  if (lane == 0) base = atomicAdd(counter, __popc(mask));
  base = __shfl_sync(0xffffffffu, base, 0);
  return base + __popc(mask & ((1u << lane) - 1));
}

template <bool Buffered>
__device__ void emit(Storage& s, const float* row, int32_t* dst, int count,
                     uint32_t cutoff) {
  const int tid = threadIdx.x;
  if (tid == 0) s.count = 0;
  __syncthreads();
  for (int base = 0; base < count; base += kThreads) {
    const int i = base + tid;
    const uint2 pair = i < count ? (Buffered ? s.candidates[i]
                                             : make_uint2(i, ordered(row[i])))
                                 : make_uint2(0, 0);
    bool keep = i < count && pair.y > cutoff;
    if (i < count && pair.y == cutoff) {
      keep = atomicSub(&s.remaining, 1) > 0;
    }
    const int offset = reserve(keep, &s.count);
    if (keep) dst[offset] = pair.x;
  }
}

template <int K>
__global__ void __launch_bounds__(kThreads)
    sampled_topk_kernel(const float* __restrict__ input,
                        const int32_t* __restrict__ lengths,
                        int32_t* __restrict__ output, int64_t stride,
                        int max_length) {
  extern __shared__ __align__(16) unsigned char smem[];
  auto& s = *reinterpret_cast<Storage*>(smem);
  const int tid = threadIdx.x;
  const int length = max(0, min(lengths[blockIdx.x], max_length));
  const float* row = input + blockIdx.x * stride;
  int32_t* dst = output + blockIdx.x * K;
  // Graph allocations can hold rows much shorter than the host length bound.
  if (length < kMinSampledLength<K>) {
    __shared__ filtered_topk::FilteredTopKStorage<K> fallback_storage;
    bool selected;
    if ((stride & 3) == 0 && (reinterpret_cast<uintptr_t>(row) & 15) == 0) {
      selected =
          filtered_topk::filtered_topk_row<float, int32_t, 4, K, false, true>(
              row, dst, length, K, fallback_storage);
    } else {
      selected =
          filtered_topk::filtered_topk_row<float, int32_t, 1, K, true, true>(
              row, dst, length, K, fallback_storage);
    }
    if (!selected) {
      const uint32_t exact_cutoff = threshold<false>(s, row, length, K);
      emit<false>(s, row, dst, length, exact_cutoff);
    }
    return;
  }
  if (tid == 0) s.remaining = max(1, (kCapacity / 2) * kSample / length);
  for (int i = tid; i < 2048; i += kThreads) s.histogram[i] = 0;
  __syncthreads();
  // Sample contiguous warps spread across the row to keep loads coalesced.
  for (int i = tid; i < kSample; i += kThreads) {
    const int index = (i / 32) * (length / (kSample / 32)) + i % 32;
    atomicAdd(&s.histogram[coarse(ordered(row[index]))], 1);
  }
  __syncthreads();
  select_bin<2048>(s);
  const uint16_t half_key = s.bin << 5;
  const uint16_t half_bits =
      (half_key & 0x8000u) ? (half_key ^ 0x8000u) : ~half_key;
  const uint32_t cutoff = ordered(__half2float(__ushort_as_half(half_bits)));
  if (tid == 0) s.count = 0;
  __syncthreads();
  constexpr int kItems = 16;
  // Compact survivors with one reservation per warp per tile. Keep counting
  // beyond capacity so overflow triggers fallback instead of using a subset.
  for (int base = 0; base < length; base += kItems * kThreads) {
    uint32_t keys[kItems];
    uint32_t hit_mask = 0;
    if ((reinterpret_cast<uintptr_t>(row) & 15) == 0 &&
        base + kItems * kThreads <= length) {
#pragma unroll
      for (int j = 0; j < kItems / 4; ++j) {
        const float4 values =
            reinterpret_cast<const float4*>(row + base)[tid + j * kThreads];
        keys[4 * j] = ordered(values.x);
        keys[4 * j + 1] = ordered(values.y);
        keys[4 * j + 2] = ordered(values.z);
        keys[4 * j + 3] = ordered(values.w);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = base + tid * 4 + (j / 4) * kThreads * 4 + j % 4;
        keys[j] = i < length ? ordered(row[i]) : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      const int i = base + tid * 4 + (j / 4) * kThreads * 4 + j % 4;
      if (i < length && keys[j] >= cutoff) hit_mask |= 1u << j;
    }
    const int lane = tid & 31;
    const int hits = __popc(hit_mask);
    int inclusive = hits;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous = __shfl_up_sync(0xffffffffu, inclusive, delta);
      if (lane >= delta) inclusive += previous;
    }
    const int total = __shfl_sync(0xffffffffu, inclusive, 31);
    int warp_base = 0;
    if (lane == 31 && total != 0) warp_base = atomicAdd(&s.count, total);
    warp_base = __shfl_sync(0xffffffffu, warp_base, 31);
    int offset = warp_base + inclusive - hits;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      if (hit_mask & (1u << j)) {
        if (offset < kCapacity) {
          s.candidates[offset] = make_uint2(
              base + tid * 4 + (j / 4) * kThreads * 4 + j % 4, keys[j]);
        }
        ++offset;
      }
    }
  }
  __syncthreads();
  const int count = s.count;
  if (count >= K && count <= kCapacity) {
    const uint32_t exact_cutoff = threshold<true>(s, row, count, K);
    emit<true>(s, row, dst, count, exact_cutoff);
  } else {
    const uint32_t exact_cutoff = threshold<false>(s, row, length, K);
    emit<false>(s, row, dst, length, exact_cutoff);
  }
}

}  // namespace vllm::sampled_topk
