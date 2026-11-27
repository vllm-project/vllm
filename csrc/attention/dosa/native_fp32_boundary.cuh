#pragma once
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace native_fp32_boundary {
constexpr int Top = 2048, Threads = 128, Items = 32, Capacity = Threads * Items;
using Sort = cub::BlockRadixSort<uint64_t, Threads, Items, int>;
struct Args {
  const float* scores;  // E scores [logical row, E ordinal], full FP32
  const int* ids;       // E physical key IDs, same ordinal order
  const int *partition, *permutation;
  const int64_t* order;  // logical row -> global output row, unique
  int *output, *error;
  int rows, cap, nk, output_rows;
  int64_t score_stride, id_stride, output_stride;
};
__device__ __forceinline__ unsigned score_key(float score) {
  unsigned bits = __float_as_uint(score);
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  return bits & 0x80000000u ? bits : bits ^ 0x7fffffffu;
}

// Four independent warps per CTA; no shared sort storage for E<=32.
// Lane ordinal breaks completely equal composite keys, retaining duplicates.
static __global__ __launch_bounds__(Threads) void finish_small(Args a) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (Threads / 32) + (threadIdx.x >> 5);
  if (row >= a.rows) return;
  const unsigned partition = unsigned(a.partition[row]);
  const int certain = int(partition >> 13), uncertain = int(partition & 8191);
  if (!partition || uncertain > 32) return;
  const int64_t global_row = a.order[row];
  bool bad = a.error[row] != 0 || certain > Top || uncertain > a.cap ||
             uncertain < Top - certain || global_row < 0 ||
             global_row >= a.output_rows;
  if (__any_sync(0xffffffffu, bad)) {
    if (!lane) atomicOr(a.error + row, 1);
    return;
  }
  if (certain == Top) return;
  unsigned hi = 0xffffffffu, lo = 0xffffffffu;
  int id = -1;
  if (lane < uncertain) {
    const float score = a.scores[int64_t(row) * a.score_stride + lane];
    id = a.ids[int64_t(row) * a.id_stride + lane];
    const bool valid = id >= 0 && id < a.nk;
    const int original = valid ? a.permutation[id] : -1;
    bad = !valid || original < 0 || original >= a.nk ||
          (__float_as_uint(score) & 0x7f800000u) == 0x7f800000u;
    hi = score_key(score);
    lo = unsigned(original);
  }
  // Completes every input read before any potentially aliased output store.
  if (__any_sync(0xffffffffu, bad)) {
    if (!lane) atomicOr(a.error + row, 1);
    return;
  }
  int rank = 0;
  for (int other = 0; other < uncertain; ++other) {
    const unsigned h = __shfl_sync(0xffffffffu, hi, other);
    const unsigned l = __shfl_sync(0xffffffffu, lo, other);
    rank += (h < hi || (h == hi && (l < lo || (l == lo && other < lane))));
  }
  __syncwarp();
  if (lane < uncertain && rank < Top - certain)
    a.output[global_row * a.output_stride + certain + rank] = id;
}

// One CTA per logical row. On success only output[order[row], certain:Top]
// changes. E scores/IDs/permutation are read completely before any output
// write, supporting same-row input/output aliases (no cross-row aliasing).
static __global__ __launch_bounds__(Threads) void finish(Args a) {
  const int row = blockIdx.x, t = threadIdx.x;
  const int64_t global_row = a.order[row];
  const unsigned partition = unsigned(a.partition[row]);
  const int certain = int(partition >> 13), uncertain = int(partition & 8191);
  if (partition == 0 || uncertain <= 32) return;  // handled by finish_small
  __shared__ unsigned bad;
  __shared__ Sort::TempStorage storage;
  if (t == 0)
    bad = a.error[row] != 0 || certain > Top || uncertain > a.cap ||
          uncertain > Capacity || uncertain < Top - certain || global_row < 0 ||
          global_row >= a.output_rows;
  __syncthreads();
  if (bad) {
    if (t == 0) atomicOr(a.error + row, 1);
    return;
  }
  if (certain == Top) return;
  uint64_t keys[Items];
  int values[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int at = t * Items + i;
    keys[i] = UINT64_MAX;
    values[i] = -1;
    if (at < uncertain) {
      const float score = a.scores[int64_t(row) * a.score_stride + at];
      const int id = a.ids[int64_t(row) * a.id_stride + at];
      const bool valid = id >= 0 && id < a.nk;
      const int original = valid ? a.permutation[id] : -1;
      if (!valid || original < 0 || original >= a.nk ||
          (__float_as_uint(score) & 0x7f800000u) == 0x7f800000u)
        atomicOr(&bad, 1u);
      keys[i] = (uint64_t(score_key(score)) << 32) | unsigned(original);
      values[i] = id;  // Preserve C298 physical-ID output; _map runs later.
    }
  }
  __syncthreads();  // All aliased input reads drain before any output writes.
  if (bad) {
    if (t == 0) atomicOr(a.error + row, 1);
    return;
  }
  Sort(storage).SortBlockedToStriped(keys, values);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int at = i * Threads + t;
    if (at < Top - certain)
      a.output[global_row * a.output_stride + certain + at] = values[i];
  }
}
}  // namespace native_fp32_boundary
