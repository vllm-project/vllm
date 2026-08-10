// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// LiteTopK DSA V3 hybrid host wrapper: DeepGEMM-2.5 scoring loop + V1 KV-split;
// scoring kernel (sm100_dsa_litetopk.cuh) with the sparse candidate epilogue,
// plus the architecture-agnostic radix-select post-kernels (copied verbatim
// from dsa_litetopk.cu). Build against the DeepGEMM 2.5 include tree + its
// bundled CUTLASS (NOT the legacy deep_gemm include tree V1 uses).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>

#include "sm100_dsa_litetopk.cuh"
#include "dense_topk_litetopk.cuh"
// Production integration of the independently qualified h2048 safe selector.
// Fast-path source SHA256:
//   305b2af3c3d2495271245df7535354b051add8ca653aa5234b67d3560ca5f7bf
// Overflow fallback source SHA256:
//   d3a3ea206f0bee5419863118e30bc48d58cb8bd6030d75398d86732656df4430
//
// The fast CTA emits physical IDs from the six-byte high24 candidate ABI.
// Status bit 5 is the sole recoverable condition.  The always-launched exact
// fallback clears it after high12/low12 radix selection; the caller performs
// one uniform late map only after both kernels complete.

namespace h2048_safe_topk {

constexpr int kBins = 256;
constexpr int kMinCap = 65536;
constexpr int kMaxCap = 1 << 20;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadCertificate = 1u << 4,
  // Matches qrita_overflow_fallback_safety.cu's recoverable contract.
  kBoundaryOverflow = 1u << 5,
};

constexpr uint32_t kNonOverflowStatusMask =
    kBadCount | kNonFinite | kBadPhysical | kBadCertificate;
static_assert(kBoundaryOverflow == 32u);
static_assert((kNonOverflowStatusMask & kBoundaryOverflow) == 0u);

__device__ __forceinline__ uint32_t candidate_score_code(
    uint16_t value, int32_t packed_index) {
  return ((static_cast<uint32_t>(packed_index) >> 20) << 16) |
         static_cast<uint32_t>(value);
}

__device__ __forceinline__ float decode_score_code(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits = (ordered & 0x80000000u)
      ? (ordered ^ 0x80000000u)
      : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int coarse_bucket(uint32_t code) {
  const float value = decode_score_code(code);
  return value < 0.0f ? 0 : (value >= 256.0f ? 255 : static_cast<int>(value));
}

template <int Scale>
__device__ __forceinline__ int coarse_bucket_scaled(uint32_t code) {
  static_assert(Scale == 8);
  constexpr int bins = kBins * Scale;
  const float value = decode_score_code(code);
  return value < 0.0f
      ? 0
      : (value >= static_cast<float>(kBins)
             ? bins - 1
             : static_cast<int>(value * static_cast<float>(Scale)));
}

__device__ __forceinline__ uint32_t fp24_code(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t ordered =
      (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return ordered >> 8;
}

template <int Bins>
__device__ __forceinline__ void find_radix_digit(
    const uint32_t* __restrict__ hist,
    uint32_t* __restrict__ desired,
    uint32_t* __restrict__ rank,
    uint32_t* __restrict__ selected_count,
    int shift) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 32) return;
  constexpr unsigned kFull = 0xffffffffu;
  static_assert(
      Bins == 256 || Bins == 512 || Bins == 1024 || Bins == 2048 ||
      Bins == 4096);
  constexpr int kGroupBins = Bins / 32;
  constexpr int kItemsPerLane = (kGroupBins + 31) / 32;
  const int lane = tid;
  const int first = lane * kGroupBins;
  uint32_t group_count = 0u;
#pragma unroll
  for (int i = 0; i < kGroupBins; ++i) group_count += hist[first + i];
  uint32_t inclusive = group_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  const uint32_t target = *rank;
  const unsigned group_mask = __ballot_sync(kFull, inclusive >= target);
  if (target == 0u || group_mask == 0u) return;
  const int winning_group = __ffs(group_mask) - 1;
  const uint32_t group_before = __shfl_sync(
      kFull, inclusive - group_count, winning_group);
  // For 2048/4096 bins the winning coarse group contains 64/128 bins.
  // Give each lane a contiguous 2/4-bin segment, scan segment totals, then
  // let the winning lane locate the exact bin locally.  The same code also
  // handles the 8-bin exact-radix histogram used below.
  const int segment_offset = lane * kItemsPerLane;
  const bool segment_valid = segment_offset < kGroupBins;
  const int segment_first =
      winning_group * kGroupBins + segment_offset;
  uint32_t segment_count = 0u;
#pragma unroll
  for (int i = 0; i < kItemsPerLane; ++i) {
    if (segment_offset + i < kGroupBins) {
      segment_count += hist[segment_first + i];
    }
  }
  uint32_t segment_inclusive = segment_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, segment_inclusive, offset);
    if (lane >= offset) segment_inclusive += other;
  }
  const unsigned segment_mask = __ballot_sync(
      kFull,
      segment_valid && group_before + segment_inclusive >= target);
  if (segment_mask == 0u) return;
  const int winning_lane = __ffs(segment_mask) - 1;
  const uint32_t segment_before = __shfl_sync(
      kFull, segment_inclusive - segment_count, winning_lane);

  uint32_t local_digit = 0u;
  uint32_t local_before = 0u;
  uint32_t local_count = 0u;
  if (lane == winning_lane) {
    const uint32_t local_target = target - group_before - segment_before;
    uint32_t running = 0u;
    bool found = false;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      const bool valid = segment_offset + i < kGroupBins;
      const uint32_t count = valid ? hist[segment_first + i] : 0u;
      if (!found && valid && running + count >= local_target) {
        local_digit = static_cast<uint32_t>(segment_offset + i);
        local_before = running;
        local_count = count;
        found = true;
      }
      running += count;
    }
  }
  local_digit = __shfl_sync(kFull, local_digit, winning_lane);
  local_before = __shfl_sync(kFull, local_before, winning_lane);
  local_count = __shfl_sync(kFull, local_count, winning_lane);
  if (lane == 0) {
    const uint32_t digit =
        static_cast<uint32_t>(winning_group * kGroupBins) + local_digit;
    *desired |= digit << static_cast<uint32_t>(shift);
    *rank = target - group_before - segment_before - local_before;
    *selected_count = local_count;
  }
}

// Mode 0: decoded-FP32 predicates + scalar pass-one histogram (old scratch).
// Mode 1: integer-code predicates + scalar pass-one histogram.
// Mode 2: integer-code predicates + match_any pass-one histogram.
template <int Threads, int BoundaryCapacity, int Mode, int HistScale>
__global__ __launch_bounds__(Threads) void coarse_tiering_topk_kernel(
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ output,
    int32_t* __restrict__ status,
    int32_t* __restrict__ diagnostics,
    int rows,
    int cap,
    int topk,
    int sequence_length) {
  static_assert(Threads == 128 || Threads == 256 || Threads == 512);
  static_assert(BoundaryCapacity == 512 || BoundaryCapacity == 1024);
  static_assert(Mode >= 0 && Mode <= 2);
  static_assert(HistScale == 8);
  constexpr int kCoarseBins = kBins * HistScale;
  constexpr unsigned kFull = 0xffffffffu;
  const int logical_block = static_cast<int>(blockIdx.x);
  // Real count argmax is near the end; longest rows launch first.
  const int row = rows - 1 - logical_block;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  if (row < 0 || row >= rows) return;

  __shared__ uint32_t hist[kCoarseBins];
  __shared__ uint32_t boundary_code[BoundaryCapacity];
  // Candidate rows may use the production 196608-record capacity.  A uint16
  // slot silently wrapped beyond the old 65536-record qualification shape.
  __shared__ uint32_t boundary_slot[BoundaryCapacity];
  __shared__ uint32_t s_status;
  __shared__ uint32_t s_desired;
  __shared__ uint32_t s_rank;
  __shared__ uint32_t s_selected_count;
  __shared__ int s_count;
  __shared__ int s_coarse_bucket;
  __shared__ int s_coarse_lt;
  __shared__ int s_coarse_need;
  __shared__ int s_boundary_count;
  __shared__ int s_strict_cursor;
  __shared__ int s_boundary_cursor;
  __shared__ int s_boundary_lt_cursor;

  if (tid == 0) {
    const int raw_count = counts[row];
    s_status = 0u;
    if (raw_count < topk || raw_count > cap || raw_count < 0) {
      s_status |= kBadCount;
    }
    s_count = max(0, min(raw_count, cap));
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(topk);
    s_selected_count = 0u;
    s_coarse_bucket = -1;
    s_coarse_lt = 0;
    s_coarse_need = 0;
    s_boundary_count = 0;
    s_strict_cursor = 0;
    s_boundary_cursor = 0;
    s_boundary_lt_cursor = 0;
    status[row] = 0;
  }
  for (int i = tid; i < kCoarseBins; i += Threads) hist[i] = 0u;
  __syncthreads();

  const int count = s_count;
  const int64_t row_base = static_cast<int64_t>(row) * cap;
  const int64_t out_base = static_cast<int64_t>(row) * topk;
  if (s_status != 0u) {
    for (int i = tid; i < topk; i += Threads) output[out_base + i] = -1;
    if (tid == 0) status[row] = static_cast<int32_t>(s_status);
    return;
  }

  // Production-compatible coarse certificate. Mode 2 makes the loop
  // warp-uniform and combines equal-bucket updates into one shared atomic.
  if constexpr (Mode == 2) {
    const int warp = tid >> 5;
    for (int base = warp * 32; base < count; base += Threads) {
      const int slot = base + lane;
      const bool valid = slot < count;
      int bucket = 0;
      bool participate = false;
      if (valid) {
        const uint16_t value = values[row_base + slot];
        const int32_t packed_index = packed_indices[row_base + slot];
        const uint32_t physical =
            static_cast<uint32_t>(packed_index) & kPhysicalMask;
        const uint32_t code = candidate_score_code(value, packed_index);
        const float decoded = decode_score_code(code);
        if (physical >= static_cast<uint32_t>(sequence_length)) {
          atomicOr(&s_status, kBadPhysical);
        } else if (!isfinite(decoded)) {
          atomicOr(&s_status, kNonFinite);
        } else {
          bucket = coarse_bucket_scaled<HistScale>(code);
          participate = true;
        }
      }
      const unsigned active = __ballot_sync(kFull, participate);
      if (participate) {
        const unsigned peers = __match_any_sync(active, bucket);
        if (lane == __ffs(peers) - 1) {
          atomicAdd(hist + bucket, static_cast<uint32_t>(__popc(peers)));
        }
      }
    }
  } else {
    for (int slot = tid; slot < count; slot += Threads) {
      const uint16_t value = values[row_base + slot];
      const int32_t packed_index = packed_indices[row_base + slot];
      const uint32_t physical =
          static_cast<uint32_t>(packed_index) & kPhysicalMask;
      if (physical >= static_cast<uint32_t>(sequence_length)) {
        atomicOr(&s_status, kBadPhysical);
        continue;
      }
      const uint32_t code = candidate_score_code(value, packed_index);
      const float decoded = decode_score_code(code);
      if (!isfinite(decoded)) {
        atomicOr(&s_status, kNonFinite);
        continue;
      }
      atomicAdd(hist + coarse_bucket_scaled<HistScale>(code), 1u);
    }
  }
  __syncthreads();
  if (s_status == 0u) {
    find_radix_digit<kCoarseBins>(
        hist, &s_desired, &s_rank, &s_selected_count, 0);
  }
  __syncthreads();
  if (tid == 0 && s_status == 0u) {
    s_coarse_bucket = static_cast<int>(s_desired);
    s_coarse_need = static_cast<int>(s_rank);
    s_coarse_lt = topk - s_coarse_need;
    s_boundary_count = static_cast<int>(s_selected_count);
    if (s_coarse_bucket < 0 || s_coarse_bucket >= kCoarseBins ||
        s_coarse_lt < 0 || s_coarse_lt >= topk ||
        s_coarse_need <= 0 || s_coarse_need > s_boundary_count) {
      s_status |= kBadCertificate;
    }
    if (s_boundary_count > BoundaryCapacity) {
      s_status |= kBoundaryOverflow;
    }
  }
  __syncthreads();
  if (s_status != 0u) {
    for (int i = tid; i < topk; i += Threads) output[out_base + i] = -1;
    if (tid == 0) {
      status[row] = static_cast<int32_t>(s_status);
      int32_t* diag = diagnostics + static_cast<int64_t>(row) * 5;
      diag[0] = count;
      diag[1] = s_coarse_bucket;
      diag[2] = s_coarse_lt;
      diag[3] = s_coarse_need;
      diag[4] = s_boundary_count;
    }
    return;
  }

  // Emit physical IDs below the coarse boundary and gather the boundary.
  const int threshold_bucket = s_coarse_bucket;
  const float threshold_edge =
      static_cast<float>(threshold_bucket) / static_cast<float>(HistScale);
  const float next_threshold_edge =
      static_cast<float>(threshold_bucket + 1) /
      static_cast<float>(HistScale);
  const uint32_t threshold_code = fp24_code(threshold_edge);
  const uint32_t next_threshold_code = fp24_code(next_threshold_edge);
  const int warp = tid >> 5;
  for (int base = warp * 32; base < count; base += Threads) {
    const int slot = base + lane;
    uint32_t code = 0u;
    int32_t packed_index = 0;
    const bool valid = slot < count;
    int bucket = 0;
    if (valid) {
      packed_index = packed_indices[row_base + slot];
      code = candidate_score_code(values[row_base + slot], packed_index);
      bucket = coarse_bucket_scaled<HistScale>(code);
    }
    const bool is_strict = Mode >= 1
        ? (valid && threshold_bucket > 0 && code < threshold_code)
        : (valid && bucket < threshold_bucket);
    const bool is_boundary = Mode >= 1
        ? (valid &&
           (threshold_bucket == kCoarseBins - 1 ||
            code < next_threshold_code) &&
           (threshold_bucket == 0 || code >= threshold_code))
        : (valid && bucket == threshold_bucket);
    const unsigned strict_mask = __ballot_sync(kFull, is_strict);
    const unsigned boundary_mask = __ballot_sync(kFull, is_boundary);
    int strict_base = 0;
    int boundary_base = 0;
    if (lane == 0) {
      const int strict_n = __popc(strict_mask);
      const int boundary_n = __popc(boundary_mask);
      if (strict_n) strict_base = atomicAdd(&s_strict_cursor, strict_n);
      if (boundary_n) {
        boundary_base = atomicAdd(&s_boundary_cursor, boundary_n);
      }
    }
    strict_base = __shfl_sync(kFull, strict_base, 0);
    boundary_base = __shfl_sync(kFull, boundary_base, 0);
    const unsigned lane_before =
        lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
    if (is_strict) {
      const int pos = strict_base + __popc(strict_mask & lane_before);
      if (pos < topk) {
        output[out_base + pos] = static_cast<int32_t>(
            static_cast<uint32_t>(packed_index) & kPhysicalMask);
      }
    }
    if (is_boundary) {
      const int pos = boundary_base + __popc(boundary_mask & lane_before);
      if (pos < BoundaryCapacity) {
        boundary_code[pos] = code;
        boundary_slot[pos] = static_cast<uint32_t>(slot);
      }
    }
  }
  __syncthreads();
  if (tid == 0 &&
      (s_strict_cursor != s_coarse_lt ||
       s_boundary_cursor != s_boundary_count)) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  if (tid == 0) {
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(s_coarse_need);
    s_selected_count = 0u;
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    for (int i = tid; i < kBins; i += Threads) hist[i] = 0u;
    __syncthreads();
    const uint32_t desired = s_desired;
    for (int j = tid; j < s_boundary_count; j += Threads) {
      const uint32_t code = boundary_code[j];
      bool keep = true;
      if (pass == 1) keep = (code >> 16) == (desired >> 16);
      if (pass == 2) keep = (code >> 8) == (desired >> 8);
      if (keep) {
        atomicAdd(hist + ((code >> (16 - 8 * pass)) & 0xffu), 1u);
      }
    }
    __syncthreads();
    find_radix_digit<kBins>(
        hist, &s_desired, &s_rank, &s_selected_count, 16 - pass * 8);
    __syncthreads();
  }
  if (tid == 0 &&
      (s_rank == 0u || s_rank > s_selected_count ||
       s_rank > static_cast<uint32_t>(s_coarse_need))) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  const uint32_t exact_pivot = s_desired;
  const int exact_equal_take = static_cast<int>(s_rank);
  const int boundary_strict = s_coarse_need - exact_equal_take;
  for (int j = tid; j < s_boundary_count; j += Threads) {
    const uint32_t code = boundary_code[j];
    bool take = code < exact_pivot;
    int output_pos = -1;
    if (take) {
      output_pos = s_coarse_lt + atomicAdd(&s_boundary_lt_cursor, 1);
    } else if (code == exact_pivot) {
      const uint32_t slot = boundary_slot[j];
      int equal_rank = 0;
#pragma unroll 1
      for (int other = 0; other < s_boundary_count; ++other) {
        equal_rank += boundary_code[other] == exact_pivot &&
                      boundary_slot[other] < slot;
      }
      if (equal_rank < exact_equal_take) {
        take = true;
        output_pos = s_coarse_lt + boundary_strict + equal_rank;
      }
    }
    if (take && output_pos >= 0 && output_pos < topk) {
      const int slot = static_cast<int>(boundary_slot[j]);
      output[out_base + output_pos] = static_cast<int32_t>(
          static_cast<uint32_t>(packed_indices[row_base + slot]) &
          kPhysicalMask);
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (s_boundary_lt_cursor != boundary_strict) {
      s_status |= kBadCertificate;
    }
    status[row] = static_cast<int32_t>(s_status);
    int32_t* diag = diagnostics + static_cast<int64_t>(row) * 5;
    diag[0] = count;
    diag[1] = threshold_bucket;
    diag[2] = s_coarse_lt;
    diag[3] = s_boundary_count;
    diag[4] = static_cast<int32_t>(s_selected_count);
  }
}

namespace overflow {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRadixBits = 12;
constexpr int kRadixBins = 1 << kRadixBits;
constexpr int kBinsPerThread = kRadixBins / kThreads;
constexpr int kTopK = 2048;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadMapped = 1u << 3,
  kHistogramFailure = 1u << 4,
  kBoundaryOverflow = 1u << 5,
  kCompactFailure = 1u << 6,
};

__device__ __forceinline__ uint32_t candidate_score_code(
    uint16_t value, int32_t packed_index) {
  return ((static_cast<uint32_t>(packed_index) >> 20) << 16) |
      static_cast<uint32_t>(value);
}

__device__ __forceinline__ float decode_candidate_score(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits = (ordered & 0x80000000u)
      ? (ordered ^ 0x80000000u)
      : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int block_exclusive_sum(
    int value, int* warp_prefix) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) warp_prefix[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int original = lane < kWarps ? warp_prefix[lane] : 0;
    int warp_inclusive = original;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(
          0xffffffffu, warp_inclusive, offset);
      if (lane >= offset) warp_inclusive += other;
    }
    if (lane < kWarps) warp_prefix[lane] = warp_inclusive - original;
  }
  __syncthreads();
  return warp_prefix[warp] + inclusive - value;
}

__device__ __forceinline__ void select_histogram_bin(
    const int* histogram,
    int target,
    int* warp_prefix,
    int* selected_bin,
    int* selected_count_lt) {
  const int begin = static_cast<int>(threadIdx.x) * kBinsPerThread;
  int segment_sum = 0;
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    segment_sum += histogram[begin + i];
  }
  const int segment_lt = block_exclusive_sum(segment_sum, warp_prefix);
  if (target > segment_lt && target <= segment_lt + segment_sum) {
    int local_lt = 0;
#pragma unroll
    for (int i = 0; i < kBinsPerThread; ++i) {
      const int count = histogram[begin + i];
      if (target <= segment_lt + local_lt + count) {
        *selected_bin = begin + i;
        *selected_count_lt = segment_lt + local_lt;
        break;
      }
      local_lt += count;
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads) void overflow_exact_topk_kernel(
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ output,
    int32_t* __restrict__ status,
    int rows,
    int cap,
    int sequence_length,
    int topk_arg = kTopK) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;

  // CTA-uniform and before any barrier: the common no-overflow case only
  // reads one cached status word and retires the block.
  const uint32_t input_status = static_cast<uint32_t>(status[row]);
  if ((input_status & kBoundaryOverflow) == 0u) return;

  __shared__ int histogram[kRadixBins];
  __shared__ int warp_scratch[kWarps];
  __shared__ int selected_bin;
  __shared__ int selected_count_lt;
  __shared__ int first_count_lt;
  __shared__ uint32_t threshold_code;
  __shared__ int warp_lt[kWarps];
  __shared__ int warp_eq[kWarps];
  __shared__ int tile_lt;
  __shared__ int tile_eq;
  __shared__ int base_lt;
  __shared__ int base_eq;
  __shared__ uint32_t block_status;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lane_before =
      lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
  const int64_t candidate_row = static_cast<int64_t>(row) * cap;
  const int64_t output_row = static_cast<int64_t>(row) * topk_arg;
  const int count = counts[row];

  if (tid == 0) {
    // Overflow is recoverable. Any other fast-path error is retained and
    // prevents the slow path from declaring success.
    block_status = input_status & ~kBoundaryOverflow;
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  if (count < topk_arg || count > cap) {
    if (tid == 0) block_status |= kBadCount;
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  __syncthreads();

  // Pass 1: all high 12 bits. It also repeats fail-closed ABI validation so
  // the fallback is independently safe from a stale/corrupt fast flag.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (!isfinite(decode_candidate_score(code))) {
      atomicOr(&block_status, static_cast<uint32_t>(kNonFinite));
    }
    if (physical >= static_cast<uint32_t>(sequence_length)) {
      atomicOr(&block_status, static_cast<uint32_t>(kBadPhysical));
    }
    atomicAdd(histogram + (code >> kRadixBits), 1);
  }
  __syncthreads();

  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  select_histogram_bin(
      histogram, topk_arg, warp_scratch, &selected_bin, &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= topk_arg) {
      block_status |= kHistogramFailure;
    } else {
      first_count_lt = selected_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }
  const int high_bin = selected_bin;
  const int remaining_rank = topk_arg - first_count_lt;

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  if (tid == 0) {
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  // Pass 2: low 12 bits inside the winning high bucket.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (static_cast<int>(code >> kRadixBits) == high_bin) {
      atomicAdd(histogram + (code & (kRadixBins - 1)), 1);
    }
  }
  __syncthreads();
  select_histogram_bin(
      histogram, remaining_rank, warp_scratch,
      &selected_bin, &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= remaining_rank) {
      block_status |= kHistogramFailure;
    } else {
      threshold_code =
          (static_cast<uint32_t>(high_bin) << kRadixBits) |
          static_cast<uint32_t>(selected_bin);
      first_count_lt += selected_count_lt;
      base_lt = 0;
      base_eq = first_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  // Pass 3: stable exact compact and fused winner mapping.
  for (int tile = 0; tile < count; tile += kThreads) {
    const int col = tile + tid;
    uint32_t code = 0xffffffffu;
    int32_t packed = 0;
    if (col < count) {
      packed = packed_indices[candidate_row + col];
      code = candidate_score_code(values[candidate_row + col], packed);
    }
    const bool is_lt = col < count && code < threshold_code;
    const bool is_eq = col < count && code == threshold_code;
    const unsigned lt_mask = __ballot_sync(0xffffffffu, is_lt);
    const unsigned eq_mask = __ballot_sync(0xffffffffu, is_eq);
    const int lane_lt = __popc(lt_mask & lane_before);
    const int lane_eq = __popc(eq_mask & lane_before);
    if (lane == 0) {
      warp_lt[warp] = __popc(lt_mask);
      warp_eq[warp] = __popc(eq_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix_lt = 0;
      int prefix_eq = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        const int count_lt = warp_lt[w];
        const int count_eq = warp_eq[w];
        warp_lt[w] = prefix_lt;
        warp_eq[w] = prefix_eq;
        prefix_lt += count_lt;
        prefix_eq += count_eq;
      }
      tile_lt = base_lt;
      tile_eq = base_eq;
      base_lt += prefix_lt;
      base_eq += prefix_eq;
    }
    __syncthreads();

    int output_col = -1;
    if (is_lt) {
      output_col = tile_lt + warp_lt[warp] + lane_lt;
    } else if (is_eq) {
      output_col = tile_eq + warp_eq[warp] + lane_eq;
      if (output_col >= topk_arg) output_col = -1;
    }
    if (output_col >= 0 && output_col < topk_arg) {
      const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
      // Match the production fast-path contract: emit physical winners here.
      // The existing uniform winner-map kernel runs once after both paths.
      output[output_row + output_col] = static_cast<int32_t>(physical);
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (base_lt != first_count_lt || base_lt > topk_arg || base_eq < topk_arg) {
      block_status |= kCompactFailure;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    // Successful completion clears the recoverable overflow bit.
    status[row] = static_cast<int32_t>(block_status);
  }
}

}  // namespace overflow

static_assert(kBoundaryOverflow == overflow::kBoundaryOverflow);
static_assert(kBoundaryOverflow == 32u);

}  // namespace h2048_safe_topk


namespace {

using CandidateValue = dsa_litetopk::CandidateValue;

namespace pair_swap_gather {

namespace cg = cooperative_groups;

constexpr int kPlanThreads = 256;
constexpr int kGatherBlockY = 32;
constexpr int kGatherThreadsX = 8;
constexpr int kGatherVecBytes = 16;
constexpr int kHotSize = 12288;

// Restore the previous epoch's swaps, mark this epoch's HOT set, collect both
// sides of the bijection, and publish the new swaps in one cooperative launch.
// HOT12288 produces exactly 48 resident CTAs on the qualified B200 path.
template <typename HotIndexT>
__global__ __launch_bounds__(kPlanThreads, 1)
void cooperative_plan_kernel(
        const HotIndexT* __restrict__ hot,
        int* __restrict__ hot_epoch,
        int* __restrict__ permutation,
        int* __restrict__ swap_a,
        int* __restrict__ swap_b,
        int* __restrict__ counts,
        int hot_size,
        int window_start,
        int common_end,
        int epoch) {
    cg::grid_group grid = cg::this_grid();
    const int i = static_cast<int>(blockIdx.x) * kPlanThreads + threadIdx.x;
    const int lane = threadIdx.x & 31;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counts[1] = 0;
        counts[2] = 0;
    }
    const int old_count = max(0, min(counts[0], hot_size));
    if (i < old_count) {
        const int a = swap_a[i];
        const int b = swap_b[i];
        permutation[a] = a;
        permutation[b] = b;
    }
    if (i < hot_size) {
        const int64_t value = static_cast<int64_t>(hot[i]);
        if (value >= window_start && value < common_end) {
            const int previous = atomicExch(
                hot_epoch + static_cast<int>(value), epoch);
            if (previous == epoch) {
                atomicOr(counts + 3, 1);
            }
        } else {
            atomicOr(counts + 3, 2);
        }
    }
    grid.sync();

    const int64_t hot_value64 = static_cast<int64_t>(hot[i]);
    const bool hot_in_range =
        hot_value64 >= window_start && hot_value64 < common_end;
    const int hot_value =
        hot_in_range ? static_cast<int>(hot_value64) : window_start;
    const int window_value = window_start + i;
    const bool take_a = hot_in_range
        && hot_value >= window_start + hot_size;
    const bool take_b = hot_epoch[window_value] != epoch;

    const unsigned a_mask = __ballot_sync(0xffffffffu, take_a);
    int a_base = 0;
    if (lane == 0 && a_mask != 0) {
        a_base = atomicAdd(counts + 1, __popc(a_mask));
    }
    a_base = __shfl_sync(0xffffffffu, a_base, 0);
    if (take_a) {
        const int rank = __popc(a_mask & ((1u << lane) - 1u));
        swap_a[a_base + rank] = hot_value;
    }

    const unsigned b_mask = __ballot_sync(0xffffffffu, take_b);
    int b_base = 0;
    if (lane == 0 && b_mask != 0) {
        b_base = atomicAdd(counts + 2, __popc(b_mask));
    }
    b_base = __shfl_sync(0xffffffffu, b_base, 0);
    if (take_b) {
        const int rank = __popc(b_mask & ((1u << lane) - 1u));
        swap_b[b_base + rank] = window_value;
    }
    grid.sync();

    const int pair_count = min(counts[1], counts[2]);
    const int metadata_error = counts[3];
    if (metadata_error != 0 || counts[1] != counts[2]) {
        if (i == 0 && counts[1] != counts[2]) {
            atomicOr(counts + 3, 8);
        }
        asm volatile("trap;");
        return;
    }
    if (i < pair_count) {
        const int a = swap_a[i];
        const int b = swap_b[i];
        permutation[a] = b;
        permutation[b] = a;
    }
    if (i == 0) {
        counts[0] = pair_count;
    }
}

__global__ void paged_gather_kernel(
        const char* __restrict__ kv_cache,
        char* __restrict__ dst_k,
        char* __restrict__ dst_scale,
        const int* __restrict__ block_table,
        const int* __restrict__ permutation,
        int64_t token_stride,
        int64_t head_dim,
        int64_t block_stride,
        int64_t cache_block_size,
        int num_tokens,
        int quant_block_size) {
    const int dst_token = blockIdx.x * blockDim.y + threadIdx.y;
    const int head_idx =
        (blockIdx.y * blockDim.x + threadIdx.x) * kGatherVecBytes;

    // Each warp has four independent 8-lane token groups.  The x=0 lane
    // performs the permutation lookup and broadcasts it within its group.
    int source = threadIdx.x == 0 && dst_token < num_tokens
        ? permutation[dst_token] : -1;
    source = __shfl_sync(0xffffffffu, source, 0, kGatherThreadsX);
    if (head_idx >= head_dim || dst_token >= num_tokens) {
        return;
    }
    if (source < 0 || source >= num_tokens) {
        asm volatile("trap;");
        return;
    }

    const int block_idx = block_table[source / cache_block_size];
    const int64_t src_block_offset = block_idx * block_stride;
    const int64_t cache_inblock_offset =
        (source % cache_block_size) * head_dim + head_idx;
    const int64_t src_inblock_offset =
        src_block_offset + cache_inblock_offset;
    const int64_t dst_inblock_offset =
        static_cast<int64_t>(dst_token) * token_stride + head_idx;

    *reinterpret_cast<float4*>(dst_k + dst_inblock_offset) =
        *reinterpret_cast<const float4*>(kv_cache + src_inblock_offset);
    if (threadIdx.x == 0) {
        const int64_t src_scale_offset =
            src_block_offset + cache_block_size * head_dim
            + cache_inblock_offset * 4 / quant_block_size;
        *reinterpret_cast<float*>(
            dst_scale + dst_inblock_offset * 4 / quant_block_size) =
            *reinterpret_cast<const float*>(kv_cache + src_scale_offset);
    }
}

void validate_plan(
        const torch::Tensor& hot,
        const torch::Tensor& hot_epoch,
        const torch::Tensor& permutation,
        const torch::Tensor& swap_a,
        const torch::Tensor& swap_b,
        const torch::Tensor& counts,
        int64_t window_start,
        int64_t common_end,
        int64_t epoch) {
    TORCH_CHECK(
        hot.is_cuda() && hot_epoch.is_cuda() && permutation.is_cuda()
            && swap_a.is_cuda() && swap_b.is_cuda() && counts.is_cuda(),
        "all pair-swap tensors must be CUDA");
    TORCH_CHECK(
        (hot.scalar_type() == torch::kInt
             || hot.scalar_type() == torch::kLong)
            && hot_epoch.scalar_type() == torch::kInt
            && permutation.scalar_type() == torch::kInt
            && swap_a.scalar_type() == torch::kInt
            && swap_b.scalar_type() == torch::kInt
            && counts.scalar_type() == torch::kInt,
        "hot must be int32/int64; pair-swap workspaces must be int32");
    TORCH_CHECK(
        hot.is_contiguous() && hot_epoch.is_contiguous()
            && permutation.is_contiguous() && swap_a.is_contiguous()
            && swap_b.is_contiguous() && counts.is_contiguous(),
        "all pair-swap tensors must be contiguous");
    TORCH_CHECK(
        hot.dim() == 1 && hot_epoch.dim() == 1 && permutation.dim() == 1
            && swap_a.dim() == 1 && swap_b.dim() == 1 && counts.dim() == 1,
        "all pair-swap tensors must be vectors");
    TORCH_CHECK(
        hot.device() == hot_epoch.device()
            && hot.device() == permutation.device()
            && hot.device() == swap_a.device()
            && hot.device() == swap_b.device()
            && hot.device() == counts.device(),
        "all pair-swap tensors must be on one CUDA device");
    TORCH_CHECK(
        hot_epoch.numel() == permutation.numel(),
        "epoch and permutation lengths must match");
    TORCH_CHECK(
        swap_a.numel() >= hot.numel() && swap_b.numel() >= hot.numel(),
        "swap workspaces must hold HOT entries");
    TORCH_CHECK(
        hot.numel() == kHotSize,
        "production pair-swap planner requires exactly 12288 hot indices");
    TORCH_CHECK(counts.numel() >= 4, "counts must hold four int32 values");
    TORCH_CHECK(
        window_start >= 0 && window_start + hot.numel() <= common_end
            && common_end <= permutation.numel()
            && permutation.numel() <= std::numeric_limits<int>::max(),
        "expected 0 <= window_start, window_start + HOT <= common_end "
        "<= sequence length");
    TORCH_CHECK(
        epoch > 0 && epoch <= std::numeric_limits<int>::max(),
        "epoch must be positive int32");
}

void validate_gather(
        const torch::Tensor& kv_cache,
        const torch::Tensor& dst_k,
        const torch::Tensor& dst_scale,
        const torch::Tensor& block_table,
        const torch::Tensor& permutation) {
    TORCH_CHECK(
        kv_cache.is_cuda() && dst_k.is_cuda() && dst_scale.is_cuda()
            && block_table.is_cuda() && permutation.is_cuda(),
        "all gather tensors must be CUDA");
    TORCH_CHECK(
        kv_cache.device() == dst_k.device()
            && kv_cache.device() == dst_scale.device()
            && kv_cache.device() == block_table.device()
            && kv_cache.device() == permutation.device(),
        "all gather tensors must be on one CUDA device");
    TORCH_CHECK(
        kv_cache.scalar_type() == torch::kUInt8
            && dst_k.scalar_type() == torch::kUInt8
            && dst_scale.scalar_type() == torch::kUInt8,
        "cache/value/scale storage must be uint8");
    TORCH_CHECK(
        block_table.scalar_type() == torch::kInt
            && permutation.scalar_type() == torch::kInt,
        "block table and permutation must be int32");
    TORCH_CHECK(
        dst_k.is_contiguous()
            && dst_scale.is_contiguous() && block_table.is_contiguous()
            && permutation.is_contiguous(),
        "all gather tensors must be contiguous");
    TORCH_CHECK(
        kv_cache.dim() == 3 && kv_cache.stride(2) == 1
            && kv_cache.stride(1) == kv_cache.size(2)
            && kv_cache.stride(0) >= kv_cache.size(1) * kv_cache.size(2),
        "kv_cache blocks must be internally contiguous "
        "(dim0 may be strided: cross-layer cache slices)");
    TORCH_CHECK(
        kv_cache.dim() == 3 && dst_k.dim() == 2 && dst_scale.dim() == 2
            && block_table.dim() == 2 && block_table.size(0) == 1
            && permutation.dim() == 1,
        "invalid gather ranks");
    TORCH_CHECK(
        permutation.numel() == dst_k.size(0),
        "permutation must have one entry per destination token");
    // fp8 rows are 128 bytes, fp4 rows are 64 packed e2m1 bytes; both carry
    // 4 scale bytes per token and the kernel derives row bytes at runtime.
    TORCH_CHECK(
        dst_scale.size(0) == dst_k.size(0) && dst_scale.size(1) == 4
            && (dst_k.size(1) == 128 || dst_k.size(1) == 64),
        "production gather outputs must be K uint8 [S,128|64] and scale "
        "bytes [S,4]");
    TORCH_CHECK(
        dst_k.size(0) <= block_table.size(1) * kv_cache.size(1),
        "single-request block table does not cover the destination sequence");
}

template <typename HotIndexT>
void launch_plan_typed(
        const torch::Tensor& hot,
        const torch::Tensor& hot_epoch,
        const torch::Tensor& permutation,
        const torch::Tensor& swap_a,
        const torch::Tensor& swap_b,
        const torch::Tensor& counts,
        int window_start,
        int common_end,
        int epoch,
        cudaStream_t stream) {
    int hot_size = static_cast<int>(hot.numel());
    const int blocks = (hot_size + kPlanThreads - 1) / kPlanThreads;
    const HotIndexT* hot_ptr = hot.data_ptr<HotIndexT>();
    int* hot_epoch_ptr = hot_epoch.data_ptr<int>();
    int* permutation_ptr = permutation.data_ptr<int>();
    int* swap_a_ptr = swap_a.data_ptr<int>();
    int* swap_b_ptr = swap_b.data_ptr<int>();
    int* counts_ptr = counts.data_ptr<int>();
    void* args[] = {
        &hot_ptr, &hot_epoch_ptr, &permutation_ptr, &swap_a_ptr,
        &swap_b_ptr, &counts_ptr, &hot_size, &window_start, &common_end,
        &epoch,
    };
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(cooperative_plan_kernel<HotIndexT>),
        dim3(blocks), dim3(kPlanThreads), args, 0, stream));
}

void launch_plan(
        const torch::Tensor& hot,
        const torch::Tensor& hot_epoch,
        const torch::Tensor& permutation,
        const torch::Tensor& swap_a,
        const torch::Tensor& swap_b,
        const torch::Tensor& counts,
        int window_start,
        int common_end,
        int epoch,
        cudaStream_t stream) {
    if (hot.scalar_type() == torch::kLong) {
        launch_plan_typed<int64_t>(
            hot, hot_epoch, permutation, swap_a, swap_b, counts,
            window_start, common_end, epoch, stream);
    } else {
        launch_plan_typed<int32_t>(
            hot, hot_epoch, permutation, swap_a, swap_b, counts,
            window_start, common_end, epoch, stream);
    }
}

void launch_gather(
        const torch::Tensor& kv_cache,
        const torch::Tensor& dst_k,
        const torch::Tensor& dst_scale,
        const torch::Tensor& block_table,
        const torch::Tensor& permutation,
        cudaStream_t stream) {
    const int num_tokens = static_cast<int>(dst_k.size(0));
    const int head_dim = static_cast<int>(dst_k.size(1));
    const int quant_block_size = static_cast<int>(
        head_dim * 4 / dst_scale.size(1));
    const dim3 grid(
        (num_tokens + kGatherBlockY - 1) / kGatherBlockY,
        (head_dim + kGatherThreadsX * kGatherVecBytes - 1)
            / (kGatherThreadsX * kGatherVecBytes));
    const dim3 block(kGatherThreadsX, kGatherBlockY);
    paged_gather_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const char*>(kv_cache.data_ptr<uint8_t>()),
        reinterpret_cast<char*>(dst_k.data_ptr<uint8_t>()),
        reinterpret_cast<char*>(dst_scale.data_ptr<uint8_t>()),
        block_table.data_ptr<int>(),
        permutation.data_ptr<int>(),
        dst_k.stride(0),
        head_dim,
        kv_cache.stride(0),
        kv_cache.size(1),
        num_tokens,
        quant_block_size);
}

void plan_and_permuted_paged_gather_out(
        const torch::Tensor& hot,
        const torch::Tensor& hot_epoch,
        const torch::Tensor& permutation,
        const torch::Tensor& swap_a,
        const torch::Tensor& swap_b,
        const torch::Tensor& counts,
        int64_t window_start,
        int64_t common_end,
        int64_t epoch,
        const torch::Tensor& kv_cache,
        const torch::Tensor& dst_k,
        const torch::Tensor& dst_scale,
        const torch::Tensor& block_table) {
    validate_plan(
        hot, hot_epoch, permutation, swap_a, swap_b, counts,
        window_start, common_end, epoch);
    validate_gather(kv_cache, dst_k, dst_scale, block_table, permutation);
    TORCH_CHECK(
        hot.device() == kv_cache.device(),
        "planner and gather tensors must be on one CUDA device");

    const c10::cuda::CUDAGuard guard(hot.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_plan(
        hot, hot_epoch, permutation, swap_a, swap_b, counts,
        static_cast<int>(window_start), static_cast<int>(common_end),
        static_cast<int>(epoch), stream);
    launch_gather(kv_cache, dst_k, dst_scale, block_table, permutation, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pair_swap_gather

// Paged-candidate post-processing ABI.  Keep these constants beside the host
// validation so the producer, finalizer, selector, and Python-owned workspace
// cannot silently drift apart.
constexpr int kPagedCandidateInline = 8192;
constexpr int kPagedCandidatePage = 4096;
constexpr int kPagedCandidatePageShift = 12;
constexpr int kPagedCandidatePageMask = kPagedCandidatePage - 1;
constexpr int kPagedCandidateInvalidPageStatus = 32;
constexpr int kSegmentCandidateMicro = 512;
constexpr int kSegmentCandidatePage = 4096;
constexpr uint32_t kSegmentCandidateInvalidStart = 0xffffffffu;
constexpr int kSegmentCandidateDescriptorWarps = 8;
constexpr int kSegmentCandidateBlockQ = 4;
constexpr int kSegmentCandidateDescriptorMaxCount = 32 * 18;
constexpr int kSegmentCandidateSpillSlots = 2;
constexpr int kSegmentCandidateInvalidDescriptorStatus = 64;
constexpr int kSegmentCandidateWriterStatus = 128;

static_assert(
    (1 << kPagedCandidatePageShift) == kPagedCandidatePage,
    "paged candidate page size must be a power of two");
static_assert(
    kPagedCandidateInline % kPagedCandidatePage == 0,
    "paged candidate inline capacity must be page aligned");

struct PagedCandidateShape {
    int rows;
    int overflow_pages;
    int table_pages_per_row;
    int logical_cap;
};

struct SegmentCandidateShape {
    int rows;
    int desc_windows;
    int micro_records;
    int pool_pages;
    int physical_records;
};

static bool tensor_storage_ranges_overlap(
    const torch::Tensor& a, const torch::Tensor& b);

static torch::TensorOptions candidate_options(
        const torch::TensorOptions& options) {
    // torch.float16 is only the owning 16-bit storage type here.  CUDA treats
    // its payload as an opaque uint16 score code; no half arithmetic occurs.
    return options.dtype(torch::kHalf);
}

static CandidateValue* candidate_data_ptr(torch::Tensor& tensor) {
    return reinterpret_cast<CandidateValue*>(
        tensor.data_ptr<at::Half>());
}

static void check_candidate_dtype(const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.scalar_type() == torch::kHalf,
        "cand_val must use float16 as opaque packed storage");
}

static PagedCandidateShape check_paged_candidate_layout(
        const torch::Tensor& inline_val,
        const torch::Tensor& inline_idx,
        const torch::Tensor& overflow_val,
        const torch::Tensor& overflow_idx,
        const torch::Tensor& page_table,
        int64_t logical_cap64) {
    TORCH_CHECK(
        inline_val.is_cuda() && inline_idx.is_cuda() &&
            overflow_val.is_cuda() && overflow_idx.is_cuda() &&
            page_table.is_cuda(),
        "paged candidate tensors must be CUDA");
    TORCH_CHECK(
        inline_val.device() == inline_idx.device() &&
            inline_val.device() == overflow_val.device() &&
            inline_val.device() == overflow_idx.device() &&
            inline_val.device() == page_table.device(),
        "paged candidate tensors must be on one CUDA device");
    TORCH_CHECK(
        inline_val.is_contiguous() && inline_idx.is_contiguous() &&
            overflow_val.is_contiguous() && overflow_idx.is_contiguous() &&
            page_table.is_contiguous(),
        "paged candidate tensors must be contiguous");
    check_candidate_dtype(inline_val);
    check_candidate_dtype(overflow_val);
    TORCH_CHECK(
        inline_idx.scalar_type() == torch::kInt &&
            overflow_idx.scalar_type() == torch::kInt &&
            page_table.scalar_type() == torch::kInt,
        "paged candidate indices and page_table must be int32");
    TORCH_CHECK(
        inline_val.dim() == 2 &&
            inline_val.size(1) == kPagedCandidateInline &&
            inline_idx.sizes() == inline_val.sizes(),
        "inline_val/inline_idx must be [Q,8192]");
    TORCH_CHECK(
        overflow_val.dim() == 2 &&
            overflow_val.size(1) == kPagedCandidatePage &&
            overflow_idx.sizes() == overflow_val.sizes(),
        "overflow_val/overflow_idx must be [P,4096]");
    TORCH_CHECK(
        page_table.dim() == 2 &&
            page_table.size(0) == inline_val.size(0),
        "page_table must be [Q,Npages]");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(inline_val, inline_idx) &&
            !tensor_storage_ranges_overlap(inline_val, overflow_val) &&
            !tensor_storage_ranges_overlap(inline_val, overflow_idx) &&
            !tensor_storage_ranges_overlap(inline_val, page_table) &&
            !tensor_storage_ranges_overlap(inline_idx, overflow_val) &&
            !tensor_storage_ranges_overlap(inline_idx, overflow_idx) &&
            !tensor_storage_ranges_overlap(inline_idx, page_table) &&
            !tensor_storage_ranges_overlap(overflow_val, overflow_idx) &&
            !tensor_storage_ranges_overlap(overflow_val, page_table) &&
            !tensor_storage_ranges_overlap(overflow_idx, page_table),
        "paged candidate arenas and page_table must not alias");
    TORCH_CHECK(
        inline_val.size(0) > 0 &&
            inline_val.size(0) <= std::numeric_limits<int>::max() &&
            overflow_val.size(0) <= std::numeric_limits<int>::max() &&
            page_table.size(1) <= std::numeric_limits<int>::max(),
        "paged candidate dimensions exceed the int32 ABI");
    TORCH_CHECK(
        logical_cap64 >= 1 &&
            logical_cap64 < std::numeric_limits<int32_t>::max(),
        "logical_cap must leave room for an int32 overflow sentinel");
    const int64_t overflow_capacity =
        std::max<int64_t>(logical_cap64 - kPagedCandidateInline, 0);
    const int64_t required_table_pages =
        (overflow_capacity + kPagedCandidatePage - 1) /
        kPagedCandidatePage;
    TORCH_CHECK(
        page_table.size(1) == required_table_pages,
        "page_table width must exactly match logical_cap");
    return {
        static_cast<int>(inline_val.size(0)),
        static_cast<int>(overflow_val.size(0)),
        static_cast<int>(page_table.size(1)),
        static_cast<int>(logical_cap64),
    };
}

static SegmentCandidateShape check_segment_candidate_layout(
        const torch::Tensor& seed_val,
        const torch::Tensor& seed_idx,
        const torch::Tensor& segment_val,
        const torch::Tensor& segment_idx,
        const torch::Tensor& segment_desc) {
    TORCH_CHECK(
        seed_val.is_cuda() && seed_idx.is_cuda() &&
            segment_val.is_cuda() && segment_idx.is_cuda() &&
            segment_desc.is_cuda(),
        "segmented candidate tensors must be CUDA");
    TORCH_CHECK(
        seed_val.device() == seed_idx.device() &&
            seed_val.device() == segment_val.device() &&
            seed_val.device() == segment_idx.device() &&
            seed_val.device() == segment_desc.device(),
        "segmented candidate tensors must be on one CUDA device");
    TORCH_CHECK(
        seed_val.is_contiguous() && seed_idx.is_contiguous() &&
            segment_val.is_contiguous() && segment_idx.is_contiguous() &&
            segment_desc.is_contiguous(),
        "segmented candidate tensors must be contiguous");
    check_candidate_dtype(seed_val);
    check_candidate_dtype(segment_val);
    TORCH_CHECK(
        seed_idx.scalar_type() == torch::kInt &&
            segment_idx.scalar_type() == torch::kInt &&
            segment_desc.scalar_type() == torch::kLong,
        "segmented indices must be int32 and descriptors must be int64");
    TORCH_CHECK(
        seed_val.dim() == 2 &&
            seed_val.size(0) > 0 &&
            seed_val.size(0) <= std::numeric_limits<int>::max() &&
            seed_val.size(1) == kPagedCandidateInline &&
            seed_idx.sizes() == seed_val.sizes(),
        "seed_val/seed_idx must be [Q,8192]");
    TORCH_CHECK(
        segment_val.dim() == 1 &&
            segment_idx.sizes() == segment_val.sizes() &&
            segment_val.numel() <= std::numeric_limits<int>::max(),
        "segment_val/segment_idx must be equally-sized flat arenas");
    TORCH_CHECK(
        segment_desc.dim() == 3 &&
            segment_desc.size(0) == seed_val.size(0) &&
            segment_desc.size(1) > 0 &&
            segment_desc.size(1) <= std::numeric_limits<int>::max() &&
            segment_desc.size(2) == kSegmentCandidateDescriptorWarps,
        "segment_desc must be int64 [Q,W,8]");

    const int64_t rows = seed_val.size(0);
    const int64_t q_blocks =
        (rows + kSegmentCandidateBlockQ - 1) /
        kSegmentCandidateBlockQ;
    const int64_t micro_records =
        q_blocks * kSegmentCandidateDescriptorWarps *
        kSegmentCandidateMicro;
    TORCH_CHECK(
        segment_val.numel() > micro_records &&
            (segment_val.numel() - micro_records) %
                kSegmentCandidatePage == 0,
        "segment arena must be micro_records + P*4096 with P>0");
    const int64_t pool_pages =
        (segment_val.numel() - micro_records) /
        kSegmentCandidatePage;
    TORCH_CHECK(
        pool_pages > 0 &&
            segment_val.numel() < kSegmentCandidateInvalidStart,
        "segmented physical arena exceeds the uint32 descriptor ABI");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(seed_val, seed_idx) &&
            !tensor_storage_ranges_overlap(seed_val, segment_val) &&
            !tensor_storage_ranges_overlap(seed_val, segment_idx) &&
            !tensor_storage_ranges_overlap(seed_val, segment_desc) &&
            !tensor_storage_ranges_overlap(seed_idx, segment_val) &&
            !tensor_storage_ranges_overlap(seed_idx, segment_idx) &&
            !tensor_storage_ranges_overlap(seed_idx, segment_desc) &&
            !tensor_storage_ranges_overlap(segment_val, segment_idx) &&
            !tensor_storage_ranges_overlap(segment_val, segment_desc) &&
            !tensor_storage_ranges_overlap(segment_idx, segment_desc),
        "seed, segment, and descriptor storage must not alias");
    return {
        static_cast<int>(rows),
        static_cast<int>(segment_desc.size(1)),
        static_cast<int>(micro_records),
        static_cast<int>(pool_pages),
        static_cast<int>(segment_val.numel()),
    };
}

static void check_segment_spill_layout(
        const torch::Tensor& spill_desc,
        const torch::Tensor& row_spill,
        const torch::Tensor& segment_desc,
        const SegmentCandidateShape& shape) {
    TORCH_CHECK(
        spill_desc.is_cuda() && row_spill.is_cuda() &&
            spill_desc.device() == segment_desc.device() &&
            row_spill.device() == segment_desc.device(),
        "spill descriptors/row flags must be CUDA on the segment device");
    TORCH_CHECK(
        spill_desc.is_contiguous() && row_spill.is_contiguous(),
        "spill descriptors/row flags must be contiguous");
    TORCH_CHECK(
        spill_desc.scalar_type() == torch::kLong &&
            row_spill.scalar_type() == torch::kInt,
        "spill_desc must be int64 and row_spill must be int32");
    TORCH_CHECK(
        spill_desc.dim() == 4 &&
            spill_desc.size(0) == shape.rows &&
            spill_desc.size(1) == shape.desc_windows &&
            spill_desc.size(2) == kSegmentCandidateDescriptorWarps &&
            spill_desc.size(3) == kSegmentCandidateSpillSlots,
        "spill_desc must be [Q,W,8,2]");
    TORCH_CHECK(
        row_spill.dim() == 1 && row_spill.numel() == shape.rows,
        "row_spill must have Q int32 elements");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(spill_desc, segment_desc) &&
            !tensor_storage_ranges_overlap(row_spill, segment_desc) &&
            !tensor_storage_ranges_overlap(row_spill, spill_desc),
        "base/spill descriptors and row flags must not alias");
}

static bool tensor_storage_ranges_overlap(
        const torch::Tensor& a,
        const torch::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) return false;
    const uintptr_t a_begin =
        reinterpret_cast<uintptr_t>(a.data_ptr());
    const uintptr_t b_begin =
        reinterpret_cast<uintptr_t>(b.data_ptr());
    const uintptr_t a_end =
        a_begin + static_cast<uintptr_t>(a.numel()) * a.element_size();
    const uintptr_t b_end =
        b_begin + static_cast<uintptr_t>(b.numel()) * b.element_size();
    return a_begin < b_end && b_begin < a_end;
}

static void* driver_handle() {
    static void* h = nullptr;
    if (!h) {
        h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        TORCH_CHECK(h, "failed to load libcuda.so.1");
    }
    return h;
}

static CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt, cuuint32_t rank,
                          void* addr, const cuuint64_t* dims, const cuuint64_t* strides,
                          const cuuint32_t* box, const cuuint32_t* estrides,
                          CUtensorMapInterleave il, CUtensorMapSwizzle sw,
                          CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
    using FT = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                            const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static FT f = nullptr;
    if (!f) {
        f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
        TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
    }
    return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer,
                           int smem_inner, int smem_outer,
                           long gmem_outer_stride, int swizzle_mode) {
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
    CUtensorMap tm;
    const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
    const cuuint32_t estrides[2] = {1, 1};
    CUtensorMapSwizzle swizzle =
        swizzle_mode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_mode == 64  ? CU_TENSOR_MAP_SWIZZLE_64B :
        swizzle_mode == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 4;         // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1;   // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200
constexpr int ONLINE_FIXED_BUCKETS = 256;

// Offline-only fixed-bucket launch prep.  The scan owns the complete per-row
// histogram in shared memory, so only its externally visible gate/count state
// and the three-word boundary certificate need initialization.
__global__ void online_fixed_init_litetopk_kernel(
        int32_t* __restrict__ th_bucket,
        int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ boundary_meta,
        int rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    th_bucket[row] = ONLINE_FIXED_BUCKETS - 1;
    cand_cnt[row] = 0;
    int32_t* meta =
        boundary_meta + static_cast<size_t>(row) * ONLINE_FIXED_BUCKETS;
    meta[0] = 0;
    meta[1] = 0;
    meta[2] = 0;
}

// Gather the compact hot sample's FP8 rows and their per-row FP32 scales in
// one launch. One warp owns one output row; its first eight lanes issue a
// single coalesced 128-byte vector copy and lane zero also copies the scale.
// A capped persistent grid avoids launching one CTA per hot row at N=8192.
template <typename IndexT>
__global__ void gather_hot_sample_litetopk_kernel(
    const uint4* __restrict__ k,
    const float* __restrict__ k_scale,
    const IndexT* __restrict__ idx,
    uint4* __restrict__ out_k,
    float* __restrict__ out_scale,
    int64_t seq_len,
    int hot_n) {
    constexpr int kWarpsPerBlock = 8;
    constexpr int kVecsPerRow = HEAD_DIM / static_cast<int>(sizeof(uint4));
    static_assert(kVecsPerRow == 8, "the fused gather requires D=128");
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_row = blockIdx.x * kWarpsPerBlock + warp;
    const int row_stride = gridDim.x * kWarpsPerBlock;

    for (int out_row = first_row; out_row < hot_n;
         out_row += row_stride) {
        int64_t src_row =
            lane == 0 ? static_cast<int64_t>(idx[out_row]) : int64_t{0};
        src_row = __shfl_sync(0xffffffffu, src_row, 0);

        // The host can validate dtype/layout/device without synchronizing,
        // but only the device sees index values. Assert instead of clamping:
        // clamping would silently duplicate a row and could tighten the DSA
        // threshold enough to reduce recall.
        if (src_row < 0 || src_row >= seq_len) {
            if (lane == 0) {
                assert(src_row >= 0 && src_row < seq_len);
            }
            continue;
        }
        if (lane < kVecsPerRow) {
            out_k[static_cast<int64_t>(out_row) * kVecsPerRow + lane] =
                k[src_row * kVecsPerRow + lane];
        }
        if (lane == 0) {
            out_scale[out_row] = k_scale[src_row];
        }
    }
}

void gather_hot_sample_litetopk_(
        torch::Tensor k,
        torch::Tensor k_scale,
        torch::Tensor idx,
        torch::Tensor out_k,
        torch::Tensor out_scale) {
    TORCH_CHECK(
        k.is_cuda() && k_scale.is_cuda() && idx.is_cuda() &&
            out_k.is_cuda() && out_scale.is_cuda(),
        "k, k_scale, idx, out_k, and out_scale must be CUDA tensors");
    TORCH_CHECK(
        k.device() == k_scale.device() && k.device() == idx.device() &&
            k.device() == out_k.device() && k.device() == out_scale.device(),
        "all fused hot-gather tensors must be on the same CUDA device");
    TORCH_CHECK(
        k.is_contiguous() && k_scale.is_contiguous() &&
            idx.is_contiguous() && out_k.is_contiguous() &&
            out_scale.is_contiguous(),
        "all fused hot-gather tensors must be contiguous");
    TORCH_CHECK(
        k.scalar_type() == torch::kFloat8_e4m3fn &&
            out_k.scalar_type() == torch::kFloat8_e4m3fn,
        "k and out_k must be fp8_e4m3fn");
    TORCH_CHECK(
        k_scale.scalar_type() == torch::kFloat &&
            out_scale.scalar_type() == torch::kFloat,
        "k_scale and out_scale must be fp32");
    TORCH_CHECK(
        idx.scalar_type() == torch::kLong ||
            idx.scalar_type() == torch::kInt,
        "idx must be int64 or int32");
    TORCH_CHECK(
        k.dim() == 2 && k.size(1) == HEAD_DIM,
        "k must have shape [S, 128]");
    TORCH_CHECK(
        k_scale.dim() == 1 && k_scale.numel() == k.size(0),
        "k_scale must have shape [S]");
    TORCH_CHECK(idx.dim() == 1, "idx must have shape [hot_n]");
    TORCH_CHECK(
        out_k.dim() == 2 && out_k.size(0) == idx.numel() &&
            out_k.size(1) == HEAD_DIM,
        "out_k must have shape [hot_n, 128]");
    TORCH_CHECK(
        out_scale.dim() == 1 && out_scale.numel() == idx.numel(),
        "out_scale must have shape [hot_n]");
    TORCH_CHECK(
        idx.numel() <= 8192,
        "fused hot gather supports at most 8192 indices");
    TORCH_CHECK(
        k.size(0) <= std::numeric_limits<int>::max(),
        "fused hot gather supports at most INT_MAX source rows");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(k.data_ptr()) % alignof(uint4) == 0 &&
            reinterpret_cast<uintptr_t>(out_k.data_ptr()) %
                    alignof(uint4) ==
                0,
        "k and out_k must be 16-byte aligned");
    TORCH_CHECK(
        k.data_ptr() != out_k.data_ptr() &&
            k_scale.data_ptr() != out_scale.data_ptr(),
        "fused hot gather does not support aliased input/output storage");

    const int hot_n = static_cast<int>(idx.numel());
    if (hot_n == 0) {
        return;
    }
    const c10::cuda::CUDAGuard device_guard(k.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    constexpr int kThreads = 256;
    constexpr int kRowsPerBlock = kThreads / 32;
    const int blocks =
        std::min((hot_n + kRowsPerBlock - 1) / kRowsPerBlock, NUM_SMS);
    if (idx.scalar_type() == torch::kLong) {
        gather_hot_sample_litetopk_kernel<<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const uint4*>(k.data_ptr()),
            k_scale.data_ptr<float>(),
            idx.data_ptr<int64_t>(),
            reinterpret_cast<uint4*>(out_k.data_ptr()),
            out_scale.data_ptr<float>(),
            k.size(0),
            hot_n);
    } else {
        gather_hot_sample_litetopk_kernel<<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const uint4*>(k.data_ptr()),
            k_scale.data_ptr<float>(),
            idx.data_ptr<int32_t>(),
            reinterpret_cast<uint4*>(out_k.data_ptr()),
            out_scale.data_ptr<float>(),
            k.size(0),
            hot_n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void seed_bcount_kernel(
    const float* __restrict__ seed_val,
    int seed_k,
    const float* __restrict__ origin,
    const float* __restrict__ inv_delta,
    int32_t* __restrict__ bcount,
    int R,
    int NB) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R) return;
    float o = origin[row];
    float inv = inv_delta[row];
    for (int i = tid; i < seed_k; i += blockDim.x) {
        float x = seed_val[(size_t)row * seed_k + i];
        int braw = static_cast<int>((x - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (braw < NB) atomicAdd(&bcount[(size_t)row * NB + b], 1);
    }
}

__global__ void refresh_threshold_from_bcount_kernel(
    int32_t* __restrict__ th_bucket,
    const int32_t* __restrict__ bcount,
    int R,
    int NB,
    int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R) return;
    int old_th = th_bucket[row];
    int cum = 0;
    int new_th = old_th;
    bool found = false;
    for (int b = 0; b < NB; ++b) {
        cum += bcount[(size_t)row * NB + b];
        if (cum >= K) {
            new_th = b;
            found = true;
            break;
        }
    }
    if (found && new_th < old_th) th_bucket[row] = new_th;
}

// Fused seed/prep kernel (one block per row, all state in smem — borrows the
// vLLM top_k_per_row engineering): from the sample scores [Q, head] derive the
// per-row bucket params (origin, inv_delta), the initial gate threshold
// (bucket of the K-th best sample score), write the FULL sample histogram into
// bcount (a valid, conservative refresh base: counting genuine row elements
// can only tighten th safely), and emit every sample position with
// bucket <= th as initial candidates — a SUPERSET of the sample top-K, which
// the exact final select trims. Replaces: aminmax + torch.topk/radix seed +
// neg/contiguous copies + host seed copies + seed_bcount_kernel (~6 passes,
// ~10 launches) with 3 passes in 1 launch.
constexpr int kSeedThreads = 256;
constexpr int kSeed12Threads = 256;

template <bool kEmitCandidates, int kRetainedHead, int BT>
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride,
    const int head, const int NB, const int K,
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket,
    CandidateValue* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx,
    int32_t* __restrict__ cand_cnt,
    const int cand_cap,
    const int physical_index_base,
    int32_t* __restrict__ bcount_out) {
    constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
    static_assert(
        kRetainedHead == 8192 || kRetainedHead == 12288,
        "production seed supports only the qualified 8K/12K layouts");
    constexpr int kRetainVecs = kRetainedHead / (BT * 4);
    const int row = gridDim.x - 1 - blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float* srow = slog + (size_t)row * slog_stride;
    extern __shared__ int s_hist[];  // NSUB * NB ints

    // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
    // when the caller passes clean_logits=True full-row logits (dense-select
    // mode) for the out-of-range causal tail; it must not poison the range.
    __shared__ float s_mx[BT / 32];
    __shared__ float s_mn[BT / 32];
    float mx = -INFINITY, mn = INFINITY;
    const auto acc = [&](const float s) {
        if (isfinite(s)) {
            mx = fmaxf(mx, s);
            mn = fminf(mn, s);
        }
    };
    // The production 8K and 12K specializations retain respectively eight
    // and twelve float4 values per thread. Keep them live across the CTA
    // reduction so histogram construction and emission never reread the
    // materialized prefix logits. Missing tail lanes carry -inf and are
    // ignored by the generic <=8K compatibility path.
    static_assert(
        BT == 256 || BT == 384 || BT == 512,
        "retained HOT seed requires a qualified CTA size");
    static_assert(BT % (NSUB * 32) == 0,
                  "each seed sub-histogram must own whole warps");
    float4 retained[kRetainVecs];
    if (head == kRetainedHead) {
#pragma unroll
        for (int it = 0; it < kRetainVecs; ++it) {
            const int j = tid * 4 + it * BT * 4;
            const float4 s4 =
                *reinterpret_cast<const float4*>(srow + j);
            retained[it] = s4;
            acc(s4.x); acc(s4.y); acc(s4.z); acc(s4.w);
        }
    } else {
#pragma unroll
        for (int it = 0; it < kRetainVecs; ++it) {
            const int j = tid * 4 + it * BT * 4;
            float4 s4 =
                make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            if (j + 3 < head) {
                s4 = *reinterpret_cast<const float4*>(srow + j);
            } else {
                if (j < head) s4.x = srow[j];
                if (j + 1 < head) s4.y = srow[j + 1];
                if (j + 2 < head) s4.z = srow[j + 2];
            }
            retained[it] = s4;
            acc(s4.x); acc(s4.y); acc(s4.z); acc(s4.w);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
    }
    if (lane == 0) { s_mx[tid >> 5] = mx; s_mn[tid >> 5] = mn; }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int wgi = 1; wgi < BT / 32; ++wgi) {
            s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
            s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
        }
    }
    __syncthreads();
    float o = -s_mx[0];         // min over x = -score
    const float hi = -s_mn[0];  // max over x
    const float span = fmaxf(hi - o, 1e-20f);
    o -= headroom * span;       // forward (above-max) drift headroom
    float inv = (NB - 1) / (span * (1.0f + headroom));
    const float vth = -o * inv;

    // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
    // smem atomic conflicts, vectorized loads.
    for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
    __syncthreads();
    int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
    const auto bucket_of = [&](const float s) -> int {
        // Use the byte-for-byte arithmetic contract consumed by both the
        // seed emitter and the main scan.  Computing (-s - o) * inv as two
        // rounded operations can put a boundary value one bucket below its
        // FMA result: the histogram would then certify K records while the
        // emitter rejects one of them, producing a silent underfill.
        const float bq = fmaf(-s, inv, vth);
        int b = static_cast<int>(bq);
        return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
    };
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
        const float4 s4 = retained[it];
        if (isfinite(s4.x))
            atomicAdd(&my_hist[bucket_of(s4.x)], 1);
        if (isfinite(s4.y))
            atomicAdd(&my_hist[bucket_of(s4.y)], 1);
        if (isfinite(s4.z))
            atomicAdd(&my_hist[bucket_of(s4.z)], 1);
        if (isfinite(s4.w))
            atomicAdd(&my_hist[bucket_of(s4.w)], 1);
    }
    __syncthreads();
    // merge sub-histograms into s_hist[0..NB)
    for (int b = tid; b < NB; b += BT) {
        int c = s_hist[b];
        #pragma unroll
        for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
        s_hist[b] = c;
    }
    __syncthreads();
    if (bcount_out != nullptr) {
        // Full-row overwrite of the sample histogram. The ring daemon warm-
        // starts its refresh base from these counts: they are genuine row
        // records in the final (origin, inv) bucket space, so adding them to
        // the daemon's subset cum can only tighten the published edge safely
        // — provided the main scan starts after the sampled prefix (the
        // exact-once contract), or the same records would count twice.
        for (int b = tid; b < NB; b += BT)
            bcount_out[(size_t)row * NB + b] = s_hist[b];
    }
    // Coarse K-th estimate on the single (o, inv) scale built above. There
    // is deliberately NO scale rebuild: th_bucket, origin/inv, the emitted
    // candidates, and the exported bcount histogram must all share one
    // bucket space — the ring warm-start base is only sound under that
    // identity. Headroom above the sample max keeps drifted scores out of
    // bucket 0 where refresh could never resolve them.
    // The production U16 contract uses emit_limit==0 and a single KV split.
    // Its scan covers the complete KV range and initializes the CTA-local
    // histogram itself, so writing Q*NB zeros to global memory is dead work.
    // Find the first histogram prefix that reaches K in parallel.  The old
    // single-thread walk serialized 256 dependent shared-memory loads while
    // the other 1023 threads waited.  NB <= BT gives every bin one owner;
    // the half-open prefix ranges are disjoint, so exactly one thread writes
    // the same threshold as the serial "first cumulative sum >= K" rule.
    __shared__ int s_th;
    __shared__ int s_wsum[BT / 32];
    if (tid == 0) s_th = NB - 1;
    const int h = (tid < NB) ? s_hist[tid] : 0;
    int x = h;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if ((tid & 31) >= off) x += y;
    }
    if ((tid & 31) == 31) s_wsum[tid >> 5] = x;
    __syncthreads();
    int base = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w)
        if (w < (tid >> 5)) base += s_wsum[w];
    const int incl = base + x;
    const int excl = incl - h;
    if (tid < NB && excl < K && K <= incl) s_th = tid;
    __syncthreads();
    if (tid == 0) {
        th_bucket[row] = s_th;
        origin[row] = o;
        inv_delta[row] = inv;
    }
    __syncthreads();
    if constexpr (kEmitCandidates) {
        // Large exact-once mode: the HOT scores retained above are the physical
        // prefix [physical_index_base, physical_index_base + head).  Emit their
        // passing records now, then let the main producer start after `head`.
        //
        // One iteration covers 256 * float4 == 1024 ordered columns.  A warp
        // scan plus eight warp totals gives every thread a deterministic CTA
        // prefix; this reserves no global counter per candidate.  The one CTA
        // owning the row publishes the true (possibly over-cap) total once.
        int emitted_before = 0;
        const float gate_edge = static_cast<float>(s_th + 1);
        const uint64_t row_base =
            static_cast<uint64_t>(row) * cand_cap;
#pragma unroll
        for (int it = 0; it < kRetainVecs; ++it) {
            const int j0 = tid * 4 + it * BT * 4;
            const float4 s4 = retained[it];
            const float score[4] = {s4.x, s4.y, s4.z, s4.w};
            float bq[4];
            bool pass[4];
            int local_count = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                bq[k] = fmaf(-score[k], inv, vth);
                pass[k] = j0 + k < head && isfinite(score[k]) &&
                    __float_as_int(bq[k]) < __float_as_int(gate_edge);
                local_count += pass[k] ? 1 : 0;
            }

            int warp_inclusive = local_count;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int other = __shfl_up_sync(
                    0xffffffffu, warp_inclusive, off);
                if (lane >= off) warp_inclusive += other;
            }
            if (lane == 31) s_wsum[tid >> 5] = warp_inclusive;
            __syncthreads();

            int warp_before = 0;
#pragma unroll
            for (int w = 0; w < BT / 32; ++w) {
                if (w < (tid >> 5)) warp_before += s_wsum[w];
            }
            const int thread_base = emitted_before + warp_before +
                warp_inclusive - local_count;
            int local_rank = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (pass[k]) {
                    const int out = thread_base + local_rank++;
                    if (out < cand_cap) {
                        const uint32_t physical_idx =
                            static_cast<uint32_t>(physical_index_base + j0 + k);
                        dsa_litetopk::store_candidate(
                            cand_val + row_base + out,
                            cand_idx + row_base + out,
                            bq[k],
                            physical_idx);
                    }
                }
            }

            int block_total = 0;
#pragma unroll
            for (int w = 0; w < BT / 32; ++w)
                block_total += s_wsum[w];
            emitted_before += block_total;
            // Do not let an early warp overwrite s_wsum for the next retained
            // group while a slower warp still consumes this group's totals.
            __syncthreads();
        }
        if (tid == 0) cand_cnt[row] = emitted_before;
    } else {
        if (tid == 0) cand_cnt[row] = 0;
    }
}

__device__ __forceinline__ uint32_t compact_enc_float(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

// Find the first radix digit whose inclusive histogram prefix reaches kfind.
//
// The old selector assigned this to tid==0, serializing 256 dependent shared
// loads while the other 255 threads waited. Warp 0 instead treats the radix
// as 32 groups of eight bins:
//   1. every lane sums one eight-bin group and warp-scans the group totals;
//   2. the winning group is broadcast, lanes 0..7 scan its eight bins.
//
// This keeps the exact "first prefix >= k" rule, including empty bins and
// ties, and needs no extra CTA barrier: callers already synchronize after the
// histogram fill and again after desired/kfind are published.
__device__ __forceinline__ void
compact_find_radix_digit_warp0(
    const uint32_t* __restrict__ hist,
    uint32_t* __restrict__ desired,
    uint32_t* __restrict__ kfind,
    const uint32_t desired_base,
    const int shift,
    const int tid) {
    if (tid >= 32) return;
    constexpr unsigned FULL = 0xffffffffu;
    const int lane = tid;
    const int group_start = lane * 8;
    uint32_t group_count = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        group_count += hist[group_start + i];

    uint32_t group_inclusive = group_count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(
            FULL, group_inclusive, offset);
        if (lane >= offset) group_inclusive += other;
    }

    const uint32_t target = *kfind;
    if (target == 0u) return;
    const unsigned group_mask = __ballot_sync(
        FULL, group_inclusive >= target);
    // Match the serial fallback exactly for an underfilled histogram: leave
    // desired/kfind unchanged instead of deriving an invalid -1 group.
    if (group_mask == 0u) return;
    const int winning_group = __ffs(group_mask) - 1;
    const uint32_t group_before = __shfl_sync(
        FULL, group_inclusive - group_count, winning_group);

    const uint32_t digit_count =
        lane < 8 ? hist[winning_group * 8 + lane] : 0u;
    uint32_t digit_inclusive = digit_count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(
            FULL, digit_inclusive, offset);
        if (lane >= offset) digit_inclusive += other;
    }
    const unsigned digit_mask = __ballot_sync(
        FULL,
        lane < 8 &&
            group_before + digit_inclusive >= target);
    if (digit_mask == 0u) return;
    const int winning_lane = __ffs(digit_mask) - 1;
    const uint32_t digit_before =
        group_before +
        __shfl_sync(
            FULL, digit_inclusive - digit_count, winning_lane);

    if (lane == 0) {
        const uint32_t digit =
            static_cast<uint32_t>(winning_group * 8 + winning_lane);
        *desired =
            desired_base | (digit << static_cast<uint32_t>(shift));
        *kfind = target - digit_before;
    }
}

// Rebuild the exact 256-bin boundary certificate after the fixed-threshold
// scan. Candidate indices remain in physical workspace space for selection;
// only final TOPK winners are mapped by the following grid-wide epilogue.
__global__ void finalize_static_hot_meta_litetopk_kernel(
        const CandidateValue* __restrict__ cand_val,
        const int32_t* __restrict__ cand_idx,
        const int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ th_bucket,
        int32_t* __restrict__ boundary_meta,
        int32_t* __restrict__ status,
        int index_limit,
        int rows,
        int cand_cap,
        int num_buckets,
        int topk) {
    constexpr int kThreads = 256;
    constexpr int kBins = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= rows) return;

    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int scan_status;
    const int raw_n = cand_cnt[row];
    const int n = raw_n < 0 ? 0 : min(raw_n, cand_cap);
    if (tid == 0) {
        int st = 0;
        if (raw_n < 0 || raw_n > cand_cap) st |= 1;
        if (n < topk) st |= 2;
        scan_status = st;
        desired = 0u;
        kfind = static_cast<uint32_t>(topk);
    }
    hist[tid] = 0u;
    __syncthreads();

    if (scan_status != 0) {
        if (tid == 0) {
            status[row] = scan_status;
            int32_t* meta =
                boundary_meta +
                static_cast<uint64_t>(row) * num_buckets;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    const uint64_t row_base =
        static_cast<uint64_t>(row) * cand_cap;
    for (int j = tid; j < n; j += kThreads) {
        const uint64_t offset = row_base + j;
        const int32_t packed_idx = cand_idx[offset];
        const int physical_idx =
            dsa_litetopk::candidate_decode_index(packed_idx);
        // The late-map production path keeps candidate indices in physical
        // workspace space until selection.  When an index bound is supplied,
        // retain the old mapped-finalizer's fail-closed check without paying
        // a random permutation read or a candidate-index writeback.
        if (index_limit > 0 &&
            (physical_idx < 0 || physical_idx >= index_limit)) {
            atomicOr(&scan_status, 16);
            continue;
        }
        const float value = dsa_litetopk::candidate_decode_score(
            cand_val[offset], packed_idx);
        if (!isfinite(value)) {
            atomicOr(&scan_status, 4);
            continue;
        }
        const int bucket = value < 0.0f ? 0 :
            (value >= static_cast<float>(num_buckets)
                ? num_buckets - 1
                : static_cast<int>(value));
        atomicAdd(hist + bucket, 1u);
    }
    __syncthreads();

    if (scan_status != 0) {
        if (tid == 0) {
            status[row] = scan_status;
            int32_t* meta =
                boundary_meta +
                static_cast<uint64_t>(row) * num_buckets;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    compact_find_radix_digit_warp0(
        hist, &desired, &kfind, 0u, 0, tid);
    __syncthreads();
    if (tid == 0) {
        const int threshold = static_cast<int>(desired);
        const int count_lt =
            topk - static_cast<int>(kfind);
        const int count_eq =
            threshold >= 0 && threshold < num_buckets
                ? static_cast<int>(hist[threshold])
                : 0;
        int st = scan_status;
        const int need = topk - count_lt;
        if (threshold >= num_buckets || count_lt < 0 ||
            count_lt >= topk || need <= 0 || need > count_eq) {
            st |= 8;
        }
        status[row] = st;
        int32_t* meta =
            boundary_meta +
            static_cast<uint64_t>(row) * num_buckets;
        if (st == 0) {
            th_bucket[row] = threshold;
            meta[0] = ~threshold;
            meta[1] = count_lt;
            meta[2] = count_eq;
        } else {
            // Keep the certificate deliberately invalid so an unchecked
            // production-selector call traps instead of returning bad top-k.
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
    }
}

__device__ __forceinline__ void
finalize_static_hot_mapped_paged_record(
        const CandidateValue value,
        int32_t* __restrict__ packed_idx_ptr,
        const int32_t* __restrict__ index_map,
        const int index_map_size,
        const int num_buckets,
        uint32_t* __restrict__ hist,
        int* __restrict__ scan_status) {
    int32_t packed_idx = *packed_idx_ptr;
    const int physical_idx =
        dsa_litetopk::candidate_decode_index(packed_idx);
    if (physical_idx < 0 || physical_idx >= index_map_size) {
        atomicOr(scan_status, 16);
        return;
    }
    const int original_idx = index_map[physical_idx];
    if (original_idx < 0 || original_idx >= index_map_size ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
        atomicOr(scan_status, 16);
        return;
    }
    packed_idx = static_cast<int32_t>(
        (static_cast<uint32_t>(packed_idx) &
         ~dsa_litetopk::kCandidateIndexMask) |
        static_cast<uint32_t>(original_idx));
    *packed_idx_ptr = packed_idx;

    const float decoded =
        dsa_litetopk::candidate_decode_score(value, packed_idx);
    if (!isfinite(decoded)) {
        atomicOr(scan_status, 4);
        return;
    }
    const int bucket = decoded < 0.0f
        ? 0
        : (decoded >= static_cast<float>(num_buckets)
               ? num_buckets - 1
               : static_cast<int>(decoded));
    atomicAdd(hist + bucket, 1u);
}

// Paged counterpart of the production mapped finalizer.  Page-table entries
// are immutable during this launch.  Thread zero fetches each page id exactly
// once and broadcasts it through shared memory; all record traffic within a
// page remains contiguous and coalesced.  Status bit 32 denotes a missing or
// out-of-range overflow page and invalidates the selector certificate.
__global__ void finalize_static_hot_meta_mapped_paged_litetopk_kernel(
        const CandidateValue* __restrict__ inline_val,
        int32_t* __restrict__ inline_idx,
        const CandidateValue* __restrict__ overflow_val,
        int32_t* __restrict__ overflow_idx,
        const int32_t* __restrict__ page_table,
        const int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ th_bucket,
        int32_t* __restrict__ boundary_meta,
        int32_t* __restrict__ status,
        const int32_t* __restrict__ index_map,
        int index_map_size,
        int rows,
        int overflow_pages,
        int table_pages_per_row,
        int logical_cap,
        int num_buckets,
        int topk) {
    constexpr int kThreads = 256;
    constexpr int kBins = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= rows) return;

    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int scan_status;
    __shared__ int page_id;

    const int raw_n = cand_cnt[row];
    const int n = raw_n < 0 ? 0 : min(raw_n, logical_cap);
    if (tid == 0) {
        int st = 0;
        if (raw_n < 0 || raw_n > logical_cap) st |= 1;
        if (n < topk) st |= 2;
        scan_status = st;
        desired = 0u;
        kfind = static_cast<uint32_t>(topk);
    }
    hist[tid] = 0u;
    __syncthreads();

    int32_t* const meta =
        boundary_meta + static_cast<uint64_t>(row) * num_buckets;
    if (scan_status != 0) {
        if (tid == 0) {
            status[row] = scan_status;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    const int inline_n = min(n, kPagedCandidateInline);
    const uint64_t inline_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    for (int j = tid; j < inline_n; j += kThreads) {
        const uint64_t offset = inline_base + j;
        finalize_static_hot_mapped_paged_record(
            inline_val[offset], inline_idx + offset,
            index_map, index_map_size, num_buckets,
            hist, &scan_status);
    }
    __syncthreads();

    const int overflow_n = max(n - kPagedCandidateInline, 0);
    const int num_pages =
        (overflow_n + kPagedCandidatePage - 1) >>
        kPagedCandidatePageShift;
    for (int logical_page = 0;
         logical_page < num_pages && scan_status == 0;
         ++logical_page) {
        if (tid == 0) {
            const int id =
                logical_page < table_pages_per_row
                    ? page_table[
                          static_cast<uint64_t>(row) *
                              table_pages_per_row +
                          logical_page]
                    : -1;
            page_id = id;
            if (id < 0 || id >= overflow_pages) {
                scan_status |= kPagedCandidateInvalidPageStatus;
            }
        }
        __syncthreads();
        if (scan_status != 0) break;

        const int page_n = min(
            overflow_n - logical_page * kPagedCandidatePage,
            kPagedCandidatePage);
        const uint64_t page_base =
            static_cast<uint64_t>(page_id) * kPagedCandidatePage;
        for (int j = tid; j < page_n; j += kThreads) {
            const uint64_t offset = page_base + j;
            finalize_static_hot_mapped_paged_record(
                overflow_val[offset], overflow_idx + offset,
                index_map, index_map_size, num_buckets,
                hist, &scan_status);
        }
        __syncthreads();
    }

    if (scan_status != 0) {
        if (tid == 0) {
            status[row] = scan_status;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    compact_find_radix_digit_warp0(
        hist, &desired, &kfind, 0u, 0, tid);
    __syncthreads();
    if (tid == 0) {
        const int threshold = static_cast<int>(desired);
        const int count_lt = topk - static_cast<int>(kfind);
        const int count_eq =
            threshold >= 0 && threshold < num_buckets
                ? static_cast<int>(hist[threshold])
                : 0;
        const int need = topk - count_lt;
        int st = scan_status;
        if (threshold >= num_buckets || count_lt < 0 ||
            count_lt >= topk || need <= 0 || need > count_eq) {
            st |= 8;
        }
        status[row] = st;
        if (st == 0) {
            th_bucket[row] = threshold;
            meta[0] = ~threshold;
            meta[1] = count_lt;
            meta[2] = count_eq;
        } else {
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
    }
}

__device__ __forceinline__ bool
segment_candidate_descriptor_valid(
        const uint32_t physical_start,
        const uint32_t count,
        const int row,
        const int producer_warp,
        const int micro_records,
        const int pool_pages,
        const int physical_records,
        const uint32_t allocated_pages) {
    if (count == 0u) {
        // The scan wrapper zero-initializes the descriptor arena.  Both an
        // unwritten out-of-range window and the writer's explicit
        // (INVALID,0) empty span are semantically empty.
        return true;
    }
    if (physical_start == kSegmentCandidateInvalidStart ||
        count > static_cast<uint32_t>(
                    kSegmentCandidateDescriptorMaxCount) ||
        static_cast<uint64_t>(physical_start) + count >
            static_cast<uint64_t>(physical_records)) {
        return false;
    }
    const int q_block = row / kSegmentCandidateBlockQ;
    const uint32_t micro_base = static_cast<uint32_t>(
        (q_block * kSegmentCandidateDescriptorWarps + producer_warp) *
        kSegmentCandidateMicro);
    if (physical_start < static_cast<uint32_t>(micro_records)) {
        return physical_start >= micro_base &&
            physical_start + count <=
                micro_base + kSegmentCandidateMicro;
    }
    const uint32_t pooled =
        physical_start - static_cast<uint32_t>(micro_records);
    const uint32_t page =
        pooled / kSegmentCandidatePage;
    const uint32_t page_offset =
        pooled % kSegmentCandidatePage;
    return page < static_cast<uint32_t>(pool_pages) &&
        page < allocated_pages &&
        page_offset + count <= kSegmentCandidatePage;
}

__device__ __forceinline__ bool
segment_candidate_spill_descriptor_valid(
        const uint32_t physical_start,
        const uint32_t count,
        const int micro_records,
        const int pool_pages,
        const int physical_records,
        const uint32_t allocated_pages) {
    if (count == 0u) return true;
    if (physical_start == kSegmentCandidateInvalidStart ||
        count > static_cast<uint32_t>(kSegmentCandidatePage) ||
        physical_start < static_cast<uint32_t>(micro_records) ||
        static_cast<uint64_t>(physical_start) + count >
            static_cast<uint64_t>(physical_records)) {
        return false;
    }
    const uint32_t pooled =
        physical_start - static_cast<uint32_t>(micro_records);
    const uint32_t page = pooled / kSegmentCandidatePage;
    return pooled % kSegmentCandidatePage == 0u &&
        page < static_cast<uint32_t>(pool_pages) &&
        page < allocated_pages;
}

// One CTA per row.  The seed prefix is processed by all 256 threads; then the
// eight CTA warps consume the eight producer-warp descriptors in parallel for
// every window.  Descriptor traffic therefore needs one lane load + shuffles
// per warp/window and no window-level CTA barriers.
__global__ void
finalize_static_hot_meta_mapped_segmented_litetopk_kernel(
        const CandidateValue* __restrict__ seed_val,
        int32_t* __restrict__ seed_idx,
        const int32_t* __restrict__ seed_cnt,
        const CandidateValue* __restrict__ segment_val,
        int32_t* __restrict__ segment_idx,
        const uint64_t* __restrict__ segment_desc,
        const uint64_t* __restrict__ spill_desc,
        const int32_t* __restrict__ row_spill,
        const int32_t* __restrict__ segment_pool_head,
        const int32_t* __restrict__ writer_status,
        int32_t* __restrict__ total_cnt,
        int32_t* __restrict__ th_bucket,
        int32_t* __restrict__ boundary_meta,
        int32_t* __restrict__ status,
        const int32_t* __restrict__ index_map,
        int index_map_size,
        int rows,
        int desc_windows,
        int micro_records,
        int pool_pages,
        int physical_records,
        int logical_cap,
        int num_buckets,
        int topk) {
    constexpr int kThreads = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int producer_warp = tid >> 5;
    if (row >= rows) return;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int scan_status;
    __shared__ int total_count;
    __shared__ uint32_t allocated_pages;
    __shared__ int has_spill;

    const int raw_seed_n = seed_cnt[row];
    const int seed_n = raw_seed_n < 0
        ? 0
        : min(raw_seed_n, kPagedCandidateInline);
    if (tid == 0) {
        int st = 0;
        if (raw_seed_n < 0 || raw_seed_n > kPagedCandidateInline) {
            st |= 1;
        }
        const int raw_pool_head = segment_pool_head[0];
        allocated_pages = raw_pool_head < 0
            ? 0u
            : static_cast<uint32_t>(raw_pool_head);
        if (raw_pool_head < 0 || raw_pool_head > pool_pages) {
            st |= kSegmentCandidateInvalidDescriptorStatus;
        }
        if (writer_status[0] != 0) {
            st |= kSegmentCandidateWriterStatus;
        }
        const int spill_flag = row_spill[row];
        if (spill_flag < 0) {
            st |= kSegmentCandidateInvalidDescriptorStatus;
        }
        has_spill = spill_flag != 0;
        scan_status = st;
        total_count = seed_n;
        desired = 0u;
        kfind = static_cast<uint32_t>(topk);
    }
    hist[tid] = 0u;
    __syncthreads();

    const uint64_t seed_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    if (scan_status == 0) {
        for (int j = tid; j < seed_n; j += kThreads) {
            const uint64_t offset = seed_base + j;
            finalize_static_hot_mapped_paged_record(
                seed_val[offset], seed_idx + offset,
                index_map, index_map_size, num_buckets,
                hist, &scan_status);
        }
    }

    int warp_segment_count = 0;
    for (int window = 0; window < desc_windows; ++window) {
        uint32_t physical_start = kSegmentCandidateInvalidStart;
        uint32_t count = 0u;
        bool valid = false;
        if (lane == 0) {
            const uint64_t descriptor = segment_desc[
                (static_cast<uint64_t>(row) * desc_windows + window) *
                    kSegmentCandidateDescriptorWarps +
                producer_warp];
            physical_start = static_cast<uint32_t>(descriptor);
            count = static_cast<uint32_t>(descriptor >> 32);
            valid = segment_candidate_descriptor_valid(
                physical_start, count, row, producer_warp,
                micro_records, pool_pages, physical_records,
                allocated_pages);
            if (!valid) {
                atomicOr(
                    &scan_status,
                    kSegmentCandidateInvalidDescriptorStatus);
                count = 0u;
            }
            warp_segment_count += static_cast<int>(count);
        }
        physical_start = __shfl_sync(0xffffffffu, physical_start, 0);
        count = __shfl_sync(0xffffffffu, count, 0);
        for (uint32_t j = static_cast<uint32_t>(lane);
             j < count;
             j += 32u) {
            const uint64_t offset =
                static_cast<uint64_t>(physical_start) + j;
            finalize_static_hot_mapped_paged_record(
                segment_val[offset], segment_idx + offset,
                index_map, index_map_size, num_buckets,
                hist, &scan_status);
        }
    }
    if (has_spill) {
        for (int window = 0; window < desc_windows; ++window) {
#pragma unroll
            for (int spill_slot = 0;
                 spill_slot < kSegmentCandidateSpillSlots;
                 ++spill_slot) {
                uint32_t physical_start = 0u;
                uint32_t count = 0u;
                if (lane == 0) {
                    const uint64_t descriptor = spill_desc[
                        ((static_cast<uint64_t>(row) * desc_windows +
                          window) * kSegmentCandidateDescriptorWarps +
                         producer_warp) * kSegmentCandidateSpillSlots +
                        spill_slot];
                    physical_start = static_cast<uint32_t>(descriptor);
                    count = static_cast<uint32_t>(descriptor >> 32);
                    const bool valid =
                        segment_candidate_spill_descriptor_valid(
                            physical_start, count, micro_records,
                            pool_pages, physical_records,
                            allocated_pages);
                    if (!valid) {
                        atomicOr(
                            &scan_status,
                            kSegmentCandidateInvalidDescriptorStatus);
                        count = 0u;
                    }
                    warp_segment_count += static_cast<int>(count);
                }
                physical_start = __shfl_sync(
                    0xffffffffu, physical_start, 0);
                count = __shfl_sync(0xffffffffu, count, 0);
                for (uint32_t j = static_cast<uint32_t>(lane);
                     j < count;
                     j += 32u) {
                    const uint64_t offset =
                        static_cast<uint64_t>(physical_start) + j;
                    finalize_static_hot_mapped_paged_record(
                        segment_val[offset], segment_idx + offset,
                        index_map, index_map_size, num_buckets,
                        hist, &scan_status);
                }
            }
        }
    }
    if (lane == 0 && warp_segment_count != 0) {
        atomicAdd(&total_count, warp_segment_count);
    }
    __syncthreads();

    if (tid == 0) {
        const int n = total_count;
        if (n < topk) scan_status |= 2;
        if (n < 0 || n > logical_cap) scan_status |= 1;
        total_cnt[row] = n;
    }
    __syncthreads();

    int32_t* const meta =
        boundary_meta + static_cast<uint64_t>(row) * num_buckets;
    if (scan_status != 0) {
        if (tid == 0) {
            status[row] = scan_status;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    compact_find_radix_digit_warp0(
        hist, &desired, &kfind, 0u, 0, tid);
    __syncthreads();
    if (tid == 0) {
        const int threshold = static_cast<int>(desired);
        const int count_lt = topk - static_cast<int>(kfind);
        const int count_eq =
            threshold >= 0 && threshold < num_buckets
                ? static_cast<int>(hist[threshold])
                : 0;
        const int need = topk - count_lt;
        int st = scan_status;
        if (threshold >= num_buckets || count_lt < 0 ||
            count_lt >= topk || need <= 0 || need > count_eq) {
            st |= 8;
        }
        status[row] = st;
        if (st == 0) {
            th_bucket[row] = threshold;
            meta[0] = ~threshold;
            meta[1] = count_lt;
            meta[2] = count_eq;
        } else {
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
    }
}

// Turn the emitted high24 records directly into an exact continuation bound.
// Three radix bytes find their exact production-ABI pivot; this kernel then
// retains every record below that pivot plus exactly enough pivot records to
// restore K entries.  The old path first launched a coarse 256-bin finalizer,
// but that certificate did not narrow any of these exact radix passes.  Doing
// count/finite validation here removes one full candidate read and one launch
// from every safety checkpoint without weakening the bound.
//
// th_bucket is phase-reused to publish the ordered32 pivot edge
// (pivot24 << 8) consumed by the exact-gate no-hist scan.  Later records with
// the same high24 code are equivalent under the production ABI, so excluding
// them is an exact tie break while keeping every refine bounded.
//
// Compaction remains tile-local and alias-safe: every source record is loaded
// before the CTA barrier, and at most one output exists per processed source,
// so no destination reaches the next unread tile.
__global__ void compact_static_hot_candidates_litetopk_kernel(
        CandidateValue* __restrict__ cand_val,
        int32_t* __restrict__ cand_idx,
        int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ th_bucket,
        int32_t* __restrict__ boundary_meta,
        int32_t* __restrict__ status,
        int rows,
        int cand_cap,
        int num_buckets,
        int topk) {
    constexpr int kThreads = 256;
    constexpr unsigned kFullMask = 0xffffffffu;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    if (row >= rows) return;

    const int raw_n = cand_cnt[row];
    const int n = raw_n < 0 ? 0 : min(raw_n, cand_cap);
    int32_t* const meta =
        boundary_meta + static_cast<uint64_t>(row) * num_buckets;

    __shared__ uint32_t exact_hist[256];
    __shared__ uint32_t exact_desired;
    __shared__ uint32_t exact_kfind;
    __shared__ int compact_status;
    __shared__ int compact_count;
    __shared__ int equal_seen;
    if (tid == 0) {
        int st = 0;
        if (raw_n < 0 || raw_n > cand_cap) st |= 1;
        if (n < topk) st |= 2;

        compact_status = st;
        compact_count = 0;
        equal_seen = 0;
        exact_desired = 0u;
        exact_kfind = static_cast<uint32_t>(topk);
        if (st != 0) {
            // Make the failure persistent across the remaining suffix scan.
            // The final mapped finalizer rewrites status from cand_cnt and the
            // rebuilt certificate, so status/meta alone could otherwise hide
            // this error.  An over-cap sentinel also prevents the producer
            // from writing any more records; in particular it normalizes a
            // corrupt negative count that would make its atomic reservation
            // return a negative output offset.
            status[row] = st;
            cand_cnt[row] = cand_cap + 1;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
    }
    __syncthreads();
    if (compact_status != 0) return;

    const uint64_t row_base =
        static_cast<uint64_t>(row) * cand_cap;

    uint32_t exact_mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = 16 - pass * 8;
        exact_hist[tid] = 0u;
        __syncthreads();
        const uint32_t desired = exact_desired;
        for (int j = tid; j < n; j += kThreads) {
            const uint64_t source = row_base + j;
            const uint32_t code =
                dsa_litetopk::candidate_load_score_code(
                    cand_val[source], cand_idx[source]);
            if (pass == 0) {
                const float value =
                    dsa_litetopk::candidate_decode_score(
                        cand_val[source], cand_idx[source]);
                if (!isfinite(value)) {
                    atomicOr(&compact_status, 4);
                }
            }
            if ((code & exact_mask) == (desired & exact_mask)) {
                atomicAdd(&exact_hist[(code >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (compact_status != 0) {
            if (tid == 0) {
                status[row] = compact_status;
                cand_cnt[row] = cand_cap + 1;
                meta[0] = 0;
                meta[1] = 0;
                meta[2] = 0;
            }
            return;
        }
        compact_find_radix_digit_warp0(
            exact_hist,
            &exact_desired,
            &exact_kfind,
            desired,
            shift,
            tid);
        __syncthreads();
        exact_mask |= 0xffu << shift;
    }

    const uint32_t pivot = exact_desired;
    const int equal_take = static_cast<int>(exact_kfind);
    if (equal_take <= 0 || equal_take > topk ||
        pivot >= 0x00ffffffu) {
        if (tid == 0) {
            status[row] = 8;
            cand_cnt[row] = cand_cap + 1;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
        return;
    }

    for (int tile = 0; tile < n; tile += kThreads) {
        const int j = tile + tid;
        CandidateValue value{};
        int32_t packed_idx = 0;
        uint32_t code = 0u;
        bool valid = false;
        if (j < n) {
            const uint64_t source = row_base + j;
            value = cand_val[source];
            packed_idx = cand_idx[source];
            code = dsa_litetopk::candidate_load_score_code(
                value, packed_idx);
            valid = true;
        }

        const bool is_equal = valid && code == pivot;
        const unsigned equal_mask =
            __ballot_sync(kFullMask, is_equal);
        int equal_base = 0;
        if (lane == 0) {
            equal_base = atomicAdd(&equal_seen, __popc(equal_mask));
        }
        equal_base = __shfl_sync(kFullMask, equal_base, 0);
        const int equal_rank =
            equal_base +
            __popc(equal_mask & ((1u << lane) - 1u));

        const bool keep = valid &&
            (code < pivot || (is_equal && equal_rank < equal_take));
        const unsigned keep_mask = __ballot_sync(kFullMask, keep);
        int warp_base = 0;
        if (lane == 0) {
            warp_base = atomicAdd(&compact_count, __popc(keep_mask));
        }
        warp_base = __shfl_sync(kFullMask, warp_base, 0);
        const int output =
            warp_base + __popc(keep_mask & ((1u << lane) - 1u));

        // All sources in this tile are resident before any warp overwrites a
        // compacted destination.
        __syncthreads();
        if (keep) {
            const uint64_t destination = row_base + output;
            cand_val[destination] = value;
            cand_idx[destination] = packed_idx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        if (compact_count != topk ||
            equal_seen < equal_take) {
            // The radix pivot and stored records disagree. Leave an over-cap
            // sentinel so a later finalizer cannot hide the failed refine.
            status[row] = 8;
            cand_cnt[row] = cand_cap + 1;
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        } else {
            cand_cnt[row] = compact_count;
            // int32 is only storage here. The continuation scan reloads this
            // bit pattern as unsigned ordered32 and applies a strict compare.
            // The pivot edge excludes later high24 ties; they are equivalent
            // to the retained pivot entries under the six-byte ABI.
            th_bucket[row] = static_cast<int32_t>(pivot << 8);
            status[row] = 0;
            // This coarse certificate no longer matches th_bucket. The final
            // mapped finalizer (or the next coarse refine) rebuilds it.
            meta[0] = 0;
            meta[1] = 0;
            meta[2] = 0;
        }
    }
}

// Deferred overflow telemetry used to be expressed as
//
//   stack({cand_cnt.max(), cand_cnt.float().mean().int()})
//
// which launches five ATen kernels before the asynchronous eight-byte D2H
// copy.  Candidate counts are already contiguous int32 values, so one CTA is
// enough for all supported Q (8128/8192).  Accumulate in signed 64 bits:
// even the supported 1M-token upper bound sums to about 8.6e9 and therefore
// cannot be represented by int32.
//
// The second output is the exact integer mean, truncated toward zero.  ATen's
// FP32 tree reduction can very rarely round that value across an integer
// boundary.  The production HOTONLY path marks every successful probe as
// strided, where the mean is telemetry-only and AUTO state is not updated;
// Python retains the old ATen expression for any future non-strided path.
__global__ void cand_count_stats_litetopk_kernel(
    const int32_t* __restrict__ cand_cnt,
    int count,
    int32_t* __restrict__ stats) {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int32_t local_max = (-2147483647 - 1);
    int64_t local_sum = 0;
    for (int i = tid; i < count; i += kThreads) {
        const int32_t value = cand_cnt[i];
        local_max = max(local_max, value);
        local_sum += static_cast<int64_t>(value);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = max(
            local_max,
            __shfl_down_sync(0xffffffffu, local_max, offset));
        local_sum +=
            __shfl_down_sync(0xffffffffu, local_sum, offset);
    }

    __shared__ int32_t warp_max[kWarps];
    __shared__ int64_t warp_sum[kWarps];
    if (lane == 0) {
        warp_max[warp] = local_max;
        warp_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        int32_t block_max =
            lane < kWarps ? warp_max[lane] : (-2147483647 - 1);
        int64_t block_sum = lane < kWarps ? warp_sum[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = max(
                block_max,
                __shfl_down_sync(0xffffffffu, block_max, offset));
            block_sum +=
                __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (lane == 0) {
            stats[0] = block_max;
            stats[1] = static_cast<int32_t>(
                block_sum / static_cast<int64_t>(count));
        }
    }
}

// Selector-fused carry votes have a much smaller value domain than their
// corpus-index domain.  A selected corpus position can receive at most one
// vote from each sampled query row, hence max_vote=ceil(Q/row_stride)<=8192
// while the histogram itself can contain up to 1M positions.  Exploit that
// bounded domain directly instead of sending the 1M int32 values through a
// general-purpose topk.
//
// The operation is deliberately split into exactly two kernels:
//
//   1. Each CTA builds a local count-of-counts histogram for a contiguous
//      8192-position tile and writes it to caller-owned int16 partial storage.
//      The CTA that receives the final completion ticket reduces those
//      partials, finds the exact vote threshold, resolves threshold ties by
//      ascending corpus index, and publishes deterministic per-CTA offsets.
//   2. CTAs stably compact the selected indices to int64 output while clearing
//      every live vote.  The output is ascending by corpus index, which is a
//      friendlier order for the next index_select than vote-sorted output.
//
// The partial workspace and state are exclusive to one ordered stream.  The
// final CTA resets the completion ticket before kernel exit, so the same
// workspace can be reused by the next call without a memset or host sync.
constexpr int kCarryTileItems = 8192;
constexpr int kCarryMaxItems = 1 << 20;
constexpr int kCarryMaxK = 12288;
constexpr int kCarryMaxVote = 8192;
constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;
constexpr int kCarryThreads = 256;
constexpr int kCarryWarps = kCarryThreads / 32;
constexpr int kCarryPlannerWindow = 8192;

enum CarryStateOffset : int {
    kCarryTicket = 0,
    kCarryThreshold = 1,
    kCarryTieBlock = 2,
    kCarryTieTake = 3,
    kCarryOutK = 4,
    kCarryNumBlocks = 5,
    kCarryBlockOffsets = 6,
};
constexpr int kCarryStateInts =
    kCarryBlockOffsets + kCarryMaxBlocks + 1;

enum CarryPlannerStateOffset : int {
    kCarryPlannerPairCount = 0,
    kCarryPlannerTicket = 1,
    kCarryPlannerStatus = 2,
    kCarryPlannerSelectedInside = 3,
    kCarryPlannerPreviousWindow = 4,
};
constexpr int kCarryPlannerStateInts = 5;

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <bool FusePlanner>
__global__ void carry_votes_plan_litetopk_kernel(
    const int32_t* __restrict__ votes,
    int count,
    int min_index,
    int out_k,
    int max_vote,
    volatile int16_t* __restrict__ partial,
    int partial_stride,
    int32_t* __restrict__ state,
    int32_t* __restrict__ permutation,
    int32_t* __restrict__ planner_state) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int block = blockIdx.x;
    const int begin =
        (FusePlanner ? min_index : 0) + block * kCarryTileItems;
    const int end = min(begin + kCarryTileItems, count);
    const int bins = max_vote + 1;
    const int previous_window_start =
        FusePlanner ? planner_state[kCarryPlannerPreviousWindow] : 0;

    // In the fused path the physical HOT window itself is the compact list of
    // the previous pair swaps. Restore those pairs while K1 is already reading
    // the next vote histogram; no epoch array or planner launch is needed.
    if constexpr (FusePlanner) {
        for (int offset = block * kCarryThreads + tid;
             offset < kCarryPlannerWindow;
             offset += gridDim.x * kCarryThreads) {
            const int b = previous_window_start + offset;
            const int a = permutation[b];
            if (a != b) {
                permutation[a] = a;
                permutation[b] = b;
            }
        }
    }

    extern __shared__ uint32_t s_freq[];
    __shared__ int s_warp_sum[kCarryWarps];
    __shared__ int s_last;
    __shared__ int s_scan_base;
    __shared__ int s_found;
    __shared__ int s_threshold;
    __shared__ int s_count_gt;
    __shared__ int s_tie_block;
    __shared__ int s_tie_take;
    __shared__ int s_block_count[kCarryMaxBlocks];

    for (int bin = tid; bin < bins; bin += kCarryThreads) {
        s_freq[bin] = 0;
    }
    __syncthreads();

    // Zero votes dominate most corpora. Count them in registers and reduce
    // once instead of serializing every zero through one shared atomic.
    int local_zero = 0;
    for (int index = begin + tid; index < end; index += kCarryThreads) {
        if (index < min_index) {
            continue;
        }
        int value = votes[index];
        // The selector emits unique winners per sampled row, so this clamp is
        // unreachable under the public ABI. Keep release builds memory-safe
        // if an upstream invariant is violated.
        value = value < 0 ? 0 : (value > max_vote ? max_vote : value);
        if (value == 0) {
            ++local_zero;
        } else {
            atomicAdd(&s_freq[value], 1u);
        }
    }
    local_zero = carry_warp_sum(local_zero);
    if (lane == 0) {
        s_warp_sum[warp] = local_zero;
    }
    __syncthreads();
    if (warp == 0) {
        int value = lane < kCarryWarps ? s_warp_sum[lane] : 0;
        value = carry_warp_sum(value);
        if (lane == 0) {
            s_freq[0] = static_cast<uint32_t>(value);
        }
    }
    __syncthreads();

    volatile int16_t* block_partial =
        partial + static_cast<size_t>(block) * partial_stride;
    for (int bin = tid; bin < bins; bin += kCarryThreads) {
        // A CTA owns at most 8192 positions, safely inside signed int16.
        block_partial[bin] = static_cast<int16_t>(s_freq[bin]);
    }
    // Every thread publishes its own global stores. A fence in tid0 alone
    // would not release the other 255 writers before the completion ticket.
    __threadfence();
    __syncthreads();

    // CUDA's canonical "last block" reduction pattern. No CTA spins: every
    // non-last block exits, while the last ticket holder sees all partial
    // writes made visible before the atomic increment.
    if (tid == 0) {
        const int old = atomicAdd(&state[kCarryTicket], 1);
        s_last = old == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }

    for (int bin = tid; bin < bins; bin += kCarryThreads) {
        int total = 0;
        for (int source_block = 0;
             source_block < gridDim.x;
             ++source_block) {
            total += static_cast<int>(
                partial[
                    static_cast<size_t>(source_block) * partial_stride +
                    bin]);
        }
        s_freq[bin] = static_cast<uint32_t>(total);
    }
    __syncthreads();

    // Descending 256-bin tiles. This is the seed-prep parallel prefix in the
    // opposite direction, extended to the dynamic [0,max_vote] domain.
    if (tid == 0) {
        s_scan_base = 0;
        s_found = 0;
        s_threshold = 0;
        s_count_gt = 0;
    }
    __syncthreads();
    for (int tile = 0; tile < bins; tile += kCarryThreads) {
        const int bin = max_vote - tile - tid;
        const int count_here =
            bin >= 0 ? static_cast<int>(s_freq[bin]) : 0;
        int inclusive = count_here;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other =
                __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset) {
                inclusive += other;
            }
        }
        if (lane == 31) {
            s_warp_sum[warp] = inclusive;
        }
        __syncthreads();
        int warp_base = 0;
#pragma unroll
        for (int source_warp = 0;
             source_warp < kCarryWarps;
             ++source_warp) {
            if (source_warp < warp) {
                warp_base += s_warp_sum[source_warp];
            }
        }
        const int exclusive =
            s_scan_base + warp_base + inclusive - count_here;
        const int inclusive_global = exclusive + count_here;
        if (bin >= 0 && exclusive < out_k &&
            out_k <= inclusive_global) {
            s_threshold = bin;
            s_count_gt = exclusive;
            s_found = 1;
        }
        __syncthreads();
        if (s_found) {
            break;
        }
        if (tid == 0) {
            int tile_total = 0;
#pragma unroll
            for (int source_warp = 0;
                 source_warp < kCarryWarps;
                 ++source_warp) {
                tile_total += s_warp_sum[source_warp];
            }
            s_scan_base += tile_total;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int need_equal = out_k - s_count_gt;
        int equal_before = 0;
        s_tie_block = gridDim.x - 1;
        s_tie_take = 0;
        for (int source_block = 0;
             source_block < gridDim.x;
             ++source_block) {
            const int equal_here = static_cast<int>(
                partial[
                    static_cast<size_t>(source_block) * partial_stride +
                    s_threshold]);
            if (equal_before < need_equal &&
                need_equal <= equal_before + equal_here) {
                s_tie_block = source_block;
                s_tie_take = need_equal - equal_before;
                break;
            }
            equal_before += equal_here;
        }
    }
    __syncthreads();

    // Compute each block's exact stable-output size. Warps read one partial
    // row at a time so the second partial pass remains coalesced.
    for (int source_block = warp;
         source_block < gridDim.x;
         source_block += kCarryWarps) {
        int selected = 0;
        for (int bin = s_threshold + 1 + lane;
             bin < bins;
             bin += 32) {
            selected += static_cast<int>(
                partial[
                    static_cast<size_t>(source_block) * partial_stride +
                    bin]);
        }
        selected = carry_warp_sum(selected);
        if (lane == 0) {
            int equal_take = 0;
            if (source_block < s_tie_block) {
                equal_take = static_cast<int>(
                    partial[
                        static_cast<size_t>(source_block) *
                            partial_stride +
                        s_threshold]);
            } else if (source_block == s_tie_block) {
                equal_take = s_tie_take;
            }
            s_block_count[source_block] = selected + equal_take;
        }
    }
    __syncthreads();

    // The last K1 CTA already owns the exact selected count for the first
    // aligned 8192-position window. It initializes all K2 planner state, so
    // the fused path needs neither a memset nor a separate init kernel.
    if constexpr (FusePlanner) {
        if (tid == 0) {
            const int selected_inside = s_block_count[0];
            planner_state[kCarryPlannerPairCount] =
                kCarryPlannerWindow - selected_inside;
            planner_state[kCarryPlannerTicket] = 0;
            planner_state[kCarryPlannerStatus] = 0;
            planner_state[kCarryPlannerSelectedInside] = selected_inside;
            planner_state[kCarryPlannerPreviousWindow] = min_index;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int offset = 0;
        for (int source_block = 0;
             source_block < gridDim.x;
             ++source_block) {
            state[kCarryBlockOffsets + source_block] = offset;
            offset += s_block_count[source_block];
        }
        state[kCarryBlockOffsets + gridDim.x] = offset;
        state[kCarryThreshold] = s_threshold;
        state[kCarryTieBlock] = s_tie_block;
        state[kCarryTieTake] = s_tie_take;
        state[kCarryOutK] = out_k;
        state[kCarryNumBlocks] = gridDim.x;
        __threadfence();
        atomicExch(&state[kCarryTicket], 0);
    }
}

template <bool FusePlanner>
__global__ void carry_votes_emit_reset_litetopk_kernel(
    int32_t* __restrict__ votes,
    int count,
    int min_index,
    int max_vote,
    int64_t* __restrict__ out_idx,
    const int32_t* __restrict__ state,
    int32_t* __restrict__ permutation,
    int32_t* __restrict__ planner_state,
    int window_start,
    int target_length) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int block = blockIdx.x;
    const int begin =
        (FusePlanner ? min_index : 0) + block * kCarryTileItems;
    const int threshold = state[kCarryThreshold];
    const int tie_block = state[kCarryTieBlock];
    const int tie_take = state[kCarryTieTake];
    const int output_base = state[kCarryBlockOffsets + block];

    __shared__ int s_warp_count[kCarryWarps];
    __shared__ int s_warp_prefix[kCarryWarps];
    __shared__ int s_tile_output_base;
    __shared__ int s_tie_seen;
    __shared__ int s_tile_total;
    if (tid == 0) {
        s_tile_output_base = 0;
        s_tie_seen = 0;
    }
    __syncthreads();

    if constexpr (FusePlanner) {
        // The aligned fused grid excludes LongCat's sink positions. Preserve
        // the existing contract that K2 clears every live vote.
        if (block == 0) {
            for (int index = tid; index < min_index; index += kCarryThreads) {
                votes[index] = 0;
            }
        }
    }

    constexpr unsigned kFullMask = 0xffffffffu;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);
    for (int tile = 0;
         tile < kCarryTileItems;
         tile += kCarryThreads) {
        const int index = begin + tile + tid;
        const bool valid = index < count;
        const int raw_value = valid ? votes[index] : 0;
        const int value =
            raw_value < 0 ? 0 :
            (raw_value > max_vote ? max_vote : raw_value);
        if (valid) {
            votes[index] = 0;
        }
        const bool eligible = valid && index >= min_index;
        const bool is_equal = eligible && value == threshold;

        bool take_equal = is_equal && block < tie_block;
        if (block == tie_block) {
            const unsigned equal_mask =
                __ballot_sync(kFullMask, is_equal);
            if (lane == 0) {
                s_warp_count[warp] = __popc(equal_mask);
            }
            __syncthreads();
            if (tid == 0) {
                int prefix = 0;
                for (int source_warp = 0;
                     source_warp < kCarryWarps;
                     ++source_warp) {
                    s_warp_prefix[source_warp] = prefix;
                    prefix += s_warp_count[source_warp];
                }
                s_tile_total = prefix;
            }
            __syncthreads();
            const int equal_rank =
                s_tie_seen + s_warp_prefix[warp] +
                __popc(equal_mask & lane_mask);
            take_equal = is_equal && equal_rank < tie_take;
            __syncthreads();
            if (tid == 0) {
                s_tie_seen += s_tile_total;
            }
            __syncthreads();
        }

        const bool selected =
            eligible && (value > threshold || take_equal);
        const unsigned selected_mask =
            __ballot_sync(kFullMask, selected);
        if (lane == 0) {
            s_warp_count[warp] = __popc(selected_mask);
        }
        __syncthreads();
        if (tid == 0) {
            int prefix = 0;
            for (int source_warp = 0;
                 source_warp < kCarryWarps;
                 ++source_warp) {
                s_warp_prefix[source_warp] = prefix;
                prefix += s_warp_count[source_warp];
            }
            s_tile_total = prefix;
        }
        __syncthreads();
        const int local_rank =
            s_warp_prefix[warp] +
            __popc(selected_mask & lane_mask);
        if (selected) {
            out_idx[
                output_base + s_tile_output_base + local_rank] =
                static_cast<int64_t>(index);
        }
        if constexpr (FusePlanner) {
            if (block == 0 && valid && !selected) {
                // Every non-selected position in the physical HOT window is a
                // hole. Reuse the already-consumed vote prefix as its compact
                // endpoint list; hole_rank needs no extra ballot or atomic.
                const int position = tile + tid;
                const int hole_rank = position -
                    (s_tile_output_base + local_rank);
                reinterpret_cast<volatile int32_t*>(votes)[hole_rank] = index;
            }
        }
        __syncthreads();
        if (tid == 0) {
            s_tile_output_base += s_tile_total;
        }
        __syncthreads();
    }

    if constexpr (FusePlanner) {
        // Qualified small-sequence grids have at most 16 CTAs on B200, so all
        // CTAs are simultaneously resident. Each producer releases its own
        // endpoint stores before publishing a ticket; the final -1 is the
        // acquire/release point for the parallel pair pass.
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int old =
                atomicAdd(&planner_state[kCarryPlannerTicket], 1);
            if (old == gridDim.x - 1) {
                atomicExch(&planner_state[kCarryPlannerTicket], -1);
            } else {
                while (atomicAdd(
                           &planner_state[kCarryPlannerTicket], 0) != -1) {
                    __nanosleep(64);
                }
            }
        }
        __syncthreads();

        const int selected_inside =
            planner_state[kCarryPlannerSelectedInside];
        const int pair_count = planner_state[kCarryPlannerPairCount];
        for (int rank = block * kCarryThreads + tid;
             rank < pair_count;
             rank += gridDim.x * kCarryThreads) {
            const int a = static_cast<int>(
                reinterpret_cast<volatile int64_t*>(out_idx)[
                    selected_inside + rank]);
            const int b =
                reinterpret_cast<volatile int32_t*>(votes)[rank];
            if (a < window_start + kCarryPlannerWindow || a >= count ||
                b < window_start ||
                b >= window_start + kCarryPlannerWindow ||
                b >= target_length) {
                atomicOr(&planner_state[kCarryPlannerStatus], 1);
                asm volatile("trap;");
            } else {
                permutation[a] = b;
                permutation[b] = a;
            }
            votes[rank] = 0;
        }
    }
}

// DSA specialization of the GitHub FlashTopK boundary-bucket strategy.
//
// The generic selector above logically restricts the radix set to bucket
// `th`, but every radix pass still rereads all `n` candidates and filters
// them. Sparse refresh normally leaves:
//
//     count(bucket < th) < K <= count(bucket <= th).
//
// Make that saving physical: one tiled pass writes bucket<th directly to the
// final output and compacts bucket==th in-place at the front of the candidate
// buffer. The four radix passes then read only that compact boundary. A tile
// is loaded completely before any write, and the compacted prefix can never
// extend beyond the end of the processed tile, so aliasing input/output is
// race-free and needs no second multi-GiB candidate slab.
//
// The two fallback modes mirror compact_topk_min_thr_litetopk_kernel:
//   * threshold too loose (lt >= K): compact/radix the lt set;
//   * threshold underfilled: compact/radix every finite buffered candidate.
__device__ __forceinline__ void
dsa_litetopk_accumulate_inplace_votes(
    const int32_t* __restrict__ out_idx,
    int K,
    int tid,
    int threads,
    int32_t* __restrict__ votes,
    int votes_len,
    int row,
    int row_stride) {
    // Keep the same total vote volume as row subsampling, but cover every
    // query row.  For stride=8 and K=2048 each row contributes one rotating
    // 256-winner slice instead of only row%8==0 contributing all 2048.  This
    // removes a fixed phase blind spot without adding winner loads/atomics.
    if (votes == nullptr || votes_len <= 0) {
        return;
    }
    // Every call site is a block-uniform exit. Wait until all winner stores
    // are visible, then count this row's phase while they are still hot.
    __syncthreads();
    const int phase = row & (row_stride - 1);
    for (int j = tid + phase * threads;
         j < K;
         j += threads * row_stride) {
        int32_t col = out_idx[j];
        col = col < 0 ? 0 : (col >= votes_len ? votes_len - 1 : col);
        atomicAdd(votes + col, 1);
    }
}

// Late-map production epilogue.  Selection stays entirely in physical
// pair-swapped workspace space; this grid-wide kernel then maps only Q*K
// winners with enough independent warps to hide the random permutation-read
// latency.  Carry voting is folded into the same pass after each winner has
// reached original corpus space.
__global__ void map_topk_indices_and_accumulate_votes_litetopk_kernel(
        int32_t* __restrict__ out_idx,
        const int32_t* __restrict__ index_map,
        const int32_t* __restrict__ status,
        int32_t* __restrict__ votes,
        int64_t total,
        int rows,
        int index_map_size,
        int topk,
        int votes_len,
        int vote_recent_rows,
        const int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ stat_run_max,
        int32_t* __restrict__ stat_over,
        int stat_watermark) {
    const int64_t step =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t global_thread =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t vote_begin =
        static_cast<int64_t>(rows - vote_recent_rows) * topk;
    // A bad/underfilled row may contain selector padding rather than K valid
    // physical indices.  Preserve fail-closed behavior before treating the
    // complete output matrix as mappable winners.  The candidate-count
    // telemetry rides the same one-thread-per-row sweep (replaces a 5-launch
    // amax/maximum/gt/sum/add_ tail on the host path).
    int stat_local_max = 0;
    int stat_local_over = 0;
    for (int row = static_cast<int>(global_thread);
         row < rows;
         row += static_cast<int>(step)) {
        if (status[row] != 0) {
            asm volatile("trap;");
            return;
        }
        if (cand_cnt != nullptr) {
            const int c = cand_cnt[row];
            stat_local_max = c > stat_local_max ? c : stat_local_max;
            stat_local_over += c > stat_watermark ? 1 : 0;
        }
    }
    if (cand_cnt != nullptr) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const int m = __shfl_down_sync(0xffffffffu, stat_local_max, off);
            stat_local_max = m > stat_local_max ? m : stat_local_max;
            stat_local_over += __shfl_down_sync(0xffffffffu, stat_local_over, off);
        }
        __shared__ int stat_smax[32], stat_sover[32];
        const int wid = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) {
            stat_smax[wid] = stat_local_max;
            stat_sover[wid] = stat_local_over;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int m = 0, ov = 0;
            const int warps = (blockDim.x + 31) >> 5;
            for (int w = 0; w < warps; ++w) {
                m = stat_smax[w] > m ? stat_smax[w] : m;
                ov += stat_sover[w];
            }
            if (m > 0) atomicMax(stat_run_max, m);
            if (ov > 0) atomicAdd(stat_over, ov);
        }
    }
    for (int64_t linear =
             global_thread;
         linear < total;
         linear += step) {
        const int32_t physical_idx = out_idx[linear];
        if (static_cast<uint32_t>(physical_idx) >=
            static_cast<uint32_t>(index_map_size)) {
            asm volatile("trap;");
            return;
        }
        const int32_t original_idx = index_map[physical_idx];
        if (static_cast<uint32_t>(original_idx) >=
                static_cast<uint32_t>(index_map_size) ||
            static_cast<uint32_t>(original_idx) >
                dsa_litetopk::kCandidateIndexMask) {
            asm volatile("trap;");
            return;
        }
        out_idx[linear] = original_idx;

        if (votes != nullptr && votes_len > 0) {
            // The next chunk is best predicted by the most recent query
            // window.  Mapping already visits every Q*K winner, so voting all
            // winners from the last 1536 rows only adds the atomics; it needs
            // no extra winner read or launch.
            if (linear >= vote_begin) {
                const int32_t vote_idx = original_idx < 0
                    ? 0
                    : (original_idx >= votes_len
                        ? votes_len - 1
                        : original_idx);
                atomicAdd(votes + vote_idx, 1);
            }
        }
    }
}

template <bool kOnlineFixedPayload = false>
__global__ void compact_topk_min_thr_inplace_idx_out_litetopk_kernel(
    CandidateValue* __restrict__ val,
    int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt,
    const int32_t* __restrict__ th_in,
    const int32_t* __restrict__ boundary_meta,
    int R,
    int CAP,
    int K,
    int NB,
    int32_t* __restrict__ out_idx,
    int32_t* __restrict__ votes,
    int votes_len,
    int vote_row_stride) {
    constexpr int BT = 256;
    constexpr int RADIX = 256;
    const unsigned FULL = 0xffffffffu;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);
    if (row >= R) return;

    CandidateValue* vrow =
        val + static_cast<size_t>(row) * CAP;
    int32_t* irow = idx + static_cast<size_t>(row) * CAP;
    int32_t* oi = out_idx + static_cast<size_t>(row) * K;
    const int raw_n = cnt[row];
    int n = raw_n;
    if (n > CAP) n = CAP;
    if (n < 0) n = 0;
    if (n == 0) {
        for (int j = tid; j < K; j += BT) {
            oi[j] = 0;
        }
        dsa_litetopk_accumulate_inplace_votes(
            oi, K, tid, BT, votes, votes_len, row, vote_row_stride);
        return;
    }

    const int th = th_in[row];
    // The packed boundary remains bit-exact only above its compile-time
    // lower bound.  Fail loudly instead of silently turning the exact path
    // into an approximation.
    constexpr int kPackedExactThreshold = 0;
    if (th < kPackedExactThreshold) {
        asm volatile("trap;");
        return;
    }

    // mode 0: standard boundary path; 1: loose threshold; 2: underfilled.
    __shared__ int s_count_lt;
    __shared__ int s_count_eq;
    __shared__ int s_count_valid;
    __shared__ int s_have_boundary_meta;
    __shared__ int s_mode;
    __shared__ int s_k_target;
    constexpr int BOUNDARY_SMEM_CAP = 256;
    __shared__ uint32_t s_boundary_val[BOUNDARY_SMEM_CAP];
    __shared__ int32_t s_boundary_idx[BOUNDARY_SMEM_CAP];
    __shared__ int s_fast_lt_cursor;
    __shared__ int s_fast_eq_cursor;
    __shared__ uint32_t s_fast_hist[RADIX];
    __shared__ uint32_t s_fast_desired;
    __shared__ uint32_t s_fast_kfind;
    __shared__ int s_fast_pivot_lt;
    __shared__ int s_fast_write_lt;
    __shared__ int s_fast_write_eq;
    __shared__ int s_fast_certificate_matches;
    if (tid == 0) {
        const int32_t* meta =
            boundary_meta + static_cast<size_t>(row) * NB;
        const int tag = meta[0];
        const int meta_th = ~tag;
        const int meta_lt = meta[1];
        const int meta_eq = meta[2];
        const int meta_need = K - meta_lt;
        s_have_boundary_meta =
            tag < 0 && meta_th == th &&
            meta_th >= 0 && meta_th < NB &&
            raw_n >= 0 && raw_n <= CAP &&
            meta_lt >= 0 && meta_eq >= 0 &&
            meta_lt < K && meta_need > 0 &&
            meta_need <= meta_eq &&
            meta_lt + meta_eq <= n;
        s_count_lt = s_have_boundary_meta ? meta_lt : 0;
        s_count_eq = s_have_boundary_meta ? meta_eq : 0;
        s_count_valid = 0;
    }
    __syncthreads();

    // The six-byte representation is conditionally exact for the certified
    // sparse-refresh boundary path.  A missing certificate could require a
    // top-K selection within collapsed bucket 0, so it is an explicit error.
    if (!s_have_boundary_meta) {
        asm volatile("trap;");
        return;
    }

    if (!s_have_boundary_meta) {
        int local_lt = 0;
        int local_eq = 0;
        int local_valid = 0;
        for (int j = tid; j < n; j += BT) {
            float v = 0.0f;
            uint32_t score_code = 0u;
            if constexpr (kOnlineFixedPayload) {
                score_code =
                    dsa_litetopk::candidate_load_score_code(
                        vrow[j], irow[j]);
            } else {
                v = dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]);
                if (!isfinite(v)) continue;
            }
            ++local_valid;
            int b;
            if constexpr (kOnlineFixedPayload) {
                b = static_cast<int>(score_code >> 16);
            } else {
                const int braw = static_cast<int>(v);
                b = braw < 0 ? 0 :
                    (braw > NB - 1 ? NB - 1 : braw);
            }
            local_lt += b < th;
            local_eq += b == th;
        }
        atomicAdd(&s_count_lt, local_lt);
        atomicAdd(&s_count_eq, local_eq);
        atomicAdd(&s_count_valid, local_valid);
    }
    __syncthreads();
    if (tid == 0) {
        const int need = K - s_count_lt;
        if (s_count_lt < K &&
            need > 0 && need <= s_count_eq) {
            s_mode = 0;
            s_k_target = need;
        } else if (s_count_lt >= K) {
            s_mode = 1;
            s_k_target = K;
        } else {
            s_mode = 2;
            s_k_target = min(K, s_count_valid);
        }
    }
    __syncthreads();

    // Production sparse-refresh distribution (Q=8192, K=2048):
    // boundary E averages ~97 candidates, P99 ~163, max 212 on the 1M
    // corpus. Keep that boundary entirely in shared memory. Unlike the
    // generic in-place path below, this pass has no aliasing stores, so
    // warp-local shared-atomic reservations need no CTA barrier per tile.
    if (s_have_boundary_meta &&
        s_count_eq <= BOUNDARY_SMEM_CAP) {
        if (tid == 0) {
            s_fast_lt_cursor = 0;
            s_fast_eq_cursor = 0;
        }
        __syncthreads();
        for (int tile = 0; tile < n; tile += BT) {
            const int j = tile + tid;
            uint32_t score_code = 0u;
            bool valid = false;
            if (j < n) {
                score_code =
                    dsa_litetopk::candidate_load_score_code(
                        vrow[j], irow[j]);
                valid = true;
            }
            // The sign-aware FP32 high24 code is monotonic across negative and
            // positive bucket-space values. Truncating the low byte cannot
            // cross an exactly represented integer bucket edge.
            uint32_t th_code;
            uint32_t next_th_code;
            if constexpr (kOnlineFixedPayload) {
                th_code = static_cast<uint32_t>(th) << 16;
                next_th_code =
                    static_cast<uint32_t>(th + 1) << 16;
            } else {
                th_code =
                    dsa_litetopk::candidate_fp24_code(
                        static_cast<float>(th));
                next_th_code =
                    dsa_litetopk::candidate_fp24_code(
                        static_cast<float>(th + 1));
            }
            const bool is_lt =
                valid && th > 0 && score_code < th_code;
            const bool is_eq =
                valid && score_code < next_th_code &&
                (th == 0 || score_code >= th_code);
            const unsigned lt_mask =
                __ballot_sync(FULL, is_lt);
            const unsigned eq_mask =
                __ballot_sync(FULL, is_eq);
            int warp_lt_base = 0;
            int warp_eq_base = 0;
            if (lane == 0) {
                const int lt_count = __popc(lt_mask);
                const int eq_count = __popc(eq_mask);
                if (lt_count != 0)
                    warp_lt_base =
                        atomicAdd(&s_fast_lt_cursor, lt_count);
                if (eq_count != 0)
                    warp_eq_base =
                        atomicAdd(&s_fast_eq_cursor, eq_count);
            }
            warp_lt_base =
                __shfl_sync(FULL, warp_lt_base, 0);
            warp_eq_base =
                __shfl_sync(FULL, warp_eq_base, 0);

            if (is_lt) {
                const int pos =
                    warp_lt_base + __popc(lt_mask & lane_mask);
                if (pos < K) {
                    const int32_t raw_idx = irow[j];
                    oi[pos] =
                        dsa_litetopk::candidate_decode_index(raw_idx);
                }
            }
            if (is_eq) {
                const int pos =
                    warp_eq_base + __popc(eq_mask & lane_mask);
                if (pos < BOUNDARY_SMEM_CAP) {
                    s_boundary_val[pos] = score_code;
                    s_boundary_idx[pos] =
                        dsa_litetopk::candidate_decode_index(
                            irow[j]);
                }
            }
        }
        __syncthreads();

        const int boundary_n = s_fast_eq_cursor;
        const int output_base = s_fast_lt_cursor;
        const int k_target = K - output_base;
        if (tid == 0) {
            // The certificate was produced by the immediately preceding
            // finalizer, but do not let a stale/corrupt certificate turn the
            // fixed-size shared boundary into an out-of-bounds access.  The
            // cursors are an independent recount using the selector's exact
            // predicates.  On any disagreement, republish those actual
            // counts and fall through to the existing capacity-independent
            // in-place selector below.
            s_fast_certificate_matches =
                output_base == s_count_lt &&
                boundary_n == s_count_eq &&
                boundary_n >= 0 &&
                boundary_n <= BOUNDARY_SMEM_CAP;
            if (!s_fast_certificate_matches) {
                s_count_lt = output_base;
                s_count_eq = boundary_n;
                s_count_valid = n;
                const int actual_need = K - output_base;
                if (output_base < K && actual_need > 0 &&
                    actual_need <= boundary_n) {
                    s_mode = 0;
                    s_k_target = actual_need;
                } else if (output_base >= K) {
                    s_mode = 1;
                    s_k_target = K;
                } else {
                    s_mode = 2;
                    s_k_target = min(K, n);
                }
            }
        }
        __syncthreads();
        if (s_fast_certificate_matches) {
            if (boundary_n == k_target) {
                for (int j = tid;
                     j < boundary_n; j += BT) {
                    oi[output_base + j] = s_boundary_idx[j];
                }
                dsa_litetopk_accumulate_inplace_votes(
                    oi, K, tid, BT, votes, votes_len,
                    row, vote_row_stride);
                return;
            }

            if (tid == 0) {
                // For th>0, boundary values lie in [th, th+1), so their
                // sign-aware FP32 high byte is fixed. Bucket zero also owns
                // every negative value, so it needs the full three-byte key.
                s_fast_desired =
                    (!kOnlineFixedPayload && th == 0)
                        ? 0u
                        : (s_boundary_val[0] & 0xff0000u);
                s_fast_kfind =
                    static_cast<uint32_t>(k_target);
            }
            __syncthreads();
            uint32_t fast_mask = 0u;
            #pragma unroll
            for (int pass = 0; pass < 3; ++ pass) {
                const bool full_key =
                    !kOnlineFixedPayload && th == 0;
                const int num_passes = full_key ? 3 : 2;
                if (pass < num_passes) {
                    const int shift =
                        (full_key ? 16 : 8) - pass * 8;
                    s_fast_hist[tid] = 0;
                    __syncthreads();
                    const uint32_t desired = s_fast_desired;
                    if (tid < boundary_n) {
                        const uint32_t encoded =
                            s_boundary_val[tid];
                        if ((encoded & fast_mask) ==
                            (desired & fast_mask)) {
                            atomicAdd(
                                &s_fast_hist[
                                    (encoded >> shift) & 0xffu],
                                1u);
                        }
                    }
                    __syncthreads();
                    compact_find_radix_digit_warp0(
                        s_fast_hist,
                        &s_fast_desired,
                        &s_fast_kfind,
                        desired,
                        shift,
                        tid);
                    __syncthreads();
                    fast_mask |= 0xffu << shift;
                }
            }
            const uint32_t pivot = s_fast_desired;

            if (tid == 0) {
                s_fast_pivot_lt = 0;
                s_fast_write_lt = 0;
                s_fast_write_eq = 0;
            }
            __syncthreads();
            if (tid < boundary_n && s_boundary_val[tid] < pivot)
                atomicAdd(&s_fast_pivot_lt, 1);
            __syncthreads();
            const int eq_take =
                max(k_target - s_fast_pivot_lt, 0);
            if (tid < boundary_n) {
                const uint32_t encoded =
                    s_boundary_val[tid];
                if (encoded < pivot) {
                    const int pos = atomicAdd(
                        &s_fast_write_lt, 1);
                    if (pos < k_target) {
                        oi[output_base + pos] = s_boundary_idx[tid];
                    }
                } else if (encoded == pivot) {
                    const int equal_rank = atomicAdd(
                        &s_fast_write_eq, 1);
                    if (equal_rank < eq_take) {
                        const int pos =
                            output_base +
                            s_fast_pivot_lt +
                            equal_rank;
                        if (pos < K) {
                            oi[pos] = s_boundary_idx[tid];
                        }
                    }
                }
            }
            dsa_litetopk_accumulate_inplace_votes(
                oi, K, tid, BT, votes, votes_len,
                row, vote_row_stride);
            return;
        }
    }

    // Tiled, alias-safe in-place compaction. In the standard mode, lt
    // candidates bypass the compact buffer and go straight to output.
    __shared__ int s_compact_base;
    __shared__ int s_direct_base;
    if (tid == 0) {
        s_compact_base = 0;
        s_direct_base = 0;
    }
    __syncthreads();

    for (int tile = 0; tile < n; tile += BT) {
        const int j = tile + tid;
        CandidateValue raw_value{};
        float v = INFINITY;
        int32_t raw_idx = 0;
        int b = NB;
        bool valid = false;
        if (j < n) {
            raw_value = vrow[j];
            raw_idx = irow[j];
            if constexpr (kOnlineFixedPayload) {
                const uint32_t score_code =
                    dsa_litetopk::candidate_load_score_code(
                        raw_value, raw_idx);
                b = static_cast<int>(score_code >> 16);
                valid = true;
            } else {
                v = dsa_litetopk::candidate_decode_score(
                    raw_value, raw_idx);
                valid = isfinite(v);
                if (valid) {
                    int braw = static_cast<int>(v);
                    b = braw < 0 ? 0 :
                        (braw > NB - 1 ? NB - 1 : braw);
                }
            }
        }

        const bool is_lt = valid && b < th;
        bool selected = false;
        if (s_mode == 0)
            selected = valid && b == th;
        else if (s_mode == 1)
            selected = is_lt;
        else
            selected = valid;
        const bool direct = s_mode == 0 && is_lt;

        const unsigned selected_mask =
            __ballot_sync(FULL, selected);
        const unsigned direct_mask =
            __ballot_sync(FULL, direct);
        int warp_compact_base = 0;
        int warp_direct_base = 0;
        if (lane == 0) {
            const int selected_count = __popc(selected_mask);
            const int direct_count = __popc(direct_mask);
            if (selected_count != 0)
                warp_compact_base =
                    atomicAdd(&s_compact_base, selected_count);
            if (direct_count != 0)
                warp_direct_base =
                    atomicAdd(&s_direct_base, direct_count);
        }
        warp_compact_base =
            __shfl_sync(FULL, warp_compact_base, 0);
        warp_direct_base =
            __shfl_sync(FULL, warp_direct_base, 0);

        // One CTA barrier per tile is sufficient for alias safety: every
        // source element is already in a register and every warp has reserved
        // its compact ranges before any in-place store starts. Compact output
        // never reaches the next (unread) tile.
        __syncthreads();

        if (direct) {
            const int pos =
                warp_direct_base +
                __popc(direct_mask & lane_mask);
            if (pos < K) {
                oi[pos] =
                    dsa_litetopk::candidate_decode_index(raw_idx);
            }
        }
        if (selected) {
            const int pos =
                warp_compact_base +
                __popc(selected_mask & lane_mask);
            vrow[pos] = raw_value;
            irow[pos] = raw_idx;
        }
    }
    __syncthreads();

    const int selected_n = s_compact_base;
    const int output_base = s_mode == 0 ? s_count_lt : 0;
    const int k_target = s_k_target;

    // Exact fallback with fewer than K finite buffered candidates.
    if (s_mode == 2 && selected_n <= K) {
        for (int j = tid; j < selected_n; j += BT) {
            oi[j] =
                dsa_litetopk::candidate_decode_index(irow[j]);
        }
        for (int j = selected_n + tid; j < K; j += BT) {
            oi[j] = 0;
        }
        dsa_litetopk_accumulate_inplace_votes(
            oi, K, tid, BT, votes, votes_len, row, vote_row_stride);
        return;
    }
    if (selected_n == 0 || k_target == 0) {
        for (int j = output_base + tid; j < K; j += BT) {
            oi[j] = 0;
        }
        dsa_litetopk_accumulate_inplace_votes(
            oi, K, tid, BT, votes, votes_len, row, vote_row_stride);
        return;
    }

    // Radix-select only the compacted set. In the expected sparse-refresh
    // case this is exactly the threshold bucket and k_target == K-count_lt.
    __shared__ uint32_t hist[RADIX];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int s_pivot_lt;
    __shared__ int s_write_lt;
    __shared__ int s_write_eq;
    if (tid == 0) {
        desired = 0u;
        kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();

    uint32_t mask = 0u;
    constexpr int kRadixPasses =
        kOnlineFixedPayload ? 3 : 4;
    constexpr int kFirstRadixShift =
        kOnlineFixedPayload ? 16 : 24;
    #pragma unroll
    for (int pass = 0; pass < kRadixPasses; ++ pass) {
        const int shift = kFirstRadixShift - pass * 8;
        hist[tid] = 0;
        __syncthreads();
        const uint32_t d = desired;
        for (int j = tid; j < selected_n; j += BT) {
            uint32_t e;
            if constexpr (kOnlineFixedPayload) {
                e = dsa_litetopk::candidate_load_score_code(
                    vrow[j], irow[j]);
            } else {
                e = compact_enc_float(
                    dsa_litetopk::candidate_decode_score(
                        vrow[j], irow[j]));
            }
            if ((e & mask) == (d & mask))
                atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        compact_find_radix_digit_warp0(
            hist, &desired, &kfind, d, shift, tid);
        __syncthreads();
        mask |= 0xffu << shift;
    }
    const uint32_t pivot = desired;

    if (tid == 0) {
        s_pivot_lt = 0;
        s_write_lt = 0;
        s_write_eq = 0;
    }
    __syncthreads();
    int pivot_lt = 0;
    for (int j = tid; j < selected_n; j += BT) {
        uint32_t e;
        if constexpr (kOnlineFixedPayload) {
            e = dsa_litetopk::candidate_load_score_code(
                vrow[j], irow[j]);
        } else {
            e = compact_enc_float(
                dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]));
        }
        pivot_lt += e < pivot;
    }
    atomicAdd(&s_pivot_lt, pivot_lt);
    __syncthreads();
    const int eq_take = max(k_target - s_pivot_lt, 0);

    for (int j = tid; j < selected_n; j += BT) {
        uint32_t e;
        if constexpr (kOnlineFixedPayload) {
            e = dsa_litetopk::candidate_load_score_code(
                vrow[j], irow[j]);
        } else {
            const float v =
                dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]);
            e = compact_enc_float(v);
        }
        if (e < pivot) {
            const int w = atomicAdd(&s_write_lt, 1);
            const int pos = output_base + w;
            if (pos < K) {
                oi[pos] =
                    dsa_litetopk::candidate_decode_index(irow[j]);
            }
        } else if (e == pivot) {
            const int equal_rank = atomicAdd(&s_write_eq, 1);
            if (equal_rank < eq_take) {
                const int pos =
                    output_base + s_pivot_lt + equal_rank;
                if (pos < K) {
                    oi[pos] =
                        dsa_litetopk::candidate_decode_index(irow[j]);
                }
            }
        }
    }
    dsa_litetopk_accumulate_inplace_votes(
        oi, K, tid, BT, votes, votes_len, row, vote_row_stride);
}

__device__ __forceinline__ void
compact_topk_paged_fast_scan_span(
        const CandidateValue* __restrict__ val,
        const int32_t* __restrict__ idx,
        const int count,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        int32_t* __restrict__ out_idx,
        uint32_t* __restrict__ boundary_val,
        int32_t* __restrict__ boundary_idx,
        int* __restrict__ lt_cursor,
        int* __restrict__ eq_cursor,
        const int topk) {
    constexpr int kThreads = 256;
    constexpr int kBoundaryCapacity = 256;
    constexpr unsigned kFullMask = 0xffffffffu;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);

    for (int tile = 0; tile < count; tile += kThreads) {
        const int j = tile + tid;
        uint32_t score_code = 0u;
        int32_t packed_idx = 0;
        const bool valid = j < count;
        if (valid) {
            packed_idx = idx[j];
            score_code = dsa_litetopk::candidate_load_score_code(
                val[j], packed_idx);
        }
        const bool is_lt =
            valid && th > 0 && score_code < th_code;
        const bool is_eq =
            valid && score_code < next_th_code &&
            (th == 0 || score_code >= th_code);
        const unsigned lt_mask =
            __ballot_sync(kFullMask, is_lt);
        const unsigned eq_mask =
            __ballot_sync(kFullMask, is_eq);
        int warp_lt_base = 0;
        int warp_eq_base = 0;
        if (lane == 0) {
            const int lt_count = __popc(lt_mask);
            const int eq_count = __popc(eq_mask);
            if (lt_count != 0) {
                warp_lt_base = atomicAdd(lt_cursor, lt_count);
            }
            if (eq_count != 0) {
                warp_eq_base = atomicAdd(eq_cursor, eq_count);
            }
        }
        warp_lt_base =
            __shfl_sync(kFullMask, warp_lt_base, 0);
        warp_eq_base =
            __shfl_sync(kFullMask, warp_eq_base, 0);

        if (is_lt) {
            const int pos =
                warp_lt_base + __popc(lt_mask & lane_mask);
            if (pos < topk) {
                out_idx[pos] =
                    dsa_litetopk::candidate_decode_index(packed_idx);
            }
        }
        if (is_eq) {
            const int pos =
                warp_eq_base + __popc(eq_mask & lane_mask);
            if (pos < kBoundaryCapacity) {
                boundary_val[pos] = score_code;
                boundary_idx[pos] =
                    dsa_litetopk::candidate_decode_index(packed_idx);
            }
        }
    }
}

__device__ __forceinline__ void
compact_topk_segment_fast_scan_warp(
        const CandidateValue* __restrict__ val,
        const int32_t* __restrict__ idx,
        const uint32_t count,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        int32_t* __restrict__ out_idx,
        uint32_t* __restrict__ boundary_val,
        int32_t* __restrict__ boundary_idx,
        int* __restrict__ lt_cursor,
        int* __restrict__ eq_cursor,
        const int topk) {
    constexpr int kBoundaryCapacity = 256;
    constexpr unsigned kFullMask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);
    for (uint32_t tile = 0; tile < count; tile += 32u) {
        const uint32_t j = tile + static_cast<uint32_t>(lane);
        uint32_t score_code = 0u;
        int32_t packed_idx = 0;
        const bool valid = j < count;
        if (valid) {
            packed_idx = idx[j];
            score_code = dsa_litetopk::candidate_load_score_code(
                val[j], packed_idx);
        }
        const bool is_lt =
            valid && th > 0 && score_code < th_code;
        const bool is_eq =
            valid && score_code < next_th_code &&
            (th == 0 || score_code >= th_code);
        const unsigned lt_mask =
            __ballot_sync(kFullMask, is_lt);
        const unsigned eq_mask =
            __ballot_sync(kFullMask, is_eq);
        int warp_lt_base = 0;
        int warp_eq_base = 0;
        if (lane == 0) {
            const int lt_count = __popc(lt_mask);
            const int eq_count = __popc(eq_mask);
            if (lt_count != 0) {
                warp_lt_base = atomicAdd(lt_cursor, lt_count);
            }
            if (eq_count != 0) {
                warp_eq_base = atomicAdd(eq_cursor, eq_count);
            }
        }
        warp_lt_base =
            __shfl_sync(kFullMask, warp_lt_base, 0);
        warp_eq_base =
            __shfl_sync(kFullMask, warp_eq_base, 0);
        if (is_lt) {
            const int pos =
                warp_lt_base + __popc(lt_mask & lane_mask);
            if (pos < topk) {
                out_idx[pos] =
                    dsa_litetopk::candidate_decode_index(packed_idx);
            }
        }
        if (is_eq) {
            const int pos =
                warp_eq_base + __popc(eq_mask & lane_mask);
            if (pos < kBoundaryCapacity) {
                boundary_val[pos] = score_code;
                boundary_idx[pos] =
                    dsa_litetopk::candidate_decode_index(packed_idx);
            }
        }
    }
}

// Cold-path primitives for a certified boundary that does not fit in shared
// memory.  They never compact or mutate candidate storage: every radix pass
// simply rereads the same coalesced inline/page spans.  This costs extra reads
// only for an exceptional row and, unlike a fixed scratch buffer, has no
// boundary-cardinality limit.
__device__ __forceinline__ bool
compact_topk_paged_is_boundary(
        const uint32_t score_code,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code) {
    return score_code < next_th_code &&
        (th == 0 || score_code >= th_code);
}

__device__ __forceinline__ void
compact_topk_paged_radix_hist_span(
        const CandidateValue* __restrict__ val,
        const int32_t* __restrict__ idx,
        const int count,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        const uint32_t prefix_mask,
        const uint32_t desired_prefix,
        const int shift,
        uint32_t* __restrict__ hist) {
    constexpr int kThreads = 256;
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = tid; j < count; j += kThreads) {
        const int32_t packed_idx = idx[j];
        const uint32_t score_code =
            dsa_litetopk::candidate_load_score_code(
                val[j], packed_idx);
        if (compact_topk_paged_is_boundary(
                score_code, th, th_code, next_th_code) &&
            (score_code & prefix_mask) ==
                (desired_prefix & prefix_mask)) {
            atomicAdd(&hist[(score_code >> shift) & 0xffu], 1u);
        }
    }
}

__device__ __forceinline__ void
compact_topk_paged_count_pivot_lt_span(
        const CandidateValue* __restrict__ val,
        const int32_t* __restrict__ idx,
        const int count,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        const uint32_t pivot,
        int* __restrict__ pivot_lt) {
    constexpr int kThreads = 256;
    const int tid = static_cast<int>(threadIdx.x);
    int local_lt = 0;
    for (int j = tid; j < count; j += kThreads) {
        const int32_t packed_idx = idx[j];
        const uint32_t score_code =
            dsa_litetopk::candidate_load_score_code(
                val[j], packed_idx);
        local_lt +=
            compact_topk_paged_is_boundary(
                score_code, th, th_code, next_th_code) &&
            score_code < pivot;
    }
    if (local_lt != 0) atomicAdd(pivot_lt, local_lt);
}

__device__ __forceinline__ void
compact_topk_paged_write_pivot_span(
        const CandidateValue* __restrict__ val,
        const int32_t* __restrict__ idx,
        const int count,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        const uint32_t pivot,
        const int output_base,
        const int pivot_lt,
        const int eq_take,
        const int topk,
        int32_t* __restrict__ out_idx,
        int* __restrict__ write_lt,
        int* __restrict__ write_eq) {
    constexpr int kThreads = 256;
    constexpr unsigned kFullMask = 0xffffffffu;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);

    for (int tile = 0; tile < count; tile += kThreads) {
        const int j = tile + tid;
        uint32_t score_code = 0u;
        int32_t packed_idx = 0;
        const bool valid = j < count;
        if (valid) {
            packed_idx = idx[j];
            score_code = dsa_litetopk::candidate_load_score_code(
                val[j], packed_idx);
        }
        const bool is_boundary =
            valid && compact_topk_paged_is_boundary(
                score_code, th, th_code, next_th_code);
        const bool is_lt = is_boundary && score_code < pivot;
        const bool is_eq = is_boundary && score_code == pivot;
        const unsigned lt_mask =
            __ballot_sync(kFullMask, is_lt);
        const unsigned eq_mask =
            __ballot_sync(kFullMask, is_eq);
        int warp_lt_base = 0;
        int warp_eq_base = 0;
        if (lane == 0) {
            const int lt_count = __popc(lt_mask);
            const int eq_count = __popc(eq_mask);
            if (lt_count != 0) {
                warp_lt_base = atomicAdd(write_lt, lt_count);
            }
            if (eq_count != 0) {
                warp_eq_base = atomicAdd(write_eq, eq_count);
            }
        }
        warp_lt_base =
            __shfl_sync(kFullMask, warp_lt_base, 0);
        warp_eq_base =
            __shfl_sync(kFullMask, warp_eq_base, 0);

        if (is_lt) {
            const int rank =
                warp_lt_base + __popc(lt_mask & lane_mask);
            const int pos = output_base + rank;
            if (rank < pivot_lt && pos < topk) {
                out_idx[pos] =
                    dsa_litetopk::candidate_decode_index(packed_idx);
            }
        }
        if (is_eq) {
            const int rank =
                warp_eq_base + __popc(eq_mask & lane_mask);
            if (rank < eq_take) {
                const int pos = output_base + pivot_lt + rank;
                if (pos < topk) {
                    out_idx[pos] =
                        dsa_litetopk::candidate_decode_index(packed_idx);
                }
            }
        }
    }
}

static __device__ __noinline__ void
compact_topk_paged_large_boundary_select(
        const CandidateValue* __restrict__ inline_val,
        const int32_t* __restrict__ inline_idx,
        const CandidateValue* __restrict__ overflow_val,
        const int32_t* __restrict__ overflow_idx,
        const int32_t* __restrict__ page_table,
        const int row,
        const int raw_n,
        const int overflow_pages,
        const int table_pages_per_row,
        const int th,
        const uint32_t th_code,
        const uint32_t next_th_code,
        const int output_base,
        const int k_target,
        const int topk,
        int32_t* __restrict__ row_out,
        int* __restrict__ page_id,
        uint32_t* __restrict__ hist,
        uint32_t* __restrict__ desired,
        uint32_t* __restrict__ kfind,
        int* __restrict__ pivot_lt,
        int* __restrict__ write_lt,
        int* __restrict__ write_eq) {
    constexpr int kThreads = 256;
    const int tid = static_cast<int>(threadIdx.x);
    const int inline_n = min(raw_n, kPagedCandidateInline);
    const uint64_t inline_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    const int overflow_n =
        max(raw_n - kPagedCandidateInline, 0);
    const int num_pages =
        (overflow_n + kPagedCandidatePage - 1) >>
        kPagedCandidatePageShift;

    if (tid == 0) {
        *desired = th == 0
            ? 0u
            : dsa_litetopk::candidate_fp24_code(
                  static_cast<float>(th)) & 0xff0000u;
        *kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();

    uint32_t prefix_mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const bool full_key = th == 0;
        const int num_passes = full_key ? 3 : 2;
        if (pass >= num_passes) continue;
        const int shift =
            (full_key ? 16 : 8) - pass * 8;
        hist[tid] = 0u;
        __syncthreads();
        const uint32_t desired_base = *desired;
        compact_topk_paged_radix_hist_span(
            inline_val + inline_base,
            inline_idx + inline_base,
            inline_n,
            th,
            th_code,
            next_th_code,
            prefix_mask,
            desired_base,
            shift,
            hist);
        __syncthreads();

        for (int logical_page = 0;
             logical_page < num_pages;
             ++logical_page) {
            if (tid == 0) {
                *page_id =
                    logical_page < table_pages_per_row
                        ? page_table[
                              static_cast<uint64_t>(row) *
                                  table_pages_per_row +
                              logical_page]
                        : -1;
            }
            __syncthreads();
            if (*page_id < 0 || *page_id >= overflow_pages) {
                asm volatile("trap;");
                return;
            }
            const int page_n = min(
                overflow_n - logical_page * kPagedCandidatePage,
                kPagedCandidatePage);
            const uint64_t page_base =
                static_cast<uint64_t>(*page_id) *
                kPagedCandidatePage;
            compact_topk_paged_radix_hist_span(
                overflow_val + page_base,
                overflow_idx + page_base,
                page_n,
                th,
                th_code,
                next_th_code,
                prefix_mask,
                desired_base,
                shift,
                hist);
            __syncthreads();
        }

        compact_find_radix_digit_warp0(
            hist, desired, kfind,
            desired_base, shift, tid);
        __syncthreads();
        prefix_mask |= 0xffu << shift;
    }
    const uint32_t pivot = *desired;

    if (tid == 0) *pivot_lt = 0;
    __syncthreads();
    compact_topk_paged_count_pivot_lt_span(
        inline_val + inline_base,
        inline_idx + inline_base,
        inline_n,
        th,
        th_code,
        next_th_code,
        pivot,
        pivot_lt);
    __syncthreads();
    for (int logical_page = 0;
         logical_page < num_pages;
         ++logical_page) {
        if (tid == 0) {
            *page_id =
                logical_page < table_pages_per_row
                    ? page_table[
                          static_cast<uint64_t>(row) *
                              table_pages_per_row +
                          logical_page]
                    : -1;
        }
        __syncthreads();
        if (*page_id < 0 || *page_id >= overflow_pages) {
            asm volatile("trap;");
            return;
        }
        const int page_n = min(
            overflow_n - logical_page * kPagedCandidatePage,
            kPagedCandidatePage);
        const uint64_t page_base =
            static_cast<uint64_t>(*page_id) *
            kPagedCandidatePage;
        compact_topk_paged_count_pivot_lt_span(
            overflow_val + page_base,
            overflow_idx + page_base,
            page_n,
            th,
            th_code,
            next_th_code,
            pivot,
            pivot_lt);
        __syncthreads();
    }

    const int exact_pivot_lt = *pivot_lt;
    const int eq_take = max(k_target - exact_pivot_lt, 0);
    if (tid == 0) {
        *write_lt = 0;
        *write_eq = 0;
    }
    __syncthreads();
    compact_topk_paged_write_pivot_span(
        inline_val + inline_base,
        inline_idx + inline_base,
        inline_n,
        th,
        th_code,
        next_th_code,
        pivot,
        output_base,
        exact_pivot_lt,
        eq_take,
        topk,
        row_out,
        write_lt,
        write_eq);
    __syncthreads();
    for (int logical_page = 0;
         logical_page < num_pages;
         ++logical_page) {
        if (tid == 0) {
            *page_id =
                logical_page < table_pages_per_row
                    ? page_table[
                          static_cast<uint64_t>(row) *
                              table_pages_per_row +
                          logical_page]
                    : -1;
        }
        __syncthreads();
        if (*page_id < 0 || *page_id >= overflow_pages) {
            asm volatile("trap;");
            return;
        }
        const int page_n = min(
            overflow_n - logical_page * kPagedCandidatePage,
            kPagedCandidatePage);
        const uint64_t page_base =
            static_cast<uint64_t>(*page_id) *
            kPagedCandidatePage;
        compact_topk_paged_write_pivot_span(
            overflow_val + page_base,
            overflow_idx + page_base,
            page_n,
            th,
            th_code,
            next_th_code,
            pivot,
            output_base,
            exact_pivot_lt,
            eq_take,
            topk,
            row_out,
            write_lt,
            write_eq);
        __syncthreads();
    }
    if (*write_lt != exact_pivot_lt || *write_eq < eq_take) {
        asm volatile("trap;");
        return;
    }
    __syncthreads();
}

// Paged production selector.  The usual boundary<=256 case stays entirely in
// shared memory.  A larger certified boundary is a valid distribution, not an
// error: the cold path radix-selects it in place by rereading the logical
// inline/page stream, without requiring a second candidate arena.  Any
// malformed count, page entry, or stale certificate still traps before the
// corresponding page is dereferenced.
__global__ void compact_topk_min_thr_paged_idx_out_litetopk_kernel(
        const CandidateValue* __restrict__ inline_val,
        const int32_t* __restrict__ inline_idx,
        const CandidateValue* __restrict__ overflow_val,
        const int32_t* __restrict__ overflow_idx,
        const int32_t* __restrict__ page_table,
        const int32_t* __restrict__ cand_cnt,
        const int32_t* __restrict__ th_in,
        const int32_t* __restrict__ boundary_meta,
        int rows,
        int overflow_pages,
        int table_pages_per_row,
        int logical_cap,
        int num_buckets,
        int topk,
        int32_t* __restrict__ out_idx,
        int32_t* __restrict__ votes,
        int votes_len,
        int vote_row_stride) {
    constexpr int kThreads = 256;
    constexpr int kRadix = 256;
    constexpr int kBoundaryCapacity = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= rows) return;

    __shared__ int count_lt;
    __shared__ int count_eq;
    __shared__ int certificate_valid;
    __shared__ int page_id;
    __shared__ int lt_cursor;
    __shared__ int eq_cursor;
    __shared__ uint32_t boundary_val[kBoundaryCapacity];
    __shared__ int32_t boundary_idx[kBoundaryCapacity];
    __shared__ uint32_t hist[kRadix];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int pivot_lt;
    __shared__ int write_lt;
    __shared__ int write_eq;

    const int raw_n = cand_cnt[row];
    const int th = th_in[row];
    if (tid == 0) {
        const int32_t* meta =
            boundary_meta +
            static_cast<uint64_t>(row) * num_buckets;
        const int tag = meta[0];
        const int meta_th = ~tag;
        const int meta_lt = meta[1];
        const int meta_eq = meta[2];
        const int need = topk - meta_lt;
        certificate_valid =
            tag < 0 && meta_th == th &&
            th >= 0 && th < num_buckets &&
            raw_n >= topk && raw_n <= logical_cap &&
            meta_lt >= 0 && meta_lt < topk &&
            meta_eq > 0 &&
            need > 0 && need <= meta_eq &&
            meta_lt + meta_eq <= raw_n;
        count_lt = meta_lt;
        count_eq = meta_eq;
        lt_cursor = 0;
        eq_cursor = 0;
    }
    __syncthreads();
    if (!certificate_valid) {
        asm volatile("trap;");
        return;
    }

    int32_t* const row_out =
        out_idx + static_cast<uint64_t>(row) * topk;
    const uint32_t th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th));
    const uint32_t next_th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th + 1));

    const int inline_n = min(raw_n, kPagedCandidateInline);
    const uint64_t inline_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    compact_topk_paged_fast_scan_span(
        inline_val + inline_base,
        inline_idx + inline_base,
        inline_n,
        th,
        th_code,
        next_th_code,
        row_out,
        boundary_val,
        boundary_idx,
        &lt_cursor,
        &eq_cursor,
        topk);
    __syncthreads();

    const int overflow_n =
        max(raw_n - kPagedCandidateInline, 0);
    const int num_pages =
        (overflow_n + kPagedCandidatePage - 1) >>
        kPagedCandidatePageShift;
    for (int logical_page = 0;
         logical_page < num_pages;
         ++logical_page) {
        if (tid == 0) {
            page_id =
                logical_page < table_pages_per_row
                    ? page_table[
                          static_cast<uint64_t>(row) *
                              table_pages_per_row +
                          logical_page]
                    : -1;
        }
        __syncthreads();
        if (page_id < 0 || page_id >= overflow_pages) {
            asm volatile("trap;");
            return;
        }

        const int page_n = min(
            overflow_n - logical_page * kPagedCandidatePage,
            kPagedCandidatePage);
        const uint64_t page_base =
            static_cast<uint64_t>(page_id) * kPagedCandidatePage;
        compact_topk_paged_fast_scan_span(
            overflow_val + page_base,
            overflow_idx + page_base,
            page_n,
            th,
            th_code,
            next_th_code,
            row_out,
            boundary_val,
            boundary_idx,
            &lt_cursor,
            &eq_cursor,
            topk);
        // The next page-table entry may not replace page_id until every warp
        // has finished dereferencing this page.
        __syncthreads();
    }

    if (tid == 0) {
        certificate_valid =
            lt_cursor == count_lt &&
            eq_cursor == count_eq;
    }
    __syncthreads();
    if (!certificate_valid) {
        asm volatile("trap;");
        return;
    }

    const int output_base = count_lt;
    const int boundary_n = count_eq;
    const int k_target = topk - output_base;
    if (boundary_n <= kBoundaryCapacity) {
      if (boundary_n == k_target) {
        for (int j = tid; j < boundary_n; j += kThreads) {
            row_out[output_base + j] = boundary_idx[j];
        }
        dsa_litetopk_accumulate_inplace_votes(
            row_out, topk, tid, kThreads, votes, votes_len,
            row, vote_row_stride);
        return;
    }

    if (tid == 0) {
        // For th>0, [th,th+1) fixes the high score-code byte, so two radix
        // bytes remain. Bucket zero includes negative values and needs all
        // three bytes, matching the existing production fast selector.
        desired = th == 0
            ? 0u
            : (boundary_val[0] & 0xff0000u);
        kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();

    uint32_t mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const bool full_key = th == 0;
        const int num_passes = full_key ? 3 : 2;
        if (pass < num_passes) {
            const int shift =
                (full_key ? 16 : 8) - pass * 8;
            hist[tid] = 0u;
            __syncthreads();
            const uint32_t desired_base = desired;
            if (tid < boundary_n) {
                const uint32_t encoded = boundary_val[tid];
                if ((encoded & mask) ==
                    (desired_base & mask)) {
                    atomicAdd(
                        &hist[(encoded >> shift) & 0xffu],
                        1u);
                }
            }
            __syncthreads();
            compact_find_radix_digit_warp0(
                hist, &desired, &kfind,
                desired_base, shift, tid);
            __syncthreads();
            mask |= 0xffu << shift;
        }
    }
    const uint32_t pivot = desired;

    if (tid == 0) {
        pivot_lt = 0;
        write_lt = 0;
        write_eq = 0;
    }
    __syncthreads();
    if (tid < boundary_n && boundary_val[tid] < pivot) {
        atomicAdd(&pivot_lt, 1);
    }
    __syncthreads();
    const int eq_take = max(k_target - pivot_lt, 0);
    if (tid < boundary_n) {
        const uint32_t encoded = boundary_val[tid];
        if (encoded < pivot) {
            const int pos = atomicAdd(&write_lt, 1);
            if (pos < k_target) {
                row_out[output_base + pos] = boundary_idx[tid];
            }
        } else if (encoded == pivot) {
            const int equal_rank = atomicAdd(&write_eq, 1);
            if (equal_rank < eq_take) {
                const int pos =
                    output_base + pivot_lt + equal_rank;
                if (pos < topk) {
                    row_out[pos] = boundary_idx[tid];
                }
            }
        }
    }
    dsa_litetopk_accumulate_inplace_votes(
        row_out, topk, tid, kThreads, votes, votes_len,
        row, vote_row_stride);
    return;
    }

    // The normal kernel has independently recounted the certificate and
    // already emitted the <threshold prefix.  A small persistent continuation
    // grid handles this exceptional row after the fast grid completes; keeping
    // that cold radix code out of this kernel preserves normal-path occupancy.
    return;
}

// Persistent cold continuation for rows whose exact coarse boundary exceeds
// the 256-record shared fast buffer.  The preceding normal selector has
// already validated/recounted the certificate and written every <threshold
// index.  Limiting this grid to roughly one CTA per SM makes the all-fast case
// only an empty metadata walk instead of inflating the hot kernel's register
// footprint or launching Q extra cold CTAs.
__global__ void
compact_topk_min_thr_paged_large_boundary_litetopk_kernel(
        const CandidateValue* __restrict__ inline_val,
        const int32_t* __restrict__ inline_idx,
        const CandidateValue* __restrict__ overflow_val,
        const int32_t* __restrict__ overflow_idx,
        const int32_t* __restrict__ page_table,
        const int32_t* __restrict__ cand_cnt,
        const int32_t* __restrict__ th_in,
        const int32_t* __restrict__ boundary_meta,
        int rows,
        int overflow_pages,
        int table_pages_per_row,
        int logical_cap,
        int num_buckets,
        int topk,
        int32_t* __restrict__ out_idx,
        int32_t* __restrict__ votes,
        int votes_len,
        int vote_row_stride) {
    constexpr int kThreads = 256;
    constexpr int kBoundaryCapacity = 256;
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ int task_count;
    __shared__ int task_rows[kThreads];
    __shared__ int selected_raw_n;
    __shared__ int selected_th;
    __shared__ int selected_output_base;
    __shared__ int selected_k_target;
    __shared__ int page_id;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int pivot_lt;
    __shared__ int write_lt;
    __shared__ int write_eq;

    const int tile_stride =
        static_cast<int>(gridDim.x) * kThreads;
    for (int tile_base =
             static_cast<int>(blockIdx.x) * kThreads;
         tile_base < rows;
         tile_base += tile_stride) {
        if (tid == 0) task_count = 0;
        __syncthreads();
        const int candidate_row = tile_base + tid;
        bool is_large = false;
        if (candidate_row < rows) {
            const int raw_n = cand_cnt[candidate_row];
            const int th = th_in[candidate_row];
            const int32_t* meta =
                boundary_meta +
                static_cast<uint64_t>(candidate_row) * num_buckets;
            const int tag = meta[0];
            const int meta_th = ~tag;
            const int meta_lt = meta[1];
            const int meta_eq = meta[2];
            const int need = topk - meta_lt;
            is_large =
                tag < 0 && meta_th == th &&
                th >= 0 && th < num_buckets &&
                raw_n >= topk && raw_n <= logical_cap &&
                meta_lt >= 0 && meta_lt < topk &&
                meta_eq > kBoundaryCapacity &&
                need > 0 && need <= meta_eq &&
                meta_lt + meta_eq <= raw_n;
        }
        if (is_large) {
            const int slot = atomicAdd(&task_count, 1);
            task_rows[slot] = candidate_row;
        }
        __syncthreads();

        const int num_tasks = task_count;
        for (int task = 0; task < num_tasks; ++task) {
          const int row = task_rows[task];
          if (tid == 0) {
            const int raw_n = cand_cnt[row];
            const int th = th_in[row];
            const int32_t* meta =
                boundary_meta +
                static_cast<uint64_t>(row) * num_buckets;
            const int meta_lt = meta[1];
            const int need = topk - meta_lt;
            selected_raw_n = raw_n;
            selected_th = th;
            selected_output_base = meta_lt;
            selected_k_target = need;
        }
        __syncthreads();

            const int th = selected_th;
            int32_t* const row_out =
                out_idx + static_cast<uint64_t>(row) * topk;
            const uint32_t th_code =
                dsa_litetopk::candidate_fp24_code(
                    static_cast<float>(th));
            const uint32_t next_th_code =
                dsa_litetopk::candidate_fp24_code(
                    static_cast<float>(th + 1));
            compact_topk_paged_large_boundary_select(
                inline_val,
                inline_idx,
                overflow_val,
                overflow_idx,
                page_table,
                row,
                selected_raw_n,
                overflow_pages,
                table_pages_per_row,
                th,
                th_code,
                next_th_code,
                selected_output_base,
                selected_k_target,
                topk,
                row_out,
                &page_id,
                hist,
                &desired,
                &kfind,
                &pivot_lt,
                &write_lt,
                &write_eq);
            dsa_litetopk_accumulate_inplace_votes(
                row_out, topk, tid, kThreads,
                votes, votes_len, row, vote_row_stride);
            __syncthreads();
        }
    }
}

__global__ void
compact_topk_min_thr_segmented_idx_out_litetopk_kernel(
        const CandidateValue* __restrict__ seed_val,
        const int32_t* __restrict__ seed_idx,
        const int32_t* __restrict__ seed_cnt,
        const CandidateValue* __restrict__ segment_val,
        const int32_t* __restrict__ segment_idx,
        const uint64_t* __restrict__ segment_desc,
        const uint64_t* __restrict__ spill_desc,
        const int32_t* __restrict__ row_spill,
        const int32_t* __restrict__ segment_pool_head,
        const int32_t* __restrict__ total_cnt,
        const int32_t* __restrict__ th_in,
        const int32_t* __restrict__ boundary_meta,
        const int32_t* __restrict__ finalizer_status,
        int rows,
        int desc_windows,
        int micro_records,
        int pool_pages,
        int physical_records,
        int logical_cap,
        int num_buckets,
        int topk,
        int32_t* __restrict__ out_idx,
        int32_t* __restrict__ votes,
        int votes_len,
        int vote_row_stride) {
    constexpr int kThreads = 256;
    constexpr int kBoundaryCapacity = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int producer_warp = tid >> 5;
    if (row >= rows) return;

    __shared__ int count_lt;
    __shared__ int count_eq;
    __shared__ int certificate_valid;
    __shared__ int lt_cursor;
    __shared__ int eq_cursor;
    __shared__ int has_spill;
    __shared__ uint32_t allocated_pages;
    __shared__ uint32_t boundary_val[kBoundaryCapacity];
    __shared__ int32_t boundary_idx[kBoundaryCapacity];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int pivot_lt;
    __shared__ int write_lt;
    __shared__ int write_eq;

    const int raw_seed_n = seed_cnt[row];
    const int raw_n = total_cnt[row];
    const int th = th_in[row];
    if (tid == 0) {
        const int spill_flag = row_spill[row];
        const int raw_pool_head = segment_pool_head[0];
        const int32_t* meta =
            boundary_meta +
            static_cast<uint64_t>(row) * num_buckets;
        const int tag = meta[0];
        const int meta_th = ~tag;
        const int meta_lt = meta[1];
        const int meta_eq = meta[2];
        const int need = topk - meta_lt;
        certificate_valid =
            finalizer_status[row] == 0 &&
            tag < 0 && meta_th == th &&
            th >= 0 && th < num_buckets &&
            raw_seed_n >= 0 &&
            raw_seed_n <= kPagedCandidateInline &&
            spill_flag >= 0 &&
            raw_pool_head >= 0 && raw_pool_head <= pool_pages &&
            raw_n >= topk && raw_n <= logical_cap &&
            meta_lt >= 0 && meta_lt < topk &&
            meta_eq > 0 && need > 0 && need <= meta_eq &&
            meta_lt + meta_eq <= raw_n;
        count_lt = meta_lt;
        count_eq = meta_eq;
        lt_cursor = 0;
        eq_cursor = 0;
        has_spill = spill_flag != 0;
        allocated_pages = raw_pool_head < 0
            ? 0u
            : static_cast<uint32_t>(raw_pool_head);
    }
    __syncthreads();
    if (!certificate_valid) {
        int32_t* const row_out =
            out_idx + static_cast<uint64_t>(row) * topk;
        for (int j = tid; j < topk; j += kThreads) row_out[j] = 0;
        return;
    }

    int32_t* const row_out =
        out_idx + static_cast<uint64_t>(row) * topk;
    const uint32_t th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th));
    const uint32_t next_th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th + 1));
    const uint64_t seed_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    compact_topk_paged_fast_scan_span(
        seed_val + seed_base,
        seed_idx + seed_base,
        raw_seed_n,
        th,
        th_code,
        next_th_code,
        row_out,
        boundary_val,
        boundary_idx,
        &lt_cursor,
        &eq_cursor,
        topk);
    __syncthreads();

    for (int window = 0; window < desc_windows; ++window) {
        uint32_t physical_start = 0u;
        uint32_t count = 0u;
        bool valid = true;
        if (lane == 0) {
            const uint64_t descriptor = segment_desc[
                (static_cast<uint64_t>(row) * desc_windows + window) *
                    kSegmentCandidateDescriptorWarps +
                producer_warp];
            physical_start = static_cast<uint32_t>(descriptor);
            count = static_cast<uint32_t>(descriptor >> 32);
            valid = segment_candidate_descriptor_valid(
                physical_start, count, row, producer_warp,
                micro_records, pool_pages, physical_records,
                allocated_pages);
            if (!valid) {
                atomicExch(&certificate_valid, 0);
                count = 0u;
            }
        }
        physical_start = __shfl_sync(0xffffffffu, physical_start, 0);
        count = __shfl_sync(0xffffffffu, count, 0);
        compact_topk_segment_fast_scan_warp(
            segment_val + physical_start,
            segment_idx + physical_start,
            count,
            th,
            th_code,
            next_th_code,
            row_out,
            boundary_val,
            boundary_idx,
            &lt_cursor,
            &eq_cursor,
            topk);
    }
    if (has_spill) {
        for (int window = 0; window < desc_windows; ++window) {
#pragma unroll
            for (int spill_slot = 0;
                 spill_slot < kSegmentCandidateSpillSlots;
                 ++spill_slot) {
                uint32_t physical_start = 0u;
                uint32_t count = 0u;
                if (lane == 0) {
                    const uint64_t descriptor = spill_desc[
                        ((static_cast<uint64_t>(row) * desc_windows +
                          window) * kSegmentCandidateDescriptorWarps +
                         producer_warp) * kSegmentCandidateSpillSlots +
                        spill_slot];
                    physical_start = static_cast<uint32_t>(descriptor);
                    count = static_cast<uint32_t>(descriptor >> 32);
                    if (!segment_candidate_spill_descriptor_valid(
                            physical_start, count, micro_records,
                            pool_pages, physical_records,
                            allocated_pages)) {
                        atomicExch(&certificate_valid, 0);
                        count = 0u;
                    }
                }
                physical_start = __shfl_sync(
                    0xffffffffu, physical_start, 0);
                count = __shfl_sync(0xffffffffu, count, 0);
                compact_topk_segment_fast_scan_warp(
                    segment_val + physical_start,
                    segment_idx + physical_start,
                    count,
                    th,
                    th_code,
                    next_th_code,
                    row_out,
                    boundary_val,
                    boundary_idx,
                    &lt_cursor,
                    &eq_cursor,
                    topk);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        certificate_valid = certificate_valid &&
            lt_cursor == count_lt && eq_cursor == count_eq;
    }
    __syncthreads();
    if (!certificate_valid) {
        for (int j = tid; j < topk; j += kThreads) row_out[j] = 0;
        return;
    }

    const int output_base = count_lt;
    const int boundary_n = count_eq;
    const int k_target = topk - output_base;
    // A separate cold continuation owns valid larger boundaries so this hot
    // kernel retains the same register/occupancy profile as the shared path.
    if (boundary_n > kBoundaryCapacity) return;
    if (boundary_n == k_target) {
        for (int j = tid; j < boundary_n; j += kThreads) {
            row_out[output_base + j] = boundary_idx[j];
        }
        dsa_litetopk_accumulate_inplace_votes(
            row_out, topk, tid, kThreads,
            votes, votes_len, row, vote_row_stride);
        return;
    }

    if (tid == 0) {
        desired = th == 0
            ? 0u
            : (boundary_val[0] & 0xff0000u);
        kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();
    uint32_t mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const bool full_key = th == 0;
        const int num_passes = full_key ? 3 : 2;
        if (pass < num_passes) {
            const int shift =
                (full_key ? 16 : 8) - pass * 8;
            hist[tid] = 0u;
            __syncthreads();
            const uint32_t desired_base = desired;
            if (tid < boundary_n) {
                const uint32_t encoded = boundary_val[tid];
                if ((encoded & mask) ==
                    (desired_base & mask)) {
                    atomicAdd(
                        &hist[(encoded >> shift) & 0xffu], 1u);
                }
            }
            __syncthreads();
            compact_find_radix_digit_warp0(
                hist, &desired, &kfind,
                desired_base, shift, tid);
            __syncthreads();
            mask |= 0xffu << shift;
        }
    }
    const uint32_t pivot = desired;
    if (tid == 0) {
        pivot_lt = 0;
        write_lt = 0;
        write_eq = 0;
    }
    __syncthreads();
    if (tid < boundary_n && boundary_val[tid] < pivot) {
        atomicAdd(&pivot_lt, 1);
    }
    __syncthreads();
    const int eq_take = max(k_target - pivot_lt, 0);
    if (tid < boundary_n) {
        const uint32_t encoded = boundary_val[tid];
        if (encoded < pivot) {
            const int pos = atomicAdd(&write_lt, 1);
            if (pos < k_target) {
                row_out[output_base + pos] = boundary_idx[tid];
            }
        } else if (encoded == pivot) {
            const int equal_rank = atomicAdd(&write_eq, 1);
            if (equal_rank < eq_take) {
                const int pos = output_base + pivot_lt + equal_rank;
                if (pos < topk) row_out[pos] = boundary_idx[tid];
            }
        }
    }
    dsa_litetopk_accumulate_inplace_votes(
        row_out, topk, tid, kThreads,
        votes, votes_len, row, vote_row_stride);
}

__device__ __forceinline__ void
compact_topk_segmented_descriptor_span(
        const uint64_t* __restrict__ segment_desc,
        const uint64_t* __restrict__ spill_desc,
        const int row,
        const int desc_windows,
        const int descriptor_id,
        const bool has_spill,
        const int micro_records,
        const int pool_pages,
        const int physical_records,
        const uint32_t allocated_pages,
        uint32_t* __restrict__ span_start,
        uint32_t* __restrict__ span_count,
        int* __restrict__ descriptors_valid) {
    const int base_descriptors =
        desc_windows * kSegmentCandidateDescriptorWarps;
    uint64_t descriptor = 0u;
    bool valid = false;
    if (descriptor_id < base_descriptors) {
        descriptor = segment_desc[
            static_cast<uint64_t>(row) * base_descriptors +
            descriptor_id];
        const uint32_t start = static_cast<uint32_t>(descriptor);
        const uint32_t count = static_cast<uint32_t>(descriptor >> 32);
        valid = segment_candidate_descriptor_valid(
            start, count, row,
            descriptor_id % kSegmentCandidateDescriptorWarps,
            micro_records, pool_pages, physical_records,
            allocated_pages);
    } else if (has_spill) {
        const int spill_id = descriptor_id - base_descriptors;
        descriptor = spill_desc[
            static_cast<uint64_t>(row) * base_descriptors *
                kSegmentCandidateSpillSlots +
            spill_id];
        const uint32_t start = static_cast<uint32_t>(descriptor);
        const uint32_t count = static_cast<uint32_t>(descriptor >> 32);
        valid = segment_candidate_spill_descriptor_valid(
            start, count, micro_records, pool_pages,
            physical_records, allocated_pages);
    }
    uint32_t start = static_cast<uint32_t>(descriptor);
    uint32_t count = static_cast<uint32_t>(descriptor >> 32);
    if (!valid) {
        *descriptors_valid = 0;
        start = 0u;
        count = 0u;
    } else if (count == 0u) {
        start = 0u;
    }
    *span_start = start;
    *span_count = count;
}

static __device__ __noinline__ void
compact_topk_segmented_large_boundary_select(
        const CandidateValue* __restrict__ seed_val,
        const int32_t* __restrict__ seed_idx,
        const CandidateValue* __restrict__ segment_val,
        const int32_t* __restrict__ segment_idx,
        const uint64_t* __restrict__ segment_desc,
        const uint64_t* __restrict__ spill_desc,
        const int row,
        const int seed_n,
        const int desc_windows,
        const bool has_spill,
        const int micro_records,
        const int pool_pages,
        const int physical_records,
        const uint32_t allocated_pages,
        const int th,
        const int output_base,
        const int k_target,
        const int topk,
        int32_t* __restrict__ row_out,
        uint32_t* __restrict__ span_start,
        uint32_t* __restrict__ span_count,
        uint32_t* __restrict__ hist,
        uint32_t* __restrict__ desired,
        uint32_t* __restrict__ kfind,
        int* __restrict__ pivot_lt,
        int* __restrict__ write_lt,
        int* __restrict__ write_eq,
        int* __restrict__ descriptors_valid) {
    constexpr int kThreads = 256;
    const int tid = static_cast<int>(threadIdx.x);
    const uint32_t th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th));
    const uint32_t next_th_code =
        dsa_litetopk::candidate_fp24_code(static_cast<float>(th + 1));
    const uint64_t seed_base =
        static_cast<uint64_t>(row) * kPagedCandidateInline;
    const int base_descriptors =
        desc_windows * kSegmentCandidateDescriptorWarps;
    const int num_descriptors = base_descriptors *
        (has_spill ? 1 + kSegmentCandidateSpillSlots : 1);

    if (tid == 0) {
        *desired = th == 0
            ? 0u
            : (th_code & 0xff0000u);
        *kfind = static_cast<uint32_t>(k_target);
        *descriptors_valid = 1;
    }
    __syncthreads();
    uint32_t prefix_mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const bool full_key = th == 0;
        const int num_passes = full_key ? 3 : 2;
        if (pass >= num_passes) continue;
        const int shift =
            (full_key ? 16 : 8) - pass * 8;
        hist[tid] = 0u;
        __syncthreads();
        const uint32_t desired_base = *desired;
        compact_topk_paged_radix_hist_span(
            seed_val + seed_base, seed_idx + seed_base,
            seed_n, th, th_code, next_th_code,
            prefix_mask, desired_base, shift, hist);
        __syncthreads();
        for (int descriptor_id = 0;
             descriptor_id < num_descriptors;
             ++descriptor_id) {
            if (tid == 0) {
                compact_topk_segmented_descriptor_span(
                    segment_desc, spill_desc, row, desc_windows,
                    descriptor_id, has_spill, micro_records,
                    pool_pages, physical_records, allocated_pages,
                    span_start, span_count, descriptors_valid);
            }
            __syncthreads();
            compact_topk_paged_radix_hist_span(
                segment_val + *span_start,
                segment_idx + *span_start,
                static_cast<int>(*span_count),
                th, th_code, next_th_code,
                prefix_mask, desired_base, shift, hist);
            __syncthreads();
        }
        compact_find_radix_digit_warp0(
            hist, desired, kfind, desired_base, shift, tid);
        __syncthreads();
        prefix_mask |= 0xffu << shift;
    }
    const uint32_t pivot = *desired;

    if (tid == 0) *pivot_lt = 0;
    __syncthreads();
    compact_topk_paged_count_pivot_lt_span(
        seed_val + seed_base, seed_idx + seed_base,
        seed_n, th, th_code, next_th_code,
        pivot, pivot_lt);
    __syncthreads();
    for (int descriptor_id = 0;
         descriptor_id < num_descriptors;
         ++descriptor_id) {
        if (tid == 0) {
            compact_topk_segmented_descriptor_span(
                segment_desc, spill_desc, row, desc_windows,
                descriptor_id, has_spill, micro_records,
                pool_pages, physical_records, allocated_pages,
                span_start, span_count, descriptors_valid);
        }
        __syncthreads();
        compact_topk_paged_count_pivot_lt_span(
            segment_val + *span_start,
            segment_idx + *span_start,
            static_cast<int>(*span_count),
            th, th_code, next_th_code, pivot, pivot_lt);
        __syncthreads();
    }

    const int exact_pivot_lt = *pivot_lt;
    const int eq_take = max(k_target - exact_pivot_lt, 0);
    if (tid == 0) {
        *write_lt = 0;
        *write_eq = 0;
    }
    __syncthreads();
    compact_topk_paged_write_pivot_span(
        seed_val + seed_base, seed_idx + seed_base,
        seed_n, th, th_code, next_th_code,
        pivot, output_base, exact_pivot_lt, eq_take,
        topk, row_out, write_lt, write_eq);
    __syncthreads();
    for (int descriptor_id = 0;
         descriptor_id < num_descriptors;
         ++descriptor_id) {
        if (tid == 0) {
            compact_topk_segmented_descriptor_span(
                segment_desc, spill_desc, row, desc_windows,
                descriptor_id, has_spill, micro_records,
                pool_pages, physical_records, allocated_pages,
                span_start, span_count, descriptors_valid);
        }
        __syncthreads();
        compact_topk_paged_write_pivot_span(
            segment_val + *span_start,
            segment_idx + *span_start,
            static_cast<int>(*span_count),
            th, th_code, next_th_code,
            pivot, output_base, exact_pivot_lt, eq_take,
            topk, row_out, write_lt, write_eq);
        __syncthreads();
    }
    if (!*descriptors_valid) {
        for (int j = tid; j < topk; j += kThreads) row_out[j] = 0;
        __syncthreads();
        return;
    }
    if (*write_lt != exact_pivot_lt || *write_eq < eq_take) {
        asm volatile("trap;");
        return;
    }
    __syncthreads();
}

__global__ void
compact_topk_min_thr_segmented_large_boundary_litetopk_kernel(
        const CandidateValue* __restrict__ seed_val,
        const int32_t* __restrict__ seed_idx,
        const int32_t* __restrict__ seed_cnt,
        const CandidateValue* __restrict__ segment_val,
        const int32_t* __restrict__ segment_idx,
        const uint64_t* __restrict__ segment_desc,
        const uint64_t* __restrict__ spill_desc,
        const int32_t* __restrict__ row_spill,
        const int32_t* __restrict__ segment_pool_head,
        const int32_t* __restrict__ total_cnt,
        const int32_t* __restrict__ th_in,
        const int32_t* __restrict__ boundary_meta,
        const int32_t* __restrict__ finalizer_status,
        int rows,
        int desc_windows,
        int micro_records,
        int pool_pages,
        int physical_records,
        int logical_cap,
        int num_buckets,
        int topk,
        int32_t* __restrict__ out_idx,
        int32_t* __restrict__ votes,
        int votes_len,
        int vote_row_stride) {
    constexpr int kThreads = 256;
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ int task_count;
    __shared__ int task_rows[kThreads];
    __shared__ int selected_seed_n;
    __shared__ int selected_th;
    __shared__ int selected_output_base;
    __shared__ int selected_k_target;
    __shared__ int selected_has_spill;
    __shared__ uint32_t selected_allocated_pages;
    __shared__ int descriptors_valid;
    __shared__ uint32_t span_start;
    __shared__ uint32_t span_count;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int pivot_lt;
    __shared__ int write_lt;
    __shared__ int write_eq;

    const int tile_stride =
        static_cast<int>(gridDim.x) * kThreads;
    for (int tile_base =
             static_cast<int>(blockIdx.x) * kThreads;
         tile_base < rows;
         tile_base += tile_stride) {
        if (tid == 0) task_count = 0;
        __syncthreads();
        const int candidate_row = tile_base + tid;
        bool is_large = false;
        if (candidate_row < rows &&
            finalizer_status[candidate_row] == 0) {
            const int spill_flag = row_spill[candidate_row];
            const int raw_pool_head = segment_pool_head[0];
            const int raw_n = total_cnt[candidate_row];
            const int th = th_in[candidate_row];
            const int32_t* meta =
                boundary_meta +
                static_cast<uint64_t>(candidate_row) * num_buckets;
            const int tag = meta[0];
            const int meta_th = ~tag;
            const int meta_lt = meta[1];
            const int meta_eq = meta[2];
            const int need = topk - meta_lt;
            is_large =
                tag < 0 && meta_th == th &&
                th >= 0 && th < num_buckets &&
                seed_cnt[candidate_row] >= 0 &&
                seed_cnt[candidate_row] <= kPagedCandidateInline &&
                spill_flag >= 0 &&
                raw_pool_head >= 0 && raw_pool_head <= pool_pages &&
                raw_n >= topk && raw_n <= logical_cap &&
                meta_lt >= 0 && meta_lt < topk &&
                meta_eq > 256 && need > 0 && need <= meta_eq &&
                meta_lt + meta_eq <= raw_n;
        }
        if (is_large) {
            const int slot = atomicAdd(&task_count, 1);
            task_rows[slot] = candidate_row;
        }
        __syncthreads();
        const int num_tasks = task_count;
        for (int task = 0; task < num_tasks; ++task) {
            const int row = task_rows[task];
            if (tid == 0) {
                const int32_t* meta =
                    boundary_meta +
                    static_cast<uint64_t>(row) * num_buckets;
                selected_seed_n = seed_cnt[row];
                selected_th = th_in[row];
                selected_output_base = meta[1];
                selected_k_target = topk - meta[1];
                selected_has_spill = row_spill[row] != 0;
                selected_allocated_pages = static_cast<uint32_t>(
                    segment_pool_head[0]);
            }
            __syncthreads();
            int32_t* const row_out =
                out_idx + static_cast<uint64_t>(row) * topk;
            compact_topk_segmented_large_boundary_select(
                seed_val, seed_idx, segment_val, segment_idx,
                segment_desc, spill_desc, row, selected_seed_n,
                desc_windows, selected_has_spill, micro_records,
                pool_pages, physical_records, selected_allocated_pages,
                selected_th,
                selected_output_base, selected_k_target, topk,
                row_out, &span_start, &span_count,
                hist, &desired, &kfind,
                &pivot_lt, &write_lt, &write_eq,
                &descriptors_valid);
            dsa_litetopk_accumulate_inplace_votes(
                row_out, topk, tid, kThreads,
                votes, votes_len, row, vote_row_stride);
            __syncthreads();
        }
    }
}

template <int kImplHeads, int kImplBlockQ>
static int compute_smem_bytes_t(bool include_hist = true) {
    const int esz_fp8 = 1, esz_f32 = 4;
    const int smem_q  = kImplBlockQ * kImplHeads * HEAD_DIM * esz_fp8;
    const int smem_w  = kImplBlockQ * kImplHeads * esz_f32;
    const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
    const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
    const int num_barriers =
        NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 +
        (MATH_THREADS / 128) * dsa_litetopk::kUmmaStages * 2;
    const int smem_barriers = num_barriers * 8;
    const int smem_slots = 4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
    constexpr int emit_record_bytes = (int)sizeof(uint32_t);
    const int smem_warpq =
        (MATH_THREADS / 32) * kImplBlockQ *
        ((int)sizeof(int32_t) +
         dsa_litetopk::kEmitLaneSlots * 32 * emit_record_bytes);
    const int smem_hist = include_hist
        ? kImplBlockQ * 256 * (int)sizeof(int32_t)
        : 0;  // per-CTA refresh histogram (NB<=256)
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_slots + smem_warpq + smem_hist;
}

static int compute_smem_bytes(bool include_hist = true) {
    return compute_smem_bytes_t<NUM_HEADS, BLOCK_Q>(include_hist);
}

constexpr int NUM_KV_STAGES_FP4 = 6;

template <int kImplHeads, int kImplBlockQ>
static int compute_smem_bytes_fp4_t(bool include_hist) {
    const int esz_i32 = 4;
    const int smem_q  = kImplBlockQ * kImplHeads * (HEAD_DIM / 2);
    const int smem_w  = kImplBlockQ * kImplHeads * 4;
    const int smem_kv = BLOCK_KV * (HEAD_DIM / 2);
    const int smem_ks = align_up(align_up(BLOCK_KV, 128) * esz_i32, 512);
    const int smem_sfq = align_up(kImplBlockQ * kImplHeads, 128) * esz_i32;
    const int num_barriers =
        NUM_Q_STAGES * 2 + NUM_KV_STAGES_FP4 * 2 + 3 * 2 +
        NUM_KV_STAGES_FP4;  // 3 TMEM stages + sf_ready
    const int smem_barriers = num_barriers * 8;
    const int smem_slots = 4 * (int)sizeof(uint32_t);
    const int smem_warpq =
        (MATH_THREADS / 32) * kImplBlockQ *
        ((int)sizeof(int32_t) +
         dsa_litetopk::kEmitLaneSlots * 32 * (int)sizeof(uint32_t));
    const int smem_hist = include_hist
        ? kImplBlockQ * 256 * (int)sizeof(int32_t)
        : 0;
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES_FP4 * smem_kv + NUM_KV_STAGES_FP4 * smem_ks + smem_sfq +
           smem_barriers + smem_slots + smem_warpq + smem_hist;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_litetopk(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        int64_t num_buckets64,
        int64_t cand_cap64,
        int64_t topk64,
        int64_t refresh_every64,
        int64_t num_kv_splits_override) {
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda() && origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
                seed_val.is_cuda() && seed_idx.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() && weights.is_contiguous() &&
                cu_start.is_contiguous() && cu_end.is_contiguous() && origin.is_contiguous() && inv_delta.is_contiguous() &&
                th_bucket.is_contiguous() && seed_val.is_contiguous() && seed_idx.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q must be fp8_e4m3fn");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat8_e4m3fn, "kv must be fp8_e4m3fn");
    TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat, "kv_scales must be fp32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be fp32");
    TORCH_CHECK(origin.scalar_type() == torch::kFloat && inv_delta.scalar_type() == torch::kFloat, "origin/inv_delta must be fp32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt, "th_bucket must be int32");
    TORCH_CHECK(seed_val.scalar_type() == torch::kFloat, "seed_val must be fp32 x=-score");
    TORCH_CHECK(seed_idx.scalar_type() == torch::kInt, "seed_idx must be int32");
    TORCH_CHECK(cu_start.scalar_type() == torch::kInt && cu_end.scalar_type() == torch::kInt,
                "cu_start/cu_end must be int32");

    const int seq_len = (int)q.size(0);
    const int num_heads = (int)q.size(1);
    const int head_dim = (int)q.size(2);
    const int seq_len_kv = (int)kv.size(0);
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    const int topk = static_cast<int>(topk64);
    // Sparse-only: honor a caller-provided cap in [topk, S).
    const int cand_cap = (cand_cap64 >= topk && cand_cap64 < seq_len_kv)
                             ? static_cast<int>(cand_cap64) : seq_len_kv;
    TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
    TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len && th_bucket.numel() == seq_len, "bucket params must have Q elements");
    const int num_buckets = static_cast<int>(num_buckets64);
    TORCH_CHECK(refresh_every64 > 0,
                "sparse refresh requires refresh_every>0");
    const int refresh_every = static_cast<int>(refresh_every64);
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
                "in-place boundary select requires 3 <= num_buckets <= 256");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1");
    TORCH_CHECK(seed_val.dim() == 2 && seed_idx.dim() == 2, "seed tensors must be [Q, seed_k]");
    TORCH_CHECK(seed_val.size(0) == seq_len && seed_idx.size(0) == seq_len && seed_val.size(1) == seed_idx.size(1),
                "seed tensor shape mismatch");
    const int seed_k = static_cast<int>(seed_val.size(1));
    TORCH_CHECK(seed_k <= cand_cap, "seed_k must be <= cand_cap");
    TORCH_CHECK(
        seed_k == 0,
        "production scan requires empty seeds; use the "
        "prepared ext API so sampled positions are not double-counted");

    auto cand_val = torch::empty(
        {seq_len, cand_cap}, candidate_options(q.options()));
    auto cand_idx = torch::empty({seq_len, cand_cap}, q.options().dtype(torch::kInt));
    auto cand_cnt = torch::full({seq_len}, seed_k, q.options().dtype(torch::kInt));
    auto bcount = torch::zeros({seq_len, num_buckets}, q.options().dtype(torch::kInt));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int esz_fp8 = 1, esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM, BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                         HEAD_DIM, seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                         ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                        NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

    const int smem = compute_smem_bytes();

    // Production packed-candidate path: one CTA owns each row's complete
    // histogram and publishes an exact boundary certificate.
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    TORCH_CHECK(
        num_kv_splits_override <= 0 || num_kv_splits_override == 1,
        "production packed candidates require num_kv_splits=1");
    constexpr int num_kv_splits = 1;
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
        SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits, 0u, 0ULL, 0u,
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u,
        nullptr, nullptr, nullptr, nullptr, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(cand_val, cand_idx, cand_cnt);
}

void launch_seed_prep(
        const float* slog,
        int64_t slog_stride,
        int Q,
        int head,
        int NB,
        int K,
        float headroom,
        float* origin,
        float* inv_delta,
        int32_t* th_bucket,
        CandidateValue* cand_val,
        int32_t* cand_idx,
        int32_t* cand_cnt,
        int cand_cap,
        int emit_limit,
        int physical_index_base,
        int32_t* bcount,
        cudaStream_t stream) {
    const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
    if (head == 12288) {
        if (emit_limit == head) {
            seed_prep_kernel<true, 12288, kSeed12Threads>
                <<<Q, kSeed12Threads, seed_smem, stream>>>(
                    slog, slog_stride, head, NB, K, headroom,
                    origin, inv_delta, th_bucket,
                    cand_val, cand_idx, cand_cnt,
                    cand_cap, physical_index_base, bcount);
        } else {
            seed_prep_kernel<false, 12288, kSeed12Threads>
                <<<Q, kSeed12Threads, seed_smem, stream>>>(
                    slog, slog_stride, head, NB, K, headroom,
                    origin, inv_delta, th_bucket,
                    nullptr, nullptr, cand_cnt, cand_cap, 0, bcount);
        }
    } else if (emit_limit == head) {
        seed_prep_kernel<true, 8192, kSeedThreads>
            <<<Q, kSeedThreads, seed_smem, stream>>>(
                slog, slog_stride, head, NB, K, headroom,
                origin, inv_delta, th_bucket,
                cand_val, cand_idx, cand_cnt,
                cand_cap, physical_index_base, bcount);
    } else {
        seed_prep_kernel<false, 8192, kSeedThreads>
            <<<Q, kSeedThreads, seed_smem, stream>>>(
                slog, slog_stride, head, NB, K, headroom,
                origin, inv_delta, th_bucket,
                nullptr, nullptr, cand_cnt, cand_cap, 0, bcount);
    }
}

// Fused seed/prep: sample scores -> (origin, inv_delta, th_bucket, cand_val,
// cand_idx, cand_cnt, bcount), everything the scan needs, in one launch.
void seed_prep_litetopk_(torch::Tensor slog, int64_t num_buckets64, int64_t topk64,
                       int64_t cand_cap64, int64_t emit_limit64, double headroom,
                       int64_t probe_stride_tok64, int64_t hist_stride64,
                       torch::Tensor origin, torch::Tensor inv_delta,
                       torch::Tensor th_bucket, torch::Tensor bcount,
                       torch::Tensor cand_val, torch::Tensor cand_idx,
                       torch::Tensor cand_cnt) {
    TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
    TORCH_CHECK(
        origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
            bcount.is_cuda() && cand_val.is_cuda() && cand_idx.is_cuda() &&
            cand_cnt.is_cuda(),
        "seed prep outputs must be CUDA tensors");
    TORCH_CHECK(
        slog.device() == origin.device() && slog.device() == inv_delta.device() &&
            slog.device() == th_bucket.device() && slog.device() == bcount.device() &&
            slog.device() == cand_val.device() && slog.device() == cand_idx.device() &&
            slog.device() == cand_cnt.device(),
        "seed prep tensors must be on one CUDA device");
    TORCH_CHECK(
        origin.is_contiguous() && inv_delta.is_contiguous() &&
            th_bucket.is_contiguous() && bcount.is_contiguous() &&
            cand_val.is_contiguous() && cand_idx.is_contiguous() &&
            cand_cnt.is_contiguous(),
        "seed prep outputs must be contiguous");
    TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
    TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
    const int Q = (int)slog.size(0);
    const int head = (int)slog.size(1);
    const int NB = (int)num_buckets64;
    const int K = (int)topk64;
    const int cap = (int)cand_cap64;
    TORCH_CHECK(
        head >= K && (head <= 8192 || head == 12288),
        "production seed prep requires topk <= HOT <= 8192 or HOT=12288");
    TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
    TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
    TORCH_CHECK(
        origin.scalar_type() == torch::kFloat &&
            inv_delta.scalar_type() == torch::kFloat &&
            th_bucket.scalar_type() == torch::kInt &&
            bcount.scalar_type() == torch::kInt &&
            cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt,
        "seed prep affine outputs must be fp32 and metadata/indices int32");
    TORCH_CHECK(
        origin.dim() == 1 && origin.numel() >= Q &&
            inv_delta.dim() == 1 && inv_delta.numel() >= Q &&
            th_bucket.dim() == 1 && th_bucket.numel() >= Q &&
            cand_cnt.dim() == 1 && cand_cnt.numel() >= Q,
        "origin/inv_delta/th_bucket/cand_cnt must cover Q rows");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_val.size(0) >= Q &&
            cand_val.size(1) == cap && cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [>=Q,cand_cap]");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        bcount.dim() == 2 && bcount.size(0) >= Q && bcount.size(1) == NB,
        "bcount must be [>=Q,num_buckets]");
    TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
                "slog rows must be 16B aligned");
    const c10::cuda::CUDAGuard device_guard(slog.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int emit_limit = emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
    TORCH_CHECK(
        emit_limit == 0 || emit_limit == head,
        "production seed prep requires emit_limit to be 0 or HOT");
    TORCH_CHECK(hist_stride64 == 1,
                "production seed prep requires hist_stride=1");
    const int64_t physical_index_base64 =
        emit_limit == head ? probe_stride_tok64 : 0;
    TORCH_CHECK(
        physical_index_base64 >= 0 &&
            physical_index_base64 + head <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "HOT physical index range exceeds the packed 20-bit candidate ABI");
    launch_seed_prep(
        slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
        static_cast<float>(headroom),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(), candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(), cap,
        emit_limit, static_cast<int>(physical_index_base64),
        bcount.data_ptr<int32_t>(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
seed_prep_litetopk(torch::Tensor slog, int64_t num_buckets64, int64_t topk64,
                 int64_t cand_cap64, int64_t emit_limit64, double headroom,
                 int64_t probe_stride_tok64, int64_t hist_stride64) {
    TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
    TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
    TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
    const int Q = (int)slog.size(0);
    const int head = (int)slog.size(1);
    const int NB = (int)num_buckets64;
    const int K = (int)topk64;
    const int cap = (int)cand_cap64;
    TORCH_CHECK(
        head >= K && (head <= 8192 || head == 12288),
        "production seed prep requires topk <= HOT <= 8192 or HOT=12288");
    TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
    TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
    // float4 pass requires 16B-aligned rows; fall back is not implemented.
    TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
                "slog rows must be 16B aligned");

    auto opts_f = slog.options();
    auto opts_i = slog.options().dtype(torch::kInt);
    auto origin = torch::empty({Q}, opts_f);
    auto inv_delta = torch::empty({Q}, opts_f);
    auto th_bucket = torch::empty({Q}, opts_i);
    auto bcount = torch::empty({Q, NB}, opts_i);  // scan publishes boundary metadata
    auto cand_val = torch::empty({Q, cap}, candidate_options(opts_f));
    auto cand_idx = torch::empty({Q, cap}, opts_i);
    auto cand_cnt = torch::empty({Q}, opts_i);

    const c10::cuda::CUDAGuard device_guard(slog.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int emit_limit = emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
    TORCH_CHECK(
        emit_limit == 0 || emit_limit == head,
        "production seed prep requires emit_limit to be 0 or HOT");
    TORCH_CHECK(hist_stride64 == 1,
                "production seed prep requires hist_stride=1");
    const int64_t physical_index_base64 =
        emit_limit == head ? probe_stride_tok64 : 0;
    TORCH_CHECK(
        physical_index_base64 >= 0 &&
            physical_index_base64 + head <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "HOT physical index range exceeds the packed 20-bit candidate ABI");
    launch_seed_prep(
        slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
        static_cast<float>(headroom),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(), candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(), cap,
        emit_limit, static_cast<int>(physical_index_base64),
        bcount.data_ptr<int32_t>(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(origin, inv_delta, th_bucket, cand_val, cand_idx,
                           cand_cnt, bcount);
}

// Scan into buffers prepared by seed_prep_litetopk (no seeding of any kind).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_litetopk_ext(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64,
        int64_t refresh_every64,
        int64_t num_kv_splits_override,
        int64_t probe_group64,
        int64_t probe_add_max64) {
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() && weights.is_contiguous() &&
                cu_start.is_contiguous() && cu_end.is_contiguous() && origin.is_contiguous() && inv_delta.is_contiguous() &&
                th_bucket.is_contiguous() && cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                cand_cnt.is_contiguous() && bcount.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && kv.scalar_type() == torch::kFloat8_e4m3fn,
                "q/kv must be fp8_e4m3fn");
    check_candidate_dtype(cand_val);
    const int seq_len = (int)q.size(0);
    const int seq_len_kv = (int)kv.size(0);
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    const int cand_cap = (int)cand_val.size(1);
    const int num_buckets = (int)num_buckets64;
    const int topk = (int)topk64;
    TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
                "prepared scan requires 3 <= num_buckets <= 256");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap,
                "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 > 0,
                "sparse refresh requires refresh_every>0");
    TORCH_CHECK(cand_val.size(0) == seq_len && cand_idx.sizes() == cand_val.sizes() &&
                cand_cnt.numel() == seq_len && bcount.size(0) == seq_len && bcount.size(1) == num_buckets,
                "prepared buffer shape mismatch");
    const int refresh_every = static_cast<int>(refresh_every64);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int esz_fp8 = 1, esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM, BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                         HEAD_DIM, seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                         ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                        NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

    const int smem = compute_smem_bytes();

    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    TORCH_CHECK(
        num_kv_splits_override <= 0 || num_kv_splits_override == 1,
        "production packed candidates require num_kv_splits=1");
    constexpr int num_kv_splits = 1;
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
        SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits, (uint32_t)probe_group64,
        probe_group64 > 0 ? (((1ULL << 42) + (uint64_t)probe_group64 - 1) / (uint64_t)probe_group64) : 0ULL,
        (uint32_t)probe_add_max64,
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u,
        nullptr, nullptr, nullptr, nullptr, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(cand_val, cand_idx, cand_cnt);
}

// LITETOPK_STATIC_HOT_AB: offline-only scan using the caller's HOT8192
// seed_prep outputs. The sample threshold remains fixed throughout the full
// score scan; the kernel builds the passing-candidate histogram and publishes
// one tight boundary certificate at completion. No buffers are allocated or
// initialized here: cand_cnt must already contain the seed-prep value (zero in
// the production HOT-only/no-emit contract).
template <int kImplHeads, int kImplBlockQ>
static void mqa_logits_dsa_static_hot_litetopk_impl_t(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64,
        bool no_hist,
        bool exact_gate,
        std::optional<torch::Tensor> dump_slab = std::nullopt,
        std::optional<torch::Tensor> dump_counts = std::nullopt,
        int64_t dump_max_windows = 0) {
    TORCH_CHECK(
        !exact_gate || no_hist,
        "exact FP24 gate requires the histogram-free static scan");
    TORCH_CHECK(
        q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
            weights.is_cuda() && cu_start.is_cuda() && cu_end.is_cuda() &&
            origin.is_cuda() && inv_delta.is_cuda() &&
            th_bucket.is_cuda() && cand_val.is_cuda() &&
            cand_idx.is_cuda() && cand_cnt.is_cuda() && bcount.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(
        q.device() == kv.device() && q.device() == kv_scales.device() &&
            q.device() == weights.device() && q.device() == cu_start.device() &&
            q.device() == cu_end.device() && q.device() == origin.device() &&
            q.device() == inv_delta.device() &&
            q.device() == th_bucket.device() &&
            q.device() == cand_val.device() && q.device() == cand_idx.device() &&
            q.device() == cand_cnt.device() && q.device() == bcount.device(),
        "all tensors must be on the same CUDA device");
    TORCH_CHECK(
        q.is_contiguous() && kv.is_contiguous() &&
            kv_scales.is_contiguous() && weights.is_contiguous() &&
            cu_start.is_contiguous() && cu_end.is_contiguous() &&
            origin.is_contiguous() && inv_delta.is_contiguous() &&
            th_bucket.is_contiguous() && cand_val.is_contiguous() &&
            cand_idx.is_contiguous() && cand_cnt.is_contiguous() &&
            bcount.is_contiguous(),
        "all tensors must be contiguous");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat8_e4m3fn &&
            kv.scalar_type() == torch::kFloat8_e4m3fn,
        "q/kv must be fp8_e4m3fn");
    TORCH_CHECK(
        kv_scales.scalar_type() == torch::kFloat &&
            weights.scalar_type() == torch::kFloat &&
            origin.scalar_type() == torch::kFloat &&
            inv_delta.scalar_type() == torch::kFloat,
        "kv_scales/weights/origin/inv_delta must be fp32");
    TORCH_CHECK(
        cu_start.scalar_type() == torch::kInt &&
            cu_end.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            bcount.scalar_type() == torch::kInt,
        "range, threshold, candidate index/count and metadata must be int32");
    check_candidate_dtype(cand_val);

    TORCH_CHECK(q.dim() == 3, "q must be [Q,32,128]");
    TORCH_CHECK(kv.dim() == 2, "kv must be [S,128]");
    const int seq_len = static_cast<int>(q.size(0));
    const int seq_len_kv = static_cast<int>(kv.size(0));
    TORCH_CHECK(seq_len > 0 && seq_len_kv > 0, "Q and S must be nonzero");
    TORCH_CHECK(
        q.size(1) == kImplHeads && q.size(2) == HEAD_DIM &&
            kv.size(1) == HEAD_DIM,
        "static HOT path requires H in {32,64}, D=128");
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    TORCH_CHECK(
        weights.dim() == 2 && weights.size(0) == seq_len &&
            weights.size(1) == kImplHeads,
        "weights must be [Q,32]");
    TORCH_CHECK(
        cu_start.dim() == 1 && cu_start.numel() == seq_len &&
            cu_end.dim() == 1 && cu_end.numel() == seq_len,
        "cu_start/cu_end must have Q elements");
    TORCH_CHECK(
        origin.dim() == 1 && origin.numel() == seq_len &&
            inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
            th_bucket.dim() == 1 && th_bucket.numel() == seq_len,
        "origin/inv_delta/th_bucket must have Q elements");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_val.size(0) == seq_len &&
            cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [Q,cand_cap]");
    const int cand_cap = static_cast<int>(cand_val.size(1));
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    TORCH_CHECK(
        num_buckets >= 3 && num_buckets <= 256,
        "static HOT path requires 3 <= num_buckets <= 256");
    TORCH_CHECK(
        topk >= 1 && topk <= cand_cap,
        "topk must be in [1,cand_cap]");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len,
        "cand_cnt must have Q elements");
    TORCH_CHECK(
        bcount.dim() == 2 && bcount.size(0) == seq_len &&
            bcount.size(1) == num_buckets,
        "bcount must be [Q,num_buckets]");

    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int esz_fp8 = 1;
    const int esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    TORCH_CHECK(
        kv_scales.numel() >= ks_aligned,
        "kv_scales storage is shorter than the aligned KV length");
    auto tm_q = make_2d(
        q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len * kImplHeads, HEAD_DIM,
        kImplBlockQ * kImplHeads, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(
        kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(
        kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(
        weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        kImplHeads, seq_len, kImplHeads, kImplBlockQ, kImplHeads, 0);

    const bool raw_dump = dump_slab.has_value();
    const int num_q_blocks = (seq_len + kImplBlockQ - 1) / kImplBlockQ;
    if (raw_dump) {
        TORCH_CHECK(
            no_hist && !exact_gate,
            "raw-dump emit requires the plain no-hist writer");
        TORCH_CHECK(
            dump_counts.has_value(),
            "raw-dump emit needs both the region slab and the counts slab");
        const auto& slab = *dump_slab;
        const auto& counts = *dump_counts;
        TORCH_CHECK(
            slab.is_cuda() && counts.is_cuda() &&
                slab.device() == q.device() &&
                counts.device() == q.device() &&
                slab.is_contiguous() && counts.is_contiguous() &&
                slab.scalar_type() == torch::kInt &&
                counts.scalar_type() == torch::kInt,
            "dump slabs must be contiguous int32 CUDA tensors");
        const int64_t region_words = static_cast<int64_t>(kImplBlockQ) *
            dsa_litetopk::kEmitLaneSlots * 32;
        const int64_t required_windows =
            (((static_cast<int64_t>(seq_len_kv) + BLOCK_KV - 1) / BLOCK_KV) +
             dsa_litetopk::kEmitChunkBlocks - 1) /
            dsa_litetopk::kEmitChunkBlocks;
        TORCH_CHECK(
            dump_max_windows >= required_windows,
            "dump_max_windows must cover the full KV range");
        const int64_t regions =
            static_cast<int64_t>(num_q_blocks) * 8 * dump_max_windows;
        TORCH_CHECK(
            slab.numel() >= regions * region_words &&
                counts.numel() >= regions * 32,
            "dump slabs are smaller than the deterministic region layout");
    }
    // seq words + shadow words + pending/stale/last rows + per-lane
    // progress cursors, appended after the re-enabled histogram region.
    constexpr int kRingScratchBytes =
        2 * 8 * kImplBlockQ * static_cast<int>(sizeof(uint32_t)) +
        3 * kImplBlockQ * static_cast<int>(sizeof(int32_t)) +
        8 * kImplBlockQ * 32;
    // The production FP8 bucket-gate/no-hist scan always uses the qualified
    // warm-started ring daemon. Exact-gate and histogram modes retain their
    // dedicated non-ring specializations; the separate FP4 graft is not
    // launched through this function.
    const int smem = compute_smem_bytes_t<kImplHeads, kImplBlockQ>(
                         !exact_gate) +
        (no_hist && !exact_gate ? kRingScratchBytes : 0);
    auto kernel = raw_dump
        ? &dsa_litetopk::sm100_dsa_litetopk<
              kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
              false, true, true, false, false, false, true, true>
        : exact_gate
        ? &dsa_litetopk::sm100_dsa_litetopk<
              kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
              false, true, true, true>
        : no_hist
            ? &dsa_litetopk::sm100_dsa_litetopk<
              kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
              false, true, true, false, false, false, true>
            : &dsa_litetopk::sm100_dsa_litetopk<
              kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
              false, true, false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
        origin.data_ptr<float>(),
        inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        // The ring always warm-starts from the seed histogram. This is sound
        // under the production exact-once prefix/suffix split; non-ring
        // specializations keep their existing bcount contract.
        bcount.data_ptr<int32_t>(),
        static_cast<uint32_t>(num_buckets),
        static_cast<uint32_t>(topk),
        1u,
        1u,
        0u,
        0ULL,
        0u,
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        static_cast<uint32_t>(cand_cap),
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u,
        nullptr, nullptr,
        raw_dump
            ? reinterpret_cast<uint32_t*>(dump_slab->data_ptr<int>())
            : nullptr,
        raw_dump
            ? reinterpret_cast<uint32_t*>(dump_counts->data_ptr<int>())
            : nullptr,
        static_cast<uint32_t>(raw_dump ? dump_max_windows : 0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void mqa_logits_dsa_static_hot_litetopk_impl(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64,
        bool no_hist,
        bool exact_gate) {
    TORCH_CHECK(q.dim() == 3, "q must be [Q,H,128]");
    const int nh = static_cast<int>(q.size(1));
    if (nh == 64) {
        mqa_logits_dsa_static_hot_litetopk_impl_t<64, 2>(
            q, kv, kv_scales, weights, cu_start, cu_end,
            origin, inv_delta, th_bucket,
            cand_val, cand_idx, cand_cnt, bcount,
            num_buckets64, topk64, no_hist, exact_gate);
    } else {
        mqa_logits_dsa_static_hot_litetopk_impl_t<32, 4>(
            q, kv, kv_scales, weights, cu_start, cu_end,
            origin, inv_delta, th_bucket,
            cand_val, cand_idx, cand_cnt, bcount,
            num_buckets64, topk64, no_hist, exact_gate);
    }
}

void mqa_logits_dsa_static_hot_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64) {
    mqa_logits_dsa_static_hot_litetopk_impl(
        q, kv, kv_scales, weights, cu_start, cu_end,
        origin, inv_delta, th_bucket,
        cand_val, cand_idx, cand_cnt, bcount,
        num_buckets64, topk64, false, false);
}

// LITETOPK_STATIC_HOT_NOHIST_AB scratch entry: same fixed HOT gate and
// candidate ABI, but candidate metadata is finalized by a separate M-only
// pass instead of atomics in the S-wide score scan.
void mqa_logits_dsa_static_hot_nohist_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64) {
    mqa_logits_dsa_static_hot_litetopk_impl(
        q, kv, kv_scales, weights, cu_start, cu_end,
        origin, inv_delta, th_bucket,
        cand_val, cand_idx, cand_cnt, bcount,
        num_buckets64, topk64, true, false);
}

void mqa_logits_dsa_static_hot_nohist_paged_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor inline_val,
        torch::Tensor inline_idx,
        torch::Tensor overflow_val,
        torch::Tensor overflow_idx,
        torch::Tensor page_table,
        torch::Tensor pool_head,
        torch::Tensor pool_status,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t logical_cap64,
        int64_t num_buckets64,
        int64_t topk64) {
    const PagedCandidateShape shape = check_paged_candidate_layout(
        inline_val, inline_idx, overflow_val, overflow_idx,
        page_table, logical_cap64);
    TORCH_CHECK(
        q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
            weights.is_cuda() && cu_start.is_cuda() && cu_end.is_cuda() &&
            origin.is_cuda() && inv_delta.is_cuda() &&
            th_bucket.is_cuda() && pool_head.is_cuda() &&
            pool_status.is_cuda() && cand_cnt.is_cuda() && bcount.is_cuda(),
        "paged static-HOT scan tensors must be CUDA");
    TORCH_CHECK(
        q.device() == kv.device() && q.device() == kv_scales.device() &&
            q.device() == weights.device() && q.device() == cu_start.device() &&
            q.device() == cu_end.device() && q.device() == origin.device() &&
            q.device() == inv_delta.device() &&
            q.device() == th_bucket.device() &&
            q.device() == inline_val.device() &&
            q.device() == pool_head.device() &&
            q.device() == pool_status.device() &&
            q.device() == cand_cnt.device() && q.device() == bcount.device(),
        "paged static-HOT scan tensors must be on one CUDA device");
    TORCH_CHECK(
        q.is_contiguous() && kv.is_contiguous() &&
            kv_scales.is_contiguous() && weights.is_contiguous() &&
            cu_start.is_contiguous() && cu_end.is_contiguous() &&
            origin.is_contiguous() && inv_delta.is_contiguous() &&
            th_bucket.is_contiguous() && pool_head.is_contiguous() &&
            pool_status.is_contiguous() && cand_cnt.is_contiguous() &&
            bcount.is_contiguous(),
        "paged static-HOT scan tensors must be contiguous");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat8_e4m3fn &&
            kv.scalar_type() == torch::kFloat8_e4m3fn,
        "q/kv must be fp8_e4m3fn");
    TORCH_CHECK(
        kv_scales.scalar_type() == torch::kFloat &&
            weights.scalar_type() == torch::kFloat &&
            origin.scalar_type() == torch::kFloat &&
            inv_delta.scalar_type() == torch::kFloat,
        "kv_scales/weights/origin/inv_delta must be fp32");
    TORCH_CHECK(
        cu_start.scalar_type() == torch::kInt &&
            cu_end.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            pool_head.scalar_type() == torch::kInt &&
            pool_status.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            bcount.scalar_type() == torch::kInt,
        "paged scan range, allocator, count, and metadata must be int32");
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 2,
                "q/kv must be [Q,32,128] and [S,128]");
    const int seq_len = static_cast<int>(q.size(0));
    const int seq_len_kv = static_cast<int>(kv.size(0));
    TORCH_CHECK(
        seq_len > 0 && seq_len == shape.rows && seq_len_kv > 0,
        "paged candidate rows must equal Q and Q/S must be nonzero");
    TORCH_CHECK(
        q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM &&
            kv.size(1) == HEAD_DIM,
        "paged static-HOT path requires GLM DSA H=32,D=128");
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    TORCH_CHECK(
        weights.dim() == 2 && weights.size(0) == seq_len &&
            weights.size(1) == NUM_HEADS,
        "weights must be [Q,32]");
    TORCH_CHECK(
        kv_scales.dim() == 1 &&
            cu_start.dim() == 1 && cu_start.numel() == seq_len &&
            cu_end.dim() == 1 && cu_end.numel() == seq_len,
        "kv_scales must be 1-D and cu_start/cu_end must have Q elements");
    TORCH_CHECK(
        origin.dim() == 1 && origin.numel() == seq_len &&
            inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
            th_bucket.dim() == 1 && th_bucket.numel() == seq_len &&
            cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len,
        "origin/inv_delta/th_bucket/cand_cnt must have Q elements");
    TORCH_CHECK(
        bcount.dim() == 2 && bcount.size(0) == seq_len &&
            bcount.size(1) == num_buckets64,
        "bcount must be [Q,num_buckets]");
    TORCH_CHECK(
        pool_head.numel() == 1 && pool_status.numel() == 1,
        "pool_head/pool_status must be one-element int32 tensors");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(pool_head, pool_status) &&
            !tensor_storage_ranges_overlap(pool_head, inline_val) &&
            !tensor_storage_ranges_overlap(pool_head, inline_idx) &&
            !tensor_storage_ranges_overlap(pool_head, overflow_val) &&
            !tensor_storage_ranges_overlap(pool_head, overflow_idx) &&
            !tensor_storage_ranges_overlap(pool_head, page_table) &&
            !tensor_storage_ranges_overlap(pool_head, cand_cnt) &&
            !tensor_storage_ranges_overlap(pool_status, inline_val) &&
            !tensor_storage_ranges_overlap(pool_status, inline_idx) &&
            !tensor_storage_ranges_overlap(pool_status, overflow_val) &&
            !tensor_storage_ranges_overlap(pool_status, overflow_idx) &&
            !tensor_storage_ranges_overlap(pool_status, page_table) &&
            !tensor_storage_ranges_overlap(pool_status, cand_cnt),
        "pool_head/pool_status must own disjoint allocator storage");
    TORCH_CHECK(
        shape.logical_cap >= kPagedCandidateInline &&
            shape.overflow_pages > 0 && shape.table_pages_per_row > 0,
        "paged scan requires logical_cap>8192 and a nonempty overflow pool/table");
    TORCH_CHECK(
        num_buckets64 >= 3 && num_buckets64 <= 256,
        "paged static-HOT path requires 3 <= num_buckets <= 256");
    TORCH_CHECK(
        topk64 >= 1 && topk64 <= logical_cap64,
        "topk must be in [1,logical_cap]");

    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const int esz_fp8 = 1;
    const int esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    TORCH_CHECK(
        kv_scales.numel() >= ks_aligned,
        "kv_scales storage is shorter than the aligned KV length");

    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        page_table.data_ptr<int32_t>(), 0xff,
        static_cast<size_t>(page_table.numel()) * sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        pool_head.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        pool_status.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    auto tm_q = make_2d(
        q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
        BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(
        kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len_kv, HEAD_DIM,
        BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(
        kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(
        weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        NUM_HEADS, seq_len, NUM_HEADS,
        BLOCK_Q, NUM_HEADS, 0);

    const int smem = compute_smem_bytes(false);
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
        SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
        false, true, true, false, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    const int num_q_blocks =
        (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
        origin.data_ptr<float>(),
        inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        static_cast<uint32_t>(num_buckets),
        static_cast<uint32_t>(topk),
        1u,
        1u,
        0u,
        0ULL,
        0u,
        candidate_data_ptr(inline_val),
        inline_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        static_cast<uint32_t>(shape.logical_cap),
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        candidate_data_ptr(overflow_val),
        overflow_idx.data_ptr<int32_t>(),
        page_table.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(pool_head.data_ptr<int32_t>()),
        pool_status.data_ptr<int32_t>(),
        static_cast<uint32_t>(shape.overflow_pages),
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u,
        nullptr, nullptr, nullptr, nullptr, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mqa_logits_dsa_static_hot_nohist_segmented_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        torch::Tensor seed_cnt,
        torch::Tensor total_cnt,
        torch::Tensor bcount,
        torch::Tensor segment_val,
        torch::Tensor segment_idx,
        torch::Tensor segment_desc,
        torch::Tensor spill_desc,
        torch::Tensor row_spill,
        torch::Tensor segment_pool_head,
        torch::Tensor writer_status,
        int64_t num_buckets64,
        int64_t topk64) {
    const SegmentCandidateShape shape = check_segment_candidate_layout(
        seed_val, seed_idx, segment_val, segment_idx, segment_desc);
    check_segment_spill_layout(
        spill_desc, row_spill, segment_desc, shape);
    TORCH_CHECK(
        q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
            weights.is_cuda() && cu_start.is_cuda() && cu_end.is_cuda() &&
            origin.is_cuda() && inv_delta.is_cuda() &&
            th_bucket.is_cuda() && seed_cnt.is_cuda() &&
            total_cnt.is_cuda() && bcount.is_cuda() &&
            segment_pool_head.is_cuda() && writer_status.is_cuda(),
        "segmented static-HOT scan tensors must be CUDA");
    TORCH_CHECK(
        q.device() == kv.device() && q.device() == kv_scales.device() &&
            q.device() == weights.device() && q.device() == cu_start.device() &&
            q.device() == cu_end.device() && q.device() == origin.device() &&
            q.device() == inv_delta.device() &&
            q.device() == th_bucket.device() &&
            q.device() == seed_val.device() &&
            q.device() == seed_cnt.device() &&
            q.device() == total_cnt.device() &&
            q.device() == bcount.device() &&
            q.device() == segment_pool_head.device() &&
            q.device() == writer_status.device(),
        "segmented static-HOT scan tensors must be on one CUDA device");
    TORCH_CHECK(
        q.is_contiguous() && kv.is_contiguous() &&
            kv_scales.is_contiguous() && weights.is_contiguous() &&
            cu_start.is_contiguous() && cu_end.is_contiguous() &&
            origin.is_contiguous() && inv_delta.is_contiguous() &&
            th_bucket.is_contiguous() && seed_cnt.is_contiguous() &&
            total_cnt.is_contiguous() && bcount.is_contiguous() &&
            segment_pool_head.is_contiguous() && writer_status.is_contiguous(),
        "segmented static-HOT scan tensors must be contiguous");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat8_e4m3fn &&
            kv.scalar_type() == torch::kFloat8_e4m3fn,
        "q/kv must be fp8_e4m3fn");
    TORCH_CHECK(
        kv_scales.scalar_type() == torch::kFloat &&
            weights.scalar_type() == torch::kFloat &&
            origin.scalar_type() == torch::kFloat &&
            inv_delta.scalar_type() == torch::kFloat,
        "kv_scales/weights/origin/inv_delta must be fp32");
    TORCH_CHECK(
        cu_start.scalar_type() == torch::kInt &&
            cu_end.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            seed_cnt.scalar_type() == torch::kInt &&
            total_cnt.scalar_type() == torch::kInt &&
            bcount.scalar_type() == torch::kInt &&
            segment_pool_head.scalar_type() == torch::kInt &&
            writer_status.scalar_type() == torch::kInt,
        "segmented scan range/count/allocator metadata must be int32");
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 2,
                "q/kv must be [Q,32,128] and [S,128]");
    const int seq_len = static_cast<int>(q.size(0));
    const int seq_len_kv = static_cast<int>(kv.size(0));
    TORCH_CHECK(
        seq_len > 0 && seq_len == shape.rows && seq_len_kv > 0 &&
            q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM &&
            kv.size(1) == HEAD_DIM,
        "segmented static-HOT path requires Q rows and GLM DSA H=32,D=128");
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    const int required_windows =
        (seq_len_kv + kSegmentCandidatePage * 16 - 1) /
        (kSegmentCandidatePage * 16);
    TORCH_CHECK(
        shape.desc_windows == required_windows,
        "segment_desc W must equal ceil(S/65536)");
    TORCH_CHECK(
        weights.dim() == 2 && weights.size(0) == seq_len &&
            weights.size(1) == NUM_HEADS &&
            kv_scales.dim() == 1 &&
            cu_start.dim() == 1 && cu_start.numel() == seq_len &&
            cu_end.dim() == 1 && cu_end.numel() == seq_len,
        "weights/ranges must match Q and kv_scales must be 1-D");
    TORCH_CHECK(
        origin.dim() == 1 && origin.numel() == seq_len &&
            inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
            th_bucket.dim() == 1 && th_bucket.numel() == seq_len &&
            seed_cnt.dim() == 1 && seed_cnt.numel() == seq_len &&
            total_cnt.dim() == 1 && total_cnt.numel() == seq_len,
        "origin/inv_delta/threshold/seed_cnt/total_cnt must have Q elements");
    TORCH_CHECK(
        bcount.dim() == 2 && bcount.size(0) == seq_len &&
            bcount.size(1) == num_buckets64,
        "bcount must be [Q,num_buckets]");
    TORCH_CHECK(
        segment_pool_head.numel() == 1 && writer_status.numel() == 1,
        "segment_pool_head/writer_status must be scalar int32 tensors");
    TORCH_CHECK(
        num_buckets64 >= 3 && num_buckets64 <= 256 &&
            topk64 >= 1 && topk64 <= seq_len_kv,
        "segmented scan requires 3<=num_buckets<=256 and 1<=topk<=S");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(seed_cnt, total_cnt) &&
            !tensor_storage_ranges_overlap(segment_pool_head, writer_status) &&
            !tensor_storage_ranges_overlap(segment_pool_head, segment_desc) &&
            !tensor_storage_ranges_overlap(writer_status, segment_desc) &&
            !tensor_storage_ranges_overlap(total_cnt, segment_desc),
        "segmented counts, allocator, status, and descriptors must not alias");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(spill_desc, seed_val) &&
            !tensor_storage_ranges_overlap(spill_desc, seed_idx) &&
            !tensor_storage_ranges_overlap(spill_desc, segment_val) &&
            !tensor_storage_ranges_overlap(spill_desc, segment_idx) &&
            !tensor_storage_ranges_overlap(spill_desc, segment_pool_head) &&
            !tensor_storage_ranges_overlap(spill_desc, writer_status) &&
            !tensor_storage_ranges_overlap(spill_desc, total_cnt) &&
            !tensor_storage_ranges_overlap(row_spill, seed_val) &&
            !tensor_storage_ranges_overlap(row_spill, seed_idx) &&
            !tensor_storage_ranges_overlap(row_spill, segment_val) &&
            !tensor_storage_ranges_overlap(row_spill, segment_idx) &&
            !tensor_storage_ranges_overlap(row_spill, segment_pool_head) &&
            !tensor_storage_ranges_overlap(row_spill, writer_status) &&
            !tensor_storage_ranges_overlap(row_spill, total_cnt),
        "spill descriptors/row flags must not alias scan storage");

    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const int esz_fp8 = 1;
    const int esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    TORCH_CHECK(
        kv_scales.numel() >= ks_aligned,
        "kv_scales storage is shorter than the aligned KV length");
    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        segment_desc.data_ptr<int64_t>(), 0,
        static_cast<size_t>(segment_desc.numel()) * sizeof(int64_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        spill_desc.data_ptr<int64_t>(), 0,
        static_cast<size_t>(spill_desc.numel()) * sizeof(int64_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        row_spill.data_ptr<int32_t>(), 0,
        static_cast<size_t>(row_spill.numel()) * sizeof(int32_t),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        segment_pool_head.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        writer_status.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    auto tm_q = make_2d(
        q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
        BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(
        kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len_kv, HEAD_DIM,
        BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(
        kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(
        weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        NUM_HEADS, seq_len, NUM_HEADS,
        BLOCK_Q, NUM_HEADS, 0);
    const int smem = compute_smem_bytes(false) + 384;
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
        SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128,
        false, true, true, false, false, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const int num_q_blocks =
        (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
        static_cast<uint32_t>(num_buckets),
        static_cast<uint32_t>(topk), 1u, 1u, 0u, 0ULL, 0u,
        candidate_data_ptr(seed_val), seed_idx.data_ptr<int32_t>(),
        total_cnt.data_ptr<int32_t>(),
        static_cast<uint32_t>(kPagedCandidateInline),
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
        candidate_data_ptr(segment_val),
        segment_idx.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t*>(segment_desc.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(
            segment_pool_head.data_ptr<int32_t>()),
        writer_status.data_ptr<int32_t>(),
        static_cast<uint32_t>(shape.pool_pages),
        static_cast<uint32_t>(shape.desc_windows),
        reinterpret_cast<uint64_t*>(spill_desc.data_ptr<int64_t>()),
        row_spill.data_ptr<int32_t>(), nullptr, nullptr, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Continuation after an exact intermediate compaction. th_bucket carries the
// ordered-FP32 pivot edge, not a coarse bucket. The scan emits only later
// records in a strictly better high24 class and performs no histogram update.
void mqa_logits_dsa_static_hot_exactgate_nohist_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64) {
    mqa_logits_dsa_static_hot_litetopk_impl(
        q, kv, kv_scales, weights, cu_start, cu_end,
        origin, inv_delta, th_bucket,
        cand_val, cand_idx, cand_cnt, bcount,
        num_buckets64, topk64, true, true);
}

// Qualified GLM production tail: h2048 physical selector followed by the
// always-launched exact overflow continuation.  The continuation has K=2048
// in both its output stride and radix target; candidate capacity is read from
// the tensor stride so production outliers can use the qualified 196608-row
// slab instead of trapping at the old 65536-record integration limit.
// Python reuses the now-dead boundary_meta allocation as a compact R*5-int
// diagnostic scratch; the kernel never assumes its original row stride.
void h2048_safe_topk_out_litetopk_(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor out_idx,
        torch::Tensor status,
        torch::Tensor diagnostic_scratch,
        int64_t index_limit64) {
    constexpr int kDiagnosticIntsPerRow = 5;
    static_assert(
        h2048_safe_topk::kBins >= kDiagnosticIntsPerRow,
        "boundary_meta must have room for h2048 diagnostic scratch");
    static_assert(
        dsa_litetopk::kCandidateIndexBits == 20,
        "h2048 safe selector requires the production 20-bit physical ID ABI");
    static_assert(sizeof(CandidateValue) == sizeof(uint16_t));

    TORCH_CHECK(
        cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
            out_idx.is_cuda() && status.is_cuda() &&
            diagnostic_scratch.is_cuda(),
        "h2048 safe selector tensors must be CUDA");
    TORCH_CHECK(
        cand_val.device() == cand_idx.device() &&
            cand_val.device() == cand_cnt.device() &&
            cand_val.device() == out_idx.device() &&
            cand_val.device() == status.device() &&
            cand_val.device() == diagnostic_scratch.device(),
        "h2048 safe selector tensors must be on one CUDA device");
    TORCH_CHECK(
        cand_val.is_contiguous() && cand_idx.is_contiguous() &&
            cand_cnt.is_contiguous() && out_idx.is_contiguous() &&
            status.is_contiguous() && diagnostic_scratch.is_contiguous(),
        "h2048 safe selector tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            out_idx.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt &&
            diagnostic_scratch.scalar_type() == torch::kInt,
        "h2048 safe selector metadata and output must be int32");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
        "h2048 candidate tensors must be [R,CAP]");
    const int64_t rows64 = cand_val.size(0);
    // Capacity floor scales with the selection width: 32x topk matches the
    // historical 65536 floor at K=2048 and admits 16384 at K=512.
    const int64_t min_cap64 = std::max<int64_t>(
        16384, out_idx.dim() == 2 ? 32 * out_idx.size(1) : h2048_safe_topk::kMinCap);
    TORCH_CHECK(
        rows64 > 0 && rows64 <= std::numeric_limits<int>::max() &&
            cand_val.size(1) >= min_cap64 &&
            cand_val.size(1) <= h2048_safe_topk::kMaxCap,
        "h2048 safe selector requires [R,CAP] candidates with "
        "max(16384, 32*topk) <= CAP <= 1M");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == rows64 &&
            status.dim() == 1 && status.numel() == rows64,
        "h2048 cand_cnt/status must have R elements");
    TORCH_CHECK(
        out_idx.dim() == 2 && out_idx.size(0) == rows64 &&
            out_idx.size(1) >= 1 &&
            out_idx.size(1) <= h2048_safe_topk::overflow::kTopK &&
            out_idx.size(1) <= cand_val.size(1),
        "h2048 safe selector output must be [R,topk<=2048]");
    TORCH_CHECK(
        diagnostic_scratch.numel() >=
            rows64 * kDiagnosticIntsPerRow,
        "h2048 diagnostic scratch must contain at least R*5 int32 values");
    TORCH_CHECK(
        index_limit64 > 0 &&
            index_limit64 <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "h2048 index_limit must be in [1,1M]");

    const int rows = static_cast<int>(rows64);
    const int cap = static_cast<int>(cand_val.size(1));
    const int topk = static_cast<int>(out_idx.size(1));
    const int index_limit = static_cast<int>(index_limit64);
    const c10::cuda::CUDAGuard device_guard(cand_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    h2048_safe_topk::coarse_tiering_topk_kernel<256, 512, 1, 8><<<
        rows, 256, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(
            cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        out_idx.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        diagnostic_scratch.data_ptr<int32_t>(),
        rows,
        cap,
        topk,
        index_limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // No-overflow rows return before the fallback's first CTA barrier.  An
    // overflow row is selected exactly in physical space and clears bit 32;
    // all non-overflow errors remain nonzero for the following map to trap.
    h2048_safe_topk::overflow::overflow_exact_topk_kernel<<<
        rows, h2048_safe_topk::overflow::kThreads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(
            cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        out_idx.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        rows,
        cap,
        index_limit,
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize_static_hot_meta_litetopk_(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        torch::Tensor status,
        int64_t num_buckets64,
        int64_t topk64,
        int64_t index_limit64) {
    TORCH_CHECK(
        cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
            th_bucket.is_cuda() && boundary_meta.is_cuda() && status.is_cuda(),
        "static HOT finalize tensors must be CUDA");
    TORCH_CHECK(
        cand_val.device() == cand_idx.device() &&
            cand_val.device() == cand_cnt.device() &&
            cand_val.device() == th_bucket.device() &&
            cand_val.device() == boundary_meta.device() &&
            cand_val.device() == status.device(),
        "static HOT finalize tensors must be on one CUDA device");
    TORCH_CHECK(
        cand_val.is_contiguous() && cand_idx.is_contiguous() &&
            cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
            boundary_meta.is_contiguous() && status.is_contiguous(),
        "static HOT finalize tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt,
        "static HOT finalize metadata/indices must be int32");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [R,cand_cap]");
    const int rows = static_cast<int>(cand_val.size(0));
    const int cand_cap = static_cast<int>(cand_val.size(1));
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    TORCH_CHECK(
        index_limit64 >= 0 &&
            index_limit64 <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "index_limit must be in [0, 1M]");
    const int index_limit = static_cast<int>(index_limit64);
    TORCH_CHECK(rows > 0 && cand_cap > 0,
                "candidate slab must be nonempty");
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
                "finalize requires 3 <= num_buckets <= 256");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap,
                "topk must be in [1,cand_cap]");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == rows &&
            status.dim() == 1 && status.numel() == rows,
        "cand_cnt/th_bucket/status must have R elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 && boundary_meta.size(0) == rows &&
            boundary_meta.size(1) == num_buckets,
        "boundary_meta must be [R,num_buckets]");

    c10::cuda::CUDAGuard device_guard(cand_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    finalize_static_hot_meta_litetopk_kernel<<<
        rows, 256, 0, stream>>>(
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        index_limit,
        rows,
        cand_cap,
        num_buckets,
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_static_hot_candidates_litetopk_(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        torch::Tensor status,
        int64_t num_buckets64,
        int64_t topk64) {
    TORCH_CHECK(
        cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
            th_bucket.is_cuda() && boundary_meta.is_cuda() && status.is_cuda(),
        "static HOT compact tensors must be CUDA");
    TORCH_CHECK(
        cand_val.device() == cand_idx.device() &&
            cand_val.device() == cand_cnt.device() &&
            cand_val.device() == th_bucket.device() &&
            cand_val.device() == boundary_meta.device() &&
            cand_val.device() == status.device(),
        "static HOT compact tensors must be on one CUDA device");
    TORCH_CHECK(
        cand_val.is_contiguous() && cand_idx.is_contiguous() &&
            cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
            boundary_meta.is_contiguous() && status.is_contiguous(),
        "static HOT compact tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt,
        "static HOT compact metadata/indices must be int32");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [R,cand_cap]");
    TORCH_CHECK(
        cand_val.size(0) <= std::numeric_limits<int>::max() &&
            cand_val.size(1) < std::numeric_limits<int>::max(),
        "candidate slab dimensions must leave room for an int32 overflow "
        "sentinel");
    TORCH_CHECK(num_buckets64 >= 3 && num_buckets64 <= 256,
                "compact requires 3 <= num_buckets <= 256");
    TORCH_CHECK(topk64 >= 1 && topk64 <= cand_val.size(1),
                "topk must be in [1,cand_cap]");
    const int rows = static_cast<int>(cand_val.size(0));
    const int cand_cap = static_cast<int>(cand_val.size(1));
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    TORCH_CHECK(rows > 0 && cand_cap > 0,
                "candidate slab must be nonempty");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == rows &&
            status.dim() == 1 && status.numel() == rows,
        "cand_cnt/th_bucket/status must have R elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 && boundary_meta.size(0) == rows &&
            boundary_meta.size(1) == num_buckets,
        "boundary_meta must be [R,num_buckets]");

    c10::cuda::CUDAGuard device_guard(cand_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compact_static_hot_candidates_litetopk_kernel<<<
        rows, 256, 0, stream>>>(
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        rows,
        cand_cap,
        num_buckets,
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize_static_hot_meta_mapped_paged_litetopk_(
        torch::Tensor inline_val,
        torch::Tensor inline_idx,
        torch::Tensor overflow_val,
        torch::Tensor overflow_idx,
        torch::Tensor page_table,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        torch::Tensor status,
        torch::Tensor index_map,
        int64_t logical_cap64,
        int64_t num_buckets64,
        int64_t topk64) {
    const PagedCandidateShape shape = check_paged_candidate_layout(
        inline_val, inline_idx, overflow_val, overflow_idx,
        page_table, logical_cap64);
    TORCH_CHECK(
        cand_cnt.is_cuda() && th_bucket.is_cuda() &&
            boundary_meta.is_cuda() && status.is_cuda() &&
            index_map.is_cuda(),
        "paged mapped-finalizer metadata must be CUDA");
    TORCH_CHECK(
        inline_val.device() == cand_cnt.device() &&
            inline_val.device() == th_bucket.device() &&
            inline_val.device() == boundary_meta.device() &&
            inline_val.device() == status.device() &&
            inline_val.device() == index_map.device(),
        "paged mapped-finalizer tensors must be on one CUDA device");
    TORCH_CHECK(
        cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
            boundary_meta.is_contiguous() && status.is_contiguous() &&
            index_map.is_contiguous(),
        "paged mapped-finalizer metadata must be contiguous");
    TORCH_CHECK(
        cand_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt &&
            index_map.scalar_type() == torch::kInt,
        "paged mapped-finalizer metadata must be int32");
    TORCH_CHECK(
        num_buckets64 >= 3 && num_buckets64 <= 256,
        "paged mapped finalizer requires 3 <= num_buckets <= 256");
    TORCH_CHECK(
        topk64 >= 1 && topk64 <= logical_cap64,
        "topk must be in [1,logical_cap]");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == shape.rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == shape.rows &&
            status.dim() == 1 && status.numel() == shape.rows,
        "cand_cnt/th_bucket/status must have Q elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 &&
            boundary_meta.size(0) == shape.rows &&
            boundary_meta.size(1) == num_buckets64,
        "boundary_meta must be [Q,num_buckets]");
    TORCH_CHECK(
        index_map.dim() == 1 && index_map.numel() > 0 &&
            index_map.numel() <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "index_map must be a nonempty int32 vector with at most 1M entries");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(inline_val, inline_idx) &&
            !tensor_storage_ranges_overlap(overflow_val, overflow_idx) &&
            !tensor_storage_ranges_overlap(inline_idx, overflow_idx) &&
            !tensor_storage_ranges_overlap(inline_idx, page_table) &&
            !tensor_storage_ranges_overlap(overflow_idx, page_table) &&
            !tensor_storage_ranges_overlap(inline_idx, index_map) &&
            !tensor_storage_ranges_overlap(overflow_idx, index_map) &&
            !tensor_storage_ranges_overlap(inline_idx, cand_cnt) &&
            !tensor_storage_ranges_overlap(inline_idx, th_bucket) &&
            !tensor_storage_ranges_overlap(inline_idx, boundary_meta) &&
            !tensor_storage_ranges_overlap(inline_idx, status) &&
            !tensor_storage_ranges_overlap(overflow_idx, cand_cnt) &&
            !tensor_storage_ranges_overlap(overflow_idx, th_bucket) &&
            !tensor_storage_ranges_overlap(overflow_idx, boundary_meta) &&
            !tensor_storage_ranges_overlap(overflow_idx, status),
        "paged mapped-finalizer writable candidate storage must not alias "
        "values, page_table, index_map, or another candidate arena");

    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const int index_map_size = static_cast<int>(index_map.numel());
    c10::cuda::CUDAGuard device_guard(inline_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    finalize_static_hot_meta_mapped_paged_litetopk_kernel<<<
        shape.rows, 256, 0, stream>>>(
        candidate_data_ptr(inline_val),
        inline_idx.data_ptr<int32_t>(),
        candidate_data_ptr(overflow_val),
        overflow_idx.data_ptr<int32_t>(),
        page_table.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        index_map.data_ptr<int32_t>(),
        index_map_size,
        shape.rows,
        shape.overflow_pages,
        shape.table_pages_per_row,
        shape.logical_cap,
        num_buckets,
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize_static_hot_meta_mapped_segmented_litetopk_(
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        torch::Tensor seed_cnt,
        torch::Tensor segment_val,
        torch::Tensor segment_idx,
        torch::Tensor segment_desc,
        torch::Tensor spill_desc,
        torch::Tensor row_spill,
        torch::Tensor segment_pool_head,
        torch::Tensor writer_status,
        torch::Tensor total_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        torch::Tensor status,
        torch::Tensor index_map,
        int64_t logical_cap64,
        int64_t num_buckets64,
        int64_t topk64) {
    const SegmentCandidateShape shape = check_segment_candidate_layout(
        seed_val, seed_idx, segment_val, segment_idx, segment_desc);
    check_segment_spill_layout(
        spill_desc, row_spill, segment_desc, shape);
    TORCH_CHECK(
        seed_cnt.is_cuda() && segment_pool_head.is_cuda() &&
            writer_status.is_cuda() && total_cnt.is_cuda() &&
            th_bucket.is_cuda() && boundary_meta.is_cuda() &&
            status.is_cuda() && index_map.is_cuda(),
        "segmented finalizer metadata must be CUDA");
    TORCH_CHECK(
        seed_val.device() == seed_cnt.device() &&
            seed_val.device() == segment_pool_head.device() &&
            seed_val.device() == writer_status.device() &&
            seed_val.device() == total_cnt.device() &&
            seed_val.device() == th_bucket.device() &&
            seed_val.device() == boundary_meta.device() &&
            seed_val.device() == status.device() &&
            seed_val.device() == index_map.device(),
        "segmented finalizer tensors must be on one CUDA device");
    TORCH_CHECK(
        seed_cnt.is_contiguous() && segment_pool_head.is_contiguous() &&
            writer_status.is_contiguous() && total_cnt.is_contiguous() &&
            th_bucket.is_contiguous() && boundary_meta.is_contiguous() &&
            status.is_contiguous() && index_map.is_contiguous(),
        "segmented finalizer metadata must be contiguous");
    TORCH_CHECK(
        seed_cnt.scalar_type() == torch::kInt &&
            segment_pool_head.scalar_type() == torch::kInt &&
            writer_status.scalar_type() == torch::kInt &&
            total_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt &&
            index_map.scalar_type() == torch::kInt,
        "segmented finalizer metadata must be int32");
    TORCH_CHECK(
        logical_cap64 >= 1 &&
            logical_cap64 < std::numeric_limits<int32_t>::max() &&
            num_buckets64 >= 3 && num_buckets64 <= 256 &&
            topk64 >= 1 && topk64 <= logical_cap64,
        "invalid segmented logical_cap/num_buckets/topk");
    TORCH_CHECK(
        seed_cnt.dim() == 1 && seed_cnt.numel() == shape.rows &&
            total_cnt.dim() == 1 && total_cnt.numel() == shape.rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == shape.rows &&
            status.dim() == 1 && status.numel() == shape.rows,
        "seed_cnt/total_cnt/th_bucket/status must have Q elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 &&
            boundary_meta.size(0) == shape.rows &&
            boundary_meta.size(1) == num_buckets64,
        "boundary_meta must be [Q,num_buckets]");
    TORCH_CHECK(
        segment_pool_head.numel() == 1 && writer_status.numel() == 1,
        "segment_pool_head/writer_status must be scalar int32 tensors");
    TORCH_CHECK(
        index_map.dim() == 1 && index_map.numel() > 0 &&
            index_map.numel() <=
                (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
        "index_map must be a nonempty int32 vector with at most 1M entries");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(seed_idx, segment_idx) &&
            !tensor_storage_ranges_overlap(seed_idx, index_map) &&
            !tensor_storage_ranges_overlap(segment_idx, index_map) &&
            !tensor_storage_ranges_overlap(seed_idx, spill_desc) &&
            !tensor_storage_ranges_overlap(segment_idx, spill_desc) &&
            !tensor_storage_ranges_overlap(index_map, spill_desc) &&
            !tensor_storage_ranges_overlap(total_cnt, seed_cnt) &&
            !tensor_storage_ranges_overlap(total_cnt, boundary_meta) &&
            !tensor_storage_ranges_overlap(status, boundary_meta),
        "segmented finalizer writable tensors must not alias inputs/metadata");

    c10::cuda::CUDAGuard device_guard(seed_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    finalize_static_hot_meta_mapped_segmented_litetopk_kernel<<<
        shape.rows, 256, 0, stream>>>(
        candidate_data_ptr(seed_val), seed_idx.data_ptr<int32_t>(),
        seed_cnt.data_ptr<int32_t>(), candidate_data_ptr(segment_val),
        segment_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(
            segment_desc.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(
            spill_desc.data_ptr<int64_t>()),
        row_spill.data_ptr<int32_t>(),
        segment_pool_head.data_ptr<int32_t>(),
        writer_status.data_ptr<int32_t>(),
        total_cnt.data_ptr<int32_t>(), th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
        index_map.data_ptr<int32_t>(),
        static_cast<int>(index_map.numel()), shape.rows,
        shape.desc_windows, shape.micro_records, shape.pool_pages,
        shape.physical_records, static_cast<int>(logical_cap64),
        static_cast<int>(num_buckets64), static_cast<int>(topk64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Offline A/B only: compute raw indexer scores and feed a fixed 256-bin online
// histogram directly from the GEMM epilogue.  Unlike the production entry
// point this path consumes no sample-derived origin/inv tensors and performs
// no allocations; the caller owns all candidate and selector state.
void mqa_logits_dsa_online_fixed_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        torch::Tensor th_bucket,
        int64_t topk64) {
    TORCH_CHECK(
        q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
            weights.is_cuda() && cu_start.is_cuda() && cu_end.is_cuda() &&
            cand_val.is_cuda() && cand_idx.is_cuda() &&
            cand_cnt.is_cuda() && bcount.is_cuda() &&
            th_bucket.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(
        q.device() == kv.device() && q.device() == kv_scales.device() &&
            q.device() == weights.device() && q.device() == cu_start.device() &&
            q.device() == cu_end.device() && q.device() == cand_val.device() &&
            q.device() == cand_idx.device() && q.device() == cand_cnt.device() &&
            q.device() == bcount.device() && q.device() == th_bucket.device(),
        "all tensors must be on the same CUDA device");
    TORCH_CHECK(
        q.is_contiguous() && kv.is_contiguous() &&
            kv_scales.is_contiguous() && weights.is_contiguous() &&
            cu_start.is_contiguous() && cu_end.is_contiguous() &&
            cand_val.is_contiguous() && cand_idx.is_contiguous() &&
            cand_cnt.is_contiguous() && bcount.is_contiguous() &&
            th_bucket.is_contiguous(),
        "all tensors must be contiguous");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat8_e4m3fn &&
            kv.scalar_type() == torch::kFloat8_e4m3fn,
        "q/kv must be fp8_e4m3fn");
    TORCH_CHECK(
        kv_scales.scalar_type() == torch::kFloat &&
            weights.scalar_type() == torch::kFloat,
        "kv_scales/weights must be fp32");
    TORCH_CHECK(
        cu_start.scalar_type() == torch::kInt &&
            cu_end.scalar_type() == torch::kInt,
        "cu_start/cu_end must be int32");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        cand_idx.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt &&
            bcount.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt,
        "cand_idx/cand_cnt/bcount/th_bucket must be int32");

    TORCH_CHECK(q.dim() == 3, "q must be [Q,32,128]");
    TORCH_CHECK(kv.dim() == 2, "kv must be [S,128]");
    const int seq_len = static_cast<int>(q.size(0));
    const int seq_len_kv = static_cast<int>(kv.size(0));
    TORCH_CHECK(seq_len > 0 && seq_len_kv > 0, "Q and S must be nonzero");
    TORCH_CHECK(
        q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM &&
            kv.size(1) == HEAD_DIM,
        "online fixed path requires GLM DSA H=32,D=128");
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    TORCH_CHECK(
        weights.dim() == 2 && weights.size(0) == seq_len &&
            weights.size(1) == NUM_HEADS,
        "weights must be [Q,32]");
    TORCH_CHECK(
        cu_start.dim() == 1 && cu_start.numel() == seq_len &&
            cu_end.dim() == 1 && cu_end.numel() == seq_len,
        "cu_start/cu_end must have Q elements");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_val.size(0) == seq_len &&
            cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [Q,cand_cap]");
    const int cand_cap = static_cast<int>(cand_val.size(1));
    const int topk = static_cast<int>(topk64);
    TORCH_CHECK(
        topk >= 1 && topk <= cand_cap,
        "topk must be in [1,cand_cap]");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len &&
            th_bucket.dim() == 1 && th_bucket.numel() == seq_len,
        "cand_cnt/th_bucket must have Q elements");
    TORCH_CHECK(
        bcount.dim() == 2 && bcount.size(0) == seq_len &&
            bcount.size(1) == ONLINE_FIXED_BUCKETS,
        "bcount must be [Q,256]");

    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    constexpr int kInitThreads = 256;
    const int init_blocks = (seq_len + kInitThreads - 1) / kInitThreads;
    online_fixed_init_litetopk_kernel<<<
        init_blocks, kInitThreads, 0, stream>>>(
        th_bucket.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int esz_fp8 = 1;
    const int esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    TORCH_CHECK(
        kv_scales.numel() >= ks_aligned,
        "kv_scales storage is shorter than the aligned KV length");
    auto tm_q = make_2d(
        q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
        BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(
        kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
        HEAD_DIM, seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(
        kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(
        weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
        NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

    const int smem = compute_smem_bytes();
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
        SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
        nullptr,
        nullptr,
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        static_cast<uint32_t>(ONLINE_FIXED_BUCKETS),
        static_cast<uint32_t>(topk),
        1u,
        1u,
        0u,
        0ULL,
        0u,
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        static_cast<uint32_t>(cand_cap),
        tm_q, tm_kv, tm_ks, tm_w, tm_q,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
        nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u,
        nullptr, nullptr, nullptr, nullptr, 0u);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cand_count_stats_litetopk_(
        torch::Tensor cand_cnt,
        torch::Tensor stats) {
    TORCH_CHECK(
        cand_cnt.is_cuda() && stats.is_cuda(),
        "cand_cnt/stats must be CUDA tensors");
    TORCH_CHECK(
        cand_cnt.is_contiguous() && stats.is_contiguous(),
        "cand_cnt/stats must be contiguous");
    TORCH_CHECK(
        cand_cnt.scalar_type() == torch::kInt &&
            stats.scalar_type() == torch::kInt,
        "cand_cnt/stats must be int32");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() > 0,
        "cand_cnt must be a non-empty 1-D tensor");
    TORCH_CHECK(
        cand_cnt.numel() <= std::numeric_limits<int32_t>::max(),
        "cand_cnt is too large for the single-CTA stats ABI");
    TORCH_CHECK(
        stats.dim() == 1 && stats.numel() == 2,
        "stats must be int32[2]");
    TORCH_CHECK(
        cand_cnt.device() == stats.device(),
        "cand_cnt/stats must be on the same CUDA device");

    cand_count_stats_litetopk_kernel<<<
        1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand_cnt.data_ptr<int32_t>(),
        static_cast<int>(cand_cnt.numel()),
        stats.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool FusePlanner>
void carry_votes_topk_reset_litetopk_impl_(
        torch::Tensor votes,
        torch::Tensor out_idx,
        torch::Tensor partial,
        torch::Tensor state,
        torch::Tensor permutation,
        torch::Tensor planner_state,
        int64_t k64,
        int64_t max_vote64,
        int64_t min_index64,
        int64_t window_start64) {
    TORCH_CHECK(
        votes.is_cuda() && out_idx.is_cuda() &&
            partial.is_cuda() && state.is_cuda(),
        "votes/out_idx/partial/state must be CUDA tensors");
    TORCH_CHECK(
        votes.is_contiguous() && out_idx.is_contiguous() &&
            partial.is_contiguous() && state.is_contiguous(),
        "votes/out_idx/partial/state must be contiguous");
    TORCH_CHECK(
        votes.scalar_type() == torch::kInt,
        "votes must be int32");
    TORCH_CHECK(
        out_idx.scalar_type() == torch::kLong,
        "out_idx must be int64");
    TORCH_CHECK(
        partial.scalar_type() == torch::kShort,
        "partial must be int16");
    TORCH_CHECK(
        state.scalar_type() == torch::kInt,
        "state must be int32");
    TORCH_CHECK(
        votes.device() == out_idx.device() &&
            votes.device() == partial.device() &&
            votes.device() == state.device(),
        "votes/out_idx/partial/state must be on the same CUDA device");
    TORCH_CHECK(
        votes.dim() == 1,
        "votes must be a 1-D histogram");
    TORCH_CHECK(
        out_idx.dim() == 1,
        "out_idx must be 1-D");
    TORCH_CHECK(
        partial.dim() == 2,
        "partial must be [blocks,bins]");
    TORCH_CHECK(
        state.dim() == 1 && state.numel() >= kCarryStateInts,
        "state is too small for the carry top-k ABI");

    if constexpr (FusePlanner) {
        TORCH_CHECK(
            permutation.is_cuda() && planner_state.is_cuda(),
            "permutation/planner_state must be CUDA tensors");
        TORCH_CHECK(
            permutation.is_contiguous() && planner_state.is_contiguous(),
            "permutation/planner_state must be contiguous");
        TORCH_CHECK(
            permutation.scalar_type() == torch::kInt &&
                planner_state.scalar_type() == torch::kInt,
            "permutation/planner_state must be int32");
        TORCH_CHECK(
            permutation.dim() == 1 &&
                planner_state.dim() == 1 &&
                planner_state.numel() >= kCarryPlannerStateInts,
            "invalid fused carry-planner workspace");
        TORCH_CHECK(
            votes.device() == permutation.device() &&
                votes.device() == planner_state.device(),
            "planner workspace must be on the carry CUDA device");
    }

    const int64_t count64 = votes.numel();
    TORCH_CHECK(
        count64 >= 1 && count64 <= kCarryMaxItems,
        "votes length must be in [1,1048576]");
    TORCH_CHECK(
        k64 >= 1 && k64 <= kCarryMaxK,
        "k must be in [1,12288]");
    TORCH_CHECK(
        max_vote64 >= 1 && max_vote64 <= kCarryMaxVote,
        "max_vote must be in [1,8192]");
    TORCH_CHECK(
        min_index64 >= 0 && min_index64 < count64,
        "min_index must be in [0,votes.numel())");
    const int count = static_cast<int>(count64);
    const int min_index = static_cast<int>(min_index64);
    const int eligible = count - min_index;
    const int out_k = static_cast<int>(min(k64, static_cast<int64_t>(eligible)));
    const int max_vote = static_cast<int>(max_vote64);
    const int bins = max_vote + 1;
    const int blocks =
        ((FusePlanner ? eligible : count) + kCarryTileItems - 1) /
        kCarryTileItems;
    if constexpr (FusePlanner) {
        TORCH_CHECK(
            out_k == kCarryPlannerWindow,
            "fused carry planner requires k=8192");
        TORCH_CHECK(
            window_start64 == min_index64,
            "fused carry planner requires window_start=min_index");
        TORCH_CHECK(
            min_index64 + kCarryPlannerWindow <= count64,
            "fused carry planner window exceeds the vote extent");
        TORCH_CHECK(
            permutation.numel() >= count64,
            "permutation must cover the complete vote extent");
        TORCH_CHECK(
            permutation.numel() <= std::numeric_limits<int>::max(),
            "permutation is too large for the int32 planner ABI");
        // The software grid barrier is qualified only for the <=128K exact
        // path: at most 16 of these 256-thread CTAs on fixed B200 hardware.
        TORCH_CHECK(
            blocks <= 16,
            "fused carry planner is restricted to at most 16 CTAs");
    }
    TORCH_CHECK(
        out_idx.numel() == out_k,
        "out_idx must have min(k,votes.numel()-min_index) elements");
    TORCH_CHECK(
        partial.size(0) >= blocks && partial.size(1) >= bins,
        "partial must provide at least [ceil(N/8192),max_vote+1]");

    const int partial_stride = static_cast<int>(partial.size(1));
    const size_t dynamic_smem =
        static_cast<size_t>(bins) * sizeof(uint32_t);
    const c10::cuda::CUDAGuard device_guard(votes.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    carry_votes_plan_litetopk_kernel<FusePlanner><<<
        blocks, kCarryThreads, dynamic_smem, stream>>>(
        votes.data_ptr<int32_t>(),
        count,
        min_index,
        out_k,
        max_vote,
        partial.data_ptr<int16_t>(),
        partial_stride,
        state.data_ptr<int32_t>(),
        FusePlanner ? permutation.data_ptr<int32_t>() : nullptr,
        FusePlanner ? planner_state.data_ptr<int32_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    carry_votes_emit_reset_litetopk_kernel<FusePlanner><<<
        blocks, kCarryThreads, 0, stream>>>(
        votes.data_ptr<int32_t>(),
        count,
        min_index,
        max_vote,
        out_idx.data_ptr<int64_t>(),
        state.data_ptr<int32_t>(),
        FusePlanner ? permutation.data_ptr<int32_t>() : nullptr,
        FusePlanner ? planner_state.data_ptr<int32_t>() : nullptr,
        static_cast<int>(window_start64),
        FusePlanner ? static_cast<int>(permutation.numel()) : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void carry_votes_topk_reset_litetopk_(
        torch::Tensor votes,
        torch::Tensor out_idx,
        torch::Tensor partial,
        torch::Tensor state,
        int64_t k64,
        int64_t max_vote64,
        int64_t min_index64) {
    carry_votes_topk_reset_litetopk_impl_<false>(
        votes, out_idx, partial, state, torch::Tensor(), torch::Tensor(),
        k64, max_vote64, min_index64, 0);
}

void map_topk_indices_and_accumulate_votes_litetopk_(
        torch::Tensor out_idx,
        torch::Tensor index_map,
        torch::Tensor status,
        torch::Tensor votes,
        int64_t vote_recent_rows64) {
    TORCH_CHECK(out_idx.is_cuda() && index_map.is_cuda() &&
                    status.is_cuda() && votes.is_cuda(),
                "out_idx/index_map/status/votes must be CUDA");
    TORCH_CHECK(out_idx.device() == index_map.device() &&
                    out_idx.device() == status.device() &&
                    out_idx.device() == votes.device(),
                "out_idx/index_map/status/votes must be on one CUDA device");
    TORCH_CHECK(out_idx.is_contiguous() && index_map.is_contiguous() &&
                    status.is_contiguous() && votes.is_contiguous(),
                "out_idx/index_map/status/votes must be contiguous");
    TORCH_CHECK(out_idx.scalar_type() == torch::kInt &&
                    index_map.scalar_type() == torch::kInt &&
                    status.scalar_type() == torch::kInt &&
                    votes.scalar_type() == torch::kInt,
                "out_idx/index_map/status/votes must be int32");
    TORCH_CHECK(out_idx.dim() == 2 && out_idx.numel() > 0,
                "out_idx must be a nonempty [R,K] matrix");
    TORCH_CHECK(index_map.dim() == 1 && index_map.numel() > 0 &&
                    index_map.numel() <=
                        (int64_t{1} <<
                         dsa_litetopk::kCandidateIndexBits),
                "index_map must be a nonempty vector with at most 1M entries");
    TORCH_CHECK(status.dim() == 1 && status.numel() == out_idx.size(0),
                "status must contain one value per output row");
    TORCH_CHECK(votes.dim() == 1,
                "votes must be a 1-D histogram (or empty)");
    TORCH_CHECK(
        vote_recent_rows64 > 0 && vote_recent_rows64 <= out_idx.size(0),
        "vote_recent_rows must be in [1,R]");

    constexpr int kThreads = 256;
    constexpr int kBlocksPerSm = 8;
    constexpr int kProductionSms = 148;
    const int64_t total = out_idx.numel();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + kThreads - 1) / kThreads,
        kProductionSms * kBlocksPerSm));
    const int votes_len = static_cast<int>(votes.numel());
    const c10::cuda::CUDAGuard device_guard(out_idx.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    map_topk_indices_and_accumulate_votes_litetopk_kernel<<<
        blocks, kThreads, 0, stream>>>(
        out_idx.data_ptr<int32_t>(),
        index_map.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        total,
        static_cast<int>(out_idx.size(0)),
        static_cast<int>(index_map.numel()),
        static_cast<int>(out_idx.size(1)),
        votes_len,
        static_cast<int>(vote_recent_rows64),
        nullptr,
        nullptr,
        nullptr,
        0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Same map/vote pass with the per-call candidate-count telemetry folded in:
// run_max = atomicMax over cand_cnt, over_events += count(cand_cnt > wm).
void map_topk_vote_stats_litetopk_(
        torch::Tensor out_idx,
        torch::Tensor index_map,
        torch::Tensor status,
        torch::Tensor votes,
        int64_t vote_recent_rows64,
        torch::Tensor cand_cnt,
        torch::Tensor run_max,
        torch::Tensor over_events,
        int64_t watermark64) {
    TORCH_CHECK(cand_cnt.is_cuda() && run_max.is_cuda() &&
                    over_events.is_cuda(),
                "stats tensors must be CUDA");
    TORCH_CHECK(cand_cnt.is_contiguous() && run_max.is_contiguous() &&
                    over_events.is_contiguous(),
                "stats tensors must be contiguous");
    TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                    run_max.scalar_type() == torch::kInt &&
                    over_events.scalar_type() == torch::kInt,
                "stats tensors must be int32");
    TORCH_CHECK(cand_cnt.numel() >= out_idx.size(0) &&
                    run_max.numel() >= 1 && over_events.numel() >= 1,
                "stats tensors too small");
    TORCH_CHECK(out_idx.is_cuda() && out_idx.is_contiguous() &&
                    out_idx.scalar_type() == torch::kInt &&
                    out_idx.dim() == 2 && out_idx.numel() > 0,
                "out_idx must be a nonempty contiguous int32 [R,K]");
    TORCH_CHECK(status.numel() == out_idx.size(0) &&
                    votes.dim() == 1 &&
                    vote_recent_rows64 > 0 &&
                    vote_recent_rows64 <= out_idx.size(0),
                "map stats: bad status/votes/recent-rows");

    constexpr int kThreads = 256;
    constexpr int kBlocksPerSm = 8;
    constexpr int kProductionSms = 148;
    const int64_t total = out_idx.numel();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + kThreads - 1) / kThreads,
        kProductionSms * kBlocksPerSm));
    const int votes_len = static_cast<int>(votes.numel());
    const c10::cuda::CUDAGuard device_guard(out_idx.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    map_topk_indices_and_accumulate_votes_litetopk_kernel<<<
        blocks, kThreads, 0, stream>>>(
        out_idx.data_ptr<int32_t>(),
        index_map.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        total,
        static_cast<int>(out_idx.size(0)),
        static_cast<int>(index_map.numel()),
        static_cast<int>(out_idx.size(1)),
        votes_len,
        static_cast<int>(vote_recent_rows64),
        cand_cnt.data_ptr<int32_t>(),
        run_max.data_ptr<int32_t>(),
        over_events.data_ptr<int32_t>(),
        static_cast<int>(watermark64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Destructive single-use selector for the fused indexer. This entry point
// consumes cand_val/cand_idx by compacting its selected subset in place.
// Candidate index space is opaque to selection: the large production path
// emits physical workspace positions and maps only the final K winners in its
// following epilogue. Gate4 values are already in bucket space, and the caller
// owns the final idx output, so this specialization allocates and writes no
// discarded values or temporary index tensor.
void compact_topk_min_thr_inplace_idx_out_litetopk(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        int64_t num_buckets64,
        int64_t k64,
        torch::Tensor out_idx,
        torch::Tensor votes,
        int64_t vote_row_stride64) {
    TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() &&
                cand_cnt.is_cuda() && th_bucket.is_cuda() &&
                boundary_meta.is_cuda() && out_idx.is_cuda() &&
                votes.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
                boundary_meta.is_contiguous() && out_idx.is_contiguous() &&
                votes.is_contiguous(),
                "tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt &&
                cand_cnt.scalar_type() == torch::kInt &&
                out_idx.scalar_type() == torch::kInt,
                "idx/cnt/out_idx must be int32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt,
                "th_bucket must be int32");
    TORCH_CHECK(boundary_meta.scalar_type() == torch::kInt,
                "boundary_meta must be int32");
    TORCH_CHECK(votes.scalar_type() == torch::kInt,
                "votes must be int32");
    TORCH_CHECK(cand_val.dim() == 2 &&
                cand_idx.sizes() == cand_val.sizes(),
                "candidate tensors must be [R,CAP]");
    const int R = static_cast<int>(cand_val.size(0));
    const int CAP = static_cast<int>(cand_val.size(1));
    TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == R,
                "cand_cnt must have R elements");
    const int K = static_cast<int>(k64);
    const int NB = static_cast<int>(num_buckets64);
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    TORCH_CHECK(NB >= 3 && NB <= 256,
                "in-place boundary select requires 3 <= num_buckets <= 256");
    TORCH_CHECK(th_bucket.numel() == R,
                "th_bucket must have R elements");
    TORCH_CHECK(boundary_meta.dim() == 2 &&
                boundary_meta.size(0) == R &&
                boundary_meta.size(1) == NB,
                "boundary_meta must be [R,num_buckets]");
    TORCH_CHECK(out_idx.dim() == 2 &&
                out_idx.size(0) == R &&
                out_idx.size(1) == K,
                "out_idx must be [R,K]");
    TORCH_CHECK(votes.dim() == 1,
                "votes must be a 1-D histogram (or empty)");
    const int votes_len = static_cast<int>(votes.numel());
    TORCH_CHECK(
        vote_row_stride64 == 1 ||
        vote_row_stride64 == 8 ||
        vote_row_stride64 == 16,
        "vote_row_stride must be one of {1, 8, 16}");
    const int vote_row_stride =
        static_cast<int>(vote_row_stride64);
    auto stream = c10::cuda::getCurrentCUDAStream();
    compact_topk_min_thr_inplace_idx_out_litetopk_kernel<false><<<
        R, 256, 0, stream>>>(
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        R,
        CAP,
        K,
        NB,
        out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len,
        vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_topk_min_thr_paged_idx_out_litetopk(
        torch::Tensor inline_val,
        torch::Tensor inline_idx,
        torch::Tensor overflow_val,
        torch::Tensor overflow_idx,
        torch::Tensor page_table,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        int64_t logical_cap64,
        int64_t num_buckets64,
        int64_t topk64,
        torch::Tensor out_idx,
        torch::Tensor votes,
        int64_t vote_row_stride64) {
    const PagedCandidateShape shape = check_paged_candidate_layout(
        inline_val, inline_idx, overflow_val, overflow_idx,
        page_table, logical_cap64);
    TORCH_CHECK(
        cand_cnt.is_cuda() && th_bucket.is_cuda() &&
            boundary_meta.is_cuda() && out_idx.is_cuda() &&
            votes.is_cuda(),
        "paged selector tensors must be CUDA");
    TORCH_CHECK(
        inline_val.device() == cand_cnt.device() &&
            inline_val.device() == th_bucket.device() &&
            inline_val.device() == boundary_meta.device() &&
            inline_val.device() == out_idx.device() &&
            inline_val.device() == votes.device(),
        "paged selector tensors must be on one CUDA device");
    TORCH_CHECK(
        cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
            boundary_meta.is_contiguous() && out_idx.is_contiguous() &&
            votes.is_contiguous(),
        "paged selector tensors must be contiguous");
    TORCH_CHECK(
        cand_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            out_idx.scalar_type() == torch::kInt &&
            votes.scalar_type() == torch::kInt,
        "paged selector metadata/output must be int32");
    TORCH_CHECK(
        num_buckets64 >= 3 && num_buckets64 <= 256,
        "paged selector requires 3 <= num_buckets <= 256");
    TORCH_CHECK(
        topk64 >= 1 && topk64 <= logical_cap64,
        "topk must be in [1,logical_cap]");
    TORCH_CHECK(
        cand_cnt.dim() == 1 && cand_cnt.numel() == shape.rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == shape.rows,
        "cand_cnt/th_bucket must have Q elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 &&
            boundary_meta.size(0) == shape.rows &&
            boundary_meta.size(1) == num_buckets64,
        "boundary_meta must be [Q,num_buckets]");
    TORCH_CHECK(
        out_idx.dim() == 2 && out_idx.size(0) == shape.rows &&
            out_idx.size(1) == topk64,
        "out_idx must be [Q,topk]");
    TORCH_CHECK(
        votes.dim() == 1 &&
            votes.numel() <= std::numeric_limits<int>::max(),
        "votes must be a 1-D int32 histogram fitting the int32 ABI");
    TORCH_CHECK(
        vote_row_stride64 == 1 ||
            vote_row_stride64 == 8 ||
            vote_row_stride64 == 16,
        "vote_row_stride must be one of {1,8,16}");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(out_idx, inline_val) &&
            !tensor_storage_ranges_overlap(out_idx, inline_idx) &&
            !tensor_storage_ranges_overlap(out_idx, overflow_val) &&
            !tensor_storage_ranges_overlap(out_idx, overflow_idx) &&
            !tensor_storage_ranges_overlap(out_idx, page_table) &&
            !tensor_storage_ranges_overlap(out_idx, cand_cnt) &&
            !tensor_storage_ranges_overlap(out_idx, th_bucket) &&
            !tensor_storage_ranges_overlap(out_idx, boundary_meta) &&
            !tensor_storage_ranges_overlap(out_idx, votes),
        "paged selector output must not alias candidate or page-table storage");

    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const int votes_len = static_cast<int>(votes.numel());
    const int vote_row_stride =
        static_cast<int>(vote_row_stride64);
    c10::cuda::CUDAGuard device_guard(inline_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compact_topk_min_thr_paged_idx_out_litetopk_kernel<<<
        shape.rows, 256, 0, stream>>>(
        candidate_data_ptr(inline_val),
        inline_idx.data_ptr<int32_t>(),
        candidate_data_ptr(overflow_val),
        overflow_idx.data_ptr<int32_t>(),
        page_table.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        shape.rows,
        shape.overflow_pages,
        shape.table_pages_per_row,
        shape.logical_cap,
        num_buckets,
        topk,
        out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len,
        vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int fallback_blocks = std::min(
        (shape.rows + 255) / 256, NUM_SMS);
    compact_topk_min_thr_paged_large_boundary_litetopk_kernel<<<
        fallback_blocks, 256, 0, stream>>>(
        candidate_data_ptr(inline_val),
        inline_idx.data_ptr<int32_t>(),
        candidate_data_ptr(overflow_val),
        overflow_idx.data_ptr<int32_t>(),
        page_table.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        shape.rows,
        shape.overflow_pages,
        shape.table_pages_per_row,
        shape.logical_cap,
        num_buckets,
        topk,
        out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len,
        vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_topk_min_thr_segmented_idx_out_litetopk(
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        torch::Tensor seed_cnt,
        torch::Tensor segment_val,
        torch::Tensor segment_idx,
        torch::Tensor segment_desc,
        torch::Tensor spill_desc,
        torch::Tensor row_spill,
        torch::Tensor segment_pool_head,
        torch::Tensor total_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        torch::Tensor status,
        int64_t logical_cap64,
        int64_t num_buckets64,
        int64_t topk64,
        torch::Tensor out_idx,
        torch::Tensor votes,
        int64_t vote_row_stride64) {
    const SegmentCandidateShape shape = check_segment_candidate_layout(
        seed_val, seed_idx, segment_val, segment_idx, segment_desc);
    check_segment_spill_layout(
        spill_desc, row_spill, segment_desc, shape);
    TORCH_CHECK(
        seed_cnt.is_cuda() && total_cnt.is_cuda() && th_bucket.is_cuda() &&
            boundary_meta.is_cuda() && status.is_cuda() &&
            segment_pool_head.is_cuda() && out_idx.is_cuda() &&
            votes.is_cuda(),
        "segmented selector tensors must be CUDA");
    TORCH_CHECK(
        seed_val.device() == seed_cnt.device() &&
            seed_val.device() == total_cnt.device() &&
            seed_val.device() == th_bucket.device() &&
            seed_val.device() == boundary_meta.device() &&
            seed_val.device() == status.device() &&
            seed_val.device() == segment_pool_head.device() &&
            seed_val.device() == out_idx.device() &&
            seed_val.device() == votes.device(),
        "segmented selector tensors must be on one CUDA device");
    TORCH_CHECK(
        seed_cnt.is_contiguous() && total_cnt.is_contiguous() &&
            th_bucket.is_contiguous() && boundary_meta.is_contiguous() &&
            status.is_contiguous() && segment_pool_head.is_contiguous() &&
            out_idx.is_contiguous() && votes.is_contiguous(),
        "segmented selector tensors must be contiguous");
    TORCH_CHECK(
        seed_cnt.scalar_type() == torch::kInt &&
            total_cnt.scalar_type() == torch::kInt &&
            th_bucket.scalar_type() == torch::kInt &&
            boundary_meta.scalar_type() == torch::kInt &&
            status.scalar_type() == torch::kInt &&
            segment_pool_head.scalar_type() == torch::kInt &&
            out_idx.scalar_type() == torch::kInt &&
            votes.scalar_type() == torch::kInt,
        "segmented selector metadata/output must be int32");
    TORCH_CHECK(
        logical_cap64 >= 1 &&
            logical_cap64 < std::numeric_limits<int32_t>::max() &&
            num_buckets64 >= 3 && num_buckets64 <= 256 &&
            topk64 >= 1 && topk64 <= logical_cap64,
        "invalid segmented logical_cap/num_buckets/topk");
    TORCH_CHECK(
        seed_cnt.dim() == 1 && seed_cnt.numel() == shape.rows &&
            total_cnt.dim() == 1 && total_cnt.numel() == shape.rows &&
            th_bucket.dim() == 1 && th_bucket.numel() == shape.rows &&
            status.dim() == 1 && status.numel() == shape.rows,
        "seed_cnt/total_cnt/th_bucket/status must have Q elements");
    TORCH_CHECK(
        boundary_meta.dim() == 2 &&
            boundary_meta.size(0) == shape.rows &&
            boundary_meta.size(1) == num_buckets64,
        "boundary_meta must be [Q,num_buckets]");
    TORCH_CHECK(
        segment_pool_head.numel() == 1,
        "segment_pool_head must be a scalar int32 tensor");
    TORCH_CHECK(
        out_idx.dim() == 2 && out_idx.size(0) == shape.rows &&
            out_idx.size(1) == topk64,
        "out_idx must be [Q,topk]");
    TORCH_CHECK(
        votes.dim() == 1 &&
            votes.numel() <= std::numeric_limits<int>::max(),
        "votes must be a 1-D int32 histogram fitting the int32 ABI");
    TORCH_CHECK(
        vote_row_stride64 == 1 || vote_row_stride64 == 8 ||
            vote_row_stride64 == 16,
        "vote_row_stride must be one of {1,8,16}");
    TORCH_CHECK(
        !tensor_storage_ranges_overlap(out_idx, seed_val) &&
            !tensor_storage_ranges_overlap(out_idx, seed_idx) &&
            !tensor_storage_ranges_overlap(out_idx, segment_val) &&
            !tensor_storage_ranges_overlap(out_idx, segment_idx) &&
            !tensor_storage_ranges_overlap(out_idx, segment_desc) &&
            !tensor_storage_ranges_overlap(out_idx, spill_desc) &&
            !tensor_storage_ranges_overlap(out_idx, row_spill) &&
            !tensor_storage_ranges_overlap(out_idx, segment_pool_head) &&
            !tensor_storage_ranges_overlap(out_idx, boundary_meta) &&
            !tensor_storage_ranges_overlap(out_idx, votes),
        "segmented selector output must not alias candidate/metadata storage");

    const int votes_len = static_cast<int>(votes.numel());
    const int vote_row_stride = static_cast<int>(vote_row_stride64);
    c10::cuda::CUDAGuard device_guard(seed_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compact_topk_min_thr_segmented_idx_out_litetopk_kernel<<<
        shape.rows, 256, 0, stream>>>(
        candidate_data_ptr(seed_val), seed_idx.data_ptr<int32_t>(),
        seed_cnt.data_ptr<int32_t>(), candidate_data_ptr(segment_val),
        segment_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(
            segment_desc.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(
            spill_desc.data_ptr<int64_t>()),
        row_spill.data_ptr<int32_t>(),
        segment_pool_head.data_ptr<int32_t>(),
        total_cnt.data_ptr<int32_t>(), th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
        shape.rows, shape.desc_windows, shape.micro_records,
        shape.pool_pages, shape.physical_records,
        static_cast<int>(logical_cap64), static_cast<int>(num_buckets64),
        static_cast<int>(topk64), out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len, vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int fallback_blocks = std::min(
        (shape.rows + 255) / 256, NUM_SMS);
    compact_topk_min_thr_segmented_large_boundary_litetopk_kernel<<<
        fallback_blocks, 256, 0, stream>>>(
        candidate_data_ptr(seed_val), seed_idx.data_ptr<int32_t>(),
        seed_cnt.data_ptr<int32_t>(), candidate_data_ptr(segment_val),
        segment_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(
            segment_desc.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(
            spill_desc.data_ptr<int64_t>()),
        row_spill.data_ptr<int32_t>(),
        segment_pool_head.data_ptr<int32_t>(),
        total_cnt.data_ptr<int32_t>(), th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
        shape.rows, shape.desc_windows, shape.micro_records,
        shape.pool_pages, shape.physical_records,
        static_cast<int>(logical_cap64), static_cast<int>(num_buckets64),
        static_cast<int>(topk64), out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len, vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Offline-only selector for the online fixed-bucket payload.  The six-byte
// record layout is unchanged, but its high eight score bits are the explicit
// FP16-coarse bucket and its low 16 bits are the raw descending-score suffix.
void compact_topk_min_thr_inplace_idx_out_online_fixed_litetopk(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        int64_t k64,
        torch::Tensor out_idx,
        torch::Tensor votes,
        int64_t vote_row_stride64) {
    TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() &&
                cand_cnt.is_cuda() && th_bucket.is_cuda() &&
                boundary_meta.is_cuda() && out_idx.is_cuda() &&
                votes.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
                boundary_meta.is_contiguous() && out_idx.is_contiguous() &&
                votes.is_contiguous(),
                "tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt &&
                cand_cnt.scalar_type() == torch::kInt &&
                th_bucket.scalar_type() == torch::kInt &&
                boundary_meta.scalar_type() == torch::kInt &&
                out_idx.scalar_type() == torch::kInt &&
                votes.scalar_type() == torch::kInt,
                "fixed selector metadata/indices must be int32");
    TORCH_CHECK(cand_val.dim() == 2 &&
                cand_idx.sizes() == cand_val.sizes(),
                "candidate tensors must be [R,CAP]");
    const int R = static_cast<int>(cand_val.size(0));
    const int CAP = static_cast<int>(cand_val.size(1));
    const int K = static_cast<int>(k64);
    constexpr int NB = ONLINE_FIXED_BUCKETS;
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == R &&
                th_bucket.dim() == 1 && th_bucket.numel() == R,
                "cand_cnt/th_bucket must have R elements");
    TORCH_CHECK(boundary_meta.dim() == 2 &&
                boundary_meta.size(0) == R &&
                boundary_meta.size(1) == NB,
                "boundary_meta must be [R,256]");
    TORCH_CHECK(out_idx.dim() == 2 &&
                out_idx.size(0) == R &&
                out_idx.size(1) == K,
                "out_idx must be [R,K]");
    TORCH_CHECK(votes.dim() == 1,
                "votes must be a 1-D histogram (or empty)");
    TORCH_CHECK(vote_row_stride64 == 1 ||
                vote_row_stride64 == 8 ||
                vote_row_stride64 == 16,
                "vote_row_stride must be one of {1, 8, 16}");
    const int votes_len = static_cast<int>(votes.numel());
    const int vote_row_stride =
        static_cast<int>(vote_row_stride64);
    auto stream = c10::cuda::getCurrentCUDAStream();
    compact_topk_min_thr_inplace_idx_out_litetopk_kernel<true><<<
        R, 256, 0, stream>>>(
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        R,
        CAP,
        K,
        NB,
        out_idx.data_ptr<int32_t>(),
        votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr,
        votes_len,
        vote_row_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Raw-dump no-hist scan entry: the flush bulk-copies each warp's staged
// ring block to a deterministic (q-block, warp, window) global region with
// no output-reservation atomic; a separate region compactor decodes the
// raw records back into the canonical candidate slab so every downstream
// consumer is unchanged.
void mqa_logits_dsa_static_hot_nohist_dump_litetopk_(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64,
        torch::Tensor dump_slab,
        torch::Tensor dump_counts,
        int64_t dump_max_windows) {
    TORCH_CHECK(q.dim() == 3, "q must be [Q,H,128]");
    const int nh = static_cast<int>(q.size(1));
    if (nh == 64) {
        mqa_logits_dsa_static_hot_litetopk_impl_t<64, 2>(
            q, kv, kv_scales, weights, cu_start, cu_end,
            origin, inv_delta, th_bucket,
            cand_val, cand_idx, cand_cnt, bcount,
            num_buckets64, topk64, true, false,
            dump_slab, dump_counts, dump_max_windows);
    } else {
        mqa_logits_dsa_static_hot_litetopk_impl_t<32, 4>(
            q, kv, kv_scales, weights, cu_start, cu_end,
            origin, inv_delta, th_bucket,
            cand_val, cand_idx, cand_cnt, bcount,
            num_buckets64, topk64, true, false,
            dump_slab, dump_counts, dump_max_windows);
    }
}

// One warp per query row: decode the scan's raw ring regions and append
// them to the [Q, cand_cap] candidate slab after the records already
// present (seed prefix plus rare lane-overflow spills). A region word is
// ((sub_row * kEmitLaneSlots + slot) * 32 + lane) and its KV index is
// kv_start + window*(kEmitChunkBlocks*BLOCK_KV) + local_block*BLOCK_KV +
// (warp*32 + lane), mirroring the scan's flush arithmetic exactly.
template <int kBlockQ>
__global__ __launch_bounds__(32) void compact_dump_regions_litetopk_kernel(
        const uint32_t* __restrict__ dump_slab,
        const uint32_t* __restrict__ dump_counts,
        const int32_t* __restrict__ cu_start,
        const int32_t* __restrict__ cu_end,
        dsa_litetopk::CandidateValue* __restrict__ cand_val,
        int32_t* __restrict__ cand_idx,
        int32_t* __restrict__ cand_cnt,
        int32_t* __restrict__ status,
        const uint32_t seq_len,
        const uint32_t seq_len_kv,
        const uint32_t cand_cap,
        const uint32_t max_windows) {
    constexpr uint32_t kScanWarps = 8;
    constexpr uint32_t kSlots = dsa_litetopk::kEmitLaneSlots;
    const uint32_t row = blockIdx.x;
    if (row >= seq_len) return;
    const uint32_t qb = row / kBlockQ;
    const uint32_t sub = row % kBlockQ;
    const uint32_t lane = threadIdx.x;

    // Reproduce the scan CTA's KV schedule (min start /4*4, max end,
    // single split, padded rows empty).
    uint32_t start = 0xffffffffu, end = 0u;
    #pragma unroll
    for (int i = 0; i < kBlockQ; ++i) {
        const uint32_t r = min(qb * kBlockQ + i, seq_len - 1);
        uint32_t s = static_cast<uint32_t>(cu_start[r]);
        uint32_t e = static_cast<uint32_t>(cu_end[r]);
        if (qb * kBlockQ + i >= seq_len) {
            s = seq_len_kv;
            e = 0;
        }
        start = min(start, min(s, seq_len_kv));
        end = max(end, min(e, seq_len_kv));
    }
    start = start / 4 * 4;
    const uint32_t nkv = end > start
        ? (end - start + BLOCK_KV - 1) / BLOCK_KV
        : 0u;
    const uint32_t windows =
        (nkv + dsa_litetopk::kEmitChunkBlocks - 1) /
        dsa_litetopk::kEmitChunkBlocks;

    int out = cand_cnt[row];
    for (uint32_t w = 0; w < kScanWarps; ++w) {
        for (uint32_t win = 0; win < windows; ++win) {
            const uint64_t region =
                (static_cast<uint64_t>(qb) * kScanWarps + w) *
                    max_windows +
                win;
            const uint32_t packed = dump_counts[region * 32u + lane];
            const uint32_t count = (packed >> (sub * 8)) & 0xffu;
            uint32_t prefix = count;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const uint32_t nsh =
                    __shfl_up_sync(0xffffffffu, prefix, off);
                if (lane >= static_cast<uint32_t>(off)) prefix += nsh;
            }
            const uint32_t total = __shfl_sync(0xffffffffu, prefix, 31);
            if (total != 0u) {
                const uint32_t my_base =
                    static_cast<uint32_t>(out) + prefix - count;
                const uint32_t kv_base =
                    start +
                    win * (dsa_litetopk::kEmitChunkBlocks * BLOCK_KV);
                const uint32_t max_count =
                    __reduce_max_sync(0xffffffffu, count);
                const uint32_t* rbase =
                    dump_slab +
                    region * (kBlockQ * kSlots * 32ull) +
                    (sub * kSlots) * 32u;
                const uint64_t row_out =
                    static_cast<uint64_t>(row) * cand_cap;
                for (uint32_t slot = 0; slot < max_count; ++slot) {
                    const bool live = slot < count;
                    const uint32_t record =
                        live ? __ldcs(rbase + slot * 32u + lane) : 0u;
                    const uint32_t pos = my_base + slot;
                    if (live && pos < cand_cap) {
                        dsa_litetopk::store_candidate_record(
                            &cand_val[row_out + pos],
                            &cand_idx[row_out + pos],
                            record >> 8,
                            kv_base + (record & 0xffu) * BLOCK_KV +
                                w * 32u + lane);
                    }
                }
            }
            out += static_cast<int>(total);
        }
    }
    if (lane == 0) {
        if (static_cast<uint32_t>(out) > cand_cap) {
            dsa_litetopk::paged_candidate_set_status(
                status + row,
                dsa_litetopk::kPagedCandidateLogicalOverflow);
        }
        cand_cnt[row] =
            min(out, static_cast<int>(cand_cap));
    }
}

void compact_dump_regions_litetopk_(
        torch::Tensor dump_slab,
        torch::Tensor dump_counts,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor status,
        int64_t seq_len_kv,
        int64_t max_windows,
        int64_t block_q) {
    TORCH_CHECK(
        dump_slab.is_cuda() && dump_counts.is_cuda() &&
            dump_slab.is_contiguous() && dump_counts.is_contiguous() &&
            dump_slab.scalar_type() == torch::kInt &&
            dump_counts.scalar_type() == torch::kInt,
        "dump slabs must be contiguous int32 CUDA tensors");
    TORCH_CHECK(
        cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
        "cand_val/cand_idx must be [Q,cand_cap]");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(
        block_q == 2 || block_q == 4,
        "block_q must match the scan dispatch (2 for H=64, 4 for H=32)");
    const int seq_len = static_cast<int>(cand_val.size(0));
    const int cand_cap = static_cast<int>(cand_val.size(1));
    TORCH_CHECK(
        cand_cnt.numel() == seq_len && status.numel() >= seq_len &&
            cu_start.numel() == seq_len && cu_end.numel() == seq_len,
        "per-row tensors must have Q elements");
    c10::cuda::CUDAGuard device_guard(cand_val.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    auto kernel = block_q == 2
        ? &compact_dump_regions_litetopk_kernel<2>
        : &compact_dump_regions_litetopk_kernel<4>;
    kernel<<<static_cast<unsigned>(seq_len), 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(dump_slab.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(dump_counts.data_ptr<int>()),
        cu_start.data_ptr<int32_t>(),
        cu_end.data_ptr<int32_t>(),
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        static_cast<uint32_t>(cand_cap),
        static_cast<uint32_t>(max_windows));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int GRAFT_Q_STAGES = 3;

template <int kImplHeads, int kImplBlockQ>
static int compute_smem_bytes_fp4graft() {
    const int smem_q = kImplBlockQ * kImplHeads * (HEAD_DIM / 2);
    const int smem_w = kImplBlockQ * kImplHeads * 4;
    const int smem_sfq = align_up(kImplBlockQ * kImplHeads, 128) * 4;
    const int smem_kv = BLOCK_KV * (HEAD_DIM / 2);
    const int smem_sfkv = align_up(BLOCK_KV, 128) * 4;
    const int num_barriers =
        GRAFT_Q_STAGES * 2 + NUM_KV_STAGES_FP4 * 3 + 3 * 2;
    // BQ2 runs the score-bank engine (kWin=32 float2 bank); BQ4 keeps
    // the staged ring.
    const int smem_ring =
        (MATH_THREADS / 32) * kImplBlockQ *
        dsa_litetopk::kEmitLaneSlots * 32 * (int)sizeof(uint32_t);
    const int smem_bank = 32 * MATH_THREADS * 8;
    const int smem_emit =
        kImplBlockQ == 2 ? (smem_bank > smem_ring ? smem_bank : smem_ring)
                         : smem_ring;
    return GRAFT_Q_STAGES * (smem_q + smem_sfq + smem_w) +
           NUM_KV_STAGES_FP4 * (smem_kv + smem_sfkv) +
           num_barriers * 8 + 4 + 12 + smem_emit +
           (257 + 514) * (int)sizeof(int) +
           (kImplBlockQ == 2 ? 2 * (int)sizeof(int) : 0);
}

void mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_(
        torch::Tensor q,          // [Q, H, 64] uint8 packed e2m1
        torch::Tensor q_sf,       // [Q, H] int32 (4x UE8M0)
        torch::Tensor kv,         // [S, 64] uint8 packed e2m1
        torch::Tensor kv_sf,      // [S] int32 (4x UE8M0)
        torch::Tensor weights,    // [Q, H] fp32 (no q_scale folded)
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64) {
    TORCH_CHECK(
        q.is_cuda() && q_sf.is_cuda() && kv.is_cuda() && kv_sf.is_cuda() &&
            q.is_contiguous() && q_sf.is_contiguous() &&
            kv.is_contiguous() && kv_sf.is_contiguous(),
        "fp4 operands must be contiguous CUDA tensors");
    TORCH_CHECK(
        q.scalar_type() == torch::kUInt8 &&
            kv.scalar_type() == torch::kUInt8 &&
            q_sf.scalar_type() == torch::kInt &&
            kv_sf.scalar_type() == torch::kInt,
        "fp4 operands must be uint8 data with int32 SF streams");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == HEAD_DIM / 2,
        "q must be [Q,H,64] packed e2m1");
    TORCH_CHECK(kv.dim() == 2 && kv.size(1) == HEAD_DIM / 2,
        "kv must be [S,64] packed e2m1");
    const int seq_len = static_cast<int>(q.size(0));
    const int nh = static_cast<int>(q.size(1));
    const int seq_len_kv = static_cast<int>(kv.size(0));
    TORCH_CHECK(nh == 32 || nh == 64, "H must be 32 or 64");
    TORCH_CHECK(
        q_sf.dim() == 2 && q_sf.size(0) == seq_len && q_sf.size(1) == nh &&
            kv_sf.numel() >= seq_len_kv,
        "SF stream shapes must match the packed operands");
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "packed candidates support at most 1M KV positions");
    check_candidate_dtype(cand_val);
    const int cand_cap = static_cast<int>(cand_val.size(1));
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256, "bad num_buckets");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "bad topk");

    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int packed = HEAD_DIM / 2;
    const int sfkv_aligned = align_up(seq_len_kv, 4);
    TORCH_CHECK(kv_sf.numel() >= sfkv_aligned,
        "kv_sf storage is shorter than the aligned KV length");
    TORCH_CHECK(
        weights.is_cuda() && weights.is_contiguous() &&
            weights.scalar_type() == torch::kFloat &&
            weights.dim() == 2 && weights.size(0) >= seq_len &&
            weights.size(1) == nh,
        "weights must be contiguous fp32 [Q, H]");
    TORCH_CHECK(
        cu_start.is_cuda() && cu_end.is_cuda() &&
            cu_start.scalar_type() == torch::kInt &&
            cu_end.scalar_type() == torch::kInt &&
            cu_start.numel() >= seq_len && cu_end.numel() >= seq_len,
        "cu_start/cu_end must be int32 with >= Q rows");
    TORCH_CHECK(
        origin.numel() >= seq_len && inv_delta.numel() >= seq_len &&
            th_bucket.numel() >= seq_len && cand_cnt.numel() >= seq_len &&
            origin.scalar_type() == torch::kFloat &&
            inv_delta.scalar_type() == torch::kFloat &&
            th_bucket.scalar_type() == torch::kInt &&
            cand_cnt.scalar_type() == torch::kInt,
        "origin/inv_delta/th_bucket/cand_cnt must cover Q rows");

    auto launch = [&](auto heads_c, auto blockq_c) {
        constexpr int kH = decltype(heads_c)::value;
        constexpr int kBQ = decltype(blockq_c)::value;
        auto tm_q = make_2d(
            q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
            packed, seq_len * kH, packed, kBQ * kH, packed, 64);
        auto tm_kv = make_2d(
            kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
            packed, seq_len_kv, packed, BLOCK_KV, packed, 64);
        auto tm_ks = make_2d(
            kv_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
            sfkv_aligned, 1, BLOCK_KV, 1, 0, 0);
        auto tm_w = make_2d(
            weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
            kH, seq_len, kH, kBQ, kH, 0);
        auto tm_sfq = make_2d(
            q_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
            align_up(seq_len * kH, 4), 1, kBQ * kH, 1, 0, 0);
        const int smem = compute_smem_bytes_fp4graft<kH, kBQ>();
        const char* tflag = std::getenv("LITETOPK_GRAFT_TIGHTEN");
        const bool tighten = tflag != nullptr && tflag[0] == '1';
        auto pick = [&](auto tight_c) {
            constexpr bool kT = decltype(tight_c)::value;
            return &dsa_litetopk::sm100_dsa_litetopk_fp4graft<
                kH, HEAD_DIM, kBQ, BLOCK_KV,
                GRAFT_Q_STAGES, NUM_KV_STAGES_FP4, NUM_SMS,
                SPEC_THREADS, MATH_THREADS, kT, MATH_THREADS / 128>;
        };
        auto kernel = tighten ? pick(std::true_type{})
                              : pick(std::false_type{});
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        dim3 grid(static_cast<unsigned>(NUM_SMS), 1u, 1u);
        kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(seq_len_kv),
            reinterpret_cast<const uint32_t*>(cu_start.data_ptr<int>()),
            reinterpret_cast<const uint32_t*>(cu_end.data_ptr<int>()),
            origin.data_ptr<float>(),
            inv_delta.data_ptr<float>(),
            th_bucket.data_ptr<int32_t>(),
            candidate_data_ptr(cand_val),
            cand_idx.data_ptr<int32_t>(),
            cand_cnt.data_ptr<int32_t>(),
            static_cast<uint32_t>(cand_cap),
            tm_q, tm_sfq, tm_kv, tm_ks, tm_w,
            static_cast<uint32_t>(topk),
            bcount.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (nh == 64) {
        launch(std::integral_constant<int, 64>{},
               std::integral_constant<int, 2>{});
    } else {
        launch(std::integral_constant<int, 32>{},
               std::integral_constant<int, 4>{});
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "plan_and_permuted_paged_gather_out",
        &pair_swap_gather::plan_and_permuted_paged_gather_out,
        "HOT12288 cooperative pair-swap planning followed by paged gather",
        pybind11::arg("hot"),
        pybind11::arg("hot_epoch"),
        pybind11::arg("permutation"),
        pybind11::arg("swap_a"),
        pybind11::arg("swap_b"),
        pybind11::arg("counts"),
        pybind11::arg("window_start"),
        pybind11::arg("common_end"),
        pybind11::arg("epoch"),
        pybind11::arg("kv_cache"),
        pybind11::arg("dst_k"),
        pybind11::arg("dst_scale"),
        pybind11::arg("block_table"));
    m.def(
        "candidate_fp24_global_litetopk",
        []() { return true; },
        "Reports the production high24 FP32 local/global candidate ABI");
    m.def(
        "candidate_value_u16_litetopk",
        []() { return true; },
        "Reports the packed six-byte candidate ABI");
    m.def(
        "dense_topk_litetopk_",
        &litetopk_dense::dense_topk_litetopk_,
        "Single-launch exact dense prefill top-k",
        pybind11::arg("logits"),
        pybind11::arg("row_starts"),
        pybind11::arg("row_ends"),
        pybind11::arg("out"),
        pybind11::arg("rows"),
        pybind11::arg("stride0"),
        pybind11::arg("stride1"),
        pybind11::arg("topk"),
        pybind11::arg("num_init_tokens"),
        pybind11::arg("num_local_tokens"));
    m.def("seed_prep_litetopk_", &seed_prep_litetopk_,
          "In-place fused sample prep (caller-owned buffers)",
          pybind11::arg("slog"), pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("cand_cap"), pybind11::arg("emit_limit"), pybind11::arg("headroom"),
          pybind11::arg("probe_stride_tok"), pybind11::arg("hist_stride"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"),
          pybind11::arg("th_bucket"), pybind11::arg("bcount"),
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"));
    m.def(
          "mqa_logits_dsa_static_hot_nohist_litetopk_",
          &mqa_logits_dsa_static_hot_nohist_litetopk_,
          "Production fixed-HOT suffix scan without an online histogram",
          pybind11::arg("q"),
          pybind11::arg("kv"),
          pybind11::arg("kv_scales"),
          pybind11::arg("weights"),
          pybind11::arg("cu_start"),
          pybind11::arg("cu_end"),
          pybind11::arg("origin"),
          pybind11::arg("inv_delta"),
          pybind11::arg("th_bucket"),
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("bcount"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"));
    m.def(
          "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_",
          &mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_,
          "MXFP4 no-hist scan on the dense reference's persistent-CTA "
          "skeleton (graft: STG epilogue replaced by bucket-gate emit)",
          pybind11::arg("q"),
          pybind11::arg("q_sf"),
          pybind11::arg("kv"),
          pybind11::arg("kv_sf"),
          pybind11::arg("weights"),
          pybind11::arg("cu_start"),
          pybind11::arg("cu_end"),
          pybind11::arg("origin"),
          pybind11::arg("inv_delta"),
          pybind11::arg("th_bucket"),
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("bcount"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"));
    m.def(
          "mqa_logits_dsa_static_hot_nohist_dump_litetopk_",
          &mqa_logits_dsa_static_hot_nohist_dump_litetopk_,
          "Raw-dump no-hist suffix scan: deterministic per-(q-block,warp,"
          "window) bulk regions, no output-reservation atomic",
          pybind11::arg("q"),
          pybind11::arg("kv"),
          pybind11::arg("kv_scales"),
          pybind11::arg("weights"),
          pybind11::arg("cu_start"),
          pybind11::arg("cu_end"),
          pybind11::arg("origin"),
          pybind11::arg("inv_delta"),
          pybind11::arg("th_bucket"),
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("bcount"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"),
          pybind11::arg("dump_slab"),
          pybind11::arg("dump_counts"),
          pybind11::arg("dump_max_windows"));
    m.def(
          "compact_dump_regions_litetopk_",
          &compact_dump_regions_litetopk_,
          "Decode raw dump regions into the canonical candidate slab",
          pybind11::arg("dump_slab"),
          pybind11::arg("dump_counts"),
          pybind11::arg("cu_start"),
          pybind11::arg("cu_end"),
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("status"),
          pybind11::arg("seq_len_kv"),
          pybind11::arg("max_windows"),
          pybind11::arg("block_q") = 4);
    m.def(
          "h2048_safe_topk_out_litetopk_",
          &h2048_safe_topk_out_litetopk_,
          "Frozen h2048 physical selector plus exact overflow fallback",
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("out_idx"),
          pybind11::arg("status"),
          pybind11::arg("diagnostic_scratch"),
          pybind11::arg("index_limit"));
    m.def(
          "finalize_static_hot_meta_litetopk_",
          &finalize_static_hot_meta_litetopk_,
          "Rebuild static-HOT metadata while retaining physical candidate "
          "indices for winner-only mapping",
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("th_bucket"),
          pybind11::arg("boundary_meta"),
          pybind11::arg("status"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"),
          pybind11::arg("index_limit") = 0);
    m.def("cand_count_stats_litetopk_", &cand_count_stats_litetopk_,
          "Single-CTA candidate-count max and exact integer mean",
          pybind11::arg("cand_cnt"), pybind11::arg("stats"));
    m.def(
          "carry_votes_topk_reset_",
          &carry_votes_topk_reset_litetopk_,
          "Deterministic carry-vote top-k with fused histogram reset",
          pybind11::arg("votes"),
          pybind11::arg("out_idx"),
          pybind11::arg("partial"),
          pybind11::arg("state"),
          pybind11::arg("k"),
          pybind11::arg("max_vote"),
          pybind11::arg("min_index") = 0);
    m.def(
          "map_topk_indices_and_accumulate_votes_litetopk_",
          &map_topk_indices_and_accumulate_votes_litetopk_,
          "Map physical TOPK winners to corpus space and vote every winner "
          "from the most recent query rows in one grid-wide pass",
          pybind11::arg("out_idx"),
          pybind11::arg("index_map"),
          pybind11::arg("status"),
          pybind11::arg("votes"),
          pybind11::arg("vote_recent_rows") = 1);
    m.def("map_topk_vote_stats_litetopk_",
          &map_topk_vote_stats_litetopk_,
          "map/vote pass with fused candidate-count telemetry");
    m.def("compact_topk_min_thr_inplace_idx_out_litetopk",
          &compact_topk_min_thr_inplace_idx_out_litetopk,
          "Single-use Gate4 threshold top-k directly into caller idx output",
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("th_bucket"),
          pybind11::arg("boundary_meta"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"),
          pybind11::arg("out_idx"),
          pybind11::arg("votes"),
          pybind11::arg("vote_row_stride") = 1);
}
