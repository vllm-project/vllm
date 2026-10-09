// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Derived from DeepGEMM commit 891d57b4db1071624b5c8fa0d1e51cb317fa709f;
// see LICENSE.deepseek-deepgemm.
// LiteTopK scoring and selection; uses DeepGEMM and CUTLASS headers.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <initializer_list>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>

#include "sm100_dsa_litetopk.cuh"
// Select physical IDs from high24 keys; exact selection applies to these keys,
// not the original FP32 scores. One kernel caches small boundaries in shared
// memory and streams larger boundaries through the same radix selection.

namespace h2048_safe_topk {

constexpr int kBins = 256;
constexpr int kTopK = 2048;
constexpr int kMinCap = 49152;
constexpr int kMaxCap = 1 << 20;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadCertificate = 1u << 4,
};

using dsa_litetopk::candidate_load_score_code;
using dsa_litetopk::candidate_fp24_code;

__device__ __forceinline__ float decode_score_code(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
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

template <int Bins>
__device__ __forceinline__ void find_radix_digit(
    const uint32_t* __restrict__ hist, uint32_t* __restrict__ desired,
    uint32_t* __restrict__ rank, uint32_t* __restrict__ selected_count,
    int shift) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 32) return;
  constexpr unsigned kFull = 0xffffffffu;
  static_assert(Bins == 256 || Bins == 512 || Bins == 1024 || Bins == 2048 ||
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
  const uint32_t prefix = *desired;
  const unsigned group_mask = __ballot_sync(kFull, inclusive >= target);
  if (target == 0u || group_mask == 0u) return;
  const int winning_group = __ffs(group_mask) - 1;
  const uint32_t group_before =
      __shfl_sync(kFull, inclusive - group_count, winning_group);
  // Scan per-lane segments, then locate the winning bin within its segment.
  const int segment_offset = lane * kItemsPerLane;
  const bool segment_valid = segment_offset < kGroupBins;
  const int segment_first = winning_group * kGroupBins + segment_offset;
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
      kFull, segment_valid && group_before + segment_inclusive >= target);
  if (segment_mask == 0u) return;
  const int winning_lane = __ffs(segment_mask) - 1;
  const uint32_t segment_before =
      __shfl_sync(kFull, segment_inclusive - segment_count, winning_lane);

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
  // Shuffle collectives do not order shared-memory loads against lane 0's
  // stores. Finish every lane's state snapshot before publishing the digit.
  __syncwarp(kFull);
  if (lane == 0) {
    const uint32_t digit =
        static_cast<uint32_t>(winning_group * kGroupBins) + local_digit;
    *desired = prefix | (digit << static_cast<uint32_t>(shift));
    *rank = target - group_before - segment_before - local_before;
    *selected_count = local_count;
  }
}

// Every lane participates, including padding lanes. Collapse equal digits in
// each warp before touching the shared counter, avoiding a hot-bin atomic per
// candidate when many high24 keys share the same digit.
__device__ __forceinline__ void add_boundary_digit(uint32_t* histogram,
                                                  uint32_t digit, bool valid) {
  constexpr unsigned kFull = 0xffffffffu;
  if (__ballot_sync(kFull, valid) == 0u) return;
  const unsigned peers = __match_any_sync(kFull, valid ? digit : 0xffffffffu);
  if (valid && (threadIdx.x & 31) == __ffs(peers) - 1) {
    atomicAdd(histogram + digit, static_cast<uint32_t>(__popc(peers)));
  }
}

template <int Threads, int BoundaryCapacity, int HistScale>
__global__ __launch_bounds__(Threads) void coarse_tiering_topk_kernel(
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts, int32_t* __restrict__ output,
    int32_t* __restrict__ status, int32_t* __restrict__ diagnostics, int rows,
    int cap, int topk, int sequence_length) {
  static_assert(Threads == 128 || Threads == 256 || Threads == 512);
  static_assert(BoundaryCapacity == 4096 || BoundaryCapacity == 8192);
  static_assert(HistScale == 8);
  constexpr int kCoarseBins = kBins * HistScale;
  constexpr unsigned kFull = 0xffffffffu;
  const int logical_block = static_cast<int>(blockIdx.x);
  // Real count argmax is near the end; longest rows launch first.
  const int row = rows - 1 - logical_block;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  if (row < 0 || row >= rows) return;

  // The coarse histogram is dead before boundary gathering. Reuse its storage
  // for cached high24 keys and 20-bit candidate slots in the 6-byte input ABI.
  union BoundaryScratch {
    uint32_t coarse_hist[kCoarseBins];
    struct {
      uint16_t value[BoundaryCapacity];
      uint32_t index[BoundaryCapacity];
    } records;
  };
  __shared__ BoundaryScratch scratch;
  __shared__ uint32_t digit_hist[kBins];
  uint32_t* const hist = scratch.coarse_hist;
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
  __shared__ int s_boundary_out_cursor;

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
    s_boundary_out_cursor = 0;
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

  for (int slot = tid; slot < count; slot += Threads) {
    const uint16_t value = values[row_base + slot];
    const int32_t packed_index = packed_indices[row_base + slot];
    const uint32_t physical =
        static_cast<uint32_t>(packed_index) & kPhysicalMask;
    if (physical >= static_cast<uint32_t>(sequence_length)) {
      atomicOr(&s_status, kBadPhysical);
      continue;
    }
    const uint32_t code = candidate_load_score_code(value, packed_index);
    const float decoded = decode_score_code(code);
    if (!isfinite(decoded)) {
      atomicOr(&s_status, kNonFinite);
      continue;
    }
    atomicAdd(hist + coarse_bucket_scaled<HistScale>(code), 1u);
  }
  __syncthreads();
  // Snapshot status across the CTA before warp 0 reuses adjacent radix state;
  // the compiler may combine these shared scalar loads into a vector load.
  const bool select_coarse = s_status == 0u;
  __syncthreads();
  if (select_coarse) {
    find_radix_digit<kCoarseBins>(hist, &s_desired, &s_rank, &s_selected_count,
                                  0);
  }
  __syncthreads();
  if (tid == 0 && select_coarse) {
    uint32_t coarse_status = 0u;
    s_coarse_bucket = static_cast<int>(s_desired);
    s_coarse_need = static_cast<int>(s_rank);
    s_coarse_lt = topk - s_coarse_need;
    s_boundary_count = static_cast<int>(s_selected_count);
    if (s_coarse_bucket < 0 || s_coarse_bucket >= kCoarseBins ||
        s_coarse_lt < 0 || s_coarse_lt >= topk || s_coarse_need <= 0 ||
        s_coarse_need > s_boundary_count) {
      coarse_status |= kBadCertificate;
    }
    s_status = coarse_status;
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
      diag[3] = s_boundary_count;
      diag[4] = 0;
    }
    return;
  }

  // Emit strict winners and cache the boundary when it fits. Larger boundaries
  // use the same selection below, reading candidates in bounded tiles.
  const bool cache_boundary = s_boundary_count <= BoundaryCapacity;
  const int threshold_bucket = s_coarse_bucket;
  const float threshold_edge =
      static_cast<float>(threshold_bucket) / static_cast<float>(HistScale);
  const float next_threshold_edge =
      static_cast<float>(threshold_bucket + 1) / static_cast<float>(HistScale);
  const uint32_t threshold_code = candidate_fp24_code(threshold_edge);
  const uint32_t next_threshold_code = candidate_fp24_code(next_threshold_edge);
  const int warp = tid >> 5;
  for (int base = warp * 32; cache_boundary && base < count; base += Threads) {
    const int slot = base + lane;
    uint32_t code = 0u;
    int32_t packed_index = 0;
    const bool valid = slot < count;
    if (valid) {
      packed_index = packed_indices[row_base + slot];
      code = candidate_load_score_code(values[row_base + slot], packed_index);
    }
    const bool is_strict =
        valid && threshold_bucket > 0 && code < threshold_code;
    const bool is_boundary =
        valid &&
        (threshold_bucket == kCoarseBins - 1 || code < next_threshold_code) &&
        (threshold_bucket == 0 || code >= threshold_code);
    const unsigned strict_mask = __ballot_sync(kFull, is_strict);
    const unsigned boundary_mask =
        __ballot_sync(kFull, cache_boundary && is_boundary);
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
    if (cache_boundary && is_boundary) {
      const int pos = boundary_base + __popc(boundary_mask & lane_before);
      if (pos < BoundaryCapacity) {
        scratch.records.value[pos] = static_cast<uint16_t>(code);
        scratch.records.index[pos] =
            ((code >> 16) << 20) | static_cast<uint32_t>(slot);
      }
    }
  }
  __syncthreads();
  if (tid == 0 && cache_boundary &&
      (s_strict_cursor != s_coarse_lt ||
       s_boundary_cursor != s_boundary_count)) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  const int records = cache_boundary ? s_boundary_count : count;
  const auto load_boundary = [&](int record, uint32_t& code,
                                 uint32_t& slot) -> bool {
    code = slot = 0u;
    if (record >= records) return false;
    const uint32_t packed = scratch.records.index[record];
    code = candidate_load_score_code(scratch.records.value[record], packed);
    slot = packed & kPhysicalMask;
    return true;
  };

  // Interior coarse buckets have a bounded high24 interval. Preserve its
  // common high bytes instead of rebuilding their single-bin histograms.
  // The clamped first/last buckets may contain out-of-range scores and must
  // still search all 24 bits.
  int first_shift = 16;
  uint32_t common_prefix = 0u;
  if (threshold_bucket > 0 && threshold_bucket < kCoarseBins - 1) {
    const uint32_t varying = threshold_code ^ (next_threshold_code - 1u);
    first_shift = varying < (1u << 8) ? 0 : (varying < (1u << 16) ? 8 : 16);
    common_prefix = threshold_code & ~((1u << (first_shift + 8)) - 1u);
  }
  if (tid == 0) {
    s_desired = common_prefix;
    s_rank = static_cast<uint32_t>(s_coarse_need);
    s_selected_count = 0u;
  }
  __syncthreads();
  for (int shift = first_shift; shift >= 0; shift -= 8) {
    for (int i = tid; i < kBins; i += Threads) digit_hist[i] = 0u;
    __syncthreads();
    const uint32_t desired = s_desired;
    if (cache_boundary) {
      for (int base = 0; base < records; base += Threads) {
        uint32_t code, slot;
        const bool valid = load_boundary(base + tid, code, slot);
        const bool keep = valid &&
                          (code >> (shift + 8)) == (desired >> (shift + 8));
        add_boundary_digit(digit_hist, (code >> shift) & 0xffu, keep);
      }
    } else {
      // Independent loads overlap the latency of streamed candidate reads.
      constexpr int kItems = 4;
      for (int base = tid; base < count; base += Threads * kItems) {
        uint32_t codes[kItems];
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          const int j = base + i * Threads;
          codes[i] = j < count
              ? candidate_load_score_code(values[row_base + j], packed_indices[row_base + j])
              : 0xffffffffu;
        }
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
          const uint32_t code = codes[i];
          if (base + i * Threads < count &&
              (threshold_bucket == 0 || code >= threshold_code) &&
              (threshold_bucket == kCoarseBins - 1 || code < next_threshold_code) &&
              (code >> (shift + 8)) == (desired >> (shift + 8))) {
            atomicAdd(digit_hist + ((code >> shift) & 0xffu), 1u);
          }
        }
      }
    }
    __syncthreads();
    find_radix_digit<kBins>(digit_hist, &s_desired, &s_rank, &s_selected_count, shift);
    __syncthreads();
  }
  if (tid == 0 && (s_rank == 0u || s_rank > s_selected_count ||
                   s_rank > static_cast<uint32_t>(s_coarse_need))) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  const uint32_t exact_pivot = s_desired;
  const int exact_equal_take = static_cast<int>(s_rank);
  const int exact_equal_count = static_cast<int>(s_selected_count);
  uint32_t equal_slot_limit = 0xffffffffu;
  // Every warp must snapshot the score pivot before reusing radix state.
  __syncthreads();
  if (cache_boundary && exact_equal_take < exact_equal_count) {
    // Boundary gathering is unordered across warps. Select the cutoff among
    // original candidate slots to retain deterministic ties, without comparing
    // each equal key against the whole boundary. Slot IDs have at most 20 bits,
    // so this needs at most 3 passes.
    if (tid == 0) {
      s_desired = 0u;
      s_rank = static_cast<uint32_t>(exact_equal_take);
    }
    __syncthreads();
    for (int shift = count > (1 << 16) ? 16 : 8; shift >= 0; shift -= 8) {
      for (int i = tid; i < kBins; i += Threads) digit_hist[i] = 0u;
      __syncthreads();
      const uint32_t desired = s_desired;
      for (int base = 0; base < records; base += Threads) {
        const int j = base + tid;
        uint32_t code, slot;
        const bool valid = load_boundary(j, code, slot);
        const bool keep = valid && code == exact_pivot &&
                          (slot >> (shift + 8)) == (desired >> (shift + 8));
        add_boundary_digit(digit_hist, (slot >> shift) & 0xffu, keep);
      }
      __syncthreads();
      find_radix_digit<kBins>(digit_hist, &s_desired, &s_rank, &s_selected_count, shift);
      __syncthreads();
    }
    equal_slot_limit = s_desired;
  }

  // Cached records use the slot cutoff. Streamed records follow slot order;
  // process four records per thread to overlap global loads and amortize CTA
  // prefix barriers. The score radix algorithm above is shared by both modes.
  if (cache_boundary) {
    for (int base = 0; base < records; base += Threads) {
      const int j = base + tid;
      uint32_t code, slot;
      const bool valid = load_boundary(j, code, slot);
      const bool take = valid &&
          (code < exact_pivot ||
           (code == exact_pivot && slot <= equal_slot_limit));
      const unsigned mask = __ballot_sync(kFull, take);
      int offset = 0;
      if (lane == 0 && mask != 0u) {
        offset = atomicAdd(&s_boundary_out_cursor, __popc(mask));
      }
      offset = __shfl_sync(kFull, offset, 0);
      const unsigned lane_before = lane == 0 ? 0u : ((1u << lane) - 1u);
      const int pos = s_coarse_lt + offset + __popc(mask & lane_before);
      if (take && pos < topk) {
        output[out_base + pos] = static_cast<int32_t>(
            static_cast<uint32_t>(packed_indices[row_base + slot]) &
            kPhysicalMask);
      }
    }
  } else {
    constexpr int kItems = 4;
    constexpr int kWarps = Threads / 32;
    static_assert(kItems * kWarps <= 32);
    uint32_t equal_seen = 0u;
    for (int base = 0; base < count; base += Threads * kItems) {
      uint32_t codes[kItems];
      uint32_t packed[kItems];
      uint32_t equal_rank[kItems];
      const unsigned lane_before = lane == 0 ? 0u : ((1u << lane) - 1u);
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const int slot = base + i * Threads + tid;
        codes[i] = 0xffffffffu;
        packed[i] = 0u;
        if (slot < count) {
          packed[i] = static_cast<uint32_t>(packed_indices[row_base + slot]);
          codes[i] = candidate_load_score_code(values[row_base + slot], packed[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const unsigned mask = __ballot_sync(kFull, codes[i] == exact_pivot);
        if (lane == 0) digit_hist[i * kWarps + warp] = __popc(mask);
        equal_rank[i] = __popc(mask & lane_before);
      }
      __syncthreads();
      if (warp == 0) {
        const uint32_t n = lane < kItems * kWarps ? digit_hist[lane] : 0u;
        uint32_t inclusive = n;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          const uint32_t other = __shfl_up_sync(kFull, inclusive, offset);
          if (lane >= offset) inclusive += other;
        }
        if (lane < kItems * kWarps) {
          digit_hist[lane] = equal_seen + inclusive - n;
        }
        if (lane == 31) s_rank = equal_seen + inclusive;
      }
      __syncthreads();
      equal_seen = s_rank;
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const bool take = codes[i] < exact_pivot ||
            (codes[i] == exact_pivot &&
             digit_hist[i * kWarps + warp] + equal_rank[i] <
                 static_cast<uint32_t>(exact_equal_take));
        const unsigned mask = __ballot_sync(kFull, take);
        int offset = 0;
        if (lane == 0 && mask != 0u) {
          offset = atomicAdd(&s_boundary_out_cursor, __popc(mask));
        }
        offset = __shfl_sync(kFull, offset, 0);
        const int pos = offset + __popc(mask & lane_before);
        if (take && pos < topk) {
          output[out_base + pos] = static_cast<int32_t>(packed[i] & kPhysicalMask);
        }
      }
      __syncthreads();
      if (s_boundary_out_cursor == topk) break;
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (s_boundary_out_cursor != (cache_boundary ? s_coarse_need : topk)) {
      s_status |= kBadCertificate;
    }
    status[row] = static_cast<int32_t>(s_status);
    int32_t* diag = diagnostics + static_cast<int64_t>(row) * 5;
    diag[0] = count;
    diag[1] = threshold_bucket;
    diag[2] = s_coarse_lt;
    diag[3] = s_boundary_count;
    diag[4] = exact_equal_count;
  }
}

}  // namespace h2048_safe_topk

namespace {

using CandidateValue = dsa_litetopk::CandidateValue;

static CandidateValue* candidate_data_ptr(torch::Tensor& tensor) {
  return reinterpret_cast<CandidateValue*>(tensor.data_ptr<at::Half>());
}

// Validate common tensor properties once; entry points still check their shapes.
static void check_tensors(std::initializer_list<torch::Tensor> tensors,
                          const torch::Tensor& reference, at::ScalarType dtype) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device() &&
                    tensor.is_contiguous() && tensor.scalar_type() == dtype,
                "expected contiguous ", dtype, " tensors on ", reference.device());
  }
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer, int smem_inner,
                           int smem_outer, long gmem_outer_stride,
                           int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  CUtensorMap tm;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                               : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : swizzle_mode == 32
                                   ? CU_TENSOR_MAP_SWIZZLE_32B
                                   : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = cuTensorMapEncodeTiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

// Page-local K values followed by FP32 scales; page stride includes both.
static CUtensorMap make_paged_values(void* ptr, int pages, int page_size,
                                     int64_t page_stride) {
  CUtensorMap tm;
  const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(page_size),
                              static_cast<cuuint64_t>(pages)};
  const cuuint64_t strides[2] = {128, static_cast<cuuint64_t>(page_stride)};
  const cuuint32_t box[3] = {128, static_cast<cuuint32_t>(page_size), 1};
  const cuuint32_t elem[3] = {1, 1, 1};
  const auto rc = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, dims,
      strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "paged tensor map failed: ", int(rc));
  return tm;
}

__global__ void gather_paged_sample_kernel(
    const uint8_t* cache, const int32_t* block_table, const int32_t* page_order,
    uint8_t* dst, float* scales, int sample_size, int page_size,
    int64_t page_stride) {
  const int token = blockIdx.x * 32 + threadIdx.y;
  if (token >= sample_size) return;
  const int page = block_table[page_order[token / page_size]];
  const int offset = token % page_size;
  const auto* src = cache + int64_t(page) * page_stride;
  reinterpret_cast<float4*>(dst + int64_t(token) * 128)[threadIdx.x] =
      reinterpret_cast<const float4*>(src + offset * 128)[threadIdx.x];
  if (threadIdx.x == 0)
    scales[token] = reinterpret_cast<const float*>(src + page_size * 128)[offset];
}

void gather_paged_sample_out(torch::Tensor cache, torch::Tensor block_table,
                            torch::Tensor page_order, torch::Tensor dst,
                            torch::Tensor scales) {
  check_tensors({cache}, cache, torch::kUInt8);
  check_tensors({block_table, page_order}, cache, torch::kInt);
  check_tensors({scales}, cache, torch::kFloat);
  check_tensors({dst}, cache, dst.scalar_type());
  TORCH_CHECK(cache.dim() == 3 && cache.size(1) == 64 && cache.size(2) == 132,
              "sample gather requires cache [pages,64,132]");
  TORCH_CHECK(dst.dim() == 2 && dst.size(1) == 128 && dst.element_size() == 1 &&
                  scales.numel() == dst.size(0), "invalid sample buffers");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1 &&
                  page_order.dim() == 1 && page_order.numel() * 64 >= dst.size(0),
              "invalid sample page tables");
  const c10::cuda::CUDAGuard guard(cache.device());
  gather_paged_sample_kernel<<<(dst.size(0)+31)/32, dim3(8,32), 0,
      c10::cuda::getCurrentCUDAStream()>>>(
      cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(),
      page_order.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(dst.data_ptr()),
      scales.data_ptr<float>(), dst.size(0), cache.size(1), cache.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 32;
constexpr int BLOCK_Q = 4;
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1;  // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200

// Initialize gate/count state and the boundary certificate for the offline scan.

// One CTA per row derives the bucket scale, K-th bucket, and full seed histogram,
// then emits all seed candidates through that bucket for later selection.
constexpr int kSeedThreads = 256;
constexpr int kSeed12Threads = 256;

template <int kRetainedHead, int BT>
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride, const int head,
    const int NB, const int K,
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // Extra sample span above the maximum; scale NB with it.
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket, CandidateValue* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx, int32_t* __restrict__ cand_cnt,
    const int cand_cap,
    int32_t* __restrict__ bcount_out) {
  constexpr int NSUB = BT == 512 ? 16 : 4;  // one histogram per large-seed warp
  static_assert(kRetainedHead == 8192 || kRetainedHead == 12288 ||
                kRetainedHead == 32768,
                "seed supports the qualified 8K/12K/32K layouts");
  constexpr int kRetainVecs = kRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float* srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[];  // NSUB * NB ints

  // pass 1: min/max of the row's FINITE scores (vectorized). Ignore any -inf
  // padding in diagnostic full-row logits so it cannot poison the range.
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  const auto acc = [&](const float s) {
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // Retain prefix logits across reduction, histogram construction, and emission.
  // Generic tail lanes contain -inf and are ignored.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "retained random seed requires a qualified CTA size");
  static_assert(BT % (NSUB * 32) == 0,
                "each seed sub-histogram must own whole warps");
  float4 retained[kRetainVecs];
  if (head == kRetainedHead) {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  } else {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      if (j + 3 < head) {
        s4 = *reinterpret_cast<const float4*>(srow + j);
      } else {
        if (j < head) s4.x = srow[j];
        if (j + 1 < head) s4.y = srow[j + 1];
        if (j + 2 < head) s4.z = srow[j + 2];
      }
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
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
  o -= headroom * span;  // forward (above-max) drift headroom
  float inv = (NB - 1) / (span * (1.0f + headroom));
  const float vth = -o * inv;

  // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
  // smem atomic conflicts, vectorized loads.
  for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
  __syncthreads();
  int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    // Match the emitter's FMA: separate rounding can certify a rejected boundary.
    const float bq = fmaf(-s, inv, vth);
    int b = static_cast<int>(bq);
    return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
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
    // Warm-start from genuine seed records; the suffix must not recount them.
    for (int b = tid; b < NB; b += BT)
      bcount_out[(size_t)row * NB + b] = s_hist[b];
  }
  // Keep one bucket scale for the histogram, gate, seed, and suffix.
  // Disjoint prefix ranges give exactly one owner of the first bin reaching K.
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
  // The compiler may vector-load s_wsum beside s_th. Complete every read
  // before publishing s_th, even though the scalar arrays do not overlap.
  __syncthreads();
  if (tid < NB && excl < K && K <= incl) s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  // Warp reservations avoid a CTA-wide prefix scan/barrier for every float4.
  __shared__ int emitted;
  if (tid == 0) emitted = 0;
  __syncthreads();
  const float edge = float(s_th + 1);
  const uint64_t row_base = uint64_t(row) * cand_cap;
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const int j = tid * 4 + it * BT * 4;
    const float4 s4 = retained[it];
    const float scores[4] = {s4.x,s4.y,s4.z,s4.w};
    float codes[4];
    bool pass[4];
    int count = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      codes[k] = fmaf(-scores[k], inv, vth);
      pass[k] = j + k < head && isfinite(scores[k]) &&
                __float_as_int(codes[k]) < __float_as_int(edge);
      count += pass[k];
    }
    int prefix = count;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu,prefix,d);
      if (lane >= d) prefix += v;
    }
    const int total = __shfl_sync(0xffffffffu,prefix,31);
    int base = 0;
    if (lane == 0 && total) base = atomicAdd(&emitted,total);
    base = __shfl_sync(0xffffffffu,base,0) + prefix - count;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (pass[k]) {
        if (base < cand_cap)
          dsa_litetopk::store_candidate(cand_val+row_base+base,
              cand_idx+row_base+base,codes[k],j+k);
        ++base;
      }
    }
  }
  __syncthreads();
  if (tid == 0) cand_cnt[row] = emitted;
}

// Map virtual page positions to corpus indices and update candidate telemetry.
__global__ void map_topk_indices_litetopk_kernel(
    int32_t* __restrict__ out_idx, const int32_t* __restrict__ index_map,
    const int32_t* __restrict__ status,
    int64_t total, int rows, int index_map_size,
    const int32_t* __restrict__ cand_cnt,
    int32_t* __restrict__ stat_run_max, int32_t* __restrict__ stat_over,
    int stat_watermark) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Reject bad rows before mapping padded IDs; update count telemetry here too.
  int stat_local_max = 0;
  int stat_local_over = 0;
  for (int row = static_cast<int>(global_thread); row < rows;
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
    __shared__ int stat_smax[32], stat_shared_over[32];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
      stat_smax[wid] = stat_local_max;
      stat_shared_over[wid] = stat_local_over;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int m = 0, ov = 0;
      const int warps = (blockDim.x + 31) >> 5;
      for (int w = 0; w < warps; ++w) {
        m = stat_smax[w] > m ? stat_smax[w] : m;
        ov += stat_shared_over[w];
      }
      if (m > 0) atomicMax(stat_run_max, m);
      if (ov > 0) atomicAdd(stat_over, ov);
    }
  }
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = (index_map[physical_idx >> 6] << 6) |
        (physical_idx & 63);
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = original_idx;

  }
}

static int compute_smem_bytes() {
  const int esz_fp8 = 1, esz_f32 = 4;
  const int smem_q = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
  const int smem_w = BLOCK_Q * NUM_HEADS * esz_f32;
  const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
  const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
  // Include the per-KV-stage scale-factor publication barriers.
  const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 3 +
                           (MATH_THREADS / 128) * dsa_litetopk::kUmmaStages * 2;
  const int smem_barriers = num_barriers * 8;
  const int smem_slots =
      4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
  constexpr int emit_record_bytes = (int)sizeof(uint32_t);
  const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                         ((int)sizeof(int32_t) + dsa_litetopk::kEmitLaneSlots *
                                                     32 * emit_record_bytes);
  const int smem_hist = BLOCK_Q * 256 * (int)sizeof(int32_t);
  return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
         NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks + smem_barriers +
         smem_slots + smem_warpq + smem_hist;
}

void launch_seed_prep(const float* slog, int64_t slog_stride, int Q, int head,
                      int NB, int K, float headroom, float* origin,
                      float* inv_delta, int32_t* th_bucket,
                      CandidateValue* cand_val, int32_t* cand_idx,
                      int32_t* cand_cnt, int cand_cap, int32_t* bcount,
                      cudaStream_t stream) {
  const int seed_smem = (head > 12288 ? 16 : 4) * NB * static_cast<int>(sizeof(int));
  if (head > 12288) {
    seed_prep_kernel<32768, 512><<<Q,512,seed_smem,stream>>>(
        slog,slog_stride,head,NB,K,headroom,origin,inv_delta,th_bucket,
        cand_val,cand_idx,cand_cnt,cand_cap,bcount);
  } else if (head == 12288) {
    seed_prep_kernel<12288, kSeed12Threads>
        <<<Q, kSeed12Threads, seed_smem, stream>>>(
            slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
            th_bucket, cand_val, cand_idx, cand_cnt, cand_cap, bcount);
  } else {
    seed_prep_kernel<8192, kSeedThreads>
        <<<Q, kSeedThreads, seed_smem, stream>>>(
            slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
            th_bucket, cand_val, cand_idx, cand_cnt, cand_cap, bcount);
  }
}

// Fused seed/prep: sample scores -> (origin, inv_delta, th_bucket, cand_val,
// cand_idx, cand_cnt, bcount), everything the scan needs, in one launch.
void seed_prep_litetopk_(torch::Tensor slog, int64_t num_buckets64,
                         int64_t topk64, int64_t cand_cap64,
                         double headroom,
                         torch::Tensor origin, torch::Tensor inv_delta,
                         torch::Tensor th_bucket, torch::Tensor bcount,
                         torch::Tensor cand_val, torch::Tensor cand_idx,
                         torch::Tensor cand_cnt) {
  TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
  check_tensors({origin, inv_delta}, slog, torch::kFloat);
  check_tensors({th_bucket, bcount, cand_idx, cand_cnt}, slog, torch::kInt);
  check_tensors({cand_val}, slog, torch::kHalf);
  TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
  TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
  const int Q = (int)slog.size(0);
  const int head = (int)slog.size(1);
  const int NB = (int)num_buckets64;
  const int K = (int)topk64;
  const int cap = (int)cand_cap64;
  TORCH_CHECK(head >= K && (head <= 8192 || head == 12288 ||
              (head > 12288 && head <= 32768)),
              "seed prep requires topk <= sample <= 8192, sample=12288, or 12288 < sample <= 32768");
  TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
  TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() >= Q &&
                  inv_delta.dim() == 1 && inv_delta.numel() >= Q &&
                  th_bucket.dim() == 1 && th_bucket.numel() >= Q &&
                  cand_cnt.dim() == 1 && cand_cnt.numel() >= Q,
              "origin/inv_delta/th_bucket/cand_cnt must cover Q rows");
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) >= Q &&
                  cand_val.size(1) == cap &&
                  cand_idx.sizes() == cand_val.sizes(),
              "cand_val/cand_idx must be [>=Q,cand_cap]");
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) >= Q && bcount.size(1) == NB,
              "bcount must be [>=Q,num_buckets]");
  TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                  (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
              "slog rows must be 16B aligned");
  const c10::cuda::CUDAGuard device_guard(slog.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_seed_prep(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), cap,
                   bcount.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Scan the suffix into buffers initialized by seed_prep_litetopk.
void mqa_logits_dsa_paged_litetopk_(
    torch::Tensor q, torch::Tensor kv, torch::Tensor block_table, torch::Tensor page_order,
    int64_t seq_len_kv, torch::Tensor weights, torch::Tensor cu_start, torch::Tensor cu_end,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor th_bucket,
    torch::Tensor cand_val, torch::Tensor cand_idx, torch::Tensor cand_cnt,
    torch::Tensor bcount, int64_t num_buckets64, int64_t topk64) {
  constexpr int kPageSize = 64;
  check_tensors({q}, q, torch::kFloat8_e4m3fn);
  check_tensors({kv}, q, torch::kUInt8);
  check_tensors({weights, origin, inv_delta}, q, torch::kFloat);
  check_tensors({cu_start, cu_end, th_bucket, cand_idx, cand_cnt, bcount,
                 block_table, page_order}, q, torch::kInt);
  check_tensors({cand_val}, q, torch::kHalf);

  TORCH_CHECK(q.dim() == 3, "q must be [Q,32,128]");
  TORCH_CHECK(kv.dim() == 3, "invalid KV rank");
  const int seq_len = static_cast<int>(q.size(0));
  TORCH_CHECK(seq_len > 0 && seq_len_kv > 0, "Q and S must be nonzero");
  TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM &&
                  kv.size(1) == kPageSize && kv.size(2) == 132,
              "GLM paged path requires H=32, D=128");
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == seq_len &&
                  weights.size(1) == NUM_HEADS,
              "weights must be [Q,32]");
  TORCH_CHECK(cu_start.dim() == 1 && cu_start.numel() == seq_len &&
                  cu_end.dim() == 1 && cu_end.numel() == seq_len,
              "cu_start/cu_end must have Q elements");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() == seq_len &&
                  inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
                  th_bucket.dim() == 1 && th_bucket.numel() == seq_len,
              "origin/inv_delta/th_bucket must have Q elements");
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) == seq_len &&
                  cand_idx.sizes() == cand_val.sizes(),
              "cand_val/cand_idx must be [Q,cand_cap]");
  const int cand_cap = static_cast<int>(cand_val.size(1));
  const int num_buckets = static_cast<int>(num_buckets64);
  const int topk = static_cast<int>(topk64);
  TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
              "paged path requires 3 <= num_buckets <= 256");
  TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1,cand_cap]");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len,
              "cand_cnt must have Q elements");
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) == seq_len &&
                  bcount.size(1) == num_buckets,
              "bcount must be [Q,num_buckets]");

  c10::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int esz_fp8 = 1;
  const int esz_f32 = 4;
  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                      HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
                      BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  CUtensorMap tm_kv, tm_ks;
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1 && page_order.dim() == 1 &&
      block_table.numel() >= (seq_len_kv + kPageSize - 1) / kPageSize &&
      page_order.numel() >= (seq_len_kv + kPageSize - 1) / kPageSize,
      "invalid paged scan tables");
  tm_kv = make_paged_values(kv.data_ptr(), kv.size(0), kPageSize, kv.stride(0));
  tm_ks = make_2d(kv.data_ptr<uint8_t>() + kPageSize * HEAD_DIM,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, kPageSize, kv.size(0),
      kPageSize, 1, kv.stride(0) / 4, 0);
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  // seq words + shadow words + pending/stale/last rows + per-lane
  // progress cursors, appended after the re-enabled histogram region.
  constexpr int kRingScratchBytes =
      2 * 8 * BLOCK_Q * static_cast<int>(sizeof(uint32_t)) +
      3 * BLOCK_Q * static_cast<int>(sizeof(int32_t)) +
      8 * BLOCK_Q * 32;
  const int smem = compute_smem_bytes() +
                   kRingScratchBytes;
  auto kernel =
      &dsa_litetopk::sm100_dsa_litetopk<
          NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES,
          NUM_KV_STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len_kv),
      reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(),
      // Seed histogram is valid only when the suffix excludes the seed prefix.
      bcount.data_ptr<int32_t>(), static_cast<uint32_t>(num_buckets),
      static_cast<uint32_t>(topk),
      candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
      cand_cnt.data_ptr<int32_t>(), static_cast<uint32_t>(cand_cap), tm_q,
      tm_kv, tm_ks, tm_w,
      block_table.data_ptr<int32_t>(), page_order.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K=2048 physical selector; candidate capacity follows tensor stride.
// The caller reuses dead boundary metadata as contiguous R*5 diagnostic integers.
void h2048_safe_topk_out_litetopk_(torch::Tensor cand_val,
                                   torch::Tensor cand_idx,
                                   torch::Tensor cand_cnt,
                                   torch::Tensor out_idx, torch::Tensor status,
                                   torch::Tensor diagnostic_scratch,
                                   int64_t index_limit64) {
  constexpr int kDiagnosticIntsPerRow = 5;
  static_assert(h2048_safe_topk::kBins >= kDiagnosticIntsPerRow,
                "boundary_meta must have room for h2048 diagnostic scratch");
  static_assert(
      dsa_litetopk::kCandidateIndexBits == 20,
      "h2048 safe selector requires the production 20-bit physical ID ABI");
  static_assert(sizeof(CandidateValue) == sizeof(uint16_t));

  check_tensors({cand_val}, cand_val, torch::kHalf);
  check_tensors({cand_idx, cand_cnt, out_idx, status, diagnostic_scratch},
                cand_val, torch::kInt);
  TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
              "h2048 candidate tensors must be [R,CAP]");
  const int64_t rows64 = cand_val.size(0);
  TORCH_CHECK(rows64 > 0 && rows64 <= std::numeric_limits<int>::max() &&
                  cand_val.size(1) >= h2048_safe_topk::kMinCap &&
                  cand_val.size(1) <= h2048_safe_topk::kMaxCap,
              "h2048 safe selector requires [R,CAP] candidates where CAP meets "
              "the qualified selection-width floor and is <= 1M");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == rows64 &&
                  status.dim() == 1 && status.numel() == rows64,
              "h2048 cand_cnt/status must have R elements");
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.size(0) == rows64 &&
                  out_idx.size(1) == h2048_safe_topk::kTopK,
              "h2048 safe selector output must be [R,2048]");
  TORCH_CHECK(
      diagnostic_scratch.numel() >= rows64 * kDiagnosticIntsPerRow,
      "h2048 diagnostic scratch must contain at least R*5 int32 values");
  TORCH_CHECK(
      index_limit64 > 0 &&
          index_limit64 <= (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
      "h2048 index_limit must be in [1,1M]");

  const int rows = static_cast<int>(rows64);
  const int cap = static_cast<int>(cand_val.size(1));
  const int topk = static_cast<int>(out_idx.size(1));
  const int index_limit = static_cast<int>(index_limit64);
  const c10::cuda::CUDAGuard device_guard(cand_val.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  h2048_safe_topk::coarse_tiering_topk_kernel<256, 4096, 8>
      <<<rows, 256, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(cand_val.data_ptr<at::Half>()),
          cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
          out_idx.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
          diagnostic_scratch.data_ptr<int32_t>(), rows, cap, topk, index_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Winner mapping with per-call candidate-count telemetry folded in:
// run_max = atomicMax over cand_cnt, over_events += count(cand_cnt > wm).
void map_topk_stats_litetopk_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor cand_cnt, torch::Tensor run_max, torch::Tensor over_events,
    int64_t watermark64, int64_t index_limit) {
  check_tensors({out_idx, index_map, status, cand_cnt, run_max, over_events},
                out_idx, torch::kInt);
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.numel() > 0,
              "out_idx must be nonempty [R,K]");
  TORCH_CHECK(status.numel() == out_idx.size(0) &&
                  cand_cnt.numel() >= out_idx.size(0) && run_max.numel() >= 1 &&
                  over_events.numel() >= 1, "status/stats tensors too small");
  TORCH_CHECK(index_map.dim() == 1 && index_limit > 0 &&
                  index_limit <= (1 << 20) && index_map.numel() * 64 >= index_limit,
              "invalid winner page map");
  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_topk_indices_litetopk_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(), total,
      static_cast<int>(out_idx.size(0)), index_limit,
      cand_cnt.data_ptr<int32_t>(),
      run_max.data_ptr<int32_t>(), over_events.data_ptr<int32_t>(),
      static_cast<int>(watermark64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_paged_sample_out", &gather_paged_sample_out);
  m.def("mqa_logits_dsa_paged_litetopk_", &mqa_logits_dsa_paged_litetopk_);
  m.def(
      "candidate_fp24_global_litetopk", []() { return true; },
      "Reports the production high24 FP32 local/global candidate ABI");
  m.def(
      "candidate_value_u16_litetopk", []() { return true; },
      "Reports the packed six-byte candidate ABI");
  m.def("seed_prep_litetopk_", &seed_prep_litetopk_,
        "In-place fused sample prep (caller-owned buffers)",
        pybind11::arg("slog"), pybind11::arg("num_buckets"),
        pybind11::arg("topk"), pybind11::arg("cand_cap"),
        pybind11::arg("headroom"),
        pybind11::arg("origin"), pybind11::arg("inv_delta"),
        pybind11::arg("th_bucket"), pybind11::arg("bcount"),
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"));
  m.def("h2048_safe_topk_out_litetopk_", &h2048_safe_topk_out_litetopk_,
        "Single h2048 physical selector with cached or streamed boundaries",
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"), pybind11::arg("out_idx"),
        pybind11::arg("status"), pybind11::arg("diagnostic_scratch"),
        pybind11::arg("index_limit"));
  m.def("map_topk_stats_litetopk_", &map_topk_stats_litetopk_,
        "Page mapping with fused candidate-count telemetry",
        pybind11::arg("out_idx"), pybind11::arg("index_map"), pybind11::arg("status"),
        pybind11::arg("cand_cnt"),
        pybind11::arg("run_max"), pybind11::arg("over_events"), pybind11::arg("watermark"),
        pybind11::arg("index_limit"));

}
