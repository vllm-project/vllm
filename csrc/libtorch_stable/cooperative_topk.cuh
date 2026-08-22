/*
 * Cooperative TopK kernel for DSA Indexer
 */

#ifndef COOPERATIVE_TOPK_CUH_
#define COOPERATIVE_TOPK_CUH_

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <algorithm>
#include <cstdint>

#include "topk_histogram_4096.cuh"

namespace vllm {
namespace cooperative {

namespace hist4096 = topk_histogram_4096;

constexpr uint32_t kHistBits = 10;
constexpr uint32_t kHistBins = 1 << kHistBits;
using ExactRadix = hist4096::ExactRadixTraits<true>;
constexpr uint32_t kExactHistBins = ExactRadix::kBins;
constexpr uint32_t kMaxTopK = 2048;
// Retain this many threshold-bin candidates before exact recovery.
constexpr uint32_t kCoarseTieCapacity = kMaxTopK;

constexpr uint32_t kElemPerStage = 16;
constexpr uint32_t kSizePerStage =
    kElemPerStage * hist4096::kBlockSize;  // 16384

// CS=4 two-pass path uses two TMA stages as a double buffer.
constexpr uint32_t kStreamingStagesCS4 = 2;
// CS=8/16 fused paths keep all loaded TMA stages resident in smem.
constexpr uint32_t kFusedStagesCS8 = 2;
constexpr uint32_t kFusedStagesCS16 = 2;

// CS=4 single-pass path
constexpr uint32_t kMaxSinglePassStages = 3;
constexpr uint32_t kMaxSinglePassPerBlock =
    kMaxSinglePassStages * kSizePerStage;  // 49152

template <uint32_t TopK = 1024>
struct CooperativeTopKParams {
  const float* __restrict__ input;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ lengths;
  hist4096::Tie* __restrict__ tie_ws;  // per-row tie workspace, see
                                       // kTieWsPerRow
  uint32_t num_rows, stride;
};

// ============================================================================
// Cooperative helpers
// ============================================================================

// only CS adjacent lanes participate (sub-warp reduce), in opposite to
// warp_reduce_sum_full
template <uint32_t N>
__device__ __forceinline__ uint32_t warp_reduce_sum_subN(uint32_t v) {
#pragma unroll
  for (uint32_t m = N >> 1; m > 0; m >>= 1)
    v += __shfl_xor_sync(0xFFFFFFFF, v, m, 32);
  return v;
}

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ uint32_t extract_coarse_bin(float x) {
  return hist4096::extract_coarse_bin_N<kHistBits>(x);
}

// Records how far the cheap overflow probe advanced exact selection.
enum class OverflowProbeStatus : uint32_t {
  kFullRescan = 0,
  kKnownPivot = 1,
  kFirstDigitReady = 2,
};

// Bounds one coarse FP16 bin in the monotonic ordered-FP32 key space.
struct CoarseBinRange {
  uint32_t base_key;
  bool finite;
};

struct OverflowProbeSummary {
  uint32_t min_key;
  uint32_t max_key;
  uint32_t flags;
};

// Reconstruct exact ordered-FP32 boundaries from reduced-precision keys.
__device__ __forceinline__ uint32_t ordered_fp32_from_fp16_key(
    uint16_t key) {
  const uint16_t bits = (key & 0x8000u)
                            ? static_cast<uint16_t>(key & 0x7FFFu)
                            : static_cast<uint16_t>(~key);
  return hist4096::convert_to_uint32_v2(
      __half2float(__ushort_as_half(bits)));
}

__device__ __forceinline__ uint32_t ordered_fp32_from_bf16_key(
    uint32_t key) {
  return (key << 16) | ((key & 0x8000u) ? 0u : 0xFFFFu);
}

__device__ __forceinline__ CoarseBinRange coarse_bin_range(
    uint32_t coarse_bin) {
  const uint16_t lower_key =
      static_cast<uint16_t>(coarse_bin << (16 - kHistBits));
  const uint16_t lower_bits =
      (lower_key & 0x8000u)
          ? static_cast<uint16_t>(lower_key & 0x7FFFu)
          : static_cast<uint16_t>(~lower_key);
  const uint32_t ordered_lower = ordered_fp32_from_fp16_key(lower_key);
  return {.base_key = ordered_lower > 8192 ? ordered_lower - 8192 : 0,
          .finite = (lower_bits & 0x7C00u) != 0x7C00u};
}

__device__ __forceinline__ void mbarrier_init(uint64_t* a, uint32_t n) {
  cuda::ptx::mbarrier_init(a, n);
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* a, uint32_t p) {
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed,
                                              cuda::ptx::scope_cta, a, p));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* a,
                                                          uint32_t t) {
  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed,
                                       cuda::ptx::scope_cta,
                                       cuda::ptx::space_shared, a, t);
}
__device__ __forceinline__ void tma_load(void* d, const void* s, uint32_t n,
                                         uint64_t* m) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global, d,
                           s, n, m);
}

// ============================================================================
// DSMEM histogram reduce
// ============================================================================

template <uint32_t CS, uint32_t NumBins>
__device__ __forceinline__ void dsmem_hist_reduce(uint32_t* histogram) {
  static_assert(NumBins % CS == 0);
  // Fold the distributed per-rank bins into cluster-visible totals.
  auto cluster = cooperative_groups::this_cluster();
  cluster.sync();
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;
  constexpr uint32_t kLocal = NumBins / CS;
  const uint32_t off = kLocal * rank;
#pragma unroll
  for (uint32_t bin = tx; bin < NumBins;
       bin += hist4096::kBlockSize) {
    auto* addr = &histogram[off + bin / CS];
    auto* src = cluster.map_shared_rank(addr, bin % CS);
    *src = warp_reduce_sum_subN<CS>(*src);
  }
  cluster.sync();
}

// ============================================================================
// Find threshold from reduced histogram
// ============================================================================

// NOTE: caller must ensure a cluster.sync() or __syncthreads() happened
// before calling this, so warp_sum writes are visible across warps.
// The first internal __syncthreads() is still needed for the warp_sum exchange.
template <uint32_t TopK>
__device__ __forceinline__ void find_threshold(uint32_t* histogram,
                                               uint32_t* warp_sum,
                                               uint32_t* counter_gt,
                                               uint32_t* counter_eq,
                                               hist4096::MatchBin* match) {
  const auto tx = threadIdx.x;
  const auto li = tx % hist4096::kWarpSize, wi = tx / hist4096::kWarpSize;
  const auto value = tx < kHistBins ? histogram[tx] : 0;
  const auto winc = hist4096::warp_inclusive_sum(li, value);
  if (li == hist4096::kWarpSize - 1) warp_sum[wi] = winc;
  __syncthreads();
  const auto tmp = warp_sum[li];
  const auto total = hist4096::warp_reduce_sum_full(tmp);
  auto pfx = hist4096::warp_reduce_sum_full(li < wi ? tmp : 0) + winc;
  const auto above = total - pfx;
  if (tx < kHistBins && above < TopK && above + value >= TopK) {
    *counter_gt = *counter_eq = 0;
    *match = {.bin = tx, .above_count = above, .equal_count = value};
  }
  __syncthreads();
}

// Locate the target digit in a 2,048-bin exact-radix histogram.
__device__ __forceinline__ void find_threshold_exact(
    uint32_t* histogram, uint32_t* warp_sum, uint32_t target,
    hist4096::MatchBin* match) {
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % hist4096::kWarpSize;
  const uint32_t warp = tx / hist4096::kWarpSize;
  const uint32_t bin0 = 2 * tx;
  const uint32_t bin1 = bin0 + 1;
  const uint32_t count0 = histogram[bin0];
  const uint32_t count1 = histogram[bin1];
  const uint32_t local = count0 + count1;
  const uint32_t warp_inclusive =
      hist4096::warp_inclusive_sum(lane, local);
  if (lane == hist4096::kWarpSize - 1) {
    warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    warp_sum[lane] =
        hist4096::warp_inclusive_sum(lane, warp_sum[lane]);
  }
  __syncthreads();

  const uint32_t total = warp_sum[hist4096::kNumWarps - 1];
  const uint32_t before_warp = warp == 0 ? 0 : warp_sum[warp - 1];
  const uint32_t before_thread = before_warp + warp_inclusive - local;
  uint32_t above = total - before_thread - local;
  if (above < target && above + count1 >= target) {
    *match = {.bin = bin1, .above_count = above, .equal_count = count1};
  } else {
    above += count1;
    if (above < target && above + count0 >= target) {
      *match = {.bin = bin0, .above_count = above, .equal_count = count0};
    }
  }
  __syncthreads();
}

// Streams data through shared memory in chunks, processing each chunk before
// loading the next overwrites each buffer after processing it (the epilogue
// prefetch loads the next chunk into the same slot)
template <typename SmemType, uint32_t kStages, uint32_t kBinBits,
          bool kIsScatter>
__device__ void tma_stream_pass(const float* scores, uint32_t length,
                                uint32_t thr_bin, int32_t* indices,
                                uint32_t* phases, SmemType* smem) {
  const auto tx = threadIdx.x;
  const auto lane = tx % hist4096::kWarpSize;
  const auto ni =
      (length + kSizePerStage - 1) / kSizePerStage;  // total stages needed
  const auto la =
      (length + 3u) & ~3u;  // length rounded up to float4 (TMA alignment)
  const auto pass =
      kIsScatter ? 1 : 0;  // barrier dim: [0] for histogram, [1] for scatter

  // Prologue: issue initial TMA loads - prefill the pipeline
  if (tx == 0) {
#pragma unroll
    for (uint32_t i = 0; i < kStages; i++) {
      if (i >= ni) {
        break;
      }
      const auto o = i * kSizePerStage;
      const auto sz = min(kSizePerStage, la - o) * sizeof(float);
      tma_load(smem->score_buffer[i], scores + o, sz,
               &smem->barrier[pass][i]);  // cp.async.bulk is non-blocking
      mbarrier_arrive_expect_tx(&smem->barrier[pass][i], sz);
    }
  }

  // Main loop: process stages
  for (uint32_t it = 0; it < ni; it++) {
    const auto b = it % kStages;  // which buffer slot (0 or 1)
    const auto o = it * kSizePerStage;
    const auto sz = min(kSizePerStage, length - o);

    if (lane == 0) {
      mbarrier_wait(&smem->barrier[pass][b],
                    phases[b] & 1);  // wait for the data
    }
    phases[b]++;  // advances the phase for next time this slot is reused
    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * hist4096::kBlockSize;
      if (li >= sz) {
        break;
      }
      const auto sc = smem->score_buffer[b][li];
      const auto bn = hist4096::extract_coarse_bin_N<kBinBits>(sc);
      if constexpr (kIsScatter) {  // compile-time branch
        // Scatter pass: place above-threshold and collect ties
        const auto gi = o + li;
        if (bn > thr_bin) {
          indices[atomicAdd(&smem->counter_gt, 1)] = gi;
        } else if (bn == thr_bin) {
          const auto p = atomicAdd(&smem->counter_eq, 1);
          if (p < kCoarseTieCapacity) {
            smem->tie_buffer[p] = {gi, sc};
          }
        }
      } else {
        // Histogram pass: just count
        atomicAdd(&smem->histogram[bn], 1);
      }
    }
    __syncthreads();  // ensures all threads finished processing their buffer
                      // before next TMA load

    // Epilogue: issue next TMA load
    if (tx == 0 && it + kStages < ni) {
      const auto no = (it + kStages) * kSizePerStage;
      const auto nsz = min(kSizePerStage, la - no) * sizeof(float);
      tma_load(smem->score_buffer[b], scores + no, nsz,
               &smem->barrier[pass][b]);
      mbarrier_arrive_expect_tx(&smem->barrier[pass][b], nsz);
    }
  }
}

// ============================================================================
// Fused path: single TMA pass, rescan smem for scatter
// ============================================================================

// Fused shared memory layout for cluster cooperative paths.
// kPasses=1 for single-pass (CS=8, CS=4 singlepass), kPasses=2 for two-pass
// (CS=4).
template <uint32_t kStages, uint32_t kPasses = 1>
struct SmemFused {
  uint64_t barrier[kPasses][kStages];
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) hist4096::MatchBin match;
  uint32_t warp_sum[hist4096::kNumWarps];
  union {
    uint32_t histogram[kExactHistBins];
    hist4096::Tie tie_buffer[kMaxTopK];
  };
  alignas(128) float score_buffer[kStages][kSizePerStage];
};

using Smem4 = SmemFused<kStreamingStagesCS4, 2>;
using SmemSinglePass = SmemFused<kMaxSinglePassStages>;

// Build one radix digit when the coarse bin cannot bound the FP32 keys.
__device__ void build_exact_histogram(const float* __restrict__ scores,
                                      uint32_t length, uint32_t prefix,
                                      uint32_t round, uint32_t* histogram) {
  const uint32_t shift = ExactRadix::shift(round);
  const uint32_t digit_mask = ExactRadix::digit_mask(round);
  const uint32_t prefix_mask = ExactRadix::prefix_mask(round);
  hist4096::scan_scores<hist4096::kBlockSize, 4>(
      scores, length, [&](uint32_t, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        if ((ordered & prefix_mask) == prefix) {
          atomicAdd(&histogram[(ordered >> shift) & digit_mask], 1);
        }
      });
}

// Reuse resident TMA data when possible; otherwise rescan global memory.
template <bool UseResident, typename SmemType, typename Visit>
__device__ __forceinline__ void for_each_partition_score(
    const float* __restrict__ scores, uint32_t length, SmemType* smem,
    Visit visit) {
  if constexpr (UseResident) {
    const uint32_t tx = threadIdx.x;
    const uint32_t stages = (length + kSizePerStage - 1) / kSizePerStage;
    for (uint32_t stage = 0; stage < stages; ++stage) {
      const uint32_t offset = stage * kSizePerStage;
      const uint32_t size = min(kSizePerStage, length - offset);
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const uint32_t local = tx + i * hist4096::kBlockSize;
        if (local >= size) break;
        visit(offset + local, smem->score_buffer[stage][local]);
      }
    }
  } else {
    hist4096::scan_scores<hist4096::kBlockSize, 4>(scores, length, visit);
  }
}

// A finite 10-bit FP16 bin spans fewer than 22 ordered-FP32 bits, so two
// radix-2048 passes are sufficient to select its exact pivot.
template <bool UseResident, typename SmemType>
__device__ void build_coarse_refine_histogram(
    const float* __restrict__ scores, uint32_t length, uint32_t coarse_bin,
    uint32_t base_key, uint32_t prefix, uint32_t round, SmemType* smem,
    uint32_t* histogram) {
  const uint32_t shift = round == 0 ? 11 : 0;
  const uint32_t prefix_mask = round == 0 ? 0u : 0xFFFFF800u;
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t, float score) {
        if (extract_coarse_bin(score) != coarse_bin) return;
        const uint32_t delta =
            hist4096::convert_to_uint32_v2(score) - base_key;
        if (round == 0 || (delta & prefix_mask) == prefix) {
          atomicAdd(&histogram[(delta >> shift) & 0x7FFu], 1);
        }
      });
}

template <bool UseResident, typename SmemType>
__device__ void collect_exact_candidates(
    const float* __restrict__ scores, uint32_t length, uint32_t pivot,
    uint32_t equal_limit, SmemType* smem, uint32_t* above_count,
    uint32_t* equal_count, int32_t* above_indices, int32_t* equal_indices) {
  for_each_partition_score<UseResident>(
      scores, length, smem, [&](uint32_t idx, float score) {
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        if (ordered > pivot) {
          above_indices[atomicAdd(above_count, 1)] =
              static_cast<int32_t>(idx);
        } else if (ordered == pivot) {
          const uint32_t pos = atomicAdd(equal_count, 1);
          if (pos < equal_limit) {
            equal_indices[pos] = static_cast<int32_t>(idx);
          }
        }
      });
}

struct ClusterCandidateOffsets {
  uint32_t above;
  uint32_t equal;
  uint32_t total_above;
};

template <uint32_t CS>
__device__ __forceinline__ ClusterCandidateOffsets candidate_offsets(
    uint32_t local_above, uint32_t local_equal) {
  constexpr uint32_t kCountBits = 16;
  constexpr uint32_t kCountMask = (1u << kCountBits) - 1;
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;
  auto cluster = cooperative_groups::this_cluster();
  __shared__ uint32_t counts[CS];
  __shared__ uint32_t packed_offset;
  __shared__ uint32_t total_above;

  if (tx < CS) {
    auto* dst = cluster.map_shared_rank(counts, tx);
    dst[rank] = (local_equal << kCountBits) | local_above;
  }
  cluster.sync();

  if (tx == 0) {
    uint32_t above = 0;
    uint32_t equal = 0;
    for (uint32_t i = 0; i < CS; ++i) {
      if (i == rank) packed_offset = (equal << kCountBits) | above;
      above += counts[i] & kCountMask;
      equal += counts[i] >> kCountBits;
    }
    total_above = above;
  }
  __syncthreads();

  return {
      .above = packed_offset & kCountMask,
      .equal = packed_offset >> kCountBits,
      .total_above = total_above,
  };
}

// Build the first exact radix digit while probing an overflowing coarse bin.
// The probe either resolves an exact FP16-grid pivot or leaves enough state
// for recovery to resume at the second radix digit.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_reduced_precision_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  constexpr uint32_t kBf16Bins = 256;
  constexpr uint32_t kDigitBits = 11;
  constexpr uint32_t kDeltaLimit = 1u << (2 * kDigitBits);
  constexpr uint32_t kValid = 1;
  constexpr uint32_t kHasNonBf16 = 2;
  constexpr uint32_t kHasNonFp16 = 4;
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;
  const uint32_t needed = TopK - coarse_above;
  auto cluster = cooperative_groups::this_cluster();
  __shared__ uint32_t rank_min[CS];
  __shared__ uint32_t rank_max[CS];
  __shared__ uint32_t rank_flags[CS];
  __shared__ OverflowProbeSummary summary;
  auto* exact_histogram = reinterpret_cast<uint32_t*>(scratch);

  const auto range = coarse_bin_range(coarse_bin);

  for (uint32_t bin = tx; bin < kBf16Bins;
       bin += hist4096::kBlockSize) {
    smem->histogram[bin] = 0;
  }
  for (uint32_t bin = tx; bin < kExactHistBins;
       bin += hist4096::kBlockSize) {
    exact_histogram[bin] = 0;
  }
  if (tx == 0) {
    smem->warp_sum[0] = 0xFFFFFFFFu;
    smem->warp_sum[1] = 0;
    smem->warp_sum[2] = range.finite;
    smem->warp_sum[3] = 0;
  }
  __syncthreads();

  uint32_t thread_min = 0xFFFFFFFFu;
  uint32_t thread_max = 0;
  uint32_t thread_invalid = 0;
  uint32_t thread_non_bf16 = 0;
  uint32_t thread_non_fp16 = 0;
  const auto visit = [&](uint32_t, float score) {
    const uint32_t bin = extract_coarse_bin(score);
    if (bin != coarse_bin) return;

    const uint32_t raw = __float_as_uint(score);
    const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
    thread_min = min(thread_min, ordered);
    thread_max = max(thread_max, ordered);
    const bool is_bf16 = (raw & 0xFFFFu) == 0;
    if (is_bf16) {
      atomicAdd(&smem->histogram[(ordered >> 16) & 0xFFu], 1);
    } else {
      thread_non_bf16 = 1;
    }
    const __half half_score = __float2half_rn(score);
    thread_non_fp16 |=
        raw ^ __float_as_uint(__half2float(half_score));
    const uint32_t delta = ordered - range.base_key;
    const bool valid = ordered >= range.base_key && delta < kDeltaLimit;
    thread_invalid |= !valid;
    if (valid && !is_bf16) {
      atomicAdd(&exact_histogram[delta >> kDigitBits], 1);
    }
  };

  for_each_partition_score<UseResident>(
      row_input + my_start, my_len, smem, visit);
  atomicMin(&smem->warp_sum[0], thread_min);
  atomicMax(&smem->warp_sum[1], thread_max);
  atomicAnd(&smem->warp_sum[2], thread_invalid == 0);
  atomicOr(&smem->warp_sum[3],
           (thread_non_bf16 ? kHasNonBf16 : 0) |
               (thread_non_fp16 ? kHasNonFp16 : 0));
  __syncthreads();

  if (tx < CS) {
    auto* min_dst = cluster.map_shared_rank(rank_min, tx);
    auto* max_dst = cluster.map_shared_rank(rank_max, tx);
    auto* flags_dst = cluster.map_shared_rank(rank_flags, tx);
    min_dst[rank] = smem->warp_sum[0];
    max_dst[rank] = smem->warp_sum[1];
    flags_dst[rank] = smem->warp_sum[2] |
                      smem->warp_sum[3];
  }
  cluster.sync();

  if (tx == 0) {
    summary = {.min_key = 0xFFFFFFFFu, .max_key = 0, .flags = 0};
    uint32_t valid = kValid;
    uint32_t features = 0;
    for (uint32_t i = 0; i < CS; ++i) {
      summary.min_key = min(summary.min_key, rank_min[i]);
      summary.max_key = max(summary.max_key, rank_max[i]);
      valid &= rank_flags[i] & kValid;
      features |= rank_flags[i] & ~kValid;
    }
    summary.flags = valid | features;
  }
  __syncthreads();

  if (summary.min_key == summary.max_key) {
    if (tx == 0) {
      smem->match = {.bin = summary.min_key,
                     .above_count = needed,
                     .equal_count = static_cast<uint32_t>(
                         OverflowProbeStatus::kKnownPivot)};
    }
    __syncthreads();
    return OverflowProbeStatus::kKnownPivot;
  }

  const uint32_t min_bf16 = summary.min_key >> 16;
  const uint32_t max_bf16 = summary.max_key >> 16;
  if ((summary.flags & kHasNonBf16) == 0 &&
      max_bf16 - min_bf16 < kBf16Bins) {
    dsmem_hist_reduce<CS, kBf16Bins>(smem->histogram);
    if (tx == 0) {
      uint32_t above = 0;
      for (uint32_t key = max_bf16;; --key) {
        const uint32_t count = smem->histogram[key & 0xFFu];
        if (above < needed && above + count >= needed) {
          const uint32_t pivot = ordered_fp32_from_bf16_key(key);
          smem->match = {.bin = pivot,
                         .above_count = needed - above,
                         .equal_count = static_cast<uint32_t>(
                             OverflowProbeStatus::kKnownPivot)};
          break;
        }
        above += count;
        if (key == min_bf16) break;
      }
    }
    __syncthreads();
    return OverflowProbeStatus::kKnownPivot;
  }

  const bool bf16_keys_unambiguous = max_bf16 - min_bf16 < kBf16Bins;
  if ((summary.flags & kValid) != 0 && bf16_keys_unambiguous) {
    for (uint32_t key = min_bf16 + tx; key <= max_bf16;
         key += hist4096::kBlockSize) {
      const uint32_t count = smem->histogram[key & 0xFFu];
      if (count == 0) continue;
      const uint32_t ordered = ordered_fp32_from_bf16_key(key);
      const uint32_t delta = ordered - range.base_key;
      atomicAdd(&exact_histogram[delta >> kDigitBits], count);
    }
    __syncthreads();
    dsmem_hist_reduce<CS, kExactHistBins>(exact_histogram);
    find_threshold_exact(exact_histogram, smem->warp_sum, needed,
                         &smem->match);
  }

  if (tx == 0) {
    uint32_t status = 0;
    uint32_t pivot = 0;
    uint32_t remaining = 0;
    if ((summary.flags & kValid) != 0 && bf16_keys_unambiguous) {
      const uint32_t first_prefix = smem->match.bin << kDigitBits;
      remaining = needed - smem->match.above_count;
      if ((summary.flags & kHasNonFp16) == 0) {
        const uint32_t first_lower = range.base_key + first_prefix;
        const uint32_t first_upper = first_lower + (1u << kDigitBits) - 1;
        for (uint32_t low = 0; low < (1u << (16 - kHistBits)); ++low) {
          const uint16_t half_key =
              static_cast<uint16_t>((coarse_bin << (16 - kHistBits)) | low);
          const uint32_t candidate = ordered_fp32_from_fp16_key(half_key);
          if (candidate >= first_lower && candidate <= first_upper) {
            status = static_cast<uint32_t>(
                OverflowProbeStatus::kKnownPivot);
            pivot = candidate;
            break;
          }
        }
      }
      if (status == 0) {
        status = static_cast<uint32_t>(
            OverflowProbeStatus::kFirstDigitReady);
        pivot = first_prefix;
      }
    }
    smem->match = {
        .bin = pivot, .above_count = remaining, .equal_count = status};
  }
  __syncthreads();
  return static_cast<OverflowProbeStatus>(smem->match.equal_count);
}

// Arbitrary-FP32 overflow path: build the first exact radix digit directly.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_arbitrary_fp32_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  constexpr uint32_t kDigitBits = 11;
  const uint32_t tx = threadIdx.x;
  const uint32_t needed = TopK - coarse_above;
  auto* exact_histogram = reinterpret_cast<uint32_t*>(scratch);

  const auto range = coarse_bin_range(coarse_bin);
  if (!range.finite) {
    return OverflowProbeStatus::kFullRescan;
  }

  for (uint32_t bin = tx; bin < kExactHistBins;
       bin += hist4096::kBlockSize) {
    exact_histogram[bin] = 0;
  }
  __syncthreads();

  // A finite coarse bin is fully covered by the two 11-bit radix digits.
  for_each_partition_score<UseResident>(
      row_input + my_start, my_len, smem, [&](uint32_t, float score) {
        if (extract_coarse_bin(score) != coarse_bin) return;
        const uint32_t ordered = hist4096::convert_to_uint32_v2(score);
        const uint32_t delta = ordered - range.base_key;
        atomicAdd(&exact_histogram[delta >> kDigitBits], 1);
      });
  __syncthreads();

  dsmem_hist_reduce<CS, kExactHistBins>(exact_histogram);
  find_threshold_exact(exact_histogram, smem->warp_sum, needed, &smem->match);

  if (tx == 0) {
    const uint32_t first_prefix = smem->match.bin << kDigitBits;
    const uint32_t remaining = needed - smem->match.above_count;
    smem->match = {
        .bin = first_prefix,
        .above_count = remaining,
        .equal_count =
            static_cast<uint32_t>(OverflowProbeStatus::kFirstDigitReady)};
  }
  __syncthreads();
  return OverflowProbeStatus::kFirstDigitReady;
}

// Every CTA reads the same sample, making the choice cluster-uniform without a
// cluster synchronization. A missed arbitrary-FP32 input only takes the slower
// general probe; both paths remain exact.
__device__ __forceinline__ bool prefer_direct_fp32_probe(
    const float* __restrict__ row_input) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t use_direct;
  if (tx < hist4096::kWarpSize) {
    const float score = row_input[tx];
    const uint32_t raw = __float_as_uint(score);
    const bool is_bf16 = (raw & 0xFFFFu) == 0;
    const bool is_fp16 =
        raw == __float_as_uint(__half2float(__float2half_rn(score)));
    const uint32_t arbitrary_mask =
        __ballot_sync(0xFFFFFFFFu, !is_bf16 && !is_fp16);
    const uint32_t unequal_mask =
        __ballot_sync(0xFFFFFFFFu,
                      raw != __float_as_uint(row_input[0]));
    if (tx == 0) {
      use_direct = arbitrary_mask != 0 && unequal_mask != 0;
    }
  }
  __syncthreads();
  return use_direct != 0;
}

template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ OverflowProbeStatus probe_coarse_overflow(
    const float* __restrict__ row_input, uint32_t my_start, uint32_t my_len,
    uint32_t coarse_bin, uint32_t coarse_above, SmemType* smem,
    int32_t* scratch) {
  if (prefer_direct_fp32_probe(row_input)) {
    return probe_arbitrary_fp32_overflow<TopK, CS, UseResident>(
        row_input, my_start, my_len, coarse_bin, coarse_above, smem, scratch);
  }
  return probe_reduced_precision_overflow<TopK, CS, UseResident>(
      row_input, my_start, my_len, coarse_bin, coarse_above, smem, scratch);
}

template <uint32_t TopK, uint32_t CS, typename SmemType>
__device__ bool emit_staged_candidates(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t pivot, uint32_t remaining,
    uint32_t staged_above, uint32_t staged_candidates, SmemType* smem,
    int32_t* scratch) {
  const uint32_t tx = threadIdx.x;
  __shared__ uint32_t rank_staging_ok[CS];
  __shared__ uint32_t staging_ok;
  auto cluster = cooperative_groups::this_cluster();

  if (tx < CS) {
    auto* dst = cluster.map_shared_rank(rank_staging_ok, tx);
    dst[blockIdx.y] = staged_above + staged_candidates <= kMaxTopK;
  }
  cluster.sync();
  if (tx == 0) {
    staging_ok = 1;
    for (uint32_t i = 0; i < CS; ++i) {
      staging_ok &= rank_staging_ok[i];
    }
  }
  __syncthreads();
  if (staging_ok == 0) return false;

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();
  for (uint32_t i = tx; i < staged_candidates;
       i += hist4096::kBlockSize) {
    const int32_t idx = scratch[kMaxTopK - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        row_input[my_start + idx]);
    if (ordered > pivot) {
      atomicAdd(&smem->counter_gt, 1);
    } else if (ordered == pivot) {
      atomicAdd(&smem->counter_eq, 1);
    }
  }
  __syncthreads();

  const auto offsets = candidate_offsets<CS>(
      staged_above + smem->counter_gt, smem->counter_eq);
  for (uint32_t i = tx; i < staged_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = scratch[i] + my_start;
  }

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();
  for (uint32_t i = tx; i < staged_candidates;
       i += hist4096::kBlockSize) {
    const int32_t idx = scratch[kMaxTopK - 1 - i];
    const uint32_t ordered = hist4096::convert_to_uint32_v2(
        row_input[my_start + idx]);
    if (ordered > pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
      row_output[offsets.above + staged_above + pos] = idx + my_start;
    } else if (ordered == pivot) {
      const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
      if (offsets.equal + pos < remaining) {
        row_output[offsets.total_above + offsets.equal + pos] =
            idx + my_start;
      }
    }
  }
  return true;
}

template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ void exact_topk_rescan_cluster(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t my_len, uint32_t coarse_bin,
    uint32_t coarse_above, uint32_t start_round, uint32_t initial_prefix,
    uint32_t initial_remaining, bool has_known_pivot, uint32_t known_pivot,
    SmemType* smem, int32_t* s_topk) {
  const uint32_t tx = threadIdx.x;
  const auto range = coarse_bin_range(coarse_bin);

  uint32_t prefix = initial_prefix;
  uint32_t remaining = has_known_pivot
                           ? initial_remaining
                           : (start_round != 0
                                  ? initial_remaining
                                  : (range.finite ? TopK - coarse_above
                                                  : TopK));
  const uint32_t rounds = range.finite ? 2 : ExactRadix::kRounds;
  const bool stage_final_candidates =
      range.finite && start_round == 1 && my_len >= 16384;
  uint32_t staged_above = 0;
  uint32_t staged_candidates = 0;
  for (uint32_t round = start_round;
       round < (has_known_pivot ? start_round : rounds); ++round) {
    for (uint32_t bin = tx; bin < kExactHistBins;
         bin += hist4096::kBlockSize) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    if (range.finite) {
      if (stage_final_candidates && round == 1) {
        if (tx == 0) {
          smem->counter_gt = 0;
          smem->counter_eq = 0;
        }
        __syncthreads();
        const uint32_t interval_lower = range.base_key + prefix;
        const uint32_t interval_upper = interval_lower + 0x7FFu;
        for_each_partition_score<UseResident>(
            row_input + my_start, my_len, smem,
            [&](uint32_t idx, float score) {
              const uint32_t ordered =
                  hist4096::convert_to_uint32_v2(score);
              if (ordered > interval_upper) {
                const uint32_t pos = atomicAdd(&smem->counter_gt, 1);
                if (pos < kMaxTopK) {
                  s_topk[pos] = static_cast<int32_t>(idx);
                }
              } else if (ordered >= interval_lower) {
                const uint32_t pos = atomicAdd(&smem->counter_eq, 1);
                if (pos < kMaxTopK) {
                  s_topk[kMaxTopK - 1 - pos] = static_cast<int32_t>(idx);
                }
                atomicAdd(&smem->histogram[ordered - interval_lower], 1);
              }
            });
        __syncthreads();
        staged_above = smem->counter_gt;
        staged_candidates = smem->counter_eq;
      } else {
        build_coarse_refine_histogram<UseResident>(
            row_input + my_start, my_len, coarse_bin, range.base_key, prefix,
            round, smem, smem->histogram);
      }
    } else {
      build_exact_histogram(row_input + my_start, my_len, prefix, round,
                            smem->histogram);
    }
    __syncthreads();

    dsmem_hist_reduce<CS, kExactHistBins>(smem->histogram);
    find_threshold_exact(smem->histogram, smem->warp_sum, remaining,
                         &smem->match);
    const uint32_t shift = range.finite ? (round == 0 ? 11 : 0)
                                        : ExactRadix::shift(round);
    prefix |= smem->match.bin << shift;
    remaining -= smem->match.above_count;
  }

  const uint32_t pivot = has_known_pivot
                             ? known_pivot
                             : (range.finite ? range.base_key + prefix
                                             : prefix);

  if (stage_final_candidates && !has_known_pivot &&
      emit_staged_candidates<TopK, CS>(
          row_input, row_output, my_start, pivot, remaining, staged_above,
          staged_candidates, smem, s_topk)) {
    return;
  }

  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();

  int32_t* equal_indices = reinterpret_cast<int32_t*>(smem->tie_buffer);
  collect_exact_candidates<UseResident>(
      row_input + my_start, my_len, pivot, remaining, smem,
      &smem->counter_gt, &smem->counter_eq, s_topk, equal_indices);
  __syncthreads();

  const uint32_t local_above = smem->counter_gt;
  const uint32_t local_equal = min(smem->counter_eq, remaining);
  const auto offsets = candidate_offsets<CS>(local_above, local_equal);
  for (uint32_t i = tx; i < local_above; i += hist4096::kBlockSize) {
    row_output[offsets.above + i] = s_topk[i] + my_start;
  }
  const uint32_t equal_to_write = offsets.equal < remaining
                                      ? min(local_equal,
                                            remaining - offsets.equal)
                                      : 0;
  for (uint32_t i = tx; i < equal_to_write; i += hist4096::kBlockSize) {
    row_output[offsets.total_above + offsets.equal + i] =
        equal_indices[i] + my_start;
  }
}

// Keep all exceptional-path state behind one call boundary so the healthy
// large kernel retains the same live ranges as the original implementation.
template <uint32_t TopK, uint32_t CS, bool UseResident, typename SmemType>
__device__ __noinline__ void recover_coarse_overflow(
    const float* __restrict__ row_input, int32_t* __restrict__ row_output,
    uint32_t my_start, uint32_t my_len, SmemType* smem, int32_t* s_topk) {
  const uint32_t coarse_bin = smem->match.bin;
  const uint32_t coarse_above = smem->match.above_count;
  const auto probe_status = probe_coarse_overflow<TopK, CS, UseResident>(
      row_input, my_start, my_len, coarse_bin, coarse_above, smem, s_topk);
  const bool has_known_pivot =
      probe_status == OverflowProbeStatus::kKnownPivot;
  const uint32_t start_round =
      probe_status == OverflowProbeStatus::kFirstDigitReady ? 1 : 0;

  exact_topk_rescan_cluster<TopK, CS, UseResident>(
      row_input, row_output, my_start, my_len, coarse_bin, coarse_above,
      start_round, start_round == 0 ? 0 : smem->match.bin,
      smem->match.above_count, has_known_pivot,
      has_known_pivot ? smem->match.bin : 0, smem, s_topk);
}

// Keep the two-candidate-per-thread refinement out of the common kernel body.
// Inlining it increases register pressure even when the coarse bin contains at
// most 1,024 candidates and this branch is never executed.
template <uint32_t TopK>
__device__ __noinline__ void refine_large_coarse_ties(
    const hist4096::Tie* ties, uint32_t num_ties, uint32_t num_above,
    int32_t* output, void* smem) {
  static_assert(TopK <= hist4096::kBlockSize);
  hist4096::tie_handle_large<TopK, kCoarseTieCapacity>(
      ties, num_ties, num_above, output, smem);
}

// Keep the second candidate copy out of the one-candidate-per-thread path.
template <uint32_t TopK>
__device__ __noinline__ void copy_extra_coarse_ties(
    const hist4096::Tie* tie_buffer, uint32_t num_ties,
    uint32_t prefix_equal, uint32_t total_above, uint32_t row_start,
    hist4096::Tie* tie_ws, int32_t* row_output) {
  const uint32_t i = hist4096::kBlockSize + threadIdx.x;
  if (i >= num_ties) {
    return;
  }

  const auto tie = tie_buffer[i];
  const uint32_t output_pos = total_above + prefix_equal + i;
  if (output_pos < TopK) {
    row_output[output_pos] = tie.idx + row_start;
  }
  const uint32_t tie_pos = prefix_equal + i;
  if (tie_pos < kCoarseTieCapacity) {
    tie_ws[tie_pos] = {tie.idx + row_start, tie.score};
  }
}

// Cluster-cooperative large path.
// kFused=true: all TMA stages resident, single-pass histogram + scatter (rescan
// from smem). kFused=false: TMA double-buffer streaming, two passes (histogram
// then scatter).
template <uint32_t TopK, uint32_t CS, typename SmemType, bool kFused>
__device__ void large_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t* phases, hist4096::Tie* tie_ws) {
  const auto rank = blockIdx.y;  // this block's position in cluster
  const auto tx = threadIdx.x;
  const auto lane = tx % hist4096::kWarpSize;

  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<SmemType*>(smem_raw);
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw + sizeof(SmemType));

  // Partition row across cluster ranks
  constexpr uint32_t kAlign = 4;
  const auto units =
      (seq_len + kAlign - 1) / kAlign;  // float4-aligned element count
  const auto base = units / CS, extra = units % CS;  // elements per block
  const auto lu = base + (rank < extra ? 1u : 0u);   // remainder blocks
  const auto ou =
      rank * base + min(rank, extra);  // this block's count (load-balanced)
  const auto my_start = ou * kAlign;   // global start offset
  const auto my_len = min(my_start + lu * kAlign, seq_len) -
                      my_start;  // actual length of this block
  const auto num_iters =
      (my_len + kSizePerStage - 1) / kSizePerStage;  // TMA stages needed
  const auto len_aligned = (my_len + 3u) & ~3u;

  if constexpr (kFused) {
    // Fused init + TMA prologue
    if (tx < kHistBins) smem->histogram[tx] = 0;
    if (tx == 0) {  // thread 0 issues TMA - then all threads continue working
                    // until mbarrier sync
      smem->counter_gt = 0;
      smem->counter_eq = 0;
      for (uint32_t i = 0; i < num_iters; i++) {
        const auto off = i * kSizePerStage;
        const auto sz = min(kSizePerStage, len_aligned - off) * sizeof(float);
        tma_load(smem->score_buffer[i], row_input + my_start + off, sz,
                 &smem->barrier[0][i]);  // cp.async.bulk of size kSizePerStage
                                         // × sizeof(float)
        mbarrier_arrive_expect_tx(&smem->barrier[0][i], sz);
      }
    }
    __syncthreads();

    // Histogram build. ILP unroll-by-2, no inter-stage sync
    for (uint32_t iter = 0; iter < num_iters; iter++) {
      const auto off = iter * kSizePerStage;
      const auto sz = min(kSizePerStage, my_len - off);
      if (lane == 0) {
        mbarrier_wait(&smem->barrier[0][iter],
                      phases[iter] & 1);  // wait for TMA
      }
      phases[iter]++;
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i += 2) {
        const auto li0 = tx + i * hist4096::kBlockSize;
        const auto li1 = tx + (i + 1) * hist4096::kBlockSize;
        if (li0 >= sz) {
          break;
        }
        const auto b0 = extract_coarse_bin(smem->score_buffer[iter][li0]);
        if (li1 < sz) {
          const auto b1 = extract_coarse_bin(smem->score_buffer[iter][li1]);
          atomicAdd(&smem->histogram[b0], 1);
          atomicAdd(&smem->histogram[b1], 1);
        } else {
          atomicAdd(&smem->histogram[b0], 1);
        }
      }
    }
  } else {
    // Twopass: init then stream histogram pass
    if (tx < kHistBins) smem->histogram[tx] = 0;
    if (tx == 0) {
      smem->counter_gt = 0;
      smem->counter_eq = 0;
    }
    __syncthreads();
    tma_stream_pass<SmemType, kStreamingStagesCS4, kHistBits, false>(
        row_input + my_start, my_len, 0, nullptr, phases, smem);
  }

  // DSMEM all-reduce + find threshold
  dsmem_hist_reduce<CS, kHistBins>(
      smem->histogram);  // each block histogram is summed across all CS blocks
  find_threshold<TopK>(smem->histogram, smem->warp_sum, &smem->counter_gt,
                       &smem->counter_eq, &smem->match);

  const auto thr = smem->match.bin;
  // A larger threshold bin needs the exact overflow path.
  if (__builtin_expect(
          smem->match.equal_count > kCoarseTieCapacity, 0)) {
    recover_coarse_overflow<TopK, CS, kFused>(
        row_input, row_output, my_start, my_len, smem, s_topk);
    return;
  }

  if constexpr (kFused) {
    // Fused scatter: rescan score_buffer (still in smem)
    for (uint32_t iter = 0; iter < num_iters; iter++) {
      const auto off = iter * kSizePerStage;
      const auto sz = min(kSizePerStage, my_len - off);
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i++) {
        const auto li = tx + i * hist4096::kBlockSize;
        if (li >= sz) {
          break;
        }
        const auto score = smem->score_buffer[iter][li];  // still in smem
        const auto bin = extract_coarse_bin(score);
        const auto gidx = off + li;
        if (bin > thr) {
          s_topk[atomicAdd(&smem->counter_gt, 1)] = gidx;  // above -> s_topk
        } else if (bin == thr) {
          const auto p = atomicAdd(&smem->counter_eq,
                                   1);  // equal -> ties (later refinement)
          if (p < kCoarseTieCapacity) {
            smem->tie_buffer[p] = {gidx, score};
          }
        }
      }
    }
    __syncthreads();
  } else {
    // Twopass scatter: re-stream data via TMA
    uint32_t scatter_phases[kStreamingStagesCS4] = {};
    tma_stream_pass<SmemType, kStreamingStagesCS4, kHistBits, true>(
        row_input + my_start, my_len, thr, s_topk, scatter_phases, smem);
  }

  // Output collection via DSMEM prefix sum
  constexpr uint32_t kAboveBits = 16;
  constexpr uint32_t kAboveMask = (1 << kAboveBits) - 1;
  static_assert(kAboveMask >= TopK);
  static_assert(kAboveMask >= kMaxSinglePassPerBlock,
                "kAboveBits must cover max per-block element count");

  const uint32_t la = smem->counter_gt;
  const uint32_t le_full = smem->counter_eq;
  const uint32_t le =
      min(le_full, kCoarseTieCapacity);  // written smem tie_buffer entries

  __shared__ uint32_t s_local_counts[CS];
  __shared__ uint32_t s_prefix_packed;
  __shared__ uint32_t s_total_above, s_total_equal;

  auto cluster = cooperative_groups::this_cluster();
  if (tx < CS) {
    // Pack written tie counts into 32-bit: (equal << 16) | above.
    // `le_full` may exceed the per-block tie buffer cap; using it here creates
    // holes in tie_ws and can make TopK=2048 refine unwritten workspace slots.
    const uint32_t packed = (le << kAboveBits) | la;
    const auto dst = cluster.map_shared_rank(s_local_counts, tx);
    dst[rank] = packed;  // write my count to every block's s_local_counts[rank]
  }
  cluster.sync();

  // Thread 0 computes serial prefix sum
  if (tx == 0) {
    uint32_t prefix = 0, ta = 0, te = 0;
    for (uint32_t i = 0; i < CS; i++) {
      if (i == rank) {
        s_prefix_packed = prefix;  // my prefix
      }
      ta += s_local_counts[i] & kAboveMask;   // total above
      te += s_local_counts[i] >> kAboveBits;  // total equal
      prefix += s_local_counts[i];
    }
    s_total_above = ta;
    s_total_equal = te;
  }
  __syncthreads();

  const uint32_t prefix_above = s_prefix_packed & kAboveMask;
  const uint32_t prefix_equal = s_prefix_packed >> kAboveBits;

  // Write to global output
  for (uint32_t i = tx; i < la; i += hist4096::kBlockSize) {
    // indices are placed contiguously starting at prefix_above
    row_output[prefix_above + i] =
        s_topk[i] + my_start;  // my_start: block-local -> row-global index
  }
  const uint32_t common_le = min(le, hist4096::kBlockSize);
  if (tx < common_le) {
    const uint32_t i = tx;
    const auto t = smem->tie_buffer[i];
    const uint32_t p = s_total_above + prefix_equal + i;
    if (p < TopK) {
      row_output[p] = t.idx + my_start;
    }
    const uint32_t tp = prefix_equal + i;
    if (tp < kCoarseTieCapacity) {
      tie_ws[tp] = hist4096::Tie{t.idx + my_start, t.score};
    }
  }
  if (__builtin_expect(le > hist4096::kBlockSize, 0)) {
    copy_extra_coarse_ties<TopK>(smem->tie_buffer, le, prefix_equal,
                                 s_total_above, my_start, tie_ws, row_output);
  }

  // Tie refinement
  cooperative_groups::this_cluster().sync();
  if (rank != 0) {  // only rank 0 does tie refinement
    return;
  }
  if (s_total_above + s_total_equal <= TopK) {  // no ties to refine
    return;
  }

  // Tie-breaking uses FP32 (4-round radix sort)
  if constexpr (TopK <= hist4096::kBlockSize) {
    if (s_total_equal <= hist4096::kMaxTies) {
      // Common case: copy one tie per thread back to smem, then refine.
      const uint32_t num_ties = s_total_equal;
      for (uint32_t i = tx; i < num_ties; i += hist4096::kBlockSize) {
        smem->tie_buffer[i] = hist4096::Tie{tie_ws[i].idx, tie_ws[i].score};
      }
      __syncthreads();
      hist4096::tie_handle<TopK>(smem->tie_buffer, num_ties, s_total_above,
                                 row_output, smem);
    } else {
      // Rare case: retain and rank two threshold-bin candidates per thread.
      refine_large_coarse_ties<TopK>(tie_ws, s_total_equal, s_total_above,
                                     row_output, smem);
    }
  } else {
    hist4096::tie_handle_large<TopK>(tie_ws, s_total_equal, s_total_above,
                                     row_output, smem);
  }
}

// ============================================================================
// Adapted from https://github.com/sgl-project/sglang/pull/23600
// sgl-project/sglang
// (python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/)
// ============================================================================

template <uint32_t TopK, uint32_t CS>
__device__ void cooperative_topk_body(CooperativeTopKParams<TopK> params) {
  const auto rank = blockIdx.y, row = blockIdx.x, tx = threadIdx.x;
  // Clamp at 0: `sl` is compared signed here but cast to uint32 below, so a
  // negative length would otherwise emit indices 0..TopK-1 as valid instead
  // of the -1 padding.
  const int32_t sl = params.lengths[row] > 0 ? params.lengths[row] : 0;
  int32_t* out = params.output + row * TopK;
  const float* in = params.input + row * params.stride;

  // Trivial: seq_len <= TopK
  if (sl <= static_cast<int32_t>(TopK)) {
    if (rank == 0) {
      for (uint32_t i = tx; i < TopK; i += hist4096::kBlockSize) {
        out[i] = (i < static_cast<uint32_t>(sl)) ? static_cast<int32_t>(i) : -1;
      }
    }
    return;
  }

  // Short-Medium path: histogram_4096_topk on rank 0 only - all data fits in RF
  if (sl <= static_cast<int32_t>(hist4096::kHist4096MaxLen)) {
    if (rank == 0) {
      extern __shared__ uint8_t sr[];
      hist4096::histogram_4096_topk<
          TopK, 12, hist4096::kHist4096VecsPerThread, false,
          hist4096::OverflowRecovery::kRescan>(
          in, out, sl, sr);  // 4096-bin (12-bit) histogram
    }
    return;
  }

  // Large path: init mbarriers + state, then dispatch fused or twopass
  const uint32_t per_block =
      (params.stride + CS - 1) / CS;  // how many elements per block
  constexpr uint32_t kFusedMax = ((CS == 16)  ? kFusedStagesCS16
                                  : (CS == 8) ? kFusedStagesCS8
                                              : kMaxSinglePassStages) *
                                 kSizePerStage;
  const bool use_singlepass =
      per_block <=
      kFusedMax;  // single pass or TMA streaming: histogram+scatter

  // Select smem type and stage count at compile time based on CS
  constexpr uint32_t kFusedStages = (CS == 16)  ? kFusedStagesCS16
                                    : (CS == 8) ? kFusedStagesCS8
                                                : kMaxSinglePassStages;
  using FusedSmem = SmemFused<kFusedStages>;

  extern __shared__ uint8_t sr[];

  constexpr uint32_t kTieWsPerRow = kCoarseTieCapacity;
  hist4096::Tie* row_tie_ws = params.tie_ws + row * kTieWsPerRow;

  if (use_singlepass) {
    auto* smem = reinterpret_cast<FusedSmem*>(sr);
    const uint32_t sp_stages = (per_block + kSizePerStage - 1) / kSizePerStage;
    if (tx < sp_stages) {
      mbarrier_init(&smem->barrier[0][tx],
                    1);  // init 1 barrier per TMA stage -
                         // signal when async copies complete
    }
    __syncthreads();
    uint32_t phases[kFusedStages] =
        {};  // tracks the parity for mbarrier wait/arrive protocol
    large_topk<TopK, CS, FusedSmem, true>(in, out, sl, phases, row_tie_ws);
  } else {
    auto* smem = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kStreamingStagesCS4) {
      mbarrier_init(&smem->barrier[0][tx], 1);
    }
    __syncthreads();
    uint32_t hp[kStreamingStagesCS4] = {};
    large_topk<TopK, CS, Smem4, false>(in, out, sl, hp, row_tie_ws);
  }
}

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, 4, 1)
        cooperative_topk_cs4(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 4>(params);
}

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, 8, 1)
        cooperative_topk_cs8(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 8>(params);
}

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, 16, 1)
        cooperative_topk_cs16(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 16>(params);
}

constexpr size_t kSmemSize4_base = sizeof(Smem4);
constexpr size_t kSmemSize4_sp = sizeof(SmemSinglePass);
constexpr size_t kSmemSize4 =
    (kSmemSize4_base > kSmemSize4_sp ? kSmemSize4_base : kSmemSize4_sp) +
    sizeof(int32_t) * 2048 + 128;
constexpr size_t kSmemSize8 =
    sizeof(SmemFused<kFusedStagesCS8>) + sizeof(int32_t) * 2048 + 128;

}  // namespace cooperative

}  // namespace vllm

#endif  // COOPERATIVE_TOPK_CUH_
