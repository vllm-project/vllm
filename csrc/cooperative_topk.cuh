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
#include "topk_histogram_4096_overflow.cuh"

namespace vllm {
namespace cooperative {

namespace hist4096 = topk_histogram_4096;

constexpr uint32_t kHistBits = 10;
constexpr uint32_t kHistBins = 1 << kHistBits;
using ExactRadix = hist4096::ExactRadixTraits;
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
constexpr uint32_t kFusedMaxStrideCS16 = 16 * kFusedStagesCS16 * kSizePerStage;

// CS=4 single-pass path
constexpr uint32_t kMaxSinglePassStages = 3;
constexpr uint32_t kMaxSinglePassPerBlock =
    kMaxSinglePassStages * kSizePerStage;  // 49152

template <uint32_t TopK = 1024>
struct CooperativeTopKParams {
  const float* __restrict__ input;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ lengths;
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
  static_assert(CS <= 32);
  auto cluster = cooperative_groups::this_cluster();
  const auto tx = threadIdx.x;
  const auto rank = blockIdx.y;
  if constexpr ((CS & (CS - 1)) != 0) {
    constexpr uint32_t kOwn = (NumBins + CS - 1) / CS;
    __shared__ uint32_t s_red[kOwn];
    const uint32_t lo = min(rank * kOwn, NumBins);
    const uint32_t n = min(kOwn, NumBins - lo);
    for (uint32_t t = tx; t < kOwn; t += hist4096::kBlockSize) {
      s_red[t] = 0;
    }
    cluster.sync();
    for (uint32_t t = tx; t < n * CS; t += hist4096::kBlockSize) {
      const uint32_t b = t / CS, src = t % CS;
      atomicAdd(&s_red[b], *cluster.map_shared_rank(&histogram[lo + b], src));
    }
    __syncthreads();
    for (uint32_t t = tx; t < n * CS; t += hist4096::kBlockSize) {
      const uint32_t b = t % n, dst = t / n;
      *cluster.map_shared_rank(&histogram[lo + b], dst) = s_red[b];
    }
    cluster.sync();
  } else {
    static_assert(NumBins % CS == 0);
    // Fold the distributed per-rank bins into cluster-visible totals.
    cluster.sync();
    constexpr auto kLocal = NumBins / CS;
    const auto off = kLocal * rank;
#pragma unroll
    for (uint32_t bin = tx; bin < NumBins; bin += hist4096::kBlockSize) {
      // Cover histograms wider than one CTA without changing the reduction.
      const auto addr = &histogram[off + bin / CS];
      const auto src = cluster.map_shared_rank(addr, bin % CS);
      *src = warp_reduce_sum_subN<CS>(*src);
    }
    cluster.sync();
  }
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

using Smem8 = SmemFused<kFusedStagesCS8>;
using Smem16 = SmemFused<kFusedStagesCS16>;
using Smem4 = SmemFused<kStreamingStagesCS4, 2>;
using SmemSinglePass = SmemFused<kMaxSinglePassStages>;

#include "cooperative_topk_overflow.cuh"

template <uint32_t TopK, uint32_t CS, typename SmemType>
__device__ void cluster_tie_refine(int32_t* __restrict__ row_output,
                                   const uint32_t* local_counts,
                                   uint32_t num_above, uint32_t seq_len,
                                   SmemType* smem) {
  constexpr uint32_t kCap = kCoarseTieCapacity;
  constexpr uint32_t kPerThread =
      (kCap + hist4096::kBlockSize - 1) / hist4096::kBlockSize;
  constexpr uint32_t kAboveBits = 16;
  constexpr uint32_t kAlign = 4;
  __shared__ uint32_t s_tie_prefix[CS + 1];
  const auto tx = threadIdx.x;
  const auto li = tx % hist4096::kWarpSize, wi = tx / hist4096::kWarpSize;
  auto cluster = cooperative_groups::this_cluster();

  if (tx == 0) {
    uint32_t acc = 0;
    for (uint32_t r = 0; r < CS; r++) {
      s_tie_prefix[r] = acc;
      acc += local_counts[r] >> kAboveBits;
    }
    s_tie_prefix[CS] = acc;
  }
  __syncthreads();
  const uint32_t num_ties = min(s_tie_prefix[CS], kCap);
  const uint32_t units = (seq_len + kAlign - 1) / kAlign;
  const uint32_t base = units / CS, extra = units % CS;

  uint32_t idxs[kPerThread], keys[kPerThread];
  bool active[kPerThread];
  uint32_t key_or = 0, key_and = ~0u;
#pragma unroll
  for (uint32_t e = 0; e < kPerThread; e++) {
    const uint32_t g = e * hist4096::kBlockSize + tx;
    active[e] = g < num_ties;
    idxs[e] = 0;
    keys[e] = 0;
    if (active[e]) {
      uint32_t r = 0;
      while (r + 1 < CS && s_tie_prefix[r + 1] <= g) {
        r++;
      }
      const hist4096::Tie t = *cluster.map_shared_rank(
          &smem->tie_buffer[g - s_tie_prefix[r]], r);
      idxs[e] = t.idx + (r * base + min(r, extra)) * kAlign;
      keys[e] = hist4096::convert_to_uint32_v2(t.score);
      key_or |= keys[e];
      key_and &= keys[e];
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  key_or = __reduce_or_sync(0xFFFFFFFFu, key_or);
  key_and = __reduce_and_sync(0xFFFFFFFFu, key_and);
  __syncthreads();
  uint32_t* s_or = smem->histogram;
  uint32_t* s_and = smem->histogram + hist4096::kNumWarps;
  if (li == 0) {
    s_or[wi] = key_or;
    s_and[wi] = key_and;
  }
  __syncthreads();
  const uint32_t vary = __reduce_or_sync(0xFFFFFFFFu, s_or[li]) ^
                        __reduce_and_sync(0xFFFFFFFFu, s_and[li]);
  if (tx == 0) {
    smem->counter_gt = 0;
  }
  __syncthreads();

  uint32_t* hist = smem->histogram;
  uint32_t remain = TopK - num_above;
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const uint32_t sh = 24 - r * 8;
    if (r < 3 && ((vary >> sh) & 0xFFu) == 0) {
      continue;
    }
    if (tx < hist4096::RADIX) {
      hist[tx] = 0;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t e = 0; e < kPerThread; e++) {
      if (active[e]) {
        atomicAdd(&hist[(keys[e] >> sh) & 0xFFu], 1);
      }
    }
    __syncthreads();
    uint32_t hv = 0, wi2 = 0;
    if (tx < hist4096::RADIX) {
      hv = hist[tx];
      wi2 = hist4096::warp_inclusive_sum(li, hv);
      if (li == hist4096::kWarpSize - 1) {
        smem->warp_sum[wi] = wi2;
      }
    }
    __syncthreads();
    if (tx < hist4096::RADIX) {
      const auto tmp = (li < hist4096::RADIX / hist4096::kWarpSize)
                           ? smem->warp_sum[li]
                           : 0u;
      const auto tot = hist4096::warp_reduce_sum_full(tmp);
      const auto inter = hist4096::warp_reduce_sum_full(li < wi ? tmp : 0u);
      const auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        smem->match = {.bin = tx,
                       .above_count = above,
                       .equal_count = remain - above};
      }
    }
    __syncthreads();
    const auto thr = smem->match.bin;
    const auto na = smem->match.above_count;
#pragma unroll
    for (uint32_t e = 0; e < kPerThread; e++) {
      if (!active[e]) {
        continue;
      }
      const uint32_t bin = (keys[e] >> sh) & 0xFFu;
      if (bin > thr) {
        const uint32_t wpos = num_above + atomicAdd(&smem->counter_gt, 1);
        if (wpos < TopK) {
          row_output[wpos] = idxs[e];
        }
        active[e] = false;
      } else if (bin < thr) {
        active[e] = false;
      } else if (r == 3) {
        const uint32_t wpos = TopK - atomicAdd(&smem->match.equal_count, -1u);
        if (wpos < TopK) {
          row_output[wpos] = idxs[e];
        }
      }
    }
    remain -= na;
  }
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Cluster-cooperative large path.
// kFused=true: all TMA stages resident, single-pass histogram + scatter (rescan
// from smem). kFused=false: TMA double-buffer streaming, two passes (histogram
// then scatter).
template <uint32_t TopK, uint32_t CS, typename SmemType, bool kFused>
__device__ void large_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t* phases) {
  const auto rank = blockIdx.y;  // this block's position in cluster
  const auto tx = threadIdx.x;
  const auto lane = tx % hist4096::kWarpSize;

  extern __shared__ __align__(128) uint8_t smem_raw[];
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
    if (tx < kHistBins) {
      smem->histogram[tx] = 0;  // all threads zero histogram
    }
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
    if (tx < kHistBins) {
      smem->histogram[tx] = 0;
    }
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
  // A threshold bin beyond tie capacity needs exact recovery; scattering
  // into the bounded buffer would discard possible TopK candidates.
  if (__builtin_expect(smem->match.equal_count > kCoarseTieCapacity, 0)) {
    recover_coarse_overflow<TopK, CS, kFused>(row_input, row_output, seq_len,
                                              my_start, my_len, smem, s_topk);
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
    uint32_t scatter_phases[kStreamingStagesCS4] = {0, 0};
    tma_stream_pass<SmemType, kStreamingStagesCS4, kHistBits, true>(
        row_input + my_start, my_len, thr, s_topk, scatter_phases, smem);
  }

  // Output collection via DSMEM prefix sum
  constexpr uint32_t kAboveBits = 16;
  constexpr uint32_t kAboveMask = (1 << kAboveBits) - 1;
  static_assert(kAboveMask >= TopK);
  static_assert(CS * kCoarseTieCapacity <= kAboveMask + 1u);
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

  const bool refine_ties = s_total_above + s_total_equal > TopK;
  for (uint32_t i = tx; i < la; i += hist4096::kBlockSize) {
    row_output[prefix_above + i] = s_topk[i] + my_start;
  }
  if (!refine_ties) {
    for (uint32_t i = tx; i < le; i += hist4096::kBlockSize) {
      const uint32_t p = s_total_above + prefix_equal + i;
      if (p < TopK) {
        row_output[p] = smem->tie_buffer[i].idx + my_start;
      }
    }
    return;
  }
  if (rank != 0) {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  cluster_tie_refine<TopK, CS>(row_output, s_local_counts, s_total_above,
                               seq_len, smem);
}

// ============================================================================
// Adapted from https://github.com/sgl-project/sglang/pull/23600
// sgl-project/sglang
// (python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/)
// ============================================================================

template <uint32_t TopK, uint32_t CS, uint32_t kStages = 0>
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
      extern __shared__ __align__(128) uint8_t sr[];
      hist4096::histogram_4096_topk<TopK, 12, hist4096::kHist4096VecsPerThread,
                                    false,
                                    hist4096::OverflowRecovery::kFp32Rescan>(
          in, out, sl, sr);  // 4096-bin (12-bit) histogram
    }
    return;
  }

  // Large path: init mbarriers + state, then dispatch fused or twopass.
  // Elements per block, sized by the row's length rather than its stride so a
  // short row in a wide logits buffer still takes the single-pass staging.
  // large_topk splits the row in 16-byte units, but kFusedMax and
  // kSizePerStage are multiples of that unit, so the path and stage count
  // come out the same.
  const uint32_t per_block = (static_cast<uint32_t>(sl) + CS - 1) / CS;
  constexpr uint32_t kFusedMax = (kStages ? kStages
                                  : (CS == 16)    ? kFusedStagesCS16
                                  : (CS == 8)     ? kFusedStagesCS8
                                                  : kMaxSinglePassStages) *
                                 kSizePerStage;
  const bool use_singlepass =
      per_block <=
      kFusedMax;  // single pass or TMA streaming: histogram+scatter

  // Select smem type and stage count at compile time based on CS
  constexpr uint32_t kFusedStages = kStages ? kStages
                                    : (CS == 16)    ? kFusedStagesCS16
                                    : (CS == 8)     ? kFusedStagesCS8
                                                    : kMaxSinglePassStages;
  using FusedSmem = SmemFused<kFusedStages>;

  extern __shared__ __align__(128) uint8_t sr[];

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
    large_topk<TopK, CS, FusedSmem, true>(in, out, sl, phases);
  } else {
    // Streaming path for cluster sizes whose per-block input exceeds the
    // single-pass staging capacity.
    auto* smem = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kStreamingStagesCS4) {
      mbarrier_init(&smem->barrier[0][tx],
                    1);  // init 2×2=4 barriers (2 passes × 2 stages)
    }
    __syncthreads();
    uint32_t hp[kStreamingStagesCS4] = {0,
                                        0};  // histogram+scatter pass counters
    large_topk<TopK, CS, Smem4, false>(in, out, sl, hp);
  }
}

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    __cluster_dims__(1, 2, 1)
        cooperative_topk_cs2(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 2>(params);
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

template <uint32_t TopK, uint32_t CS, uint32_t Stages>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1)
    cooperative_topk_staged(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, CS, Stages>(params);
}

constexpr size_t kSmemSize4_base = sizeof(Smem4);
constexpr size_t kSmemSize4_sp = sizeof(SmemSinglePass);
constexpr size_t kSmemSize4 =
    (kSmemSize4_base > kSmemSize4_sp ? kSmemSize4_base : kSmemSize4_sp) +
    sizeof(int32_t) * 2048 + 128;
constexpr size_t kSmemSize2 = kSmemSize4;
constexpr size_t kSmemSizeStaged3 = kSmemSize4;
constexpr size_t kSmemSizeStaged4 =
    sizeof(SmemFused<4>) + sizeof(int32_t) * 2048 + 128;
constexpr size_t kSmemSize8 =
    sizeof(SmemFused<kFusedStagesCS8>) + sizeof(int32_t) * 2048 + 128;

}  // namespace cooperative

}  // namespace vllm

#endif  // COOPERATIVE_TOPK_CUH_
