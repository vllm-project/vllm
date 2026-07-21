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
constexpr uint32_t kMaxTopK = 2048;

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

template <uint32_t CS>
__device__ __forceinline__ void dsmem_hist_reduce(uint32_t* histogram) {
  static_assert(kHistBins <= hist4096::kBlockSize);
  auto cluster = cooperative_groups::this_cluster();
  cluster.sync();
  const auto tx = threadIdx.x;
  const auto rank = blockIdx.y;
  constexpr auto kLocal = kHistBins / CS;
  const auto off = kLocal * rank;
  if (tx < kHistBins) {
    const auto addr = &histogram[off + tx / CS];
    const auto src = cluster.map_shared_rank(addr, tx % CS);
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
          if (p < hist4096::kMaxTies) {
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
    uint32_t histogram[kHistBins];
    hist4096::Tie tie_buffer[kMaxTopK];
  };
  alignas(128) float score_buffer[kStages][kSizePerStage];
};

using Smem8 = SmemFused<kFusedStagesCS8>;
using Smem16 = SmemFused<kFusedStagesCS16>;
using Smem4 = SmemFused<kStreamingStagesCS4, 2>;
using SmemSinglePass = SmemFused<kMaxSinglePassStages>;

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
  dsmem_hist_reduce<CS>(
      smem->histogram);  // each block histogram is summed across all CS blocks
  find_threshold<TopK>(smem->histogram, smem->warp_sum, &smem->counter_gt,
                       &smem->counter_eq, &smem->match);

  const auto thr = smem->match.bin;

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
          if (p < hist4096::kMaxTies) {
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
  static_assert(kAboveMask >= kMaxSinglePassPerBlock,
                "kAboveBits must cover max per-block element count");

  const uint32_t la = smem->counter_gt;
  const uint32_t le_full = smem->counter_eq;
  const uint32_t le =
      min(le_full, hist4096::kMaxTies);  // written smem tie_buffer entries

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
  for (uint32_t i = tx; i < le; i += hist4096::kBlockSize) {
    const auto t = smem->tie_buffer[i];
    uint32_t p = s_total_above + prefix_equal + i;
    if (p < TopK) {
      row_output[p] = t.idx + my_start;
    }
    uint32_t tp = prefix_equal + i;
    if (tp < (TopK <= hist4096::kBlockSize ? hist4096::kMaxTies : TopK)) {
      tie_ws[tp] = hist4096::Tie{t.idx + my_start, t.score};
    }
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
    // copy ties from tie_ws back to smem, then refine
    const uint32_t num_ties = min(s_total_equal, hist4096::kMaxTies);
    // TODO (roberto): could vectorize with uint2 (8 bytes = exactly one Tie)
    for (uint32_t i = tx; i < num_ties; i += hist4096::kBlockSize) {
      smem->tie_buffer[i] = hist4096::Tie{tie_ws[i].idx, tie_ws[i].score};
    }
    __syncthreads();
    hist4096::tie_handle<TopK>(smem->tie_buffer, num_ties, s_total_above,
                               row_output, smem);
  } else {
    // TopK=2048: process directly from tie_ws (GMEM)
    const uint32_t num_ties = min(s_total_equal, static_cast<uint32_t>(TopK));
    hist4096::tie_handle_large<TopK>(tie_ws, num_ties, s_total_above,
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
  const auto sl = params.lengths[row];
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
      hist4096::histogram_4096_topk<TopK, 12>(
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

  constexpr uint32_t kTieWsPerRow =
      TopK <= hist4096::kBlockSize ? hist4096::kMaxTies : TopK;
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
    // Two-pass: only CS=4 in practice (CS=8 always fits in singlepass)
    auto* smem = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kStreamingStagesCS4) {
      mbarrier_init(&smem->barrier[0][tx],
                    1);  // init 2×2=4 barriers (2 passes × 2 stages)
    }
    __syncthreads();
    uint32_t hp[kStreamingStagesCS4] = {0,
                                        0};  // histogram+scatter pass counters
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
