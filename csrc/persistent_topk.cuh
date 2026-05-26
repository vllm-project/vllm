/*
 * Persistent TopK Scheduler for DSA Indexer
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

namespace vllm {
namespace cooperative {

// TopK is now a template parameter (512 or 1024)
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kHistBits = 10;
constexpr uint32_t kHistBins = 1 << kHistBits;
constexpr uint32_t RADIX = 256;
constexpr uint32_t kMaxTies = 1024;
static_assert(kMaxTies <= kBlockSize, "tie_handle requires kMaxTies <= kBlockSize");
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;

constexpr uint32_t kElemPerStage = 16;
constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;  // 16384

// CS=4: 2 TMA stages (double buffer), two-pass
constexpr uint32_t kNumStages4 = 2;
// CS=8: 4 TMA stages, single-pass (all data stays in smem)
constexpr uint32_t kNumStages8 = 2;  // 2 stages × 16K = 32K per block (enough for CS=8)
// Max data per block with CS=8: ceil(262144/8) = 32768 ≤ 4 × 8192 = 32768
constexpr uint32_t kMaxSeqLen8 = kNumStages8 * kSizePerStage * 8;  // 262144

// Register path
constexpr uint32_t kRegHistBits = 12;
constexpr uint32_t kRegHistBins = 1 << kRegHistBits;  // 4096
constexpr uint32_t kRegVecsPerThread = 4;
constexpr uint32_t kRegMaxLen = kRegVecsPerThread * 4 * kBlockSize;  // 16384
constexpr uint32_t kRegHistItems = kRegHistBins / kBlockSize;  // 4

// CS=4 single-pass path
constexpr uint32_t kMaxSinglePassStages = 3;
constexpr uint32_t kMaxSinglePassPerBlock = kMaxSinglePassStages * kSizePerStage;  // 49152

constexpr uint32_t kStreamNumStages = 2;

struct alignas(16) MatchBin { uint32_t bin, above_count, equal_count; };
struct alignas(8) Tie { uint32_t idx; float score; };
struct ClusterState { int output_counter; };

template <uint32_t TopK = 1024>
struct CooperativeTopKParams {
  const float* __restrict__ input;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ lengths;
  ClusterState* __restrict__ states;
  Tie* __restrict__ tie_ws;  // per-cluster tie workspace, kMaxTies entries each
  uint32_t num_rows, stride;
};

// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane, uint32_t v) {
#pragma unroll
  for (uint32_t o = 1; o < 32; o *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, v, o); if (lane >= o) v += n;
  }
  return v;
}

__device__ __forceinline__ uint32_t warp_reduce_sum_full(uint32_t v) {
  uint32_t r;
  asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
  return r;
}

template <uint32_t N>
__device__ __forceinline__ uint32_t warp_reduce_sum_subN(uint32_t v) {
#pragma unroll
  for (uint32_t m = N >> 1; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, m, 32);
  return v;
}

// ============================================================================
// Cluster cooperative TopK
// ============================================================================

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ uint32_t extract_coarse_bin(float x) {
  return extract_coarse_bin_N<kHistBits>(x);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* a, uint32_t n) { cuda::ptx::mbarrier_init(a, n); }
__device__ __forceinline__ void mbarrier_wait(uint64_t* a, uint32_t p) {
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, a, p));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* a, uint32_t t) {
  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, a, t);
}
__device__ __forceinline__ void tma_load(void* d, const void* s, uint32_t n, uint64_t* m) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global, d, s, n, m);
}

// ============================================================================
// Tie refinement (single CTA): 4-round radix topK
// ============================================================================

template <uint32_t TopK>
__device__ void tie_handle(const Tie* ties, uint32_t num_ties, uint32_t num_above,
                            int32_t* output, void* _smem) {
  struct TS { alignas(128) uint32_t counter; alignas(128) MatchBin match;
              uint32_t histogram[RADIX]; uint32_t warp_sum[kNumWarps]; };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;
  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = convert_to_uint32_v2(tie.score);
  bool active = has; uint32_t remain = TopK - num_above, wpos = TopK;
  s->counter = 0; __syncthreads();
#pragma unroll
  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8, bin = (key >> sh) & 0xFF;
    if (tx < RADIX) s->histogram[tx] = 0; __syncthreads();
    if (active) atomicAdd(&s->histogram[bin], 1); __syncthreads();
    uint32_t hv = 0, wi2 = 0;
    if (tx < RADIX) { hv = s->histogram[tx]; wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize-1) s->warp_sum[wi] = wi2; }
    __syncthreads();
    if (tx < RADIX) {
      auto tmp = (li < RADIX/kWarpSize) ? s->warp_sum[li] : 0;
      auto tot = warp_reduce_sum_full(tmp);
      auto inter = warp_reduce_sum_full(li < wi ? tmp : 0);
      auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) s->match = {tx, above, remain - above};
    } __syncthreads();
    auto [thr, na, _] = s->match;
    if (active) {
      if (bin > thr) { wpos = num_above + atomicAdd(&s->counter, 1); active = false; }
      else if (bin < thr) active = false;
      else if (r == 3) wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
    }
    remain -= na; if (!remain) break;
  }
  if (wpos < TopK) output[wpos] = tie.idx;
}

// Extended tie_handle for TopK > kBlockSize (e.g. TopK=2048).
// Each thread handles ceil(TopK/kBlockSize) ties from global memory.
template <uint32_t TopK>
__device__ void tie_handle_large(const Tie* ties, uint32_t num_ties,
                                  uint32_t num_above, int32_t* output, void* _smem) {
  static_assert(TopK > kBlockSize);
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize;
  const auto wi = tx / kWarpSize;

  constexpr uint32_t kPerThread = (TopK + kBlockSize - 1) / kBlockSize;
  Tie my_ties[kPerThread];
  uint32_t keys[kPerThread];
  bool active[kPerThread];

  for (uint32_t e = 0; e < kPerThread; e++) {
    uint32_t idx = e * kBlockSize + tx;
    if (idx < num_ties) {
      my_ties[e] = ties[idx];
      keys[e] = convert_to_uint32_v2(ties[idx].score);
      active[e] = true;
    } else {
      my_ties[e] = {0, 0.0f};
      keys[e] = 0;
      active[e] = false;
    }
  }

  uint32_t remain = TopK - num_above;
  s->counter = 0;
  __syncthreads();

  for (int r = 0; r < 4; r++) {
    uint32_t sh = 24 - r * 8;
    if (tx < RADIX) {
      s->histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (active[e]) {
        atomicAdd(&s->histogram[(keys[e] >> sh) & 0xFF], 1);
      }
    }
    __syncthreads();

    uint32_t hv = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      auto wi2 = warp_inclusive_sum(li, hv);
      if (li == kWarpSize - 1) {
        s->warp_sum[wi] = wi2;
      }
    }
    __syncthreads();
    if (tx < RADIX) {
      auto tmp2 = (li < RADIX / kWarpSize) ? s->warp_sum[li] : 0;
      auto total = warp_reduce_sum_full(tmp2);
      auto inter = warp_reduce_sum_full(li < wi ? tmp2 : 0);
      auto wi2 = warp_inclusive_sum(li, hv);
      auto above = total - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {.bin = tx, .above_count = above, .equal_count = remain - above};
      }
    }
    __syncthreads();

    auto thr = s->match.bin;
    auto na = s->match.above_count;

    for (uint32_t e = 0; e < kPerThread; e++) {
      if (!active[e]) {
        continue;
      }
      uint32_t bin = (keys[e] >> sh) & 0xFF;
      if (bin > thr) {
        uint32_t wpos = num_above + atomicAdd(&s->counter, 1);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
        active[e] = false;
      } else if (bin < thr) {
        active[e] = false;
      } else if (r == 3) {
        uint32_t wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
        if (wpos < TopK) {
          output[wpos] = my_ties[e].idx;
        }
      }
    }

    num_above += na;
    remain -= na;
    __syncthreads();
    s->counter = 0;
    __syncthreads();
  }
}


// ============================================================================
// DSMEM histogram reduce
// ============================================================================

template <uint32_t CS>
__device__ __forceinline__ void dsmem_hist_reduce(uint32_t* histogram) {
  static_assert(kHistBins <= kBlockSize);
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
__device__ __forceinline__ void find_threshold(uint32_t* histogram, uint32_t* warp_sum,
                                uint32_t* counter_gt, uint32_t* counter_eq,
                                MatchBin* match) {
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;
  const auto value = tx < kHistBins ? histogram[tx] : 0;
  const auto winc = warp_inclusive_sum(li, value);
  if (li == kWarpSize - 1) warp_sum[wi] = winc;
  __syncthreads();
  const auto tmp = warp_sum[li];
  const auto total = warp_reduce_sum_full(tmp);
  auto pfx = warp_reduce_sum_full(li < wi ? tmp : 0) + winc;
  const auto above = total - pfx;
  if (tx < kHistBins && above < TopK && above + value >= TopK) {
    *counter_gt = *counter_eq = 0;
    *match = {.bin = tx, .above_count = above, .equal_count = value};
  }
  __syncthreads();
}

// ============================================================================
// Register-based single-CTA fast path for seq_len <= 16384
// 4 float4 per thread × 1024 threads = 16384 elements max
// Uses 4096-bin (12-bit) histogram for better precision
// ============================================================================

struct RegSmem {
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[kRegHistBins];
    Tie tie_buffer[kMaxTies];
  };
};

template <uint32_t TopK, uint32_t HIST_BITS, uint32_t VECS_PER_THREAD = kRegVecsPerThread>
__device__ void register_topk(const float* __restrict__ scores,
                               int32_t* __restrict__ output,
                               uint32_t length, void* _smem) 
{
  constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  constexpr uint32_t ITEMS_PER_THREAD = HIST_BINS >= kBlockSize ? HIST_BINS / kBlockSize : 1;
  constexpr bool MULTI_ITEM = HIST_BINS >= kBlockSize;

  // Reinterpret smem with the right histogram size
  struct LocalSmem {
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    union { uint32_t histogram[HIST_BINS]; Tie tie_buffer[kMaxTies]; };
    static_assert(sizeof(uint32_t) * HIST_BINS >= TopK * sizeof(Tie),
                  "histogram union must be large enough for TopK ties");
  };
  auto* smem = static_cast<LocalSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Fused: zero histogram + load data (both independent, saves one __syncthreads__)
  float4 vecs[VECS_PER_THREAD];
  if constexpr (MULTI_ITEM && ITEMS_PER_THREAD >= 4) {
    for (uint32_t i = 0; i < ITEMS_PER_THREAD / 4; i++)
      reinterpret_cast<uint4*>(smem->histogram)[tx * (ITEMS_PER_THREAD / 4) + i] = make_uint4(0, 0, 0, 0);
  } else {
    if (tx < HIST_BINS) smem->histogram[tx] = 0;
  }
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
    const uint32_t base = (tx + v * kBlockSize) * 4;
    if (base < length) {
      vecs[v] = *reinterpret_cast<const float4*>(scores + base);
    }
  }
  __syncthreads();  // single barrier: histogram zeroed + data loaded

  // Build histogram from registers
  {
    bool done = false;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
      const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
      for (uint32_t e = 0; e < 4 && !done; e++) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx >= length) {
          done = true;
        } else {
          atomicAdd(&smem->histogram[extract_coarse_bin_N<HIST_BITS>(elems[e])], 1);
        }
      }
    }
  }
  __syncthreads();

  // Find threshold via prefix scan
  if constexpr (MULTI_ITEM) {
    // Multi-element scan (4096 bins: 4 per thread, 256 bins: not used here)
    uint32_t orig[ITEMS_PER_THREAD];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      orig[i] = smem->histogram[tx * ITEMS_PER_THREAD + i];
      local_sum += orig[i];
    }
    const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix = warp_reduce_sum_full(lane_id < warp_id ? tmp : 0);
    prefix += warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      prefix += orig[i];
      const auto above = length - prefix;
      if (above < TopK && above + orig[i] >= TopK) {
        smem->match = {.bin = tx * ITEMS_PER_THREAD + i,
                       .above_count = above, .equal_count = orig[i]};
      }
    }
  } else {
    // 1 bin per thread (256 bins, 1024 threads: only first 256 active)
    const uint32_t value = tx < HIST_BINS ? smem->histogram[tx] : 0;
    const auto warp_inc = warp_inclusive_sum(lane_id, value);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();
    const auto tmp = smem->warp_sum[lane_id];
    const auto total = warp_reduce_sum_full(tmp);
    auto pfx = warp_reduce_sum_full(lane_id < warp_id ? tmp : 0) + warp_inc;
    const auto above = total - pfx;
    if (tx < HIST_BINS && above < TopK && above + value >= TopK) {
      smem->match = {.bin = tx, .above_count = above, .equal_count = value};
    }
  }
  __syncthreads();

  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > TopK);

  // Scatter from registers
  {
    bool done = false;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD && !done; v++) {
      const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
      for (uint32_t e = 0; e < 4 && !done; e++) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx >= length) {
          done = true;
        } else {
          const uint32_t bin = extract_coarse_bin_N<HIST_BITS>(elems[e]);
          if (bin > thr_bin) {
            output[atomicAdd(&smem->counter_gt, 1)] = idx;
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1);
            if (!need_tie) {
              if (pos + num_above < TopK) {
                output[pos + num_above] = idx;
              }
            } else {
              if (pos < TopK) {
                smem->tie_buffer[pos] = {idx, elems[e]};
              }
            }
          }
        }
      }
    }
  }
  if (!need_tie) return;
  __syncthreads();

  // Fast warp-ballot tie-breaking for small tie counts
  const uint32_t num_ties = min(num_equal, static_cast<uint32_t>(TopK));
  const uint32_t topk_remain = TopK - num_above;

  auto is_greater = [](const Tie& a, const Tie& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (num_ties <= kWarpSize) {
    // Warp-level: each warp evaluates one candidate, ballot across lanes
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = smem->tie_buffer[lane_id];
    const auto target = smem->tie_buffer[warp_id];
    const bool pred = is_greater(tie, target);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      output[num_above + rank] = target.idx;
    }
  } else if (num_ties <= kWarpSize * 2) {
    // 64×64: each thread handles 2 elements
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto lane1 = lane_id + kWarpSize;
    const auto warp1 = warp_id + kWarpSize;
    const auto invalid = Tie{0xFFFFFFFF, -__FLT_MAX__};
    const auto tie0 = smem->tie_buffer[lane_id];
    const auto tie1 = lane1 < num_ties ? smem->tie_buffer[lane1] : invalid;
    if (warp_id < num_ties) {
      const auto target = smem->tie_buffer[warp_id];
      const auto r0 = __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 = __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
    if (warp1 < num_ties) {
      const auto target = smem->tie_buffer[warp1];
      const auto r0 = __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie0, target)));
      const auto r1 = __popc(__ballot_sync(0xFFFFFFFF, is_greater(tie1, target)));
      if (lane_id == 0 && r0 + r1 < topk_remain)
        output[num_above + r0 + r1] = target.idx;
    }
  } else {
    // Large tie count: fall back to full radix refinement
    if constexpr (TopK <= kBlockSize) {
      tie_handle<TopK>(smem->tie_buffer, num_ties, num_above, output, smem);
    } else {
      tie_handle_large<TopK>(smem->tie_buffer, num_ties, num_above, output, smem);
    }
  }
}

// ============================================================================
// Streaming single-CTA path for seq_len 16K–32K
// TMA double-buffer, 4096-bin histogram, no cluster sync
struct StreamSmem {
  uint64_t barrier[2][kStreamNumStages];
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[kRegHistBins];
    Tie tie_buffer[kMaxTies];
  };
  alignas(128) float score_buffer[kStreamNumStages][kSizePerStage];
};

// Streams data through shared memory in chunks, processing each chunk before loading the next
// overwrites each buffer after processing it (the epilogue prefetch loads the next chunk into the same slot)
template <typename SmemType, uint32_t kStages, uint32_t kBinBits, bool kIsScatter>
__device__ void tma_stream_pass(const float* scores, 
                                uint32_t length,
                                uint32_t thr_bin, 
                                int32_t* indices,
                                uint32_t* phases, 
                                SmemType* smem) 
{
  const auto tx   = threadIdx.x;
  const auto lane = tx % kWarpSize;
  const auto ni   = (length + kSizePerStage - 1) / kSizePerStage; // total stages needed
  const auto la   = (length + 3u) & ~3u; // length rounded up to float4 (TMA alignment)
  const auto pass = kIsScatter ? 1 : 0;  // barrier dim: [0] for histogram, [1] for scatter

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
                       &smem->barrier[pass][i]); // cp.async.bulk is non-blocking
      mbarrier_arrive_expect_tx(&smem->barrier[pass][i], sz);
    }
  }

  // Main loop: process stages
  for (uint32_t it = 0; it < ni; it++) {
    const auto b = it % kStages; // which buffer slot (0 or 1)
    const auto o = it * kSizePerStage;
    const auto sz = min(kSizePerStage, length - o);

    if (lane == 0) {
      mbarrier_wait(&smem->barrier[pass][b], phases[b] & 1); // wait for the data
    }
    phases[b]++; // advances the phase for next time this slot is reused
    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) {
        break;
      }
      const auto sc = smem->score_buffer[b][li];
      const auto bn = extract_coarse_bin_N<kBinBits>(sc);      
      if constexpr (kIsScatter) { // compile-time branch
        // Scatter pass: place above-threshold and collect ties
        const auto gi = o + li;
        if (bn > thr_bin) {
          indices[atomicAdd(&smem->counter_gt, 1)] = gi;
        } else if (bn == thr_bin) {
          const auto p = atomicAdd(&smem->counter_eq, 1);
          if (p < kMaxTies) {
            smem->tie_buffer[p] = {gi, sc};
          }
        }
      } else {
        // Histogram pass: just count
        atomicAdd(&smem->histogram[bn], 1);
      }
    }
    __syncthreads(); // ensures all threads finished processing their buffer before next TMA load

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
// kPasses=1 for single-pass (CS=8, CS=4 singlepass), kPasses=2 for two-pass (CS=4).
template <uint32_t kStages, uint32_t kPasses = 1>
struct SmemFused {
  uint64_t barrier[kPasses][kStages];
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union { uint32_t histogram[kHistBins]; Tie tie_buffer[kMaxTies]; };
  alignas(128) float score_buffer[kStages][kSizePerStage];
};

using Smem8 = SmemFused<kNumStages8>;
using Smem4 = SmemFused<kNumStages4, 2>;
using SmemSinglePass = SmemFused<kMaxSinglePassStages>;

template <uint32_t TopK, uint32_t CS, typename SmemType>
__device__ void large_topk_fused(const float* __restrict__ row_input,
                                 int32_t* __restrict__ row_output,
                                 uint32_t seq_len,
                                 uint32_t* phases, Tie* tie_ws) 
{
  const auto rank = blockIdx.y; // this block's position in cluster
  const auto tx = threadIdx.x;
  const auto lane = tx % kWarpSize;

  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<SmemType*>(smem_raw);
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw + sizeof(SmemType));

  // Phase 1: Partition row across cluster ranks.
  constexpr uint32_t kAlign = 4;
  const auto units    = (seq_len + kAlign - 1) / kAlign; //float4-aligned element count
  const auto base     = units / CS, extra = units % CS;  //elements per block
  const auto lu       = base + (rank < extra ? 1u : 0u); //remainder blocks
  const auto ou       = rank * base + min(rank, extra);  //this block's count (load-balanced)
  const auto my_start = ou * kAlign;                     //global start offset
  const auto my_len   = min(my_start + lu * kAlign, seq_len) - my_start; //actual length of this block
  const auto num_iters = (my_len + kSizePerStage - 1) / kSizePerStage;   //TMA stages needed
  const auto len_aligned = (my_len + 3u) & ~3u;

  // Fused init + TMA prologue
  if (tx < kHistBins) {
    smem->histogram[tx] = 0; // all threads zero histogram
  }
  if (tx == 0) { // thread 0 issues TMA - then all threads continue working until mbarrier sync
    smem->counter_gt = 0;
    smem->counter_eq = 0;
    for (uint32_t i = 0; i < num_iters; i++) {
      const auto off = i * kSizePerStage;
      const auto sz = min(kSizePerStage, len_aligned - off) * sizeof(float);
      tma_load(smem->score_buffer[i], row_input + my_start + off,
               sz, &smem->barrier[0][i]); //cp.async.bulk of size kSizePerStage × sizeof(float)
      mbarrier_arrive_expect_tx(&smem->barrier[0][i], sz);
    }
  }
  __syncthreads();

  // Phase 2: Histogram build. ILP unroll-by-2, no inter-stage sync
  for (uint32_t iter = 0; iter < num_iters; iter++) {
    const auto off = iter * kSizePerStage;
    const auto sz = min(kSizePerStage, my_len - off);
    if (lane == 0) {
      mbarrier_wait(&smem->barrier[0][iter], phases[iter] & 1); // wait for TMA
    }
    phases[iter]++;
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i += 2) {
      const auto li0 = tx + i * kBlockSize;
      const auto li1 = tx + (i + 1) * kBlockSize;
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

  // Phase 3: DSMEM all-reduce + find threshold
  dsmem_hist_reduce<CS>(smem->histogram); // each block histogram is summed across all CS blocks
  find_threshold<TopK>(smem->histogram, smem->warp_sum,
                       &smem->counter_gt, &smem->counter_eq, &smem->match);

  const auto thr = smem->match.bin;

  // Phase 4: Scatter. Rescan score_buffer (still in smem)
  for (uint32_t iter = 0; iter < num_iters; iter++) {
    const auto off = iter * kSizePerStage;
    const auto sz = min(kSizePerStage, my_len - off);
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) {
        break;
      }
      const auto score = smem->score_buffer[iter][li]; // still in smem
      const auto bin = extract_coarse_bin(score);
      const auto gidx = off + li;
      if (bin > thr) {
        s_topk[atomicAdd(&smem->counter_gt, 1)] = gidx; // above -> s_topk
      } else if (bin == thr) {
        const auto p = atomicAdd(&smem->counter_eq, 1); // equal -> ties (later refinement)
        if (p < kMaxTies) {
          smem->tie_buffer[p] = {gidx, score};
        }
      }
    }
  }
  __syncthreads();

  // Phase 5: Cross-block output collection via DSMEM prefix sum
  constexpr uint32_t kAboveBits = 16;
  constexpr uint32_t kAboveMask = (1 << kAboveBits) - 1;
  static_assert(kAboveMask >= TopK);
  static_assert(kAboveMask >= kMaxSinglePassPerBlock,
                "kAboveBits must cover max per-block element count");

  const uint32_t la = smem->counter_gt;
  const uint32_t le_full = smem->counter_eq;
  const uint32_t le = min(le_full, kMaxTies);  // smem tie_buffer cap

  __shared__ uint32_t s_local_counts[CS];
  __shared__ uint32_t s_prefix_packed;
  __shared__ uint32_t s_total_above, s_total_equal;
  
  auto cluster = cooperative_groups::this_cluster();
  if (tx < CS) {
    // Pack local counts into 32-bit: (equal << 16) | above
    const uint32_t packed = (le_full << kAboveBits) | la;
    const auto dst = cluster.map_shared_rank(s_local_counts, tx);
    dst[rank] = packed; // write my count to every block's s_local_counts[rank]
  }
  cluster.sync();

  // Thread 0 computes serial prefix sum
  if (tx == 0) {
    uint32_t prefix = 0, ta = 0, te = 0;
    for (uint32_t i = 0; i < CS; i++) {
      if (i == rank) {
        s_prefix_packed = prefix; // my prefix
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
  for (uint32_t i = tx; i < la; i += kBlockSize) {
    // indices are placed contiguously starting at prefix_above
    row_output[prefix_above + i] = s_topk[i] + my_start; // my_start: block-local -> row-global index                                 
  }
  for (uint32_t i = tx; i < le; i += kBlockSize) {
    const auto t = smem->tie_buffer[i];
    uint32_t p = s_total_above + prefix_equal + i;
    if (p < TopK) {
      row_output[p] = t.idx + my_start; 
    }
    uint32_t tp = prefix_equal + i;
    if (tp < (TopK <= kBlockSize ? kMaxTies : TopK)) {
      tie_ws[tp] = Tie{t.idx + my_start, t.score}; // Ties are copied for cross-block refinement
    }
  }

  // Phase 6: Tie refinement
  cooperative_groups::this_cluster().sync();
  if (rank != 0) { // only rank 0 does tie refinement
    return;
  }
  if (s_total_above + s_total_equal <= TopK) { // no ties to refine
    return;
  }

  if constexpr (TopK <= kBlockSize) {
    // copy ties from tie_ws back to smem, then refine
    const uint32_t num_ties = min(s_total_equal, kMaxTies);
    // TODO (roberto): could vectorize with uint2 (8 bytes = exactly one Tie)
    for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
      smem->tie_buffer[i] = Tie{tie_ws[i].idx, tie_ws[i].score};
    }
    __syncthreads();
    tie_handle<TopK>(smem->tie_buffer, num_ties, s_total_above, row_output, smem);
  } else {
    // TopK=2048: process directly from tie_ws (GMEM)
    const uint32_t num_ties = min(s_total_equal, static_cast<uint32_t>(TopK));
    tie_handle_large<TopK>(tie_ws, num_ties, s_total_above, row_output, smem);
  }
}

// Fallback path when data doesn't fit in SMEM for single-pass.
// Instead of keeping all TMA stages resident and rescanning, it streams data twice:
// once for histogram, once for scatter.
template <uint32_t TopK, uint32_t CS>
__device__ void large_topk_twopass(const float* __restrict__ ri,
                                   int32_t* __restrict__ ro,
                                   uint32_t sl, 
                                   ClusterState* state,
                                   uint32_t* hp, uint32_t* sp, Tie* tie_ws) 
{
  const auto rank = blockIdx.y, tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<Smem4*>(smem_raw);
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw + sizeof(Smem4));

  constexpr uint32_t kA = 4;
  const auto u = (sl + kA-1)/kA, b = u/CS, e = u%CS;
  const auto lu = b + (rank < e ? 1u : 0u);
  const auto ou = rank * b + min(rank, e);
  const auto ms = ou * kA, ml = min(ms + lu * kA, sl) - ms;

  // Phase 0: Init. Histogram zeroed, counters zeroed. Unlike large_topk_fused, 
  // there's no TMA prologue here.
  // tma_stream_pass handles its own TMA issuing internally.
  if (tx < kHistBins) smem->histogram[tx] = 0;
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
  }
  __syncthreads();

  // Phase 1: Histogram pass.
  tma_stream_pass<Smem4, kNumStages4, kHistBits, false>(ri + ms, ml, 0, nullptr, hp, smem);
  __syncthreads();

  // DSMEM all-reduce + find threshold
  dsmem_hist_reduce<CS>(smem->histogram);
  find_threshold<TopK>(smem->histogram, smem->warp_sum,
                  &smem->counter_gt, &smem->counter_eq, &smem->match);

  // Phase 2: Scatter pass
  tma_stream_pass<Smem4, kNumStages4, kHistBits, true>(ri + ms, ml, smem->match.bin, s_topk, sp, smem);
  __syncthreads();

  const uint32_t la = smem->counter_gt;
  const uint32_t le_full = smem->counter_eq;
  const uint32_t le = min(le_full, kMaxTies);  // smem tie_buffer cap

  // Phase 3: Output collection. Unlike large_topk_fused which uses DSMEM prefix sums, twopass uses
  // global atomicAdd on state->output_counter.
  __shared__ uint32_t s_off;
  // Step 1: Reserve slots for above-threshold indices.
  if (tx == 0) s_off = atomicAdd(&state->output_counter, (int)la);
  __syncthreads();
  for (uint32_t i = tx; i < la; i += kBlockSize) { ro[s_off + i] = s_topk[i] + ms; }

  // Step 2: cluster.sync() - all blocks done writting above-threshold
  cooperative_groups::this_cluster().sync();

  __shared__ uint32_t s_toff, s_ta;
  // Step 3: Read total above count (= how many positions filled so far)
  if (tx == 0) s_ta = state->output_counter;
  __syncthreads();
  cooperative_groups::this_cluster().sync(); // ensure all blocks read same s_ta
  // Step 4: Reserve slots for ties
  if (tx == 0) s_toff = atomicAdd(&state->output_counter, (int)le);
  __syncthreads();
  if (s_ta >= TopK) return; // all TopK already filled, no ties needed

  // Phase 4: Write ties to global output and workspace for refinement
  for (uint32_t i = tx; i < le; i += kBlockSize) {
    const auto t = smem->tie_buffer[i];
    uint32_t p = s_toff + i;
    if (p < TopK) { ro[p] = t.idx + ms; }
    uint32_t tp = s_toff - s_ta + i;
    if (tp < (TopK <= kBlockSize ? kMaxTies : TopK)) { 
      tie_ws[tp] = Tie{t.idx + ms, t.score}; 
    }
  }

  // Phase 5: Tie refinement
  cooperative_groups::this_cluster().sync();
  if (rank != 0) return;

  const auto tt = state->output_counter - s_ta; // total ties across all blocks
  if (tt <= TopK - s_ta) return; // all fit, no refinement needed
  if constexpr (TopK <= kBlockSize) {
    // copy ties from tie_ws back to smem, then refine
    // TODO (roberto): could vectorize with uint2 (8 bytes = exactly one Tie)
    for (uint32_t i = tx; i < min(tt, kMaxTies); i += kBlockSize) {
      smem->tie_buffer[i] = Tie{tie_ws[i].idx, tie_ws[i].score};
    }
    __syncthreads();
    tie_handle<TopK>(smem->tie_buffer, min(tt, kMaxTies), s_ta, ro, smem);
  } else {
    // TopK=2048: process directly from tie_ws (GMEM)
    tie_handle_large<TopK>(tie_ws, min(tt, static_cast<uint32_t>(TopK)), s_ta, ro, smem);
  }
}

// ============================================================================
// Adapted from https://github.com/sgl-project/sglang/pull/23600
// sgl-project/sglang (python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/)
// ============================================================================

template <uint32_t TopK, uint32_t CS>
__device__ void cooperative_topk_body(CooperativeTopKParams<TopK> params) 
{
  const auto rank = blockIdx.y, row = blockIdx.x, tx = threadIdx.x;
  const auto sl   = params.lengths[row];
  int32_t* out    = params.output + row * TopK;
  const float* in = params.input + row * params.stride;

  // Trivial: seq_len <= TopK
  if (sl <= static_cast<int32_t>(TopK)) {
    if (rank == 0) {
      for (uint32_t i = tx; i < TopK; i += kBlockSize) {
        out[i] = (i < static_cast<uint32_t>(sl)) ? static_cast<int32_t>(i) : -1;
      }
    }
    return;
  }

  // Short-Medium path: register_topk on rank 0 only - all data fits in RF
  if (sl <= static_cast<int32_t>(kRegMaxLen)) {
    if (rank == 0) {
      extern __shared__ uint8_t sr[];
      register_topk<TopK, 12>(in, out, sl, sr); // 4096-bin (12-bit) histogram
    }
    return;
  }

  // Large path: init mbarriers + state, then dispatch fused or twopass
  auto* st = &params.states[row];
  const uint32_t per_block = (params.stride + CS - 1) / CS; // how many elements per block
  constexpr uint32_t kFusedMax = ((CS == 8) ? kNumStages8 : kMaxSinglePassStages) * kSizePerStage;
  const bool use_singlepass = per_block <= kFusedMax; // single pass or TMA streaming: histogram+scatter

  // Select smem type and stage count at compile time based on CS
  constexpr uint32_t kFusedStages = (CS == 8) ? kNumStages8 : kMaxSinglePassStages;
  using FusedSmem = SmemFused<kFusedStages>;

  extern __shared__ uint8_t sr[];

  if (use_singlepass) {
    auto* smem = reinterpret_cast<FusedSmem*>(sr);
    const uint32_t sp_stages = (per_block + kSizePerStage - 1) / kSizePerStage;
    if (tx < sp_stages) {
      mbarrier_init(&smem->barrier[0][tx], 1); // init 1 barrier per TMA stage -
                                               // signal when async copies complete
    }
    __syncthreads();
    uint32_t phases[kFusedStages] = {}; // tracks the parity for mbarrier wait/arrive protocol
    large_topk_fused<TopK, CS, FusedSmem>(in, out, sl, phases, params.tie_ws + row * TopK);
  } else {
    // Two-pass: only CS=4 in practice (CS=8 always fits in singlepass)
    auto* smem = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kNumStages4) {
      mbarrier_init(&smem->barrier[0][tx], 1); // init 2×2=4 barriers (2 passes × 2 stages)
    }
    if (rank == 0 && tx == 0) {
      st->output_counter = 0;
      __threadfence(); // ensure visibility to other blocks via GMEM - needed for atomicAdd
    }
    __syncthreads();
    uint32_t hp[kNumStages4] = {0, 0}, sp_ph[kNumStages4] = {0, 0}; // histogram+scatter pass counters
    large_topk_twopass<TopK, CS>(in, out, sl, st, hp, sp_ph, params.tie_ws + row * TopK);
  }
}

template <uint32_t TopK>
__global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 4, 1) cooperative_topk_cs4(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 4>(params);
}

template <uint32_t TopK>
__global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 8, 1) cooperative_topk_cs8(CooperativeTopKParams<TopK> params) {
  cooperative_topk_body<TopK, 8>(params);
}

constexpr size_t kSmemSize4_base = (sizeof(Smem4) > sizeof(StreamSmem) ? sizeof(Smem4) : sizeof(StreamSmem));
constexpr size_t kSmemSize4_sp = sizeof(SmemSinglePass);
constexpr size_t kSmemSize4 = (kSmemSize4_base > kSmemSize4_sp ? kSmemSize4_base : kSmemSize4_sp) + sizeof(int32_t) * 2048 + 128;
constexpr size_t kSmemSize8 = std::max(sizeof(SmemFused<kNumStages8>), sizeof(StreamSmem)) + sizeof(int32_t) * 2048 + 128;

}  // namespace cooperative

// ============================================================================
// FilteredTopK — single CTA per row for bs > 32
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }

  FLASHINFER_INLINE void cast_store(T* ptr) const {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      ptr[i] = data[i];
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4
// refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return;
  }

  // Short path
  if (length <= 32768) {
    extern __shared__ uint8_t _smem_reg[];
    cooperative::register_topk<MAX_K, 12, 8>(score, dst, length, _smem_reg);
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  if (topk == 0) {
    // Collect indices where bin > threshold
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = base + j;
        }
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = i;
      }
    }
    __syncthreads();
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        filter_and_add_to_histogram(score_vec[j], base + j);
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      filter_and_add_to_histogram(score[i], i);
    }
    __syncthreads();

    // Stage 2: refine with 8bit radix passes
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      __shared__ int s_last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
      const auto num_input =
          (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_histogram[tx + 1];
      }
      __syncthreads();

      const auto threshold = s_threshold_bin_id;
      topk -= s_histogram[threshold + 1];

      const int offset = FIRST_SHIFT - round * 8;
      const bool is_last_round = (round == NUM_ROUNDS - 1);

      if (topk == 0) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto raw_input = score[idx];
          const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (static_cast<int>(bin) == threshold) {
            if (is_last_round) {
              const auto pos = atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                s_indices[top_k - pos] = idx;
              }
            } else {
              const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
              if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                s_input_idx[r_idx ^ 1][pos] = idx;
                const auto bin32 = Traits::ToOrdered(raw_input);
                const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Output phase - mode-specific
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    dst[base] = static_cast<IdType>(idx);
  }
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) 
{
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(DType* input, IdType* output_indices,
                                        IdType* lengths, uint32_t num_rows,
                                        uint32_t top_k_val, uint32_t max_len,
                                        cudaStream_t stream = 0) 
{
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                               \
  if (vec_size == VS) {                                                     \
    auto kernel = FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K>;      \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                              \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));   \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, \
                                          smem_size, stream));              \
    return cudaSuccess;                                                     \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace vllm

#endif  // COOPERATIVE_TOPK_CUH_
