#pragma once
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cstdint>

namespace cluster_topk {

constexpr uint32_t K = 1024;
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kHistBits = 10;
constexpr uint32_t kHistBins = 1 << kHistBits;
constexpr uint32_t RADIX = 256;
constexpr uint32_t kMaxTies = 1024;
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

struct alignas(16) MatchBin { uint32_t bin, above_count, equal_count; };
struct alignas(8) Tie { uint32_t idx; float score; };
struct ClusterState { int output_counter; };

struct Params {
  const float* __restrict__ input;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ lengths;
  ClusterState* __restrict__ states;
  Tie* __restrict__ tie_ws;  // per-cluster tie workspace, kMaxTies entries each
  uint32_t num_rows, stride;
};

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ uint32_t extract_coarse_bin(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? (uint16_t)(~bits) : (uint16_t)(bits | 0x8000);
  return key >> (16 - kHistBits);
}

template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? (uint16_t)(~bits) : (uint16_t)(bits | 0x8000);
  return key >> (16 - kBits);
}

__device__ __forceinline__ uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
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

namespace ptx_h {
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
}

// ============================================================================
// Tie refinement (single CTA)
// ============================================================================

__device__ void tie_handle(const Tie* ties, uint32_t num_ties, uint32_t num_above,
                            int32_t* output, void* _smem) {
  struct TS { alignas(128) uint32_t counter; alignas(128) MatchBin match;
              uint32_t histogram[RADIX]; uint32_t warp_sum[kNumWarps]; };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;
  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = extract_exact_bin(tie.score);
  bool active = has; uint32_t remain = K - num_above, wpos = K;
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
      else if (r == 3) wpos = K - atomicAdd(&s->match.equal_count, -1u);
    }
    remain -= na; if (!remain) break;
  }
  if (wpos < K) output[wpos] = tie.idx;
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
  if (tx < kHistBins && above < K && above + value >= K) {
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

constexpr uint32_t kRegHistBits = 12;
constexpr uint32_t kRegHistBins = 1 << kRegHistBits;  // 4096
constexpr uint32_t kRegVecsPerThread = 4;
constexpr uint32_t kRegMaxLen = kRegVecsPerThread * 4 * kBlockSize;  // 16384
constexpr uint32_t kRegHistItems = kRegHistBins / kBlockSize;  // 4

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

template <uint32_t HIST_BITS, uint32_t VECS_PER_THREAD = kRegVecsPerThread>
__device__ void register_topk(const float* __restrict__ scores,
                               int32_t* __restrict__ output,
                               uint32_t length, void* _smem) {
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
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
    const uint32_t base = (tx + v * kBlockSize) * 4;
    if (base < length) {
      vecs[v] = *reinterpret_cast<const float4*>(scores + base);
    }
  }
  __syncthreads();  // single barrier: histogram zeroed + data loaded

  // Build histogram from registers
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) goto hist_done;
      atomicAdd(&smem->histogram[extract_coarse_bin_N<HIST_BITS>(elems[e])], 1);
    }
  }
hist_done:
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
      if (above < K && above + orig[i] >= K) {
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
    if (tx < HIST_BINS && above < K && above + value >= K) {
      smem->match = {.bin = tx, .above_count = above, .equal_count = value};
    }
  }
  __syncthreads();

  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > K);

  // Scatter from registers
#pragma unroll
  for (uint32_t v = 0; v < VECS_PER_THREAD; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) goto scatter_done;
      const uint32_t bin = extract_coarse_bin_N<HIST_BITS>(elems[e]);
      if (bin > thr_bin) {
        output[atomicAdd(&smem->counter_gt, 1)] = idx;
      } else if (bin == thr_bin) {
        const auto pos = atomicAdd(&smem->counter_eq, 1);
        if (!need_tie) {
          if (pos + num_above < K) output[pos + num_above] = idx;
        } else {
          if (pos < kMaxTies) smem->tie_buffer[pos] = {idx, elems[e]};
        }
      }
    }
  }
scatter_done:
  if (!need_tie) return;
  __syncthreads();

  // Fast warp-ballot tie-breaking for small tie counts
  const uint32_t num_ties = min(num_equal, kMaxTies);
  const uint32_t topk_remain = K - num_above;

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
    tie_handle(smem->tie_buffer, num_ties, num_above, output, smem);
  }
}

// ============================================================================
// Streaming single-CTA path for seq_len 16K–32K
// TMA double-buffer, 4096-bin histogram, no cluster sync
// ============================================================================

// Streaming path disabled — cluster path is faster at all sizes > 16K
constexpr uint32_t kStreamMaxLen = kRegMaxLen;  // effectively skip streaming
constexpr uint32_t kStreamNumStages = 2;

struct StreamSmem {
  uint64_t barrier[2][kStreamNumStages];  // 2 passes × 2 stages
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union {
    uint32_t histogram[kRegHistBins];  // 4096 bins (12-bit)
    Tie tie_buffer[kMaxTies];
  };
  alignas(128) float score_buffer[kStreamNumStages][kSizePerStage];
};

template <bool kIsScatter>
__device__ void stream_pass_single(const float* scores, uint32_t length,
                                    uint32_t thr_bin, int32_t* indices,
                                    uint32_t* phases, StreamSmem* smem) {
  const auto tx = threadIdx.x;
  const auto lane = tx % kWarpSize;
  const auto ni = (length + kSizePerStage - 1) / kSizePerStage;
  const auto la = (length + 3u) & ~3u;
  const auto pass = kIsScatter ? 1 : 0;

  if (tx == 0) {
#pragma unroll
    for (uint32_t i = 0; i < kStreamNumStages; i++) {
      if (i >= ni) break;
      const auto o = i * kSizePerStage;
      const auto sz = min(kSizePerStage, la - o) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[i], scores + o, sz,
                        &smem->barrier[pass][i]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[pass][i], sz);
    }
  }

  for (uint32_t it = 0; it < ni; it++) {
    const auto b = it % kStreamNumStages;
    const auto o = it * kSizePerStage;
    const auto sz = min(kSizePerStage, length - o);

    if (lane == 0)
      ptx_h::mbarrier_wait(&smem->barrier[pass][b], phases[b] & 1);
    phases[b]++;
    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      const auto sc = smem->score_buffer[b][li];
      const auto bn = extract_coarse_bin_N<kRegHistBits>(sc);
      if constexpr (kIsScatter) {
        const auto gi = o + li;
        if (bn > thr_bin) indices[atomicAdd(&smem->counter_gt, 1)] = gi;
        else if (bn == thr_bin) {
          const auto p = atomicAdd(&smem->counter_eq, 1);
          if (p < kMaxTies) smem->tie_buffer[p] = {gi, sc};
        }
      } else {
        atomicAdd(&smem->histogram[bn], 1);
      }
    }
    __syncthreads();
    if (tx == 0 && it + kStreamNumStages < ni) {
      const auto no = (it + kStreamNumStages) * kSizePerStage;
      const auto nsz = min(kSizePerStage, la - no) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[b], scores + no, nsz,
                        &smem->barrier[pass][b]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[pass][b], nsz);
    }
  }
}

__device__ void streaming_topk(const float* __restrict__ scores,
                                int32_t* __restrict__ output,
                                uint32_t length, void* _smem,
                                uint32_t* hist_phases, uint32_t* scatter_phases) {
  auto* smem = static_cast<StreamSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Init histogram + counters
  for (uint32_t i = tx; i < kRegHistBins; i += kBlockSize)
    smem->histogram[i] = 0;
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
  __syncthreads();

  // Phase 1: TMA streaming histogram
  stream_pass_single<false>(scores, length, 0, nullptr, hist_phases, smem);
  __syncthreads();

  // Find threshold (4 bins per thread, same as register path)
  {
    uint32_t orig[kRegHistItems];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kRegHistItems; i++) {
      orig[i] = smem->histogram[tx * kRegHistItems + i];
      local_sum += orig[i];
    }
    const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix = warp_reduce_sum_full(lane_id < warp_id ? tmp : 0);
    prefix += warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kRegHistItems; i++) {
      prefix += orig[i];
      const auto above = length - prefix;
      if (above < K && above + orig[i] >= K) {
        smem->counter_gt = smem->counter_eq = 0;
        smem->match = {.bin = tx * kRegHistItems + i,
                       .above_count = above, .equal_count = orig[i]};
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > K);

  // Phase 2: TMA streaming scatter
  stream_pass_single<true>(scores, length, thr_bin, output,
                            scatter_phases, smem);
  __syncthreads();

  if (!need_tie) {
    // Ties fit directly — already written in scatter pass
    return;
  }

  // Tie refinement
  tie_handle(smem->tie_buffer, min(num_equal, kMaxTies), num_above, output, smem);
}

// ============================================================================
// CS=8 Fused path: single TMA pass, rescan smem for scatter
// ============================================================================

struct Smem8 {
  uint64_t barrier[kNumStages8];
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union { uint32_t histogram[kHistBins]; Tie tie_buffer[kMaxTies]; };
  alignas(128) float score_buffer[kNumStages8][kSizePerStage];
};

__device__ void large_topk_fused8(const float* __restrict__ row_input,
                                   int32_t* __restrict__ row_output,
                                   uint32_t seq_len, ClusterState* state,
                                   uint32_t* phases, Tie* tie_ws) {
  constexpr uint32_t CS = 8;
  const auto rank = blockIdx.y;
  const auto tx = threadIdx.x;

  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<Smem8*>(smem_raw);
  __shared__ int32_t s_topk[K];

  // Partition
  constexpr uint32_t kAlign = 4;
  const auto units = (seq_len + kAlign - 1) / kAlign;
  const auto base = units / CS, extra = units % CS;
  const auto lu = base + (rank < extra ? 1u : 0u);
  const auto ou = rank * base + min(rank, extra);
  const auto my_start = ou * kAlign;
  const auto my_len = min(my_start + lu * kAlign, seq_len) - my_start;
  const auto num_iters = (my_len + kSizePerStage - 1) / kSizePerStage;
  const auto len_aligned = (my_len + 3u) & ~3u;

  // Fused init + TMA prologue: zero histogram AND issue TMA concurrently
  // TMA writes to score_buffer (independent of histogram), so no ordering conflict
  if (tx < kHistBins) smem->histogram[tx] = 0;
  if (tx == 0) {
    smem->counter_gt = 0; smem->counter_eq = 0;
    // Issue TMA while other threads zero histogram
#pragma unroll
    for (uint32_t i = 0; i < kNumStages8; i++) {
      if (i >= num_iters) break;
      const auto off = i * kSizePerStage;
      const auto sz = min(kSizePerStage, len_aligned - off) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[i], row_input + my_start + off,
                        sz, &smem->barrier[i]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[i], sz);
    }
  }
  __syncthreads();  // histogram zeroed + TMA issued

  // Histogram pass: no sync between stages (each uses its own score_buffer)
  {
    const auto lane = tx % kWarpSize;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
      const auto off = iter * kSizePerStage;
      const auto sz = min(kSizePerStage, my_len - off);
      if (lane == 0) ptx_h::mbarrier_wait(&smem->barrier[iter], phases[iter] & 1);
      phases[iter]++;
      __syncwarp();
      // Unrolled by 2 for ILP: compute 2 bins before issuing atomicAdds
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i += 2) {
        const auto li0 = tx + i * kBlockSize;
        const auto li1 = tx + (i+1) * kBlockSize;
        if (li0 >= sz) break;
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
  }
  // No __syncthreads needed — dsmem_hist_reduce starts with cluster.sync()
  // which is a superset (synchronizes all threads across all blocks)

  // DSMEM all-reduce + find threshold (2 cluster.sync() total)
  dsmem_hist_reduce<CS>(smem->histogram);
  find_threshold(smem->histogram, smem->warp_sum,
                  &smem->counter_gt, &smem->counter_eq, &smem->match);

  const auto thr = smem->match.bin;

  // Scatter: rescan score_buffer (still in smem — no re-streaming!)
  for (uint32_t iter = 0; iter < num_iters; iter++) {
    const auto off = iter * kSizePerStage;
    const auto sz = min(kSizePerStage, my_len - off);
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      const auto score = smem->score_buffer[iter][li];
      const auto bin = extract_coarse_bin(score);
      const auto gidx = off + li;
      if (bin > thr) s_topk[atomicAdd(&smem->counter_gt, 1)] = gidx;
      else if (bin == thr) {
        const auto p = atomicAdd(&smem->counter_eq, 1);
        if (p < kMaxTies) smem->tie_buffer[p] = {gidx, score};
      }
    }
  }
  __syncthreads();

  // Cross-block output collection via DSMEM prefix sum
  // ONE cluster.sync() instead of two sequential global atomics
  constexpr uint32_t kAboveBits2 = 16;
  constexpr uint32_t kAboveMask2 = (1 << kAboveBits2) - 1;
  static_assert(kAboveMask2 >= K);

  const uint32_t la = smem->counter_gt, le = smem->counter_eq;
  const auto midx = tx < la ? s_topk[tx] : 0;
  const auto mtie = tx < le ? smem->tie_buffer[tx] : Tie{0, 0.0f};

  // Push packed (above, equal) counts to all blocks via DSMEM
  __shared__ uint32_t s_local_counts[CS];
  __shared__ uint32_t s_prefix_packed;
  __shared__ uint32_t s_total_above, s_total_equal;
  {
    auto cluster = cooperative_groups::this_cluster();
    if (tx < CS) {
      const uint32_t packed = (le << kAboveBits2) | la;
      const auto dst = cluster.map_shared_rank(s_local_counts, tx);
      dst[rank] = packed;
    }
    cluster.sync();  // single sync: counts visible + all scatter writes done

    // Compute prefix + totals in one shot (thread 0 only)
    if (tx == 0) {
      uint32_t prefix = 0, ta = 0, te = 0;
      for (uint32_t i = 0; i < CS; i++) {
        if (i == rank) s_prefix_packed = prefix;
        ta += s_local_counts[i] & kAboveMask2;
        te += s_local_counts[i] >> kAboveBits2;
        prefix += s_local_counts[i];
      }
      s_total_above = ta;
      s_total_equal = te;
    }
  }
  __syncthreads();  // single sync: prefix + totals visible

  const uint32_t prefix_above = s_prefix_packed & kAboveMask2;
  const uint32_t prefix_equal = s_prefix_packed >> kAboveBits2;

  // Write above-threshold indices at prefix offset
  if (tx < la) row_output[prefix_above + tx] = midx + my_start;

  // Write ties within K to output, and ALL ties to workspace for refinement
  // tie_ws passed as parameter
  if (tx < le) {
    uint32_t p = s_total_above + prefix_equal + tx;
    if (p < K) row_output[p] = mtie.idx + my_start;
    uint32_t tp = prefix_equal + tx;
    if (tp < kMaxTies) tie_ws[tp] = Tie{mtie.idx + my_start, mtie.score};
  }

  cooperative_groups::this_cluster().sync();
  if (rank != 0) return;

  if (s_total_above + s_total_equal <= K) return;

  const uint32_t num_ties = min(s_total_equal, kMaxTies);
  __syncthreads();
  for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
    smem->tie_buffer[i] = Tie{tie_ws[i].idx, tie_ws[i].score};
  }
  __syncthreads();
  tie_handle(smem->tie_buffer, num_ties, s_total_above, row_output, smem);
}

// ============================================================================
// CS=4 Two-pass path (unchanged from before)
// ============================================================================

struct Smem4 {
  uint64_t barrier[2][kNumStages4];  // 2 passes × 2 stages
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union { uint32_t histogram[kHistBins]; Tie tie_buffer[kMaxTies]; };
  alignas(128) float score_buffer[kNumStages4][kSizePerStage];
  // 4 stages × 8192 × 4 = 128KB for score_buffer + ~20KB overhead ≈ 148KB total
};

template <bool kIsScatter>
__device__ void stream_pass4(const float* scores, uint32_t length,
                              uint32_t thr_bin, int32_t* topk_indices,
                              uint32_t* phases, Smem4* smem) {
  const auto tx = threadIdx.x;
  const auto lane = tx % kWarpSize;
  const auto ni = (length + kSizePerStage - 1) / kSizePerStage;
  const auto la = (length + 3u) & ~3u;
  const auto pass = kIsScatter ? 1 : 0;

  // Prologue: issue initial TMA loads
  if (tx == 0) {
#pragma unroll
    for (uint32_t i = 0; i < kNumStages4; i++) {
      if (i >= ni) break;
      const auto o = i * kSizePerStage;
      const auto sz = min(kSizePerStage, la - o) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[i], scores + o, sz, &smem->barrier[pass][i]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[pass][i], sz);
    }
  }

  for (uint32_t it = 0; it < ni; it++) {
    const auto b = it % kNumStages4;
    const auto o = it * kSizePerStage;
    const auto sz = min(kSizePerStage, length - o);

    // Wait for current buffer
    if (lane == 0) ptx_h::mbarrier_wait(&smem->barrier[pass][b], phases[b] & 1);
    phases[b]++;
    __syncwarp();

    // Process current buffer
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      const auto sc = smem->score_buffer[b][li];
      const auto bn = extract_coarse_bin(sc);
      if constexpr (kIsScatter) {
        if (bn > thr_bin) topk_indices[atomicAdd(&smem->counter_gt, 1)] = o + li;
        else if (bn == thr_bin) {
          const auto p = atomicAdd(&smem->counter_eq, 1);
          if (p < kMaxTies) smem->tie_buffer[p] = {o + li, sc};
        }
      } else { atomicAdd(&smem->histogram[bn], 1); }
    }
    __syncthreads();
    if (tx == 0 && it + kNumStages4 < ni) {
      const auto no = (it + kNumStages4) * kSizePerStage;
      const auto nsz = min(kSizePerStage, la - no) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[b], scores + no, nsz,
                        &smem->barrier[pass][b]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[pass][b], nsz);
    }
  }
}

// Single-pass smem: all TMA stages resident simultaneously
constexpr uint32_t kMaxSinglePassStages = 3;  // covers up to ~196K with CS=4 at 16K/stage
constexpr uint32_t kMaxSinglePassPerBlock = kMaxSinglePassStages * kSizePerStage; // 49152

struct SmemSinglePass {
  uint64_t barrier[kMaxSinglePassStages];
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBin match;
  uint32_t warp_sum[kNumWarps];
  union { uint32_t histogram[kHistBins]; Tie tie_buffer[kMaxTies]; };
  alignas(128) float score_buffer[kMaxSinglePassStages][kSizePerStage];
};

__device__ void large_topk_singlepass4(const float* __restrict__ ri,
                                        int32_t* __restrict__ ro,
                                        uint32_t sl, ClusterState* state,
                                        uint32_t* phases, Tie* tie_ws) {
  constexpr uint32_t CS = 4;
  const auto rank = blockIdx.y, tx = threadIdx.x;
  const auto lane = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;
  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<SmemSinglePass*>(smem_raw);
  __shared__ int32_t s_topk[K];

  constexpr uint32_t kA = 4;
  const auto u = (sl + kA-1)/kA, b = u/CS, e = u%CS;
  const auto lu = b + (rank < e ? 1u : 0u);
  const auto ou = rank * b + min(rank, e);
  const auto ms = ou * kA, ml = min(ms + lu * kA, sl) - ms;
  const auto num_iters = (ml + kSizePerStage - 1) / kSizePerStage;
  const auto ml_aligned = (ml + 3u) & ~3u;

  // Init
  if (tx < kHistBins) smem->histogram[tx] = 0;
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
  __syncthreads();

  // TMA prologue: load ALL stages at once (data stays in smem for rescan)
  if (tx == 0) {
    for (uint32_t i = 0; i < num_iters; i++) {
      const auto off = i * kSizePerStage;
      const auto sz = min(kSizePerStage, ml_aligned - off) * sizeof(float);
      ptx_h::tma_load(smem->score_buffer[i], ri + ms + off, sz, &smem->barrier[i]);
      ptx_h::mbarrier_arrive_expect_tx(&smem->barrier[i], sz);
    }
  }

  // Phase 1: Histogram — process all stages, no sync between stages
  // (each stage uses its own score_buffer[it], histogram updates are atomic)
  for (uint32_t it = 0; it < num_iters; it++) {
    const auto off = it * kSizePerStage;
    const auto sz = min(kSizePerStage, ml - off);
    if (lane == 0) ptx_h::mbarrier_wait(&smem->barrier[it], phases[it] & 1);
    phases[it]++;
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      atomicAdd(&smem->histogram[extract_coarse_bin(smem->score_buffer[it][li])], 1);
    }
  }
  __syncthreads();  // single sync after all stages — before DSMEM reduce

  // DSMEM all-reduce + find threshold
  dsmem_hist_reduce<CS>(smem->histogram);
  find_threshold(smem->histogram, smem->warp_sum,
                  &smem->counter_gt, &smem->counter_eq, &smem->match);

  const auto thr_bin = smem->match.bin;

  // Phase 2: Scatter — rescan score_buffer (still in smem, no re-streaming!)
  for (uint32_t it = 0; it < num_iters; it++) {
    const auto off = it * kSizePerStage;
    const auto sz = min(kSizePerStage, ml - off);
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      const auto score = smem->score_buffer[it][li];
      const auto bin = extract_coarse_bin(score);
      const auto gidx = off + li;
      if (bin > thr_bin) s_topk[atomicAdd(&smem->counter_gt, 1)] = gidx;
      else if (bin == thr_bin) {
        const auto p = atomicAdd(&smem->counter_eq, 1);
        if (p < kMaxTies) smem->tie_buffer[p] = {gidx, score};
      }
    }
  }
  __syncthreads();

  // Output collection: above first, then ties (2 cluster.sync())
  const uint32_t la = smem->counter_gt, le = smem->counter_eq;
  const auto midx = tx < la ? s_topk[tx] : 0;
  const auto mtie = tx < le ? smem->tie_buffer[tx] : Tie{0, 0.0f};

  __shared__ uint32_t s_off;
  if (tx == 0) s_off = atomicAdd(&state->output_counter, (int)la);
  __syncthreads();
  if (tx < la) ro[s_off + tx] = midx + ms;

  cooperative_groups::this_cluster().sync();

  // Read total above count BEFORE any block does atomicAdd for ties
  __shared__ uint32_t s_toff, s_ta;
  if (tx == 0) s_ta = state->output_counter;
  __syncthreads();
  // Now all threads see the same s_ta. Only then reserve tie slots.
  // But s_ta is per-block shared memory — all blocks read state->output_counter
  // at the same point (right after cluster.sync). Use cluster.sync to ensure ordering.
  cooperative_groups::this_cluster().sync();
  if (tx == 0) s_toff = atomicAdd(&state->output_counter, (int)le);
  __syncthreads();
  if (s_ta >= K) return;
  if (tx < le) {
    uint32_t p = s_toff + tx;
    if (p < K) ro[p] = mtie.idx + ms;
    // Store all ties to workspace for tie refinement
    uint32_t tp = s_toff - s_ta + tx;
    if (tp < kMaxTies) tie_ws[tp] = Tie{mtie.idx + ms, mtie.score};
  }

  cooperative_groups::this_cluster().sync();
  if (rank != 0) return;

  const auto tt = state->output_counter - s_ta;
  if (tt <= K - s_ta) return;
  __syncthreads();
  for (uint32_t i = tx; i < min(tt, kMaxTies); i += kBlockSize) {
    smem->tie_buffer[i] = Tie{tie_ws[i].idx, tie_ws[i].score};
  }
  __syncthreads();
  tie_handle(smem->tie_buffer, min(tt, kMaxTies), s_ta, ro, smem);
}

__device__ void large_topk_twopass4(const float* __restrict__ ri,
                                     int32_t* __restrict__ ro,
                                     uint32_t sl, ClusterState* state,
                                     uint32_t* hp, uint32_t* sp, Tie* tie_ws) {
  constexpr uint32_t CS = 4;
  const auto rank = blockIdx.y, tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];
  auto* smem = reinterpret_cast<Smem4*>(smem_raw);
  __shared__ int32_t s_topk[K];

  constexpr uint32_t kA = 4;
  const auto u = (sl + kA-1)/kA, b = u/CS, e = u%CS;
  const auto lu = b + (rank < e ? 1u : 0u);
  const auto ou = rank * b + min(rank, e);
  const auto ms = ou * kA, ml = min(ms + lu * kA, sl) - ms;

  if (tx < kHistBins) smem->histogram[tx] = 0;
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
  __syncthreads();

  stream_pass4<false>(ri + ms, ml, 0, nullptr, hp, smem);
  __syncthreads();

  dsmem_hist_reduce<CS>(smem->histogram);
  find_threshold(smem->histogram, smem->warp_sum,
                  &smem->counter_gt, &smem->counter_eq, &smem->match);

  stream_pass4<true>(ri + ms, ml, smem->match.bin, s_topk, sp, smem);
  __syncthreads();

  const uint32_t la = smem->counter_gt, le = smem->counter_eq;
  const auto midx = tx < la ? s_topk[tx] : 0;
  const auto mtie = tx < le ? smem->tie_buffer[tx] : Tie{0, 0.0f};

  __shared__ uint32_t s_off;
  if (tx == 0) s_off = atomicAdd(&state->output_counter, (int)la);
  __syncthreads();
  if (tx < la) ro[s_off + tx] = midx + ms;

  cooperative_groups::this_cluster().sync();

  __shared__ uint32_t s_toff, s_ta;
  if (tx == 0) s_ta = state->output_counter;
  __syncthreads();
  cooperative_groups::this_cluster().sync();
  if (tx == 0) s_toff = atomicAdd(&state->output_counter, (int)le);
  __syncthreads();
  if (s_ta >= K) return;
    if (tx < le) {
    uint32_t p = s_toff + tx;
    if (p < K) ro[p] = mtie.idx + ms;
    uint32_t tp = s_toff - s_ta + tx;
    if (tp < kMaxTies) tie_ws[tp] = Tie{mtie.idx + ms, mtie.score};
  }

  // cluster.sync() before tie refinement
  cooperative_groups::this_cluster().sync();
  if (rank != 0) return;

  const auto tt = state->output_counter - s_ta;
  if (tt <= K - s_ta) return;
  __syncthreads();
  for (uint32_t i = tx; i < min(tt, kMaxTies); i += kBlockSize) {
    smem->tie_buffer[i] = Tie{tie_ws[i].idx, tie_ws[i].score};
  }
  __syncthreads();
  tie_handle(smem->tie_buffer, min(tt, kMaxTies), s_ta, ro, smem);
}

// ============================================================================
// Non-cluster kernel for small seq_lens (≤16K)
// Plain <<<num_rows, kBlockSize>>> launch — no cluster overhead
// ============================================================================

__global__ void __launch_bounds__(kBlockSize, 1)
    register_topk_kernel(Params params) {
  const auto row = blockIdx.x;
  if (row >= params.num_rows) return;
  const auto tx = threadIdx.x;
  const uint32_t sl = params.lengths[row];
  int32_t* out = params.output + row * K;
  const float* in = params.input + row * params.stride;

  if (sl <= K) {
    if (tx < sl) out[tx] = tx;
    else if (tx < K) out[tx] = -1;
  } else if (sl <= kRegMaxLen) {
    extern __shared__ uint8_t smem[];
    register_topk<12>(in, out, sl, smem);
  } else {
    // Streaming single-CTA path for large seq_lens (no cluster overhead)
    extern __shared__ uint8_t smem[];
    auto* ss = reinterpret_cast<StreamSmem*>(smem);

    // Init mbarriers for this CTA
    if (tx < 2 * kStreamNumStages) {
      ptx_h::mbarrier_init(&ss->barrier[0][0] + tx, 1);
    }
    if (tx < kRegHistBins) ss->histogram[tx] = 0;
    if (tx == 0) { ss->counter_gt = 0; ss->counter_eq = 0; }
    __syncthreads();

    // Phase 1: TMA streaming histogram (4096-bin)
    const auto lane = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto ni = (sl + kSizePerStage - 1) / kSizePerStage;
    const auto la = (sl + 3u) & ~3u;

    if (tx == 0) {
      for (uint32_t i = 0; i < kStreamNumStages && i < ni; i++) {
        const auto o = i * kSizePerStage;
        const auto sz = min(kSizePerStage, la - o) * sizeof(float);
        ptx_h::tma_load(ss->score_buffer[i], in + o, sz, &ss->barrier[0][i]);
        ptx_h::mbarrier_arrive_expect_tx(&ss->barrier[0][i], sz);
      }
    }

    uint32_t hp[kStreamNumStages] = {0, 0};
    for (uint32_t it = 0; it < ni; it++) {
      const auto b = it % kStreamNumStages;
      const auto o = it * kSizePerStage;
      const auto sz = min(kSizePerStage, sl - o);
      if (lane == 0) ptx_h::mbarrier_wait(&ss->barrier[0][b], hp[b] & 1);
      hp[b]++;
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i++) {
        const auto li = tx + i * kBlockSize;
        if (li >= sz) break;
        atomicAdd(&ss->histogram[extract_coarse_bin_N<kRegHistBits>(ss->score_buffer[b][li])], 1);
      }
      __syncthreads();
      if (tx == 0 && it + kStreamNumStages < ni) {
        const auto no = (it + kStreamNumStages) * kSizePerStage;
        const auto nsz = min(kSizePerStage, la - no) * sizeof(float);
        ptx_h::tma_load(ss->score_buffer[b], in + no, nsz, &ss->barrier[0][b]);
        ptx_h::mbarrier_arrive_expect_tx(&ss->barrier[0][b], nsz);
      }
    }

    // Find threshold (4096-bin, 4 items per thread)
    {
      const auto lane_id = tx % kWarpSize;
      uint32_t orig[kRegHistItems];
      uint32_t local_sum = 0;
      for (uint32_t i = 0; i < kRegHistItems; i++) {
        orig[i] = ss->histogram[tx * kRegHistItems + i];
        local_sum += orig[i];
      }
      const auto warp_inc = warp_inclusive_sum(lane_id, local_sum);
      if (lane_id == kWarpSize - 1) ss->warp_sum[warp_id] = warp_inc;
      __syncthreads();
      const auto tmp = ss->warp_sum[lane_id];
      uint32_t prefix = warp_reduce_sum_full(lane_id < warp_id ? tmp : 0);
      prefix += warp_inc - local_sum;
      for (uint32_t i = 0; i < kRegHistItems; i++) {
        prefix += orig[i];
        const auto above = sl - prefix;
        if (above < K && above + orig[i] >= K) {
          ss->counter_gt = ss->counter_eq = 0;
          ss->match = {.bin = tx * kRegHistItems + i,
                       .above_count = above, .equal_count = orig[i]};
        }
      }
      __syncthreads();
    }

    const auto thr_bin = ss->match.bin;

    // Phase 2: TMA streaming scatter
    if (tx == 0) {
      for (uint32_t i = 0; i < kStreamNumStages && i < ni; i++) {
        const auto o = i * kSizePerStage;
        const auto sz = min(kSizePerStage, la - o) * sizeof(float);
        ptx_h::tma_load(ss->score_buffer[i], in + o, sz, &ss->barrier[1][i]);
        ptx_h::mbarrier_arrive_expect_tx(&ss->barrier[1][i], sz);
      }
    }

    uint32_t sp2[kStreamNumStages] = {0, 0};
    for (uint32_t it = 0; it < ni; it++) {
      const auto b = it % kStreamNumStages;
      const auto o = it * kSizePerStage;
      const auto sz = min(kSizePerStage, sl - o);
      if (lane == 0) ptx_h::mbarrier_wait(&ss->barrier[1][b], sp2[b] & 1);
      sp2[b]++;
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; i++) {
        const auto li = tx + i * kBlockSize;
        if (li >= sz) break;
        const auto sc = ss->score_buffer[b][li];
        const auto bn = extract_coarse_bin_N<kRegHistBits>(sc);
        const auto gi = o + li;
        if (bn > thr_bin) out[atomicAdd(&ss->counter_gt, 1)] = gi;
        else if (bn == thr_bin) {
          const auto p = atomicAdd(&ss->counter_eq, 1);
          if (p < kMaxTies) ss->tie_buffer[p] = {gi, sc};
        }
      }
      __syncthreads();
      if (tx == 0 && it + kStreamNumStages < ni) {
        const auto no = (it + kStreamNumStages) * kSizePerStage;
        const auto nsz = min(kSizePerStage, la - no) * sizeof(float);
        ptx_h::tma_load(ss->score_buffer[b], in + no, nsz, &ss->barrier[1][b]);
        ptx_h::mbarrier_arrive_expect_tx(&ss->barrier[1][b], nsz);
      }
    }

    // Tie handling
    const auto [_, num_above, num_equal] = ss->match;
    if (num_equal + num_above > K) {
      __syncthreads();
      tie_handle(ss->tie_buffer, min(num_equal, kMaxTies), num_above, out, ss);
    }
  }
}

// ============================================================================
// Cluster kernel variants
// ============================================================================

template <uint32_t CS> __global__ void __launch_bounds__(kBlockSize, 1)
    cluster_persistent_topk_impl(Params params);

template <> __global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 4, 1) cluster_persistent_topk_impl<4>(Params params) {
  const auto rank = blockIdx.y, row = blockIdx.x, tx = threadIdx.x;
  const auto sl = params.lengths[row];
  int32_t* out = params.output + row * K;
  const float* in = params.input + row * params.stride;

  // Register/trivial: only rank 0 works, then return
  if (sl <= K) {
    if (rank == 0) { if (tx < sl) out[tx] = tx; else if (tx < K) out[tx] = -1; }
    return;
  }
  if (sl <= kRegMaxLen) {
    if (rank == 0) { extern __shared__ uint8_t sr[]; register_topk<12>(in, out, sl, sr); }
    return;
  }

  // Large path: init mbarriers + state, then dispatch
  auto* st = &params.states[row];
  constexpr uint32_t CS = 4;
  const uint32_t per_block = (params.stride + CS - 1) / CS;
  const bool use_singlepass = per_block <= kMaxSinglePassPerBlock;
  const uint32_t sp_stages = (per_block + kSizePerStage - 1) / kSizePerStage;

  extern __shared__ uint8_t sr[];
  if (use_singlepass) {
    auto* sp = reinterpret_cast<SmemSinglePass*>(sr);
    if (tx < sp_stages) ptx_h::mbarrier_init(&sp->barrier[tx], 1);
  } else {
    auto* s4 = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kNumStages4) ptx_h::mbarrier_init(&s4->barrier[0][0] + tx, 1);
  }
  if (rank == 0 && tx == 0) st->output_counter = 0;
  __syncthreads();

  uint32_t hp[kNumStages4]={0,0}, sp_ph[kNumStages4]={0,0};
  uint32_t spp[kMaxSinglePassStages] = {};
  if (use_singlepass) {
    large_topk_singlepass4(in, out, sl, st, spp, params.tie_ws + row * kMaxTies);
  } else {
    large_topk_twopass4(in, out, sl, st, hp, sp_ph, params.tie_ws + row * kMaxTies);
  }
}

template <> __global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 8, 1) cluster_persistent_topk_impl<8>(Params params) {
  const auto rank = blockIdx.y, row = blockIdx.x, tx = threadIdx.x;
  const auto sl = params.lengths[row];
  int32_t* out = params.output + row * K;
  const float* in = params.input + row * params.stride;

  // Register/trivial: only rank 0 works, then return
  if (sl <= K) {
    if (rank == 0) { if (tx < sl) out[tx] = tx; else if (tx < K) out[tx] = -1; }
    return;
  }
  if (sl <= kRegMaxLen) {
    if (rank == 0) { extern __shared__ uint8_t sr[]; register_topk<12>(in, out, sl, sr); }
    return;
  }

  // Large path: init mbarriers + state
  auto* st = &params.states[row];
  extern __shared__ uint8_t sr[];
  auto* s = reinterpret_cast<Smem8*>(sr);
  if (tx < kNumStages8) ptx_h::mbarrier_init(&s->barrier[tx], 1);
  if (rank == 0 && tx == 0) st->output_counter = 0;
  __syncthreads();

  uint32_t ph[kNumStages8] = {0,0};
  large_topk_fused8(in, out, sl, st, ph, params.tie_ws + row * kMaxTies);
}

constexpr size_t kSmemSize4_base = (sizeof(Smem4) > sizeof(StreamSmem) ? sizeof(Smem4) : sizeof(StreamSmem));
constexpr size_t kSmemSize4_sp = sizeof(SmemSinglePass);
constexpr size_t kSmemSize4 = (kSmemSize4_base > kSmemSize4_sp ? kSmemSize4_base : kSmemSize4_sp) + sizeof(int32_t) * K + 128;
constexpr size_t kSmemSize8 = std::max(sizeof(Smem8), sizeof(StreamSmem)) + sizeof(int32_t) * K + 128;

}  // namespace cluster_topk
