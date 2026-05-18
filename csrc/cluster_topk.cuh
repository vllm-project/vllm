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

constexpr uint32_t kElemPerStage = 8;
constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;  // 8192

// CS=4: 2 TMA stages (double buffer), two-pass
constexpr uint32_t kNumStages4 = 2;
// CS=8: 4 TMA stages, single-pass (all data stays in smem)
constexpr uint32_t kNumStages8 = 4;
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
#pragma unroll
  for (uint32_t o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xFFFFFFFF, v, o, 32);
  return v;
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
__device__ void dsmem_hist_reduce(uint32_t* histogram) {
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

__device__ void find_threshold(uint32_t* histogram, uint32_t* warp_sum,
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

__device__ void register_topk(const float* __restrict__ scores,
                               int32_t* __restrict__ output,
                               uint32_t length, void* _smem) {
  auto* smem = static_cast<RegSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Zero histogram (4 bins per thread, vectorized)
  smem->histogram[tx * kRegHistItems] = 0;
  smem->histogram[tx * kRegHistItems + 1] = 0;
  smem->histogram[tx * kRegHistItems + 2] = 0;
  smem->histogram[tx * kRegHistItems + 3] = 0;
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
  __syncthreads();

  // Load all data into registers (4 float4 per thread)
  float regs[kRegVecsPerThread * 4];
#pragma unroll
  for (uint32_t v = 0; v < kRegVecsPerThread; v++) {
    const uint32_t base = (tx + v * kBlockSize) * 4;
    if (base < length) {
      const float4 val = *reinterpret_cast<const float4*>(scores + base);
      regs[v * 4] = val.x; regs[v * 4 + 1] = val.y;
      regs[v * 4 + 2] = val.z; regs[v * 4 + 3] = val.w;
    }
  }

  // Build 4096-bin histogram from registers
#pragma unroll
  for (uint32_t v = 0; v < kRegVecsPerThread; v++) {
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) goto hist_done;
      const uint32_t bin = extract_coarse_bin_N<kRegHistBits>(regs[v * 4 + e]);
      atomicAdd(&smem->histogram[bin], 1);
    }
  }
hist_done:
  __syncthreads();

  // Find threshold via multi-element prefix scan (4 bins per thread)
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
        smem->match = {.bin = tx * kRegHistItems + i,
                       .above_count = above, .equal_count = orig[i]};
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > K);

  // Scatter from registers
#pragma unroll
  for (uint32_t v = 0; v < kRegVecsPerThread; v++) {
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) goto scatter_done;
      const uint32_t bin = extract_coarse_bin_N<kRegHistBits>(regs[v * 4 + e]);
      if (bin > thr_bin) {
        output[atomicAdd(&smem->counter_gt, 1)] = idx;
      } else if (bin == thr_bin) {
        const auto pos = atomicAdd(&smem->counter_eq, 1);
        if (!need_tie) {
          if (pos + num_above < K) output[pos + num_above] = idx;
        } else {
          if (pos < kMaxTies) smem->tie_buffer[pos] = {idx, regs[v * 4 + e]};
        }
      }
    }
  }
scatter_done:
  if (!need_tie) return;
  __syncthreads();
  tie_handle(smem->tie_buffer, min(num_equal, kMaxTies), num_above, output, smem);
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
                                   uint32_t* phases) {
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

  // Init
  if (tx < kHistBins) smem->histogram[tx] = 0;
  if (tx == 0) { smem->counter_gt = 0; smem->counter_eq = 0; }
  __syncthreads();

  // TMA prologue: load ALL stages (data stays in smem for rescan)
  if (tx == 0) {
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

  // Histogram pass: process all stages, data stays in smem
  for (uint32_t iter = 0; iter < num_iters; iter++) {
    const auto off = iter * kSizePerStage;
    const auto sz = min(kSizePerStage, my_len - off);
    const auto lane = tx % kWarpSize;
    if (lane == 0) ptx_h::mbarrier_wait(&smem->barrier[iter], phases[iter] & 1);
    phases[iter]++;
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < kElemPerStage; i++) {
      const auto li = tx + i * kBlockSize;
      if (li >= sz) break;
      atomicAdd(&smem->histogram[extract_coarse_bin(smem->score_buffer[iter][li])], 1);
    }
    __syncthreads();
  }

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

  // Cross-block output collection
  const uint32_t la = smem->counter_gt, le = smem->counter_eq;
  const auto midx = tx < la ? s_topk[tx] : 0;
  const auto mtie = tx < le ? smem->tie_buffer[tx] : Tie{0, 0.0f};

  __shared__ uint32_t s_off;
  if (tx == 0) s_off = atomicAdd(&state->output_counter, (int)la);
  __syncthreads();
  if (tx < la) row_output[s_off + tx] = midx + my_start;

  cooperative_groups::this_cluster().sync();  // 3rd sync: ensure above writes done

  __shared__ uint32_t s_toff, s_ta;
  if (tx == 0) {
    s_ta = state->output_counter;
    s_toff = atomicAdd(&state->output_counter, (int)le);
  }
  __syncthreads();
  if (s_ta >= K) return;
  if (tx < le) { uint32_t p = s_toff + tx; if (p < K) row_output[p] = mtie.idx + my_start; }

  cooperative_groups::this_cluster().sync();  // 4th sync before tie refinement
  if (rank != 0) return;

  const auto tt = state->output_counter - s_ta;
  if (tt <= K - s_ta) return;
  __syncthreads();
  for (uint32_t i = tx; i < min(tt, kMaxTies); i += kBlockSize) {
    uint32_t idx = row_output[s_ta + i];
    smem->tie_buffer[i] = {(uint32_t)idx, row_input[idx]};
  }
  __syncthreads();
  tie_handle(smem->tie_buffer, min(tt, kMaxTies), s_ta, row_output, smem);
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

__device__ void large_topk_twopass4(const float* __restrict__ ri,
                                     int32_t* __restrict__ ro,
                                     uint32_t sl, ClusterState* state,
                                     uint32_t* hp, uint32_t* sp) {
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

  // Phase 1: TMA histogram (4-stage pipeline)
  stream_pass4<false>(ri + ms, ml, 0, nullptr, hp, smem);
  __syncthreads();

  // DSMEM all-reduce + find threshold (2 cluster.sync() — the only syncs)
  dsmem_hist_reduce<CS>(smem->histogram);
  find_threshold(smem->histogram, smem->warp_sum,
                  &smem->counter_gt, &smem->counter_eq, &smem->match);

  // Phase 2: TMA scatter (4-stage pipeline)
  stream_pass4<true>(ri + ms, ml, smem->match.bin, s_topk, sp, smem);
  __syncthreads();

  // Output collection: above first, then ties
  const uint32_t la = smem->counter_gt, le = smem->counter_eq;
  const auto midx = tx < la ? s_topk[tx] : 0;
  const auto mtie = tx < le ? smem->tie_buffer[tx] : Tie{0, 0.0f};

  // Phase A: write above-threshold via global atomic
  __shared__ uint32_t s_off;
  if (tx == 0) s_off = atomicAdd(&state->output_counter, (int)la);
  __syncthreads();
  if (tx < la) ro[s_off + tx] = midx + ms;

  // cluster.sync() to ensure all above writes visible + get total above
  cooperative_groups::this_cluster().sync();

  // Phase B: write ties
  __shared__ uint32_t s_toff, s_ta;
  if (tx == 0) {
    s_ta = state->output_counter;
    s_toff = atomicAdd(&state->output_counter, (int)le);
  }
  __syncthreads();
  if (s_ta >= K) return;
  if (tx < le) { uint32_t p = s_toff + tx; if (p < K) ro[p] = mtie.idx + ms; }

  // cluster.sync() before tie refinement
  cooperative_groups::this_cluster().sync();
  if (rank != 0) return;

  const auto tt = state->output_counter - s_ta;
  if (tt <= K - s_ta) return;
  __syncthreads();
  for (uint32_t i = tx; i < min(tt, kMaxTies); i += kBlockSize) {
    uint32_t idx = ro[s_ta + i]; smem->tie_buffer[i] = {(uint32_t)idx, ri[idx]};
  }
  __syncthreads();
  tie_handle(smem->tie_buffer, min(tt, kMaxTies), s_ta, ro, smem);
}

// ============================================================================
// Kernel variants
// ============================================================================

template <uint32_t CS> __global__ void __launch_bounds__(kBlockSize, 1)
    cluster_persistent_topk_impl(Params params);

template <> __global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 4, 1) cluster_persistent_topk_impl<4>(Params params) {
  const auto rank = blockIdx.y, cid = blockIdx.x, nc = gridDim.x, tx = threadIdx.x;
  auto* st = &params.states[cid];
  { extern __shared__ uint8_t sr[];
    // Init mbarriers for cluster TMA path
    auto* s4 = reinterpret_cast<Smem4*>(sr);
    if (tx < 2 * kNumStages4) ptx_h::mbarrier_init(&s4->barrier[0][0] + tx, 1);
    // Init mbarriers for streaming path (only block 0 uses them)
    if (rank == 0) {
      auto* ss = reinterpret_cast<StreamSmem*>(sr);
      if (tx < 2 * kStreamNumStages) ptx_h::mbarrier_init(&ss->barrier[0][0] + tx, 1);
    }
    if (rank == 0 && tx == 0) st->output_counter = 0;
  }
  __syncthreads();
  uint32_t hp[kNumStages4]={0,0}, sp[kNumStages4]={0,0};
  uint32_t shp[kStreamNumStages]={0,0}, ssp[kStreamNumStages]={0,0};
  for (uint32_t r = cid; r < params.num_rows; r += nc) {
    auto sl = params.lengths[r];
    if (sl <= K) { if (rank == 0) { if (tx < sl) params.output[r*K+tx] = tx;
        else if (tx < K) params.output[r*K+tx] = -1; }
    } else if (sl <= kRegMaxLen) {
      if (rank == 0) {
        extern __shared__ uint8_t sr2[];
        register_topk(params.input + r * params.stride,
                       params.output + r * K, sl, sr2);
      }
    } else if (sl <= kStreamMaxLen) {
      if (rank == 0) {
        extern __shared__ uint8_t sr2[];
        streaming_topk(params.input + r * params.stride,
                        params.output + r * K, sl, sr2, shp, ssp);
      }
    } else { large_topk_twopass4(params.input + r * params.stride,
                                  params.output + r * K, sl, st, hp, sp); }
    if (rank == 0 && tx == 0) st->output_counter = 0;
    cooperative_groups::this_cluster().sync();
  }
}

template <> __global__ void __launch_bounds__(kBlockSize, 1)
    __cluster_dims__(1, 8, 1) cluster_persistent_topk_impl<8>(Params params) {
  const auto rank = blockIdx.y, cid = blockIdx.x, nc = gridDim.x, tx = threadIdx.x;
  auto* st = &params.states[cid];
  { extern __shared__ uint8_t sr[];
    auto* s = reinterpret_cast<Smem8*>(sr);
    if (tx < kNumStages8) ptx_h::mbarrier_init(&s->barrier[tx], 1);
    if (rank == 0) {
      auto* ss = reinterpret_cast<StreamSmem*>(sr);
      if (tx < 2 * kStreamNumStages) ptx_h::mbarrier_init(&ss->barrier[0][0] + tx, 1);
    }
    if (rank == 0 && tx == 0) st->output_counter = 0;
  }
  __syncthreads();
  uint32_t ph[kNumStages8] = {0,0,0,0};
  uint32_t shp[kStreamNumStages]={0,0}, ssp[kStreamNumStages]={0,0};
  for (uint32_t r = cid; r < params.num_rows; r += nc) {
    auto sl = params.lengths[r];
    if (sl <= K) { if (rank == 0) { if (tx < sl) params.output[r*K+tx] = tx;
        else if (tx < K) params.output[r*K+tx] = -1; }
    } else if (sl <= kRegMaxLen) {
      if (rank == 0) {
        extern __shared__ uint8_t sr2[];
        register_topk(params.input + r * params.stride,
                       params.output + r * K, sl, sr2);
      }
    } else if (sl <= kStreamMaxLen) {
      if (rank == 0) {
        extern __shared__ uint8_t sr2[];
        streaming_topk(params.input + r * params.stride,
                        params.output + r * K, sl, sr2, shp, ssp);
      }
    } else { large_topk_fused8(params.input + r * params.stride,
                                params.output + r * K, sl, st, ph); }
    if (rank == 0 && tx == 0) st->output_counter = 0;
    cooperative_groups::this_cluster().sync();
  }
}

constexpr size_t kSmemSize4 = (sizeof(Smem4) > sizeof(StreamSmem) ? sizeof(Smem4) : sizeof(StreamSmem)) + sizeof(int32_t) * K + 128;
constexpr size_t kSmemSize8 = std::max(sizeof(Smem8), sizeof(StreamSmem)) + sizeof(int32_t) * K + 128;

}  // namespace cluster_topk
