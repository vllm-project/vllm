// SPDX-License-Identifier: MIT
// Persistent fused FP8 MQA logits + per-row top-2048 (SM100),
// DeepGEMM/FA4-style: one CTA per SM, barriers and TMEM initialised once, every
// warp role runs its own loop over blocks claimed (heaviest first) from a
// global counter, and the sample and scan tiles of a block flow through ONE
// continuous Q/KV/TMEM pipeline whose stage/phase state is carried across
// blocks.
//
// Roles (384 threads):
//   warp 8   claimer + Q/weights TMA (+ drain of math WG1 rows, > 16K keys)
//   warp 9   KV TMA (sample tiles via the strided map, then the scan tiles)
//   warp 10  UMMA issuer
//   warp 11  drain of math WG0 rows (> 16K keys), idle otherwise
//   warps 0-7 math: sample histogram -> gate -> scan/emit -> selection
// A block whose candidate count leaves [2048, 16384] pauses the role loops; the
// CTA rescans it exactly (fallback.cuh) and resumes the loops with freshly
// initialised barriers.
#pragma once
#include <algorithm>
#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "candidates.cuh"
#include "deep_gemm_compat.cuh"
#include "fallback.cuh"
#include "fast_select.cuh"
#include "row_select.cuh"
#include "sample.cuh"

namespace fused_persist {
using namespace deep_gemm;
using TBar = cutlass::arch::ClusterTransactionBarrier;

constexpr unsigned kTerm = 0xffffffffu, kPause = 0xfffffffeu;
// Selection scratch starts after Q + weights so the next block's Q can land
// during selection.
constexpr unsigned kSelBase = 33792;
constexpr unsigned kScratchS =
    8192;  // per-warp selection scratch (<= 16K keys)
constexpr unsigned kCandCap = 3072;
constexpr unsigned kCandExt = 8u * kCandCap + 256u;
constexpr unsigned kQBytes = 8 * 32 * 128, kKVBytes = 256 * 128,
                   kWBytes = 8 * 32 * 4;

template <unsigned KVS>
struct PStorage {
  alignas(1024) cutlass::float_e4m3_t q[kQBytes];
  alignas(128) float weights[8 * 32];
  alignas(1024) cutlass::float_e4m3_t kv[KVS][kKVBytes];
  // <= 16K: the per-key scales have their own ring, twice as deep as the KV
  // stages (see BarIdx::sf_full).
  alignas(128) float sf_kv[2 * KVS][256];
};
// > 16K keys (rebalanced scan): the 16 linear-term rows of each math
// warpgroup bracket Q, so WG0's N=144 UMMA reads B = [lin WG0 | Q rows 0..127]
// (D = [lin | heads]) and WG1's reads B = [Q rows 128..255 | lin WG1] (D =
// [heads | lin]).
template <>
struct PStorage<3> {
  alignas(1024) uint8_t lin_lo[16 * 128];
  cutlass::float_e4m3_t q[kQBytes];
  uint8_t lin_hi[16 * 128];
  alignas(128) float weights[8 * 32];
  float lin_scale[8];
  alignas(1024) cutlass::float_e4m3_t kv[3][kKVBytes];
  alignas(128) float sf_kv[3][256];
};
static_assert(offsetof(PStorage<3>, q) == 16 * 128 &&
                  offsetof(PStorage<3>, lin_hi) == 16 * 128 + kQBytes,
              "lin rows must bracket Q");
static_assert(offsetof(PStorage<2>, kv) == kSelBase,
              "selection scratch base (<= 16K)");
// > 16K: selection scratch starts after lin + Q + lin + weights + row scales
// (the next block's linear term is written during this block's selection).
constexpr unsigned kSelBase3 = offsetof(PStorage<3>, kv);

// q'_r = sum_h (w_rh / 2) q_rh, split into 4 FP8 e4m3 terms: terms 0,1 scale
// s_r, terms 2,3 s_r * 2^-8 (lin_scale[r]); term t of row r is B row (r % 4) *
// 4 + t of lin_lo (rows 0-3) / lin_hi (4-7), 128B-swizzled K-major like Q. One
// warp; lane l owns dims 4l..4l+3.
__device__ __forceinline__ void compute_lin_rows(PStorage<3>& sm,
                                                 unsigned lane) {
  const uint8_t* qbytes = reinterpret_cast<const uint8_t*>(sm.q);
#pragma unroll 1
  for (unsigned r = 0; r < 8; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (unsigned h = 0; h < 32; ++h) {
      const unsigned R = r * 32 + h;
      const unsigned off = R * 128u + ((((lane * 4u) >> 4) ^ (R & 7u)) << 4) +
                           ((lane * 4u) & 15u);
      const uint32_t packed =
          ptx::ld_shared(reinterpret_cast<const uint32_t*>(qbytes + off));
      const float wv = ptx::ld_shared(sm.weights + r * 32 + h) * 0.5f;
      const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
      const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
      acc[0] = fmaf(wv, f0.x, acc[0]);
      acc[1] = fmaf(wv, f0.y, acc[1]);
      acc[2] = fmaf(wv, f1.x, acc[2]);
      acc[3] = fmaf(wv, f1.y, acc[3]);
    }
    float m = fmaxf(fmaxf(fabsf(acc[0]), fabsf(acc[1])),
                    fmaxf(fabsf(acc[2]), fabsf(acc[3])));
#pragma unroll
    for (unsigned o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    int e = -120;  // pow2 scale s >= m / 448, clamped so 1/s stays finite
    if (m > 0.f && isfinite(m)) {
      frexpf(m * (1.0f / 448.0f), &e);
      e = max(e, -120);
    }
    const float s = ldexpf(1.0f, e), inv_s = ldexpf(1.0f, -e);
    float res[4];
#pragma unroll
    for (unsigned dd = 0; dd < 4; ++dd) res[dd] = acc[dd] * inv_s;
    uint8_t* lin_base = r < 4 ? sm.lin_lo : sm.lin_hi;
#pragma unroll
    for (unsigned tt = 0; tt < 4; ++tt) {
      const __nv_fp8x2_storage_t c01 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[0], res[1]), __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t c23 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[2], res[3]), __NV_SATFINITE, __NV_E4M3);
      const __half2_raw d01 = __nv_cvt_fp8x2_to_halfraw2(c01, __NV_E4M3);
      const __half2_raw d23 = __nv_cvt_fp8x2_to_halfraw2(c23, __NV_E4M3);
      const float2 g01 =
          __half22float2(*reinterpret_cast<const __half2*>(&d01));
      const float2 g23 =
          __half22float2(*reinterpret_cast<const __half2*>(&d23));
      res[0] -= g01.x;
      res[1] -= g01.y;
      res[2] -= g23.x;
      res[3] -= g23.y;
      // Terms 2, 3 in units of s * 2^-8: the residual after two terms is <=
      // 0.5, so x256 <= 128 (no saturation), and the e4m3 subnormal floor drops
      // from s * 2^-10 to s * 2^-18 per dim.
      if (tt == 1) {
        res[0] *= 256.0f;
        res[1] *= 256.0f;
        res[2] *= 256.0f;
        res[3] *= 256.0f;
      }
      const unsigned n = (r % 4u) * 4u + tt;
      const unsigned off = n * 128u + ((((lane * 4u) >> 4) ^ (n & 7u)) << 4) +
                           ((lane * 4u) & 15u);
      ptx::st_shared(
          reinterpret_cast<uint32_t*>(lin_base + off),
          static_cast<uint32_t>(c01) | (static_cast<uint32_t>(c23) << 16));
    }
    if (lane == 0) sm.lin_scale[r] = s;
  }
}

// > 16K gate: the k-th largest sample (k = count * target / visible) of one row
// from its 2048-bin histogram; one warp per row. Same result as finish_sample
// (count = histogram total).
__device__ __noinline__ void finish_sample_histogram(const unsigned* hist,
                                                     float* gate,
                                                     unsigned visible,
                                                     unsigned target,
                                                     unsigned lane) {
  if (visible <= target) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  // Lane l owns the 64 bins of chunks 63-2l and 62-2l (bins descending with the
  // lane).
  const unsigned* const h0 = hist + (63 - 2 * lane) * 33;
  const unsigned* const h1 = hist + (62 - 2 * lane) * 33;
  unsigned sum = 0;
#pragma unroll 8
  for (unsigned j = 0; j < 32; ++j) sum += h0[j] + h1[j];
  unsigned incl = sum;
#pragma unroll
  for (unsigned dd = 1; dd < 32; dd *= 2) {
    const unsigned u = __shfl_up_sync(0xffffffffu, incl, dd);
    if (lane >= dd) incl += u;
  }
  const unsigned tot = __shfl_sync(0xffffffffu, incl, 31);
  if (tot == 0) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  const unsigned rank =
      max(1u, min(tot, (tot * target + visible - 1) / visible));
  const unsigned w = __ffs(__ballot_sync(0xffffffffu, incl >= rank)) - 1;
  unsigned above = __shfl_sync(0xffffffffu, incl - sum, w), chosen = 0;
#pragma unroll 1
  for (unsigned cc = 0; cc < 2; ++cc) {
    const unsigned c = 63 - 2 * w - cc;
    unsigned v = hist[c * 33 + 31 - lane];  // lane 0 = highest bin of the chunk
#pragma unroll
    for (unsigned dd = 1; dd < 32; dd *= 2) {
      const unsigned u = __shfl_up_sync(0xffffffffu, v, dd);
      if (lane >= dd) v += u;
    }
    const unsigned mask = __ballot_sync(0xffffffffu, above + v >= rank);
    if (mask) {
      chosen = c * 32 + 31 - (__ffs(mask) - 1);
      break;
    }
    above += __shfl_sync(0xffffffffu, v, 31);
  }
  if (lane == 0) *gate = sample_floor(chosen);
  __syncwarp();
}

struct alignas(16) PHandoffData {
  uint32_t records[2][8][4][8][32];
  uint32_t counts[2][8][4][32];
  uint32_t bases[2][8];
};

struct ClaimEntry {
  unsigned vblock, all_short, kv_base, splits, slot;
  unsigned starts[8], ends[8];
};

// Barrier words (static shared, never aliased with data).
template <unsigned KVS>
struct BarIdx {
  // > 16K (KVS == 3): 3 per-warpgroup TMEM slots + the linear-term barrier.
  static constexpr unsigned NT = KVS == 3 ? 3 : 2;
  static constexpr unsigned full_q = 0, empty_q = 1, full_kv = 2,
                            empty_kv = 2 + KVS, full_tmem = 2 + 2 * KVS,
                            empty_tmem = full_tmem + NT,
                            claim_full = empty_tmem + NT,
                            claim_empty = claim_full + 2,
                            sel_done = claim_empty + 2, h_full = sel_done + 1,
                            h_empty = h_full + 16, full_lin = h_empty + 16,
                            sf_full = full_lin + (KVS == 3 ? 1 : 0),
                            total = sf_full + (KVS == 3 ? 0 : 2 * KVS);
};

__device__ __forceinline__ void mbar_inval(const void* b) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(b)))
               : "memory");
}

// Expect `bytes` of TMA transactions and arrive `count` times on one barrier.
__device__ __forceinline__ void mbar_arrive_expect_tx(const void* b,
                                                      uint32_t bytes,
                                                      uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(a),
      "r"(bytes)
      : "memory");
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a),
      "r"(count)
      : "memory");
}

// > 16K self-drain, out of line: stores this warp's ring records of the last
// handoff interval. recs = records[bank][warp] ([4][8][32]).
__device__ __noinline__ void self_drain_last(
    const uint32_t* __restrict__ recs, unsigned n0, unsigned n1, unsigned n2,
    unsigned n3, int* __restrict__ counts, unsigned rbase, uint32_t lane_kv,
    unsigned flush_base, int64_t row_delta, uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices) {
  constexpr uint32_t kCap = 16384;
  const unsigned lane = threadIdx.x & 31;
  const unsigned n[4] = {n0, n1, n2, n3};
  unsigned my_total = 0;
#pragma unroll
  for (unsigned row = 0; row < 4; ++row) {
    const unsigned total = __reduce_add_sync(0xffffffffu, n[row]);
    if (lane == row) my_total = total;
  }
  unsigned my_start = 0;
  if (lane < 4 && my_total)
    my_start = atomicAdd(counts + rbase + lane, my_total);
  unsigned p01 = n[0] | (n[1] << 16), p23 = n[2] | (n[3] << 16);
#pragma unroll
  for (int dd = 1; dd < 32; dd *= 2) {
    const unsigned a = __shfl_up_sync(0xffffffffu, p01, dd),
                   b = __shfl_up_sync(0xffffffffu, p23, dd);
    if (lane >= dd) {
      p01 += a;
      p23 += b;
    }
  }
  const unsigned prefix[4] = {p01 & 65535u, p01 >> 16, p23 & 65535u, p23 >> 16};
#pragma unroll
  for (unsigned row = 0; row < 4; ++row) {
    const unsigned start = __shfl_sync(0xffffffffu, my_start, row);
    for (unsigned j = 0; j < n[row]; ++j) {
      const unsigned slot2 = start + prefix[row] - n[row] + j;
      const unsigned record = recs[(row * 8 + j) * 32 + lane];
      if (slot2 < kCap) {
        const uint32_t physical =
            lane_kv + (flush_base * 2 + (record & 255u)) * 128;
        const uint64_t off =
            static_cast<uint64_t>(int64_t(rbase + row) + row_delta) * kCap +
            slot2;
        fused_candidates::store_candidate_record(vals + off, indices + off,
                                                 record >> 8, physical);
      }
    }
  }
  __syncwarp();
}

template <class Store>
__device__ __forceinline__ void ho_drain(PHandoffData& d, TBar* hfull,
                                         TBar* hempty, int producer,
                                         unsigned epoch, unsigned row_base,
                                         int* totals, unsigned cap, bool skip,
                                         Store store) {
  const int lane = threadIdx.x & 31, bank = epoch & 1;
  hfull[bank * 8 + producer].wait((epoch >> 1) & 1);
  __syncwarp();
  if (skip) {
    hempty[bank * 8 + producer].arrive();
    return;
  }  // self-drained by the producer
  const unsigned base = d.bases[bank][producer];
  unsigned n[4];
  unsigned my_total = 0;
#pragma unroll
  for (unsigned row = 0; row < 4; ++row) {
    n[row] = d.counts[bank][producer][row][lane];
    unsigned total = __reduce_add_sync(0xffffffffu, n[row]);
    if (lane == row) my_total = total;
  }
  unsigned my_start = 0;
  if (lane < 4 && my_total)
    my_start = atomicAdd(totals + row_base + lane, my_total);
  unsigned p01 = n[0] | (n[1] << 16), p23 = n[2] | (n[3] << 16);
#pragma unroll
  for (int dd = 1; dd < 32; dd *= 2) {
    unsigned a = __shfl_up_sync(0xffffffffu, p01, dd);
    unsigned b = __shfl_up_sync(0xffffffffu, p23, dd);
    if (lane >= dd) {
      p01 += a;
      p23 += b;
    }
  }
  unsigned prefix[4] = {p01 & 65535u, p01 >> 16, p23 & 65535u, p23 >> 16};
#pragma unroll
  for (unsigned row = 0; row < 4; ++row) {
    unsigned start = __shfl_sync(0xffffffffu, my_start, row);
    for (unsigned j = 0; j < n[row]; ++j) {
      unsigned slot = start + prefix[row] - n[row] + j;
      unsigned record = d.records[bank][producer][row][j][lane];
      if (slot < cap)
        store(row_base + row, slot, record, base, producer * 32 + lane);
    }
  }
  __syncwarp();
  hempty[bank * 8 + producer].arrive();
}

template <unsigned Interval, unsigned KVS>
__global__ __launch_bounds__(384, 1) void fused_mqa_topk_fp8_prefill_kernel(
    unsigned rows, unsigned keys, const unsigned* starts_g,
    const unsigned* ends_g, uint16_t* vals, int* indices, int* counts,
    FallbackBuffers fallback, unsigned phase_bytes, unsigned sample_target,
    unsigned sample_stride, unsigned sample_groups, unsigned sample_tiles,
    const float* scales, const __grid_constant__ cute::TmaDescriptor tks,
    const __grid_constant__ cute::TmaDescriptor tq,
    const __grid_constant__ cute::TmaDescriptor ts,
    const __grid_constant__ cute::TmaDescriptor tk,
    const __grid_constant__ cute::TmaDescriptor tw, bool relative,
    unsigned nblocks, unsigned* counter, int* defer_next) {
  constexpr bool kSmem =
      Interval <= 8;  // <= 16K keys: shared candidates, direct emission
  constexpr uint32_t kMathRegs = kSmem ? 232 : 224,
                     kSpecRegs = 168 * 3 - 2 * kMathRegs;
  constexpr uint32_t kCandGlobalCap = 16384;
  using St = PStorage<KVS>;
  using B = BarIdx<KVS>;
  // > 16K keys: rebalanced scan (linear term on the tensor core, per-warpgroup
  // N=144 MMAs).
  constexpr bool kLin = !kSmem;
  static_assert(kLin == (KVS == 3),
                "rebalanced scan needs the 3-KV-stage lin layout");
  constexpr uint32_t kSlotCols =
      160;  // 4 rows x 32 heads + 16 linear-term columns
  constexpr unsigned kSelB = KVS == 3 ? kSelBase3 : kSelBase;
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& sm = *reinterpret_cast<St*>(smem_buffer);
  uint8_t* const region = smem_buffer + sizeof(St);
  auto& stream = *reinterpret_cast<SampleShared*>(region);
  auto& hdata = *reinterpret_cast<PHandoffData*>(region);

  __shared__ alignas(8) uint64_t bar_words[B::total];
  __shared__ uint32_t tmem_ptr;
  __shared__ ClaimEntry entries[2];
  __shared__ float gate_s[8];
  __shared__ unsigned row_count[8], pool_cnt[8];
  __shared__ unsigned sel_recover, sel_staged, sel_short, row_offset[9];
  __shared__ unsigned defer_head, unsafe_s, pause_req, epoch_kind;

  TBar* const bars = reinterpret_cast<TBar*>(bar_words);
  TBar& full_q = bars[B::full_q];
  TBar& empty_q = bars[B::empty_q];
  TBar* const full_kv = bars + B::full_kv;
  // > 16K the 32 producer lanes stage the sample scales, so each arrives on
  // full_kv.
  constexpr unsigned kFullKvArrivals = KVS == 3 ? 32 : 1;
  // <= 16K: the MMA warp frees each KV stage once its MMAs complete, and the
  // per-key scales arrive through a separate ring (2 * KVS deep, sf_full), so
  // the math warps never wait on or release KV stages. The producer leads the
  // MMA by <= KVS tiles and the MMA leads the math warps by <= 1 tile (2 TMEM
  // stages), so a scale stage is never overwritten while being read.
  constexpr bool kRel = KVS == 2;
  TBar* const sf_full = bars + B::sf_full;
  TBar* const empty_kv = bars + B::empty_kv;
  TBar* const full_tmem = bars + B::full_tmem;
  TBar* const empty_tmem = bars + B::empty_tmem;
  TBar* const claim_full = bars + B::claim_full;
  TBar* const claim_empty = bars + B::claim_empty;
  TBar& sel_done = bars[B::sel_done];
  TBar* const h_full = bars + B::h_full;
  TBar* const h_empty = bars + B::h_empty;
  TBar* const full_lin = bars + B::full_lin;  // > 16K only

  const unsigned warp_idx = cutlass::canonical_warp_idx_sync();
  const unsigned lane = ptx::get_lane_idx();
  constexpr unsigned kClaimConsumers =
      kSmem ? 10 : 11;  // warps 9, 10, 0-7 (+11)

  if (warp_idx == 8 && lane == 0) {
    cute::prefetch_tma_descriptor(&tq);
    cute::prefetch_tma_descriptor(&tw);
    cute::prefetch_tma_descriptor(&tk);
    cute::prefetch_tma_descriptor(&ts);
    cute::prefetch_tma_descriptor(&tks);
  }
  if (warp_idx == 9 && cute::elect_one_sync()) {
    defer_head = kTerm;
    pause_req = 0;
  }
  // Epochs: the persistent role loops run until the work runs out (kTerm) or a
  // block needs the exact rescan (kPause); rare blocks are then rescanned with
  // the fallback core and the loops resume with freshly initialised barriers.
  // The first epoch is emitted outside any loop (nesting the role loops in an
  // epoch loop costs ~1.5 percent in the >16K scan: uniform address math is
  // rematerialised every tile). Later epochs only run after a recovery pause.
  __shared__ unsigned row_bounds[2][8];
  __shared__ unsigned fslot;
  bool more = false;
  {
    constexpr unsigned epoch = 0;
#include "persistent_epoch.inc"
  }
#pragma unroll 1
  for (unsigned epoch = 1; more; ++epoch) {
#include "persistent_epoch.inc"
  }
  // counter = {next block, finished CTAs}: the last CTA to finish (every claim
  // is done by then) zeroes both, so the caller's workspace is ready for the
  // next launch / graph replay.
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(counter + 1, 1u) == gridDim.x - 1) {
      atomicExch(counter, 0u);
      atomicExch(counter + 1, 0u);
    }
  }
}

// ---------------- host: launch configuration (> kResidentKeys keys)
// ----------------
using PersistentKernel = decltype(&fused_mqa_topk_fp8_prefill_kernel<4, 2>);

struct LaunchConfig {
  PersistentKernel kernel;
  int smem;              // dynamic shared memory bytes
  unsigned phase_bytes;  // selection scratch bytes past the Q / weights region
  // Gate sample: sample_tiles tiles of 256 keys on a uniform grid (stride
  // sample_stride).
  unsigned sample_target, sample_stride, sample_groups, sample_tiles;
};

// Picks the kernel instantiation, sizes its shared memory (setting the opt-in
// limit) and the gate sample for `keys` keys.
inline cudaError_t launch_config(unsigned keys, LaunchConfig& c) {
  // <= 16K keys: shared candidates (cap kCandCap), so aim lower: 2560
  // candidates. Up to 128K: 3072. Both size the sample so the gate's quantile
  // rank is ~100 (about 10% count error). Longer contexts aim for 4096 with
  // rank 64, clamped to 16-32 sample tiles (rank ~32 at 1M keys): a sample tile
  // costs about two scan tiles, so a larger sample costs more than the smaller
  // candidate set saves.
  const bool small = keys <= 16384, long_ctx = keys > 131072;
  c.sample_target = small ? 2560u : (long_ctx ? 4096u : 3072u);
  const unsigned rank = long_ctx ? 64u : 100u;
  const unsigned rank_tiles = std::max(
      2u, unsigned((uint64_t(rank) * keys + 256ull * c.sample_target - 1) /
                   (256ull * c.sample_target)));
  c.sample_tiles =
      long_ctx ? std::min(32u, std::max(16u, rank_tiles)) : rank_tiles;
  c.sample_stride = std::max(
      1u, (keys + 256u * c.sample_tiles - 1) / (256u * c.sample_tiles));
  c.sample_groups =
      (keys - c.sample_stride / 2 + c.sample_stride - 1) / c.sample_stride;
  c.kernel = keys <= 8192     ? fused_mqa_topk_fp8_prefill_kernel<4, 2>
             : small          ? fused_mqa_topk_fp8_prefill_kernel<8, 2>
             : keys <= 32768  ? fused_mqa_topk_fp8_prefill_kernel<16, 3>
             : keys <= 131072 ? fused_mqa_topk_fp8_prefill_kernel<32, 3>
             : keys <= 524288 ? fused_mqa_topk_fp8_prefill_kernel<64, 3>
                              : fused_mqa_topk_fp8_prefill_kernel<128, 3>;

  // The recovery rescan reuses the same dynamic shared memory with the fallback
  // core's layout.
  constexpr int kFallbackSmem =
      int(sizeof(fused_mqa_dense::SharedStorage) + sizeof(FallbackShared));
  constexpr int kSmallSmem = int(sizeof(PStorage<2>) + kCandExt * 5u);
  constexpr int kLongSmem =
      int(sizeof(PStorage<3>) +
          std::max(sizeof(PHandoffData), sizeof(SampleShared)));
  static_assert(sizeof(SampleShared) <= kCandExt * 5u,
                "sample state must fit the candidate region");
  constexpr int kMaxSmem = 232448;  // sm_100 opt-in limit per block
  if (small) {
    c.smem = std::max(kSmallSmem, kFallbackSmem);
  } else {
    // > 16K: the staged selection may use everything past Q + weights, up to
    // the opt-in limit.
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(
            &attr, reinterpret_cast<const void*>(c.kernel));
        e != cudaSuccess)
      return e;
    c.smem = std::max(kFallbackSmem, kMaxSmem - int(attr.sharedSizeBytes));
    if (c.smem < kLongSmem) return cudaErrorInvalidConfiguration;
  }
  c.phase_bytes = unsigned(c.smem) - (small ? kSelBase : kSelBase3);
  return cudaFuncSetAttribute(reinterpret_cast<const void*>(c.kernel),
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              c.smem);
}

}  // namespace fused_persist
