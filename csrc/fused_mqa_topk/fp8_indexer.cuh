// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from SGLang PR #47469
// (https://github.com/sgl-project/sglang/pull/47469, Apache-2.0) at
// 599923b6f48d638d2256abbd0550c8fde141f4f5:
// python/sglang/kernels/jit/csrc/dsa/litetopk_prefill/fp8_indexer.cuh.
// GPU recovery (fp8_recovery_kernel, needs_recovery, the kRecovery paths) added
// on top. The tensor-core scoring pipeline follows DeepGEMM's SM100 FP8 MQA
// logits kernel (https://github.com/deepseek-ai/DeepGEMM, MIT).
//
// FP8 DSA prefill indexer with sampled-threshold candidate filtering (SM100).
//
// One CTA scores 256 / heads query rows (8 at 32 heads, 4 at 64) against every
// key of their ragged ranges on the tensor cores, DeepGEMM `sm100_mqa_logits`
// style: KV tile (256 keys) = MMA A, Q x heads = MMA B, accumulator in TMEM,
// each math thread owns one key of a 128-key half and reduces the heads in
// registers. Before the scan, a strided sample of single keys is scored the
// same way and per row the j'-th largest sample becomes the threshold T, found
// exactly on fp16 codes of each sample's distance to a per-row reference (the
// row's largest sample in its first sample tile), so a common offset of the
// scores does not coarsen it; the scan then appends every (score, key) with
// score >= T to the row's global slab. No top-k here: the slab, the counts and
// T are the output.
//
// Layout (vLLM #60618): each math warpgroup owns half of the rows (128 row x
// head columns) and scans both halves of every tile, so one KV tile serves all
// the CTA's rows.
// Linear-term trick (vLLM #60618): relu(s) w = (|s| w + s w) / 2; the s w half
// comes from the tensor core as 16 extra B rows per warpgroup (N = 144), so
// the scores are not bitwise DeepGEMM's.
//
// Roles (384 threads): warps 0-7 math (two warpgroups), warp 8 TMA producer,
// warp 9 UMMA issuer, warps 10-11 candidate drains (warp 10 also allocates
// TMEM, warp 11 writes the linear-term rows the math warps did not).
#pragma once

#include "device_utils.cuh"
#include "exact_topk.cuh"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace fused_mqa_topk {

namespace device::ptx {

FMT_DEVICE void st_release_gpu(uint32_t* addr, uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(addr), "r"(value)
               : "memory");
}

}  // namespace device::ptx

// The tensor maps are a separate `__grid_constant__` kernel argument: inside
// the (address-taken) parameter struct they make ptxas read every scalar field
// as non-uniform, which moves the block loop and the pipeline rings of the math
// warps out of the uniform registers (+10 % instructions in the scan loop).
struct Fp8IndexerTensorMaps {
  CUtensorMap q;  // q as [rows * heads, 128] u8, 128 B swizzle, kBlockQ * heads
                  // rows per box
  CUtensorMap weights;  // weights as [rows, heads] fp32, kBlockQ rows per box
  CUtensorMap kv;       // kv as [keys, 128] u8, 128 B swizzle, 256 rows per box
  CUtensorMap sf_kv;    // kv_scale as [1, keys] fp32, 256 keys per box
  // threshold samples: kv as [keys / S, S, 128] u8 (key = row * S + phase), 256
  // rows of one phase per box, one map per odd sample stride S (see
  // sample_stride)
  CUtensorMap kv_strided[2];
};

struct Fp8IndexerParams;
// fp8_indexer.cu: the candidate producer or, with `recovery`, the recovery
// kernel for 32 or 64 heads.
using Fp8Kernel = void (*)(Fp8IndexerParams, Fp8IndexerTensorMaps);
Fp8Kernel fp8_kernel(int64_t heads, bool recovery);

struct Fp8IndexerParams {
  const int32_t* __restrict__ ks;      // [rows] first visible key
  const int32_t* __restrict__ ke;      // [rows] one past the last visible key
  const float* __restrict__ kv_scale;  // [keys]
  uint32_t* __restrict__ block_ready;  // [num_blocks] 1 once the block's
                                       // thresholds are out (0 between
                                       // launches)
  uint32_t* __restrict__ block_done;   // [num_blocks] scan tiles of the block
                                       // done (0 between launches)
  float* __restrict__ cand_scores;     // [rows, cap]
  int32_t* __restrict__ cand_indices;  // [rows, cap], key - ks[row]
  int32_t* __restrict__ cand_count;  // [rows] number of scores >= T (may exceed
                                     // cap)
  float* __restrict__ thresholds;    // [rows] T
  int32_t* __restrict__ overflow;    // [rows] 1 when count > cap
  uint32_t num_rows;
  uint32_t num_blocks;  // ceil(rows / rows per CTA)
  uint32_t num_keys;
  uint32_t cap;
  uint32_t topk;
  uint32_t dense_bound;  // rows with at most this many keys take T = -inf
  uint32_t tail_chunks;  // pieces per block of the last, partial wave (1: whole
                         // blocks)
  uint2* __restrict__ sample_codes;  // [grid][Impl::kCodesPerCta] a block's
                                     // sample codes between the two passes
  uint32_t sample_limit[2];  // keys < limit[m] = (num_keys / S_m) * S_m are
                             // reachable through kv_strided[m]
  float* recovery_scores;
  int32_t* recovery_out;
  int64_t recovery_stride;
  uint32_t recovery_block;
  uint32_t recovery_slot, recovery_slots;
  const float* recovery_weights;
};

namespace device::fp8_indexer {

// Order-preserving 16-bit code of an fp16 bit pattern (ascending with the
// value).
FMT_DEVICE uint32_t f16_code(uint32_t h) {
  return (h & 0x8000u) ? (~h & 0xffffu) : (h | 0x8000u);
}
FMT_DEVICE float f16_code_value(uint32_t code) {
  const uint32_t h = (code & 0x8000u) ? (code ^ 0x8000u) : (~code & 0xffffu);
  return __half2float(__ushort_as_half(static_cast<uint16_t>(h)));
}
// Code of a sample score: its distance to the row's reference, rounded down
// (so ref + the code's value <= score). Never 0, which marks "not a sample".
FMT_DEVICE uint32_t sample_code(float score, float ref) {
  return f16_code(__half_as_ushort(__float2half_rd(__fsub_rd(score, ref))));
}
// Threshold of a selected code: ref + its distance, rounded down; -inf for the
// code of -inf (a distance below -65504).
FMT_DEVICE float code_threshold(float ref, uint32_t code) {
  return code < 0x400u ? -INFINITY : __fadd_rd(ref, f16_code_value(code));
}

// `values[i]` through a select chain: a run-time index into a register array
// would put it on the stack.
template <typename T, uint32_t N>
FMT_DEVICE T select_at(const T (&values)[N], uint32_t i) {
  T v = values[0];
#pragma unroll
  for (uint32_t j = 1; j < N; ++j) {
    if (i == j) v = values[j];
  }
  return v;
}

// Byte offset of 16 B chunk `chunk` of row `row` in a 128 B-swizzled K-major
// tile.
FMT_DEVICE uint32_t sw128_off(uint32_t row, uint32_t chunk) {
  return row * 128u + ((chunk ^ (row & 7u)) << 4);
}

// 256 bins of 16 bits per row, two bins per word: bin b -> word b >> 1, half b
// & 1.
FMT_DEVICE uint32_t hist_get(const uint32_t* h, uint32_t bin) {
  return (h[bin >> 1] >> ((bin & 1u) * 16)) & 0xffffu;
}
FMT_DEVICE void hist_add(uint32_t* h, uint32_t bin) {
  atomicAdd(h + (bin >> 1), 1u << ((bin & 1u) * 16));
}

// Warp-collective walk of a 256-bin histogram from the top: the bin holding
// rank `rank` (1-based from the top), with `above` the count in higher bins.
// `total` receives the histogram sum. Requires 1 <= rank <= total.
FMT_DEVICE uint32_t find_rank_bin(const uint32_t* h, uint32_t rank,
                                  uint32_t& above, uint32_t& total) {
  const uint32_t lane_id = get_lane_id();
  // Lane l owns bins 255 - 8l .. 248 - 8l (descending with the lane).
  uint32_t counts[8];
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t k = 0; k < 8; ++k) {
    counts[k] = hist_get(h, 255 - 8 * lane_id - k);
    sum += counts[k];
  }
  const uint32_t incl = warp::inclusive_sum(sum);
  total = __shfl_sync(warp::kFullMask, incl, 31);
  const uint32_t hit = __ballot_sync(warp::kFullMask, incl >= rank);
  const uint32_t w = hit ? __ffs(hit) - 1 : 31;
  uint32_t bin = 0, c = __shfl_sync(warp::kFullMask, incl - sum, w);
  above = c;
  if (lane_id == w) {
    bool found = false;
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
      if (!found && c + counts[k] >= rank) {
        bin = 255 - 8 * lane_id - k;
        above = c;
        found = true;
      }
      if (!found) c += counts[k];
    }
  }
  bin = __shfl_sync(warp::kFullMask, bin, w);
  above = __shfl_sync(warp::kFullMask, above, w);
  return bin;
}

// The head weights of one row as the reduce consumes them (pairs j, j + 1).
struct WeightsReg {  // all in registers (the scan)
  const float* w;
  FMT_DEVICE float2 pair(uint32_t j) const {
    return make_float2(w[j], w[j + 1]);
  }
};
struct WeightsSmem {  // broadcast loads from shared memory (the sample tiles)
  const float* w;
  FMT_DEVICE float2 pair(uint32_t j) const {
    AlignedVector<float, 2> v;
    v.load(w + j);
    return make_float2(v[0], v[1]);
  }
};

// Partial reduction over the 16 head columns [j0, j0 + 16) of one row (`acc`
// holds them), accumulating the two chains of reduce_heads_lin in its order: a
// 16-column part of the row can reduce while the next part lands.
template <typename Weights>
FMT_DEVICE void reduce_part_lin(const float* acc, const Weights& w, uint32_t j0,
                                float2& sum_0, float2& sum_1) {
#pragma unroll
  for (uint32_t j = 0; j < 16; j += 4) {
    const auto a_0 = make_float2(fabsf(acc[j]), fabsf(acc[j + 1]));
    const auto a_1 = make_float2(fabsf(acc[j + 2]), fabsf(acc[j + 3]));
    sum_0 = math::fma_f32x2(a_0, w.pair(j0 + j), sum_0);
    sum_1 = math::fma_f32x2(a_1, w.pair(j0 + j + 2), sum_1);
  }
}
// Linear-term reduction: (sum_h |s_h| w'_h + L0 + L1 + 2^-8 (L2 + L3)) * scale,
// with w' = w / (2 s_r) and L_t = term_t(q') . k from the extra B rows; the
// caller multiplies by s_r.
template <uint32_t kNumHeads, typename Weights>
FMT_DEVICE float reduce_heads_lin(const float* acc, const float* lin,
                                  const Weights& w, float scale) {
  float2 sum_0 = make_float2(lin[0], lin[1]);
  float2 sum_1 = math::mul_f32x2(make_float2(lin[2], lin[3]),
                                 make_float2(0x1p-8f, 0x1p-8f));
#pragma unroll
  for (uint32_t j = 0; j < kNumHeads; j += 4) {
    sum_0 = math::fma_f32x2(make_float2(fabsf(acc[j]), fabsf(acc[j + 1])),
                            w.pair(j), sum_0);
    sum_1 = math::fma_f32x2(make_float2(fabsf(acc[j + 2]), fabsf(acc[j + 3])),
                            w.pair(j + 2), sum_1);
  }
  const auto sum = math::add_f32x2(sum_0, sum_1);
  return (sum.x + sum.y) * scale;
}

// kNumHeads    index heads (32 or 64).
// kNumKVStages KV pipeline depth (32 KB per stage).
// kNumQStages  Q pipeline depth.
template <uint32_t kNumHeads, uint32_t kNumKVStages, uint32_t kNumQStages,
          bool kRecovery = false>
struct Fp8IndexerImpl {
  // 64 heads at most: a uint2 sample code holds 4 rows, a drain warp serves 4
  // rows (8 lanes each).
  static_assert(kNumHeads == 32 || kNumHeads == 64, "32 or 64 index heads");
  static constexpr uint32_t kHeadDim = 128;
  // A math warpgroup reduces 128 row x head columns: 4 rows at 32 heads, 2
  // at 64.
  static constexpr uint32_t kRowsPerWG = 128 / kNumHeads;
  static constexpr uint32_t kBlockQ = 2 * kRowsPerWG;
  static constexpr uint32_t kKeysPerThread =
      2;  // halves of a tile per warpgroup
  static constexpr uint32_t kSplitKV = 256;
  static constexpr uint32_t kUmmaM = 128;
  static constexpr uint32_t kUmmaK = 32;
  static constexpr uint32_t kLinTerms = 4;  // e4m3 terms of q' per row
  // Linear-term B rows per warpgroup, padded to 16 (an M = 128 UMMA takes N in
  // steps of 16); the padding rows are never written and their TMEM columns
  // never read.
  static constexpr uint32_t kLinRows =
      div_ceil(kRowsPerWG * kLinTerms, 16u) * 16u;
  static constexpr uint32_t kHeadCols =
      kRowsPerWG * kNumHeads;  // per warpgroup
  static constexpr uint32_t kUmmaN = kHeadCols + kLinRows;
  static constexpr uint32_t kNumTmemStages = 3;
  static constexpr uint32_t kNumTmemCols = 512;
  // Threshold samples are single keys on an odd stride: 65 up to 512K keys of
  // span, 129 up to 1M (at most kMaxSampleTiles tiles of 256; the op takes at
  // most 1M keys). Top keys sit in runs, so a block of consecutive keys lands
  // in one or misses it and makes the threshold noisy.
  static constexpr uint32_t sample_stride(uint32_t m) {
    return m == 0 ? 65u : 129u;
  }
  // Q stage: [lin WG0 16 | Q 256 | lin WG1 16] (each warpgroup's B is kUmmaN
  // contiguous rows), 128 B swizzled rows.
  static constexpr uint32_t kQTmaRows = kBlockQ * kNumHeads;
  static constexpr uint32_t kQTmaBytes = kQTmaRows * kHeadDim;
  static constexpr uint32_t kQTmaRowOffset = kLinRows;
  static constexpr uint32_t kQRows = kQTmaRows + 2 * kLinRows;
  static constexpr uint32_t kQBytes = kQRows * kHeadDim;
  static constexpr uint32_t kKVBytes = kSplitKV * kHeadDim;  // 32 KB
  static constexpr uint32_t kWBytes = kBlockQ * kNumHeads * 4;
  static constexpr uint32_t kSFBytes = kSplitKV * 4;  // 1 KB
  static constexpr uint32_t kNumMathWarps = 8,
                            kNumMathThreads = 32 * kNumMathWarps;
  // At most 32 sample tiles per block (see make_block_info). Their codes (16
  // bits per row and key) wait in a global scratch for the second, low-byte
  // pass: per CTA [tile][half][math thread] x up to 4 rows.
  static constexpr uint32_t kMaxSampleTiles = 32;
  static constexpr uint32_t kCodesPerCta =
      kMaxSampleTiles * kKeysPerThread * kNumMathThreads;
  static constexpr uint32_t kNumThreads = 384;
  // Candidate emission: per (bank, warp, row) the passing lanes store their
  // fp32 score at a fixed slot [tile][half][lane] plus the ballot mask per
  // (tile, half); the warps hand over every kInterval tiles (double-banked
  // mbarriers) and a drain warp per warpgroup gathers the set lanes' scores
  // into the slab (the key is rebuilt from tile, half, warp, lane).
  static constexpr uint32_t kInterval = 2;  // tiles per handoff
  // Slots a (warp, row) list holds per interval: one per lane and half tile.
  static constexpr uint32_t kListCap = kInterval * 2 * 32;
  // 64 heads: 216 / 72 (nvcc 13.0: 6-9% less time at 24K-32K keys, 0.3-2% from
  // 64K keys; nvcc 13.4: under 1%). 32 heads keep 224 / 56: 216 spills in
  // their scan loops.
  static constexpr uint32_t kMathRegs = kNumHeads == 64 ? 216 : 224,
                            kSpecRegs = kNumHeads == 64 ? 72 : 56;
  static_assert(kNumTmemStages * kUmmaN <= kNumTmemCols);
  static_assert(kHeadCols == kUmmaM && kUmmaN % 16 == 0 && kUmmaN <= 256,
                "UMMA shape");
  static_assert(kQBytes % 1024 == 0 && kQTmaRowOffset * kHeadDim % 1024 == 0,
                "swizzle atoms");
  using tcgen05 = ptx::TCGen05;
  enum Barrier { BAR_MATH = 1, BAR_WG = 2 };  // BAR_WG + wg: one warpgroup

  // B operand base row of warpgroup `wg` within the Q stage.
  FMT_DEVICE static constexpr uint32_t b_row_base(uint32_t wg) {
    return wg * kUmmaN;
  }
  // TMEM column of row i's heads / linear terms within a slot, for warpgroup
  // `wg`.
  FMT_DEVICE static constexpr uint32_t slot_heads_col(uint32_t wg) {
    return wg == 0 ? kLinRows : 0;
  }
  FMT_DEVICE static constexpr uint32_t slot_lin_col(uint32_t wg) {
    return wg == 0 ? 0 : kHeadCols;
  }
  // Byte offset of the linear-term rows of warpgroup `wg` in the Q stage.
  FMT_DEVICE static constexpr uint32_t lin_bytes(uint32_t wg) {
    return wg == 0 ? 0 : (kLinRows + kQTmaRows) * kHeadDim;
  }

  struct Smem {
    alignas(1024) uint8_t q[kNumQStages][kQBytes];
    alignas(1024) uint8_t kv[kNumKVStages][kKVBytes];
    alignas(128) float sf_kv[kNumKVStages][kSplitKV];
    alignas(128) float weights[kNumQStages][kBlockQ * kNumHeads];
    alignas(16) uint32_t hist[kBlockQ][128];  // high byte of the sample codes
    alignas(16) uint32_t
        lowhist[kBlockQ][128];  // low byte, within the chosen high-byte bin
    // Interval drain: per bank, math warp and row the score slots, the ballot
    // masks per (tile, half) of the rows, and the handoff barriers.
    alignas(16) uint32_t recs[2][kNumMathWarps][kRowsPerWG][kListCap];
    alignas(16) uint32_t masks[2][kNumMathWarps][kInterval][2][kRowsPerWG];
    uint64_t h_full[2][2];  // [bank][warpgroup]: arrived by the warpgroup's 128
                            // threads
    uint64_t h_empty[2][2];  // [bank][warpgroup]: arrived by its drain warp
    uint64_t drain_done;
    // Linear term: per Q stage the row scales s_r and the weights as w / (2
    // s_r).
    float lin_scale[kNumQStages][kBlockQ];
    alignas(16) float wscaled[kNumQStages][kBlockQ * kNumHeads];
    uint64_t full_lin[kNumQStages];
    float thr[kBlockQ];
    float ref[kBlockQ];  // the rows' sample references (see sample_code)
    float ref_part[kNumMathWarps][kRowsPerWG];  // per math warp and row
    uint32_t cnt[kBlockQ];  // candidates of a whole block's rows
    uint32_t sel_bin[kBlockQ];
    uint32_t sel_above[kBlockQ];
    uint32_t sel_rank[kBlockQ];
    uint64_t full_q[kNumQStages];
    uint64_t empty_q[kNumQStages];
    uint64_t full_kv[kNumKVStages];
    uint64_t empty_kv[kNumKVStages];
    uint64_t full_tmem[kNumTmemStages];
    uint64_t empty_tmem[kNumTmemStages];
    uint32_t tmem_base;
  };
  static constexpr uint32_t kSmemBytes = sizeof(Smem);
  static_assert(kSmemBytes <= 232448,
                "shared memory exceeds the 227 KB opt-in limit");

  // Ring of `kStages` slots: `advance()` returns the slot and its phase parity.
  template <uint32_t kStages>
  struct Ring {
    uint32_t idx = 0;
    uint32_t phase = 0;
    struct Pos {
      uint32_t idx, phase;
    };
    FMT_DEVICE Pos advance() {
      const Pos p{idx, phase};
      if (++idx == kStages) {
        idx = 0;
        phase ^= 1;
      }
      return p;
    }
    FMT_DEVICE Pos idx_phase() const { return {idx, phase}; }
    // The current slot, then moves `n` slots forward.
    FMT_DEVICE Pos advance(uint32_t n) {
      const Pos p{idx, phase};
      for (uint32_t i = 0; i < n; ++i) {
        advance();
      }
      return p;
    }
  };

  struct BlockInfo {
    uint32_t ks[kBlockQ];
    uint32_t len[kBlockQ];
    uint32_t kv_lo;
    uint32_t splits;
    uint32_t stride;  // keys between consecutive samples
    uint32_t sbase;   // first sampled key
    uint32_t smap;    // kv_strided map of the stride
    uint32_t slimit;  // samples at or past this key are not reachable through
                      // the map
    uint32_t sample_tiles;  // 0 when no row needs a threshold
    uint32_t blk;           // row block index (rows blk * kBlockQ ..)
    bool any;
  };

  FMT_DEVICE static bool needs_recovery(const Fp8IndexerParams& params,
                                        uint32_t row, uint32_t length) {
    const int32_t count = __ldg(params.cand_count + row);
    return count < static_cast<int32_t>(min(length, params.topk)) ||
           count > static_cast<int32_t>(params.cap) ||
           __ldg(params.overflow + row) != 0;
  }

  FMT_DEVICE static uint32_t visible_keys(const Fp8IndexerParams& params,
                                          uint32_t row) {
    const uint32_t start =
        min(uint32_t(max(params.ks[row], 0)), params.num_keys);
    const uint32_t end = min(uint32_t(max(params.ke[row], 0)), params.num_keys);
    return max(start, end) - start;
  }

  FMT_DEVICE static BlockInfo make_block_info(const Fp8IndexerParams& params,
                                              uint32_t blk) {
    BlockInfo b{};
    b.blk = blk;
    uint32_t lo = 0xffffffffu, hi = 0;
    bool need_t = false;
#pragma unroll
    for (uint32_t i = 0; i < kBlockQ; ++i) {
      const uint32_t row = blk * kBlockQ + i;
      uint32_t s = 0, e = 0;
      if (row < params.num_rows) {
        s = min(static_cast<uint32_t>(max(__ldg(params.ks + row), 0)),
                params.num_keys);
        e = min(static_cast<uint32_t>(max(__ldg(params.ke + row), 0)),
                params.num_keys);
        e = max(e, s);
        if constexpr (kRecovery) {
          if (!needs_recovery(params, row, e - s)) s = e = 0;
        }
      }
      b.ks[i] = s;
      b.len[i] = e - s;
      if (e > s) {
        lo = min(lo, s);
        hi = max(hi, e);
        need_t |= !kRecovery && (e - s) > params.dense_bound;
      }
    }
    b.any = hi > 0;
    if (!b.any) return b;
    b.kv_lo = lo / 4 * 4;  // 16 B aligned scale loads
    const uint32_t span = hi - b.kv_lo;
    b.splits = div_ceil(span, kSplitKV);
    if (need_t) {
      // a per-block phase in [0, S) keeps the grid off any fixed alignment of
      // the keys
      const uint32_t m = span <= 524288u ? 0u : 1u;
      const uint32_t stride = sample_stride(m);
      b.stride = stride;
      b.smap = m;
      b.slimit = params.sample_limit[m];
      b.sbase = b.kv_lo + ((blk * 2654435761u) >> 8) % stride;
      const uint32_t n = b.sbase < hi ? div_ceil(hi - b.sbase, stride) : 0u;
      b.sample_tiles = min(div_ceil(n, kSplitKV), kMaxSampleTiles);
    }
    return b;
  }

  // A CTA's work. Blocks go round-robin to the CTAs as whole units (sample,
  // thresholds, scan) for the full waves; the last, partial wave of r blocks is
  // split so it keeps every SM busy: each tail block gets a sample unit (on CTA
  // j for tail block j, publishing its thresholds to global memory) and
  // `tail_chunks` scan pieces (key fractions i / C of the block), ordered
  // chunk-major so the pieces running at the same time scan the same keys and
  // share L2. Each round of `grid` pieces gives CTA j < r the piece of its own
  // block j and CTAs >= r the rest, so a piece only waits for its own CTA or a
  // lower one: CTAs start in index order, and the grid needs no co-residency
  // (kernels on other streams may hold SMs).
  enum class UnitKind { WHOLE, SAMPLE, PIECE };
  struct Unit {
    BlockInfo b;
    UnitKind kind;
    uint32_t sample_tiles;  // sample tiles the unit scores (0 for a piece)
    uint32_t t0, t1;        // the unit's scan tiles [t0, t1)
    bool valid;             // false past the CTA's last unit
    // Whether the unit runs the pipeline: a block with keys, and a piece with
    // tiles.
    FMT_DEVICE bool live() const {
      return valid && b.any && (kind != UnitKind::PIECE || t0 < t1);
    }
  };
  FMT_DEVICE static Unit unit_at(const Fp8IndexerParams& params, uint32_t k) {
    if constexpr (kRecovery) {
      Unit u{};
      u.kind = UnitKind::WHOLE;
      u.valid = k == 0;
      if (u.valid) {
        u.b = make_block_info(params, params.recovery_block);
        u.t1 = u.b.splits;
      }
      return u;
    }
    const uint32_t nb = params.num_blocks, grid = gridDim.x, c = blockIdx.x;
    const uint32_t full = nb / grid, tail = nb % grid;
    Unit u{};
    uint32_t blk = c + k * grid, chunk = 0;
    u.kind = UnitKind::WHOLE;
    u.valid = true;
    if (k >= full) {
      if (params.tail_chunks == 1) {
        u.valid = k == full && c < tail;
      } else if (k == full && c < tail) {
        u.kind = UnitKind::SAMPLE;
      } else {
        // The CTA's round of pieces [base, base + grid), piece = chunk * tail +
        // tail block: CTA c >= tail takes piece base + c, CTA c < tail the one
        // of its own block c.
        const bool own = c < tail;
        const uint32_t base = (k - full - (own ? 1 : 0)) * grid;
        const uint32_t n = own ? base : base + c, t = max(tail, 1u);
        const uint32_t q = n / t, r = n % t;
        blk = full * grid + (own ? c : r);
        chunk = own && c < r ? q + 1 : q;
        u.kind = UnitKind::PIECE;
        u.valid = tail > 0 && chunk < params.tail_chunks;
      }
    }
    if (!u.valid) return u;
    u.b = make_block_info(params, blk);
    // Piece starts are multiples of kInterval, so a piece's handoff intervals
    // line up with the tile index.
    const uint32_t splits = u.b.splits;
    const auto cut = [&](uint32_t i) {
      return i == params.tail_chunks
                 ? splits
                 : min(splits * i / params.tail_chunks / kInterval * kInterval,
                       splits);
    };
    switch (u.kind) {
      case UnitKind::WHOLE:
        u.sample_tiles = u.b.sample_tiles;
        u.t1 = splits;
        break;
      case UnitKind::SAMPLE:
        u.sample_tiles = u.b.sample_tiles;
        break;
      case UnitKind::PIECE:
        u.t0 = cut(chunk);
        u.t1 = cut(chunk + 1);
        break;
    }
    return u;
  }

  // One warp: the linear-term rows [r_begin, r_end) of Q stage `qs`. q'_r =
  // sum_h (w_rh / 2) q_rh (lane l owns dims 4l..4l+3) is split into four e4m3
  // terms: terms 0, 1 at the power-of-two row scale s_r, terms 2, 3 at s_r 2^-8
  // (the residual after two terms is <= 0.5 ulp, so x256 does not saturate).
  // Term t of row r is B row 4 (r % 4) + t of its warpgroup's linear rows,
  // swizzled like Q. Also writes w / (2 s_r) and s_r, and fences the stores for
  // the UMMA.
  FMT_DEVICE static void write_lin_rows(Smem& smem, uint32_t qs,
                                        uint32_t lane_id, uint32_t r_begin,
                                        uint32_t r_end) {
    uint8_t* const qstage = smem.q[qs];
    const uint8_t* q = qstage + kQTmaRowOffset * kHeadDim;
    const float* const weights = smem.weights[qs];
#pragma unroll 1
    for (uint32_t r = r_begin; r < r_end; ++r) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (uint32_t h = 0; h < kNumHeads; ++h) {
        const uint32_t row = r * kNumHeads + h;
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(
            q + sw128_off(row, lane_id >> 2) + (lane_id & 3u) * 4u);
        const float wv = weights[r * kNumHeads + h] * 0.5f;
        const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
        const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
        const float2 f0 =
            __half22float2(*reinterpret_cast<const __half2*>(&lo));
        const float2 f1 =
            __half22float2(*reinterpret_cast<const __half2*>(&hi));
        acc[0] = fmaf(wv, f0.x, acc[0]);
        acc[1] = fmaf(wv, f0.y, acc[1]);
        acc[2] = fmaf(wv, f1.x, acc[2]);
        acc[3] = fmaf(wv, f1.y, acc[3]);
      }
      float m = fmaxf(fmaxf(fabsf(acc[0]), fabsf(acc[1])),
                      fmaxf(fabsf(acc[2]), fabsf(acc[3])));
      m = warp::reduce_max(m);
      // s >= m / 448 keeps term 0 finite; clamped so 1 / s stays finite too.
      int e = -120;
      if (m > 0.f && isfinite(m)) {
        frexpf(m * (1.0f / 448.0f), &e);
        e = max(e, -120);
      }
      const float sr = ldexpf(1.0f, e), inv_s = ldexpf(1.0f, -e);
      float res[4];
#pragma unroll
      for (uint32_t d = 0; d < 4; ++d) {
        res[d] = acc[d] * inv_s;
      }
      uint8_t* lin_rows = qstage + lin_bytes(r / kRowsPerWG);
#pragma unroll
      for (uint32_t t = 0; t < kLinTerms; ++t) {
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
        if (t == 1) {
#pragma unroll
          for (uint32_t d = 0; d < 4; ++d) {
            res[d] *= 256.0f;
          }
        }
        const uint32_t n = (r % kRowsPerWG) * kLinTerms + t;
        *reinterpret_cast<uint32_t*>(lin_rows + sw128_off(n, lane_id >> 2) +
                                     (lane_id & 3u) * 4u) =
            static_cast<uint32_t>(c01) | (static_cast<uint32_t>(c23) << 16);
      }
#pragma unroll
      for (uint32_t h = 0; h < kNumHeads; h += 32) {
        smem.wscaled[qs][r * kNumHeads + h + lane_id] =
            weights[r * kNumHeads + h + lane_id] * 0.5f * inv_s;
      }
      if (lane_id == 0) smem.lin_scale[qs][r] = sr;
    }
    ptx::fence_proxy_async_shared();  // generic smem writes before the UMMA
                                      // reads them
    __syncwarp();
  }

  // The TMA warp: per unit its block's Q and weights, then a whole or sample
  // unit's sample tiles, then the scan tiles.
  FMT_DEVICE static void load_warp(const Fp8IndexerParams& params,
                                   const Fp8IndexerTensorMaps& tmaps,
                                   Smem& smem, uint32_t lane_id) {
    ptx::setmaxnreg_dec<kSpecRegs>();
    Ring<kNumQStages> q_pipe;
    Ring<kNumKVStages> kv_pipe;
    for (uint32_t ui = 0;; ++ui) {
      const Unit u = unit_at(params, ui);
      if (!u.valid) break;
      if (!u.live()) continue;
      const BlockInfo& b = u.b;
      const auto [qs, qph] = q_pipe.advance();
      if (warp::elect_one_lane()) {
        ptx::mbar_wait_parity(&smem.empty_q[qs], qph ^ 1);
        ptx::mbar_arrive_expect_tx(&smem.full_q[qs], kQTmaBytes + kWBytes);
        ptx::cp_async_bulk_tensor_2d(
            ptx::to_shared(smem.q[qs] + kQTmaRowOffset * kHeadDim), &tmaps.q, 0,
            static_cast<int32_t>(b.blk * kQTmaRows), &smem.full_q[qs]);
        ptx::cp_async_bulk_tensor_2d(
            ptx::to_shared(smem.weights[qs]), &tmaps.weights, 0,
            static_cast<int32_t>(b.blk * kBlockQ), &smem.full_q[qs]);
      }
      __syncwarp();
      // Sample tiles: one strided TMA load of 256 keys (one lane) and, per
      // lane, the scales of 8 samples (4 B each, too far apart for a TMA box);
      // the 32 lanes arrive on `full_kv` once their scales are in place.
#pragma unroll 1
      for (uint32_t t = 0; t < u.sample_tiles; ++t) {
        const auto [s, ph] = kv_pipe.advance();
        ptx::mbar_wait_parity(&smem.empty_kv[s], ph ^ 1);
        const uint32_t key0 =
            b.sbase + t * kSplitKV * b.stride;  // first sample of the tile
        {
          float v[8];
#pragma unroll
          for (uint32_t e = 0; e < 8; ++e) {
            const uint32_t key = key0 + (lane_id * 8 + e) * b.stride;
            v[e] = key < b.slimit ? __ldg(params.kv_scale + key) : 0.0f;
          }
          reinterpret_cast<float4*>(smem.sf_kv[s] + lane_id * 8)[0] =
              make_float4(v[0], v[1], v[2], v[3]);
          reinterpret_cast<float4*>(smem.sf_kv[s] + lane_id * 8)[1] =
              make_float4(v[4], v[5], v[6], v[7]);
        }
        ptx::fence_proxy_async_shared();
        if (warp::elect_one_lane()) {
          ptx::mbar_expect_tx(&smem.full_kv[s], kKVBytes);
          ptx::cp_async_bulk_tensor_3d(
              ptx::to_shared(smem.kv[s]), &tmaps.kv_strided[b.smap], 0,
              static_cast<int32_t>(key0 % b.stride),
              static_cast<int32_t>(key0 / b.stride), &smem.full_kv[s]);
        }
        __syncwarp();
        ptx::mbar_arrive(&smem.full_kv[s]);
      }
#pragma unroll 1
      for (uint32_t t = u.t0; t < u.t1; ++t) {
        const auto [s, ph] = kv_pipe.advance();
        const auto key = static_cast<int32_t>(b.kv_lo + t * kSplitKV);
        if (warp::elect_one_lane()) {
          ptx::mbar_wait_parity(&smem.empty_kv[s], ph ^ 1);
          ptx::mbar_expect_tx(&smem.full_kv[s], kKVBytes + kSFBytes);
          ptx::cp_async_bulk_tensor_2d(ptx::to_shared(smem.kv[s]), &tmaps.kv, 0,
                                       key, &smem.full_kv[s]);
          ptx::cp_async_bulk_tensor_2d(ptx::to_shared(smem.sf_kv[s]),
                                       &tmaps.sf_kv, key, 0, &smem.full_kv[s]);
        }
        __syncwarp();
        ptx::mbar_arrive(&smem.full_kv[s]);
      }
    }
  }

  // The UMMA warp: every lane walks the rings and waits on the barriers, so the
  // loop state is warp-uniform and ptxas keeps it in uniform registers; the
  // lane elected once at the start issues all MMAs, commits and arrivals, so
  // every commit comes from the lane that issued the MMAs it tracks.
  FMT_DEVICE static void mma_warp(const Fp8IndexerParams& params, Smem& smem,
                                  uint32_t tmem_base) {
    ptx::setmaxnreg_dec<kSpecRegs>();
    const bool leader = warp::elect_one_lane();
    Ring<kNumQStages> q_pipe;
    Ring<kNumKVStages> kv_pipe;
    Ring<kNumTmemStages> tmem_pipe;
    const auto idesc = tcgen05::make_instr_desc_f8<kUmmaM, kUmmaN>();
    const auto kv_desc0 = tcgen05::make_desc_kmajor_sw128(smem.kv[0]);
    const auto q_desc0 = tcgen05::make_desc_kmajor_sw128(smem.q[0]);
    for (uint32_t ui = 0;; ++ui) {
      const Unit u = unit_at(params, ui);
      if (!u.valid) break;
      if (!u.live()) continue;
      const BlockInfo& b = u.b;
      const auto [qs, qph] = q_pipe.advance();
      ptx::mbar_wait_parity(&smem.full_q[qs], qph);
      ptx::mbar_wait_parity(&smem.full_lin[qs], qph);
      tcgen05::fence_after_thread_sync();
      const auto b_desc = q_desc0.advance_bytes(qs * kQBytes);
      const uint32_t total = u.sample_tiles + u.t1 - u.t0;
      auto [s, ph] = kv_pipe.idx_phase();
      if (total != 0) {
        const auto first = kv_pipe.advance();
        s = first.idx;
        ph = first.phase;
        ptx::mbar_wait_parity(&smem.full_kv[s], ph);
      }
#pragma unroll 1
      for (uint32_t t = 0; t < total; ++t) {
        const uint32_t cur = s;
        // Per KV half one MMA per math warpgroup (slot order: half 0 WG0, half
        // 0 WG1, half 1 WG0, half 1 WG1).
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
          for (uint32_t g = 0; g < 2; ++g) {
            const auto [ts, tph] = tmem_pipe.advance();
            ptx::mbar_wait_parity(&smem.empty_tmem[ts], tph ^ 1);
            tcgen05::fence_after_thread_sync();
            const auto a_desc =
                kv_desc0.advance_bytes(cur * kKVBytes + h * kUmmaM * kHeadDim);
            const auto b_desc_g =
                b_desc.advance_bytes(b_row_base(g) * kHeadDim);
#pragma unroll
            for (uint32_t k = 0; k < kHeadDim / kUmmaK; ++k) {
              // The next tile's stage is awaited before this tile's last MMA so
              // the tensor core never idles on the issue path.
              if (h == 1 && g == 1 && k == kHeadDim / kUmmaK - 1 &&
                  t + 1 < total) {
                const auto nxt = kv_pipe.advance();
                s = nxt.idx;
                ph = nxt.phase;
                ptx::mbar_wait_parity(&smem.full_kv[s], ph);
              }
              if (leader) {
                tcgen05::mma_f8_ss(tmem_base + ts * kUmmaN,
                                   a_desc.advance_bytes(k * kUmmaK).bits,
                                   b_desc_g.advance_bytes(k * kUmmaK).bits,
                                   idesc.bits, k);
              }
            }
            if (leader) tcgen05::commit(&smem.full_tmem[ts]);
          }
        }
      }
      if (leader) ptx::mbar_arrive(&smem.empty_q[qs]);
    }
  }

  // Coarse selection, warp r for row r: the high-byte bin of the row's rank-j'
  // sample (j = topk S / M, alpha = 4 sqrt(2 / j), j' = ceil((1 + alpha) j), S
  // the row's sample count). Rank 0: every key is a candidate. With single-key
  // samples the hit count has dispersion ~0.75 on real GLM-5.3 data and alpha 4
  // left no row short in 7M.
  FMT_DEVICE static void select_coarse(const Fp8IndexerParams& params,
                                       Smem& smem, uint32_t r, uint32_t len_r) {
    uint32_t above = 0, total = 0, bin = 0, rank = 0;
    if (len_r > params.dense_bound) {
      bin = find_rank_bin(smem.hist[r], 1, above, total);
      const auto num_samples = static_cast<float>(total);
      const float j = static_cast<float>(params.topk) * num_samples /
                      static_cast<float>(len_r);
      const float jp = ceilf((1.0f + 4.0f * sqrtf(2.0f / j)) * j);
      rank = static_cast<uint32_t>(jp);
      if (total == 0 || jp > num_samples || rank == 0) {
        rank = 0;
      } else {
        bin = find_rank_bin(smem.hist[r], rank, above, total);
      }
    }
    if (get_lane_id() == 0) {
      smem.sel_bin[r] = bin;
      smem.sel_above[r] = above;
      smem.sel_rank[r] = rank;
      smem.thr[r] = len_r == 0 ? INFINITY : -INFINITY;
    }
  }
  // Fine selection, warp r for row r: the low byte within the chosen bin gives
  // the exact code, and T from it.
  FMT_DEVICE static void select_fine(Smem& smem, uint32_t r) {
    const uint32_t rank = smem.sel_rank[r];
    if (rank == 0) return;
    uint32_t above = 0, total = 0;
    const uint32_t low =
        find_rank_bin(smem.lowhist[r], rank - smem.sel_above[r], above, total);
    if (get_lane_id() == 0)
      smem.thr[r] = code_threshold(smem.ref[r], (smem.sel_bin[r] << 8) | low);
  }

  // Tiles [first, last) are inside every live row's range: no per-key range
  // check there.
  FMT_DEVICE static void full_tile_range(const uint32_t (&ks)[kRowsPerWG],
                                         const uint32_t (&len)[kRowsPerWG],
                                         uint32_t kv_lo, uint32_t splits,
                                         uint32_t& first, uint32_t& last) {
    first = 0;
    last = splits;
#pragma unroll
    for (uint32_t i = 0; i < kRowsPerWG; ++i) {
      if (len[i] != 0) {
        first = max(first, div_ceil(ks[i] - kv_lo, kSplitKV));
        last = min(last, (ks[i] + len[i] - kv_lo) / kSplitKV);
      }
    }
    last = max(last, first);
  }

  // A piece's scan is drained: count its tiles; the block's last piece writes
  // the rows' overflow flags and resets the block's workspace for the next
  // launch.
  FMT_DEVICE static void finish_piece(const Fp8IndexerParams& params,
                                      uint32_t blk, uint32_t tiles,
                                      uint32_t splits) {
    __threadfence();
    if (atomicAdd(params.block_done + blk, tiles) + tiles != splits) return;
    __threadfence();
#pragma unroll
    for (uint32_t i = 0; i < kBlockQ; ++i) {
      const uint32_t row = blk * kBlockQ + i;
      if (row < params.num_rows)
        params.overflow[row] =
            static_cast<uint32_t>(__ldcg(params.cand_count + row)) > params.cap;
    }
    params.block_done[blk] = 0;
    params.block_ready[blk] = 0;
  }

  // One tile half's ballots of the warpgroup's rows into the drain's mask
  // table.
  FMT_DEVICE static void store_masks(uint32_t* dst,
                                     const uint32_t (&m)[kRowsPerWG]) {
    AlignedVector<uint32_t, kRowsPerWG> v;
#pragma unroll
    for (uint32_t i = 0; i < kRowsPerWG; ++i) {
      v[i] = m[i];
    }
    v.store(dst);
  }

  // The math warps (two warpgroups, kRowsPerWG rows each): per block the sample
  // passes and the thresholds, then the scan with the candidate slots handed to
  // the drains.
  FMT_DEVICE static void math_warps(const Fp8IndexerParams& params, Smem& smem,
                                    uint32_t tmem_base, uint32_t warp_id,
                                    uint32_t lane_id) {
    ptx::setmaxnreg_inc<kMathRegs>();
    const uint32_t wg = warp_id / 4;
    const uint32_t tid_wg = threadIdx.x % 128;
    // TMEM lane quadrant of this warp.
    const uint32_t taddr = tmem_base + ((warp_id % 4) * 32u << 16);
    const uint32_t heads_col = slot_heads_col(wg);
    const uint32_t lin_col = slot_lin_col(wg);
    const uint32_t row0 =
        wg * kRowsPerWG;  // this warpgroup's first row of the block
    // Whether this lane writes the block's per-row outputs of row row0 +
    // lane_id.
    const bool row_owner = warp_id % 4 == 0 && lane_id < kRowsPerWG;
    Ring<kNumQStages> q_pipe;
    Ring<kNumKVStages> kv_pipe;
    Ring<kNumTmemStages> tmem_pipe;
    tmem_pipe.advance(wg);
    uint32_t emit_epoch = 0;  // interval handoffs so far (bank = epoch & 1)
    uint32_t drain_phase = 0;
    float weights[kRowsPerWG][kNumHeads];
    for (uint32_t ui = 0;; ++ui) {
      const Unit u = unit_at(params, ui);
      if (!u.valid) break;
      const BlockInfo& b = u.b;
      if (!b.any) {
        // Rows without keys: no pipeline, the sample unit writes their outputs.
        const uint32_t row = b.blk * kBlockQ + row0 + lane_id;
        if (!kRecovery && u.kind != UnitKind::PIECE && row_owner &&
            row < params.num_rows) {
          params.thresholds[row] = INFINITY;
          params.cand_count[row] = 0;
          params.overflow[row] = 0;
        }
        continue;
      }
      if (!u.live()) continue;
      // This warpgroup's rows and the unit's scalars, copied out of `u`:
      // capturing it in the lambdas below, a run-time index into it, or keeping
      // it alive across the scan puts it on the stack or spills registers.
      uint32_t ks[kRowsPerWG], len[kRowsPerWG];
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
        ks[i] = wg ? b.ks[kRowsPerWG + i]
                   : b.ks[i];  // constant indices: the arrays stay in registers
        len[i] = wg ? b.len[kRowsPerWG + i] : b.len[i];
      }
      const uint32_t kv_lo = b.kv_lo, stride = b.stride,
                     sample_tiles = u.sample_tiles, blk = b.blk,
                     splits = b.splits;
      const uint32_t sbase = b.sbase, slimit = b.slimit;
      const uint32_t t0 = u.t0, t1 = u.t1;
      const UnitKind kind = u.kind;
      // Length of row `warp_id` of the block, for the per-row selection warps.
      const uint32_t len_w = warp_id < kBlockQ ? select_at(b.len, warp_id) : 0u;
      const auto [qs, qph] = q_pipe.advance();
      // Unit setup: zero the histograms (the previous unit's readers passed its
      // final barrier), then wait for Q, the weights and the linear-term rows.
      for (uint32_t j = threadIdx.x; j < 2 * kBlockQ * 128 / 4;
           j += kNumMathThreads) {
        AlignedVector<uint32_t, 4> zero;
        zero.fill(0);
        zero.store(&smem.hist[0][0], j);  // hist and lowhist are adjacent
      }
      if (row_owner) {
        smem.cnt[row0 + lane_id] = 0;
        smem.ref[row0 + lane_id] = 0.0f;
      }
      ptx::mbar_wait_parity(&smem.full_q[qs], qph);
      ptx::mbar_wait_parity(&smem.full_lin[qs], qph);
      float row_s[kRowsPerWG];
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
        row_s[i] = smem.lin_scale[qs][row0 + i];
      }
      const float* const w_src = smem.wscaled[qs] + row0 * kNumHeads;
      // The sample tiles read the weights from shared memory; the registers are
      // loaded right before the scan (not live across the sample code).
      const auto row_weights_s = [&](uint32_t i) {
        return WeightsSmem{&w_src[i * kNumHeads]};
      };
      const auto row_weights = [&](uint32_t i) {
        return WeightsReg{weights[i]};
      };
      ptx::bar_sync(BAR_MATH, kNumMathThreads);

      uint32_t need_t = 0;
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
        need_t |= (len[i] > params.dense_bound ? 1u : 0u) << i;
      }

      // This thread's sample codes of the block (see kCodesPerCta).
      uint2* const codes = params.sample_codes +
                           (kRecovery ? 0u : blockIdx.x) * kCodesPerCta +
                           threadIdx.x;
      // Per row the reference of its codes and the sample tile it comes from:
      // the tile of the row's first sample (~0u: the row samples nothing).
      float ref[kRowsPerWG];
      uint32_t ref_tile[kRowsPerWG];
      // One sample tile: score the warpgroup's keys, take the references of
      // the rows whose first sample is here, histogram the code's high byte
      // and stash the codes.
      const auto sample_tile = [&](uint32_t t) {
        const auto [s, ph] = kv_pipe.advance();
        ptx::mbar_wait_parity(&smem.full_kv[s], ph);
        float score[kKeysPerThread][kRowsPerWG];
        uint32_t valid = 0;  // bit h * kRowsPerWG + i
#pragma unroll
        for (uint32_t h = 0; h < kKeysPerThread; ++h) {
          const uint32_t idx = h * 128 + tid_wg;  // key within the tile
          const float scale = smem.sf_kv[s][idx];
          const auto [ts, tph] = tmem_pipe.advance(2);
          ptx::mbar_wait_parity(&smem.full_tmem[ts], tph);
          tcgen05::fence_after_thread_sync();
          if (h == kKeysPerThread - 1) {
            ptx::fence_proxy_async_shared();
            ptx::mbar_arrive(&smem.empty_kv[s]);
          }
          const uint32_t key = sbase + (t * kSplitKV + idx) * stride;
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            float acc[kNumHeads];
            float lin[kLinTerms];
            tcgen05::ld_32x32b<kNumHeads>(
                taddr + ts * kUmmaN + heads_col + i * kNumHeads, acc);
            tcgen05::ld_32x32b<kLinTerms>(
                taddr + ts * kUmmaN + lin_col + i * kLinTerms, lin);
            tcgen05::wait_ld();
            if (i == kRowsPerWG - 1) {
              tcgen05::fence_before_thread_sync();
              ptx::mbar_arrive(&smem.empty_tmem[ts]);
            }
            score[h][i] =
                reduce_heads_lin<kNumHeads>(acc, lin, row_weights_s(i), scale) *
                row_s[i];
            const bool ok =
                ((need_t >> i) & 1u) && (key - ks[i]) < len[i] && key < slimit;
            valid |= (ok ? 1u : 0u) << (h * kRowsPerWG + i);
          }
        }
        // References: the max of the row's valid samples in this tile over the
        // warpgroup (the rows and their reference tiles are warpgroup-uniform).
        bool any_ref = false;
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          any_ref |= ref_tile[i] == t;
        }
        if (any_ref) {
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            if (ref_tile[i] != t) continue;
            float m = -INFINITY;
#pragma unroll
            for (uint32_t h = 0; h < kKeysPerThread; ++h) {
              if ((valid >> (h * kRowsPerWG + i)) & 1u)
                m = fmaxf(m, score[h][i]);
            }
            m = warp::reduce_max(m);
            if (lane_id == 0) smem.ref_part[warp_id][i] = m;
          }
          ptx::bar_sync(BAR_WG + wg, 128);
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            if (ref_tile[i] != t) continue;
            float m = smem.ref_part[wg * 4][i];
#pragma unroll
            for (uint32_t w = 1; w < 4; ++w) {
              m = fmaxf(m, smem.ref_part[wg * 4 + w][i]);
            }
            ref[i] = isfinite(m) ? m : 0.0f;
            if (tid_wg == 0) smem.ref[row0 + i] = ref[i];
          }
        }
#pragma unroll
        for (uint32_t h = 0; h < kKeysPerThread; ++h) {
          uint32_t code[kRowsPerWG];
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            const bool ok = (valid >> (h * kRowsPerWG + i)) & 1u;
            code[i] = ok ? sample_code(score[h][i], ref[i]) : 0u;
            if (ok) hist_add(smem.hist[row0 + i], code[i] >> 8);
          }
          // The rows' 16-bit codes in one uint2 (row i: word i / 2, half i %
          // 2).
          uint32_t packed[2] = {0u, 0u};
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            packed[i / 2] =
                i % 2 == 0 ? code[i] : packed[i / 2] | code[i] << 16;
          }
          codes[(t * kKeysPerThread + h) * kNumMathThreads] =
              make_uint2(packed[0], packed[1]);
        }
      };

      // ---- sample tiles, then the threshold per row ----
      float thr[kRowsPerWG];
      if (sample_tiles != 0) {
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          // sample n is key sbase + n * stride
          const uint32_t first =
              ks[i] > sbase ? div_ceil(ks[i] - sbase, stride) : 0u;
          ref[i] = 0.0f;
          ref_tile[i] = ((need_t >> i) & 1u) ? first / kSplitKV : ~0u;
        }
#pragma unroll 1
        for (uint32_t t = 0; t < sample_tiles; ++t) {
          sample_tile(t);
        }
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        if (warp_id < kBlockQ) select_coarse(params, smem, warp_id, len_w);
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        // Low byte of the chosen bin from the stashed codes.
        uint32_t sel[kRowsPerWG];
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          sel[i] =
              smem.sel_rank[row0 + i] ? smem.sel_bin[row0 + i] : 0xffffffffu;
        }
#pragma unroll 1
        for (uint32_t j = 0; j < sample_tiles * kKeysPerThread; ++j) {
          const uint2 c = codes[j * kNumMathThreads];
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            const uint32_t code =
                ((i < 2 ? c.x : c.y) >> (i % 2 * 16)) & 0xffffu;
            if (code != 0 && (code >> 8) == sel[i])
              hist_add(smem.lowhist[row0 + i], code & 0xffu);
          }
        }
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        if (warp_id < kBlockQ) select_fine(smem, warp_id);
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          thr[i] = smem.thr[row0 + i];
        }
      } else if (kind != UnitKind::PIECE) {
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          thr[i] = len[i] == 0 ? INFINITY : -INFINITY;
        }
      } else {
        // A piece: the sample unit of the block published the thresholds.
        if (warp_id == 0 && lane_id == 0) {
          while (deep_gemm::ptx::ld_acq(params.block_ready + blk) == 0) {
            __nanosleep(64);
          }
        }
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          const uint32_t row = blk * kBlockQ + row0 + i;
          thr[i] = row < params.num_rows ? __ldcg(params.thresholds + row)
                                         : INFINITY;
        }
      }

      // ---- main scan ----
      // One tile; `edge`: the per-key range check is needed (some row does not
      // see the whole tile). TMEM is read in 16-column parts into two landing
      // buffers: a part lands while the previous one reduces, also across rows
      // and slots (36 registers next to the register weights, no spills).
      float acc_a[16], lin_a[kLinTerms];
      float acc_b[16], lin_b[kLinTerms];
      // Part of a row: columns [c0, c0 + 16) of row r (the linear columns with
      // the first part).
      const auto load_half = [&](uint32_t ts, uint32_t r, uint32_t c0,
                                 float* acc, float* lin) {
        tcgen05::ld_32x32b<16>(
            taddr + ts * kUmmaN + heads_col + r * kNumHeads + c0, acc);
        if (c0 == 0)
          tcgen05::ld_32x32b<kLinTerms>(
              taddr + ts * kUmmaN + lin_col + r * kLinTerms, lin);
      };
      const auto scan_tile = [&](uint32_t t, auto edge) {
        const uint32_t bank = emit_epoch & 1;
        if (t % kInterval == 0) {
          // The bank of this interval was drained (first use passes).
          ptx::mbar_wait_parity(&smem.h_empty[bank][wg],
                                ((emit_epoch >> 1) & 1) ^ 1);
        }
        // The tile's MMAs waited for its KV stage, so the stage's wait
        // (ordering this thread's scale reads) follows the slot's first TMEM
        // loads.
        const auto [s, ph] = kv_pipe.advance();
        auto [ts, tph] = tmem_pipe.advance(2);
        ptx::mbar_wait_parity(&smem.full_tmem[ts], tph);
        tcgen05::fence_after_thread_sync();
        load_half(ts, 0, 0, acc_a, lin_a);
        ptx::mbar_wait_parity(&smem.full_kv[s], ph);
#pragma unroll
        for (uint32_t h = 0; h < kKeysPerThread; ++h) {
          const uint32_t half = h;  // the KV half this thread scans
          const uint32_t idx = half * 128 + tid_wg;
          const uint32_t key = kv_lo + t * kSplitKV + idx;
          const float scale = smem.sf_kv[s][idx];
          float score[kRowsPerWG];
          bool pass[kRowsPerWG];
          bool any = false;
          // The half's slot is read out: release it and step to the next half's
          // slot (true), or release the KV stage after the last half.
          const auto next_slot = [&]() {
            tcgen05::fence_before_thread_sync();
            ptx::mbar_arrive(&smem.empty_tmem[ts]);
            if (h + 1 < kKeysPerThread) {
              const auto nxt = tmem_pipe.advance(2);
              ts = nxt.idx;
              tph = nxt.phase;
              ptx::mbar_wait_parity(&smem.full_tmem[ts], tph);
              tcgen05::fence_after_thread_sync();
              return true;
            }
            ptx::fence_proxy_async_shared();
            ptx::mbar_arrive(&smem.empty_kv[s]);
            return false;
          };
          // Per row, in 32-column steps: the step's first 16 columns reduce
          // while its other 16 land, then the next step's (or the next row's)
          // first 16 land while the second 16 reduce. The reads of a TMEM load
          // wait on its register scoreboard; `wait::ld` is only needed before
          // the slot release.
#pragma unroll
          for (uint32_t r = 0; r < kRowsPerWG; ++r) {
            float2 s0 = make_float2(lin_a[0], lin_a[1]);
            float2 s1 = math::mul_f32x2(make_float2(lin_a[2], lin_a[3]),
                                        make_float2(0x1p-8f, 0x1p-8f));
            bool more = true;
#pragma unroll
            for (uint32_t c = 0; c < kNumHeads; c += 32) {
              load_half(ts, r, c + 16, acc_b, lin_b);
              reduce_part_lin(acc_a, row_weights(r), c, s0, s1);
              if (c + 32 < kNumHeads) {
                load_half(ts, r, c + 32, acc_a, lin_a);
              } else if (r + 1 < kRowsPerWG) {
                load_half(ts, r + 1, 0, acc_a, lin_a);
              } else {
                tcgen05::wait_ld();  // every load of the slot landed: the slot
                                     // is free
                more = next_slot();
              }
              reduce_part_lin(acc_b, row_weights(r), c + 16, s0, s1);
            }
            const auto sum = math::add_f32x2(s0, s1);
            score[r] = ((sum.x + sum.y) * scale) * row_s[r];
            pass[r] = kRecovery ? len[r] != 0 : score[r] >= thr[r];
            if constexpr (decltype(edge)::value)
              pass[r] = pass[r] && (key - ks[r]) < len[r];
            any |= pass[r];
            if (r + 1 == kRowsPerWG && more) load_half(ts, 0, 0, acc_a, lin_a);
          }
          // Emission: one any-vote, then the row ballots back to back and
          // predicated slot stores (fixed slot [tile in interval][half][lane]
          // per row, exact fp32); the votes stay off the reduce chain.
          uint32_t* const masks =
              smem.masks[bank][warp_id][t % kInterval][half];
          const bool emit_any = __any_sync(warp::kFullMask, any);
          if (!emit_any && lane_id == 0) {
            const uint32_t zero[kRowsPerWG] = {};
            store_masks(masks, zero);
          }
          if (emit_any) {
            uint32_t m[kRowsPerWG];
#pragma unroll
            for (uint32_t i = 0; i < kRowsPerWG; ++i) {
              m[i] = __ballot_sync(warp::kFullMask, pass[i]);
            }
            const uint32_t slot = ((t % kInterval) * 2 + half) * 32 + lane_id;
#pragma unroll
            for (uint32_t i = 0; i < kRowsPerWG; ++i) {
              if (pass[i])
                smem.recs[bank][warp_id][i][slot] = __float_as_uint(score[i]);
            }
            if (lane_id == 0) store_masks(masks, m);
          }
        }
        if ((t + 1) % kInterval == 0 || t + 1 == t1) {
          // Hand the interval's slots to the drain warp.
          __syncwarp();
          ptx::mbar_arrive(&smem.h_full[bank][wg]);
          ++emit_epoch;
        }
      };
      uint32_t full_beg, full_end;
      full_tile_range(ks, len, kv_lo, splits, full_beg, full_end);
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; j += 4) {
          AlignedVector<float, 4> w4;
          w4.load(&w_src[i * kNumHeads + j]);
#pragma unroll
          for (uint32_t k = 0; k < 4; ++k) {
            weights[i][j + k] = w4[k];
          }
        }
      }
      {
        uint32_t t = t0;
#pragma unroll 1
        for (; t < min(full_beg, t1); ++t) {
          scan_tile(t, std::true_type{});
        }
#pragma unroll 1
        for (; t < min(full_end, t1); ++t) {
          scan_tile(t, std::false_type{});
        }
#pragma unroll 1
        for (; t < t1; ++t) {
          scan_tile(t, std::true_type{});
        }
      }
      // The next block's linear-term rows, one row per math warp, while the
      // drains finish (its Q landed long ago: the stage was free since the
      // block before this one ended).
      if (warp_id < kBlockQ && unit_at(params, ui + 1).live()) {
        Ring<kNumQStages> peek = q_pipe;
        const auto nxt = peek.advance();
        ptx::mbar_wait_parity(&smem.full_q[nxt.idx], nxt.phase);
        write_lin_rows(smem, nxt.idx, lane_id, warp_id, warp_id + 1);
        // kBlockQ warps x (32 / kBlockQ) lanes = the barrier's 32 arrivals.
        if (lane_id < 32 / kBlockQ) ptx::mbar_arrive(&smem.full_lin[nxt.idx]);
      }
      if (kind == UnitKind::WHOLE) {
        ptx::mbar_wait_parity(&smem.drain_done, drain_phase);
        drain_phase ^= 1;
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        const uint32_t row = blk * kBlockQ + row0 + lane_id;
        if (!kRecovery && row_owner && row < params.num_rows) {
          const uint32_t c = smem.cnt[row0 + lane_id];
          params.cand_count[row] = static_cast<int32_t>(c);
          params.overflow[row] = c > params.cap ? 1 : 0;
          params.thresholds[row] = select_at(thr, lane_id);
        }
      } else if (kind == UnitKind::SAMPLE) {
        // Publish the block: its thresholds and zeroed counters, then the ready
        // flag the pieces wait for.
        const uint32_t row = blk * kBlockQ + row0 + lane_id;
        if (row_owner && row < params.num_rows) {
          params.thresholds[row] = select_at(thr, lane_id);
          params.cand_count[row] = 0;
          __threadfence();
        }
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        if (warp_id == 0 && lane_id == 0)
          ptx::st_release_gpu(params.block_ready + blk, 1);
      } else {
        ptx::mbar_wait_parity(&smem.drain_done, drain_phase);
        drain_phase ^= 1;
        ptx::bar_sync(BAR_MATH, kNumMathThreads);
        if (warp_id == 0 && lane_id == 0)
          finish_piece(params, blk, t1 - t0, splits);
      }
      ptx::fence_proxy_async_shared();
      ptx::mbar_arrive(&smem.empty_q[qs]);
    }
    tcgen05::fence_before_thread_sync();
    ptx::bar_sync(BAR_MATH, kNumMathThreads);
    if (warp_id == 0) {
      tcgen05::fence_after_thread_sync();
      tcgen05::dealloc(tmem_base, kNumTmemCols);
    }
  }

  // One interval of drain `g`: the 4 math warps' slots of each row appended to
  // the row's slab with one atomic on the row's counter. Lanes 8r..8r+7 serve
  // row r (lanes past the warpgroup's rows idle); lane q of the octet holds the
  // masks of groups (warp, tile, half) 2q and 2q + 1 (16 groups per row: 4
  // warps x 2 tiles x 2 halves) and stores their set lanes in order.
  FMT_DEVICE static void drain_interval(
      const Fp8IndexerParams& params, Smem& smem, uint32_t g, uint32_t bank,
      uint32_t tiles_here,  // tiles of the interval (the block's last interval
                            // may hold fewer)
      uint32_t
          key_base,  // key of the interval's first tile, minus the row's ks
      uint32_t row,  // this lane's row
      bool piece,    // a tail piece: the row's counter is global (the block's
                     // pieces share it)
      uint32_t lane_id) {
    const uint32_t lane_r = lane_id >> 3, q = lane_id & 7u;
    const bool row_lane =
        kRowsPerWG == 4 || lane_r < kRowsPerWG;  // a constant at 4 rows
    const auto group_mask =
        [&](uint32_t grp) {  // grp = (w * kInterval + t) * 2 + h
          const uint32_t w = grp >> 2, t = (grp >> 1) & 1u, hh = grp & 1u;
          const bool valid = row_lane && t < tiles_here;
          return valid ? smem.masks[bank][g * 4 + w][t][hh][lane_r] : 0u;
        };
    const uint32_t m0 = group_mask(2 * q), m1 = group_mask(2 * q + 1);
    if constexpr (kRecovery) {
      float* const scores =
          params.recovery_scores +
          (static_cast<int64_t>(params.recovery_slot) * kBlockQ +
           row % kBlockQ) *
              params.recovery_stride;
      const uint32_t* const slots =
          &smem.recs[bank][g * 4 + (q >> 1)][lane_r][(q & 1) * 64];
      const uint32_t key0 = key_base + (q & 1) * kSplitKV + (q >> 1) * 32;
      uint32_t masks[2] = {m0, m1};
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        for (uint32_t mask = masks[half]; mask; mask &= mask - 1) {
          const uint32_t bit = __ffs(mask) - 1;
          scores[key0 + half * 128 + bit] =
              __uint_as_float(slots[half * 32 + bit]);
        }
      }
      return;
    }
    const uint32_t c0 = __popc(m0), c1 = __popc(m1);
    // Inclusive scan of the group counts over the octet, total from its last
    // lane.
    uint32_t incl = c0 + c1;
#pragma unroll
    for (uint32_t d = 1; d < 8; d <<= 1) {
      const uint32_t u = __shfl_up_sync(warp::kFullMask, incl, d);
      if (q >= d) incl += u;
    }
    const uint32_t total = __shfl_sync(warp::kFullMask, incl, lane_id | 7u);
    uint32_t rstart = 0;
    if (q == 0 && total != 0) {
      rstart = piece ? atomicAdd(
                           reinterpret_cast<uint32_t*>(params.cand_count) + row,
                           total)
                     : atomicAdd(&smem.cnt[row % kBlockQ], total);
    }
    rstart = __shfl_sync(warp::kFullMask, rstart, lane_id & ~7u);
    // Groups 2q and 2q + 1 are the two halves of one (warp, tile): adjacent
    // slots, keys 128 apart. Only the entries that fit the slab are written, so
    // the loop needs no per-entry check.
    const uint32_t pos = rstart + incl - c0 - c1;
    const uint32_t n = min(c0 + c1, pos < params.cap ? params.cap - pos : 0u);
    const auto off = static_cast<int64_t>(row) * params.cap + pos;
    float* const score_ptr = params.cand_scores + off;
    int32_t* const index_ptr = params.cand_indices + off;
    const uint32_t* const slots =
        &smem.recs[bank][g * 4 + (q >> 1)][lane_r][(q & 1) * 64];
    const uint32_t key0 = key_base + (q & 1) * kSplitKV + (q >> 1) * 32;
    uint32_t lo = m0, hi = m1;
    for (uint32_t i = 0; i < n; ++i) {
      const bool first = lo != 0;
      const uint32_t mk = first ? lo : hi;
      const uint32_t bit = __ffs(mk) - 1;
      const uint32_t rest = mk & (mk - 1);
      lo = first ? rest : lo;
      hi = first ? hi : rest;
      score_ptr[i] = __uint_as_float(slots[first ? bit : bit + 32]);
      index_ptr[i] = static_cast<int32_t>(key0 + (first ? bit : bit + 128));
    }
  }

  // Drain warp `g` (warp 10 + g): the slots of warpgroup g's rows; drain 1 also
  // writes the linear-term rows of the blocks the math warps did not (the CTA's
  // first block, one after an empty block).
  FMT_DEVICE static void drain_warp(const Fp8IndexerParams& params, Smem& smem,
                                    uint32_t g, uint32_t lane_id) {
    // setmaxnreg is warpgroup-collective: these warps take part too.
    ptx::setmaxnreg_dec<kSpecRegs>();
    Ring<kNumQStages> q_pipe;
    uint32_t epoch = 0;
    for (uint32_t ui = 0;; ++ui) {
      const Unit u = unit_at(params, ui);
      if (!u.valid) break;
      if (!u.live()) continue;
      const BlockInfo& b = u.b;
      if (g == 1) {
        const auto [qs, qph] = q_pipe.advance();
        if (ui == 0 || !unit_at(params, ui - 1).live()) {
          ptx::mbar_wait_parity(&smem.full_q[qs], qph);
          write_lin_rows(smem, qs, lane_id, 0, kBlockQ);
          ptx::mbar_arrive(&smem.full_lin[qs]);
        }
      }
      const uint32_t rb = g * kRowsPerWG + (lane_id >> 3);
      const uint32_t ks_r = select_at(b.ks, rb);
      for (uint32_t base = u.t0; base < u.t1; base += kInterval, ++epoch) {
        const uint32_t bank = epoch & 1, par = (epoch >> 1) & 1;
        ptx::mbar_wait_parity(&smem.h_full[bank][g], par);
        drain_interval(params, smem, g, bank, min(kInterval, u.t1 - base),
                       b.kv_lo + base * kSplitKV - ks_r, b.blk * kBlockQ + rb,
                       u.kind == UnitKind::PIECE, lane_id);
        __syncwarp();
        ptx::mbar_arrive(&smem.h_empty[bank][g]);
      }
      if (u.t1 > u.t0) {
        __syncwarp();
        if (lane_id == 0) ptx::mbar_arrive(&smem.drain_done);
      }
    }
  }

  FMT_DEVICE static void run(const Fp8IndexerParams& params,
                             const Fp8IndexerTensorMaps& tmaps, Smem& smem) {
    const uint32_t warp_id =
        __shfl_sync(warp::kFullMask, threadIdx.x / kWarpThreads, 0);
    const uint32_t lane_id = get_lane_id();
    enum WarpRole : uint32_t {
      WARP_LOAD = kNumMathWarps + 0,
      WARP_MMA = kNumMathWarps + 1,
      WARP_DRAIN_START = kNumMathWarps + 2,
    };

    if (warp_id == 8 && lane_id == 0) {
      ptx::prefetch_tensormap(&tmaps.q);
      ptx::prefetch_tensormap(&tmaps.weights);
      ptx::prefetch_tensormap(&tmaps.kv);
      ptx::prefetch_tensormap(&tmaps.sf_kv);
      ptx::prefetch_tensormap(&tmaps.kv_strided[0]);
      ptx::prefetch_tensormap(&tmaps.kv_strided[1]);
    } else if (warp_id == 9 && lane_id == 0) {
#pragma unroll
      for (uint32_t i = 0; i < kNumQStages; ++i) {
        ptx::mbar_init(&smem.full_q[i], 1);
        ptx::mbar_init(&smem.empty_q[i], kNumMathThreads + 1);
        ptx::mbar_init(&smem.full_lin[i], 32);
      }
#pragma unroll
      for (uint32_t i = 0; i < kNumKVStages; ++i) {
        ptx::mbar_init(&smem.full_kv[i], 32);
        ptx::mbar_init(&smem.empty_kv[i], kNumMathThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < kNumTmemStages; ++i) {
        ptx::mbar_init(&smem.full_tmem[i], 1);
        ptx::mbar_init(&smem.empty_tmem[i], 128);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        ptx::mbar_init(&smem.h_full[i / 2][i % 2], 128);
        ptx::mbar_init(&smem.h_empty[i / 2][i % 2], 32);
      }
      ptx::mbar_init(&smem.drain_done, 2);
      ptx::fence_barrier_init();
    } else if (warp_id == 10) {
      tcgen05::alloc(&smem.tmem_base, kNumTmemCols);
      tcgen05::fence_before_thread_sync();
    }
    __syncthreads();
    tcgen05::fence_after_thread_sync();
    const uint32_t tmem_base = smem.tmem_base;

    // Every role walks the same unit sequence (unit_at) and the same tile
    // sequence per unit: the sample tiles, then the scan tiles.
    if (warp_id == WARP_LOAD) {
      load_warp(params, tmaps, smem, lane_id);
    } else if (warp_id == WARP_MMA) {
      mma_warp(params, smem, tmem_base);
    } else if (warp_id < kNumMathWarps) {
      math_warps(params, smem, tmem_base, warp_id, lane_id);
    } else {
      drain_warp(params, smem, warp_id - WARP_DRAIN_START, lane_id);
    }
  }
};

template <typename Impl>
__global__ void __launch_bounds__(Impl::kNumThreads, 1)
    fp8_indexer_kernel(const Fp8IndexerParams params,
                       const __grid_constant__ Fp8IndexerTensorMaps tmaps) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  Impl::run(params, tmaps,
            *reinterpret_cast<typename Impl::Smem*>(smem_buffer));
}

// Each failed query block gets its own CTA and borrows a bounded scratch slot.
// Healthy blocks never initialize the TMA/MMA pipeline or touch the scratch.
template <typename Impl>
__global__ void __launch_bounds__(Impl::kNumThreads, 1)
    fp8_recovery_kernel(const Fp8IndexerParams params,
                        const __grid_constant__ Fp8IndexerTensorMaps tmaps) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  const uint32_t block = blockIdx.x, row = block * Impl::kBlockQ + threadIdx.x;
  bool bad = false;
  if (threadIdx.x < Impl::kBlockQ && row < params.num_rows) {
    bad = Impl::needs_recovery(params, row, Impl::visible_keys(params, row));
  }
  if (!__syncthreads_or(bad)) return;
  uint32_t zero_rows = 0;
  bool needs_scoring = false;
  for (uint32_t r = 0; r < Impl::kBlockQ; ++r) {
    const uint32_t row_id = block * Impl::kBlockQ + r;
    if (row_id >= params.num_rows) break;
    const uint32_t length = Impl::visible_keys(params, row_id);
    if (!Impl::needs_recovery(params, row_id, length)) continue;
    constexpr uint32_t heads = 256 / Impl::kBlockQ;
    const bool nonzero =
        threadIdx.x < heads &&
        __ldg(params.recovery_weights + static_cast<int64_t>(row_id) * heads +
              threadIdx.x) != 0.0f;
    if (!__syncthreads_or(nonzero)) {
      // For finite operands, zero head weights prove that every score is zero.
      for (uint32_t i = threadIdx.x; i < params.topk; i += blockDim.x) {
        params.recovery_out[static_cast<int64_t>(row_id) * params.topk + i] =
            i < length ? static_cast<int32_t>(i) : -1;
      }
      zero_rows |= 1u << r;
    } else {
      needs_scoring = true;
    }
  }
  if (!needs_scoring) return;
  if (threadIdx.x == 0) {
    *reinterpret_cast<uint32_t*>(smem_buffer) = litetopk_recovery::acquire_slot(
        params.block_ready, params.recovery_slots);
  }
  __syncthreads();
  const uint32_t slot = *reinterpret_cast<uint32_t*>(smem_buffer);
  Fp8IndexerParams p = params;
  p.recovery_block = block;
  p.recovery_slot = slot;
  Impl::run(p, tmaps, *reinterpret_cast<typename Impl::Smem*>(smem_buffer));
  __threadfence();
  __syncthreads();
  for (uint32_t r = 0; r < Impl::kBlockQ; ++r) {
    const uint32_t row_id = block * Impl::kBlockQ + r;
    if (row_id >= params.num_rows) break;
    const uint32_t length = Impl::visible_keys(params, row_id);
    if (!(zero_rows & (1u << r)) &&
        Impl::needs_recovery(params, row_id, length)) {
      litetopk_recovery::exact_topk<Impl>(
          params.recovery_scores +
              (static_cast<int64_t>(slot) * Impl::kBlockQ + r) *
                  params.recovery_stride,
          nullptr, length, params.topk,
          params.recovery_out + static_cast<int64_t>(row_id) * params.topk,
          *reinterpret_cast<litetopk_recovery::Scratch*>(smem_buffer));
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    litetopk_recovery::release_slot(params.block_ready, slot);
}

// The producer (kRecovery false) and recovery instances fp8_indexer.cu builds
// and fused_mqa_topk.cu launches: 3 KV stages, 2 Q stages.
template <uint32_t kHeads, bool kRecovery>
using Fp8Impl = Fp8IndexerImpl<kHeads, 3, 2, kRecovery>;

}  // namespace device::fp8_indexer

}  // namespace fused_mqa_topk
