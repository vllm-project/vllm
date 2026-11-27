// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from SGLang (Apache-2.0):
// https://github.com/sgl-project/sglang/blob/656331295619a4d0267b40b7da0b5cd5b549fcbc/python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh
// which adapts DeepSelect's init-window select
// (https://github.com/deepseek-ai/DeepSelect, MIT, see below).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The candidate consumers' top-k and remap (sparse_topk_small.py). Changes from
// SGLang: the sgl_kernel helpers it uses are inlined, PDL is always on, and the
// host entry takes torch tensors. Warps take their output slots from a prefix
// in warp order instead of shared-memory atomics in arrival order, and the
// thread at the tie quota keeps its lowest-numbered ties, so the output is
// deterministic: the selected columns in ascending order, ties broken toward
// the lowest column.

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>

#include <cstdint>

namespace {

using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kFullMask = 0xFFFFFFFFu;

/**
 * \brief bf16 top-k of one row that fits in registers (rows of at most 16384
 *        scores: the DeepSeek-V4.1 sparse indexer's consumer rows), fused with
 *        a page-table transform.
 *
 * One CTA of 512 threads per row, up to 32 scores per thread held in registers.
 * Two radix passes over the raw bf16 bytes locate the k-th largest value
 * exactly, a census places every element relative to it, and the selected
 * indices are staged in shared memory before one coalesced, page-transformed
 * store. This is DeepSelect's init-window select.
 *
 * \note The value order used everywhere is the "distorted" order of the raw
 *       bf16 bits (`x ^ (x < 0 ? 0xFFFF : 0x8000)`, negatives below positives,
 *       -0 below +0). The histograms are indexed by the *raw* byte instead, and
 *       the pivot search undoes the permutation once per lane, so no element
 *       pays the distortion.
 * \note NaN scores are not selected: the ordered compares never match them, so
 *       a row with n positive NaNs yields its top (k - n) real scores and -1 in
 *       the remaining slots (a negative NaN orders below -inf and is simply
 *       never picked).
 */
struct TopKBF16Config {
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  static constexpr uint32_t kOccupancy = 3;
  static constexpr uint32_t kVecSize = 8;
  static constexpr uint32_t kMaxVecs = 4;
  static constexpr uint32_t kElemsPerThread = kVecSize * kMaxVecs;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kElemsPerThread;
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kNumBins = 256;
  // the LSB pass sends out-of-bucket elements here
  static constexpr uint32_t kSinkBin = kNumBins;
  /// NOTE: in the MSB row the negative half lives 16 words further up. Raw
  /// bytes 128 apart share a bank, so without it +x and -x with the same
  /// exponent (the common case for centered data) collide on every histogram
  /// update; measured as half of all atomic wavefronts.
  static constexpr uint32_t kNegShift = 16;
  // keeps both rows 16 B aligned
  static constexpr uint32_t kHistStride = kNumBins + kNegShift + 4;
  /// NOTE: a negative NaN. In the distorted order it sits below -inf, and every
  /// ordered bf16 comparison against it is false, so padding is never counted
  /// nor selected.
  static constexpr uint32_t kPadElem = 0xFFFFu;
  static constexpr uint32_t kNegZeroBits = 0x8000u;
  // the census counters pack in 16 bits
  static_assert(kMaxSeqLen == 16384 && kMaxSeqLen <= 0xFFFF);
  // one census bit per element of the slice, in a uint32_t
  static_assert(kElemsPerThread == 32);

  struct alignas(16) Vec {
    bf16x2_t pairs[kVecSize / 2];
  };

  struct Smem {
    // each warp's census, packed (gt << 16 | eq)
    uint32_t warp_gt_eq[kNumWarps];
    uint32_t pivot_bin;
    uint32_t pivot_remain;
    union {
      alignas(16) uint32_t histogram[2][kHistStride];
      alignas(16) uint32_t stage[kMaxTopK];
    };
  };
};

struct TopKBF16Params {
  const bf16_t* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  int64_t page_indices_stride;
  uint32_t topk;
  uint32_t page_bits;
};

__device__ __forceinline__ void pdl_wait_primary() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_trigger_secondary() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t x,
                                                       uint32_t lane_id) {
#pragma unroll
  for (uint32_t offset = 1; offset < kWarpThreads; offset <<= 1) {
    const auto n = __shfl_up_sync(kFullMask, x, offset);
    if (lane_id >= offset) x += n;
  }
  return x;
}

/// \brief Exclusive suffix scan: lane `L` gets the sum over lanes `> L`.
__device__ __forceinline__ uint32_t
warp_exclusive_suffix_sum(uint32_t x, uint32_t lane_id) {
  uint32_t inc = x;
#pragma unroll
  for (uint32_t offset = 1; offset < kWarpThreads; offset <<= 1) {
    const auto t = __shfl_down_sync(kFullMask, inc, offset);
    if (lane_id + offset < kWarpThreads) inc += t;
  }
  return inc - x;
}

template <typename To, typename From>
__device__ __forceinline__ To bitcast(const From& f) {
  static_assert(sizeof(From) == sizeof(To));
  return reinterpret_cast<const To&>(f);
}

struct TopKBF16Pivot {
  uint32_t bin;     // in distorted (value-ascending) order, [0, 256)
  uint32_t remain;  // how many elements of `bin` still have to be taken
};

/// Locate the bin holding the k-th largest element in a 256-bin histogram
/// indexed by a raw byte; one warp, exactly one lane writes `smem.pivot_*`.
/// `msb_mode`: lanes < 16 cover raw 0xFF..0x80 (negatives, reversed), lanes >=
/// 16 raw 0x00..0x7F. `negative` (LSB mode only): the pivot bucket is negative,
/// so the whole byte is reversed.
__device__ __forceinline__ void topk_bf16_find_pivot_warp(
    const uint32_t* hist, uint32_t k, bool msb_mode, bool negative,
    uint32_t lane_id, TopKBF16Config::Smem& smem) {
  using C = TopKBF16Config;
  // lane L owns distorted bins [8L, 8L + 8)
  const bool reverse = msb_mode ? lane_id < 16 : negative;
  uint32_t raw_base =
      reverse ? 0xF8 - 8 * lane_id : 8 * lane_id - (msb_mode ? 0x80 : 0);
  if (msb_mode && reverse) raw_base += C::kNegShift;
  const uint4 lo = *reinterpret_cast<const uint4*>(hist + raw_base);
  const uint4 hi = *reinterpret_cast<const uint4*>(hist + raw_base + 4);
  const uint32_t fwd[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  uint32_t count[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    count[i] = reverse ? fwd[7 - i] : fwd[i];
  }
  uint32_t local = 0;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    local += count[i];
  }
  // suffix[j] = number of elements in bins >= 8L + j
  uint32_t suffix[9];
  suffix[8] = warp_exclusive_suffix_sum(local, lane_id);
#pragma unroll
  for (int32_t j = 7; j >= 0; --j) {
    suffix[j] = suffix[j + 1] + count[j];
  }
  // exactly one lane satisfies suffix[8] < k <= suffix[0]; inside it, the pivot
  // is the largest offset j with suffix[j] >= k
  const bool found = suffix[8] < k && k <= suffix[0];
  uint32_t offset = 0;
  uint32_t next = suffix[1];
#pragma unroll
  for (uint32_t j = 1; j < 8; ++j) {
    if (suffix[j] >= k) {
      offset = j;
      next = suffix[j + 1];
    }
  }
  if (found) {
    smem.pivot_bin = 8 * lane_id + offset;
    smem.pivot_remain = k - next;
  }
}

/// One hit bit per element of an 8-wide vector, from the per-pair 16-bit masks
/// of `__hgt2_mask` and friends.
__device__ __forceinline__ uint32_t
topk_bf16_pack_hits(const uint32_t (&m)[4]) {
  // one flag byte per element (0xFF / 0x00), then signed dot products with
  // -1, -2, -4, -8 and -16 .. -128 turn them into bits
  const auto lo = __byte_perm(m[0], m[1], 0x7531);
  const auto hi = __byte_perm(m[2], m[3], 0x7531);
  const auto nib =
      __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);
  return __dp4a(static_cast<int>(hi), static_cast<int>(0x80C0E0F0u), nib);
}

__global__ void __launch_bounds__(TopKBF16Config::kBlockSize,
                                  TopKBF16Config::kOccupancy)
    topk_bf16_small_kernel(const __grid_constant__ TopKBF16Params params) {
  using C = TopKBF16Config;
  using vec_t = C::Vec;
  __shared__ C::Smem smem;

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = tx / kWarpThreads;
  const auto topk = params.topk;
  // a selected index i maps through this row's table to slot
  // table[i >> page_bits] << page_bits | (i & mask); -1 past what the row has
  const auto* __restrict__ table =
      params.page_table + bx * params.page_table_stride;
  auto* __restrict__ out =
      params.page_indices + bx * params.page_indices_stride;
  const auto page_bits = params.page_bits;
  const auto page_mask = (1u << page_bits) - 1;
  const auto transform = [&](uint32_t idx) -> int32_t {
    return (table[idx >> page_bits] << page_bits) |
           static_cast<int32_t>(idx & page_mask);
  };

  {
    constexpr uint32_t kZeroVecs = sizeof(smem.histogram) / sizeof(uint4);
    static_assert(sizeof(smem.histogram) % sizeof(uint4) == 0);
#pragma unroll
    for (uint32_t idx = tx; idx < kZeroVecs; idx += C::kBlockSize) {
      reinterpret_cast<uint4*>(smem.histogram)[idx] = make_uint4(0, 0, 0, 0);
    }
  }

  // The row lengths are written by the candidate expansion, which completes
  // before the logits kernel this one waits on starts, so they load early.
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto* __restrict__ scores_row =
      params.scores + bx * params.score_stride;
  if (seq_len <= topk) {  // every element is selected, -1 past the row
    pdl_wait_primary();
    for (uint32_t t = tx; t < topk; t += C::kBlockSize) {
      out[t] = t < seq_len ? transform(t) : -1;
    }
    return pdl_trigger_secondary();
  }
  pdl_wait_primary();

  // Contiguous slices of whole vectors, balanced so short rows still spread
  // over the block. Only the last vector of a row can be partial; it is padded
  // with NaNs (see kPadElem).
  const uint32_t num_vecs = (seq_len + C::kVecSize - 1) / C::kVecSize;
  const uint32_t num_full = seq_len / C::kVecSize;
  const uint32_t vecs_per_thread = num_vecs / C::kBlockSize;
  const uint32_t vecs_rem = num_vecs % C::kBlockSize;
  const uint32_t vec_start = tx * vecs_per_thread + min(tx, vecs_rem);
  const uint32_t num_my = vecs_per_thread + (tx < vecs_rem ? 1 : 0);
  vec_t vecs[C::kMaxVecs];
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
    const auto v = vec_start + i;
    if (v < num_full) {
      const uint4 raw = reinterpret_cast<const uint4*>(scores_row)[v];
      vecs[i] = bitcast<vec_t>(raw);
    } else {
      const auto* ptr =
          reinterpret_cast<const uint16_t*>(scores_row) + v * C::kVecSize;
      const auto n = seq_len - v * C::kVecSize;  // in [1, kVecSize)
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        vecs[i].pairs[j].x =
            bitcast<bf16_t>(2 * j + 0 < n ? ptr[2 * j + 0]
                                          : static_cast<uint16_t>(C::kPadElem));
        vecs[i].pairs[j].y =
            bitcast<bf16_t>(2 * j + 1 < n ? ptr[2 * j + 1]
                                          : static_cast<uint16_t>(C::kPadElem));
      }
    }
  }

  __syncthreads();

  // Pass 1: histogram of the raw high byte (sign + 7 exponent bits)
  const auto hist_msb = smem.histogram[0];
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      const auto raw = bitcast<uint32_t>(vecs[i].pairs[j]);
      /// NOTE: spelled as byte extraction so the address is one PRMT + one
      /// LEA per element
      const auto b0 = __byte_perm(raw, 0, 0x4441);
      const auto b1 = __byte_perm(raw, 0, 0x4443);
      atomicAdd(hist_msb + b0 + (b0 >> 7) * C::kNegShift, 1);
      atomicAdd(hist_msb + b1 + (b1 >> 7) * C::kNegShift, 1);
    }
  }
  __syncthreads();

  const auto pivot_of = [&](const uint32_t* hist, uint32_t k, bool msb_mode,
                            bool neg) -> TopKBF16Pivot {
    if (warp_id == 0)
      topk_bf16_find_pivot_warp(hist, k, msb_mode, neg, lane_id, smem);
    __syncthreads();
    return {smem.pivot_bin, smem.pivot_remain};
  };
  const auto msb = pivot_of(hist_msb, topk, true, false);
  const bool negative = msb.bin < 0x80;
  // the raw high byte
  const auto pivot_hi = negative ? 0xFF - msb.bin : msb.bin - 0x80;

  // Pass 2: among elements sharing the pivot's high byte, histogram the raw low
  // byte. The high bytes are compared as tiny positive bf16 values (exact), the
  // others land in the sink bin.
  const auto hist_lsb = smem.histogram[1];
  const auto pivot_hi_x2 = bitcast<bf16x2_t>(pivot_hi << 16 | pivot_hi);
  constexpr uint32_t kSinkBinX2 = C::kSinkBin << 16 | C::kSinkBin;
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      const auto raw = bitcast<uint32_t>(vecs[i].pairs[j]);
      const auto hi = __byte_perm(raw, 0, 0x5341);  // {byte1, 0, byte3, 0}
      const auto sel = __heq2_mask(bitcast<bf16x2_t>(hi), pivot_hi_x2);
      const auto lo = raw & 0x00FF00FFu;
      // sel ? lo : kSinkBin
      const auto bins = (sel & lo) | (~sel & kSinkBinX2);
      atomicAdd(hist_lsb + __byte_perm(bins, 0, 0x4410), 1);  // bins & 0xFFFF
      atomicAdd(hist_lsb + __byte_perm(bins, 0, 0x4432), 1);  // bins >> 16
    }
  }
  __syncthreads();

  const auto lsb = pivot_of(hist_lsb, msb.remain, false, negative);
  const uint32_t pivot_lo = negative ? 0xFF - lsb.bin : lsb.bin;
  const uint32_t pivot_bits = pivot_hi << 8 | pivot_lo;
  const auto pivot_x2 = bitcast<bf16x2_t>(pivot_bits << 16 | pivot_bits);

  // Census: one bit per element of the slice, in element order (vector i
  // fills byte i)
  uint32_t gt_mask = 0;
  uint32_t eq_mask = 0;
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
    uint32_t gt[4], eq[4];
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      gt[j] = __hgt2_mask(vecs[i].pairs[j], pivot_x2);
      eq[j] = __heq2_mask(vecs[i].pairs[j], pivot_x2);
    }
    // drop the new byte into slot i, keeping the other three
    constexpr uint32_t kInsert[4] = {0x3214, 0x3240, 0x3410, 0x4210};
    gt_mask = __byte_perm(gt_mask, topk_bf16_pack_hits(gt), kInsert[i]);
    eq_mask = __byte_perm(eq_mask, topk_bf16_pack_hits(eq), kInsert[i]);
  }
  const uint32_t cnt_gt = __popc(gt_mask);
  const uint32_t cnt_eq = __popc(eq_mask);

  // Block-wide exclusive prefix of (gt, eq), packed, in thread order: a warp
  // scan, then each warp sums the totals of the warps before it.
  const uint32_t local = cnt_gt << 16 | cnt_eq;
  const uint32_t warp_inc = warp_inclusive_sum(local, lane_id);
  if (lane_id == kWarpThreads - 1) smem.warp_gt_eq[warp_id] = warp_inc;
  __syncthreads();
  const uint32_t warp_total =
      lane_id < C::kNumWarps ? smem.warp_gt_eq[lane_id] : 0;
  const uint32_t totals = __reduce_add_sync(kFullMask, warp_total);
  const uint32_t warp_base =
      __reduce_add_sync(kFullMask, lane_id < warp_id ? warp_total : 0);
  const uint32_t before = warp_base + warp_inc - local;

  // Everything above the pivot is taken, plus `remain` of the elements equal
  // to it.
  uint32_t eq_total = lsb.remain;
  if (pivot_bits == C::kNegZeroBits) {
    /// NOTE: the census compares as floats, so a -0 pivot also sees +0 as equal
    /// while the histogram ranked +0 above it. Both are worth the same, so let
    /// the equal quota absorb them: the quota then has to come from the census
    /// total.
    eq_total = topk - (totals >> 16);
  }
  const uint32_t gt_before = before >> 16;
  const uint32_t eq_before = before & 0xFFFF;
  const uint32_t eq_start = min(eq_before, eq_total);
  const uint32_t eq_quota = min(eq_before + cnt_eq, eq_total) - eq_start;
  // keep only the lowest-numbered `eq_quota` of the equal bits
  if (eq_quota == 0) {
    eq_mask = 0;
  } else {
#pragma unroll 1
    for (uint32_t n = cnt_eq; n > eq_quota; --n) {
      eq_mask ^= 1u << (31 - __clz(eq_mask));
    }
  }

  uint32_t hits = gt_mask | eq_mask;
  auto* dst = smem.stage + gt_before + eq_start;
  const uint32_t elem_base = vec_start * C::kVecSize;
  while (hits != 0) {
    const auto e = __ffs(hits) - 1;
    hits &= hits - 1;
    *dst++ = elem_base + e;
  }

  pdl_trigger_secondary();
  __syncthreads();

  // Slots past the census total were never staged (only NaN scores cause that:
  // counted by the histogram, never selected); write -1 there.
  const uint32_t num_staged = (totals >> 16) + min(totals & 0xFFFFu, eq_total);
  for (uint32_t t = tx; t < topk; t += C::kBlockSize) {
    out[t] = t < num_staged ? transform(smem.stage[t]) : -1;
  }
}

void topk_bf16_small(const at::Tensor& scores, const at::Tensor& seq_lens,
                     const at::Tensor& page_table, at::Tensor& page_indices,
                     int64_t page_bits) {
  const TopKBF16Params params{
      .scores = reinterpret_cast<const bf16_t*>(scores.data_ptr()),
      .seq_lens = seq_lens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = page_indices.data_ptr<int32_t>(),
      .score_stride = scores.stride(0),
      .page_table_stride = page_table.stride(0),
      .page_indices_stride = page_indices.stride(0),
      .topk = static_cast<uint32_t>(page_indices.size(1)),
      .page_bits = static_cast<uint32_t>(page_bits),
  };
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<uint32_t>(scores.size(0)));
  config.blockDim = dim3(TopKBF16Config::kBlockSize);
  config.stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  config.attrs = &attr;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, topk_bf16_small_kernel, params));
}

}  // namespace

TORCH_LIBRARY(sd_sparse_topk_small, m) {
  m.def(
      "topk(Tensor scores, Tensor seq_lens, Tensor page_table, "
      "Tensor(a!) page_indices, int page_bits) -> ()",
      &topk_bf16_small);
}
