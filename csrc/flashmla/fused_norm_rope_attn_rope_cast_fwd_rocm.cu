// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include "rocm/ops.h"

namespace {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;

struct DenseKV {
  const c10::BFloat16* data;
  const int32_t* indices;
  const int32_t* lengths;
  int64_t stride, index_stride, rows, topk;

  __device__ int64_t length(int64_t token) const {
    int64_t n = lengths ? lengths[token] : topk;
    return n < 0 ? 0 : (n > topk ? topk : n);
  }
  __device__ bool load(int64_t token, int64_t i, int dim, float& even,
                       float& odd) const {
    const int32_t slot = indices[token * index_stride + i];
    if (slot < 0 || slot >= rows) return false;
    even = static_cast<float>(data[int64_t(slot) * stride + dim]);
    odd = static_cast<float>(data[int64_t(slot) * stride + dim + 1]);
    return true;
  }
};

__device__ float wave_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

template <typename Position, int Splits>
__global__ void fused_prefill_kernel(
    const c10::BFloat16* q, DenseKV kv, const float* sink,
    const Position* positions, const float* cos_sin, c10::Float8_e4m3fn* out,
    uint8_t* out_scale, int64_t q_stride_t, int64_t q_stride_h,
    int64_t out_stride, int64_t scale_stride, int64_t num_positions,
    float sm_scale, bool enable_q_norm, float eps, bool apply_q_rope) {
  constexpr int Pairs = kHeadDim / (2 * 64);
  const int lane = threadIdx.x % 64;
  const int split = threadIdx.x / 64;
  const int64_t token = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t position = positions[token];
  const bool valid_position = position >= 0 && position < num_positions;
  float qe[Pairs], qo[Pairs], cos[Pairs], sin[Pairs];
  float ae[Pairs] = {}, ao[Pairs] = {};
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < Pairs; ++j) {
    const int dim = 2 * lane + j * 128;
    qe[j] = float(q[token * q_stride_t + head * q_stride_h + dim]);
    qo[j] = float(q[token * q_stride_t + head * q_stride_h + dim + 1]);
    ss += qe[j] * qe[j] + qo[j] * qo[j];
    cos[j] = 1.0f;
    sin[j] = 0.0f;
    if (dim >= kHeadDim - kRopeDim && valid_position) {
      const int pair = (dim - (kHeadDim - kRopeDim)) / 2;
      cos[j] = cos_sin[position * kRopeDim + pair];
      sin[j] = cos_sin[position * kRopeDim + kRopeDim / 2 + pair];
    }
  }
  const float norm =
      enable_q_norm ? rsqrtf(wave_sum(ss) / kHeadDim + eps) : 1.0f;
#pragma unroll
  for (int j = 0; j < Pairs; ++j) {
    float even = qe[j] * norm, odd = qo[j] * norm;
    qe[j] = float(
        c10::BFloat16(apply_q_rope ? even * cos[j] - odd * sin[j] : even));
    qo[j] =
        float(c10::BFloat16(apply_q_rope ? odd * cos[j] + even * sin[j] : odd));
  }
  // Each wave owns a disjoint KV subsequence; the sink is counted once.
  float maximum = sink && split == 0 ? sink[head] : -INFINITY;
  float denominator = isfinite(maximum) ? 1.0f : 0.0f;
  const int64_t length = valid_position ? kv.length(token) : 0;
  for (int64_t i = split; i < length; i += Splits) {
    const int32_t slot = kv.indices[token * kv.index_stride + i];
    if (slot < 0 || slot >= kv.rows) continue;
    float ve[Pairs], vo[Pairs];
    float score = 0.0f;
#pragma unroll
    for (int j = 0; j < Pairs; ++j) {
      const int dim = 2 * lane + j * 128;
      ve[j] = float(kv.data[int64_t(slot) * kv.stride + dim]);
      vo[j] = float(kv.data[int64_t(slot) * kv.stride + dim + 1]);
      score += qe[j] * ve[j] + qo[j] * vo[j];
    }
    score = wave_sum(score) * sm_scale;
    const float next_max = fmaxf(maximum, score);
    const float alpha = denominator == 0.0f ? 0.0f : expf(maximum - next_max);
    const float beta = expf(score - next_max);
#pragma unroll
    for (int j = 0; j < Pairs; ++j) {
      ae[j] = ae[j] * alpha + ve[j] * beta;
      ao[j] = ao[j] * alpha + vo[j] * beta;
    }
    denominator = denominator * alpha + beta;
    maximum = next_max;
  }
  __shared__ float partial[Splits][kHeadDim];
  __shared__ float maxima[Splits], denominators[Splits];
#pragma unroll
  for (int j = 0; j < Pairs; ++j) {
    partial[split][2 * lane + j * 128] = ae[j];
    partial[split][2 * lane + j * 128 + 1] = ao[j];
  }
  if (lane == 0) {
    maxima[split] = maximum;
    denominators[split] = denominator;
  }
  __syncthreads();
  if (split != 0) return;
  maximum = -INFINITY;
#pragma unroll
  for (int s = 0; s < Splits; ++s) maximum = fmaxf(maximum, maxima[s]);
  float weights[Splits];
  denominator = 0.0f;
#pragma unroll
  for (int s = 0; s < Splits; ++s) {
    weights[s] = denominators[s] > 0.0f ? expf(maxima[s] - maximum) : 0.0f;
    denominator += denominators[s] * weights[s];
  }
#pragma unroll
  for (int j = 0; j < Pairs; ++j) {
    const int dim = 2 * lane + j * 128;
    float even = 0.0f, odd = 0.0f;
#pragma unroll
    for (int s = 0; s < Splits; ++s) {
      even += partial[s][dim] * weights[s];
      odd += partial[s][dim + 1] * weights[s];
    }
    even = float(c10::BFloat16(denominator > 0.0f ? even / denominator : 0.0f));
    odd = float(c10::BFloat16(denominator > 0.0f ? odd / denominator : 0.0f));
    const float re = even * cos[j] + odd * sin[j];
    const float ro = odd * cos[j] - even * sin[j];
    float amax = fmaxf(fabsf(re), fabsf(ro));
    for (int offset = 8; offset > 0; offset /= 2) {
      amax = fmaxf(amax, __shfl_xor(amax, offset, 16));
    }
    const float biased = ceilf(log2f(fmaxf(amax, 0x1p-126f) / 448.0f)) + 127.0f;
    const uint8_t bits =
        static_cast<uint8_t>(fminf(fmaxf(biased, 0.0f), 254.0f));
    const float reciprocal = exp2f(127.0f - bits);
    out[token * out_stride + head * kHeadDim + dim] =
        c10::Float8_e4m3fn(fminf(fmaxf(re * reciprocal, -448.0f), 448.0f));
    out[token * out_stride + head * kHeadDim + dim + 1] =
        c10::Float8_e4m3fn(fminf(fmaxf(ro * reciprocal, -448.0f), 448.0f));
    if (lane % 16 == 0) {
      out_scale[token * scale_stride + head * (kHeadDim / 32) + dim / 32] =
          bits;
    }
  }
}

using BF16x8 = __bf16 __attribute__((ext_vector_type(8)));
using BF16x4 = __bf16 __attribute__((ext_vector_type(4)));
using BF16x2 = __bf16 __attribute__((ext_vector_type(2)));
using F4 = float __attribute__((ext_vector_type(4)));
using Short4 = short __attribute__((ext_vector_type(4)));
using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
using U32x2 = uint32_t __attribute__((ext_vector_type(2)));
using I32x4 = int32_t __attribute__((ext_vector_type(4)));

// Keep paged loads in the global address space so they only count against
// vmcnt; flat loads would also stall every LDS wait behind the prefetch.
template <typename T>
using Global = const T __attribute__((address_space(1)))*;
template <typename T>
using Local = T __attribute__((address_space(3)))*;

__device__ __forceinline__ F4 mfma_k32(BF16x8 a, BF16x8 b, F4 c) {
#if defined(__gfx950__)
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
#else
  c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
      __builtin_bit_cast(Short4, a.lo), __builtin_bit_cast(Short4, b.lo), c, 0,
      0, 0);
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
      __builtin_bit_cast(Short4, a.hi), __builtin_bit_cast(Short4, b.hi), c, 0,
      0, 0);
#endif
}

__device__ __forceinline__ F4 mfma_k16(BF16x4 a, BF16x4 b, F4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
      __builtin_bit_cast(Short4, a), __builtin_bit_cast(Short4, b), c, 0, 0, 0);
}

struct PagedCache {
  const uint8_t* data = nullptr;
  const int32_t* table = nullptr;
  int64_t block_stride = 0, table_stride = 0;
  int block_size = 1, blocks = 0, table_width = 0;
  bool fnuz = false;

  __device__ int rows() const { return blocks * block_size; }
  // Maps a request-local position through the block table to a cache slot.
  // The table read is unconditional so batched lookups stay in flight.
  __device__ int lookup(int request, int logical) const {
    const int index = logical / block_size;
    const int block = table[int64_t(request) * table_stride +
                            max(0, min(index, table_width - 1))];
    return logical < 0 || index >= table_width || block < 0 || block >= blocks
               ? -1
               : block * block_size + logical % block_size;
  }
};

// Keys [0, split) come from `first`, keys [split, length) from `second`.
struct KeySpan {
  int length = 0, split = 0, first_begin = 0, second_begin = 0, request = 0,
      seq = 0;
};

struct DecodeSource {
  PagedCache first, second;
  const int32_t *first_indices = nullptr, *first_indptr = nullptr;
  const int32_t *second_indices = nullptr, *second_indptr = nullptr;
  int64_t first_capacity = 0, second_capacity = 0;

  __device__ KeySpan span(int64_t token) const {
    auto clamp = [](int64_t v, int64_t hi) {
      return int(max(int64_t(0), min(v, hi)));
    };
    KeySpan s;
    s.first_begin = clamp(first_indptr[token], first_capacity);
    s.split =
        max(clamp(first_indptr[token + 1], first_capacity) - s.first_begin, 0);
    s.length = s.split;
    if (second_indptr) {
      s.second_begin = clamp(second_indptr[token], second_capacity);
      s.length += max(
          clamp(second_indptr[token + 1], second_capacity) - s.second_begin, 0);
    }
    return s;
  }
  __device__ int prefetch(int64_t, int) const { return 0; }
  __device__ int logical(int64_t, const KeySpan& s, int i, int) const {
    return *(i < s.split ? first_indices + s.first_begin + i
                         : second_indices + s.second_begin + i - s.split);
  }
  __device__ int resolve(const KeySpan& s, int i, int slot) const {
    return slot >= 0 && slot < (i < s.split ? first : second).rows() ? slot
                                                                     : -1;
  }
};

// Prefill attends to causal top-k compressed rows, then its SWA window.
struct PrefillSource {
  PagedCache first, second;
  const int32_t *topk = nullptr, *starts = nullptr, *seq_lens = nullptr,
                *gather_lens = nullptr;
  int64_t topk_stride = 0;
  int requests = 0, topk_width = 0, window = 0, compress_ratio = 0;

  __device__ KeySpan span(int64_t token) const {
    const int64_t global_token = token + starts[0];
    int request = 0, hi = requests;
    while (request + 1 < hi) {
      const int mid = (request + hi) / 2;
      if (starts[mid] <= global_token)
        request = mid;
      else
        hi = mid;
    }
    KeySpan s;
    s.request = request;
    s.seq = seq_lens[request];
    const int pos = s.seq - (starts[request + 1] - starts[request]) +
                    (global_token - starts[request]);
    s.split = compress_ratio > 0
                  ? max(0, min((pos + 1) / compress_ratio, topk_width))
                  : 0;
    s.second_begin =
        max(max(pos - window + 1, 0), s.seq - gather_lens[request]);
    s.length = global_token < starts[requests]
                   ? s.split + max(pos - s.second_begin + 1, 0)
                   : 0;
    return s;
  }
  // The top-k read only depends on the token, so it overlaps span().
  __device__ int prefetch(int64_t token, int i) const {
    return topk_width > 0 ? topk[token * topk_stride + min(i, topk_width - 1)]
                          : -1;
  }
  __device__ int logical(int64_t, const KeySpan& s, int i, int index) const {
    if (i >= s.split) return s.second_begin + i - s.split;
    return index < s.seq / compress_ratio ? index : -1;
  }
  __device__ int resolve(const KeySpan& s, int i, int logical) const {
    return (i < s.split ? first : second).lookup(s.request, logical);
  }
};

// One lane's share of a 584-byte record: 16 NoPE FP8 bytes from each of the
// seven 64-dim scale groups, 16 RoPE dims, and the row's E8M0 scales.
struct RowFragment {
  U32x4 nope[7] = {};
  U32x4 rope[2] = {};
  U32x2 scales = {};
  bool fnuz = false;
};

template <bool Aligned, typename T>
__device__ __forceinline__ T load_global(const uint8_t* address) {
  if constexpr (Aligned) {
    return *(Global<T>)address;
  } else {
    uint8_t bytes[sizeof(T)];
#pragma unroll
    for (int i = 0; i < int(sizeof(T)); ++i)
      bytes[i] = ((Global<uint8_t>)address)[i];
    return __builtin_bit_cast(T, bytes);
  }
}

// Invalid keys read this record instead of branching, so every tile issues
// the same loads and the compiler can wait on one prefetch at a time.
__device__ __align__(16) uint8_t kZeroRecord[592];

template <bool Aligned>
__device__ __forceinline__ RowFragment load_row(const PagedCache& cache,
                                                int slot, int part) {
  RowFragment f;
  const bool valid = slot >= 0;
  const int offset = valid ? slot % cache.block_size : 0;
  const uint8_t* page =
      cache.data +
      int64_t(valid ? slot / cache.block_size : 0) * cache.block_stride;
  const uint8_t* row = valid ? page + offset * 576 : kZeroRecord;
  const uint8_t* scales =
      valid ? page + cache.block_size * 576 + offset * 8 : kZeroRecord + 576;
#pragma unroll
  for (int g = 0; g < 7; ++g)
    f.nope[g] = load_global<Aligned, U32x4>(row + g * 64 + part * 16);
#pragma unroll
  for (int h = 0; h < 2; ++h)
    f.rope[h] = load_global<Aligned, U32x4>(row + 448 + part * 32 + h * 16);
  f.scales = load_global<Aligned, U32x2>(scales);
  f.fnuz = cache.fnuz;
  return f;
}

template <bool AnyFnuz>
__device__ __forceinline__ BF16x8 dequant_eight(U32x2 bits, float scale,
                                                bool fnuz) {
  BF16x8 out;
#if defined(__gfx950__)
  if (!AnyFnuz || !fnuz) {
    const BF16x2 a =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits.x, scale, false);
    const BF16x2 b =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits.x, scale, true);
    const BF16x2 c =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits.y, scale, false);
    const BF16x2 d =
        __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits.y, scale, true);
    return BF16x8{a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y};
  }
#endif
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const uint8_t byte = (e < 4 ? bits.x : bits.y) >> (8 * (e % 4));
    float value;
    if (AnyFnuz && fnuz) {
      c10::Float8_e4m3fnuz fp8;
      fp8.x = byte;
      value = float(fp8);
    } else {
      c10::Float8_e4m3fn fp8;
      fp8.x = byte;
      value = float(fp8);
    }
    out[e] = static_cast<__bf16>(value * scale);
  }
  return out;
}

constexpr int kWaves = 4;
constexpr int kTileKeys = 16;
constexpr int kMaxKeys = 1024;
constexpr int kKStride = kHeadDim;
constexpr int kOStride = kHeadDim + 4;
constexpr int kKBytes = kWaves * 2 * kTileKeys * kKStride * 2;
constexpr int kOBytes = 2 * 16 * kOStride * 4;
constexpr int kStageBytes = kKBytes > kOBytes ? kKBytes : kOBytes;

// Staged K rows permute their 16-byte chunks within each 128-byte bank line:
// b128 writes of rows 0..7 and transposed b64 reads of rows 4*lg + [0, 4)
// then hit distinct banks.
__device__ __forceinline__ int swizzle_chunk(int row, int chunk) {
  return chunk ^ ((row % 4) * 2 + (row / 4) % 2);
}

// One CTA owns a query and sixteen heads. Its four waves split 32-key steps
// and run independent online softmaxes, so the main loop has no barrier:
// S^T = K Q^T reads K straight from registers, P^T stays in the MFMA output
// layout, and each wave stages its own K rows in LDS for a transposed V read.
template <typename Position, typename Source, bool RoundAttention, bool Aligned,
          bool AnyFnuz>
__global__ __launch_bounds__(kWaves * 64) void fused_paged_attn_kernel(
    const __bf16* q, Source src, const float* sink, const Position* positions,
    const float* cos_sin, c10::Float8_e4m3fn* out, uint8_t* out_scale,
    int64_t q_stride_t, int64_t q_stride_h, int64_t out_stride,
    int64_t scale_stride, int64_t num_positions, int heads, float sm_scale) {
  // kStageBytes only fits gfx950's 160 KiB LDS.
#if defined(__gfx950__)
  constexpr float kLog2e = 1.4426950408889634f;
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
  const int lr = lane % 16, lg = lane / 16;
  const int64_t token = blockIdx.x;
  const int hbase = blockIdx.y * 16;
  const int head = hbase + lr;
  const int64_t rope_pos = positions[token];
  const bool valid_pos = rope_pos >= 0 && rope_pos < num_positions;

  __shared__ __align__(16) uint8_t stage[kStageBytes];
  __shared__ __align__(16) int slots[kMaxKeys];
  __shared__ float wave_max[kWaves][16], wave_sum[kWaves][16];

  // MFMA step 2g+h contracts dims g*64 + lg*16 + h*8 + [0, 8): sixteen
  // contiguous FP8 bytes per lane share one scale group.
  BF16x8 qf[16];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      qf[2 * g + h] = head < heads ? *(Global<BF16x8>)(q + token * q_stride_t +
                                                       head * q_stride_h +
                                                       g * 64 + lg * 16 + h * 8)
                                   : BF16x8{};
    }
  }

  constexpr int kSlotsPerThread = kMaxKeys / (kWaves * 64);
  int indices[kSlotsPerThread];
#pragma unroll
  for (int r = 0; r < kSlotsPerThread; ++r)
    indices[r] = src.prefetch(token, tid + r * kWaves * 64);
  const KeySpan span = src.span(token);
  const int length = valid_pos ? span.length : 0;
  const float scale_log2 = sm_scale * kLog2e;
  __bf16* kv =
      reinterpret_cast<__bf16*>(stage) + wave * 2 * kTileKeys * kKStride;
  F4 acc[32];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d] = F4{};
  float row_max = -INFINITY, row_sum = 0.0f;

  // Scores one 16-key half and stages its K rows for the PV step.
  auto score = [&](const RowFragment& rows, int half) {
    BF16x8 k[16];
#pragma unroll
    for (int g = 0; g < 7; ++g) {
      const uint32_t bits =
          ((g < 4 ? rows.scales.x : rows.scales.y) >> (8 * (g % 4))) & 0xff;
      const float scale = __uint_as_float(bits ? bits << 23 : 0x00400000u);
      k[2 * g] = dequant_eight<AnyFnuz>(rows.nope[g].xy, scale, rows.fnuz);
      k[2 * g + 1] = dequant_eight<AnyFnuz>(rows.nope[g].zw, scale, rows.fnuz);
    }
    k[14] = __builtin_bit_cast(BF16x8, rows.rope[0]);
    k[15] = __builtin_bit_cast(BF16x8, rows.rope[1]);

    // Four independent accumulators keep the MFMA pipe full.
    F4 s[4] = {};
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i % 4] = mfma_k32(k[i], qf[i], s[i % 4]);
    __bf16* row = kv + (half * kTileKeys + lr) * kKStride;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      *(Local<BF16x8>)(row + swizzle_chunk(lr, (i / 2) * 8 + lg * 2 + i % 2) *
                                 8) = k[i];
    return (s[0] + s[1]) + (s[2] + s[3]);
  };

  // This lane holds S^T for keys 4*lg + [0, 4) of each half, for head lr.
  // PV contracts those eight keys in that order, so P^T needs no shuffle and
  // V^T is two transposed reads, one per half.
  auto attend = [&](F4 lo, F4 hi, int step, int n) {
    const int key = step * 2 * kTileKeys + 4 * lg;
    const I32x4 lo_slots = *(Local<I32x4>)(slots + key);
    const I32x4 hi_slots = *(Local<I32x4>)(slots + key + kTileKeys);
    float logit[8], step_max = -INFINITY;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      logit[e] =
          key + e < n && lo_slots[e] >= 0 ? lo[e] * scale_log2 : -INFINITY;
      logit[4 + e] = key + kTileKeys + e < n && hi_slots[e] >= 0
                         ? hi[e] * scale_log2
                         : -INFINITY;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) step_max = fmaxf(step_max, logit[e]);
    step_max = fmaxf(step_max, __shfl_xor(step_max, 16));
    step_max = fmaxf(step_max, __shfl_xor(step_max, 32));
    const float next_max = fmaxf(row_max, step_max);
    const float base = next_max == -INFINITY ? 0.0f : next_max;
    const float alpha = __builtin_amdgcn_exp2f(row_max - base);
    BF16x8 p;
    float step_sum = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float value = __builtin_amdgcn_exp2f(logit[e] - base);
      p[e] = static_cast<__bf16>(value);
      step_sum += value;
    }
    row_sum = row_sum * alpha + step_sum;
    row_max = next_max;
    if (__any(alpha != 1.0f)) {
#pragma unroll
      for (int d = 0; d < 32; ++d) acc[d] *= alpha;
    }

    // ds_read_tr16_b64 hands lane (lr, lg) V[4*lg + e][16*d + lr].
#if defined(__gfx950__)
    const int v_row = 4 * lg + lr / 4;
    const __bf16* v_base = kv + v_row * kKStride + 4 * (lr % 2);
#endif
    constexpr int kBatch = 8;
#pragma unroll
    for (int d0 = 0; d0 < 32; d0 += kBatch) {
      BF16x8 v[kBatch];
#pragma unroll
      for (int d = 0; d < kBatch; ++d) {
#if defined(__gfx950__)
        const int chunk = swizzle_chunk(v_row, 2 * (d0 + d) + (lr % 4) / 2) * 8;
        const auto lo_v = __builtin_amdgcn_ds_read_tr16_b64_v4i16(
            (Local<Short4>)(v_base + chunk));
        const auto hi_v = __builtin_amdgcn_ds_read_tr16_b64_v4i16(
            (Local<Short4>)(v_base + kTileKeys * kKStride + chunk));
        v[d] = __builtin_bit_cast(
            BF16x8,
            __builtin_shufflevector(lo_v, hi_v, 0, 1, 2, 3, 4, 5, 6, 7));
#else
  #pragma unroll
        for (int e = 0; e < 8; ++e)
          v[d][e] =
              kv[((e / 4) * kTileKeys + 4 * lg + e % 4) * kKStride +
                 swizzle_chunk(4 * lg + e % 4, 2 * (d0 + d) + lr / 8) * 8 +
                 lr % 8];
#endif
      }
#pragma unroll
      for (int d = 0; d < kBatch; ++d)
        acc[d0 + d] = mfma_k32(v[d], p, acc[d0 + d]);
      // Keep the batch's LDS reads ahead of its MFMAs.
      __builtin_amdgcn_sched_group_barrier(0x100, 2 * kBatch, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, kBatch, 0);
    }
  };

  for (int chunk = 0; chunk < length; chunk += kMaxKeys) {
    const int n = min(length - chunk, kMaxKeys);
    if (chunk) __syncthreads();
    // Issue every index load, then every table load, before waiting.
    int keys[kSlotsPerThread], logical[kSlotsPerThread];
#pragma unroll
    for (int r = 0; r < kSlotsPerThread; ++r) {
      if (chunk)
        indices[r] = src.prefetch(token, chunk + tid + r * kWaves * 64);
      keys[r] = chunk + min(tid + r * kWaves * 64, n - 1);
      logical[r] = src.logical(token, span, keys[r], indices[r]);
    }
#pragma unroll
    for (int r = 0; r < kSlotsPerThread; ++r) {
      const int slot = src.resolve(span, keys[r], logical[r]);
      if (tid + r * kWaves * 64 < n) slots[tid + r * kWaves * 64] = slot;
    }
    __syncthreads();
    const int steps = (n + 2 * kTileKeys - 1) / (2 * kTileKeys);
    auto fetch = [&](int t) {
      const int i = t * kTileKeys + lr;
      const int slot = i < n ? slots[min(i, kMaxKeys - 1)] : -1;
      return load_row<Aligned>(chunk + i < span.split ? src.first : src.second,
                               slot, lg);
    };
    // Each half is refetched for the wave's next step as soon as it is
    // scored, so a full step hides its load latency.
    RowFragment lo = fetch(2 * wave), hi = fetch(2 * wave + 1);
    for (int step = wave; step < steps; step += kWaves) {
      const F4 lo_scores = score(lo, 0);
      lo = fetch(2 * (step + kWaves));
      const F4 hi_scores = score(hi, 1);
      hi = fetch(2 * (step + kWaves) + 1);
      attend(lo_scores, hi_scores, step, n);
    }
  }

  // Merge the waves' softmax states, then the sink, as the Triton baseline.
  row_sum += __shfl_xor(row_sum, 16);
  row_sum += __shfl_xor(row_sum, 32);
  if (lg == 0) {
    wave_max[wave][lr] = row_max;
    wave_sum[wave][lr] = row_sum;
  }
  __syncthreads();
  const float sink_log2 =
      sink && head < heads ? sink[head] * kLog2e : -INFINITY;
  float final_max = sink_log2;
#pragma unroll
  for (int w = 0; w < kWaves; ++w)
    final_max = fmaxf(final_max, wave_max[w][lr]);
  float total = isfinite(sink_log2)
                    ? __builtin_amdgcn_exp2f(sink_log2 - final_max)
                    : 0.0f;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    if (wave_max[w][lr] != -INFINITY)
      total +=
          wave_sum[w][lr] * __builtin_amdgcn_exp2f(wave_max[w][lr] - final_max);
  }
  const float factor =
      row_max == -INFINITY || total <= 0.0f
          ? 0.0f
          : __builtin_amdgcn_exp2f(row_max - final_max) / total;
  // Two fixed-order passes keep the merge deterministic.
  float* merged = reinterpret_cast<float*>(stage);
  float* mine = merged + ((wave % 2) * 16 + lr) * kOStride + 4 * lg;
  if (wave < 2) {
#pragma unroll
    for (int d = 0; d < 32; ++d) *(Local<F4>)(mine + d * 16) = acc[d] * factor;
  }
  __syncthreads();
  if (wave >= 2) {
#pragma unroll
    for (int d = 0; d < 32; ++d) *(Local<F4>)(mine + d * 16) += acc[d] * factor;
  }
  __syncthreads();

  const int dim = tid * 2;
  float cos = 1.0f, sin = 0.0f;
  if (dim >= kHeadDim - kRopeDim && valid_pos) {
    const int pair = (dim - (kHeadDim - kRopeDim)) / 2;
    cos = cos_sin[rope_pos * kRopeDim + pair];
    sin = cos_sin[rope_pos * kRopeDim + kRopeDim / 2 + pair];
  }
  for (int h = 0; h < 16 && hbase + h < heads; ++h) {
    float a = merged[h * kOStride + dim] + merged[(16 + h) * kOStride + dim];
    float b =
        merged[h * kOStride + dim + 1] + merged[(16 + h) * kOStride + dim + 1];
    if constexpr (RoundAttention) {
      // Prefill materializes BF16 attention; decode keeps FP32.
      a = float(static_cast<__bf16>(a));
      b = float(static_cast<__bf16>(b));
    }
    const float even = a * cos + b * sin, odd = b * cos - a * sin;
    float amax = fmaxf(fabsf(even), fabsf(odd));
    for (int offset = 8; offset > 0; offset /= 2)
      amax = fmaxf(amax, __shfl_xor(amax, offset, 16));
    const float biased = ceilf(log2f(fmaxf(amax, 0x1p-126f) / 448.0f)) + 127.0f;
    const uint8_t bits =
        static_cast<uint8_t>(fminf(fmaxf(biased, 0.0f), 254.0f));
    const float reciprocal = exp2f(127.0f - bits);
    const int64_t row = token * out_stride + (hbase + h) * kHeadDim;
    const c10::Float8_e4m3fn pair[2] = {
        c10::Float8_e4m3fn(fminf(fmaxf(even * reciprocal, -448.0f), 448.0f)),
        c10::Float8_e4m3fn(fminf(fmaxf(odd * reciprocal, -448.0f), 448.0f))};
    *reinterpret_cast<uint16_t*>(out + row + dim) =
        __builtin_bit_cast(uint16_t, pair);
    if (tid % 16 == 0)
      out_scale[token * scale_stride + (hbase + h) * (kHeadDim / 32) +
                dim / 32] = bits;
  }
#endif
}

// gfx942 has 64 KiB of LDS and no transposed LDS read, scaled FP8 convert or
// K=32 BF16 MFMA. paged_attn_bf16_kernel gives each query two waves that keep
// the NoPE keys in FP8 and run FP8 MFMAs on them.
namespace gfx942 {

using F2 = float __attribute__((ext_vector_type(2)));

// Two waves per CTA own 16 heads: wave h takes QK steps 16 h + [0, 16), i.e.
// FP8 scale groups 0-3, or groups 4-6 plus RoPE, which keeps each wave under
// 256 registers so two waves share a SIMD. Lane (lr, lg) loads the 16-dim
// chunks 4 m + lg of key lr. QK step 4 m + s contracts dims
// 64 m + 16 lg + 4 s + [0, 4) of K and Q.
constexpr int kDimSteps = kHeadDim / 16;
constexpr int kNopeChunks = (kHeadDim - kRopeDim) / 64;

__device__ __forceinline__ int wave_dim(int lg, int i) {
  return 64 * (i / 4) + 16 * lg + 4 * (i % 4);
}

// page << 8 | offset, or -1. Block sizes are powers of two up to 256.
__device__ __forceinline__ int wave_lookup(const PagedCache& c, int request,
                                           int logical) {
  const int shift = __builtin_ctz(c.block_size);
  const int index = logical >> shift;
  const int block = c.table[int64_t(request) * c.table_stride +
                            max(0, min(index, c.table_width - 1))];
  return logical < 0 || index >= c.table_width || block < 0 || block >= c.blocks
             ? -1
             : block << 8 | (logical & (c.block_size - 1));
}

__device__ __forceinline__ F4 mfma_fp8(long a, long b, F4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo),
                               0x07060302u);
}

template <typename Position, typename Source, bool Aligned>
__global__ __launch_bounds__(128) void paged_attn_bf16_kernel(
    const __bf16* q, Source src, const float* sink, const Position* positions,
    const float* cos_sin, __bf16* out, int64_t q_stride_t, int64_t q_stride_h,
    int64_t out_stride_t, int64_t out_stride_h, int64_t num_positions,
    int heads, float sm_scale, bool rotate_q) {
#if defined(__gfx942__)
  __shared__ F4 exchange[2][2][64];
  const int half = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
  auto body = [&](auto half_c) {
    constexpr int kHalf = decltype(half_c)::value;
    constexpr int kSteps2 = kDimSteps / 2;
    constexpr int kFirst = kHalf * kSteps2;
    // Scale groups held as FP8: 4 for wave 0, 3 for wave 1 (plus RoPE).
    constexpr int kGroups = kHalf == 0 ? 4 : kNopeChunks - 4;
    constexpr float kLog2e = 1.4426950408889634f;
    const int lane = threadIdx.x % 64, lr = lane % 16, lg = lane / 16;
    const int64_t token = blockIdx.x;
    const int head = blockIdx.y * 16 + lr;
    const int64_t rope_pos = positions[token];
    const bool valid_pos = rope_pos >= 0 && rope_pos < num_positions;

    BF16x8 qv[kSteps2 / 2];
#pragma unroll
    for (int i = 0; i < kSteps2 / 2; ++i)
      qv[i] = head < heads ? *(Global<BF16x8>)(q + token * q_stride_t +
                                               head * q_stride_h +
                                               wave_dim(lg, kFirst + 2 * i))
                           : BF16x8{};

    const KeySpan span = src.span(token);
    const int length = valid_pos ? span.length : 0;
    const int steps = (length + kTileKeys - 1) / kTileKeys;
    const float scale_log2 = sm_scale * kLog2e;

    // Keys resolve three steps ahead: the top-k read, the block table, then
    // the record, so each wave's shorter step still covers the load latency.
    auto index_of = [&](int step) {
      return src.prefetch(token, step * kTileKeys + lr);
    };
    auto slot_of = [&](int step, int index) {
      const int key = step * kTileKeys + lr;
      if (key >= length) return -1;
      const int logical = src.logical(token, span, key, index);
      return wave_lookup(key < span.split ? src.first : src.second,
                         span.request, logical);
    };
    // This wave's share of a 584-byte record: its FP8 scale groups, RoPE on
    // wave 1, and the wave's four E8M0 scale bytes.
    struct Frag {
      U32x4 nope[kGroups] = {};
      U32x4 rope[2] = {};
      uint32_t scales = 0;
      bool fnuz = false;
    };
    auto fetch = [&](int step, int slot) {
      const PagedCache& cache =
          step * kTileKeys + lr < span.split ? src.first : src.second;
      Frag f;
      const bool valid = slot >= 0;
      const uint8_t* page =
          cache.data + int64_t(valid ? slot >> 8 : 0) * cache.block_stride;
      const int offset = valid ? slot & 255 : 0;
      const uint8_t* row = valid ? page + offset * 576 : kZeroRecord;
      const uint8_t* scales = valid ? page + cache.block_size * 576 + offset * 8
                                    : kZeroRecord + 576;
#pragma unroll
      for (int m = 0; m < kGroups; ++m)
        f.nope[m] =
            load_global<Aligned, U32x4>(row + 64 * (4 * kHalf + m) + 16 * lg);
      if constexpr (kHalf == 1) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
          f.rope[h] = load_global<Aligned, U32x4>(row + 448 + 32 * lg + 16 * h);
      }
      f.scales = load_global<Aligned, uint32_t>(scales + 4 * kHalf);
      f.fnuz = cache.fnuz;
      return f;
    };

    int slot_cur = slot_of(0, index_of(0));
    int slot_next = slot_of(1, index_of(1));
    int slot_next2 = slot_of(2, index_of(2));
    int index_next3 = index_of(3);
    Frag frag_a = fetch(0, slot_cur), frag_b = fetch(1, slot_next), frag_c;

    // NoPE Q as FNUZ hi + lo of q / q_scale, q_scale a power of two per head
    // and wave with max |q| / q_scale < 128. The max runs on the BF16 bit
    // patterns with the sign cleared.
    using U16x2 = uint16_t __attribute__((ext_vector_type(2)));
    U16x2 q_bits = {};
#pragma unroll
    for (int i = 0; i < 2 * kGroups; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        q_bits = __builtin_elementwise_max(
            q_bits,
            __builtin_bit_cast(
                U16x2, __builtin_bit_cast(U32x4, qv[i])[j] & 0x7fff7fffu));
    uint32_t q_amax_bits = max(q_bits.x, q_bits.y);
    q_amax_bits = max(q_amax_bits, uint32_t(__shfl_xor(int(q_amax_bits), 16)));
    q_amax_bits = max(q_amax_bits, uint32_t(__shfl_xor(int(q_amax_bits), 32)));
    const float q_amax = __uint_as_float(q_amax_bits << 16);
    const int q_exp = max(__builtin_amdgcn_frexp_expf(q_amax), -100);
    const float q_inv = __builtin_ldexpf(1.0f, 7 - q_exp);
    const float q_scale = __builtin_ldexpf(1.0f, q_exp - 7);
    long qhi[kGroups][2], qlo[kGroups][2];
#pragma unroll
    for (int m = 0; m < kGroups; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t hw[2], lw[2];
        const U32x4 qw = __builtin_bit_cast(U32x4, qv[2 * m + h]);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          const uint32_t w0 = qw[2 * w], w1 = qw[2 * w + 1];
          const F2 x0 = F2{__uint_as_float(w0 << 16),
                           __uint_as_float(w0 & 0xffff0000u)} * q_inv;
          const F2 x1 = F2{__uint_as_float(w1 << 16),
                           __uint_as_float(w1 & 0xffff0000u)} * q_inv;
          int hb = __builtin_amdgcn_cvt_pk_fp8_f32(x0.x, x0.y, int(w0), false);
          hb = __builtin_amdgcn_cvt_pk_fp8_f32(x1.x, x1.y, hb, true);
          const F2 r0 = x0 - __builtin_amdgcn_cvt_pk_f32_fp8(hb, false);
          const F2 r1 = x1 - __builtin_amdgcn_cvt_pk_f32_fp8(hb, true);
          int lb = __builtin_amdgcn_cvt_pk_fp8_f32(r0.x, r0.y, int(w1), false);
          lb = __builtin_amdgcn_cvt_pk_fp8_f32(r1.x, r1.y, lb, true);
          hw[w] = uint32_t(hb);
          lw[w] = uint32_t(lb);
        }
        qhi[m][h] = __builtin_bit_cast(long, U32x2{hw[0], hw[1]});
        qlo[m][h] = __builtin_bit_cast(long, U32x2{lw[0], lw[1]});
      }
    BF16x4 qr[4] = {};
    if constexpr (kHalf == 1) {
      // GPT-J RoPE on Q, rounded to BF16, with the FMAs the Q prep in
      // fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert compiles to, so the
      // rotated Q matches it bit for bit. Lane (lr, lg) holds RoPE dims
      // 16 lg + 8 i + [0, 8), i.e. pairs 8 lg + 4 i + [0, 4).
      if (rotate_q && valid_pos) {
        const float* cs = cos_sin + rope_pos * kRopeDim;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          BF16x8& x = qv[2 * kGroups + i];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int pair = 8 * lg + 4 * i + j;
            const float cos = cs[pair], sin = cs[kRopeDim / 2 + pair];
            const float even = float(x[2 * j]), odd = float(x[2 * j + 1]);
            x[2 * j] = static_cast<__bf16>(fmaf(even, cos, -(odd * sin)));
            x[2 * j + 1] = static_cast<__bf16>(fmaf(odd, cos, even * sin));
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        qr[2 * i] = qv[2 * kGroups + i].lo;
        qr[2 * i + 1] = qv[2 * kGroups + i].hi;
      }
    }

    F4 acc[kSteps2];
#pragma unroll
    for (int d = 0; d < kSteps2; ++d) acc[d] = F4{};
    float row_max = -INFINITY, row_sum = 0.0f;

    BF16x4 identity;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      identity[e] = static_cast<__bf16>(4 * lg + e == lr ? 1.0f : 0.0f);
    // FNUZ selection: K (16 keys x 32 dims) times sel[par] hands lane (lr, lg)
    // dim 16 (lr / 4) + 4 par + lr % 4 of the 8-byte half, keys 4 lg + [0, 4).
    long sel[2];
#pragma unroll
    for (int par = 0; par < 2; ++par)
      sel[par] = lg == lr / 4 ? long(0x40ull << (8 * (4 * par + lr % 4))) : 0l;

    // Three rotating fragments so the in-flight loads never alias the
    // fragment being read.
    auto run_step = [&](int step, Frag& frag, Frag& next) {
      const uint32_t valid_keys = uint32_t(__ballot(slot_cur >= 0)) & 0xffffu;

      // NoPE K stays FP8 and the MFMAs read it as FNUZ: an OCP byte is half
      // its value there, so its scale byte gets +1. OCP -0 (0x80) would be
      // the FNUZ NaN; the compressed-cache writer stores +0 on FNUZ platforms.
      if (!frag.fnuz) frag.scales += 0x01010101u;
      auto kfp8 = [&](int m, int h) {
        return __builtin_bit_cast(
            long, U32x2{frag.nope[m][2 * h], frag.nope[m][2 * h + 1]});
      };
      BF16x4 kr[4] = {};
      if constexpr (kHalf == 1) {
#pragma unroll
        for (int t = 0; t < 4; ++t)
          kr[t] = __builtin_bit_cast(
              BF16x4, t % 2 ? frag.rope[t / 2].zw : frag.rope[t / 2].xy);
      }
      // Scale words of keys 4 lg + e: the keys of this lane's S^T and P.
      uint32_t ws[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) ws[e] = __shfl(frag.scales, 4 * lg + e);
      auto kscale = [&](int m, int e) {
        return __uint_as_float((ws[e] >> (8 * m) & 0xffu) << 23);
      };

      const int slot_next3 = slot_of(step + 3, index_next3);
      index_next3 = index_of(step + 4);
      next = fetch(step + 2, slot_next2);

      F4 nope = {};
#pragma unroll
      for (int m = 0; m < kGroups; ++m) {
        F4 c = {};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          c = mfma_fp8(kfp8(m, h), qhi[m][h], c);
          c = mfma_fp8(kfp8(m, h), qlo[m][h], c);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) nope[e] = fmaf(c[e], kscale(m, e), nope[e]);
      }
      F4 part;
      if constexpr (kHalf == 1) {
        F4 sr[2] = {};
#pragma unroll
        for (int t = 0; t < 4; ++t)
          sr[t % 2] = mfma_k16(kr[t], qr[t], sr[t % 2]);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          part[e] = fmaf(nope[e], q_scale, sr[0][e] + sr[1][e]);
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) part[e] = nope[e] * q_scale;
      }
      // Both waves add the halves in the same order, so they see the same
      // scores and take the same softmax branches. Alternating buffers need
      // only one barrier per step.
      exchange[step % 2][kHalf][lane] = part;
      __syncthreads();
      const F4 score =
          exchange[step % 2][0][lane] + exchange[step % 2][1][lane];

      // This lane holds S^T for keys 4*lg + [0, 4) of head lr.
      float logit[4], local_max = -INFINITY;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = 4 * lg + e;
        logit[e] = valid_keys >> key & 1 ? score[e] * scale_log2 : -INFINITY;
        local_max = fmaxf(local_max, logit[e]);
      }
      // The running max only moves when a key beats it by 2^8; until then P
      // stays <= 256, which BF16 and the f32 sums absorb, and the step needs
      // no cross-lane max. row_sum stays per lane until the epilogue. P at the
      // max is then not exactly 1, so row_sum adds the rounded P the PV MFMA
      // sees.
      if (__any(local_max > row_max + 8.0f)) {
        float step_max = fmaxf(local_max, __shfl_xor(local_max, 16));
        step_max = fmaxf(step_max, __shfl_xor(step_max, 32));
        const float next_max = step_max > row_max + 8.0f ? step_max : row_max;
        const float alpha = __builtin_amdgcn_exp2f(
            row_max - (next_max == -INFINITY ? 0.0f : next_max));
        row_sum *= alpha;
        row_max = next_max;
#pragma unroll
        for (int d = 0; d < kSteps2; ++d) acc[d] *= alpha;
      }
      const float base = row_max == -INFINITY ? 0.0f : row_max;
      BF16x4 p;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        p[e] = static_cast<__bf16>(__builtin_amdgcn_exp2f(logit[e] - base));
        row_sum += float(p[e]);
      }

      // K sel (FP8) or K I (RoPE) hands lane (lr, lg) keys 4 lg + [0, 4) of
      // QK step d's dim lr, unscaled; group m's per-key scales ride on P.
      constexpr int kBatch = 4;
#pragma unroll
      for (int m = 0; m < kSteps2 / kBatch; ++m) {
        F4 t[kBatch];
        BF16x4 pm;
        if (m < kGroups) {
#pragma unroll
          for (int d = 0; d < kBatch; ++d)
            t[d] = mfma_fp8(kfp8(m, d / 2), sel[d % 2], F4{});
          pm = __builtin_bit_cast(
              BF16x4,
              (U32x2{pack_bf16(float(p[0]) * kscale(m, 0),
                               float(p[1]) * kscale(m, 1)),
                     pack_bf16(float(p[2]) * kscale(m, 2),
                               float(p[3]) * kscale(m, 3))}));
        } else {
#pragma unroll
          for (int d = 0; d < kBatch; ++d)
            t[d] = mfma_k16(kr[d], identity, F4{});
          pm = p;
        }
#pragma unroll
        for (int d = 0; d < kBatch; ++d) {
          const BF16x4 v = __builtin_bit_cast(
              BF16x4, (U32x2{pack_bf16(t[d][0], t[d][1]),
                             pack_bf16(t[d][2], t[d][3])}));
          acc[kBatch * m + d] = mfma_k16(v, pm, acc[kBatch * m + d]);
        }
        __builtin_amdgcn_sched_group_barrier(0x008, kBatch, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, 2 * kBatch, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, kBatch, 0);
      }

      slot_cur = slot_next;
      slot_next = slot_next2;
      slot_next2 = slot_next3;
    };
    for (int step = 0; step < steps; step += 3) {
      run_step(step, frag_a, frag_c);
      if (step + 1 == steps) break;
      run_step(step + 1, frag_b, frag_a);
      if (step + 2 == steps) break;
      run_step(step + 2, frag_c, frag_b);
    }

    row_sum += __shfl_xor(row_sum, 16);
    row_sum += __shfl_xor(row_sum, 32);
    const float sink_log2 =
        sink && head < heads ? sink[head] * kLog2e : -INFINITY;
    const float final_max = fmaxf(row_max, sink_log2);
    const float total =
        (row_max == -INFINITY
             ? 0.0f
             : row_sum * __builtin_amdgcn_exp2f(row_max - final_max)) +
        (isfinite(sink_log2) ? __builtin_amdgcn_exp2f(sink_log2 - final_max)
                             : 0.0f);
    const float factor =
        row_max == -INFINITY || total <= 0.0f
            ? 0.0f
            : __builtin_amdgcn_exp2f(row_max - final_max) / total;
    if (head >= heads) return;

    // acc[d] holds dims wave_dim(lg, kFirst + d) + [0, 4) of head lr: two
    // RoPE pairs.
    __bf16* row = out + token * out_stride_t + head * out_stride_h;
#pragma unroll
    for (int d = 0; d < kSteps2; ++d) {
      const int dim = wave_dim(lg, kFirst + d);
      float o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = float(static_cast<__bf16>(acc[d][e] * factor));
      if (dim >= kHeadDim - kRopeDim && valid_pos) {
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int pair = (dim + e - (kHeadDim - kRopeDim)) / 2;
          const float cos = cos_sin[rope_pos * kRopeDim + pair];
          const float sin = cos_sin[rope_pos * kRopeDim + kRopeDim / 2 + pair];
          const float a = o[e], b = o[e + 1];
          o[e] = a * cos + b * sin;
          o[e + 1] = b * cos - a * sin;
        }
      }
      BF16x4 r;
#pragma unroll
      for (int e = 0; e < 4; ++e) r[e] = static_cast<__bf16>(o[e]);
      *reinterpret_cast<BF16x4*>(row + dim) = r;
    }
  };
  if (half == 0)
    body(std::integral_constant<int, 0>{});
  else
    body(std::integral_constant<int, 1>{});
#endif
}

}  // namespace gfx942

bool on_arch(const at::Tensor& t, std::string_view arch) {
  const auto* props = at::cuda::getDeviceProperties(t.device().index());
  return std::string_view(props->gcnArchName).substr(0, arch.size()) == arch;
}

PagedCache paged_cache(const at::Tensor& q, const at::Tensor& cache,
                       bool fnuz) {
  TORCH_CHECK(
      cache.device() == q.device() && cache.scalar_type() == at::kByte &&
          cache.dim() == 3 && cache.size(1) > 0 && cache.size(2) == 584 &&
          cache.stride(2) == 1 && cache.stride(1) == 584 &&
          cache.stride(0) >= cache.size(1) * 584 &&
          cache.size(0) * cache.size(1) <= std::numeric_limits<int>::max(),
      "cache must have packed 584-byte records [blocks, block_size, 584]");
  PagedCache result;
  result.data = cache.data_ptr<uint8_t>();
  result.block_stride = cache.stride(0);
  result.block_size = int(cache.size(1));
  result.blocks = int(cache.size(0));
  result.fnuz = fnuz;
  return result;
}

bool aligned_cache(const PagedCache& cache) {
  return !cache.data || (reinterpret_cast<uintptr_t>(cache.data) % 16 == 0 &&
                         cache.block_stride % 16 == 0);
}

template <bool RoundAttention, typename Source>
void launch_paged_attn(const at::Tensor& q, const Source& src,
                       const std::optional<at::Tensor>& attn_sink,
                       double sm_scale, const at::Tensor& positions,
                       const at::Tensor& cos_sin_cache, at::Tensor& out,
                       at::Tensor& out_scale) {
  if (q.size(0) == 0) return;
  TORCH_CHECK(on_arch(q, "gfx950"),
              "the paged MXFP8 mega-attention kernel is gfx950-only");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "q rows must be 16-byte aligned");
  TORCH_CHECK(out.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 2 == 0,
              "out rows must be 2-byte aligned");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(q.size(0), (q.size(1) + 15) / 16);
  const bool aligned = aligned_cache(src.first) && aligned_cache(src.second);
  const bool fnuz = src.first.fnuz || src.second.fnuz;
#define LAUNCH_FNUZ(POSITION_TYPE, ALIGNED, FNUZ)                           \
  fused_paged_attn_kernel<POSITION_TYPE, Source, RoundAttention, ALIGNED,   \
                          FNUZ><<<grid, kWaves * 64, 0, stream>>>(          \
      reinterpret_cast<const __bf16*>(q.data_ptr<c10::BFloat16>()), src,    \
      attn_sink ? attn_sink->data_ptr<float>() : nullptr,                   \
      positions.data_ptr<POSITION_TYPE>(), cos_sin_cache.data_ptr<float>(), \
      out.data_ptr<c10::Float8_e4m3fn>(), out_scale.data_ptr<uint8_t>(),    \
      q.stride(0), q.stride(1), out.stride(0), out_scale.stride(0),         \
      cos_sin_cache.size(0), int(q.size(1)), sm_scale)
#define LAUNCH(POSITION_TYPE, ALIGNED)          \
  if (fnuz) {                                   \
    LAUNCH_FNUZ(POSITION_TYPE, ALIGNED, true);  \
  } else {                                      \
    LAUNCH_FNUZ(POSITION_TYPE, ALIGNED, false); \
  }
  if (positions.scalar_type() == at::kInt) {
    if (aligned) {
      LAUNCH(int32_t, true);
    } else {
      LAUNCH(int32_t, false);
    }
  } else {
    if (aligned) {
      LAUNCH(int64_t, true);
    } else {
      LAUNCH(int64_t, false);
    }
  }
#undef LAUNCH
#undef LAUNCH_FNUZ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Source>
void launch_paged_attn_bf16(const at::Tensor& q, const Source& src,
                            const std::optional<at::Tensor>& attn_sink,
                            double sm_scale, const at::Tensor& positions,
                            const at::Tensor& cos_sin_cache, bool rotate_q,
                            at::Tensor& out) {
  if (q.size(0) == 0) return;
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "q rows must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (const PagedCache* cache : {&src.first, &src.second})
    TORCH_CHECK(
        !cache->data || (cache->block_size <= 256 &&
                         (cache->block_size & (cache->block_size - 1)) == 0 &&
                         cache->blocks < (1 << 23)),
        "the CDNA3 kernel needs a power-of-two block_size <= 256 "
        "and < 2^23 blocks");
  const dim3 grid(q.size(0), (q.size(1) + 15) / 16);
  const bool aligned = aligned_cache(src.first) && aligned_cache(src.second);
#define ARGS(POSITION_TYPE)                                                  \
  reinterpret_cast<const __bf16*>(q.data_ptr<c10::BFloat16>()), src,         \
      attn_sink ? attn_sink->data_ptr<float>() : nullptr,                    \
      positions.data_ptr<POSITION_TYPE>(), cos_sin_cache.data_ptr<float>(),  \
      reinterpret_cast<__bf16*>(out.data_ptr<c10::BFloat16>()), q.stride(0), \
      q.stride(1), out.stride(0), out.stride(1), cos_sin_cache.size(0),      \
      int(q.size(1)), sm_scale, rotate_q
#define LAUNCH(POSITION_TYPE, ALIGNED)                           \
  gfx942::paged_attn_bf16_kernel<POSITION_TYPE, Source, ALIGNED> \
      <<<grid, 128, 0, stream>>>(ARGS(POSITION_TYPE))
  if (positions.scalar_type() == at::kInt) {
    if (aligned) {
      LAUNCH(int32_t, true);
    } else {
      LAUNCH(int32_t, false);
    }
  } else {
    if (aligned) {
      LAUNCH(int64_t, true);
    } else {
      LAUNCH(int64_t, false);
    }
  }
#undef LAUNCH
#undef ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_query_inputs(const at::Tensor& q,
                        const std::optional<at::Tensor>& attn_sink,
                        double sm_scale, const at::Tensor& positions,
                        const at::Tensor& cos_sin_cache, double rms_norm_eps) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.scalar_type() == at::kBFloat16,
              "q must be a GPU BF16 tensor [tokens, heads, 512]");
  const auto tokens = q.size(0), heads = q.size(1);
  TORCH_CHECK(q.size(2) == kHeadDim && heads > 0 && heads <= 65535 &&
                  tokens <= std::numeric_limits<int>::max(),
              "expected head_dim=512 and a supported query grid");
  const auto device = q.device();
  auto check_tensor = [&](const at::Tensor& tensor, at::ScalarType dtype,
                          const char* name) {
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype,
                name, " has an incorrect device or dtype");
    TORCH_CHECK(tensor.dim() > 0 && tensor.stride(-1) == 1, name,
                " must be contiguous along the last dimension");
  };
  check_tensor(q, at::kBFloat16, "q");
  check_tensor(cos_sin_cache, at::kFloat, "cos_sin_cache");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim &&
                  cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous [num_positions, 64]");
  TORCH_CHECK(
      positions.device() == device && positions.dim() == 1 &&
          positions.size(0) == tokens && positions.is_contiguous() &&
          (positions.scalar_type() == at::kInt ||
           positions.scalar_type() == at::kLong),
      "positions must be a contiguous int32/int64 vector of token positions");
  if (attn_sink) {
    check_tensor(*attn_sink, at::kFloat, "attn_sink");
    TORCH_CHECK(attn_sink->dim() == 1 && attn_sink->size(0) == heads,
                "attn_sink must have one float32 logit per head");
  }
  TORCH_CHECK(std::isfinite(sm_scale) && std::isfinite(rms_norm_eps) &&
                  rms_norm_eps >= 0.0,
              "scale and epsilon must be finite; epsilon must be nonnegative");
}

void check_query_output(const at::Tensor& q,
                        const std::optional<at::Tensor>& attn_sink,
                        double sm_scale, const at::Tensor& positions,
                        const at::Tensor& cos_sin_cache, double rms_norm_eps,
                        const at::Tensor& out, const at::Tensor& out_scale) {
  check_query_inputs(q, attn_sink, sm_scale, positions, cos_sin_cache,
                     rms_norm_eps);
  const auto tokens = q.size(0), heads = q.size(1);
  TORCH_CHECK(out.device() == q.device() &&
                  out.scalar_type() == at::kFloat8_e4m3fn && out.dim() > 0 &&
                  out.stride(-1) == 1 && out_scale.device() == q.device() &&
                  out_scale.scalar_type() == at::kByte &&
                  out_scale.dim() > 0 && out_scale.stride(-1) == 1,
              "out and out_scale have an incorrect device, dtype or stride");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens &&
                  out.size(1) == heads * kHeadDim && out_scale.dim() == 2 &&
                  out_scale.size(0) == tokens &&
                  out_scale.size(1) == heads * (kHeadDim / 32),
              "outputs must be [tokens, heads*512] e4m3 and "
              "[tokens, heads*16] uint8 E8M0");
  TORCH_CHECK(
      out.stride(0) >= out.size(1) && out_scale.stride(0) >= out_scale.size(1),
      "output rows must not overlap");
}

void check_bf16_output(const at::Tensor& q, const at::Tensor& out) {
  TORCH_CHECK(out.device() == q.device() &&
                  out.scalar_type() == at::kBFloat16 && out.dim() == 3 &&
                  out.size(0) == q.size(0) && out.size(1) == q.size(1) &&
                  out.size(2) == kHeadDim && out.stride(2) == 1,
              "out must be a BF16 tensor [tokens, heads, 512]");
  TORCH_CHECK(out.stride(1) >= kHeadDim &&
                  out.stride(0) >= out.size(1) * out.stride(1),
              "output rows must not overlap");
  TORCH_CHECK(out.stride(0) % 4 == 0 && out.stride(1) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "out rows must be 8-byte aligned");
}

PrefillSource prefill_source(
    const at::Tensor& q, const at::Tensor& main_cache,
    const at::Tensor& main_table, const std::optional<at::Tensor>& extra_cache,
    const std::optional<at::Tensor>& extra_table, const at::Tensor& topk,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    const at::Tensor& gather_lens, int64_t window_size, int64_t compress_ratio,
    bool main_use_fnuz) {
  const auto requests = seq_lens.numel();
  TORCH_CHECK(window_size > 0 &&
                  window_size <= std::numeric_limits<int>::max() &&
                  compress_ratio >= 0 &&
                  compress_ratio <= std::numeric_limits<int>::max(),
              "invalid window_size or compress_ratio");
  TORCH_CHECK(extra_cache.has_value() == extra_table.has_value() &&
                  extra_cache.has_value() == (compress_ratio > 0),
              "compressed cache and table are required iff compress_ratio > 0");
  for (const auto& tensor : {query_start_loc, seq_lens, gather_lens}) {
    TORCH_CHECK(tensor.device() == q.device() &&
                    tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
                    tensor.is_contiguous(),
                "prefill metadata must be contiguous GPU int32 vectors");
  }
  TORCH_CHECK(query_start_loc.numel() == requests + 1 &&
                  gather_lens.numel() == requests &&
                  requests <= std::numeric_limits<int>::max(),
              "inconsistent prefill request metadata");
  TORCH_CHECK(topk.device() == q.device() && topk.scalar_type() == at::kInt &&
                  topk.dim() == 2 && topk.size(0) == q.size(0) &&
                  topk.stride(1) == 1 &&
                  topk.size(1) <= std::numeric_limits<int>::max(),
              "topk must be GPU int32 [tokens, topk_width]");
  auto paged_table = [&](const at::Tensor& cache, const at::Tensor& table,
                         bool fnuz) {
    TORCH_CHECK(table.device() == q.device() &&
                    table.scalar_type() == at::kInt && table.dim() == 2 &&
                    table.size(0) == requests && table.stride(1) == 1 &&
                    table.size(1) > 0 &&
                    table.size(1) <= std::numeric_limits<int>::max(),
                "block table must be GPU int32 [requests, max_blocks]");
    PagedCache result = paged_cache(q, cache, fnuz);
    result.table = table.data_ptr<int32_t>();
    result.table_stride = table.stride(0);
    result.table_width = int(table.size(1));
    return result;
  };
  PrefillSource src;
  src.second = paged_table(main_cache, main_table, main_use_fnuz);
  if (extra_cache) src.first = paged_table(*extra_cache, *extra_table, false);
  if (q.size(0) == 0) return src;
  TORCH_CHECK(requests > 0, "nonempty prefill requires request metadata");
  src.topk = topk.data_ptr<int32_t>();
  src.starts = query_start_loc.data_ptr<int32_t>();
  src.seq_lens = seq_lens.data_ptr<int32_t>();
  src.gather_lens = gather_lens.data_ptr<int32_t>();
  src.topk_stride = topk.stride(0);
  src.requests = int(requests);
  src.topk_width = int(topk.size(1));
  src.window = int(window_size);
  src.compress_ratio = int(compress_ratio);
  return src;
}

}  // namespace

void fused_norm_rope_attn_rope_cast_fwd_rocm(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink, double sm_scale,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool enable_q_norm, double rms_norm_eps, bool apply_q_rope, at::Tensor& out,
    at::Tensor& out_scale) {
  check_query_output(q, attn_sink, sm_scale, positions, cos_sin_cache,
                     rms_norm_eps, out, out_scale);
  const auto tokens = q.size(0), heads = q.size(1);
  const auto device = q.device();
  auto check_tensor = [&](const at::Tensor& tensor, at::ScalarType dtype,
                          const char* name) {
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype &&
                    tensor.dim() > 0 && tensor.stride(-1) == 1,
                name, " has an incorrect device, dtype or stride");
  };
  check_tensor(kv, at::kBFloat16, "kv");
  check_tensor(indices, at::kInt, "indices");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == kHeadDim,
              "kv must be [kv_rows, 1, 512] BF16 gathered rows");
  TORCH_CHECK(
      indices.dim() == 2 && indices.size(0) == tokens,
      "indices must be [tokens, topk] int32; invalid slots are skipped");
  if (topk_length) {
    check_tensor(*topk_length, at::kInt, "topk_length");
    TORCH_CHECK(topk_length->dim() == 1 && topk_length->size(0) == tokens,
                "topk_length must have one int32 length per token");
  }
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(tokens, heads);
  const auto* lengths =
      topk_length ? topk_length->data_ptr<int32_t>() : nullptr;
  const auto* sink = attn_sink ? attn_sink->data_ptr<float>() : nullptr;
  const DenseKV rows{kv.data_ptr<c10::BFloat16>(),
                     indices.data_ptr<int32_t>(),
                     lengths,
                     kv.stride(0),
                     indices.stride(0),
                     kv.size(0),
                     indices.size(1)};
#define LAUNCH(POSITION_TYPE, SPLITS)                                          \
  fused_prefill_kernel<POSITION_TYPE, SPLITS>                                  \
      <<<grid, SPLITS * 64, 0, stream>>>(                                      \
          q.data_ptr<c10::BFloat16>(), rows, sink,                             \
          positions.data_ptr<POSITION_TYPE>(),                                 \
          cos_sin_cache.data_ptr<float>(), out.data_ptr<c10::Float8_e4m3fn>(), \
          out_scale.data_ptr<uint8_t>(), q.stride(0), q.stride(1),             \
          out.stride(0), out_scale.stride(0), cos_sin_cache.size(0), sm_scale, \
          enable_q_norm, rms_norm_eps, apply_q_rope)
  if (positions.scalar_type() == at::kInt) {
    if (tokens * heads <= 1024) {
      LAUNCH(int32_t, 16);
    } else {
      LAUNCH(int32_t, 8);
    }
  } else {
    if (tokens * heads <= 1024) {
      LAUNCH(int64_t, 16);
    } else {
      LAUNCH(int64_t, 8);
    }
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_norm_rope_attn_rope_cast_decode_rocm(
    const at::Tensor& q, const at::Tensor& main_cache,
    const at::Tensor& main_indices, const at::Tensor& main_indptr,
    const std::optional<at::Tensor>& extra_cache,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_indptr,
    const std::optional<at::Tensor>& attn_sink, double sm_scale,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool main_use_fnuz, at::Tensor& out, at::Tensor& out_scale) {
  check_query_output(q, attn_sink, sm_scale, positions, cos_sin_cache, 0.0, out,
                     out_scale);
  TORCH_CHECK(extra_cache.has_value() == extra_indices.has_value() &&
                  extra_cache.has_value() == extra_indptr.has_value(),
              "extra cache, indices and indptr must be supplied together");
  auto check_ragged = [&](const at::Tensor& indices, const at::Tensor& indptr) {
    for (const auto& tensor : {indices, indptr}) {
      TORCH_CHECK(
          tensor.device() == q.device() && tensor.scalar_type() == at::kInt &&
              tensor.dim() == 1 && tensor.is_contiguous(),
          "ragged indices and indptr must be contiguous GPU int32 vectors");
    }
    TORCH_CHECK(indptr.size(0) == q.size(0) + 1,
                "indptr must have tokens + 1 entries");
  };
  DecodeSource src;
  check_ragged(main_indices, main_indptr);
  src.first = paged_cache(q, main_cache, main_use_fnuz);
  src.first_indices = main_indices.data_ptr<int32_t>();
  src.first_indptr = main_indptr.data_ptr<int32_t>();
  src.first_capacity = main_indices.numel();
  if (extra_cache) {
    check_ragged(*extra_indices, *extra_indptr);
    src.second = paged_cache(q, *extra_cache, false);
    src.second_indices = extra_indices->data_ptr<int32_t>();
    src.second_indptr = extra_indptr->data_ptr<int32_t>();
    src.second_capacity = extra_indices->numel();
  }
  launch_paged_attn<false>(q, src, attn_sink, sm_scale, positions,
                           cos_sin_cache, out, out_scale);
}

void fused_norm_rope_attn_rope_cast_prefill_rocm(
    const at::Tensor& q, const at::Tensor& main_cache,
    const at::Tensor& main_table, const std::optional<at::Tensor>& extra_cache,
    const std::optional<at::Tensor>& extra_table, const at::Tensor& topk,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    const at::Tensor& gather_lens, int64_t window_size, int64_t compress_ratio,
    const std::optional<at::Tensor>& attn_sink, double sm_scale,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool main_use_fnuz, at::Tensor& out, at::Tensor& out_scale) {
  check_query_output(q, attn_sink, sm_scale, positions, cos_sin_cache, 0.0, out,
                     out_scale);
  const PrefillSource src = prefill_source(
      q, main_cache, main_table, extra_cache, extra_table, topk,
      query_start_loc, seq_lens, gather_lens, window_size, compress_ratio,
      main_use_fnuz);
  launch_paged_attn<true>(q, src, attn_sink, sm_scale, positions, cos_sin_cache,
                          out, out_scale);
}

void fused_norm_rope_attn_rope_prefill_bf16_rocm(
    const at::Tensor& q, const at::Tensor& main_cache,
    const at::Tensor& main_table, const std::optional<at::Tensor>& extra_cache,
    const std::optional<at::Tensor>& extra_table, const at::Tensor& topk,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    const at::Tensor& gather_lens, int64_t window_size, int64_t compress_ratio,
    const std::optional<at::Tensor>& attn_sink, double sm_scale,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool main_use_fnuz, bool rotate_q, at::Tensor& out) {
  check_query_inputs(q, attn_sink, sm_scale, positions, cos_sin_cache, 0.0);
  check_bf16_output(q, out);
  TORCH_CHECK(on_arch(q, "gfx942"),
              "the BF16 mega-attention prefill is gfx942-only");
  const PrefillSource src = prefill_source(
      q, main_cache, main_table, extra_cache, extra_table, topk,
      query_start_loc, seq_lens, gather_lens, window_size, compress_ratio,
      main_use_fnuz);
  launch_paged_attn_bf16(q, src, attn_sink, sm_scale, positions, cos_sin_cache,
                         rotate_q, out);
}
