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

void check_query_output(const at::Tensor& q,
                        const std::optional<at::Tensor>& attn_sink,
                        double sm_scale, const at::Tensor& positions,
                        const at::Tensor& cos_sin_cache, double rms_norm_eps,
                        const at::Tensor& out, const at::Tensor& out_scale) {
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
  check_tensor(out, at::kFloat8_e4m3fn, "out");
  check_tensor(out_scale, at::kByte, "out_scale");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim &&
                  cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous [num_positions, 64]");
  TORCH_CHECK(
      positions.device() == device && positions.dim() == 1 &&
          positions.size(0) == tokens && positions.is_contiguous() &&
          (positions.scalar_type() == at::kInt ||
           positions.scalar_type() == at::kLong),
      "positions must be a contiguous int32/int64 vector of token positions");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens &&
                  out.size(1) == heads * kHeadDim && out_scale.dim() == 2 &&
                  out_scale.size(0) == tokens &&
                  out_scale.size(1) == heads * (kHeadDim / 32),
              "outputs must be [tokens, heads*512] e4m3 and "
              "[tokens, heads*16] uint8 E8M0");
  TORCH_CHECK(
      out.stride(0) >= out.size(1) && out_scale.stride(0) >= out_scale.size(1),
      "output rows must not overlap");
  if (attn_sink) {
    check_tensor(*attn_sink, at::kFloat, "attn_sink");
    TORCH_CHECK(attn_sink->dim() == 1 && attn_sink->size(0) == heads,
                "attn_sink must have one float32 logit per head");
  }
  TORCH_CHECK(std::isfinite(sm_scale) && std::isfinite(rms_norm_eps) &&
                  rms_norm_eps >= 0.0,
              "scale and epsilon must be finite; epsilon must be nonnegative");
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
  if (q.size(0) == 0) return;
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
  launch_paged_attn<true>(q, src, attn_sink, sm_scale, positions, cos_sin_cache,
                          out, out_scale);
}
