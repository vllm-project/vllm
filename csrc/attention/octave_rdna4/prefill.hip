// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float8 = float __attribute__((ext_vector_type(8)));

constexpr int kHead = 256;
constexpr int kGroup = 6;
constexpr int kFragment = 16;
constexpr int kWave = 32;
// A workgroup handles ten tokens of a GQA6 request. Pairs of waves share
// score computation and cover disjoint halves of the value dimensions.
constexpr int kRowWaves = 4;
constexpr int kDimParts = 2;
constexpr int kKeyTile = 32;
constexpr int kThreads = kWave * kRowWaves * kDimParts;
constexpr int kQueries = kRowWaves * kFragment / kGroup;
constexpr float kLog2e = 1.4426950408889634f;

__device__ inline __attribute__((convergent)) void prefill_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ float other_half(float value) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlanex16(
                 __builtin_bit_cast(int, value), __builtin_bit_cast(int, value),
                 0x76543210u, 0xFEDCBA98u, true, false));
}

__device__ __forceinline__ int prefill_request(
    int block, const int32_t* starts, int requests) {
  int low = 0, high = requests;
  while (low + 1 < high) {
    const int mid = (low + high) / 2;
    if (starts[mid] / kQueries + mid <= block) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low;
}

// Q and output are token-major. Expanded K is token-major within a KV head.
// V is stored as [KV tile of 16, output dimension, key within tile].
__global__ __launch_bounds__(kThreads) void dense_prefill(
    float* out, const _Float16* query, const _Float16* keys,
    const _Float16* values, const int32_t* starts, const int32_t* lengths,
    int requests, int kv_heads, int capacity, float scale) {
  constexpr int Keys = kKeyTile / kFragment;
  constexpr int Steps = kHead / (kDimParts * kFragment);
  constexpr int Vectors = kHead / 8;


  const int block = blockIdx.x;
  const int req = prefill_request(block, starts, requests);
  const int start = starts[req], end = starts[req + 1];
  const int q_base = start + (block - start / kQueries - req) * kQueries;
  if (q_base >= end) return;
  const int q_count = min(kQueries, end - q_base);
  const int length = lengths[req], prefix = length - (end - start);
  const int key_end = prefix + q_base - start + q_count;
  const int kv_head = blockIdx.y;

  const int tid = threadIdx.x, lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
  const int column = lane % kFragment, half = lane / kFragment;
  const int row = (wave / kDimParts) * kFragment + column;
  const int part = wave % kDimParts;
  const int q = q_base + min(row / kGroup, q_count - 1);
  const int q_head = kv_head * kGroup + row % kGroup;
  const int query_end = prefix + q - start + 1;
  const int64_t head_offset = int64_t(req * kv_heads + kv_head) * capacity * kHead;
  const int64_t row_offset = (int64_t(q) * kv_heads * kGroup + q_head) * kHead;
  if (prefix < 0 || length > capacity) {
    // A replay can outgrow its temporary buffer. Mark these rows for the
    // packed path without reading the expanded allocations.
    if (row < q_count * kGroup && part == 0 && half == 0)
      out[row_offset] = __builtin_nanf("");
    return;
  }
  const _Float16* key_base = keys + head_offset;
  const _Float16* value_base = values + head_offset;
  const _Float16* q_row = query + row_offset;

  __shared__ alignas(16) _Float16 key_tile[kKeyTile][kHead];
  __shared__ alignas(16) _Float16 value_tile[Keys][kHead][kFragment];
  __shared__ alignas(16) Float4 exchange[kRowWaves * kDimParts][Keys][2][kWave];

  Half8 q_fragment[Steps];
  float query_max = 0.f;
#pragma unroll
  for (int d = 0; d < Steps; ++d) {
    q_fragment[d] = *reinterpret_cast<const Half8*>(
        q_row + (part * Steps + d) * kFragment + half * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      query_max = fmaxf(query_max, fabsf(float(q_fragment[d][e])));
  }
  query_max = fmaxf(query_max, other_half(query_max));
  const bool query_safe = query_max == 0.f ||
      (query_max >= 1.e-4f && query_max * scale <= 2.f);
  Float8 accum[Steps] = {};
  float running_max = -INFINITY;
  float running_mass = 0.f;
  const float log2_scale = scale * kLog2e;

  for (int key_start = 0; key_start < key_end; key_start += kKeyTile) {
    // Swizzle the 16-byte vector addresses to distribute the LDS reads
    // across banks. Expansion has already zeroed the last partial tile.
#pragma unroll
    for (int unit = tid; unit < kKeyTile * Vectors; unit += kThreads) {
      const int token = unit / Vectors;
      const int col = unit % Vectors;
      const int pos = key_start + token;
      const auto* k_src = reinterpret_cast<const Half8*>(key_base + pos * kHead + col * 8);
      *reinterpret_cast<Half8*>(&key_tile[token][(col ^ (token & 7)) * 8]) = *k_src;

      const int sub_tile = unit / (2 * kHead);
      const int v_dim = unit / 2 % kHead;
      const int key = unit % 2;
      const auto* v_src = reinterpret_cast<const Half8*>(
          value_base + (key_start + sub_tile * kFragment) * kHead
          + v_dim * kFragment + key * 8);
      const int swizzled_key = key ^ ((v_dim >> 2) & 1);
      *reinterpret_cast<Half8*>(&value_tile[sub_tile][v_dim][swizzled_key * 8]) = *v_src;
    }
    prefill_lds_barrier();

    // Place keys on the accumulator rows and queries on the columns. The
    // softmax probabilities already have the layout needed by the second WMMA.
    Float8 scores[Keys] = {};
#pragma unroll
    for (int d = 0; d < Steps; ++d) {
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
        const Half8 key = *reinterpret_cast<const Half8*>(
            &key_tile[k * kFragment + column][(((part * Steps + d) * 2 + half) ^ (column & 7)) * 8]);
        scores[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            key, q_fragment[d], scores[k]);
      }
    }
    {
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
        const Float8 x = scores[k];
        exchange[wave][k][0][lane] = __builtin_shufflevector(x, x, 0, 1, 2, 3);
        exchange[wave][k][1][lane] = __builtin_shufflevector(x, x, 4, 5, 6, 7);
      }
      prefill_lds_barrier();
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
        const Float4 lo = exchange[wave ^ 1][k][0][lane];
        const Float4 hi = exchange[wave ^ 1][k][1][lane];
        scores[k] += __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      }
    }
    if (key_start + kKeyTile > prefix + q_base - start + 1) {
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (key_start + k * kFragment + half * 8 + e >= query_end)
            scores[k][e] = -INFINITY;
        }
      }
    }
    float tile_max = -INFINITY;
#pragma unroll
    for (int k = 0; k < Keys; ++k) {
#pragma unroll
      for (int e = 0; e < 8; ++e) tile_max = fmaxf(tile_max, scores[k][e]);
    }
    tile_max = fmaxf(tile_max, other_half(tile_max)) * log2_scale;
    // Keep the exponent base fixed for small changes to the running maximum.
    // Exp2 of an offset up to eight remains well within FP16's range.
    const bool update = tile_max > running_max + 8.f;
    float alpha = 1.f;
    if (__builtin_amdgcn_ballot_w32(update)) {
      const float next_max = update ? tile_max : running_max;
      const float safe_max = next_max == -INFINITY ? 0.f : next_max;
      alpha = __builtin_amdgcn_exp2f(running_max - safe_max);
      running_max = next_max;
#pragma unroll
      for (int d = 0; d < Steps; ++d) accum[d] *= alpha;
    }
    const float safe_max = running_max == -INFINITY ? 0.f : running_max;
    Half8 probabilities[Keys];
    float tile_mass = 0.f;
#pragma unroll
    for (int k = 0; k < Keys; ++k) {
      Float8 p;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        p[e] = __builtin_amdgcn_exp2f(fmaf(scores[k][e], log2_scale, -safe_max));
        tile_mass += p[e];
      }
      probabilities[k] = __builtin_convertvector(p, Half8);
    }
    running_mass = fmaf(running_mass, alpha, tile_mass + other_half(tile_mass));
#pragma unroll
    for (int d = 0; d < Steps; ++d) {
      const int dim = (part * Steps + d) * kFragment + column;
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
        const Half8 v = *reinterpret_cast<const Half8*>(&value_tile[k][dim][(half ^ ((dim >> 2) & 1)) * 8]);
        accum[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            v, probabilities[k], accum[d]);
      }
    }
    prefill_lds_barrier();
  }
  if (row >= q_count * kGroup) return;
  // The existing packed FP32 path repairs rows marked by the producer.
  const float inv_mass = query_safe ? 1.f / running_mass : __builtin_nanf("");
#pragma unroll
  for (int d = 0; d < Steps; ++d) {
    const int dim = (part * Steps + d) * kFragment + half * 8;
    *reinterpret_cast<Float8*>(out + row_offset + dim) = accum[d] * inv_mass;
  }
}

}  // namespace

namespace octave_rdna4 {

int launch_dense_prefill(
    float* out, const _Float16* query, const _Float16* keys,
    const _Float16* values, const int32_t* starts, const int32_t* lengths,
    int requests, int tokens, int kv_heads, int capacity, float scale,
    hipStream_t stream) {
  dense_prefill<<<dim3(tokens / kQueries + requests, kv_heads), kThreads,
                  0, stream>>>(out, query, keys, values, starts, lengths,
                                requests, kv_heads, capacity, scale);
  return hipGetLastError();
}

}  // namespace octave_rdna4
