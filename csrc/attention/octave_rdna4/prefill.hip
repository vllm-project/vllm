// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {

using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float8 = float __attribute__((ext_vector_type(8)));

constexpr int kHead = 256;
constexpr int kFragment = 16;
constexpr int kWave = 32;
// Each expanded-prefill wave owns 16 query/head rows and their full output.
constexpr int kKeyTile = 32;

constexpr float kProbabilityScale = 16384.f;
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

template <int BatchTile>
__device__ __forceinline__ int prefill_request(int block, const int32_t* starts,
                                               int requests, int group) {
  int low = 0, high = requests;
  while (low + 1 < high) {
    const int mid = (low + high) / 2;
    if (int64_t(starts[mid]) * group / BatchTile + mid <= block) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low;
}

using octave_rdna4::Activation;

template <typename T>
__device__ __forceinline__ Half8 read8(const T* ptr) {
  using Vec = T __attribute__((ext_vector_type(8)));
  Vec x;
  __builtin_memcpy(&x, ptr, sizeof(x));
  return __builtin_convertvector(__builtin_convertvector(x, Float8), Half8);
}

// Only the live chunk is converted and transposed. Its storage reuses the
// fixed-point query buffer once the prefix kernel has consumed it.
template <typename T>
__global__ __launch_bounds__(128) void stage_chunk(
    _Float16* k_out, _Float16* v_out, Activation key, Activation value,
    const int32_t* starts, int requests) {
  const int tile = blockIdx.x, kv = blockIdx.y;
  const int req = prefill_request<kKeyTile>(tile, starts, requests, 1);
  const int start = starts[req], end = starts[req + 1];
  const int base = start + (tile - start / kKeyTile - req) * kKeyTile;
  if (base >= end) return;
  const int64_t offset = (int64_t(tile) * gridDim.y + kv) * kKeyTile * kHead;
  __shared__ alignas(16) _Float16 v_tile[2][kHead][24];
  const int tid = threadIdx.x;
#pragma unroll
  for (int u = tid; u < kKeyTile * kHead / 8; u += 128) {
    const int t = u / (kHead / 8), d = 8 * (u % (kHead / 8));
    Half8 k = {}, v = {};
    if (base + t < end) {
      k = read8(static_cast<const T*>(key.data) +
                (base + t) * key.token_stride + kv * key.head_stride + d);
      v = read8(static_cast<const T*>(value.data) +
                (base + t) * value.token_stride + kv * value.head_stride + d);
    }
    *reinterpret_cast<Half8*>(k_out + offset + 8 * u) = k;
#pragma unroll
    for (int e = 0; e < 8; ++e) v_tile[t / 16][d + e][t % 16] = v[e];
  }
  prefill_lds_barrier();
#pragma unroll
  for (int u = tid; u < kKeyTile * kHead / 8; u += 128)
    *reinterpret_cast<Half8*>(v_out + offset + 8 * u) =
        *reinterpret_cast<const Half8*>(
            &v_tile[u / (2 * kHead)][(u / 2) % kHead][(u % 2) * 8]);
}

// Q and output are token-major. Expanded K is token-major within a KV head.
// V is stored as [KV tile of 16, output dimension, key within tile].
template <typename T, bool ChunkOnly, bool Precise, int RowWaves>
__global__
__launch_bounds__(kWave * RowWaves * (ChunkOnly ? 2 : 1)) void dense_prefill(
    float* out, Activation query, Activation keys, Activation values,
    const _Float16* residual, const float* errors, float* chunk_lse,
    uint8_t* repair, const int32_t* starts, const int32_t* lengths,
    int requests, int kv_heads, int group, int capacity, float scale) {
  constexpr int kDimParts = ChunkOnly ? 2 : 1;
  constexpr int RowTile = RowWaves * kFragment;
  constexpr int kThreads = RowWaves * kDimParts * kWave;
  constexpr int Keys = kKeyTile / kFragment;
  constexpr int Steps = kHead / (kDimParts * kFragment);
  constexpr int Vectors = kHead / 8;

  const int block = blockIdx.x;
  const int req = prefill_request<RowTile>(block, starts, requests, group);
  const int start = starts[req], end = starts[req + 1];
  // Tiles span query/head rows within one request and one KV head. A tile
  // can begin or end in the middle of a token, including for group > RowTile.
  const int64_t first_row =
      (block - int64_t(start) * group / RowTile - req) * RowTile;
  const int live_rows =
      min(int64_t(RowTile), int64_t(end - start) * group - first_row);
  if (live_rows <= 0) return;
  const int length = ChunkOnly ? end - start : lengths[req];
  const int prefix = length - (end - start);
  const int first_token = first_row / group;
  const int head_in_group = first_row - int64_t(first_token) * group;
  const int key_end =
      prefix + first_token + (head_in_group + live_rows - 1) / group + 1;
  const int kv_head = blockIdx.y;

  const int tid = threadIdx.x, lane = tid % kWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
  const int column = lane % kFragment, half = lane / kFragment;
  const int row = (wave / kDimParts) * kFragment + column;
  const int part = wave % kDimParts;
  const int local_row = head_in_group + min(row, live_rows - 1);
  const int token_delta = local_row / group;
  const int q = start + first_token + token_delta;
  const int q_head = kv_head * group + (local_row - token_delta * group);
  const int query_end = prefix + q - start + 1;
  const int64_t head_offset = int64_t(req * kv_heads + kv_head) * capacity * kHead;
  const int64_t row_offset = (int64_t(q) * kv_heads * group + q_head) * kHead;
  if constexpr (!ChunkOnly) {
    if (prefix < 0 || length > capacity) {
      if (row < live_rows && half == 0) {
        repair[2 * (row_offset / kHead)] = 1;
        repair[2 * (row_offset / kHead) + 1] = 0;
      }
      return;
    }
  }
  const T* q_row =
      static_cast<const T*>(query.data) +
      (ChunkOnly ? q * query.token_stride + q_head * query.head_stride
                 : row_offset);

  // Pad expanded K rows by one vector to distribute reads across LDS banks.
  // Each unrolled WMMA step then uses a constant offset from one lane address.
  constexpr int KeyStride = kHead + (ChunkOnly ? 0 : 8);
  constexpr int KeyXorMask = ChunkOnly ? 7 : 0;
  __shared__ alignas(16) _Float16 key_tile[kKeyTile][KeyStride];
  __shared__ alignas(16) _Float16 value_tile[Keys][kHead][kFragment];

  __shared__ alignas(16)
      Float4 exchange[ChunkOnly ? RowWaves * 2 : 1][Keys][2][kWave];
  Half8 q_fragment[Steps];
  float query_max = 0.f, query_energy = 0.f;
#pragma unroll
  for (int d = 0; d < Steps; ++d) {
    const T* fragment = q_row + (part * Steps + d) * kFragment + half * 8;
    if constexpr (ChunkOnly)
      q_fragment[d] = read8(fragment);
    else
      q_fragment[d] = *reinterpret_cast<const Half8*>(fragment);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x = float(q_fragment[d][e]);
      query_max = fmaxf(query_max, fabsf(x));
      if constexpr (!ChunkOnly) query_energy = fmaf(x, x, query_energy);
    }
  }
  if constexpr (!ChunkOnly) {
    query_energy = (query_energy + other_half(query_energy)) * scale * scale;
    if (prefix > 0) {
      // scale^2 * ||Q||^2 * max(||K - half(K)||^2) bounds squared logit
      // error. Both launches make the same workgroup-wide choice on replay.
      float bound = errors[(int64_t(req) * kv_heads + kv_head) * 32 + lane];
#pragma unroll
      for (int d = 16; d; d /= 2) bound = fmaxf(bound, __shfl_xor(bound, d));
      constexpr float max_logit_error = 1.f / 64.f;
      const int precise = __syncthreads_or(query_energy * bound >
                                           max_logit_error * max_logit_error);
      if (bool(precise) != Precise) return;
    } else if constexpr (Precise) {
      return;
    }
  }
  query_max = fmaxf(query_max, other_half(query_max));
  const bool query_safe = query_max == 0.f ||
      (query_max >= 1.e-4f && query_max * scale <= 2.f);
  if constexpr (!ChunkOnly) {
    if (row < live_rows && half == 0) {
      repair[2 * (row_offset / kHead)] = !query_safe;
      repair[2 * (row_offset / kHead) + 1] = 0;
    }
  }
  Float8 accum[Steps] = {};
  float running_max = -INFINITY;
  float running_mass = 0.f;
  const float log2_scale = scale * kLog2e;

  for (int key_start = 0; key_start < key_end; key_start += kKeyTile) {
    const int64_t tile_offset =
        ChunkOnly ? (int64_t(start / kKeyTile + req + key_start / kKeyTile) *
                         kv_heads +
                     kv_head) *
                        kKeyTile * kHead
                  : head_offset + int64_t(key_start) * kHead;
    const auto* k_src = reinterpret_cast<const Half8*>(
        static_cast<const _Float16*>(keys.data) + tile_offset);
    const auto* v_src = reinterpret_cast<const Half8*>(
        static_cast<const _Float16*>(values.data) + tile_offset);
    // Both inputs use contiguous 16-byte vectors within the staged tile.
    // Expansion has already zeroed the last partial tile.
#pragma unroll
    for (int unit = tid; unit < kKeyTile * Vectors; unit += kThreads) {
      const int token = unit / Vectors;
      const int col = unit % Vectors;
      const int k_col = col ^ (token & KeyXorMask);
      *reinterpret_cast<Half8*>(&key_tile[token][k_col * 8]) = k_src[unit];
      const int sub_tile = unit / (2 * kHead);
      const int v_dim = unit / 2 % kHead;
      const int key = unit % 2;
      const int swizzled_key = key ^ ((v_dim >> 2) & 1);
      *reinterpret_cast<Half8*>(&value_tile[sub_tile][v_dim][swizzled_key * 8]) = v_src[unit];
    }
    prefill_lds_barrier();

    // Place keys on the accumulator rows and queries on the columns. The
    // softmax probabilities already have the layout needed by the second WMMA.
    Float8 scores[Keys] = {};
#pragma unroll
    for (int d = 0; d < Steps; ++d) {
#pragma unroll
      for (int k = 0; k < Keys; ++k) {
        const int k_col = ((part * Steps + d) * 2 + half) ^ (column & KeyXorMask);
        const Half8 key = *reinterpret_cast<const Half8*>(
            &key_tile[k * kFragment + column][k_col * 8]);
        scores[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            key, q_fragment[d], scores[k]);
      }
    }
    if constexpr (ChunkOnly) {
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
    if constexpr (Precise) {
      if (key_start < prefix) {
#pragma unroll
        for (int k = 0; k < Keys; ++k) {
          Float8 correction = {};
#pragma unroll
          for (int d = 0; d < Steps; ++d) {
            const int64_t base =
                head_offset + int64_t(key_start + k * kFragment) * kHead;
            const Half8 lo = *reinterpret_cast<const Half8*>(
                residual + base + ((part * Steps + d) * 32 + lane) * 8);
            correction = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                lo, q_fragment[d], correction);
          }
          scores[k] += correction * (1.f / 1024.f);
        }
      }
    }
    if (key_start + kKeyTile > prefix + first_token + 1) {
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
    // An exact power of two keeps small probabilities representable in FP16
    // even when they multiply large V. The lazy maximum limits p to two.
    const bool update = tile_max > running_max + 1.f;
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
      probabilities[k] = __builtin_convertvector(p * kProbabilityScale, Half8);
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
  if (row >= live_rows) return;
  if constexpr (ChunkOnly) {
    if (half == 0 && part == 0)
      chunk_lse[row_offset / kHead] = running_max + log2f(running_mass);
  }
  const float inv_mass = (1.f / kProbabilityScale) / running_mass;
#pragma unroll
  for (int d = 0; d < Steps; ++d) {
    const int dim = (part * Steps + d) * kFragment + half * 8;
    *reinterpret_cast<Float8*>(out + row_offset + dim) = accum[d] * inv_mass;
  }
}

}  // namespace

namespace octave_rdna4 {

int launch_dense_prefill(float* out, const _Float16* query,
                         const _Float16* keys, const _Float16* residual,
                         const float* errors, const _Float16* values,
                         const int32_t* starts, const int32_t* lengths,
                         int requests, int tokens, int kv_heads, int group,
                         int capacity, float scale, uint8_t* repair,
                         hipStream_t stream) {
  const Activation q{query, int64_t(kv_heads) * group * kHead, kHead};
  auto launch = [&](auto waves) {
    constexpr int W = decltype(waves)::value;
    const dim3 grid(int64_t(tokens) * group / (W * kFragment) + requests,
                    kv_heads);
    auto issue = [&](auto kernel) {
      kernel<<<grid, W * kWave, 0, stream>>>(
          out, q, {keys, 0, 0}, {values, 0, 0}, residual, errors, nullptr,
          repair, starts, lengths, requests, kv_heads, group, capacity, scale);
    };
    issue(dense_prefill<_Float16, false, false, W>);
    issue(dense_prefill<_Float16, false, true, W>);
  };
  if (tokens >= 1536 || (requests > 1 && tokens >= 512)) {
    launch(std::integral_constant<int, 20>{});
  } else {
    // For a single request a partial final workgroup still scans the whole
    // prefix. Prefer the narrowest tile that fills the fewest 32-group waves.
    const int groups_per_head = std::max(1, 32 / kv_heads);
    auto passes = [&](int waves) {
      const int width = groups_per_head * waves * kFragment;
      return (int64_t(tokens) * group + width - 1) / width;
    };
    if (requests > 1 || passes(12) <= passes(20))
      launch(std::integral_constant<int, 12>{});
    else if (passes(16) <= passes(20))
      launch(std::integral_constant<int, 16>{});
    else
      launch(std::integral_constant<int, 20>{});
  }
  return hipGetLastError();
}

int launch_chunk_only(float* out, Activation query, Activation keys,
                      Activation values, void* packed_chunk, float* chunk_lse,
                      const int32_t* starts, int requests, int tokens,
                      int kv_heads, int group, float scale, int dtype,
                      hipStream_t stream) {
  constexpr int RowWaves = 4;
  const dim3 grid(int64_t(tokens) * group / (RowWaves * kFragment) + requests,
                  kv_heads);
  const int tiles = tokens / kKeyTile + requests;
  auto* k = static_cast<_Float16*>(packed_chunk);
  auto* v = k + int64_t(tiles) * kv_heads * kKeyTile * kHead;
  auto stage = [&](auto scalar) {
    stage_chunk<decltype(scalar)><<<dim3(tiles, kv_heads), 128, 0, stream>>>(
        k, v, keys, values, starts, requests);
  };
  if (dtype == 0)
    stage(_Float16{});
  else
    stage(__bf16{});
  const int status = hipGetLastError();
  if (status != hipSuccess) return status;
  auto launch = [&](auto scalar) {
    dense_prefill<decltype(scalar), true, false, RowWaves>
        <<<grid, RowWaves * 2 * kWave, 0, stream>>>(
            out, query, {k, 0, 0}, {v, 0, 0}, nullptr, nullptr, chunk_lse,
            nullptr, starts, nullptr, requests, kv_heads, group, 0, scale);
  };
  if (dtype == 0)
    launch(_Float16{});
  else
    launch(__bf16{});
  return hipGetLastError();
}

}  // namespace octave_rdna4
