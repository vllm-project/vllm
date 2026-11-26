// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Specialized gfx12 dense kernels for 256 head and 64 rotary dimensions.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "kernels.h"
#include "split_merge.cuh"

__device__ __forceinline__ uint32_t octave_xor_var(uint32_t value, int mask) {
  uint32_t selector = (uint32_t(threadIdx.x) ^ uint32_t(mask)) & 15u;
  if (mask & 16)
    return __builtin_amdgcn_permlanex16_var(value, value, selector, false,
                                            false);
  return __builtin_amdgcn_permlane16_var(value, value, selector, false, false);
}

constexpr uint32_t octave_dpp8_xor_selector(int mask) {
  uint32_t selector = 0;
  for (int i = 0; i < 8; ++i) selector |= uint32_t(i ^ mask) << (3 * i);
  return selector;
}

__device__ __forceinline__ uint32_t octave_xor_dpp(uint32_t value, int mask) {
  if (mask == 1)
    return __builtin_amdgcn_mov_dpp8(value, octave_dpp8_xor_selector(1));
  if (mask == 2)
    return __builtin_amdgcn_mov_dpp8(value, octave_dpp8_xor_selector(2));
  if (mask == 4)
    return __builtin_amdgcn_mov_dpp8(value, octave_dpp8_xor_selector(4));
  return octave_xor_var(value, mask);
}

__device__ __forceinline__ uint32_t octave_index_var(uint32_t value,
                                                     int index) {
  uint32_t selector = uint32_t(index) & 15u;
  uint32_t same =
      __builtin_amdgcn_permlane16_var(value, value, selector, false, false);
  uint32_t cross =
      __builtin_amdgcn_permlanex16_var(value, value, selector, false, false);
  return ((threadIdx.x ^ uint32_t(index)) & 16u) ? cross : same;
}

template <bool Dpp>
__device__ __forceinline__ uint32_t octave_xor(uint32_t value, int mask,
                                               int width) {
  return Dpp ? octave_xor_dpp(value, mask) : octave_xor_var(value, mask);
}
template <bool Dpp>
__device__ __forceinline__ float octave_xor(float value, int mask, int width) {
  return __uint_as_float(octave_xor<Dpp>(__float_as_uint(value), mask, width));
}
__device__ __forceinline__ uint32_t octave_index(uint32_t value, int index,
                                                 int width) {
  return octave_index_var(value, index);
}

namespace {

using octave_rdna4::Activation;
using octave_rdna4::PagedCache;

constexpr int kHead = 256;
constexpr int kRope = 64;
constexpr int kWave = 32;
constexpr int kLaneValues = kHead / kWave;
constexpr int kPartialSize = kHead + 2;

__device__ __forceinline__ int ceil_pow2_device(int value) {
  int power = 1;
  while (power < value && power < 1024) power <<= 1;
  return power;
}

// Keep the host-planned grid/scratch stride stable for graph replay, but avoid
// launching useful work for context partitions beyond the live query length.
__device__ __forceinline__ int active_decode_splits(int length, int maximum,
                                                     int query_tokens,
                                                     int kv_heads) {
  if (length <= 0) return 1;
  const int query_ctas = max(1, query_tokens * kv_heads);
  const int occupancy_floor =
      ceil_pow2_device((128 - 1) / query_ctas + 1);
  const int context_segment = query_tokens >= 8 ? 4096 :
                              query_tokens >= 4 ? 2048 : 1024;
  const int context_floor =
      ceil_pow2_device((length - 1) / context_segment + 1);
  const int minimum_partition = query_tokens <= 2 ? 64 : 128;
  const int minimum_tile_cap =
      ceil_pow2_device((length - 1) / minimum_partition + 1);
  return max(1, min(maximum, min(minimum_tile_cap,
                                 max(occupancy_floor, context_floor))));
}

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
  #error "Octave RDNA4 requires a gfx12 target"
#endif

template <bool Compact, int ValueBits>
struct Format {
  static constexpr bool compact = Compact;
  static constexpr int v_bits = ValueBits;
  static constexpr int k3_offset = Compact ? 0 : kRope / 2;
  static constexpr int k3_dims = Compact ? kHead : kHead - kRope;
  static constexpr int v_offset = k3_offset + k3_dims * 3 / 8;
  static constexpr int scale_offset = v_offset + kHead * ValueBits / 8;
  static constexpr int v_scale = Compact ? 1 : 4;
  static constexpr int payload = scale_offset + 2 * (v_scale + 1);
  static constexpr int alignment = Compact ? 4 : 8;
  static constexpr int slot_bytes =
      (payload + alignment - 1) / alignment * alignment;
};

template <int Bits>
constexpr int centroid(int code) {
  if constexpr (Bits == 3) {
    constexpr int lut[] = {-127, -79, -45, -14, 14, 45, 79, 127};
    return lut[code];
  } else {
    constexpr int lut[] = {-126, -95, -74, -58, -43, -30, -18, -6,
                           6,    18,  30,  43,  58,  74,  95,  126};
    return lut[code];
  }
}

template <int Bits>
constexpr uint32_t table_word(int first) {
  uint32_t word = 0;
  for (int i = 0; i < 4; ++i) {
    word |= uint32_t(uint8_t(centroid<Bits>(first + i))) << (8 * i);
  }
  return word;
}

template <int Bits>
__device__ __forceinline__ uint32_t lookup_four(uint32_t codes) {
  uint32_t low = __builtin_amdgcn_perm(table_word<Bits>(4), table_word<Bits>(0),
                                       codes & 0x07070707u);
  if constexpr (Bits == 3) {
    return low;
  } else {
    uint32_t high = __builtin_amdgcn_perm(
        table_word<Bits>(12), table_word<Bits>(8), codes & 0x07070707u);
    uint32_t mask = ((codes >> 3) & 0x01010101u) * 0xFFu;
    return (low & ~mask) | (high & mask);
  }
}

__device__ __forceinline__ float signed_byte(uint32_t word, int i) {
  return float(int8_t(word >> (8 * i)));
}

// Each lane owns four consecutive dimensions from each half of the head.
__device__ __forceinline__ int dim_at(int lane, int i) {
  return 4 * lane + (i & 3) + (i / 4) * (kHead / 2);
}

template <typename T>
__device__ __forceinline__ void load_activation(float (&x)[kLaneValues],
                                                Activation view, int token,
                                                int head, int lane) {
  const auto* row = static_cast<const T*>(view.data) +
                    token * view.token_stride + head * view.head_stride;
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) x[i] = float(row[dim_at(lane, i)]);
}

__device__ __forceinline__ void flip_signs(float (&x)[kLaneValues],
                                           const uint32_t* signs, int lane) {
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) {
    int dim = dim_at(lane, i);
    if ((signs[dim / 32] >> (dim % 32)) & 1u) x[i] = -x[i];
  }
}

__device__ __forceinline__ void butterfly(float& a, float& b) {
  float difference = a - b;
  a += b;
  b = difference;
}

template <bool WholeHead>
__device__ __forceinline__ void hadamard(float (&x)[kLaneValues], int lane) {
#pragma unroll
  for (int i = 0; i < kLaneValues; i += 4) {
    butterfly(x[i], x[i + 1]);
    butterfly(x[i + 2], x[i + 3]);
    butterfly(x[i], x[i + 2]);
    butterfly(x[i + 1], x[i + 3]);
  }
#pragma unroll
  for (int delta = 1; delta <= (WholeHead ? 16 : 8); delta *= 2) {
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      float other = octave_xor<false>(x[i], delta, kWave);
      x[i] = (lane & delta) ? other - x[i] : x[i] + other;
    }
  }
  if constexpr (WholeHead) {
#pragma unroll
    for (int i = 0; i < 4; ++i) butterfly(x[i], x[i + 4]);
  }
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) x[i] *= WholeHead ? 1.f / 16 : 1.f / 8;
}

// Use HIP lane shuffles for reductions and packing. ROCm 7.2.3 produces
// incorrect scales and packed codes with raw permlane instructions here.
template <int Width>
__device__ __forceinline__ float sum_lanes(float x) {
#pragma unroll
  for (int delta = Width / 2; delta > 0; delta /= 2) {
    x += __shfl_xor(x, delta, kWave);
  }
  return x;
}

template <int Bits>
__device__ __forceinline__ uint32_t quantize_four(const float* x, float rms) {
  uint32_t codes = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float z = (x[i] / rms) * (Bits == 3 ? 59.f : 46.f);
    uint32_t code = 0;
#pragma unroll
    for (int j = 1; j < (1 << Bits); ++j) {
      float midpoint = (centroid<Bits>(j - 1) + centroid<Bits>(j)) * 0.5f;
      code += z > midpoint;
    }
    codes |= code << (8 * i);
  }
  return codes;
}

__device__ __forceinline__ float dot_four(const float* x, uint32_t lut) {
  float dot = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) dot = fmaf(x[i], signed_byte(lut, i), dot);
  return dot;
}

template <int Bits>
__device__ __forceinline__ void pack_four(uint8_t* dst, int dims, int group,
                                          uint32_t codes) {
  auto* words = reinterpret_cast<uint32_t*>(dst);
  if constexpr (Bits == 4) {
    uint32_t upper = __shfl_xor(codes, 1, kWave);
    if (group % 2 == 0) words[group / 2] = codes | (upper << 4);
  } else {
    uint32_t low = (codes & 0x03030303u) << (2 * (group % 4));
    low |= __shfl_xor(low, 1, kWave);
    low |= __shfl_xor(low, 2, kWave);
    if (group % 4 == 0) words[group / 4] = low;
    uint32_t high = ((codes >> 2) & 0x01010101u) << (group % 8);
    high |= __shfl_xor(high, 1, kWave);
    high |= __shfl_xor(high, 2, kWave);
    high |= __shfl_xor(high, 4, kWave);
    if (group % 8 == 0) words[dims / 16 + group / 8] = high;
  }
}

template <int Bits>
__device__ __forceinline__ uint32_t load_four(const uint8_t* src, int dims,
                                              int group) {
  const auto* words = reinterpret_cast<const uint32_t*>(src);
  if constexpr (Bits == 4) {
    return lookup_four<4>((words[group / 2] >> (4 * (group % 2))) &
                          0x0F0F0F0Fu);
  } else {
    uint32_t low = (words[group / 4] >> (2 * (group % 4))) & 0x03030303u;
    uint32_t high = (words[dims / 16 + group / 8] >> (group % 8)) & 0x01010101u;
    return lookup_four<3>(low | (high << 2));
  }
}

__device__ __forceinline__ float energy_four(const float* x) {
  return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
}

__device__ __forceinline__ __half encode_scale(float energy, float dot) {
  return __float2half_rn(dot > 0 ? (energy / fmaxf(dot, 1e-30f)) * 64.f : 0.f);
}

template <int Bits>
__device__ __forceinline__ void quantize_head(const float (&x)[kLaneValues],
                                              uint8_t* dst, __half* scale,
                                              int lane) {
  float energy = sum_lanes<kWave>(energy_four(x) + energy_four(x + 4));
  float rms = fmaxf(sqrtf(energy / kHead), 1e-12f);
  uint32_t a = quantize_four<Bits>(x, rms);
  uint32_t b = quantize_four<Bits>(x + 4, rms);
  float dot = sum_lanes<kWave>(dot_four(x, lookup_four<Bits>(a)) +
                               dot_four(x + 4, lookup_four<Bits>(b)));
  if (lane == 0) *scale = encode_scale(energy, dot);
  pack_four<Bits>(dst, kHead, lane, a);
  pack_four<Bits>(dst, kHead, lane + kWave, b);
}

template <int Bits>
__device__ __forceinline__ void quantize_block(const float* x, uint8_t* dst,
                                               int dims, int group,
                                               __half* scale, int lane) {
  float energy = sum_lanes<16>(energy_four(x));
  float rms = fmaxf(sqrtf(energy / kRope), 1e-12f);
  uint32_t codes = quantize_four<Bits>(x, rms);
  float dot = sum_lanes<16>(dot_four(x, lookup_four<Bits>(codes)));
  if (lane % 16 == 0) *scale = encode_scale(energy, dot);
  pack_four<Bits>(dst, dims, group, codes);
}

template <typename T, typename F>
__global__ void cache_store(Activation key, Activation value, PagedCache cache,
                            const int64_t* slots, const uint32_t* signs) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int lane = threadIdx.x;
  int64_t slot = slots[token];
  if (slot < 0) return;
  uint8_t* dst = cache.at(slot / cache.page_size, head, slot % cache.page_size);
  auto* scales = reinterpret_cast<__half*>(dst + F::scale_offset);
  float k[kLaneValues], v[kLaneValues];
  load_activation<T>(k, key, token, head, lane);
  load_activation<T>(v, value, token, head, lane);
  flip_signs(k, signs, lane);
  flip_signs(v, signs, lane);
  hadamard<F::compact>(k, lane);
  hadamard<true>(v, lane);
  if constexpr (F::compact) {
    quantize_head<3>(k, dst, scales, lane);
  } else {
    if (lane < 16) {
      quantize_block<4>(k, dst, kRope, lane, scales, lane);
    } else {
      quantize_block<3>(k, dst + F::k3_offset, F::k3_dims, lane - 16,
                        scales + 1, lane);
    }
    quantize_block<3>(k + 4, dst + F::k3_offset, F::k3_dims, lane + 16,
                      scales + 2 + lane / 16, lane);
  }
  quantize_head<F::v_bits>(v, dst + F::v_offset, scales + F::v_scale, lane);
  if (F::payload + lane < F::slot_bytes) dst[F::payload + lane] = 0;
}

template <typename F>
__device__ __forceinline__ float packed_qk(const float* query,
                                           const uint8_t* slot, int lane) {
  const auto* scales = reinterpret_cast<const __half*>(slot + F::scale_offset);
  if constexpr (F::compact) {
    uint32_t a = load_four<3>(slot, kHead, lane);
    uint32_t b = load_four<3>(slot, kHead, lane + kWave);
    return (dot_four(query, a) + dot_four(query + 4, b)) *
           (__half2float(scales[0]) / 64.f);
  } else {
    uint32_t a = lane < 16
                     ? load_four<4>(slot, kRope, lane)
                     : load_four<3>(slot + F::k3_offset, F::k3_dims, lane - 16);
    uint32_t b = load_four<3>(slot + F::k3_offset, F::k3_dims, lane + 16);
    return (dot_four(query, a) * __half2float(scales[lane / 16]) +
            dot_four(query + 4, b) * __half2float(scales[2 + lane / 16])) /
           64.f;
  }
}

template <typename F>
__device__ __forceinline__ void packed_pv(float* out, const uint8_t* slot,
                                          int lane, float rescale,
                                          float weight) {
  const auto* scales = reinterpret_cast<const __half*>(slot + F::scale_offset);
  float scaled_weight = weight * (__half2float(scales[F::v_scale]) / 64.f);
#pragma unroll
  for (int i = 0; i < kLaneValues; i += 4) {
    uint32_t v = load_four<F::v_bits>(slot + F::v_offset, kHead, lane + i * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[i + j] = fmaf(scaled_weight, signed_byte(v, j), out[i + j] * rescale);
    }
  }
}

__device__ __forceinline__ int request_for_query(const int32_t* starts,
                                                 int requests, int q) {
  int lo = 0, hi = requests;
  while (lo + 1 < hi) {
    int mid = lo + (hi - lo) / 2;
    if (starts[mid] <= q)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

template <int Bits>
__device__ __forceinline__ uint32_t load_four(uint32_t words, int word_offset,
                                              int dims, int group) {
  // Every active lane supplies one contiguous word. Exchange before any
  // half-wave codebook selection, so a source lane is always active.
  if constexpr (Bits == 4) {
    uint32_t packed = octave_index(words, word_offset + group / 2, kWave);
    return lookup_four<4>((packed >> (4 * (group % 2))) & 0x0F0F0F0Fu);
  } else {
    uint32_t low_word = octave_index(words, word_offset + group / 4, kWave);
    uint32_t high_word =
        octave_index(words, word_offset + dims / 16 + group / 8, kWave);
    uint32_t low = (low_word >> (2 * (group % 4))) & 0x03030303u;
    uint32_t high = (high_word >> (group % 8)) & 0x01010101u;
    return lookup_four<3>(low | (high << 2));
  }
}

__device__ __forceinline__ float scale_half(uint32_t word, int half) {
  return __half2float(__ushort_as_half(uint16_t(word >> (16 * half))));
}

template <typename F>
__device__ __forceinline__ float packed_qk(const float* query,
                                           const uint8_t* slot, int lane,
                                           float& value_scale) {
  constexpr int key_words = F::v_offset / 4;
  constexpr int scale_words = (F::payload - F::scale_offset + 3) / 4;
  // K occupies lanes [0,key_words); the remaining lanes supply packed FP16
  // scales. Tail lanes repeat the last valid scale word. All loads need only
  // four-byte alignment, and the final scale word fits inside each wire slot.
  int word_index =
      lane < key_words
          ? lane
          : F::scale_offset / 4 + min(lane - key_words, scale_words - 1);
  uint32_t words = reinterpret_cast<const uint32_t*>(slot)[word_index];
  if constexpr (F::compact) {
    uint32_t a = load_four<3>(words, 0, kHead, lane);
    uint32_t b = load_four<3>(words, 0, kHead, lane + kWave);
    uint32_t scales = octave_index(words, key_words, kWave);
    value_scale = scale_half(scales, 1);
    return (dot_four(query, a) + dot_four(query + 4, b)) *
           (scale_half(scales, 0) / 64.f);
  } else {
    uint32_t a4 = load_four<4>(words, 0, kRope, lane % 16);
    uint32_t a3 = load_four<3>(words, F::k3_offset / 4, F::k3_dims, lane % 16);
    uint32_t b = load_four<3>(words, F::k3_offset / 4, F::k3_dims, lane + 16);
    uint32_t a = lane < 16 ? a4 : a3;
    float scale_a =
        scale_half(octave_index(words, key_words, kWave), lane / 16);
    float scale_b =
        scale_half(octave_index(words, key_words + 1, kWave), lane / 16);
    value_scale = scale_half(octave_index(words, key_words + 2, kWave), 0);
    return (dot_four(query, a) * scale_a + dot_four(query + 4, b) * scale_b) /
           64.f;
  }
}

template <typename F>
__device__ __forceinline__ void packed_pv(float* out, const uint8_t* slot,
                                          int lane, float rescale, float weight,
                                          float value_scale) {
  constexpr int value_words = kHead * F::v_bits / 32;
  // V needs 24 or 32 contiguous words; inactive tail suppliers repeat the
  // final valid word, with no read past the packed V payload.
  uint32_t words = reinterpret_cast<const uint32_t*>(
      slot + F::v_offset)[min(lane, value_words - 1)];
  float scaled_weight = weight * (value_scale / 64.f);
#pragma unroll
  for (int i = 0; i < kLaneValues; i += 4) {
    uint32_t v = load_four<F::v_bits>(words, 0, kHead, lane + i * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[i + j] = fmaf(scaled_weight, signed_byte(v, j), out[i + j] * rescale);
    }
  }
}

// Reuse the rotated-query registers for two signed-byte words and their
// reconstruction factors. Each four-dimensional group has its own maximum;
// this avoids adding a wave reduction or mixing different K segment scales.
// A query group reconstructs as byte * (maximum / 127). Predivide that factor
// by 64 here because the wire format reconstructs K as centroid * scale / 64.
__device__ __forceinline__ bool prepare_integer_qk(float (&query)[kLaneValues],
                                                   float attention_scale) {
  float maxima[2];
  bool safe = true;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    float maximum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x = query[4 * group + i];
      maximum = fmaxf(maximum, fabsf(x));
      safe &= __builtin_isfinite(x);
    }
    maxima[group] = maximum;
    // Keep FP32 for tiny ranges whose reciprocal can overflow, and for large
    // scaled queries where byte rounding can be amplified by sharp softmax.
    safe &= maximum == 0 ||
            (maximum >= 1e-30f && maximum * fabsf(attention_scale) <= 2.f);
  }
  if (!__all(safe)) return false;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    float maximum = maxima[group];
    float inverse = maximum > 0 ? 127.f / maximum : 0.f;
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int byte =
          max(-127, min(127, __float2int_rn(query[4 * group + i] * inverse)));
      word |= uint32_t(uint8_t(byte)) << (8 * i);
    }
    query[4 * group] = __uint_as_float(word);
    query[4 * group + 1] = maximum * (1.f / (127.f * 64.f));
    query[4 * group + 2] = query[4 * group + 3] = 0;
  }
  return true;
}

__device__ __forceinline__ int signed_dot_four(uint32_t a, uint32_t b) {
  // The installed gfx12 sudot4 intrinsic has explicit operand signedness.
  // Both true flags select signed int8; the ISA mnemonic's "iu8" is not an
  // unsigned-only contract. Four products fit exactly in an int32 accumulator.
  return __builtin_amdgcn_sudot4(true, int(a), true, int(b), 0, false);
}

template <typename F>
__device__ __forceinline__ float packed_qk(const float* query,
                                           const uint8_t* slot, int lane,
                                           bool integer_qk) {
  const auto* scales = reinterpret_cast<const __half*>(slot + F::scale_offset);
  if constexpr (F::compact) {
    uint32_t a = load_four<3>(slot, kHead, lane);
    uint32_t b = load_four<3>(slot, kHead, lane + kWave);
    if (integer_qk) {
      float da =
          float(signed_dot_four(__float_as_uint(query[0]), a)) * query[1];
      float db =
          float(signed_dot_four(__float_as_uint(query[4]), b)) * query[5];
      // The query factors differ, even though compact K has one stored scale.
      return (da + db) * __half2float(scales[0]);
    }
    return (dot_four(query, a) + dot_four(query + 4, b)) *
           (__half2float(scales[0]) / 64.f);
  } else {
    uint32_t a = lane < 16
                     ? load_four<4>(slot, kRope, lane)
                     : load_four<3>(slot + F::k3_offset, F::k3_dims, lane - 16);
    uint32_t b = load_four<3>(slot + F::k3_offset, F::k3_dims, lane + 16);
    if (integer_qk) {
      float da =
          float(signed_dot_four(__float_as_uint(query[0]), a)) * query[1];
      float db =
          float(signed_dot_four(__float_as_uint(query[4]), b)) * query[5];
      // Noncompact K has four independent 64-dimensional scales; lane / 16
      // selects segments 0/1 (including 4-bit segment 0), then segments 2/3.
      return da * __half2float(scales[lane / 16]) +
             db * __half2float(scales[2 + lane / 16]);
    }
    return (dot_four(query, a) * __half2float(scales[lane / 16]) +
            dot_four(query + 4, b) * __half2float(scales[2 + lane / 16])) /
           64.f;
  }
}

template <bool WholeHead>
__device__ __forceinline__ void hadamard_legacy(float (&x)[kLaneValues],
                                                int lane) {
#pragma unroll
  for (int i = 0; i < kLaneValues; i += 4) {
    butterfly(x[i], x[i + 1]);
    butterfly(x[i + 2], x[i + 3]);
    butterfly(x[i], x[i + 2]);
    butterfly(x[i + 1], x[i + 3]);
  }
#pragma unroll
  for (int delta = 1; delta <= (WholeHead ? 16 : 8); delta *= 2) {
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      float other = __shfl_xor(x[i], delta, kWave);
      x[i] = (lane & delta) ? other - x[i] : x[i] + other;
    }
  }
  if constexpr (WholeHead) {
#pragma unroll
    for (int i = 0; i < 4; ++i) butterfly(x[i], x[i + 4]);
  }
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) x[i] *= WholeHead ? 1.f / 16 : 1.f / 8;
}

enum class DecodePath { Scalar, Blockwise, Cooperative, Dot4 };

template <typename T, typename F, bool Prefill,
          DecodePath Path = DecodePath::Scalar>
__device__ __forceinline__ void attention_row(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* map_or_starts, const int32_t* lengths,
    int requests, const uint32_t* signs, float scale, int q, int head,
    int split, int heads, int splits, int lane, int query_tokens,
    int prefix_override = -1) {
  int req = 0, prefix = 0, length = 0, chunk_start = 0;
  if constexpr (Prefill) {
    req = request_for_query(map_or_starts, requests, q);
    chunk_start = map_or_starts[req];
    prefix = prefix_override >= 0
                 ? prefix_override
                 : lengths[req] - (map_or_starts[req + 1] - chunk_start);
    length = prefix + q - chunk_start + 1;
  } else {
    length = max(0, lengths[q]);
    prefix = length;
    if (length > 0) req = map_or_starts[q];
  }

  int active_splits =
      (Prefill || heads / cache.heads != 6)
          ? splits
          : active_decode_splits(length, splits, query_tokens, cache.heads);
  if (split >= active_splits) return;
  int tokens_per_split = (length + active_splits - 1) / active_splits;
  int begin = split * tokens_per_split;
  int end = min(begin + tokens_per_split, length);
  int kv_head = head / (heads / cache.heads);

  float packed_out[kLaneValues] = {}, chunk_out[kLaneValues] = {};
  float running_max = -INFINITY, mass = 0;
  if (begin < end) {
    float raw_q[kLaneValues], rotated_q[kLaneValues];
    bool integer_qk = false;
    load_activation<T>(raw_q, query, q, head, lane);
    if (begin < prefix) {
#pragma unroll
      for (int i = 0; i < kLaneValues; ++i) rotated_q[i] = raw_q[i];
      flip_signs(rotated_q, signs, lane);
      if constexpr (Path == DecodePath::Dot4)
        hadamard_legacy<F::compact>(rotated_q, lane);
      else
        hadamard<F::compact>(rotated_q, lane);
      if constexpr (Path == DecodePath::Dot4) {
        integer_qk = prepare_integer_qk(rotated_q, scale);
      }
    }
    if constexpr (Path == DecodePath::Blockwise) {
      // Buffer only eight logits and their actual paged addresses. QK remains
      // wave32-cooperative, but the online state advances once per microtile.
      constexpr int kTileTokens = 8;
      for (int tile_begin = begin; tile_begin < end;
           tile_begin += kTileTokens) {
        float logits[kTileTokens];
        const uint8_t* slots[kTileTokens];
        float tile_max = -INFINITY;
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
          int pos = tile_begin + t;
          const uint8_t* slot = nullptr;
          float logit = -INFINITY;
          // A split can start inside a page, and a tile can straddle either a
          // page boundary or the packed-prefix/raw-chunk boundary.
          if (pos < end) {
            logit = 0;
            if (!Prefill || pos < prefix) {
              int page = table[req * table_stride + pos / cache.page_size];
              slot = cache.at(page, kv_head, pos % cache.page_size);
              logit = packed_qk<F>(rotated_q, slot, lane);
            } else {
              float raw_k[kLaneValues];
              load_activation<T>(raw_k, key, chunk_start + pos - prefix,
                                 kv_head, lane);
#pragma unroll
              for (int i = 0; i < kLaneValues; ++i)
                logit = fmaf(raw_q[i], raw_k[i], logit);
            }
            logit = sum_lanes<kWave>(logit) * scale;
          }
          slots[t] = slot;
          logits[t] = logit;
          tile_max = fmaxf(tile_max, logit);
        }

        float next_max = fmaxf(running_max, tile_max);
        float rescale = __expf(running_max - next_max);
        float tile_mass = 0;
        float tile_packed[kLaneValues] = {}, tile_chunk[kLaneValues] = {};
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
          int pos = tile_begin + t;
          if (pos < end) {
            float weight = __expf(logits[t] - next_max);
            tile_mass += weight;
            if (slots[t]) {
              packed_pv<F>(tile_packed, slots[t], lane, 1.f, weight);
            } else if constexpr (Prefill) {
              float raw_v[kLaneValues];
              load_activation<T>(raw_v, value, chunk_start + pos - prefix,
                                 kv_head, lane);
#pragma unroll
              for (int i = 0; i < kLaneValues; ++i) {
                tile_chunk[i] = fmaf(weight, raw_v[i], tile_chunk[i]);
              }
            }
          }
        }
        // Both coordinate systems share this maximum and normalization, even
        // when the microtile contains the last packed and first raw tokens.
        mass = fmaf(mass, rescale, tile_mass);
        running_max = next_max;
#pragma unroll
        for (int i = 0; i < kLaneValues; ++i) {
          packed_out[i] = fmaf(packed_out[i], rescale, tile_packed[i]);
          if constexpr (Prefill)
            chunk_out[i] = fmaf(chunk_out[i], rescale, tile_chunk[i]);
        }
      }
    } else {
      for (int pos = begin; pos < end; ++pos) {
        const uint8_t* slot = nullptr;
        float logit = 0, value_scale = 0;
        if (!Prefill || pos < prefix) {
          int page = table[req * table_stride + pos / cache.page_size];
          slot = cache.at(page, kv_head, pos % cache.page_size);
          if constexpr (Path == DecodePath::Cooperative) {
            logit = packed_qk<F>(rotated_q, slot, lane, value_scale);
          } else if constexpr (Path == DecodePath::Dot4) {
            logit = packed_qk<F>(rotated_q, slot, lane, integer_qk);
          } else {
            logit = packed_qk<F>(rotated_q, slot, lane);
          }
        } else {
          float raw_k[kLaneValues];
          load_activation<T>(raw_k, key, chunk_start + pos - prefix, kv_head,
                             lane);
#pragma unroll
          for (int i = 0; i < kLaneValues; ++i)
            logit = fmaf(raw_q[i], raw_k[i], logit);
        }
        logit = sum_lanes<kWave>(logit) * scale;
        float next_max = fmaxf(running_max, logit);
        float rescale = __expf(running_max - next_max);
        float weight = __expf(logit - next_max);
        mass = fmaf(mass, rescale, weight);
        running_max = next_max;
        if (slot) {
          if constexpr (Path == DecodePath::Cooperative) {
            packed_pv<F>(packed_out, slot, lane, rescale, weight, value_scale);
          } else {
            packed_pv<F>(packed_out, slot, lane, rescale, weight);
          }
        } else if constexpr (Prefill) {
          float raw_v[kLaneValues];
          load_activation<T>(raw_v, value, chunk_start + pos - prefix, kv_head,
                             lane);
#pragma unroll
          for (int i = 0; i < kLaneValues; ++i) {
            packed_out[i] *= rescale;
            chunk_out[i] = fmaf(weight, raw_v[i], chunk_out[i] * rescale);
          }
        }
      }
    }
    if (begin < prefix) {
      if constexpr (Path == DecodePath::Dot4)
        hadamard_legacy<true>(packed_out, lane);
      else
        hadamard<true>(packed_out, lane);
      flip_signs(packed_out, signs, lane);
    }
  }
  int64_t row = int64_t(q) * heads + head;
  if (splits == 1) {
    float inv_mass = mass > 0 ? 1.f / mass : 0;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      out[row * kHead + dim_at(lane, i)] =
          (packed_out[i] + chunk_out[i]) * inv_mass;
    }
  } else {
    float* partial = partials + (row * splits + split) * kPartialSize;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      partial[dim_at(lane, i)] = packed_out[i] + chunk_out[i];
    }
    if (lane == 0) {
      partial[kHead] = running_max;
      partial[kHead + 1] = mass;
    }
  }
}

// gfx12 wave32 has eight distinct input elements per lane (no gfx11
// replication). A is column-fragmented, B and FP32 C are row-fragmented:
// A[lane%16][8*(lane/16)+e], B[8*(lane/16)+e][lane%16].
using WmmaHalf8 = _Float16 __attribute__((ext_vector_type(8)));
using WmmaFloat8 = float __attribute__((ext_vector_type(8)));
using WmmaInt2 = int __attribute__((ext_vector_type(2)));
using WmmaInt8 = int __attribute__((ext_vector_type(8)));
constexpr int kQkGroup = 64;
constexpr int kQkGroups = kHead / kQkGroup;
static_assert(kQkGroup >= 16 && 64 % kQkGroup == 0,
              "align integer partials with K scales");

// A 16-token wire tile uses eight DWORD registers per thread. Each wave owns
// four tokens; lanes cooperatively read the complete aligned compressed slot.
// Exchange both banks before choosing one: lane-dependent source-bank selection
// before a shuffle would corrupt values supplied by another lane.
template <int First, int Last>
__device__ __forceinline__ uint32_t staged_word(uint32_t lo, uint32_t hi,
                                                int index) {
  if constexpr (Last < 32) return octave_index(lo, index, kWave);
  if constexpr (First >= 32) return octave_index(hi, index - 32, kWave);
  uint32_t a = octave_index(lo, index & 31, kWave);
  uint32_t b = octave_index(hi, index & 31, kWave);
  return index < 32 ? a : b;
}

template <int Bits, int Offset, int Dims>
__device__ __forceinline__ uint32_t staged_four(uint32_t lo, uint32_t hi,
                                                int group) {
  constexpr int Last = Offset + Dims * Bits / 32 - 1;
  if constexpr (Bits == 4) {
    uint32_t word = staged_word<Offset, Last>(lo, hi, Offset + group / 2);
    return lookup_four<4>((word >> (4 * (group % 2))) & 0x0F0F0F0Fu);
  } else {
    uint32_t low = staged_word<Offset, Last>(lo, hi, Offset + group / 4);
    uint32_t high =
        staged_word<Offset, Last>(lo, hi, Offset + Dims / 16 + group / 8);
    return lookup_four<3>(((low >> (2 * (group % 4))) & 0x03030303u) |
                          (((high >> (group % 8)) & 0x01010101u) << 2));
  }
}

template <typename F>
__device__ __forceinline__ void stage_decode_tile(
    uint32_t (&lo)[4], uint32_t (&hi)[4], int count, int wave, int lane,
    PagedCache cache, const int32_t* pages, int page_index, int offset,
    const uint8_t* page, int head) {
  const bool page_local = count <= cache.page_size - offset;
  const uint8_t* tile = page + offset * cache.token_stride;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int token = wave + 4 * r;
    lo[r] = hi[r] = 0;
    if (token < count) {
      const uint8_t* slot;
      if (page_local) {
        slot = tile + token * cache.token_stride;
      } else {
        const int position = offset + token;
        const int next = position / cache.page_size;
        slot = cache.at(pages[page_index + next], head,
                        position - next * cache.page_size);
      }
      const auto* words = reinterpret_cast<const uint32_t*>(slot);
      lo[r] = words[lane];
      if (lane + 32 < F::slot_bytes / 4) hi[r] = words[lane + 32];
    }
  }
}

// Only LDS is exchanged by the waves during the tile loop.
__device__ inline __attribute__((convergent)) void oneq_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Two half elements of row padding avoid the 256-column LDS bank stride.
// Four waves unpack each KV tile and compute disjoint QK shards and PV
// columns. Wave zero combines the QK shards and updates FP32 softmax.
struct Fp16OneqShared {
  union Workspace {
    _Float16 prepare[16][258];
    struct {
      float partial[4][16][16];
      _Float16 key[16][258], value[16][258];
    } loop;
    float output[16][256];
  } work;
  _Float16 probability[16][16];
  float rescale[16], maximum[16], mass[16];
  int fallback;
};
static_assert(sizeof(Fp16OneqShared) == 21316, "expected FP16 one-query LDS");

template <typename T, typename F>
__global__ __launch_bounds__(128) void attention_wmma_fp16_oneq(
    float* out, float* partials, Activation query, PagedCache cache,
    const int32_t* table, int64_t table_stride, const int32_t* mapping,
    const int32_t* lengths, const uint32_t* signs, float scale,
    int tokens, int heads) {
  const int q = blockIdx.x, kv_head = blockIdx.y;
  const int split = blockIdx.z, splits = gridDim.z;
  const int length = max(0, lengths[q]);
  const int active = active_decode_splits(length, splits, tokens, cache.heads);
  if (split >= active || length == 0) return;
  const int req = mapping[q];
  const int per_split = (length + active - 1) / active;
  const int begin = split * per_split, end = min(begin + per_split, length);
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int column = lane % 16, half_wave = lane / 16;
  __shared__ Fp16OneqShared shared;
  if (tid == 0) shared.fallback = 0;
  if (tid < 16) {
    shared.maximum[tid] = -INFINITY;
    shared.mass[tid] = 0;
  }
  __syncthreads();
  for (int row = wave; row < 16; row += 4) {
    float rotated[kLaneValues] = {};
    if (row < 6)
      load_activation<T>(rotated, query, q, kv_head * 6 + row, lane);
    flip_signs(rotated, signs, lane);
    hadamard<F::compact>(rotated, lane);
    bool safe = true;
    float maxima[2] = {};
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      maxima[i / 4] = fmaxf(maxima[i / 4], fabsf(rotated[i]));
      safe &= __builtin_isfinite(rotated[i]);
    }
#pragma unroll
    for (int delta = 1; delta < kQkGroup / 4; delta *= 2) {
      maxima[0] = fmaxf(maxima[0], octave_xor<false>(maxima[0], delta, kWave));
      maxima[1] = fmaxf(maxima[1], octave_xor<false>(maxima[1], delta, kWave));
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
      safe &= maxima[i] == 0 ||
              (maxima[i] >= 1e-30f && maxima[i] <= 65504.f &&
               maxima[i] * fabsf(scale) <= 2.f);
    if (!__all(safe) && lane == 0 && row < 6)
      atomicOr(&shared.fallback, 1 << row);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      shared.work.prepare[row][dim_at(lane, i)] = (_Float16)rotated[i];
  }
  __syncthreads();
  if (shared.fallback == 63) {
    for (int row = wave; row < 6; row += 4) {
      int64_t output_row = int64_t(q) * heads + kv_head * 6 + row;
      float* dst = splits == 1 ? out + output_row * kHead
                              : partials + (output_row * splits + split) * kPartialSize;
      if (lane == 0) dst[0] = __builtin_nanf("");
    }
    return;
  }
  WmmaHalf8 query_fragment[4];
#pragma unroll
  for (int step = 0; step < 4; ++step) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      query_fragment[step][e] =
          shared.work.prepare[column][wave * 64 + step * 16 + half_wave * 8 + e];
  }
  __syncthreads();
  WmmaFloat8 packed[4] = {}, running_max, running_mass = {};
#pragma unroll
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;
  // Keep the page lookup and division out of the common in-page tile.
  const int32_t* pages = table + int64_t(req) * table_stride;
  int page_index = begin / cache.page_size;
  int offset = begin - page_index * cache.page_size;
  const uint8_t* page =
      begin < end ? cache.at(pages[page_index], kv_head, 0) : nullptr;
  uint32_t staged_lo[4], staged_hi[4];
  if (begin < end)
    stage_decode_tile<F>(staged_lo, staged_hi, min(16, end - begin), wave, lane,
                         cache, pages, page_index, offset, page, kv_head);
  for (int pos_base = begin; pos_base < end; pos_base += 16) {
    const int count = min(16, end - pos_base);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int t = wave + 4 * r;
      float k[kLaneValues] = {}, v[kLaneValues] = {};
      if (t < count) {
        uint32_t lo = staged_lo[r], hi = staged_hi[r];
        constexpr int scale_word = F::scale_offset / 4;
        uint32_t scale0 = staged_word<scale_word, scale_word>(lo, hi, scale_word);
        uint32_t a, b;
        float sa, sb;
        if constexpr (F::compact) {
          a = staged_four<3, 0, kHead>(lo, hi, lane);
          b = staged_four<3, 0, kHead>(lo, hi, lane + kWave);
          sa = sb = scale_half(scale0, 0) / 64.f;
        } else {
          uint32_t a4 = staged_four<4, 0, kRope>(lo, hi, lane % 16);
          uint32_t a3 = staged_four<3, F::k3_offset / 4, F::k3_dims>(lo, hi, lane % 16);
          a = lane < 16 ? a4 : a3;
          b = staged_four<3, F::k3_offset / 4, F::k3_dims>(lo, hi, lane + 16);
          uint32_t scale1 = staged_word<scale_word + 1, scale_word + 1>(lo, hi, scale_word + 1);
          sa = scale_half(scale0, lane / 16) / 64.f;
          sb = scale_half(scale1, lane / 16) / 64.f;
        }
        constexpr int value_scale_word = scale_word + F::v_scale / 2;
        uint32_t scales = staged_word<value_scale_word, value_scale_word>(lo, hi, value_scale_word);
        float sv = scale_half(scales, F::v_scale % 2) / 64.f;
        uint32_t va = staged_four<F::v_bits, F::v_offset / 4, kHead>(lo, hi, lane);
        uint32_t vb = staged_four<F::v_bits, F::v_offset / 4, kHead>(lo, hi, lane + kWave);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          k[i] = signed_byte(a, i) * sa;
          k[i + 4] = signed_byte(b, i) * sb;
          v[i] = signed_byte(va, i) * sv;
          v[i + 4] = signed_byte(vb, i) * sv;
        }
      }
#pragma unroll
      for (int i = 0; i < kLaneValues; ++i) {
        shared.work.loop.key[t][dim_at(lane, i)] = (_Float16)k[i];
        shared.work.loop.value[t][dim_at(lane, i)] = (_Float16)v[i];
      }
    }
    oneq_lds_barrier();
    // The current tile is in LDS, so its wire registers can hold the next
    // tile while QK, softmax, and PV consume the expanded operands.
    const int next_pos = pos_base + 16;
    if (next_pos < end) {
      offset += 16;
      if (offset >= cache.page_size) {
        const int crossed = cache.page_size >= 16 ? 1 : offset / cache.page_size;
        page_index += crossed;
        offset -= crossed * cache.page_size;
        page = cache.at(pages[page_index], kv_head, 0);
      }
      stage_decode_tile<F>(staged_lo, staged_hi, min(16, end - next_pos),
                           wave, lane, cache, pages, page_index, offset,
                           page, kv_head);
    }
    WmmaFloat8 dots = {};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      WmmaHalf8 b;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        b[e] = shared.work.loop.key[column][wave * 64 + step * 16 + half_wave * 8 + e];
      dots = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(query_fragment[step], b, dots);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e)
      shared.work.loop.partial[wave][half_wave * 8 + e][column] = dots[e];
    oneq_lds_barrier();
    if (wave == 0) {
      WmmaFloat8 score = {};
#pragma unroll
      for (int shard = 0; shard < 4; ++shard) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          score[e] += shared.work.loop.partial[shard][half_wave * 8 + e][column];
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e;
        bool live = row < 6 && column < count;
        float s = live ? score[e] * scale : -INFINITY;
        float m = s;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          m = fmaxf(m, octave_xor<false>(m, delta, kWave));
        float next_max = fmaxf(running_max[e], m);
        float rescale = running_mass[e] > 0 ? __expf(running_max[e] - next_max) : 0;
        float weight = live ? __expf(s - next_max) : 0;
        float tile_mass = weight;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          tile_mass += octave_xor<false>(tile_mass, delta, kWave);
        running_mass[e] = fmaf(running_mass[e], rescale, tile_mass);
        running_max[e] = next_max;
        shared.probability[row][column] = (_Float16)weight;
        if (column == 0) {
          shared.rescale[row] = rescale;
          shared.maximum[row] = next_max;
          shared.mass[row] = running_mass[e];
        }
      }
    }
    oneq_lds_barrier();
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      WmmaHalf8 a, b;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e;
        packed[d][e] *= shared.rescale[row];
        a[e] = shared.probability[column][half_wave * 8 + e];
        b[e] = shared.work.loop.value[half_wave * 8 + e][wave * 64 + d * 16 + column];
      }
      packed[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, packed[d]);
    }
    oneq_lds_barrier();
  }
#pragma unroll
  for (int d = 0; d < 4; ++d) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      shared.work.output[half_wave * 8 + e][wave * 64 + d * 16 + column] = packed[d][e];
  }
  __syncthreads();
  for (int row = wave; row < 6; row += 4) {
    int64_t output_row = int64_t(q) * heads + kv_head * 6 + row;
    float* dst = splits == 1 ? out + output_row * kHead
                            : partials + (output_row * splits + split) * kPartialSize;
    if (shared.fallback & (1 << row)) {
      if (lane == 0) dst[0] = __builtin_nanf("");
      continue;
    }
    float result[kLaneValues];
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      result[i] = shared.work.output[row][dim_at(lane, i)];
    hadamard<true>(result, lane);
    flip_signs(result, signs, lane);
    float mass = shared.mass[row];
    float inv_mass = mass > 0 && __builtin_isfinite(mass) ? 1.f / mass : __builtin_nanf("");
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      dst[dim_at(lane, i)] = splits == 1 ? result[i] * inv_mass : result[i];
    if (splits > 1 && lane == 0) {
      dst[kHead] = shared.maximum[row];
      dst[kHead + 1] = mass;
    }
  }
}


// Six-query prefill with four-wave FP16 QK and a single staged KV workspace.
struct Fp16ReuseShared {
  union Workspace {
    _Float16 prepare[3][16][258];
    struct {
      float partial[3][4][16][16];
      _Float16 key[16][258], value[16][258];
    } loop;
    struct { float packed[16][256], chunk[12][256]; } output;
  } work;
  _Float16 probability[3][16][16];
  float rescale[3][16], maximum[3][16], mass[3][16];
  int req[6], length[6], prefix[6], chunk_start[6], begin[6], end[6];
  int fallback[3];
};
static_assert(sizeof(Fp16ReuseShared) == 31068, "expected FP16 reuse layout");

template <int Rows>
__device__ __forceinline__ bool reuse_live(int pos, const int* begins,
                                           const int* ends) {
  bool live = false;
#pragma unroll
  for (int i = 0; i < Rows; ++i) live |= pos >= begins[i] && pos < ends[i];
  return live;
}

template <typename F, int Pairs>
__device__ __forceinline__ void stage_reuse_tile(
    uint32_t (&lo)[4], uint32_t (&hi)[4], int first, int last, int prefix,
    int wave, int lane, PagedCache cache, const int32_t* table,
    int64_t table_stride, int req, int head, const int* begins,
    const int* ends) {
  if (first >= last) return;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    int pos = first + wave + 4 * r;
    lo[r] = hi[r] = 0;
    if (pos < last && pos < prefix &&
        reuse_live<2 * Pairs>(pos, begins, ends)) {
      int page = table[req * table_stride + pos / cache.page_size];
      const auto* words = reinterpret_cast<const uint32_t*>(
          cache.at(page, head, pos % cache.page_size));
      lo[r] = words[lane];
      if (lane + 32 < F::slot_bytes / 4) hi[r] = words[lane + 32];
    }
  }
}

template <typename T, typename F, bool Prefill>
__global__ __launch_bounds__(384) void attention_wmma_fp16_reuse(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* map_or_starts, const int32_t* lengths,
    int requests, const uint32_t* signs, float scale, int tokens, int heads) {
  __shared__ Fp16ReuseShared shared;
  const int tid = threadIdx.x, lane = tid % kWave, wave = (tid / kWave) % 4;
  const int pair = tid / 128, local_tid = tid % 128;
  const int column = lane % 16, half_wave = lane / 16;
  const int q_base = 6 * blockIdx.x + 2 * pair, kv_head = blockIdx.y;
  const int split = blockIdx.z, splits = gridDim.z;
  if (tid < 3) shared.fallback[tid] = 0;
  if (tid < 6) {
    int q = 6 * blockIdx.x + tid, req = 0, prefix = 0, length = 0,
        chunk_start = 0;
    if (q < tokens) {
      if constexpr (Prefill) {
        req = request_for_query(map_or_starts, requests, q);
        chunk_start = map_or_starts[req];
        prefix = lengths[req] - (map_or_starts[req + 1] - chunk_start);
        length = prefix + q - chunk_start + 1;
      } else {
        length = max(0, lengths[q]);
        prefix = length;
        if (length > 0) req = map_or_starts[q];
      }
    }
    int per_split = (length + splits - 1) / splits;
    shared.req[tid] = req;
    shared.length[tid] = length;
    shared.prefix[tid] = prefix;
    shared.chunk_start[tid] = chunk_start;
    shared.begin[tid] = split * per_split;
    shared.end[tid] = min((split + 1) * per_split, length);
  }
  if (local_tid < 16) {
    shared.maximum[pair][local_tid] = -INFINITY;
    shared.mass[pair][local_tid] = 0;
  }
  __syncthreads();

  // Six adjacent queries can straddle ragged/request boundaries. Such a
  // workgroup uses independent FP32 rows and never aliases their KV pages.
  bool different_requests = false;
#pragma unroll
  for (int i = 1; i < 6; ++i)
    different_requests |=
        shared.length[i] > 0 && shared.req[i] != shared.req[0];
  if (different_requests) {
    for (int row = wave; row < 12; row += 4) {
      int q = q_base + row / 6;
      if (q < tokens)
        attention_row<T, F, Prefill>(
            out, partials, query, key, value, cache, table, table_stride,
            map_or_starts, lengths, requests, signs, scale, q,
            kv_head * 6 + row % 6, split, heads, splits, lane, tokens);
    }
    return;
  }

  const bool packed_queries =
      !Prefill || shared.prefix[shared.length[0] > 0 ? 0 : 1] > 0;
  for (int row = wave; row < 16; row += 4) {
    float raw[kLaneValues] = {}, rotated[kLaneValues];
    int q = q_base + row / 6;
    if (row < 12 && q < tokens)
      load_activation<T>(raw, query, q, kv_head * 6 + row % 6, lane);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) rotated[i] = raw[i];
    if (packed_queries) {
      flip_signs(rotated, signs, lane);
      hadamard<F::compact>(rotated, lane);
    }
    float maxima[2] = {};
    bool safe = true;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      float x = packed_queries ? rotated[i] : raw[i];
      maxima[i / 4] = fmaxf(maxima[i / 4], fabsf(x));
      safe &= __builtin_isfinite(x);
    }
#pragma unroll
    for (int delta = 1; delta < kQkGroup / 4; delta *= 2) {
      maxima[0] = fmaxf(maxima[0], octave_xor<false>(maxima[0], delta, kWave));
      maxima[1] = fmaxf(maxima[1], octave_xor<false>(maxima[1], delta, kWave));
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      safe &= maxima[i] == 0 ||
              (maxima[i] >= 1e-30f && maxima[i] <= 65504.f &&
               maxima[i] * fabsf(scale) <= 2.f);
    }
    if (!__all(safe) && lane == 0 && row < 12)
      atomicOr(&shared.fallback[pair], 1 << row);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      int d = dim_at(lane, i);
      float x = packed_queries ? rotated[i] : raw[i];
      shared.work.prepare[pair][row][d] = (_Float16)x;
    }
  }
  __syncthreads();
  if (shared.fallback[0] == 4095 &&
      shared.fallback[1] == 4095 &&
      shared.fallback[2] == 4095) {
    for (int row = wave; row < 12; row += 4) {
      int q = q_base + row / 6;
      if (q < tokens)
        attention_row<T, F, Prefill>(
            out, partials, query, key, value, cache, table, table_stride,
            map_or_starts, lengths, requests, signs, scale, q,
            kv_head * 6 + row % 6, split, heads, splits, lane, tokens);
    }
    return;
  }

  int first = 0x7fffffff, last = 0;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    if (shared.begin[i] < shared.end[i]) {
      first = min(first, shared.begin[i]);
      last = max(last, shared.end[i]);
    }
  }
  int owner = shared.length[0] > 0 ? 0 : 1;
  int req = shared.req[owner], prefix = shared.prefix[owner];
  int chunk_start = shared.chunk_start[owner];
  WmmaFloat8 packed[4] = {}, chunk[4] = {};
  WmmaFloat8 running_max, running_mass = {};
#pragma unroll
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;

  // Cache four 16-D steps for this wave's contiguous 64-D QK shard.
  WmmaHalf8 query_fragment[4] = {};
#pragma unroll
  for (int step = 0; step < 4; ++step) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int dim = wave * 64 + step * 16 + half_wave * 8 + e;
      query_fragment[step][e] = shared.work.prepare[pair][column][dim];
    }
  }
  __syncthreads();

  uint32_t staged_lo[4], staged_hi[4];
  bool raw_queries_loaded = false;

  for (int pos_base = first; pos_base < last;) {
    bool is_packed = !Prefill || pos_base < prefix;
    int count = min(16, last - pos_base);
    if constexpr (Prefill) {
      if (is_packed) count = min(count, prefix - pos_base);
    }
    if (!is_packed && !raw_queries_loaded) {
#pragma unroll
      for (int step = 0; step < 4; ++step) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int dim = wave * 64 + step * 16 + half_wave * 8 + e;
          int q = q_base + column / 6;
          if (column < 12 && q < tokens) {
            const auto* query_row = static_cast<const T*>(query.data) +
                q * query.token_stride + (kv_head * 6 + column % 6) * query.head_stride;
            query_fragment[step][e] = (_Float16)query_row[dim];
          } else {
            query_fragment[step][e] = (_Float16)0;
          }
        }
      }
      raw_queries_loaded = true;
    }
    if (pair == 0) {
      stage_reuse_tile<F, 3>(staged_lo, staged_hi, pos_base, last, prefix, wave,
                             lane, cache, table, table_stride, req, kv_head,
                             shared.begin, shared.end);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int t = wave + 4 * r;
        int pos = pos_base + t;
        bool live = t < count && reuse_live<6>(pos, shared.begin, shared.end);
        float k[kLaneValues] = {}, v[kLaneValues] = {};
        uint32_t a = 0, b = 0;
        float sa = 0, sb = 0;
        if (live) {
          if (is_packed) {
            uint32_t lo = staged_lo[r], hi = staged_hi[r];
            constexpr int scale_word = F::scale_offset / 4;
            uint32_t scale0 = staged_word<scale_word, scale_word>(lo, hi, scale_word);
            if constexpr (F::compact) {
              a = staged_four<3, 0, kHead>(lo, hi, lane);
              b = staged_four<3, 0, kHead>(lo, hi, lane + kWave);
              sa = sb = scale_half(scale0, 0) / 64.f;
            } else {
              uint32_t a4 = staged_four<4, 0, kRope>(lo, hi, lane % 16);
              uint32_t a3 = staged_four<3, F::k3_offset / 4, F::k3_dims>(lo, hi, lane % 16);
              a = lane < 16 ? a4 : a3;
              b = staged_four<3, F::k3_offset / 4, F::k3_dims>(lo, hi, lane + 16);
              uint32_t scale1 = staged_word<scale_word + 1, scale_word + 1>(lo, hi, scale_word + 1);
              sa = scale_half(scale0, lane / 16) / 64.f;
              sb = scale_half(scale1, lane / 16) / 64.f;
            }
            constexpr int value_scale_word = scale_word + F::v_scale / 2;
            uint32_t scales = staged_word<value_scale_word, value_scale_word>(lo, hi, value_scale_word);
            float sv = scale_half(scales, F::v_scale % 2) / 64.f;
            uint32_t va = staged_four<F::v_bits, F::v_offset / 4, kHead>(lo, hi, lane);
            uint32_t vb = staged_four<F::v_bits, F::v_offset / 4, kHead>(lo, hi, lane + kWave);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              k[i] = signed_byte(a, i) * sa;
              k[i + 4] = signed_byte(b, i) * sb;
              v[i] = signed_byte(va, i) * sv;
              v[i + 4] = signed_byte(vb, i) * sv;
            }
          } else if constexpr (Prefill) {
            int token = chunk_start + pos - prefix;
            load_activation<T>(k, key, token, kv_head, lane);
            load_activation<T>(v, value, token, kv_head, lane);
          }
        }
#pragma unroll
        for (int i = 0; i < kLaneValues; ++i) {
          int d = dim_at(lane, i);
          shared.work.loop.key[t][d] = (_Float16)k[i];
          shared.work.loop.value[t][d] = (_Float16)v[i];
        }
      }
    }
    __work_group_barrier(__CLK_LOCAL_MEM_FENCE);

    WmmaFloat8 dots = {};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      WmmaHalf8 b;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int dim = wave * 64 + step * 16 + half_wave * 8 + e;
        b[e] = shared.work.loop.key[column][dim];
      }
      dots = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
          query_fragment[step], b, dots);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int row = half_wave * 8 + e;
      shared.work.loop.partial[pair][wave][row][column] = dots[e];
    }
    __work_group_barrier(__CLK_LOCAL_MEM_FENCE);

    if (wave == 0) {
      WmmaFloat8 score = {};
#pragma unroll
      for (int shard = 0; shard < 4; ++shard) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int row = half_wave * 8 + e;
          score[e] += shared.work.loop.partial[pair][shard][row][column];
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e, qi = row / 6;
        int pos = pos_base + column;
        bool live = row < 12 && q_base + qi < tokens && column < count &&
                    pos >= shared.begin[2 * pair + qi] &&
                    pos < shared.end[2 * pair + qi];
        float s = live ? score[e] * scale : -INFINITY;
        float m = s;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          m = fmaxf(m, octave_xor<false>(m, delta, kWave));
        float next_max = fmaxf(running_max[e], m);
        float rescale =
            running_mass[e] > 0 ? __expf(running_max[e] - next_max) : 0;
        float weight = live ? __expf(s - next_max) : 0;
        float tile_mass = weight;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          tile_mass += octave_xor<false>(tile_mass, delta, kWave);
        running_mass[e] = fmaf(running_mass[e], rescale, tile_mass);
        running_max[e] = next_max;
        shared.probability[pair][row][column] = (_Float16)weight;
        if (column == 0) {
          shared.rescale[pair][row] = rescale;
          shared.maximum[pair][row] = next_max;
          shared.mass[pair][row] = running_mass[e];
        }
      }
    }
    __work_group_barrier(__CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int d = 0; d < 4; ++d) {
      WmmaHalf8 a, b;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e;
        packed[d][e] *= shared.rescale[pair][row];
        chunk[d][e] *= shared.rescale[pair][row];
        a[e] = shared.probability[pair][column][half_wave * 8 + e];
        b[e] = shared.work.loop.value[half_wave * 8 + e][wave * 64 + d * 16 + column];
      }
      if (is_packed) {
        packed[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, packed[d]);
      } else if constexpr (Prefill) {
        chunk[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, chunk[d]);
      }
    }
    __work_group_barrier(__CLK_LOCAL_MEM_FENCE);
    pos_base += count;
  }

  for (int output_pair = 0; output_pair < 3; ++output_pair) {
    if (pair == output_pair) {
      // Reuse dead operand storage for the FP32 output layout conversion.
#pragma unroll
      for (int d = 0; d < 4; ++d) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int row = half_wave * 8 + e, dim = wave * 64 + d * 16 + column;
          shared.work.output.packed[row][dim] = packed[d][e];
          if (row < 12) shared.work.output.chunk[row][dim] = chunk[d][e];
        }
      }
    }
    __syncthreads();
    if (pair == output_pair) {
      for (int row = wave; row < 12; row += 4) {
        int q = q_base + row / 6;
        if (q >= tokens) continue;
        if (shared.fallback[pair] & (1 << row)) {
          attention_row<T, F, Prefill>(
              out, partials, query, key, value, cache, table, table_stride,
              map_or_starts, lengths, requests, signs, scale, q,
              kv_head * 6 + row % 6, split, heads, splits, lane, tokens);
          continue;
        }
        float packed_out[kLaneValues];
#pragma unroll
        for (int i = 0; i < kLaneValues; ++i)
          packed_out[i] = shared.work.output.packed[row][dim_at(lane, i)];
        hadamard<true>(packed_out, lane);
        flip_signs(packed_out, signs, lane);
        int64_t output_row = int64_t(q) * heads + kv_head * 6 + row % 6;
        float mass = shared.mass[pair][row];
        float inv_mass = mass > 0 && __builtin_isfinite(mass)
                             ? 1.f / mass
                             : __builtin_nanf("");
        float* dst = splits == 1 ? out + output_row * kHead
                                 : partials + (output_row * splits + split) *
                                                  kPartialSize;
#pragma unroll
        for (int i = 0; i < kLaneValues; ++i) {
          int dim = dim_at(lane, i);
          float result = packed_out[i] + shared.work.output.chunk[row][dim];
          dst[dim] = splits == 1 ? result * inv_mass : result;
        }
        if (splits > 1 && lane == 0) {
          dst[kHead] = shared.maximum[pair][row];
          dst[kHead + 1] = mass;
        }
      }
    }
    __syncthreads();
  }
}

// Expand one packed prefix row into unrotated K/V. The current chunk
// already supplies unquantized K/V in the same coordinates.
template <typename T, typename F>
__device__ __forceinline__ void expand_prefill_row(
    float (&k)[kLaneValues], float (&v)[kLaneValues], Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const uint32_t* signs, int req, int kv_head,
    int pos, int prefix, int chunk_start, int lane) {
  if (pos < prefix) {
    const int page = table[int64_t(req) * table_stride + pos / cache.page_size];
    const auto* slot = cache.at(page, kv_head, pos % cache.page_size);
    const auto* words = reinterpret_cast<const uint32_t*>(slot);
    const uint32_t lo = words[lane];
    const uint32_t hi = lane + 32 < F::slot_bytes / 4 ? words[lane + 32] : 0;
    constexpr int scale_word = F::scale_offset / 4;
    const uint32_t scale0 = staged_word<scale_word, scale_word>(lo, hi, scale_word);
    uint32_t ka, kb;
    float ska, skb;
    if constexpr (F::compact) {
      ka = staged_four<3, 0, kHead>(lo, hi, lane);
      kb = staged_four<3, 0, kHead>(lo, hi, lane + kWave);
      ska = skb = scale_half(scale0, 0) / 64.f;
    } else {
      const uint32_t ka4 = staged_four<4, 0, kRope>(lo, hi, lane % 16);
      const uint32_t ka3 = staged_four<3, F::k3_offset / 4, F::k3_dims>(
          lo, hi, lane % 16);
      ka = lane < 16 ? ka4 : ka3;
      kb = staged_four<3, F::k3_offset / 4, F::k3_dims>(lo, hi, lane + 16);
      const uint32_t scale1 =
          staged_word<scale_word + 1, scale_word + 1>(lo, hi, scale_word + 1);
      ska = scale_half(scale0, lane / 16) / 64.f;
      skb = scale_half(scale1, lane / 16) / 64.f;
    }
    constexpr int value_scale_word = scale_word + F::v_scale / 2;
    const uint32_t value_scales = staged_word<value_scale_word,
                                               value_scale_word>(
        lo, hi, value_scale_word);
    const float sv = scale_half(value_scales, F::v_scale % 2) / 64.f;
    const uint32_t va = staged_four<F::v_bits, F::v_offset / 4, kHead>(
        lo, hi, lane);
    const uint32_t vb = staged_four<F::v_bits, F::v_offset / 4, kHead>(
        lo, hi, lane + kWave);
#pragma unroll
    for (int i = 0; i < kLaneValues / 2; ++i) {
      k[i] = signed_byte(ka, i) * ska;
      k[i + 4] = signed_byte(kb, i) * skb;
      v[i] = signed_byte(va, i) * sv;
      v[i + 4] = signed_byte(vb, i) * sv;
    }
    hadamard<F::compact>(k, lane);
    flip_signs(k, signs, lane);
    hadamard<true>(v, lane);
    flip_signs(v, signs, lane);
  } else {
    const int token = chunk_start + pos - prefix;
    load_activation<T>(k, key, token, kv_head, lane);
    load_activation<T>(v, value, token, kv_head, lane);
  }
}

template <typename T, typename F>
__device__ __forceinline__ void expand_prefill_tile(
    _Float16* key_out, _Float16* residual, float* errors, _Float16* value_out,
    _Float16 (&transposed)[kHead][24], Activation key, Activation value,
    PagedCache cache, const int32_t* table, int64_t table_stride,
    const int32_t* starts, const int32_t* lengths, int capacity,
    const uint32_t* signs) {
  constexpr int Tile = 16;
  constexpr int Threads = 128;
  const int req = blockIdx.z, kv_head = blockIdx.y, first = blockIdx.x * Tile;
  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int chunk_start = starts[req], chunk_end = starts[req + 1];
  const int length = lengths[req], prefix = length - (chunk_end - chunk_start);
  // The attention producer reads a full 32-token tile on the last step.
  if (chunk_end == chunk_start || prefix < 0 || length > capacity ||
      first >= ((length + 31) / 32) * 32)
    return;

  // Residuals use the WMMA lane order. The last 32-key tile can straddle
  // the prefix boundary, so its live and padding rows also get initialized.
  __shared__ float wave_maxima[Threads / kWave];
  float max_error = 0.f;
  const int64_t head_offset =
      (int64_t(req) * cache.heads + kv_head) * capacity * kHead;
#pragma unroll
  for (int t = wave; t < Tile; t += Threads / kWave) {
    float k[kLaneValues] = {}, v[kLaneValues] = {};
    const int pos = first + t;
    float err2 = 0.f;
    if (pos < length)
      expand_prefill_row<T, F>(k, v, key, value, cache, table, table_stride,
                               signs, req, kv_head, pos, prefix, chunk_start,
                               lane);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      const int dim = dim_at(lane, i);
      const _Float16 high = (_Float16)k[i];
      key_out[head_offset + int64_t(pos) * kHead + dim] = high;
      if (first / 32 * 32 < prefix) {
        const int fragment = dim / 16, half = dim % 16 / 8;
        const int64_t index = head_offset + int64_t(first) * kHead +
                              (fragment * 32 + half * 16 + t) * 8 + dim % 8;
        const float delta = k[i] - float(high);
        residual[index] = (_Float16)(delta * 1024.f);
        err2 = fmaf(delta, delta, err2);
      }
      transposed[dim][t] = (_Float16)v[i];
    }
    if (first / 32 * 32 < prefix) {
      err2 = sum_lanes<kWave>(err2);
      max_error = fmaxf(max_error, err2);
    }
  }
  if (lane == 0) wave_maxima[wave] = max_error;
  __syncthreads();
  if (tid == 0 && first < prefix) {
    float peak = fmaxf(fmaxf(wave_maxima[0], wave_maxima[1]),
                       fmaxf(wave_maxima[2], wave_maxima[3]));
    // Distribute maxima over 32 locations to avoid a hot atomic on long
    // prefixes.
    atomicMax(reinterpret_cast<unsigned*>(errors) +
                  (int64_t(req) * cache.heads + kv_head) * 32 + blockIdx.x % 32,
              __float_as_uint(peak));
  }
  // Store V in the vector layout consumed by the attention producer.
  _Float16* dst = value_out + head_offset + int64_t(first) * kHead;
#pragma unroll
  for (int unit = tid; unit < kHead * 2; unit += Threads) {
    const int dim = unit / 2, key_offset = unit % 2 * 8;
    *reinterpret_cast<WmmaHalf8*>(dst + unit * 8) =
        *reinterpret_cast<const WmmaHalf8*>(&transposed[dim][key_offset]);
  }
}

__global__ __launch_bounds__(128) void expand_prefill(
    _Float16* key_out, _Float16* residual, float* errors, _Float16* value_out,
    Activation key, Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* starts, const int32_t* lengths,
    int capacity, const uint32_t* signs, int dtype, int format) {
  __shared__ alignas(16) _Float16 transposed[kHead][24];
  // Uniform per-workgroup dispatch keeps a single packaged kernel entry.
#define EXPAND_PREFILL(T, Compact, Bits)                                   \
  expand_prefill_tile<T, Format<Compact, Bits>>(                           \
      key_out, residual, errors, value_out, transposed, key, value, cache, \
      table, table_stride, starts, lengths, capacity, signs)
  if (dtype == 0) {
    if (format == 3) EXPAND_PREFILL(__half, false, 3);
    else if (format == 4) EXPAND_PREFILL(__half, false, 4);
    else EXPAND_PREFILL(__half, true, 3);
  } else {
    if (format == 3) EXPAND_PREFILL(hip_bfloat16, false, 3);
    else if (format == 4) EXPAND_PREFILL(hip_bfloat16, false, 4);
    else EXPAND_PREFILL(hip_bfloat16, true, 3);
  }
#undef EXPAND_PREFILL
}

template <bool Prefill, bool CompactRows = false>
struct Fp16WmmaShared {
  // Decode and selected long-prefill launches store only the twelve live rows.
  static constexpr int kQueryRows = Prefill && !CompactRows ? 16 : 12;
  union QueryOrOutput {
    _Float16 q[2][kQueryRows][kHead + 2];  // raw and sign/Hadamard query
    float chunk[kQueryRows][kHead];
  } query;
  union KvOrOutput {
    struct Tile {
      _Float16 k[16][kHead + 2];
      _Float16 v[16][kHead + 2];
    } tile;
    float packed[16][kHead];
  } kv;
  _Float16 probability[16][16];
  float rescale[16], maximum[16], mass[16];
  int req[2], length[2], prefix[2], chunk_start[2], begin[2], end[2];
};
static_assert(sizeof(Fp16WmmaShared<false>) <= 32 * 1024,
              "two decode workgroups fit the LDS budget");
static_assert(sizeof(Fp16WmmaShared<true>) <= 64 * 1024,
              "gfx12 workgroup LDS limit");

// Two query tokens and six adjacent GQA heads form twelve active matrix rows.
// Four waves cooperatively unpack each 16-token KV tile, wave zero computes
// QK and FP32 softmax, and all four waves compute disjoint 64-column PV tiles.
// Packed and raw outputs keep one common FP32 max/mass but separate FP32
// accumulators, so the whole-head inverse transform occurs before combining.
template <typename T, typename F, bool Prefill, bool CompactRows = false>
__global__ __launch_bounds__(128) void attention_wmma_fp16(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* map_or_starts, const int32_t* lengths,
    int requests, const uint32_t* signs, float scale, int tokens, int heads) {
  __shared__ Fp16WmmaShared<Prefill, CompactRows> shared;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int column = lane % 16, half_wave = lane / 16;
  const int q_base = 2 * blockIdx.x, kv_head = blockIdx.y;
  const int split = blockIdx.z, splits = gridDim.z;
  if (tid < 2) {
    int q = q_base + tid, req = 0, prefix = 0, length = 0, chunk_start = 0;
    if (q < tokens) {
      if constexpr (Prefill) {
        req = request_for_query(map_or_starts, requests, q);
        chunk_start = map_or_starts[req];
        prefix = lengths[req] - (map_or_starts[req + 1] - chunk_start);
        length = prefix + q - chunk_start + 1;
      } else {
        length = max(0, lengths[q]);
        prefix = length;
        if (length > 0) req = map_or_starts[q];
      }
    }
    int per_split = (length + splits - 1) / splits;
    shared.req[tid] = req;
    shared.length[tid] = length;
    shared.prefix[tid] = prefix;
    shared.chunk_start[tid] = chunk_start;
    shared.begin[tid] = split * per_split;
    shared.end[tid] = min((split + 1) * per_split, length);
  }
  if (tid < 16) {
    shared.maximum[tid] = -INFINITY;
    shared.mass[tid] = 0;
  }
  __syncthreads();

  // Arbitrary decode maps and request boundaries cannot share KV. Retain the
  // original FP32 row algorithm for this workgroup, without touching an invalid
  // request ID on a zero-length/padding row.
  if (shared.length[0] > 0 && shared.length[1] > 0 &&
      shared.req[0] != shared.req[1]) {
    for (int row = wave; row < 12; row += 4) {
      int q = q_base + row / 6;
      if (q < tokens) {
        attention_row<T, F, Prefill>(
            out, partials, query, key, value, cache, table, table_stride,
            map_or_starts, lengths, requests, signs, scale, q,
            kv_head * 6 + row % 6, split, heads, splits, lane, tokens);
      }
    }
    return;
  }

  for (int row = wave; row < Fp16WmmaShared<Prefill, CompactRows>::kQueryRows;
       row += 4) {
    float raw[kLaneValues] = {}, rotated[kLaneValues];
    int q = q_base + row / 6;
    if (row < 12 && q < tokens) {
      load_activation<T>(raw, query, q, kv_head * 6 + row % 6, lane);
    }
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) rotated[i] = raw[i];
    flip_signs(rotated, signs, lane);
    hadamard<F::compact>(rotated, lane);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      int d = dim_at(lane, i);
      shared.query.q[0][row][d] = (_Float16)raw[i];
      shared.query.q[1][row][d] = (_Float16)rotated[i];
    }
  }
  __syncthreads();

  int first = 0x7fffffff, last = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (shared.begin[i] < shared.end[i]) {
      first = min(first, shared.begin[i]);
      last = max(last, shared.end[i]);
    }
  }
  int owner = shared.length[0] > 0 ? 0 : 1;
  int req = shared.req[owner], prefix = shared.prefix[owner];
  int chunk_start = shared.chunk_start[owner];
  WmmaFloat8 packed[4] = {}, chunk[4] = {};
  WmmaFloat8 running_max, running_mass = {};
#pragma unroll
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;

  for (int pos_base = first; pos_base < last;) {
    bool is_packed = !Prefill || pos_base < prefix;
    int count = min(16, last - pos_base);
    if constexpr (Prefill) {
      if (is_packed) count = min(count, prefix - pos_base);
    }
    for (int t = wave; t < 16; t += 4) {
      int pos = pos_base + t;
      bool live =
          t < count && ((pos >= shared.begin[0] && pos < shared.end[0]) ||
                        (pos >= shared.begin[1] && pos < shared.end[1]));
      float k[kLaneValues] = {}, v[kLaneValues] = {};
      if (live) {
        if (is_packed) {
          int page = table[req * table_stride + pos / cache.page_size];
          const uint8_t* slot = cache.at(page, kv_head, pos % cache.page_size);
          const auto* scales =
              reinterpret_cast<const __half*>(slot + F::scale_offset);
          uint32_t a, b;
          float sa, sb;
          if constexpr (F::compact) {
            a = load_four<3>(slot, kHead, lane);
            b = load_four<3>(slot, kHead, lane + kWave);
            sa = sb = __half2float(scales[0]) / 64.f;
          } else {
            a = lane < 16
                    ? load_four<4>(slot, kRope, lane)
                    : load_four<3>(slot + F::k3_offset, F::k3_dims, lane - 16);
            b = load_four<3>(slot + F::k3_offset, F::k3_dims, lane + 16);
            sa = __half2float(scales[lane / 16]) / 64.f;
            sb = __half2float(scales[2 + lane / 16]) / 64.f;
          }
          float sv = __half2float(scales[F::v_scale]) / 64.f;
          uint32_t va = load_four<F::v_bits>(slot + F::v_offset, kHead, lane);
          uint32_t vb =
              load_four<F::v_bits>(slot + F::v_offset, kHead, lane + kWave);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            k[i] = signed_byte(a, i) * sa;
            k[i + 4] = signed_byte(b, i) * sb;
            v[i] = signed_byte(va, i) * sv;
            v[i + 4] = signed_byte(vb, i) * sv;
          }
        } else if constexpr (Prefill) {
          int token = chunk_start + pos - prefix;
          load_activation<T>(k, key, token, kv_head, lane);
          load_activation<T>(v, value, token, kv_head, lane);
        }
      }
#pragma unroll
      for (int i = 0; i < kLaneValues; ++i) {
        int d = dim_at(lane, i);
        shared.kv.tile.k[t][d] = (_Float16)k[i];
        shared.kv.tile.v[t][d] = (_Float16)v[i];
      }
    }
    __syncthreads();

    if (wave == 0) {
      WmmaFloat8 score = {};
// Bound operand liveness on ROCm 7.2 while overlapping four QK steps.
#pragma unroll 4
      for (int d = 0; d < kHead; d += 16) {
        WmmaHalf8 a, b;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int kdim = d + half_wave * 8 + e;
          if constexpr (Prefill && !CompactRows) {
            a[e] = shared.query.q[is_packed ? 1 : 0][column][kdim];
          } else {
            // Padded matrix rows have no query storage in the compact LDS
            // layout.
            a[e] = column < 12 ? shared.query.q[is_packed ? 1 : 0][column][kdim]
                               : (_Float16)0;
          }
          b[e] = shared.kv.tile.k[column][kdim];
        }
        score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e, qi = row / 6;
        int pos = pos_base + column;
        bool live = row < 12 && q_base + qi < tokens && column < count &&
                    pos >= shared.begin[qi] && pos < shared.end[qi];
        float s = live ? score[e] * scale : -INFINITY;
        float m = s;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          m = fmaxf(m, octave_xor<false>(m, delta, kWave));
        float next_max = fmaxf(running_max[e], m);
        float rescale =
            running_mass[e] > 0 ? __expf(running_max[e] - next_max) : 0;
        float weight = live ? __expf(s - next_max) : 0;
        float tile_mass = weight;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          tile_mass += octave_xor<false>(tile_mass, delta, kWave);
        running_mass[e] = fmaf(running_mass[e], rescale, tile_mass);
        running_max[e] = next_max;
        shared.probability[row][column] = (_Float16)weight;
        if (column == 0) {
          shared.rescale[row] = rescale;
          shared.maximum[row] = next_max;
          shared.mass[row] = running_mass[e];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int d = 0; d < 4; ++d) {
      WmmaHalf8 a, b;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e;
        packed[d][e] *= shared.rescale[row];
        chunk[d][e] *= shared.rescale[row];
        a[e] = shared.probability[column][half_wave * 8 + e];
        b[e] = shared.kv.tile.v[half_wave * 8 + e][wave * 64 + d * 16 + column];
      }
      if (is_packed) {
        packed[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, packed[d]);
      } else if constexpr (Prefill) {
        chunk[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, chunk[d]);
      }
    }
    __syncthreads();
    pos_base += count;
  }

  // Reuse dead operand storage for the FP32 output layout conversion.
#pragma unroll
  for (int d = 0; d < 4; ++d) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int row = half_wave * 8 + e, dim = wave * 64 + d * 16 + column;
      shared.kv.packed[row][dim] = packed[d][e];
      if constexpr (Prefill && !CompactRows) {
        shared.query.chunk[row][dim] = chunk[d][e];
      } else if (row < 12) {
        shared.query.chunk[row][dim] = chunk[d][e];
      }
    }
  }
  __syncthreads();
  for (int row = wave; row < 12; row += 4) {
    int q = q_base + row / 6;
    if (q >= tokens) continue;
    float packed_out[kLaneValues];
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      packed_out[i] = shared.kv.packed[row][dim_at(lane, i)];
    hadamard<true>(packed_out, lane);
    flip_signs(packed_out, signs, lane);
    int64_t output_row = int64_t(q) * heads + kv_head * 6 + row % 6;
    float mass = shared.mass[row];
    float inv_mass =
        mass > 0 && __builtin_isfinite(mass) ? 1.f / mass : __builtin_nanf("");
    float* dst = splits == 1
                     ? out + output_row * kHead
                     : partials + (output_row * splits + split) * kPartialSize;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      int dim = dim_at(lane, i);
      float result = packed_out[i] + shared.query.chunk[row][dim];
      dst[dim] = splits == 1 ? result * inv_mass : result;
    }
    if (splits > 1 && lane == 0) {
      dst[kHead] = shared.maximum[row];
      dst[kHead + 1] = mass;
    }
  }
}

// Generic head/query packing: use all sixteen native WMMA rows.
template <bool Prefill, int Group>
struct GroupedWmmaShared {
  static constexpr int Queries = 16 / Group, Rows = Queries * Group;
  union QueryOrOutput {
    _Float16 q[2][Rows][kHead + 2];
    float chunk[Rows][kHead];
  } query;
  union KvOrOutput {
    struct Tile {
      _Float16 k[16][kHead + 2], v[16][kHead + 2];
    } tile;
    float packed[16][kHead];
  } kv;
  _Float16 probability[16][16];
  float rescale[16], maximum[16], mass[16];
  int unsafe_rows;
  int req[Queries], length[Queries], prefix[Queries], chunk_start[Queries];
  int begin[Queries], end[Queries];
};

template <typename T, typename F, bool Prefill, int Group>
__global__ __launch_bounds__(128) void attention_wmma_grouped(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* map_or_starts, const int32_t* lengths,
    int requests, const uint32_t* signs, float scale, int tokens, int heads) {
  constexpr int Queries = 16 / Group, Rows = Queries * Group;
  __shared__ GroupedWmmaShared<Prefill, Group> shared;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int column = lane % 16, half_wave = lane / 16;
  const int q_base = Queries * blockIdx.x, kv_head = blockIdx.y;
  const int split = blockIdx.z, splits = gridDim.z;
  if (tid == 0) shared.unsafe_rows = 0;
  if (tid < Queries) {
    int q = q_base + tid, req = 0, prefix = 0, length = 0, chunk_start = 0;
    if (q < tokens) {
      if constexpr (Prefill) {
        req = request_for_query(map_or_starts, requests, q);
        chunk_start = map_or_starts[req];
        prefix = lengths[req] - (map_or_starts[req + 1] - chunk_start);
        length = prefix + q - chunk_start + 1;
      } else {
        length = max(0, lengths[q]);
        prefix = length;
        if (length > 0) req = map_or_starts[q];
      }
    }
    int per_split = (length + splits - 1) / splits;
    shared.req[tid] = req;
    shared.length[tid] = length;
    shared.prefix[tid] = prefix;
    shared.chunk_start[tid] = chunk_start;
    shared.begin[tid] = split * per_split;
    shared.end[tid] = min((split + 1) * per_split, length);
  }
  if (tid < 16) {
    shared.maximum[tid] = -INFINITY;
    shared.mass[tid] = 0;
  }
  __syncthreads();

  // Arbitrary decode maps and request boundaries cannot share KV. Retain the
  // original FP32 row algorithm for this workgroup, without touching an invalid
  // request ID on a zero-length/padding row.
  int owner = 0;
  for (int i = 0; i < Queries; ++i)
    if (shared.length[i] > 0) {
      owner = i;
      break;
    }
  bool different_requests = false;
#pragma unroll
  for (int i = 0; i < Queries; ++i)
    different_requests |=
        shared.length[i] > 0 && shared.req[i] != shared.req[owner];
  if (different_requests) return;

  for (int row = wave; row < Rows; row += 4) {
    float raw[kLaneValues] = {}, rotated[kLaneValues];
    int q = q_base + row / Group;
    if (row < Rows && q < tokens) {
      load_activation<T>(raw, query, q, kv_head * Group + row % Group, lane);
    }
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) rotated[i] = raw[i];
    flip_signs(rotated, signs, lane);
    hadamard<F::compact>(rotated, lane);
    bool safe = true;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      safe &= __builtin_isfinite(raw[i]) && __builtin_isfinite(rotated[i]);
      safe &= fabsf(raw[i]) * fabsf(scale) <= 2.f &&
              fabsf(rotated[i]) * fabsf(scale) <= 2.f;
    }
    if (!__all(safe) && lane == 0) atomicOr(&shared.unsafe_rows, 1 << row);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      int d = dim_at(lane, i);
      shared.query.q[0][row][d] = (_Float16)raw[i];
      shared.query.q[1][row][d] = (_Float16)rotated[i];
    }
  }
  __syncthreads();

  if (shared.unsafe_rows == 65535) return;

  int first = 0x7fffffff, last = 0;
#pragma unroll
  for (int i = 0; i < Queries; ++i) {
    if (shared.begin[i] < shared.end[i]) {
      first = min(first, shared.begin[i]);
      last = max(last, shared.end[i]);
    }
  }
  int req = shared.req[owner], prefix = shared.prefix[owner];
  int chunk_start = shared.chunk_start[owner];
  WmmaFloat8 packed[4] = {}, chunk[4] = {};
  WmmaFloat8 running_max, running_mass = {};
#pragma unroll
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;

  for (int pos_base = first; pos_base < last;) {
    bool is_packed = !Prefill || pos_base < prefix;
    int count = min(16, last - pos_base);
    if constexpr (Prefill) {
      if (is_packed) count = min(count, prefix - pos_base);
    }
    for (int t = wave; t < 16; t += 4) {
      int pos = pos_base + t;
      bool live =
          t < count && reuse_live<Queries>(pos, shared.begin, shared.end);
      float k[kLaneValues] = {}, v[kLaneValues] = {};
      if (live) {
        if (is_packed) {
          int page = table[req * table_stride + pos / cache.page_size];
          const uint8_t* slot = cache.at(page, kv_head, pos % cache.page_size);
          const auto* scales =
              reinterpret_cast<const __half*>(slot + F::scale_offset);
          uint32_t a, b;
          float sa, sb;
          if constexpr (F::compact) {
            a = load_four<3>(slot, kHead, lane);
            b = load_four<3>(slot, kHead, lane + kWave);
            sa = sb = __half2float(scales[0]) / 64.f;
          } else {
            a = lane < 16
                    ? load_four<4>(slot, kRope, lane)
                    : load_four<3>(slot + F::k3_offset, F::k3_dims, lane - 16);
            b = load_four<3>(slot + F::k3_offset, F::k3_dims, lane + 16);
            sa = __half2float(scales[lane / 16]) / 64.f;
            sb = __half2float(scales[2 + lane / 16]) / 64.f;
          }
          float sv = __half2float(scales[F::v_scale]) / 64.f;
          uint32_t va = load_four<F::v_bits>(slot + F::v_offset, kHead, lane);
          uint32_t vb =
              load_four<F::v_bits>(slot + F::v_offset, kHead, lane + kWave);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            k[i] = signed_byte(a, i) * sa;
            k[i + 4] = signed_byte(b, i) * sb;
            v[i] = signed_byte(va, i) * sv;
            v[i + 4] = signed_byte(vb, i) * sv;
          }
        } else if constexpr (Prefill) {
          int token = chunk_start + pos - prefix;
          load_activation<T>(k, key, token, kv_head, lane);
          load_activation<T>(v, value, token, kv_head, lane);
        }
      }
#pragma unroll
      for (int i = 0; i < kLaneValues; ++i) {
        int d = dim_at(lane, i);
        shared.kv.tile.k[t][d] = (_Float16)k[i];
        shared.kv.tile.v[t][d] = (_Float16)v[i];
      }
    }
    __syncthreads();

    if (wave == 0) {
      WmmaFloat8 score = {};
// Four steps retain instruction overlap without spilling on ROCm 7.2.
#pragma unroll 4
      for (int d = 0; d < kHead; d += 16) {
        WmmaHalf8 a, b;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          int kdim = d + half_wave * 8 + e;
          if constexpr (false) {
            a[e] = shared.query.q[is_packed ? 1 : 0][column][kdim];
          } else {
            // Padded matrix rows have no query storage in the compact LDS
            // layout.
            a[e] = column < Rows
                       ? shared.query.q[is_packed ? 1 : 0][column][kdim]
                       : (_Float16)0;
          }
          b[e] = shared.kv.tile.k[column][kdim];
        }
        score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e, qi = row / Group;
        int pos = pos_base + column;
        bool live = row < Rows && q_base + qi < tokens && column < count &&
                    pos >= shared.begin[qi] && pos < shared.end[qi];
        float s = live ? score[e] * scale : -INFINITY;
        float m = s;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          m = fmaxf(m, octave_xor<false>(m, delta, kWave));
        float next_max = fmaxf(running_max[e], m);
        float rescale =
            running_mass[e] > 0 ? __expf(running_max[e] - next_max) : 0;
        float weight = live ? __expf(s - next_max) : 0;
        float tile_mass = weight;
#pragma unroll
        for (int delta = 8; delta; delta /= 2)
          tile_mass += octave_xor<false>(tile_mass, delta, kWave);
        running_mass[e] = fmaf(running_mass[e], rescale, tile_mass);
        running_max[e] = next_max;
        shared.probability[row][column] = (_Float16)weight;
        if (column == 0) {
          shared.rescale[row] = rescale;
          shared.maximum[row] = next_max;
          shared.mass[row] = running_mass[e];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int d = 0; d < 4; ++d) {
      WmmaHalf8 a, b;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half_wave * 8 + e;
        packed[d][e] *= shared.rescale[row];
        chunk[d][e] *= shared.rescale[row];
        a[e] = shared.probability[column][half_wave * 8 + e];
        b[e] = shared.kv.tile.v[half_wave * 8 + e][wave * 64 + d * 16 + column];
      }
      if (is_packed) {
        packed[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, packed[d]);
      } else if constexpr (Prefill) {
        chunk[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, chunk[d]);
      }
    }
    __syncthreads();
    pos_base += count;
  }

  // Reuse dead operand storage for the FP32 output layout conversion.
#pragma unroll
  for (int d = 0; d < 4; ++d) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int row = half_wave * 8 + e, dim = wave * 64 + d * 16 + column;
      shared.kv.packed[row][dim] = packed[d][e];
      if constexpr (false) {
        shared.query.chunk[row][dim] = chunk[d][e];
      } else if (row < Rows) {
        shared.query.chunk[row][dim] = chunk[d][e];
      }
    }
  }
  __syncthreads();
  for (int row = wave; row < Rows; row += 4) {
    int q = q_base + row / Group;
    if (q >= tokens || (shared.unsafe_rows & (1 << row))) continue;
    float packed_out[kLaneValues];
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      packed_out[i] = shared.kv.packed[row][dim_at(lane, i)];
    hadamard<true>(packed_out, lane);
    flip_signs(packed_out, signs, lane);
    int64_t output_row = int64_t(q) * heads + kv_head * Group + row % Group;
    float mass = shared.mass[row];
    float inv_mass =
        mass > 0 && __builtin_isfinite(mass) ? 1.f / mass : __builtin_nanf("");
    float* dst = splits == 1
                     ? out + output_row * kHead
                     : partials + (output_row * splits + split) * kPartialSize;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      int dim = dim_at(lane, i);
      float result = packed_out[i] + shared.query.chunk[row][dim];
      dst[dim] = splits == 1 ? result * inv_mass : result;
    }
    if (splits > 1 && lane == 0) {
      dst[kHead] = shared.maximum[row];
      dst[kHead + 1] = mass;
    }
  }
}

template <bool Prefill>
__global__ __launch_bounds__(256) void merge_splits(
    float* out, const float* partials, const int32_t* lengths, int tokens,
    int kv_heads, int splits) {
  int64_t row = int64_t(blockIdx.x) * gridDim.y + blockIdx.y;
  int lane = threadIdx.x;
  const int q = blockIdx.x;
  const int active =
      (Prefill || gridDim.y / kv_heads != 6)
          ? splits
          : active_decode_splits(max(0, lengths[q]), splits, tokens, kv_heads);
  partials += row * splits * kPartialSize;
  if (!Prefill && lengths[q] <= 0) {
    for (int d = lane; d < kHead; d += blockDim.x)
      out[row * kHead + d] = 0.f;
    return;
  }
  extern __shared__ float shared[];
  if (blockDim.x > kWave && active >= octave_rdna4::kParallelMergeMinSplits) {
    octave_rdna4::merge_parallel<kHead, false>(out + row * kHead, partials,
                                              active, shared);
    return;
  }
  if (lane >= kWave) return;
  float m = -INFINITY;
  for (int s = 0; s < active; ++s)
    m = fmaxf(m, partials[s * kPartialSize + kHead]);
  float mass = 0, result[kLaneValues] = {};
  for (int s = 0; s < active; ++s) {
    const float* partial = partials + s * kPartialSize;
    if (partial[kHead + 1] == 0) continue;
    float weight = __expf(partial[kHead] - m);
    mass = fmaf(weight, partial[kHead + 1], mass);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      result[i] = fmaf(weight, partial[dim_at(lane, i)], result[i]);
    }
  }
  float inv_mass = mass > 0 ? 1.f / mass : 0;
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i)
    out[row * kHead + dim_at(lane, i)] = result[i] * inv_mass;
}

// Repair only native WMMA work; scalar paths keep their original launch count.
template <typename T, typename F, bool P>
__global__ __launch_bounds__(256) void merge_splits_safe(
    float* out, const float* partials, int splits, Activation query,
    Activation key, Activation value, PagedCache cache, const int32_t* table,
    int64_t stride, const int32_t* map, const int32_t* lengths, int requests,
    const uint32_t* signs, float scale) {
  const int q = blockIdx.x, head = blockIdx.y, heads = gridDim.y,
            lane = threadIdx.x;
  const int64_t row = int64_t(q) * heads + head;
  const int active =
      (P || heads / cache.heads != 6)
          ? splits
          : active_decode_splits(max(0, lengths[q]), splits, gridDim.x,
                                 cache.heads);
  if constexpr (!P) {
    if (lengths[q] <= 0) {
      for (int d = lane; d < kHead; d += blockDim.x)
        out[row * kHead + d] = 0.f;
      return;
    }
  }
  extern __shared__ float shared[];
  if (blockDim.x > kWave && active >= octave_rdna4::kParallelMergeMinSplits) {
    const float* partial = partials + row * splits * kPartialSize;
    const bool safe = octave_rdna4::merge_parallel<kHead, true>(
        out + row * kHead, partial, active, shared);
    if (!safe && lane < kWave)
      attention_row<T, F, P>(out, nullptr, query, key, value, cache, table,
                             stride, map, lengths, requests, signs, scale, q,
                             head, 0, heads, 1, lane, gridDim.x);
    return;
  }
  if (lane >= kWave) return;
  if (splits == 1) {
    bool safe = true;
    float largest = 0;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      float x = out[row * kHead + dim_at(lane, i)];
      safe &= __builtin_isfinite(x);
      largest = fmaxf(largest, fabsf(x));
    }
    for (int d = 16; d; d /= 2)
      largest = fmaxf(largest, octave_xor<false>(largest, d, 32));
    if (__all(safe) && largest >= 1.e-4f) return;
    attention_row<T, F, P>(out, nullptr, query, key, value, cache, table,
                           stride, map, lengths, requests, signs, scale, q,
                           head, 0, heads, 1, lane, gridDim.x);
    return;
  }
  const float* src = partials + row * splits * kPartialSize;
  bool safe = true;
  float m = -INFINITY;
  for (int s = 0; s < active; ++s) {
    const float* partial = src + s * kPartialSize;
    float marker = partial[0], pm = partial[kHead],
          mass = src[s * kPartialSize + kHead + 1];
    // Native tiles may partition at their longest query; empty splits are
    // valid.
    safe &= __builtin_isfinite(marker) && __builtin_isfinite(mass) && mass >= 0 &&
            (mass == 0 || __builtin_isfinite(pm));
    m = fmaxf(m, pm);
  }
  if (!safe) {
    attention_row<T, F, P>(out, nullptr, query, key, value, cache, table,
                           stride, map, lengths, requests, signs, scale, q,
                           head, 0, heads, 1, lane, gridDim.x);
    return;
  }
  float mass = 0, result[kLaneValues] = {};
  for (int s = 0; s < active; ++s) {
    const float* p = src + s * kPartialSize;
    if (p[kHead + 1] == 0) continue;
    float weight = __expf(p[kHead] - m);
    mass = fmaf(weight, p[kHead + 1], mass);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i)
      result[i] = fmaf(weight, p[dim_at(lane, i)], result[i]);
  }
  const float inv = mass > 0 ? 1.f / mass : 0;
  float largest = 0;
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) {
    result[i] *= inv;
    safe &= __builtin_isfinite(result[i]);
    largest = fmaxf(largest, fabsf(result[i]));
  }
  for (int d = 16; d; d /= 2)
    largest = fmaxf(largest, octave_xor<false>(largest, d, 32));
  if (!__all(safe) || (mass > 0 && largest < 1.e-4f)) {
    attention_row<T, F, P>(out, nullptr, query, key, value, cache, table,
                           stride, map, lengths, requests, signs, scale, q,
                           head, 0, heads, 1, lane, gridDim.x);
    return;
  }
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i)
    out[row * kHead + dim_at(lane, i)] = result[i];
}
template <typename T, typename Launch>
int dispatch_format(int format, Launch launch) {
  switch (format) {
    case 3:
      launch(T{}, Format<false, 3>{});
      break;
    case 4:
      launch(T{}, Format<false, 4>{});
      break;
    case 19:
      launch(T{}, Format<true, 3>{});
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

template <typename T, typename F, bool Prefill,
          DecodePath Path = DecodePath::Scalar>
__global__ void attention(float* out, float* partials, Activation query,
                          Activation key, Activation value, PagedCache cache,
                          const int32_t* table, int64_t table_stride,
                          const int32_t* map_or_starts, const int32_t* lengths,
                          int requests, const uint32_t* signs, float scale) {
  attention_row<T, F, Prefill, Path>(
      out, partials, query, key, value, cache, table, table_stride,
      map_or_starts, lengths, requests, signs, scale, blockIdx.x, blockIdx.y,
      blockIdx.z, gridDim.y, gridDim.z, threadIdx.x, gridDim.x);
}

// Keep the prefill guard separate: combining it with attention slows ragged
// boundary rows, which need the full scalar calculation.
template <typename T, typename F>
__global__ void attention_prefill_unsafe(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* starts, const int32_t* lengths,
    int requests, const uint32_t* signs, float scale, int tokens, int heads) {
  const int q = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
  const int lane = threadIdx.x, splits = gridDim.z;
  const int group = heads / cache.heads, queries = 16 / group;
  const int base = (q / queries) * queries;
  const bool boundary =
      request_for_query(starts, requests, base) !=
      request_for_query(starts, requests, min(base + queries, tokens) - 1);
  if (!boundary) {
    float raw[kLaneValues], rotated[kLaneValues];
    load_activation<T>(raw, query, q, head, lane);
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) rotated[i] = raw[i];
    flip_signs(rotated, signs, lane);
    hadamard<F::compact>(rotated, lane);
    bool safe = true;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      safe &= __builtin_isfinite(raw[i]) && __builtin_isfinite(rotated[i]);
      safe &= fabsf(raw[i]) * fabsf(scale) <= 2.f &&
              fabsf(rotated[i]) * fabsf(scale) <= 2.f;
    }
    if (__all(safe)) return;
  }
  attention_row<T, F, true>(out, partials, query, key, value, cache, table,
                            table_stride, starts, lengths, requests, signs,
                            scale, q, head, split, heads, splits, lane, tokens);
}

template <typename Launch>
int dispatch(int dtype, int format, Launch launch) {
  switch (dtype) {
    case 0:
      return dispatch_format<__half>(format, launch);
    case 1:
      return dispatch_format<hip_bfloat16>(format, launch);
    default:
      return hipErrorInvalidValue;
  }
}

template <bool Prefill>
int launch_attention(float* out, float* partials, Activation query,
                     Activation key, Activation value, PagedCache cache,
                     const int32_t* table, int64_t table_stride,
                     const int32_t* map_or_starts, const int32_t* lengths,
                     int requests, const uint32_t* signs, int tokens, int heads,
                     int splits, float scale, int dtype, int format,
                     hipStream_t stream) {
  if (!tokens) return hipSuccess;
  // Row stride bounds the page-table capacity for the normal contiguous-row
  // layouts. It is a launch hint only; kernels use each query's live length.
  const int64_t capacity_hint = table_stride * cache.page_size;
  const bool single_query_gqa6 = tokens == 1 && heads / cache.heads == 6;
  bool used_wmma = false;
  int status = dispatch(dtype, format, [&](auto scalar, auto layout) {
    using T = decltype(scalar);
    using F = decltype(layout);
    if constexpr (!Prefill && !F::compact && F::v_bits == 3) {
      if (single_query_gqa6 && splits == 1 && capacity_hint > 0 &&
          capacity_hint <= 512) {
        attention<T, F, false, DecodePath::Cooperative>
            <<<dim3(tokens, heads, splits), kWave, 0, stream>>>(
                out, partials, query, key, value, cache, table, table_stride,
                map_or_starts, lengths, requests, signs, scale);
        return;
      }
    }
    if constexpr (Prefill) {
      const int group = heads / cache.heads;
      if ((group == 1 || group == 2 ||
           (F::compact && (group == 4 || group == 8))) &&
          tokens >= 256 && tokens / requests >= 32) {
        used_wmma = true;
        if (group == 1)
          attention_wmma_grouped<T, F, true, 1>
              <<<dim3((tokens + 15) / 16, cache.heads, splits), 128, 0,
                 stream>>>(out, partials, query, key, value, cache, table,
                           table_stride, map_or_starts, lengths, requests,
                           signs, scale, tokens, heads);
        else if (group == 2)
          attention_wmma_grouped<T, F, true, 2>
              <<<dim3((tokens + 7) / 8, cache.heads, splits), 128, 0, stream>>>(
                  out, partials, query, key, value, cache, table, table_stride,
                  map_or_starts, lengths, requests, signs, scale, tokens,
                  heads);
        else if constexpr (F::compact) {
          if (group == 4)
            attention_wmma_grouped<T, F, true, 4>
                <<<dim3((tokens + 3) / 4, cache.heads, splits), 128, 0,
                   stream>>>(out, partials, query, key, value, cache, table,
                             table_stride, map_or_starts, lengths, requests,
                             signs, scale, tokens, heads);
          else
            attention_wmma_grouped<T, F, true, 8>
                <<<dim3((tokens + 1) / 2, cache.heads, splits), 128, 0,
                   stream>>>(out, partials, query, key, value, cache, table,
                             table_stride, map_or_starts, lengths, requests,
                             signs, scale, tokens, heads);
        }
        attention_prefill_unsafe<T, F>
            <<<dim3(tokens, heads, splits), kWave, 0, stream>>>(
                out, partials, query, key, value, cache, table, table_stride,
                map_or_starts, lengths, requests, signs, scale, tokens, heads);
        return;
      }
    }
    if constexpr (Prefill) {
      if (heads / cache.heads == 6 && tokens >= 256 &&
          capacity_hint >= (int64_t(tokens) + requests - 1) / requests + 1024 &&
          (requests == 1 || tokens / requests >= 128)) {
        used_wmma = true;
        attention_wmma_fp16_reuse<T, F, true>
            <<<dim3((tokens + 5) / 6, cache.heads, splits), 384, 0, stream>>>(
                out, partials, query, key, value, cache, table, table_stride,
                map_or_starts, lengths, requests, signs, scale, tokens, heads);
        return;
      }
    }
    if constexpr (!Prefill) {
      if (heads / cache.heads == 6 && tokens >= 1 && capacity_hint >= 128 &&
          (tokens >= 32 || capacity_hint >= 512)) {
        used_wmma = true;
        attention_wmma_fp16_oneq<T, F>
            <<<dim3(tokens, cache.heads, splits), 128, 0, stream>>>(
                out, partials, query, cache, table, table_stride,
                map_or_starts, lengths, signs, scale, tokens, heads);
        return;
      }
    }
    attention<T, F, Prefill><<<dim3(tokens, heads, splits), kWave, 0, stream>>>(
        out, partials, query, key, value, cache, table, table_stride,
        map_or_starts, lengths, requests, signs, scale);
  });
  if (status != hipSuccess) return status;
  const int merge_threads = octave_rdna4::merge_threads(
      int64_t(tokens) * heads, splits, Prefill && used_wmma);
  const int merge_shared =
      octave_rdna4::merge_shared_bytes<kHead>(merge_threads, splits);
  if (!used_wmma) {
    if (splits > 1)
      merge_splits<Prefill><<<dim3(tokens, heads), merge_threads, merge_shared,
                             stream>>>(
          out, partials, lengths, tokens, cache.heads, splits);
    return hipGetLastError();
  }
  return dispatch(dtype, format, [&](auto scalar, auto layout) {
    using T = decltype(scalar);
    using F = decltype(layout);
    merge_splits_safe<T, F, Prefill>
        <<<dim3(tokens, heads), merge_threads, merge_shared, stream>>>(
            out, partials, splits, query, key, value, cache, table, table_stride,
            map_or_starts, lengths, requests, signs, scale);
  });
}

}  // namespace

int octave_rdna4::cache_slot_bytes(int64_t format) {
  switch (format) {
    case 3:
      return Format<false, 3>::slot_bytes;
    case 4:
      return Format<false, 4>::slot_bytes;
    case 19:
      return Format<true, 3>::slot_bytes;
    default:
      return 0;
  }
}

int octave_rdna4::launch_store(Activation key, Activation value,
                              PagedCache cache, const int64_t* slots,
                              const uint32_t* signs, int tokens, int dtype,
                              int format, hipStream_t stream) {
  if (!tokens) return hipSuccess;
  return dispatch(dtype, format, [&](auto scalar, auto layout) {
    cache_store<decltype(scalar), decltype(layout)>
        <<<dim3(tokens, cache.heads), kWave, 0, stream>>>(key, value, cache,
                                                          slots, signs);
  });
}

int octave_rdna4::launch_decode(
    float* out, float* partials, Activation query, PagedCache cache,
    const int32_t* table, int64_t table_stride, const int32_t* q_to_req,
    const int32_t* q_to_klen, const uint32_t* signs, int tokens, int heads,
    int splits, float scale, int dtype, int format, hipStream_t stream) {
  return launch_attention<false>(
      out, partials, query, {}, {}, cache, table, table_stride, q_to_req,
      q_to_klen, 0, signs, tokens, heads, splits, scale, dtype, format, stream);
}

int octave_rdna4::launch_prefill(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table,
    int64_t table_stride, const int32_t* query_starts, const int32_t* seq_lens,
    int requests, const uint32_t* signs, int tokens, int heads, int splits,
    float scale, int dtype, int format, hipStream_t stream) {
  return launch_attention<true>(out, partials, query, key, value, cache, table,
                                table_stride, query_starts, seq_lens, requests,
                                signs, tokens, heads, splits, scale, dtype,
                                format, stream);
}

namespace {

// Round each rotated query element once. Two signed base-256 digits let the
// integer matrix cores accumulate the score before applying the wire scale.
template <typename T, bool Compact>
__global__ void prepare_prefix_query(Activation query, const uint32_t* signs,
                                     int8_t* q8, float* qscale, uint8_t* repair,
                                     float scale, const int32_t* starts,
                                     const int32_t* lengths, int requests) {
  const int q = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int64_t row = int64_t(q) * gridDim.y + head;
  float x[kLaneValues], raw_max = 0.f, rotated_max = 0.f;
  load_activation<T>(x, query, q, head, lane);
  bool safe = true;
#pragma unroll
  for (float v : x) {
    safe &= __builtin_isfinite(v);
    raw_max = fmaxf(raw_max, fabsf(v));
  }
  const int req = requests == 1 ? 0 : request_for_query(starts, requests, q);
  if (int64_t(lengths[req]) <= int64_t(starts[req + 1]) - starts[req]) {
    const bool tiny = !__any(raw_max >= 1.e-4f) && __any(raw_max > 0.f);
    const bool ok =
        __all(safe && raw_max <= 65504.f && raw_max * scale <= 2.f) && !tiny;
    if (!lane) repair[row] = !ok;
    return;
  }
  flip_signs(x, signs, lane);
  hadamard<Compact>(x, lane);
#pragma unroll
  for (float v : x) {
    safe &= __builtin_isfinite(v);
    rotated_max = fmaxf(rotated_max, fabsf(v));
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    rotated_max = fmaxf(rotated_max, octave_xor<false>(rotated_max, d, 32));
    raw_max = fmaxf(raw_max, octave_xor<false>(raw_max, d, 32));
  }
  safe = __all(safe) && raw_max <= 65504.f &&
         fmaxf(raw_max, rotated_max) * scale <= 2.f &&
         (raw_max == 0.f || raw_max >= 1.e-4f);
  const float inv = safe && rotated_max > 0.f ? 127.f / rotated_max : 0.f;
  if (!lane) {
    repair[row] = !safe;
    qscale[row] = safe ? rotated_max / (127.f * 256.f) : 0.f;
  }
#pragma unroll
  for (int i = 0; i < kLaneValues; ++i) {
    const float scaled = safe ? x[i] * inv : 0.f;
    const int fixed = __float2int_rn(scaled * 256.f);
    const int main = (fixed + 128) >> 8;
    const int residual = fixed - main * 256;
    const int64_t off = row * 2 * kHead + dim_at(lane, i);
    q8[off] = main;
    q8[off + kHead] = residual;
  }
}

// Merge prefix splits in the rotated V basis, invert once per row, then
// combine with the raw causal chunk using its log2 normalizer.
template <typename T, typename F, bool Streamed>
__global__ void finish_prefill(float* out, const float* partials,
                               const float* chunk_lse, const uint8_t* repair,
                               Activation query, Activation key,
                               Activation value, PagedCache cache,
                               const int32_t* table, int64_t stride,
                               const int32_t* starts, const int32_t* lengths,
                               int requests, const uint32_t* signs, int splits,
                               float scale, int64_t prefix_capacity) {
  const int q = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int64_t row = int64_t(q) * gridDim.y + head;
  bool safe =
      Streamed ? !repair[row] : !(repair[2 * row] | repair[2 * row + 1]);
  int used = 0, prefix_length = -1;
  if constexpr (Streamed) {
    const int req = request_for_query(starts, requests, q);
    prefix_length = octave_rdna4::bounded_prefix(
        lengths[req], starts[req + 1] - starts[req], prefix_capacity);
    used = octave_rdna4::prefix_splits(prefix_length, splits);
  }
  float combined[kLaneValues];
  if (safe) {
    float prefix[kLaneValues] = {}, weight = 1.f, inv_mass = 1.f;
    if constexpr (Streamed) {
      if (used) {
        const int64_t step = int64_t(gridDim.x) * gridDim.y * kPartialSize;
        const float* p = partials + row * kPartialSize;
        const float lse = chunk_lse[row];
        float m = lse;
        for (int s = 0; s < used; ++s) m = fmaxf(m, p[s * step + kHead]);
        weight = __builtin_amdgcn_exp2f(lse - m);
        float mass = weight;
        for (int s = 0; s < used; ++s) {
          const float w = __builtin_amdgcn_exp2f(p[kHead] - m);
          mass = fmaf(w, p[kHead + 1], mass);
#pragma unroll
          for (int i = 0; i < kLaneValues; ++i)
            prefix[i] = fmaf(w, p[dim_at(lane, i)], prefix[i]);
          p += step;
        }
        inv_mass = 1.f / mass;
        hadamard<true>(prefix, lane);
        flip_signs(prefix, signs, lane);
      }
    }
    float largest = 0.f;
#pragma unroll
    for (int i = 0; i < kLaneValues; ++i) {
      float v = out[row * kHead + dim_at(lane, i)];
      if constexpr (Streamed) v = fmaf(weight, v, prefix[i]) * inv_mass;
      safe &= __builtin_isfinite(v);
      largest = fmaxf(largest, fabsf(v));
      combined[i] = v;
    }
#pragma unroll
    for (int d = 16; d; d >>= 1)
      largest = fmaxf(largest, octave_xor<false>(largest, d, 32));
    if (__all(safe) && largest >= 1.e-4f) {
      if constexpr (Streamed) {
#pragma unroll
        for (int i = 0; i < kLaneValues; ++i)
          out[row * kHead + dim_at(lane, i)] = combined[i];
      }
      return;
    }
  }
  attention_row<T, F, true>(out, nullptr, query, key, value, cache, table,
                            stride, starts, lengths, requests, signs, scale, q,
                            head, 0, gridDim.y, 1, lane, gridDim.x,
                            prefix_length);
}

}  // namespace

int octave_rdna4::launch_prefill_expanded(
    float* out, void* keys, void* values, const void* query_fp16,
    Activation query, Activation key, Activation value, PagedCache cache,
    const int32_t* table, int64_t table_stride, const int32_t* starts,
    const int32_t* lengths, int requests, int capacity, const uint32_t* signs,
    int tokens, int heads, float scale, int dtype, int format, void* correction,
    uint8_t* repair, hipStream_t stream) {
  if (requests <= 0 || tokens <= 0 || capacity <= 0 || capacity % 32 ||
      cache.heads <= 0 || heads <= 0 || heads % cache.heads != 0 ||
      !(scale > 0.f) || (dtype != 0 && dtype != 1) ||
      (format != 3 && format != 4 && format != 19))
    return hipErrorInvalidValue;
  auto* keys_fp16 = static_cast<_Float16*>(keys);
  auto* residual = static_cast<_Float16*>(correction);
  auto* errors = reinterpret_cast<float*>(
      residual + int64_t(requests) * cache.heads * capacity * kHead);
  auto* values_fp16 = static_cast<_Float16*>(values);
  const auto cleared = hipMemsetAsync(
      errors, 0, int64_t(requests) * cache.heads * 32 * sizeof(float), stream);
  if (cleared != hipSuccess) return cleared;
  expand_prefill<<<dim3(capacity / 16, cache.heads, requests), 128, 0,
                   stream>>>(keys_fp16, residual, errors, values_fp16, key,
                             value, cache, table, table_stride, starts, lengths,
                             capacity, signs, dtype, format);
  int status = hipGetLastError();
  if (status != hipSuccess) return status;
  status = octave_rdna4::launch_dense_prefill(
      out, static_cast<const _Float16*>(query_fp16), keys_fp16, residual,
      errors, values_fp16, starts, lengths, requests, tokens, cache.heads,
      heads / cache.heads, capacity, scale, repair, stream);
  if (status != hipSuccess) return status;

  return dispatch(dtype, format, [&](auto scalar, auto layout) {
    finish_prefill<decltype(scalar), decltype(layout), false>
        <<<dim3(tokens, heads), kWave, 0, stream>>>(
            out, nullptr, nullptr, repair, query, key, value, cache, table,
            table_stride, starts, lengths, requests, signs, 1, scale, 0);
  });
}

int octave_rdna4::launch_prefill_streamed(
    float* out, Activation query, Activation key, Activation value,
    PagedCache cache, const int32_t* table, int64_t table_stride,
    int64_t prefix_capacity, const int32_t* starts, const int32_t* lengths,
    int requests, const uint32_t* signs, int tokens, int heads, float scale,
    int dtype, int format, int splits, float* partials, int8_t* staging,
    float* chunk_lse, uint8_t* repair, hipStream_t stream) {
  // Query scales share storage with the later causal chunk normalizers.
  const dim3 rows(tokens, heads);
  int status = dispatch(dtype, format, [&](auto scalar, auto layout) {
    prepare_prefix_query<decltype(scalar), decltype(layout)::compact>
        <<<rows, kWave, 0, stream>>>(query, signs, staging, chunk_lse, repair,
                                     scale, starts, lengths, requests);
  });
  if (status != hipSuccess) return status;
  status =
      launch_prefix_tiles(staging, chunk_lse, cache, table, table_stride,
                          prefix_capacity, starts, lengths, partials, scale,
                          requests, heads, tokens, splits, format, stream);
  if (status != hipSuccess) return status;
  status = launch_chunk_only(out, query, key, value, staging, chunk_lse, starts,
                             requests, tokens, cache.heads, heads / cache.heads,
                             scale, dtype, stream);
  if (status != hipSuccess) return status;
  return dispatch(dtype, format, [&](auto scalar, auto layout) {
    finish_prefill<decltype(scalar), decltype(layout), true>
        <<<rows, kWave, 0, stream>>>(out, partials, chunk_lse, repair, query,
                                     key, value, cache, table, table_stride,
                                     starts, lengths, requests, signs, splits,
                                     scale, prefix_capacity);
  });
}
