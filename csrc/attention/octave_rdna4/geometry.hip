// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Fixed D=256/R=64 Octave matrix attention and sparse-cache kernels.
#include <cmath>
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "kernels.h"
#include "split_merge.cuh"

namespace {
using octave_rdna4::Activation;
using octave_rdna4::PagedCache;
using octave_rdna4::Shadow;

constexpr int LD = 256, LR = 64, W = 32, Partial = LD + 2;
template <bool C, int VB>
struct Format {
  static constexpr bool compact = C;
  static constexpr int vbits = VB;
  static constexpr int D = LD, R = C ? 0 : LR;
  static constexpr int VL = D / 32, KS = C ? 1 : D / 64, KOff = R / 2,
                       VOff = KOff + (D - R) * 3 / 8;
  static constexpr int SOff = VOff + D * VB / 8, Align = C ? 4 : 8,
                       Bytes =
                           (SOff + 2 * (KS + 1) + Align - 1) / Align * Align;
  static constexpr bool VWhole = (D & (D - 1)) == 0;
  static constexpr int Waves = D > 512 ? 8 : 4;
  static constexpr int Cols = (D + 16 * Waves - 1) / (16 * Waves) * 16;
  static constexpr bool Wide = D > 512;
  static constexpr int Rows = 16, Tile = Wide ? 8 : 16;
  static constexpr int OutputRows = Wide ? 8 : Rows;
  static constexpr int VStride = D == 512 ? Tile + 8 : Tile;
  static_assert(D <= 1024, "storage head limit");
  __device__ static int logical(int d) {
    if constexpr (C) return d < LD ? d : -1;
    return d < LR ? d : (d >= R && d < R + LD - LR ? LR + d - R : -1);
  }
};
// gfx12 register-lane exchange avoids the LDS permutation path.
__device__ __forceinline__ float sx(float x, int delta) {
  uint32_t bits = __float_as_uint(x);
  uint32_t selector = (uint32_t(threadIdx.x) ^ uint32_t(delta)) & 15u;
  uint32_t result =
      (delta & 16)
          ? __builtin_amdgcn_permlanex16_var(bits, bits, selector, false, false)
          : __builtin_amdgcn_permlane16_var(bits, bits, selector, false, false);
  return __uint_as_float(result);
}
__device__ float sum(float x) {
  for (int d = 16; d; d /= 2) x += sx(x, d);
  return x;
}
__device__ float maximum(float x) {
  for (int d = 16; d; d /= 2) x = fmaxf(x, sx(x, d));
  return x;
}
template <class F, class T>
__device__ void load(float (&x)[F::VL], Activation a, int token, int head,
                     int lane) {
  const T* p = static_cast<const T*>(a.data) + token * a.token_stride +
               head * a.head_stride;
#pragma unroll
  for (int i = 0; i < F::VL; ++i) {
    int l = F::logical(lane + 32 * i);
    x[i] = l >= 0 ? float(p[l]) : 0.f;
  }
}
template <class F>
__device__ void signs(float (&x)[F::VL], const uint32_t* s, int lane) {
#pragma unroll
  for (int i = 0; i < F::VL; ++i)
    if ((s[i] >> lane) & 1u) x[i] = -x[i];
}
template <class F, bool Whole>
__device__ void transform(float (&x)[F::VL], int lane) {
#pragma unroll
  for (int delta = 1; delta < 32; delta *= 2) {
#pragma unroll
    for (int i = 0; i < F::VL; ++i) {
      float o = sx(x[i], delta);
      x[i] = (lane & delta) ? o - x[i] : x[i] + o;
    }
  }
  constexpr int regs = Whole ? F::VL : 2;
#pragma unroll
  for (int delta = 1; delta < regs; delta *= 2) {
#pragma unroll
    for (int i = 0; i < F::VL; ++i)
      if (!(i & delta)) {
        float a = x[i], b = x[i + delta];
        x[i] = a + b;
        x[i + delta] = a - b;
      }
  }
  constexpr float inv = Whole ? (F::D == 64    ? .125f
                                 : F::D == 128 ? .08838834764831845f
                                 : F::D == 256 ? .0625f
                                 : F::D == 512 ? .04419417382415922f
                                               : .03125f)
                              : .125f;
#pragma unroll
  for (int i = 0; i < F::VL; ++i) x[i] *= inv;
}
template <int Bits>
__device__ int centroid(int c) {
  if constexpr (Bits == 3) {
    return int(
        int8_t(__builtin_amdgcn_perm(0x7f4f2d0eu, 0xf2d3b181u, uint32_t(c))));
  } else {
    uint32_t lo =
        __builtin_amdgcn_perm(0xfaeee2d5u, 0xc6b6a182u, uint32_t(c & 7));
    uint32_t hi =
        __builtin_amdgcn_perm(0x7e5f4a3au, 0x2b1e1206u, uint32_t(c & 7));
    return int(int8_t(c < 8 ? lo : hi));
  }
}
template <int Bits>
__device__ int code(float x, float rms) {
  float z = x / rms * (Bits == 3 ? 59.f : 46.f);
  int c = 0;
#pragma unroll
  for (int j = 1; j < (1 << Bits); ++j)
    c += z > (centroid<Bits>(j - 1) + centroid<Bits>(j)) * .5f;
  return c;
}
template <int Bits>
__device__ int unpack(const uint8_t* p, int dims, int d) {
  const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
  if constexpr (Bits == 4)
    return (w[d / 8] >> (8 * (d % 4) + 4 * ((d % 8) / 4))) & 15;
  else
    return ((w[d / 16] >> (8 * (d % 4) + 2 * ((d % 16) / 4))) & 3) |
           (((w[dims / 16 + d / 32] >> (8 * (d % 4) + (d % 32) / 4)) & 1) << 2);
}
template <class F>
__device__ float kval(const uint8_t* p, int d) {
  const __half* s = reinterpret_cast<const __half*>(p + F::SOff);
  if constexpr (F::compact)
    return centroid<3>(unpack<3>(p, F::D, d)) * (__half2float(s[0]) / 64.f);
  else if (d < F::R)
    return centroid<4>(unpack<4>(p, F::R, d)) *
           (__half2float(s[d / 64]) / 64.f);
  else
    return centroid<3>(unpack<3>(p + F::KOff, F::D - F::R, d - F::R)) *
           (__half2float(s[d / 64]) / 64.f);
}
template <class F>
__device__ float vval(const uint8_t* p, int d) {
  return centroid<F::vbits>(unpack<F::vbits>(p + F::VOff, F::D, d)) *
         (__half2float(reinterpret_cast<const __half*>(p + F::SOff)[F::KS]) /
          64.f);
}
__device__ int request_for(const int32_t* starts, int n, int q) {
  int lo = 0, hi = n;
  while (lo + 1 < hi) {
    int m = (lo + hi) / 2;
    if (starts[m] <= q)
      lo = m;
    else
      hi = m;
  }
  return lo;
}
template <bool P>
__device__ void metadata(int q, const int32_t* map, const int32_t* lengths,
                         int requests, int& req, int& prefix, int& length,
                         int& start) {
  req = prefix = length = start = 0;
  if constexpr (P) {
    req = request_for(map, requests, q);
    start = map[req];
    prefix = lengths[req] - (map[req + 1] - start);
    length = prefix + q - start + 1;
  } else {
    length = max(0, lengths[q]);
    prefix = length;
    if (length > 0) req = map[q];
  }
}
template <class F, class T>
__global__ void store(Activation key, Activation value, PagedCache cache,
                      const int64_t* slots, const uint32_t* sgn) {
  const int lane = threadIdx.x, t = blockIdx.x, h = blockIdx.y;
  int64_t slot = slots[t];
  if (slot < 0) return;
  __shared__ uint8_t kc[F::D], vc[F::D];
  float k[F::VL], v[F::VL];
  load<F, T>(k, key, t, h, lane);
  load<F, T>(v, value, t, h, lane);
  signs<F>(k, sgn, lane);
  signs<F>(v, sgn, lane);
  transform<F, F::compact>(k, lane);
  transform<F, F::VWhole>(v, lane);
  uint8_t* dst = cache.at(slot / cache.page_size, h, slot % cache.page_size);
  for (int w = lane; w < F::Bytes / 4; w += 32)
    reinterpret_cast<uint32_t*>(dst)[w] = 0;
  __syncthreads();
  __half* sc = reinterpret_cast<__half*>(dst + F::SOff);
  if constexpr (F::compact) {
    float e = 0, dot = 0;
#pragma unroll
    for (int i = 0; i < F::VL; ++i) e += k[i] * k[i];
    e = sum(e);
    float rms = fmaxf(sqrtf(e / F::D), 1e-12f);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) {
      int c = code<3>(k[i], rms);
      kc[lane + 32 * i] = c;
      dot += k[i] * centroid<3>(c);
    }
    dot = sum(dot);
    if (lane == 0)
      sc[0] = __float2half_rn(dot > 0 ? e / fmaxf(dot, 1e-30f) * 64 : 0);
  } else {
#pragma unroll
    for (int b = 0; b < F::D / 64; ++b) {
      float e = sum(k[2 * b] * k[2 * b] + k[2 * b + 1] * k[2 * b + 1]);
      float rms = fmaxf(sqrtf(e / 64.f), 1e-12f);
      float dot = 0;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int d = lane + (2 * b + i) * 32;
        int c =
            d < F::R ? code<4>(k[2 * b + i], rms) : code<3>(k[2 * b + i], rms);
        kc[d] = c;
        dot += k[2 * b + i] * (d < F::R ? centroid<4>(c) : centroid<3>(c));
      }
      dot = sum(dot);
      if (lane == 0)
        sc[b] = __float2half_rn(dot > 0 ? e / fmaxf(dot, 1e-30f) * 64 : 0);
    }
  }
  float e = 0, dot = 0;
#pragma unroll
  for (int i = 0; i < F::VL; ++i) e += v[i] * v[i];
  e = sum(e);
  float rms = fmaxf(sqrtf(e / F::D), 1e-12f);
#pragma unroll
  for (int i = 0; i < F::VL; ++i) {
    int c = code<F::vbits>(v[i], rms);
    vc[lane + 32 * i] = c;
    dot += v[i] * centroid<F::vbits>(c);
  }
  dot = sum(dot);
  if (lane == 0)
    sc[F::KS] = __float2half_rn(dot > 0 ? e / fmaxf(dot, 1e-30f) * 64 : 0);
  __syncthreads();
  for (int section = 0; section < 3; ++section) {
    const uint8_t* src = section == 2 ? vc : kc;
    int base = section == 1 ? F::R : 0;
    int n = section == 0 ? F::R : section == 1 ? F::D - F::R : F::D;
    int bits = section == 0 ? 4 : section == 1 ? 3 : F::vbits;
    uint8_t* out = dst + (section == 0 ? 0 : section == 1 ? F::KOff : F::VOff);
    if (bits == 4) {
      for (int w = lane; w < n / 8; w += 32) {
        uint32_t x = 0;
        for (int j = 0; j < 4; ++j)
          x |=
              uint32_t(src[base + 8 * w + j] | (src[base + 8 * w + j + 4] << 4))
              << (8 * j);
        reinterpret_cast<uint32_t*>(out)[w] = x;
      }
    } else {
      for (int w = lane; w < n / 16; w += 32) {
        uint32_t x = 0;
        for (int j = 0; j < 16; ++j)
          x |= uint32_t(src[base + 16 * w + j] & 3)
               << (8 * (j % 4) + 2 * (j / 4));
        reinterpret_cast<uint32_t*>(out)[w] = x;
      }
      for (int w = lane; w < n / 32; w += 32) {
        uint32_t x = 0;
        for (int j = 0; j < 32; ++j)
          x |= uint32_t(src[base + 32 * w + j] >> 2) << (8 * (j % 4) + j / 4);
        reinterpret_cast<uint32_t*>(out + n / 4)[w] = x;
      }
    }
  }
}
__device__ __forceinline__ float scalar_shuffle_sum(float x) {
#pragma unroll
  for (int d = 16; d; d /= 2) x += __shfl_xor(x, d, 32);
  return x;
}
template <class F, class T, bool P, bool Shuffle = false>
__device__ void scalar_row(float* out, float* partials, Activation query,
                           Activation key, Activation value, PagedCache cache,
                           const int32_t* table, int64_t stride,
                           const int32_t* map, const int32_t* lengths,
                           int requests, const uint32_t* sgn, float scale,
                           int q, int head, int split, int heads, int splits,
                           int lane) {
  int req, prefix, length, start;
  metadata<P>(q, map, lengths, requests, req, prefix, length, start);
  int width = (length + splits - 1) / splits, begin = split * width,
      end = min(length, begin + width), kv = head / (heads / cache.heads);
  float packed[F::VL] = {}, rawout[F::VL] = {}, qr[F::VL], qt[F::VL];
  float m = -INFINITY, mass = 0;
  if (begin < end) {
    load<F, T>(qr, query, q, head, lane);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) qt[i] = qr[i];
    signs<F>(qt, sgn, lane);
    transform<F, F::compact>(qt, lane);
    for (int pos = begin; pos < end; ++pos) {
      bool pk = !P || pos < prefix;
      const uint8_t* slot = nullptr;
      float dot = 0, k[F::VL], v[F::VL];
      if (pk) {
        int page = table[req * stride + pos / cache.page_size];
        slot = cache.at(page, kv, pos % cache.page_size);
#pragma unroll
        for (int i = 0; i < F::VL; ++i)
          dot = fmaf(qt[i], kval<F>(slot, lane + 32 * i), dot);
      } else {
        load<F, T>(k, key, start + pos - prefix, kv, lane);
        load<F, T>(v, value, start + pos - prefix, kv, lane);
#pragma unroll
        for (int i = 0; i < F::VL; ++i) dot = fmaf(qr[i], k[i], dot);
      }
      float logit = (Shuffle ? scalar_shuffle_sum(dot) : sum(dot)) * scale,
            nm = fmaxf(m, logit), rs = mass > 0 ? __expf(m - nm) : 0,
            w = __expf(logit - nm);
      mass = fmaf(mass, rs, w);
      m = nm;
#pragma unroll
      for (int i = 0; i < F::VL; ++i) {
        packed[i] = pk ? fmaf(w, vval<F>(slot, lane + 32 * i), packed[i] * rs)
                       : packed[i] * rs;
        rawout[i] = pk ? rawout[i] * rs : fmaf(w, v[i], rawout[i] * rs);
      }
    }
    transform<F, F::VWhole>(packed, lane);
    signs<F>(packed, sgn, lane);
  }
  float* dst =
      splits == 1
          ? out + (int64_t(q) * heads + head) * LD
          : partials + ((int64_t(q) * heads + head) * splits + split) * Partial;
#pragma unroll
  for (int i = 0; i < F::VL; ++i) {
    int l = F::logical(lane + 32 * i);
    if (l >= 0)
      dst[l] = (packed[i] + rawout[i]) *
               (splits == 1 ? (mass > 0 ? 1 / mass : 0) : 1);
  }
  if (splits > 1 && lane == 0) {
    dst[LD] = m;
    dst[LD + 1] = mass;
  }
}
template <class F, class T, bool P, bool Shuffle = false>
__global__ __launch_bounds__(32) void scalar(
    float* out, float* partials, Activation q, Activation k, Activation v,
    PagedCache c, const int32_t* table, int64_t stride, const int32_t* map,
    const int32_t* lengths, int requests, const uint32_t* sgn, float scale,
    int tokens, int heads, bool repair_only) {
  const int qid = repair_only ? blockIdx.y : blockIdx.x;
  const int head = repair_only ? blockIdx.x : blockIdx.y;
  if constexpr (!Shuffle) {
    if (repair_only) {
      int req, prefix, length, start;
      metadata<P>(qid, map, lengths, requests, req, prefix, length, start);
      if (length > 0) {
        const float* current =
            gridDim.z == 1
                ? out + (int64_t(qid) * heads + head) * LD
                : partials +
                      ((int64_t(qid) * heads + head) * gridDim.z + blockIdx.z) *
                          Partial;
        if (!__builtin_isnan(current[0])) return;
      }
    }
  }
  scalar_row<F, T, P, Shuffle>(
      out, partials, q, k, v, c, table, stride, map, lengths, requests, sgn,
      scale, qid, head, blockIdx.z, heads, gridDim.z, threadIdx.x);
}
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
// Only shared memory is exchanged by the matrix kernel's workgroup.
// Match HIP's local-memory barrier; global K/V and metadata are read-only.
__device__ inline __attribute__((convergent)) void matrix_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
template <class F, bool Wide>
struct MatrixInput;
template <class F>
struct MatrixInput<F, false> {
  alignas(16) _Float16 q[F::Rows][F::D + 8], k[F::Tile][F::D + 8];
  alignas(16) _Float16 v[F::D][F::VStride];
};
template <class F>
struct MatrixInput<F, true> {
  alignas(16) _Float16 q[F::Rows][F::D + 8];
  union {
    alignas(16) _Float16 k[F::Tile][F::D + 8];
    alignas(16) _Float16 v[F::D][F::VStride];
  };
};
template <class F>
struct alignas(16) Shared {
  union Data {
    MatrixInput<F, F::Wide> input;
    float output[F::OutputRows][F::D];
  } data;
  float partial[F::Waves][F::Wide ? F::Rows : 16][F::Wide ? F::Tile : 16];
  _Float16 prob[16][16];
  float rs[16], mx[16], mass[16];
  int req[F::Rows], prefix[F::Rows], start[F::Rows], length[F::Rows],
      begin[F::Rows], end[F::Rows];
  uint32_t unsafe, valid, kv_unsafe;
  int boundary;
};
template <class F, class T, bool P, bool Parallel>
__global__ __launch_bounds__(32 * F::Waves) void matrix(
    float* out, float* partials, Activation query, Activation key,
    Activation value, PagedCache cache, const int32_t* table, int64_t stride,
    const int32_t* map, const int32_t* lengths, int requests,
    const uint32_t* sgn, float scale, int tokens, int heads, bool request_tiles) {
  static_assert(sizeof(Shared<F>) <= 65536, "geometry LDS bound");
  __shared__ Shared<F> s;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32,
            col = lane % 16, half = lane / 16, split = blockIdx.z,
            splits = gridDim.z;
  const int group = heads / cache.heads, bh = min(group, F::Rows),
            ht = (group + bh - 1) / bh, kv = blockIdx.y / ht,
            hbase = (blockIdx.y % ht) * bh;
  const int tile_queries = P ? F::Rows / bh : 1;
  int queries = tile_queries, qbase = blockIdx.x * tile_queries;
  // Request boundaries change tile assignment, not the matrix calculation.
  if constexpr (P) {
    if (request_tiles) {
      const int regular = (tokens + tile_queries - 1) / tile_queries;
      if (blockIdx.x < regular) {
        const int req = request_for(map, requests, qbase);
        queries = min(tile_queries, map[req + 1] - qbase);
      } else {
        const int req = blockIdx.x - regular;
        qbase = map[req];
        if (qbase == map[req + 1] || qbase % tile_queries == 0) return;
        queries = min(map[req + 1] - qbase,
                      tile_queries - qbase % tile_queries);
      }
    }
  }
  if (tid == 0) {
    s.unsafe = s.valid = s.kv_unsafe = 0;
    s.boundary = 0;
  }
  if (tid < F::Rows) {
    int req = 0, p = 0, l = 0, start = 0, q = qbase + tid;
    if (tid < queries && q < tokens)
      metadata<P>(q, map, lengths, requests, req, p, l, start);
    int width = (l + splits - 1) / splits;
    s.req[tid] = req;
    s.prefix[tid] = p;
    s.length[tid] = l;
    s.start[tid] = start;
    s.begin[tid] = split * width;
    s.end[tid] = min(l, (split + 1) * width);
  }
  if (tid < 16) {
    s.mx[tid] = -INFINITY;
    s.mass[tid] = 0;
  }
  matrix_lds_barrier();
  if (tid == 0) {
    for (int i = 1; i < queries; ++i)
      if (s.length[i] > 0 && s.req[i] != s.req[0]) s.boundary = 1;
  }
  matrix_lds_barrier();
  if (s.boundary) return;
  for (int row = wave; row < F::Rows; row += F::Waves) {
    int qi = row / bh, h = hbase + row % bh, q = qbase + qi;
    bool live = qi < queries && q < tokens && h < group && s.length[qi] > 0;
    float raw[F::VL] = {}, rot[F::VL];
    if (live) load<F, T>(raw, query, q, kv * group + h, lane);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) rot[i] = raw[i];
    signs<F>(rot, sgn, lane);
    transform<F, F::compact>(rot, lane);
    bool safe = true;
#pragma unroll
    for (int i = 0; i < F::VL; ++i) {
      safe &= __builtin_isfinite(raw[i]) && __builtin_isfinite(rot[i]);
      safe &= fabsf(raw[i]) * fabsf(scale) <= 2.f &&
              fabsf(rot[i]) * fabsf(scale) <= 2.f;
      s.data.input.q[row][lane + 32 * i] = (_Float16)rot[i];
    }
    if (live && lane == 0) atomicOr(&s.valid, 1u << row);
    if (live && !__all(safe) && lane == 0) atomicOr(&s.unsafe, 1u << row);
  }
  matrix_lds_barrier();
  // Publish the range decision once; the guard kernel need not re-rotate Q.
  for (int row = tid; row < F::Rows; row += blockDim.x) {
    if (!(s.unsafe & (1u << row))) continue;
    int qi = row / bh, h = hbase + row % bh, q = qbase + qi;
    float* dst =
        splits == 1
            ? out + (int64_t(q) * heads + kv * group + h) * LD
            : partials +
                  ((int64_t(q) * heads + kv * group + h) * splits + split) *
                      Partial;
    dst[0] = __builtin_nanf("");
  }
  const uint32_t unsafe_mask = s.unsafe;
  if (s.valid == s.unsafe) return;
  int first = 0x7fffffff, last = 0;
  for (int i = 0; i < queries; ++i)
    if (s.begin[i] < s.end[i]) {
      first = min(first, s.begin[i]);
      last = max(last, s.end[i]);
    }
  const int req = s.req[0], prefix = s.prefix[0], start = s.start[0];
  constexpr int Vecs = F::Cols / 16;
  F8 accum[Vecs] = {};
  F8 running_max, running_mass = {};
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;
  bool query_packed = true;
  for (int posbase = first; posbase < last;) {
    bool packed = !P || posbase < prefix;
    int count = min(F::Tile, last - posbase);
    if (P && packed) count = min(count, prefix - posbase);

    if (packed != query_packed) {
      for (int row = wave; row < F::Rows; row += F::Waves) {
        int qi = row / bh, h = hbase + row % bh, q = qbase + qi;
        bool live = qi < queries && q < tokens && h < group && s.length[qi] > 0;
        float x[F::VL] = {};
        if (live) load<F, T>(x, query, q, kv * group + h, lane);
        if (packed) {
          signs<F>(x, sgn, lane);
          transform<F, F::compact>(x, lane);
        }
#pragma unroll
        for (int i = 0; i < F::VL; ++i)
          s.data.input.q[row][lane + 32 * i] = (_Float16)x[i];
      }
      matrix_lds_barrier();
      query_packed = packed;
    }
    for (int t = wave; t < F::Tile; t += F::Waves) {
      int pos = posbase + t;
      bool live = t < count;
      bool any = false;
      for (int i = 0; i < queries; ++i)
        any |= pos >= s.begin[i] && pos < s.end[i];
      live &= any;
      if constexpr (F::Wide) {
        float k[F::VL] = {};
        if (live) {
          if (packed) {
            int page = table[req * stride + pos / cache.page_size];
            const uint8_t* slot = cache.at(page, kv, pos % cache.page_size);
#pragma unroll
            for (int i = 0; i < F::VL; ++i) k[i] = kval<F>(slot, lane + 32 * i);
          } else
            load<F, T>(k, key, start + pos - prefix, kv, lane);
        }
#pragma unroll
        for (int i = 0; i < F::VL; ++i)
          s.data.input.k[t][lane + 32 * i] = (_Float16)k[i];
      } else {
        float k[F::VL] = {}, v[F::VL] = {};
        if (live) {
          if (packed) {
            int page = table[req * stride + pos / cache.page_size];
            const uint8_t* slot = cache.at(page, kv, pos % cache.page_size);
#pragma unroll
            for (int i = 0; i < F::VL; ++i) {
              k[i] = kval<F>(slot, lane + 32 * i);
              v[i] = vval<F>(slot, lane + 32 * i);
            }
          } else {
            load<F, T>(k, key, start + pos - prefix, kv, lane);
            load<F, T>(v, value, start + pos - prefix, kv, lane);
            signs<F>(v, sgn, lane);
            transform<F, F::VWhole>(v, lane);
          }
        }
#pragma unroll
        for (int i = 0; i < F::VL; ++i) {
          s.data.input.k[t][lane + 32 * i] = (_Float16)k[i];
          s.data.input.v[lane + 32 * i][t] = (_Float16)v[i];
        }
      }
    }
    matrix_lds_barrier();
    F8 score = {};
    if (Parallel || wave == 0) {
      for (int d = Parallel ? wave * 16 : 0; d < F::D;
           d += Parallel ? 16 * F::Waves : 16) {
        H8 a = {}, b = {};
        if (col < F::Rows)
          a = *reinterpret_cast<const H8*>(&s.data.input.q[col][d + half * 8]);
        if (col < F::Tile)
          b = *reinterpret_cast<const H8*>(&s.data.input.k[col][d + half * 8]);
        score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score);
      }
    }
    if constexpr (Parallel) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (!F::Wide || col < F::Tile)
          s.partial[wave][half * 8 + e][col] = score[e];
      matrix_lds_barrier();
      if (wave == 0) {
        score = {};
        for (int g = 0; g < F::Waves; ++g) {
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (!F::Wide || col < F::Tile)
              score[e] += s.partial[g][half * 8 + e][col];
        }
      }
    }
    if (wave == 0) {
      bool score_safe = true;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half * 8 + e, qi = row / bh, pos = posbase + col;
        bool live = row < F::Rows && qi < queries && qbase + qi < tokens &&
                    hbase + row % bh < group && col < count &&
                    pos >= s.begin[qi] && pos < s.end[qi];
        float x = live ? score[e] * scale : -INFINITY, m = x;
        score_safe &=
            !live || (unsafe_mask & (1u << row)) || __builtin_isfinite(x);
        for (int d = 8; d; d /= 2) m = fmaxf(m, sx(m, d));
        float nm = fmaxf(running_max[e], m),
              rs = running_mass[e] > 0 ? __expf(running_max[e] - nm) : 0,
              w = live ? __expf(x - nm) : 0, mass = w;
        for (int d = 8; d; d /= 2) mass += sx(mass, d);
        running_mass[e] = fmaf(running_mass[e], rs, mass);
        running_max[e] = nm;
        s.prob[row][col] = (_Float16)w;
        if (col == 0) {
          s.rs[row] = rs;
          s.mx[row] = nm;
          s.mass[row] = running_mass[e];
        }
      }
      if (!__all(score_safe) && lane == 0) atomicOr(&s.kv_unsafe, 1u);
    }
    matrix_lds_barrier();
    if (s.kv_unsafe) {
      // The following guard kernel recomputes these logical rows in FP32.
      // Each query/head/split has exactly one writer, including partial tiles.
      for (int row = tid; row < F::Rows; row += blockDim.x) {
        if (!(s.valid & (1u << row))) continue;
        int qi = row / bh, h = hbase + row % bh, q = qbase + qi;
        float* dst =
            splits == 1
                ? out + (int64_t(q) * heads + kv * group + h) * LD
                : partials +
                      ((int64_t(q) * heads + kv * group + h) * splits + split) *
                          Partial;
        dst[0] = __builtin_nanf("");
      }
      return;
    }
    if constexpr (F::Wide) {
      // All QK readers have completed at the preceding barrier. Reuse the
      // same plane for V, keeping the K and V decode vectors disjoint.
      for (int t = wave; t < F::Tile; t += F::Waves) {
        int pos = posbase + t;
        bool live = t < count, any = false;
        for (int i = 0; i < queries; ++i)
          any |= pos >= s.begin[i] && pos < s.end[i];
        live &= any;
        float v[F::VL] = {};
        if (live) {
          if (packed) {
            int page = table[req * stride + pos / cache.page_size];
            const uint8_t* slot = cache.at(page, kv, pos % cache.page_size);
#pragma unroll
            for (int i = 0; i < F::VL; ++i) v[i] = vval<F>(slot, lane + 32 * i);
          } else {
            load<F, T>(v, value, start + pos - prefix, kv, lane);
            signs<F>(v, sgn, lane);
            transform<F, F::VWhole>(v, lane);
          }
        }
#pragma unroll
        for (int i = 0; i < F::VL; ++i)
          s.data.input.v[lane + 32 * i][t] = (_Float16)v[i];
      }
      matrix_lds_barrier();
    }
#pragma unroll
    for (int d = 0; d < Vecs; ++d) {
      H8 a = *reinterpret_cast<const H8*>(&s.prob[col][half * 8]);
      H8 b = {};
      const int dim = wave * F::Cols + d * 16 + col;
      if (half * 8 < F::Tile && dim < F::D)
        b = *reinterpret_cast<const H8*>(&s.data.input.v[dim][half * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) accum[d][e] *= s.rs[half * 8 + e];
      accum[d] =
          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, accum[d]);
    }
    matrix_lds_barrier();
    posbase += count;
  }
  for (int output_base = 0; output_base < F::Rows;
       output_base += F::OutputRows) {
#pragma unroll
    for (int d = 0; d < Vecs; ++d) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int row = half * 8 + e;
        if (row >= output_base && row < output_base + F::OutputRows &&
            wave * F::Cols + d * 16 + col < F::D)
          s.data.output[row - output_base][wave * F::Cols + d * 16 + col] =
              accum[d][e];
      }
    }
    matrix_lds_barrier();
    for (int row = output_base + wave; row < output_base + F::OutputRows;
         row += F::Waves) {
      int qi = row / bh, h = hbase + row % bh, q = qbase + qi;
      if (!(s.valid & (1u << row)) || (s.unsafe & (1u << row))) continue;
      float v[F::VL];
#pragma unroll
      for (int i = 0; i < F::VL; ++i)
        v[i] = s.data.output[row - output_base][lane + 32 * i];
      transform<F, F::VWhole>(v, lane);
      signs<F>(v, sgn, lane);
      float* dst =
          splits == 1
              ? out + (int64_t(q) * heads + kv * group + h) * LD
              : partials +
                    ((int64_t(q) * heads + kv * group + h) * splits + split) *
                        Partial;
      float mass = s.mass[row];
      bool output_safe = true;
      float largest = 0;
#pragma unroll
      for (int i = 0; i < F::VL; ++i) {
        int l = F::logical(lane + 32 * i);
        if (l >= 0) {
          const float result =
              v[i] * (splits == 1 ? (mass > 0 ? 1 / mass : 0) : 1);
          output_safe &= __builtin_isfinite(result);
          largest = fmaxf(
              largest, fabsf(result) *
                           (splits == 1 ? 1.f : (mass > 0 ? 1.f / mass : 0.f)));
          dst[l] = result;
        }
      }
      // Logical dimension zero is owned by lane zero; publish after its store.
      // FP16 subnormal rounding can dominate a tiny output. Use the normalized
      // partial's magnitude, so the decision is independent of split mass.
      bool tiny = mass > 0 && maximum(largest) < 1.e-4f;
      if ((!__all(output_safe) || tiny) && lane == 0)
        dst[0] = __builtin_nanf("");
      if (splits > 1 && lane == 0) {
        dst[LD] = s.mx[row];
        dst[LD + 1] = mass;
      }
    }
    matrix_lds_barrier();
  }
}
__global__ void merge(float* out, const float* partial, int splits) {
  const int64_t row = int64_t(blockIdx.x) * gridDim.y + blockIdx.y;
  octave_rdna4::merge_wave<LD>(
      out + row * LD, partial + row * splits * Partial, splits);
}

__global__ __launch_bounds__(256) void merge_parallel(float* out,
                                                      const float* partial,
                                                      int splits) {
  extern __shared__ float shared[];
  const int64_t row = int64_t(blockIdx.x) * gridDim.y + blockIdx.y;
  octave_rdna4::merge_parallel<LD, false, true>(
      out + row * LD, partial + row * splits * Partial, splits, shared);
}

void launch_merge(float* out, const float* partial, int tokens, int heads,
                  int splits, hipStream_t stream) {
  const dim3 grid(tokens, heads);
  const int threads = octave_rdna4::merge_threads(int64_t(tokens) * heads, splits);
  if (threads > W)
    merge_parallel
        <<<grid, threads, octave_rdna4::merge_shared_bytes<LD>(threads, splits),
           stream>>>(out, partial, splits);
  else
    merge<<<grid, W, 0, stream>>>(out, partial, splits);
}

template <class T, class Fn>
int format_dispatch(int format, Fn fn) {
  switch (format) {
    case 3:
      fn(T{}, Format<false, 3>{});
      break;
    case 4:
      fn(T{}, Format<false, 4>{});
      break;
    case 19:
      fn(T{}, Format<true, 3>{});
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipGetLastError();
}
template <class Fn>
int dispatch(int dtype, int format, Fn fn) {
  if (dtype == 0) return format_dispatch<__half>(format, fn);
  if (dtype == 1) return format_dispatch<hip_bfloat16>(format, fn);
  return hipErrorInvalidValue;
}
template <bool P>
int attention(float* out, float* partial, Activation q, Activation k,
              Activation v, PagedCache cache, const int32_t* table,
              int64_t stride, const int32_t* map, const int32_t* lengths,
              int requests, const uint32_t* signs, int tokens, int heads,
              int splits, float scale, int dtype, int format,
              hipStream_t stream) {
  if (!tokens) return hipSuccess;
  int rc = dispatch(dtype, format, [&](auto t, auto f) {
    using T = decltype(t);
    using F = decltype(f);
    int group = heads / cache.heads;
    int bh = min(group, F::Rows), tiles = (group + bh - 1) / bh,
        qs = P ? F::Rows / bh : 1;
    auto launch_scalar = [&]() {
      if constexpr (!P && F::D <= 256) {
        if (int64_t(tokens) * heads > 256) {
          scalar<F, T, P, true><<<dim3(tokens, heads, splits), 32, 0, stream>>>(
              out, partial, q, k, v, cache, table, stride, map, lengths,
              requests, signs, scale, tokens, heads, false);
          return;
        }
      }
      scalar<F, T, P, false><<<dim3(tokens, heads, splits), 32, 0, stream>>>(
          out, partial, q, k, v, cache, table, stride, map, lengths, requests,
          signs, scale, tokens, heads, false);
    };
    // Tiny requests and independent one-query prefill avoid matrix setup.
    bool mat = P ? (tokens >= 16 && tokens > requests) : (group >= 4);
    if constexpr (LD < 32) {
      // A padded WMMA head wastes most lanes here and amplifies cancellation
      // in the cropped logical output. Keep this edge domain in FP32.
      launch_scalar();
    } else if (mat) {
      if constexpr (P) {
        if (qs > 1 && requests > 1)
          matrix<F, T, P, true>
              <<<dim3((tokens + qs - 1) / qs + requests,
                      cache.heads * tiles, splits),
                 32 * F::Waves, 0, stream>>>(
              out, partial, q, k, v, cache, table, stride, map, lengths,
              requests, signs, scale, tokens, heads, true);
        else
          matrix<F, T, P, true>
              <<<dim3((tokens + qs - 1) / qs, cache.heads * tiles, splits),
                 32 * F::Waves, 0, stream>>>(
                  out, partial, q, k, v, cache, table, stride, map, lengths,
                  requests, signs, scale, tokens, heads, false);
      } else {
        matrix<F, T, P, true>
            <<<dim3(tokens, cache.heads * tiles, splits), 32 * F::Waves, 0,
               stream>>>(out, partial, q, k, v, cache, table, stride, map,
                         lengths, requests, signs, scale, tokens, heads, false);
      }
      scalar<F, T, P, false><<<dim3(heads, tokens, splits), 32, 0, stream>>>(
          out, partial, q, k, v, cache, table, stride, map, lengths, requests,
          signs, scale, tokens, heads, true);
    } else
      launch_scalar();
  });
  if (rc != hipSuccess || splits == 1) return rc;
  launch_merge(out, partial, tokens, heads, splits, stream);
  return hipGetLastError();
}
}  // namespace
int octave_rdna4::geometry::launch_store(
    Activation k, Activation v, PagedCache c, const int64_t* slots,
    const uint32_t* signs, int tokens, int dtype, int format, hipStream_t stream) {
  if (!tokens) return hipSuccess;
  return dispatch(dtype, format, [&](auto t, auto f) {
    store<decltype(f), decltype(t)>
        <<<dim3(tokens, c.heads), 32, 0, stream>>>(k, v, c, slots, signs);
  });
}
int octave_rdna4::geometry::launch_decode(
    float* out, float* partial, Activation q, PagedCache c, const int32_t* table,
    int64_t stride, const int32_t* map, const int32_t* lengths,
    const uint32_t* signs, int tokens, int heads, int splits, float scale,
    int dtype, int format, hipStream_t stream) {
  return attention<false>(out, partial, q, {}, {}, c, table, stride, map,
                          lengths, 0, signs, tokens, heads, splits, scale,
                          dtype, format, stream);
}
int octave_rdna4::geometry::launch_prefill(
    float* out, float* partial, Activation q, Activation k, Activation v,
    PagedCache c, const int32_t* table, int64_t stride, const int32_t* starts,
    const int32_t* lengths, int requests, const uint32_t* signs, int tokens,
    int heads, int splits, float scale, int dtype, int format,
    hipStream_t stream) {
  return attention<true>(out, partial, q, k, v, c, table, stride, starts,
                         lengths, requests, signs, tokens, heads, splits, scale,
                         dtype, format, stream);
}

namespace {
// One wave owns a complete shadow row while holding its entry lock. A
// collision evicts a whole row; the tag makes readers fall back to packed KV.
template <class T>
__global__ __launch_bounds__(32) void window_store(
    Activation key, Activation value, const int64_t* slots,
    const int32_t* positions, Shadow recent, Shadow sinks, int heads) {
  const int token = blockIdx.x, lane = threadIdx.x;
  const int64_t slot = slots[token];
  if (slot < 0) return;
  auto put = [&](Shadow table) {
    const int entry = slot & (table.entries - 1);
    if (lane == 0) {
      while (atomicCAS(table.locks + entry, 0, 1) != 0) {
      }
      atomicExch(table.tags + entry, -1);
    }
    __threadfence();
    for (int head = 0; head < heads; ++head) {
      T* row = static_cast<T*>(table.data) +
               (int64_t(entry) * heads + head) * (2 * LD);
      const T* k = static_cast<const T*>(key.data) + token * key.token_stride +
                   head * key.head_stride;
      const T* v = static_cast<const T*>(value.data) +
                   token * value.token_stride + head * value.head_stride;
      for (int d = lane; d < LD; d += 32) {
        row[d] = k[d];
        row[LD + d] = v[d];
      }
    }
    __threadfence();
    if (lane == 0) {
      atomicExch(table.tags + entry, int(slot));
      __threadfence();
      atomicExch(table.locks + entry, 0);
    }
  };
  put(recent);
  if (positions[token] >= 0 && positions[token] < 4) put(sinks);
}

// QSA selects logical positions, including holes. Exact recent/sink hits
// share one online softmax with packed KV; an evicted tag uses packed KV.
template <class F, class T>
__device__ __forceinline__ void sparse_row(
    float* out, float* partial, Activation query, PagedCache cache,
    const int32_t* table, int64_t table_stride, int columns, int requests,
    int pages, const int32_t* mapping, const int32_t* indices,
    int64_t index_stride, int topk, const uint32_t* sgn,
    const int32_t* positions, Shadow recent, Shadow sinks, int heads,
    float scale, int q, int head, int split, int splits, int lane) {
  const int req = mapping[q], kv = head / (heads / cache.heads),
            width = (topk + splits - 1) / splits, begin = split * width,
            end = min(topk, begin + width);
  float qr[F::VL], qt[F::VL], packed[F::VL] = {}, raw[F::VL] = {};
  float m = -INFINITY, mass = 0;
  load<F, T>(qr, query, q, head, lane);
#pragma unroll
  for (int i = 0; i < F::VL; ++i) qt[i] = qr[i];
  signs<F>(qt, sgn, lane);
  transform<F, F::compact>(qt, lane);
  if (req >= 0 && req < requests) {
    for (int index = begin; index < end; ++index) {
      const int pos = indices[int64_t(q) * index_stride + index];
      if (pos < 0 || pos / cache.page_size >= columns) continue;
      const int page =
          table[int64_t(req) * table_stride + pos / cache.page_size];
      if (page < 0 || page >= pages) continue;
      const int slot_id = page * cache.page_size + pos % cache.page_size;
      const uint8_t* slot = cache.at(page, kv, pos % cache.page_size);
      const T* exact = nullptr;
      if (positions && positions[q] >= pos) {
        const int r = slot_id & (recent.entries - 1),
                  s = slot_id & (sinks.entries - 1);
        if (positions[q] - pos < 64 && recent.tags[r] == slot_id)
          exact = static_cast<const T*>(recent.data) +
                  (int64_t(r) * cache.heads + kv) * (2 * LD);
        else if (pos < 4 && sinks.tags[s] == slot_id)
          exact = static_cast<const T*>(sinks.data) +
                  (int64_t(s) * cache.heads + kv) * (2 * LD);
      }
      float dot = 0;
#pragma unroll
      for (int i = 0; i < F::VL; ++i) {
        const int d = lane + 32 * i, logical = F::logical(d);
        const float k = exact ? (logical >= 0 ? float(exact[logical]) : 0.f)
                              : kval<F>(slot, d);
        dot = fmaf(exact ? qr[i] : qt[i], k, dot);
      }
      const float score = sum(dot) * scale, nm = fmaxf(m, score),
                  rescale = mass > 0 ? __expf(m - nm) : 0.f,
                  weight = __expf(score - nm);
      mass = fmaf(mass, rescale, weight);
      m = nm;
#pragma unroll
      for (int i = 0; i < F::VL; ++i) {
        const int d = lane + 32 * i, logical = F::logical(d);
        packed[i] *= rescale;
        raw[i] *= rescale;
        if (exact) {
          if (logical >= 0)
            raw[i] = fmaf(weight, float(exact[LD + logical]), raw[i]);
        } else
          packed[i] = fmaf(weight, vval<F>(slot, d), packed[i]);
      }
    }
  }
  transform<F, F::VWhole>(packed, lane);
  signs<F>(packed, sgn, lane);
  float* dst =
      splits == 1
          ? out + (int64_t(q) * heads + head) * LD
          : partial + ((int64_t(q) * heads + head) * splits + split) * Partial;
#pragma unroll
  for (int i = 0; i < F::VL; ++i) {
    const int logical = F::logical(lane + 32 * i);
    if (logical >= 0)
      dst[logical] = (packed[i] + raw[i]) *
                     (splits == 1 ? (mass > 0 ? 1.f / mass : 0.f) : 1.f);
  }
  if (splits > 1 && lane == 0) {
    dst[LD] = m;
    dst[LD + 1] = mass;
  }
}

template <class F, class T>
__global__ __launch_bounds__(32) void sparse(
    float* out, float* partial, Activation query, PagedCache cache,
    const int32_t* table, int64_t table_stride, int columns, int requests,
    int pages, const int32_t* mapping, const int32_t* indices,
    int64_t index_stride, int topk, const uint32_t* sgn,
    const int32_t* positions, Shadow recent, Shadow sinks, int heads,
    float scale, bool repair) {
  const int64_t row = int64_t(blockIdx.x) * heads + blockIdx.y;
  const float* dst = gridDim.z == 1
                         ? out + row * LD
                         : partial + (row * gridDim.z + blockIdx.z) * Partial;
  if (repair && __builtin_isfinite(dst[0])) return;
  sparse_row<F, T>(out, partial, query, cache, table, table_stride, columns,
                   requests, pages, mapping, indices, index_stride, topk, sgn,
                   positions, recent, sinks, heads, scale, blockIdx.x,
                   blockIdx.y, blockIdx.z, gridDim.z, threadIdx.x);
}
// A sparse row's selected slots are shared by every query head of a KV head.
// Keep the QK/PV accumulators and softmax in FP32, and stage one FP16 tile.
template <class F, class T>
__global__ __launch_bounds__(128) void sparse_wmma(
    float* out, float* partial, Activation query, PagedCache cache,
    const int32_t* table, int64_t table_stride, int columns, int requests,
    int pages, const int32_t* mapping, const int32_t* indices,
    int64_t index_stride, int topk, const uint32_t* sgn,
    const int32_t* positions, Shadow recent, Shadow sinks, int heads,
    float scale) {
  static_assert(F::D == 256 && F::Waves == 4 && F::Tile == 16);
  __shared__ Shared<F> s;
  // A wave can stage KV while another wave still reads the query vote.
  __shared__ int live_cols[16], query_unsafe[4], kv_unsafe[4];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32,
            col = lane & 15, half = lane / 16;
  const int q = blockIdx.x, kv = blockIdx.y, split = blockIdx.z,
            splits = gridDim.z, group = heads / cache.heads;
  const int req = mapping[q], width = (topk + splits - 1) / splits,
            begin = split * width, end = min(topk, begin + width);
  // The following scalar launch repairs marked rows. Inlining sparse_row
  // here keeps both algorithms' cache metadata live and spills SGPRs on 7.2.
  auto mark = [&](int h) {
    const int64_t row = int64_t(q) * heads + kv * group + h;
    float* dst = splits == 1 ? out + row * LD
                             : partial + (row * splits + split) * Partial;
    if (lane == 0) dst[0] = __builtin_nanf("");
  };
  auto fallback = [&] {
    for (int h = wave; h < group; h += 4) mark(h);
  };
  if (req < 0 || req >= requests || begin >= end) {
    fallback();
    return;
  }
  const int qpos = positions ? positions[q] : -1;
  bool q_safe = true;
  for (int h = wave; h < 16; h += 4) {
    float x[F::VL] = {};
    if (h < group) load<F, T>(x, query, q, kv * group + h, lane);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) q_safe &= fabsf(x[i]) * fabsf(scale) <= 2.f;
    signs<F>(x, sgn, lane);
    transform<F, F::compact>(x, lane);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) {
      q_safe &= fabsf(x[i]) * fabsf(scale) <= 2.f && fabsf(x[i]) <= 65504.f;
      s.data.input.q[h][lane + 32 * i] = (_Float16)x[i];
    }
  }
  const bool all_q_safe = __all(q_safe);
  if (lane == 0) query_unsafe[wave] = !all_q_safe;
  matrix_lds_barrier();
  if (query_unsafe[0] | query_unsafe[1] | query_unsafe[2] | query_unsafe[3]) {
    fallback();
    return;
  }

  F8 accum[4] = {}, running_mass = {}, running_max;
#pragma unroll
  for (int e = 0; e < 8; ++e) running_max[e] = -INFINITY;
  const int32_t* row = indices + int64_t(q) * index_stride;
  const int32_t* pages_for_request = table + int64_t(req) * table_stride;
  for (int base = begin; base < end; base += 16) {
    bool kv_safe = true;
    for (int t = wave; t < 16; t += 4) {
      const int pos = base + t < end ? row[base + t] : -1;
      int page = -1, offset = 0;
      if (pos >= 0) {
        const int logical_page = pos / cache.page_size;
        offset = pos - logical_page * cache.page_size;
        if (logical_page < columns) page = pages_for_request[logical_page];
      }
      const bool live = page >= 0 && page < pages;
      if (lane == 0) live_cols[t] = live;
      float k[F::VL] = {}, v[F::VL] = {};
      if (live) {
        const int slot_id = page * cache.page_size + offset;
        const T* exact = nullptr;
        if (qpos >= pos) {
          const int r = slot_id & (recent.entries - 1),
                    h = slot_id & (sinks.entries - 1);
          if (qpos - pos < 64 && recent.tags[r] == slot_id)
            exact = static_cast<const T*>(recent.data) +
                    (int64_t(r) * cache.heads + kv) * (2 * LD);
          else if (pos < 4 && sinks.tags[h] == slot_id)
            exact = static_cast<const T*>(sinks.data) +
                    (int64_t(h) * cache.heads + kv) * (2 * LD);
        }
        if (exact) {
#pragma unroll
          for (int i = 0; i < F::VL; ++i) {
            const int d = lane + 32 * i;
            k[i] = float(exact[d]);
            v[i] = float(exact[LD + d]);
          }
          signs<F>(k, sgn, lane);
          transform<F, F::compact>(k, lane);
          signs<F>(v, sgn, lane);
          transform<F, F::VWhole>(v, lane);
        } else {
          const uint8_t* slot = cache.at(page, kv, offset);
#pragma unroll
          for (int i = 0; i < F::VL; ++i) {
            const int d = lane + 32 * i;
            k[i] = kval<F>(slot, d);
            v[i] = vval<F>(slot, d);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < F::VL; ++i) {
        kv_safe &= fabsf(k[i]) <= 65504.f && fabsf(v[i]) <= 65504.f;
        s.data.input.k[t][lane + 32 * i] = (_Float16)k[i];
        s.data.input.v[lane + 32 * i][t] = (_Float16)v[i];
      }
    }
    const bool all_kv_safe = __all(kv_safe);
    if (lane == 0) kv_unsafe[wave] = !all_kv_safe;
    matrix_lds_barrier();
    if (kv_unsafe[0] | kv_unsafe[1] | kv_unsafe[2] | kv_unsafe[3]) {
      fallback();
      return;
    }
    F8 score = {};
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int dim = wave * 16 + 64 * d + half * 8;
      H8 a = *reinterpret_cast<const H8*>(&s.data.input.q[col][dim]);
      H8 b = *reinterpret_cast<const H8*>(&s.data.input.k[col][dim]);
      score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) s.partial[wave][half * 8 + e][col] = score[e];
    matrix_lds_barrier();
    if (wave == 0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = half * 8 + e;
        const bool live = h < group && live_cols[col];
        const float score_total =
            (score[e] + s.partial[1][h][col] + s.partial[2][h][col] +
             s.partial[3][h][col]) *
            scale;
        const float x = live ? score_total : -INFINITY;
        float m = x;
#pragma unroll
        for (int d = 8; d; d >>= 1) m = fmaxf(m, sx(m, d));
        const float nm = fmaxf(running_max[e], m);
        const float rs =
            running_mass[e] > 0 ? __expf(running_max[e] - nm) : 0.f;
        const float p = live ? __expf(x - nm) : 0.f;
        float sum_p = p;
#pragma unroll
        for (int d = 8; d; d >>= 1) sum_p += sx(sum_p, d);
        running_max[e] = nm;
        running_mass[e] = fmaf(running_mass[e], rs, sum_p);
        s.prob[h][col] = (_Float16)p;
        if (col == 0) {
          s.rs[h] = rs;
          s.mx[h] = nm;
          s.mass[h] = running_mass[e];
        }
      }
    }
    matrix_lds_barrier();
    const H8 p = *reinterpret_cast<const H8*>(&s.prob[col][half * 8]);
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int dim = wave * 64 + d * 16 + col;
      const H8 v = *reinterpret_cast<const H8*>(&s.data.input.v[dim][half * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) accum[d][e] *= s.rs[half * 8 + e];
      accum[d] =
          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p, v, accum[d]);
    }
    matrix_lds_barrier();
  }
#pragma unroll
  for (int d = 0; d < 4; ++d)
#pragma unroll
    for (int e = 0; e < 8; ++e)
      s.data.output[half * 8 + e][wave * 64 + d * 16 + col] = accum[d][e];
  matrix_lds_barrier();
  for (int h = wave; h < group; h += 4) {
    float v[F::VL];
#pragma unroll
    for (int i = 0; i < F::VL; ++i) v[i] = s.data.output[h][lane + 32 * i];
    transform<F, F::VWhole>(v, lane);
    signs<F>(v, sgn, lane);
    const float mass = s.mass[h];
    const float inv = mass > 0 ? 1.f / mass : 0.f;
    float largest = 0.f;
    bool finite = __builtin_isfinite(mass);
#pragma unroll
    for (int i = 0; i < F::VL; ++i) {
      finite &= __builtin_isfinite(v[i]);
      largest = fmaxf(largest, fabsf(v[i] * inv));
    }
    if (!__all(finite) || (mass > 0 && maximum(largest) < 1.e-4f)) {
      mark(h);
      continue;
    }
    const int64_t output_row = int64_t(q) * heads + kv * group + h;
    float* dst = splits == 1
                     ? out + output_row * LD
                     : partial + (output_row * splits + split) * Partial;
#pragma unroll
    for (int i = 0; i < F::VL; ++i)
      dst[lane + 32 * i] = v[i] * (splits == 1 ? inv : 1.f);
    if (splits > 1 && lane == 0) {
      dst[LD] = s.mx[h];
      dst[LD + 1] = mass;
    }
  }
}
}  // namespace

int octave_rdna4::geometry::launch_window_store(
    Activation key, Activation value, const int64_t* slots,
    const int32_t* positions, Shadow recent, Shadow sinks, int tokens,
    int heads, int dtype, hipStream_t stream) {
  if (tokens == 0) return hipSuccess;
  if (dtype == 0)
    window_store<__half><<<tokens, 32, 0, stream>>>(
        key, value, slots, positions, recent, sinks, heads);
  else if (dtype == 1)
    window_store<hip_bfloat16><<<tokens, 32, 0, stream>>>(
        key, value, slots, positions, recent, sinks, heads);
  else
    return hipErrorInvalidValue;
  return hipGetLastError();
}

int octave_rdna4::geometry::launch_decode_sparse(
    float* out, float* partial, Activation q, PagedCache cache,
    const int32_t* table, int64_t table_stride, int columns, int requests,
    int pages, const int32_t* mapping, const int32_t* indices,
    int64_t index_stride, int topk, const uint32_t* sgn,
    const int32_t* positions, Shadow recent, Shadow sinks, int tokens,
    int heads, int splits, float scale, int dtype, int format,
    hipStream_t stream) {
  if (tokens == 0) return hipSuccess;
  const int group = heads / cache.heads;
  const bool use_wmma = group >= 8 && group <= 16;
  int rc = dispatch(dtype, format, [&](auto t, auto f) {
    if (use_wmma)
      sparse_wmma<decltype(f), decltype(t)>
          <<<dim3(tokens, cache.heads, splits), 128, 0, stream>>>(
              out, partial, q, cache, table, table_stride, columns, requests,
              pages, mapping, indices, index_stride, topk, sgn, positions,
              recent, sinks, heads, scale);
    sparse<decltype(f), decltype(t)>
        <<<dim3(tokens, heads, splits), 32, 0, stream>>>(
            out, partial, q, cache, table, table_stride, columns, requests,
            pages, mapping, indices, index_stride, topk, sgn, positions, recent,
            sinks, heads, scale, use_wmma);
  });
  if (rc != hipSuccess || splits == 1) return rc;
  launch_merge(out, partial, tokens, heads, splits, stream);
  return hipGetLastError();
}
