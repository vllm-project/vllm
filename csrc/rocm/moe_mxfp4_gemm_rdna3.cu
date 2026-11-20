// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 fused MoE GEMM for RDNA3 (gfx1100). Fork of
// moe_q_gemm_rdna3.cu: expert routing (sorted_token_ids / expert_ids) + the
// MXFP4 scalar dequant+dot, in one kernel. No zero-point; E8M0 block scale is
// [E, K/32, N] uint8. Per-expert weights are [E, K/8, N] uint32 (E2M1).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_mxfp4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

#define MOE_MXFP4_BLOCK_KN 256
#define MOE_MXFP4_THREADS 256

namespace vllm {
namespace moe_mxfp4_rdna3 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;
using mxfp4_rdna3::dequant_mxfp4_8_f32;

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename T>
__device__ __forceinline__ T tzero();
template <>
__device__ __forceinline__ half tzero<half>() {
  return __float2half_rn(0.0f);
}
template <>
__device__ __forceinline__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

__device__ __forceinline__ float mxfp4_to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float mxfp4_to_f(bf16_t v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ void atomic_add_pk4(half* addr, half2 v01,
                                               half2 v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}
__device__ __forceinline__ void atomic_add_pk4(bf16_t* addr, bf162_t v01,
                                               bf162_t v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using elem = _Float16;
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using elem = __bf16;
  using v16 = v16bf16;
};

template <typename FROM, typename TO>
__device__ __forceinline__ TO wmoe_bitcast(FROM x) {
  static_assert(sizeof(FROM) == sizeof(TO), "size");
  TO r;
  __builtin_memcpy(&r, &x, sizeof(TO));
  return r;
}

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna3(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,
    const uint8_t* __restrict__ b_scales_e8m0,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int groups, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * MOE_MXFP4_BLOCK_KN * 4;
  const int offset_k = blockIdx.z * MOE_MXFP4_BLOCK_KN;
  const int end_k = min(offset_k + MOE_MXFP4_BLOCK_KN, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const uint8_t* expert_scales =
      b_scales_e8m0 + (int64_t)expert_id * expert_scales_stride;

  constexpr int LDS_PAD = 8;
  __shared__ T block_a[BLOCK_SIZE_M][MOE_MXFP4_BLOCK_KN + LDS_PAD];
  static_assert(MOE_MXFP4_BLOCK_KN == MOE_MXFP4_THREADS,
                "BLOCK_KN must equal THREADS");
  const int offset_m_base = token_block * BLOCK_SIZE_M;

  // E2M1 signed-magnitude LUT (exact fp32): one lookup replaces the per-nibble
  // bit decode; the E8M0 scale folds in once per group as a multiply. The
  // decode is compute-bound on gfx1100 (~2.8x over the arithmetic decode).
  __shared__ float w_lut[16];
  if (t < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    w_lut[t] = (t & 0x8 ? -1.f : 1.f) * mag[t & 0x7];
  }

  // Stage activations into LDS via the routing table.
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      int32_t token_id = sorted_token_ids[offset_m_base + m];
      int token_row = token_id / top_k;
      block_a[m][t] = (token_row < size_m)
                          ? a[(int64_t)token_row * size_k + offset_k + t]
                          : tzero<T>();
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;  // == 32 for MXFP4
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  // Per-group E8M0 scale as a pure bit construct: 2^(s8-127) is the fp32 whose
  // exponent field is s8 (IEEE bias 127), i.e. uint_as_float(s8 << 23).
  float gscale[4];
  auto refresh = [&](int g) {
    const uint8_t* sc = expert_scales + (int64_t)g * size_n;
  #pragma unroll
    for (int col = 0; col < 4; ++col)
      gscale[col] = __uint_as_float((uint32_t)sc[n + col] << 23);
  };
  refresh(group);

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int col = 0; col < 4; ++col) block_c[m][col] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh(group);
    }
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) b_w[j] = *(const int4*)(b_ptr + j * size_n);
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      uint32_t wcol[4];
      __builtin_memcpy(wcol, &b_w[j], sizeof(int4));
  #pragma unroll
      for (int col = 0; col < 4; ++col) {
        const uint32_t qa = wcol[col];
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          const T* ap = &block_a[m][a_off];
          float s = 0.0f;
  #pragma unroll
          for (int i = 0; i < 8; ++i)
            s += w_lut[(qa >> (4 * i)) & 0xFu] * mxfp4_to_f(ap[i]);
          block_c[m][col] += gscale[col] * s;
        }
      }
    }
    k += 32;
  }

  // Epilogue: topk weight + atomic-add (with optional output_topk reduction).
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;
    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }
    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4(out, r01, r23);
    } else {
      bf162_t r01, r23;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4(out, r01, r23);
    }
  }
}

template <typename T>
struct PairOfT;
template <>
struct PairOfT<half> {
  using type = half2;
};
template <>
struct PairOfT<bf16_t> {
  using type = bf162_t;
};
template <typename T>
using PairOf = typename PairOfT<T>::type;

// ---------------------------------------------------------------------------
// WMMA path for batches that give each expert tens of rows (BLOCK_SIZE_M 32 /
// 64): the scalar kernel above re-decodes the weights for every few tokens. A
// block of WM x WN waves owns a BM x BN tile, so each weight is decoded once
// per BM tokens. K advances one 32-wide E8M0 group per step: A and the decoded
// weights go through LDS, with the next group's global loads issued before the
// current group's WMMAs.
// ---------------------------------------------------------------------------

// E2M1 magnitude bytes of T for v_perm (index = 3-bit magnitude code).
template <typename T>
struct E2m1Perm;
template <>
struct E2m1Perm<half> {
  // 0, .5, 1, 1.5 | 2, 3, 4, 6 as fp16: low bytes are all zero.
  static constexpr uint32_t HI_LO = 0x3E3C3800u, HI_HI = 0x46444240u;
  static constexpr bool kLowBytes = false;
  static constexpr uint32_t LO_LO = 0u, LO_HI = 0u;
  static constexpr int kMantBits = 10;
};
template <>
struct E2m1Perm<bf16_t> {
  static constexpr uint32_t HI_LO = 0x3F3F3F00u, HI_HI = 0x40404040u;
  static constexpr bool kLowBytes = true;
  static constexpr uint32_t LO_LO = 0xC0800000u, LO_HI = 0xC0804000u;
  static constexpr int kMantBits = 7;
};

using u16x2 = uint16_t __attribute__((ext_vector_type(2)));

// Two adjacent E2M1 codes (low nibbles of t: code0 bits 0-3, code1 bits 4-7)
// -> two T values in one word, scaled by 2^(e8m0 - 127) as an exponent add
// that skips zero magnitudes (same rounding as dequant_mxfp4_8_lut).
template <typename T>
__device__ __forceinline__ uint32_t mxfp4_pair(uint32_t t, uint16_t bias) {
  using P = E2m1Perm<T>;
  const uint32_t m0 = t & 7u;
  const uint32_t m1 = (t >> 4) & 7u;
  const uint32_t x = m0 | (m1 << 16);
  uint32_t mag =
      __builtin_amdgcn_perm(P::HI_HI, P::HI_LO, (x << 8) | 0x000C000Cu);
  if constexpr (P::kLowBytes)
    mag |= __builtin_amdgcn_perm(P::LO_HI, P::LO_LO, x | 0x0C000C00u);
  u16x2 bv;
  bv.x = m0 ? bias : (uint16_t)0;
  bv.y = m1 ? bias : (uint16_t)0;
  const u16x2 r = __builtin_bit_cast(u16x2, mag) + bv;
  const uint32_t sign = ((t & 0x8u) << 12) | ((t & 0x80u) << 24);
  return __builtin_bit_cast(uint32_t, r) | sign;
}

template <typename T>
__device__ __forceinline__ uint2 pack4(float a, float b, float c, float d);
template <>
__device__ __forceinline__ uint2 pack4<half>(float a, float b, float c,
                                             float d) {
  uint2 u;
  u.x = wmoe_bitcast<half2, uint32_t>(__floats2half2_rn(a, b));
  u.y = wmoe_bitcast<half2, uint32_t>(__floats2half2_rn(c, d));
  return u;
}
template <>
__device__ __forceinline__ uint2 pack4<bf16_t>(float a, float b, float c,
                                               float d) {
  bf162_t p, q;
  p.x = __float2bfloat16(a);
  p.y = __float2bfloat16(b);
  q.x = __float2bfloat16(c);
  q.y = __float2bfloat16(d);
  uint2 u;
  u.x = wmoe_bitcast<bf162_t, uint32_t>(p);
  u.y = wmoe_bitcast<bf162_t, uint32_t>(q);
  return u;
}

template <typename T, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) moe_gemm_mxfp4_tiled_rdna3(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,
    const uint8_t* __restrict__ b_scales_e8m0,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk,
    const T* __restrict__ bias) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM / 16;
  constexpr int TN = BN / WN / 16;
  constexpr int KP = 32 + 8;  // padded LDS row of one group
  constexpr int CP = BN + 4;
  constexpr int P = NT / BN;   // threads per weight column
  constexpr int WPER = 4 / P;  // packed words (8 codes each) per thread
  static_assert(NT % BN == 0 && (P == 1 || P == 2 || P == 4));

  const int t = threadIdx.x;
  const int wave = t >> 5;
  const int lane = t & 31;
  const int lane_lo = lane & 15;
  const int wm = wave % WM;
  const int wn = wave / WM;

  // N tiles of one token block are adjacent in launch order so they share
  // its A rows in L2.
  const int m_base = blockIdx.y * BM;
  const int n_base = blockIdx.x * BN;
  if (m_base >= num_tokens_post_padded[0]) return;
  const int e = expert_ids[blockIdx.y];
  if (e < 0) return;

  const int groups = size_k / 32;
  const int num_valid = size_m * top_k;
  const uint32_t* w_e = b_q_weight + (int64_t)e * expert_weight_stride;
  const uint8_t* s_e = b_scales_e8m0 + (int64_t)e * expert_scales_stride;

  // One LDS stage (the next group waits in registers): half the LDS of a
  // double buffer, so more blocks stay resident to cover the load latency.
  constexpr int AB_BYTES = (BM + BN) * KP * 2;
  constexpr int C_BYTES = BM * CP * 4;
  __shared__ __attribute__((
      aligned(16))) char smem[AB_BYTES > C_BYTES ? AB_BYTES : C_BYTES];
  T* sa = reinterpret_cast<T*>(smem);
  T* sb = reinterpret_cast<T*>(smem + BM * KP * 2);
  __shared__ int32_t s_tok[BM];

  if (t < BM) s_tok[t] = sorted_token_ids[m_base + t];
  __syncthreads();

  // A: BM rows x four 16-byte chunks per group.
  constexpr int A_CHUNKS = BM * 4;
  constexpr int A_PER_T = (A_CHUNKS + NT - 1) / NT;
  const T* a_src[A_PER_T];
  int a_dst[A_PER_T];
  #pragma unroll
  for (int i = 0; i < A_PER_T; ++i) {
    const int idx = t + i * NT;
    const int row = idx >> 2;
    a_src[i] = nullptr;
    a_dst[i] = -1;
    if (idx < A_CHUNKS) {
      a_dst[i] = row * KP + (idx & 3) * 8;
      const int tok = s_tok[row];
      if (tok < num_valid)
        a_src[i] = a + (int64_t)(tok / top_k) * size_k + (idx & 3) * 8;
    }
  }

  // Weights: thread -> (column, part); a part covers WPER of the 4 words.
  const int bcol = t % BN;
  const int bpart = t / BN;
  const int n_glob = n_base + bcol;
  const bool bvalid = n_glob < size_n;

  uint4 ra[A_PER_T];
  uint32_t rw[WPER];
  uint8_t rs;

  auto gload = [&](int g) {
  #pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      ra[i] = make_uint4(0, 0, 0, 0);
      if (a_src[i]) ra[i] = *reinterpret_cast<const uint4*>(a_src[i] + g * 32);
    }
    if (bvalid) {
      const uint32_t* wp =
          w_e + (int64_t)(g * 4 + bpart * WPER) * size_n + n_glob;
  #pragma unroll
      for (int j = 0; j < WPER; ++j) rw[j] = wp[j * size_n];
      rs = s_e[(int64_t)g * size_n + n_glob];
    } else {
  #pragma unroll
      for (int j = 0; j < WPER; ++j) rw[j] = 0;
      rs = 127;
    }
  };
  auto lstore = [&]() {
  #pragma unroll
    for (int i = 0; i < A_PER_T; ++i)
      if (a_dst[i] >= 0) *reinterpret_cast<uint4*>(&sa[a_dst[i]]) = ra[i];
    const uint16_t sbias =
        (uint16_t)(((int32_t)rs - 127) << E2m1Perm<T>::kMantBits);
    uint4* dst = reinterpret_cast<uint4*>(&sb[bcol * KP + bpart * WPER * 8]);
  #pragma unroll
    for (int j = 0; j < WPER; ++j) {
      const uint32_t q = rw[j];
      dst[j] = make_uint4(mxfp4_pair<T>(q, sbias), mxfp4_pair<T>(q >> 8, sbias),
                          mxfp4_pair<T>(q >> 16, sbias),
                          mxfp4_pair<T>(q >> 24, sbias));
    }
  };

  v8fp32 acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j)
  #pragma unroll
      for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.0f;

  gload(0);
  lstore();
  __syncthreads();
  for (int g = 0; g < groups; ++g) {
    if (g + 1 < groups) gload(g + 1);
  #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      V16 af[TM], bf[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const V16*>(
            &sa[((wm * TM + i) * 16 + lane_lo) * KP + ks * 16]);
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const V16*>(
            &sb[((wn * TN + j) * 16 + lane_lo) * KP + ks * 16]);
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = wmma_mma(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
    if (g + 1 < groups) lstore();
    __syncthreads();
  }

  // C fragment: lane holds column lane_lo, rows 2 * r + lane_hi. Restage
  // through LDS so each thread writes four adjacent columns.
  const int lane_hi = lane >> 4;
  float* sc = reinterpret_cast<float*>(smem);
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = (wn * TN + j) * 16 + lane_lo;
  #pragma unroll
      for (int r = 0; r < 8; ++r)
        sc[((wm * TM + i) * 16 + 2 * r + lane_hi) * CP + col] = acc[i][j][r];
    }
  __syncthreads();
  for (int idx = t; idx < BM * BN / 4; idx += NT) {
    const int row = idx / (BN / 4);
    const int col = (idx % (BN / 4)) * 4;
    const int tok = s_tok[row];
    if (tok >= num_valid || n_base + col >= size_n) continue;
    const float* v = &sc[row * CP + col];
    float b[4] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
      const T* bp = bias + (int64_t)e * size_n + n_base + col;
  #pragma unroll
      for (int i = 0; i < 4; ++i) b[i] = (float)bp[i];
    }
    const float tw =
        (mul_topk_weight && topk_weights != nullptr) ? topk_weights[tok] : 1.f;
    const uint2 u = pack4<T>((v[0] + b[0]) * tw, (v[1] + b[1]) * tw,
                             (v[2] + b[2]) * tw, (v[3] + b[3]) * tw);
    if (output_topk > 0) {
      T* dst = c + (int64_t)(tok / output_topk) * size_n + n_base + col;
      atomic_add_pk4(dst, wmoe_bitcast<uint32_t, PairOf<T>>(u.x),
                     wmoe_bitcast<uint32_t, PairOf<T>>(u.y));
    } else {
      *reinterpret_cast<uint2*>(c + (int64_t)tok * size_n + n_base + col) = u;
    }
  }
}

#else  // non-RDNA3: empty stub for symbol parity
template <typename T, int BM, int BN, int WM, int WN>
__global__ void moe_gemm_mxfp4_tiled_rdna3(
    const T*, T*, const uint32_t*, const uint8_t*, const float*, const int32_t*,
    const int32_t*, const int32_t*, const int, const int, const int, const int,
    const int, const int, const bool, const int, const T*) {}
template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna3(
    const T*, T*, const uint32_t*, const uint8_t*, const float*, const int32_t*,
    const int32_t*, const int32_t*, const int, const int, const int, const int,
    const int, const int, const int, const bool, const int) {}
#endif

// Tiled launcher: one block per (BN columns, BM-token block); N tiles of a
// token block are adjacent in launch order.
template <typename T, int BM, int BN, int WM, int WN>
void launch_moe_gemm_mxfp4_tiled(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales_e8m0,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int top_k,
    int expert_weight_stride, int expert_scales_stride, bool mul_topk_weight,
    int output_topk, const T* bias, cudaStream_t stream) {
  dim3 grid((size_n + BN - 1) / BN, num_token_blocks, 1);
  moe_gemm_mxfp4_tiled_rdna3<T, BM, BN, WM, WN>
      <<<grid, WM * WN * 32, 0, stream>>>(
          a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, size_m, size_n, size_k, top_k,
          expert_weight_stride, expert_scales_stride, mul_topk_weight,
          output_topk, bias);
}

template <typename T, int BLOCK_SIZE_M>
void launch_moe_gemm_mxfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales_e8m0,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int groups,
    int top_k, int expert_weight_stride, int expert_scales_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  dim3 block(MOE_MXFP4_THREADS);
  dim3 grid(num_token_blocks,
            (size_n + MOE_MXFP4_BLOCK_KN * 4 - 1) / (MOE_MXFP4_BLOCK_KN * 4),
            (size_k + MOE_MXFP4_BLOCK_KN - 1) / MOE_MXFP4_BLOCK_KN);
  moe_gemm_mxfp4_kernel_rdna3<T, BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, mul_topk_weight, output_topk);
}

template <typename T>
void dispatch_moe_gemm_mxfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales_e8m0,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int groups,
    int top_k, int block_size_m, int expert_weight_stride,
    int expert_scales_stride, bool mul_topk_weight, int output_topk,
    const T* bias, cudaStream_t stream) {
  TORCH_CHECK(bias == nullptr || block_size_m >= 32,
              "moe_mxfp4_gemm_rdna3: bias needs block_size_m >= 32");
  auto L = [&](auto bsm) {
    launch_moe_gemm_mxfp4<T, decltype(bsm)::value>(
        a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
        expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
        size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
        mul_topk_weight, output_topk, stream);
  };
  switch (block_size_m) {
    case 1:
      L(std::integral_constant<int, 1>{});
      break;
    case 2:
      L(std::integral_constant<int, 2>{});
      break;
    case 4:
      L(std::integral_constant<int, 4>{});
      break;
    case 8:
      L(std::integral_constant<int, 8>{});
      break;
    case 32:
      launch_moe_gemm_mxfp4_tiled<T, 32, 64, 2, 2>(
          a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, bias, stream);
      break;
    case 64:
      launch_moe_gemm_mxfp4_tiled<T, 64, 64, 2, 2>(
          a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, top_k, expert_weight_stride, expert_scales_stride,
          mul_topk_weight, output_topk, bias, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_mxfp4_gemm_rdna3: block_size_m must be 1/2/4/8/32/64");
  }
}

}  // namespace moe_mxfp4_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
//   a                      [M, K] or [M*top_k, K]  half/bf16
//   c                      [M*top_k, N] or reduced  same dtype (pre-zeroed!)
//   b_q_weight             [E, K/8, N]              uint32 (E2M1)
//   b_scales_e8m0          [E, K/32, N]             uint8  (E8M0)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]             int32
//   num_tokens_post_padded [1]                      int32
// ---------------------------------------------------------------------------
void moe_mxfp4_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales_e8m0,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk,
                          const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda() && b_q_weight.is_cuda(),
              "tensors must be CUDA/HIP");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be [E, K/8, N]");
  TORCH_CHECK(b_scales_e8m0.dim() == 3, "b_scales_e8m0 must be [E, K/32, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(b_scales_e8m0.scalar_type() == torch::kUInt8,
              "b_scales_e8m0 must be uint8 (E8M0)");
  for (const auto& t : {c, b_q_weight, b_scales_e8m0, topk_weights,
                        sorted_token_ids, expert_ids, num_tokens_post_padded}) {
    TORCH_CHECK(t.numel() == 0 || t.device() == a.device(),
                "all tensors must be on the same device");
  }
  TORCH_CHECK(a.is_contiguous() && c.is_contiguous() &&
                  b_q_weight.is_contiguous() && b_scales_e8m0.is_contiguous(),
              "a, c and the weights must be contiguous");
  TORCH_CHECK(b_q_weight.size(2) % 4 == 0,
              "N must be a multiple of 4 (vector loads, packed atomics)");
  TORCH_CHECK(block_size_m > 0, "block_size_m must be positive");
  if (bias) {
    TORCH_CHECK(
        bias->scalar_type() == a.scalar_type() && bias->is_contiguous() &&
            bias->dim() == 2 && bias->size(0) == b_q_weight.size(0) &&
            bias->size(1) == b_q_weight.size(2) && bias->device() == a.device(),
        "bias must be [E, N], contiguous, same dtype and device as a");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales_e8m0.size(1);  // K/32
  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride =
      (int)(b_scales_e8m0.size(1) * b_scales_e8m0.size(2));
  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  TORCH_CHECK(expert_ids.size(0) >= num_token_blocks,
              "expert_ids must have one entry per token block");

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  using bf16_t = vllm::moe_mxfp4_rdna3::bf16_t;
  const uint32_t* bq = (const uint32_t*)b_q_weight.data_ptr<int32_t>();
  const uint8_t* bs = (const uint8_t*)b_scales_e8m0.data_ptr();
  const int32_t* sti = sorted_token_ids.data_ptr<int32_t>();
  const int32_t* eid = expert_ids.data_ptr<int32_t>();
  const int32_t* ntp = num_tokens_post_padded.data_ptr<int32_t>();

  if (a.scalar_type() == torch::kHalf) {
    vllm::moe_mxfp4_rdna3::dispatch_moe_gemm_mxfp4<half>(
        (const half*)a.data_ptr(), (half*)c.data_ptr(), bq, bs, topk_w_ptr, sti,
        eid, ntp, num_token_blocks, size_m, size_n, size_k, groups, (int)top_k,
        (int)block_size_m, expert_weight_stride, expert_scales_stride,
        mul_topk_weight, (int)output_topk,
        bias ? (const half*)bias->data_ptr() : nullptr, stream);
  } else {
    vllm::moe_mxfp4_rdna3::dispatch_moe_gemm_mxfp4<bf16_t>(
        (const bf16_t*)a.data_ptr(), (bf16_t*)c.data_ptr(), bq, bs, topk_w_ptr,
        sti, eid, ntp, num_token_blocks, size_m, size_n, size_k, groups,
        (int)top_k, (int)block_size_m, expert_weight_stride,
        expert_scales_stride, mul_topk_weight, (int)output_topk,
        bias ? (const bf16_t*)bias->data_ptr() : nullptr, stream);
  }
}
