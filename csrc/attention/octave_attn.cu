// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Octave KV cache kernels, head_size 256 with 64 RoPE dims. Portable across
// the AMD targets vLLM builds (octave_arch.cuh); the WMMA decode tier for
// gfx11 lives in octave_decode_wmma_rdna3.cu.
// Format: vllm/v1/attention/ops/rocm_octave.py (the PyTorch reference there
// is the source of truth for the byte layout).
//
//   octave_cache_store   quantize + pack K/V into slots (one wave per slot).
//   octave_decode        split-KV attention per query group; K dot products
//                        with int8 dot4 on the codebook values of the packed
//                        codes, the query quantized to int8 once per block.
//                        Consecutive query tokens of one request (MTP verify)
//                        share the KV read.
//   octave_rotate        the store-time rotations, for the prefill path.
//
// Everything is in the rotated space: K and V are sign-flipped and
// Hadamard-rotated at store time; the query is rotated like K and the
// attention output is rotated back in the reduce kernel.

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  #include "octave_arch.cuh"
  #include "octave_format.cuh"

namespace octave {

constexpr int DPL = D / 32;  // dims per lane when a wave owns one slot

template <typename T>
__device__ __forceinline__ float to_f(T x) {
  return (float)x;
}
template <>
__device__ __forceinline__ float to_f<half>(half x) {
  return __half2float(x);
}
template <typename T>
__device__ __forceinline__ T from_f(float x) {
  return (T)x;
}
template <>
__device__ __forceinline__ half from_f<half>(float x) {
  return __float2half(x);
}

__device__ __forceinline__ float half_bits_to_f(uint32_t bits16) {
  return __half2float(__ushort_as_half((unsigned short)bits16));
}
__device__ __forceinline__ uint16_t f_to_half_bits(float x) {
  return __half_as_ushort(__float2half(x));
}

template <int MASK>
__device__ __forceinline__ float xor_sum(float v) {
  #pragma unroll
  for (int m = 1; m <= MASK; m <<= 1) v += __shfl_xor(v, m);
  return v;
}
template <int MASK>
__device__ __forceinline__ float xor_max(float v) {
  #pragma unroll
  for (int m = 1; m <= MASK; m <<= 1) v = fmaxf(v, __shfl_xor(v, m));
  return v;
}
template <int MASK>
__device__ __forceinline__ uint32_t xor_or(uint32_t v) {
  #pragma unroll
  for (int m = 1; m <= MASK; m <<= 1) v |= (uint32_t)__shfl_xor((int)v, m);
  return v;
}

// Walsh-Hadamard butterflies over DPL consecutive dims per lane, then across
// lanes with xor distance 1..LANE_MASK. Unnormalized.
template <int LANE_MASK>
__device__ __forceinline__ void fwht_wave(float (&x)[DPL], int lane) {
  #pragma unroll
  for (int h = 1; h < DPL; h <<= 1) {
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      if ((i & h) == 0) {
        float a = x[i], b = x[i + h];
        x[i] = a + b;
        x[i + h] = a - b;
      }
    }
  }
  #pragma unroll
  for (int m = 1; m <= LANE_MASK; m <<= 1) {
    const bool hi = lane & m;
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      float o = __shfl_xor(x[i], m);
      x[i] = hi ? o - x[i] : x[i] + o;
    }
  }
}

__device__ __forceinline__ float sign_of(const int* bits, int i) {
  return ((bits[i >> 5] >> (i & 31)) & 1) ? -1.0f : 1.0f;
}

// Rotate a lane's 8 dims like K (64-dim blocks) or like V (one 256-dim
// block).
__device__ __forceinline__ void rotate_k(float (&x)[DPL], int lane,
                                         const int* signs) {
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= sign_of(signs, lane * DPL + i);
  fwht_wave<4>(x, lane);
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= 0.125f;
}

__device__ __forceinline__ void rotate_full(float (&x)[DPL], int lane,
                                            const int* signs) {
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= sign_of(signs, lane * DPL + i);
  fwht_wave<16>(x, lane);
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= 0.0625f;
}

// Midpoints of the int8 codebooks (rocm_octave.LUT), in codebook units.
__device__ constexpr float kThr3[7] = {-103.f, -62.f, -29.5f, 0.f,
                                       29.5f,  62.f,  103.f};
__device__ constexpr float kThr4[15] = {-110.5f, -84.5f, -66.f, -50.5f, -36.5f,
                                        -24.f,   -12.f,  0.f,   12.f,   24.f,
                                        36.5f,   50.5f,  66.f,  84.5f,  110.5f};

// Nearest codebook entry on the group RMS (4- or 3-bit per lane), codes in
// [0, 2^bits). Returns the stored scale, times LUT_ONE: the one that makes
// x . x_hat = |x|^2, so quantization does not shrink scores or outputs.
template <int MASK>
__device__ __forceinline__ float quantize_group(const float (&x)[DPL],
                                                int (&codes)[DPL],
                                                float group_dims, bool four) {
  float ss = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) ss += x[i] * x[i];
  ss = xor_sum<MASK>(ss);
  const float to_lut =
      (four ? 46.0f : 59.0f) / fmaxf(sqrtf(ss / group_dims), 1e-12f);
  float num = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    const float y = x[i] * to_lut;
    int c = 0;
  #pragma unroll
    for (int k = 0; k < 15; ++k)
      c += (four ? y > kThr4[k] : (k < 7 && y > kThr3[k])) ? 1 : 0;
    codes[i] = c;
    num += x[i] * (float)(int8_t)(four ? lut4((uint32_t)c) : lut3((uint32_t)c));
  }
  num = xor_sum<MASK>(num);
  return num > 0.f ? ss / num * LUT_ONE : 0.f;
}

// 4-bit: one word per 8 dims (lane-local). 3-bit: this lane's contribution to
// the 2-bit plane word (16 dims, 2 lanes) and the 1-bit plane word (32 dims,
// 4 lanes); callers OR across lanes.
__device__ __forceinline__ uint32_t pack4(const int (&c)[DPL]) {
  uint32_t w = 0;
  #pragma unroll
  for (int i = 0; i < DPL; ++i)
    w |= (uint32_t)c[i] << (8 * (i & 3) + 4 * (i >> 2));
  return w;
}
__device__ __forceinline__ void pack3(const int (&c)[DPL], int idx,
                                      uint32_t& lo, uint32_t& hi) {
  lo = 0;
  hi = 0;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    const int e16 = (idx & 1) * DPL + i;  // dim within the 16-dim lo word
    const int e32 = (idx & 3) * DPL + i;  // dim within the 32-dim hi word
    lo |= (uint32_t)(c[i] & 3) << (8 * (e16 & 3) + 2 * (e16 >> 2));
    hi |= (uint32_t)(c[i] >> 2) << (8 * (e32 & 3) + (e32 >> 2));
  }
}

// ---------------------------------------------------------------------------
// Store: grid (num_tokens, num_kv_heads), block 32.
// ---------------------------------------------------------------------------
template <int VB, bool KC, typename KT>
__global__ void __launch_bounds__(32)
    store_kernel(const KT* __restrict__ key, const KT* __restrict__ value,
                 uint8_t* __restrict__ cache,
                 const int64_t* __restrict__ slot_mapping,
                 const int* __restrict__ k_signs,
                 const int* __restrict__ v_signs, int block_size, int64_t sk0,
                 int64_t sk1, int64_t sv0, int64_t sv1, int64_t scb,
                 int64_t sch, int64_t sct) {
  using F = Format<VB, KC>;
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  uint8_t* dst =
      cache + (slot / block_size) * scb + h * sch + (slot % block_size) * sct;

  float k[DPL], v[DPL];
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    k[i] = to_f<KT>(key[t * sk0 + h * sk1 + lane * DPL + i]);
    v[i] = to_f<KT>(value[t * sv0 + h * sv1 + lane * DPL + i]);
  }

  // K, block layout: rotate, then quantize per 64-dim block (8 lanes): the
  // RoPE block at 4 bits, the NoPE blocks at 3. Compact layout: rotate over
  // all dims like V, 3 bits, one scale.
  const bool rope_lane = !KC && lane < R / DPL;
  int kc[DPL];
  float k_scale;
  if constexpr (KC) {
    rotate_full(k, lane, k_signs);
    k_scale = quantize_group<16>(k, kc, (float)D, false);
  } else {
    rotate_k(k, lane, k_signs);
    k_scale = quantize_group<4>(k, kc, 64.f, rope_lane);
  }
  const int nl = lane - F::KR / DPL;  // 3-bit lane index (valid when >= 0)
  {
    uint32_t lo, hi;
    pack3(kc, nl & 3, lo, hi);
    lo = xor_or<1>(lo);
    hi = xor_or<2>(hi);
    if (rope_lane) {
      *reinterpret_cast<uint32_t*>(dst + 4 * lane) = pack4(kc);
    } else {
      if ((nl & 1) == 0)
        *reinterpret_cast<uint32_t*>(dst + F::OFF_KN + 4 * (nl >> 1)) = lo;
      if ((nl & 3) == 0)
        *reinterpret_cast<uint32_t*>(dst + F::OFF_KN + F::KN / 4 +
                                     4 * (nl >> 2)) = hi;
    }
  }

  // V: rotate all 256 dims, one scale.
  rotate_full(v, lane, v_signs);
  int vc[DPL];
  const float v_scale = quantize_group<16>(v, vc, (float)D, VB == 4);
  if constexpr (VB == 4) {
    *reinterpret_cast<uint32_t*>(dst + F::OFF_V + 4 * lane) = pack4(vc);
  } else {
    uint32_t lo, hi;
    pack3(vc, lane, lo, hi);
    lo = xor_or<1>(lo);
    hi = xor_or<2>(hi);
    if ((lane & 1) == 0)
      *reinterpret_cast<uint32_t*>(dst + F::OFF_V + 4 * (lane >> 1)) = lo;
    if ((lane & 3) == 0)
      *reinterpret_cast<uint32_t*>(dst + F::OFF_V + D / 4 + 4 * (lane >> 2)) =
          hi;
  }

  uint16_t* sc = reinterpret_cast<uint16_t*>(dst + F::OFF_SC);
  if (KC ? lane == 0 : (lane & 7) == 0) sc[lane >> 3] = f_to_half_bits(k_scale);
  if (lane == 0) sc[F::NUM_KSC] = f_to_half_bits(v_scale);
}

// ---------------------------------------------------------------------------
// Decode stage 1: grid (num_q, num_kv_heads, num_splits), block 128.
// A block serves up to QG consecutive query tokens of one request (only the
// first token of each group of a run launches work) and all G query heads of
// its KV head: ROWS = QG * G attention rows share every K/V load.
// ---------------------------------------------------------------------------
constexpr int WAVES = 4;
constexpr int CHUNK = 32 * WAVES;

template <int VB, bool KC, int G, int QG, typename QT>
__global__ void __launch_bounds__(128)
    decode_kernel(const QT* __restrict__ Q, const uint8_t* __restrict__ cache,
                  const int* __restrict__ block_table,
                  const int* __restrict__ q_to_req,
                  const int* __restrict__ q_to_klen, float* __restrict__ mid_o,
                  const int* __restrict__ k_signs, float sm_scale_log2,
                  int num_q, int block_size, int max_blocks, int num_reqs,
                  int num_phys_blocks, int num_splits, int64_t sq0, int64_t sq1,
                  int64_t scb, int64_t sch, int64_t sct, int64_t smo,
                  int64_t smh, int64_t sms) {
  using F = Format<VB, KC>;
  constexpr int ROWS = QG * G;
  constexpr int QW = D / 4;  // int8 query words per row

  __shared__ int q8[ROWS][QW];
  __shared__ float q_bs[ROWS][NKB];  // query scale per K block
  __shared__ float S[ROWS][CHUNK];
  __shared__ float m_s[ROWS], l_s[ROWS], alpha_s[ROWS];
  __shared__ const uint8_t* tok_ptr[CHUNK];
  __shared__ float tok_vs[CHUNK];

  const int qi = blockIdx.x;
  const int kvh = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;

  int req = q_to_req[qi];
  const bool req_ok = req >= 0 && req < num_reqs;
  int nq = 1;
  if constexpr (QG > 1) {
    if (req_ok) {
      int s = qi;
      while (s > 0 && qi - s < 256 && q_to_req[s - 1] == req) --s;
      if ((qi - s) % QG != 0) return;  // served by the group leader
      while (nq < QG && qi + nq < num_q && q_to_req[qi + nq] == req) ++nq;
    }
  }
  const int max_kv = max_blocks * block_size;
  int klen[QG];
  int kv_max = 0;
  #pragma unroll
  for (int j = 0; j < QG; ++j) {
    int kl = (req_ok && j < nq) ? q_to_klen[qi + j] : 0;
    kl = min(max(kl, 0), max_kv);
    klen[j] = kl;
    kv_max = max(kv_max, kl);
  }
  if (!req_ok) req = 0;

  const int tps = (kv_max + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_max);

  auto write_empty = [&]() {
    for (int r = wave; r < ROWS; r += WAVES) {
      const int j = r / G, g = r % G;
      if (j >= nq) continue;
      float* op = mid_o + (qi + j) * smo + (kvh * G + g) * smh + si * sms;
      for (int d = lane; d < D; d += 32) op[d] = 0.f;
      if (lane == 0) {
        op[D] = -INFINITY;
        op[D + 1] = 0.f;
      }
    }
  };
  if (start >= end) {
    write_empty();
    return;
  }

  // Query: rotate like K, quantize to int8 with one scale per 64-dim block.
  for (int r = wave; r < ROWS; r += WAVES) {
    const int j = r / G, g = r % G;
    float x[DPL];
    if (j < nq) {
      const QT* qp = Q + (qi + j) * sq0 + (kvh * G + g) * sq1 + lane * DPL;
  #pragma unroll
      for (int i = 0; i < DPL; ++i) x[i] = to_f<QT>(qp[i]);
    } else {
  #pragma unroll
      for (int i = 0; i < DPL; ++i) x[i] = 0.f;
    }
    if constexpr (KC)
      rotate_full(x, lane, k_signs);
    else
      rotate_k(x, lane, k_signs);
    float amax = 0.f;
  #pragma unroll
    for (int i = 0; i < DPL; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float s = fmaxf(xor_max<4>(amax), 1e-20f) / 127.0f;
    uint32_t w[2] = {0, 0};
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      int q = __float2int_rn(x[i] / s);
      q = min(max(q, -127), 127);
      w[i >> 2] |= (uint32_t)(q & 0xFF) << (8 * (i & 3));
    }
    q8[r][2 * lane] = (int)w[0];
    q8[r][2 * lane + 1] = (int)w[1];
    if ((lane & 7) == 0) q_bs[r][lane >> 3] = s;
  }
  __syncthreads();

  if (tid < ROWS) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.f;
  }
  float o0[ROWS], o1[ROWS];  // this lane's two V dims (rotated space)
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    o0[r] = 0.f;
    o1[r] = 0.f;
  }
  const int my_d = wave * 64 + 2 * lane;  // first of the two V dims owned

  for (int base = start; base < end; base += CHUNK) {
    // ---- scores: lane = token ----
    const int tok = base + wave * 32 + lane;
    const bool valid = tok < end;
    if (valid) {
      const int lb = tok / block_size;
      int pb = block_table[req * max_blocks + lb];
      if (pb < 0 || pb >= num_phys_blocks) pb = 0;
      const uint8_t* p =
          cache + pb * scb + kvh * sch + (int64_t)(tok - lb * block_size) * sct;
      tok_ptr[wave * 32 + lane] = p;

      constexpr int KW = F::OFF_V / 4;  // K code words
      uint32_t kw[KW];
      const uint32_t* p4 = reinterpret_cast<const uint32_t*>(p);
  #pragma unroll
      for (int w = 0; w < KW; ++w) kw[w] = p4[w];
      const uint32_t* kn = kw + F::OFF_KN / 4;
      const uint32_t* ps = reinterpret_cast<const uint32_t*>(p + F::OFF_SC);
      float ks[NKB];
      if constexpr (KC) {
        const uint32_t u = ps[0];
  #pragma unroll
        for (int b = 0; b < NKB; ++b) ks[b] = half_bits_to_f(u & 0xFFFF);
        tok_vs[wave * 32 + lane] = half_bits_to_f(u >> 16) * (1.0f / LUT_ONE);
      } else {
        const uint32_t u0 = ps[0], u1 = ps[1];
        ks[0] = half_bits_to_f(u0 & 0xFFFF);
        ks[1] = half_bits_to_f(u0 >> 16);
        ks[2] = half_bits_to_f(u1 & 0xFFFF);
        ks[3] = half_bits_to_f(u1 >> 16);
        tok_vs[wave * 32 + lane] =
            half_bits_to_f(ps[2] & 0xFFFF) * (1.0f / LUT_ONE);
      }

  #pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int* qr = q8[r];
        float sk = 0.f;
        if constexpr (F::KR > 0) {
          int dr = 0;
  #pragma unroll
          for (int u = 0; u < F::KR / 8; ++u) {
            const uint32_t w = kw[u];
            dr = dot4_i8(qr[2 * u], (int)lut4(w & 0x0F0F0F0Fu), dr);
            dr = dot4_i8(qr[2 * u + 1], (int)lut4((w >> 4) & 0x0F0F0F0Fu), dr);
          }
          sk = q_bs[r][0] * ks[0] * (float)dr;
        }
  #pragma unroll
        for (int b = 0; b < F::KN / 64; ++b) {
          constexpr int B0 = F::KR / 64;  // first 3-bit block
          int dn = 0;
  #pragma unroll
          for (int v = 0; v < 4; ++v) {  // 2-bit plane word: 16 dims
            const uint32_t lo = kn[4 * b + v];
            const uint32_t hi = kn[F::KN / 16 + 2 * b + (v >> 1)];
  #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const uint32_t c =
                  ((lo >> (2 * i)) & 0x03030303u) |
                  (((hi >> (4 * (v & 1) + i)) & 0x01010101u) << 2);
              dn =
                  dot4_i8(qr[F::KR / 4 + 16 * b + 4 * v + i], (int)lut3(c), dn);
            }
          }
          sk += q_bs[r][B0 + b] * ks[B0 + b] * (float)dn;
        }
        const float sc = sk * (sm_scale_log2 / LUT_ONE);
        S[r][wave * 32 + lane] = tok < klen[r / G] ? sc : -INFINITY;
      }
    } else {
  #pragma unroll
      for (int r = 0; r < ROWS; ++r) S[r][wave * 32 + lane] = -INFINITY;
    }
    __syncthreads();

    // ---- online softmax: each wave owns rows r = wave, wave + WAVES, ... ----
    for (int r = wave; r < ROWS; r += WAVES) {
      float x[CHUNK / 32];
      float mx = -INFINITY;
  #pragma unroll
      for (int c = 0; c < CHUNK / 32; ++c) {
        x[c] = S[r][32 * c + lane];
        mx = fmaxf(mx, x[c]);
      }
      mx = xor_max<16>(mx);
      const float m_old = m_s[r];
      const float m_new = fmaxf(m_old, mx);
      float ps = 0.f;
  #pragma unroll
      for (int c = 0; c < CHUNK / 32; ++c) {
        const float p = x[c] == -INFINITY ? 0.f : exp2f(x[c] - m_new);
        S[r][32 * c + lane] = p;
        ps += p;
      }
      ps = xor_sum<16>(ps);
      if (lane == 0) {
        const float a = m_new == -INFINITY ? 1.f : exp2f(m_old - m_new);
        alpha_s[r] = a;
        l_s[r] = l_s[r] * a + ps;
        m_s[r] = m_new;
      }
    }
    __syncthreads();
  #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float a = alpha_s[r];
      o0[r] *= a;
      o1[r] *= a;
    }

    // ---- P @ V: wave owns 64 dims, lane owns 2 ----
    const int n_valid = min(CHUNK, end - base);
    for (int t = 0; t < n_valid; ++t) {
      const uint8_t* p = tok_ptr[t];
      uint32_t c;  // codes of dims my_d, my_d + 1 in bytes 0, 1
      if constexpr (VB == 4) {
        const uint32_t w =
            *reinterpret_cast<const uint32_t*>(p + F::OFF_V + 4 * (my_d >> 3));
        const int e = my_d & 7;  // even
        c = ((w >> (8 * (e & 3) + 4 * (e >> 2))) & 15) |
            (((w >> (8 * ((e + 1) & 3) + 4 * ((e + 1) >> 2))) & 15) << 8);
      } else {
        const uint32_t lo =
            *reinterpret_cast<const uint32_t*>(p + F::OFF_V + 4 * (my_d >> 4));
        const uint32_t hi = *reinterpret_cast<const uint32_t*>(
            p + F::OFF_V + D / 4 + 4 * (my_d >> 5));
        const int e16 = my_d & 15, e32 = my_d & 31;
        const int f16 = e16 + 1, f32 = e32 + 1;
        c = (((lo >> (8 * (e16 & 3) + 2 * (e16 >> 2))) & 3) |
             (((hi >> (8 * (e32 & 3) + (e32 >> 2))) & 1) << 2)) |
            ((((lo >> (8 * (f16 & 3) + 2 * (f16 >> 2))) & 3) |
              (((hi >> (8 * (f32 & 3) + (f32 >> 2))) & 1) << 2))
             << 8);
      }
      const uint32_t val = lut<VB>(c);
      const float vs = tok_vs[t];
      const float c0 = (float)(int8_t)(val & 0xFF) * vs;
      const float c1 = (float)(int8_t)((val >> 8) & 0xFF) * vs;
  #pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float pr = S[r][t];
        o0[r] += pr * c0;
        o1[r] += pr * c1;
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int j = r / G, g = r % G;
    if (j >= nq) continue;
    float* op = mid_o + (qi + j) * smo + (kvh * G + g) * smh + si * sms;
    op[my_d] = o0[r];
    op[my_d + 1] = o1[r];
    if (tid == 0) {
      op[D] = m_s[r];
      op[D + 1] = l_s[r];
    }
  }
}

// ---------------------------------------------------------------------------
// Decode stage 2: combine splits, rotate V back. grid (num_q, num_q_heads),
// block 256. Up to 1024 splits: their weights at once, then 8 waves take the
// splits round-robin (lane = 8 dims) and skip the ones that weigh zero.
// ---------------------------------------------------------------------------
template <typename OutT>
__global__ void __launch_bounds__(256)
    reduce_kernel(const float* __restrict__ mid_o, OutT* __restrict__ out,
                  const int* __restrict__ v_signs, int num_splits, int64_t smo,
                  int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  __shared__ float sw[1024], swm[8], swl[8];
  __shared__ float so[8][D];
  const int qi = blockIdx.x, hi = blockIdx.y;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const float* base = mid_o + qi * smo + hi * smh;

  float ms[4], ls[4];
  float m = -INFINITY;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int s = tid + 256 * k;
    ms[k] = s < num_splits ? base[s * sms + D] : -INFINITY;
    ls[k] = s < num_splits ? base[s * sms + D + 1] : 0.0f;
    m = fmaxf(m, ms[k]);
  }
  m = xor_max<16>(m);
  if (lane == 0) swm[w] = m;
  __syncthreads();
  float mg = swm[0];
  #pragma unroll
  for (int i = 1; i < 8; ++i) mg = fmaxf(mg, swm[i]);
  float l = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float a = ms[k] == -INFINITY ? 0.0f : exp2f(ms[k] - mg);
    if (tid + 256 * k < 1024) sw[tid + 256 * k] = a;
    l += ls[k] * a;
  }
  l = xor_sum<16>(l);
  if (lane == 0) swl[w] = l;
  __syncthreads();

  float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int s = w; s < num_splits; s += 8) {
    const float as = sw[s];
    if (as == 0.0f) continue;
    const float4* v = (const float4*)(base + s * sms + lane * 8);
    const float4 x = v[0], y = v[1];
    o[0] += x.x * as;
    o[1] += x.y * as;
    o[2] += x.z * as;
    o[3] += x.w * as;
    o[4] += y.x * as;
    o[5] += y.y * as;
    o[6] += y.z * as;
    o[7] += y.w * as;
  }
  #pragma unroll
  for (int d = 0; d < 8; ++d) so[w][lane * 8 + d] = o[d];
  __syncthreads();
  float lg = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) lg += swl[i];
  float acc = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc += so[i][tid];
  __syncthreads();
  // Inverse rotation of V over the 256 dims (so[0] reused as scratch).
  float* x = so[0];
  x[tid] = acc;
  __syncthreads();
  for (int hh = 1; hh < D; hh <<= 1) {
    float a = 0.f, b = 0.f;
    const bool lo = (tid & hh) == 0;
    if (lo) {
      a = x[tid];
      b = x[tid + hh];
    }
    __syncthreads();
    if (lo) {
      x[tid] = a + b;
      x[tid + hh] = a - b;
    }
    __syncthreads();
  }
  const float inv_l = lg > 0.f ? 1.0f / lg : 0.f;
  out[qi * soo + hi * soh + tid] =
      from_f<OutT>(x[tid] * 0.0625f * sign_of(v_signs, tid) * inv_l);
}

// ---------------------------------------------------------------------------
// In-place rotation of [T, H, 256] vectors for the prefill path, like K
// (signs, then 64-dim Hadamard blocks) or like V (one 256-dim block);
// `inverse` applies the transform then the signs. grid (T, H), block 32.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(32)
    rotate_kernel(T* __restrict__ x, const int* __restrict__ signs,
                  bool k_layout, bool inverse, int64_t s0, int64_t s1) {
  const int lane = threadIdx.x;
  T* p = x + blockIdx.x * s0 + blockIdx.y * s1 + lane * DPL;
  float v[DPL];
  #pragma unroll
  for (int i = 0; i < DPL; ++i) v[i] = to_f<T>(p[i]);
  if (!inverse) {
  #pragma unroll
    for (int i = 0; i < DPL; ++i) v[i] *= sign_of(signs, lane * DPL + i);
  }
  float scale;
  if (k_layout) {
    fwht_wave<4>(v, lane);
    scale = 0.125f;
  } else {
    fwht_wave<16>(v, lane);
    scale = 0.0625f;
  }
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    float y = v[i] * scale;
    if (inverse) y *= sign_of(signs, lane * DPL + i);
    p[i] = from_f<T>(y);
  }
}


// ---------------------------------------------------------------------------
// Exact window (sparse decode). Direct-mapped fp16 shadow tables keyed by the
// cache slot: entry slot & mask holds that token's exact K and V ([hkv, 2D])
// and is valid while tags[entry] == slot. The recent table takes every stored
// token, the sink table the first kSinks positions of a sequence. A lost
// entry only means the token is read from the packed cache instead.
// ---------------------------------------------------------------------------
constexpr int kWindow = 64;
constexpr int kSinks = 4;
constexpr int kMaxWindowHits = 128;

// One wave per token. The per-entry lock keeps two tokens that share an
// entry in the same launch from leaving a mixed row behind.
template <typename T>
__global__ void __launch_bounds__(32)
    window_store_kernel(const T* __restrict__ key, const T* __restrict__ value,
                        const int64_t* __restrict__ slot_mapping,
                        const int* __restrict__ positions, T* __restrict__ wtab,
                        int* __restrict__ wtags, int* __restrict__ wlocks,
                        int wmask, T* __restrict__ stab,
                        int* __restrict__ stags, int* __restrict__ slocks,
                        int smask, int hkv, int64_t sk0, int64_t sk1,
                        int64_t sv0, int64_t sv1) {
  const int i = blockIdx.x, lane = threadIdx.x;
  const int64_t slot = slot_mapping[i];
  if (slot < 0) return;
  const int pos = positions[i];
  auto put = [&](T* tab, int* tags, int* locks, int e) {
    if (lane == 0) {
      while (atomicCAS(&locks[e], 0, 1) != 0) {
      }
      atomicExch(&tags[e], -1);
    }
    __threadfence();
    for (int h = 0; h < hkv; ++h) {
      T* row = tab + ((int64_t)e * hkv + h) * (2 * D);
  #pragma unroll
      for (int d = 0; d < DPL; ++d) {
        row[lane * DPL + d] = key[i * sk0 + h * sk1 + lane * DPL + d];
        row[D + lane * DPL + d] = value[i * sv0 + h * sv1 + lane * DPL + d];
      }
    }
    __threadfence();
    if (lane == 0) {
      atomicExch(&tags[e], (int)slot);
      __threadfence();
      atomicExch(&locks[e], 0);
    }
  };
  put(wtab, wtags, wlocks, (int)(slot & wmask));
  if (pos >= 0 && pos < kSinks) put(stab, stags, slocks, (int)(slot & smask));
}

// Attention of one query row over its selected positions that sit in the
// exact window, written as one more split of mid_o (in the rotated V space,
// like the packed-cache splits). grid (num_q, hkv), block 256.
template <typename T>
__global__ void __launch_bounds__(256)
    window_attn_kernel(const T* __restrict__ query,
                       const int* __restrict__ positions,
                       const int* __restrict__ q_to_req,
                       const int* __restrict__ block_table,
                       const int* __restrict__ indices,
                       const T* __restrict__ wtab, const int* __restrict__ wtags,
                       int wmask, const T* __restrict__ stab,
                       const int* __restrict__ stags, int smask,
                       float* __restrict__ mid_o, const int* __restrict__ v_signs,
                       float sm_scale, int split, int topk, int64_t sidx,
                       int block_size, int max_blocks, int num_reqs,
                       int num_phys_blocks, int hq, int hkv, int64_t sq0,
                       int64_t sq1, int64_t smo, int64_t smh, int64_t sms) {
  __shared__ int s_entry[kMaxWindowHits];
  __shared__ float s_score[8][kMaxWindowHits];
  __shared__ int s_count;
  const int qi = blockIdx.x, kvh = blockIdx.y;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int hpk = hq / hkv;
  if (tid == 0) s_count = 0;
  __syncthreads();
  const int req = q_to_req[qi];
  const int p = positions[qi];
  if (req >= 0 && req < num_reqs) {
    for (int j = tid; j < topk; j += 256) {
      const int t = indices[qi * sidx + j];
      const int d = p - t;
      if (t < 0 || d < 0 || (d >= kWindow && t >= kSinks)) continue;
      const int lb = t / block_size;
      if (lb >= max_blocks) continue;
      const int pb = block_table[req * max_blocks + lb];
      if (pb < 0 || pb >= num_phys_blocks) continue;
      const int slot = pb * block_size + (t - lb * block_size);
      int e = -1;
      if (d < kWindow && wtags[slot & wmask] == slot)
        e = slot & wmask;
      else if (t < kSinks && stags[slot & smask] == slot)
        e = (slot & smask) | (1 << 30);
      if (e < 0) continue;
      const int k = atomicAdd(&s_count, 1);
      if (k < kMaxWindowHits) s_entry[k] = e;
    }
  }
  __syncthreads();
  const int n = min(s_count, kMaxWindowHits);
  const float sl2 = sm_scale * 1.4426950408889634f;
  for (int g = w; g < hpk; g += 8) {
    const int head = kvh * hpk + g;
    float q[DPL];
  #pragma unroll
    for (int d = 0; d < DPL; ++d)
      q[d] = to_f<T>(query[qi * sq0 + head * sq1 + lane * DPL + d]);
    auto row_of = [&](int e) {
      const T* tab = (e >> 30) ? stab : wtab;
      return tab + ((int64_t)(e & ((1 << 30) - 1)) * hkv + kvh) * (2 * D);
    };
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) {
      const T* r = row_of(s_entry[i]);
      float acc = 0.f;
  #pragma unroll
      for (int d = 0; d < DPL; ++d) acc += q[d] * to_f<T>(r[lane * DPL + d]);
      acc = xor_sum<16>(acc) * sl2;
      if (lane == 0) s_score[w][i] = acc;
      m = fmaxf(m, acc);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    float l = 0.f, o[DPL] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
      const float pr = exp2f(s_score[w][i] - m);
      l += pr;
      const T* r = row_of(s_entry[i]) + D;
  #pragma unroll
      for (int d = 0; d < DPL; ++d) o[d] += pr * to_f<T>(r[lane * DPL + d]);
    }
    rotate_full(o, lane, v_signs);
    float* op = mid_o + qi * smo + head * smh + (int64_t)split * sms;
  #pragma unroll
    for (int d = 0; d < DPL; ++d) op[lane * DPL + d] = o[d];
    if (lane == 0) {
      op[D] = n > 0 ? m : -INFINITY;
      op[D + 1] = l;
    }
  }
}

}  // namespace octave

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
using namespace octave;

// fmt: V fmt (3 or 4), plus kCompactFlag for compact K (3-bit V only).
int octave_slot_bytes(int64_t fmt) {
  switch (fmt) {
    case 3:
      return Format<3>::SLOT;
    case 4:
      return Format<4>::SLOT;
    case 3 + kCompactFlag:
      return Format<3, true>::SLOT;
  }
  TORCH_CHECK(false, "octave: unknown format id ", fmt);
  return 0;
}

static void check_cache(const torch::Tensor& cache, int64_t fmt) {
  TORCH_CHECK(cache.dtype() == at::kByte && cache.dim() == 4,
              "octave: cache must be uint8 [blocks, heads, block_size, slot]");
  const int slot = octave_slot_bytes(fmt);
  TORCH_CHECK(cache.size(3) == slot && cache.stride(3) == 1,
              "octave: slot size mismatch, expected ", slot, " got ",
              cache.size(3));
}

void octave_cache_store(torch::Tensor key, torch::Tensor value,
                        torch::Tensor cache, torch::Tensor slot_mapping,
                        torch::Tensor k_signs, torch::Tensor v_signs,
                        int64_t fmt) {
  const int n = slot_mapping.size(0);
  if (n == 0) return;
  check_cache(cache, fmt);
  TORCH_CHECK(key.size(-1) == D && key.stride(-1) == 1 && value.stride(-1) == 1,
              "octave: head_size must be 256 with contiguous last dim");
  TORCH_CHECK(slot_mapping.dtype() == at::kLong);
  const int hkv = cache.size(1);
  const at::cuda::OptionalCUDAGuard guard(device_of(key));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid(n, hkv);
  #define SQ_STORE(B, KC, T)                                             \
    store_kernel<B, KC, T><<<grid, 32, 0, stream>>>(                     \
        (const T*)key.data_ptr(), (const T*)value.data_ptr(),            \
        (uint8_t*)cache.data_ptr(), slot_mapping.data_ptr<int64_t>(),    \
        k_signs.data_ptr<int>(), v_signs.data_ptr<int>(), cache.size(2), \
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),  \
        cache.stride(0), cache.stride(1), cache.stride(2))
  const bool bf = key.dtype() == at::kBFloat16;
  if (fmt == 4) {
    if (bf)
      SQ_STORE(4, false, __hip_bfloat16);
    else
      SQ_STORE(4, false, half);
  } else if (fmt == 3) {
    if (bf)
      SQ_STORE(3, false, __hip_bfloat16);
    else
      SQ_STORE(3, false, half);
  } else {
    if (bf)
      SQ_STORE(3, true, __hip_bfloat16);
    else
      SQ_STORE(3, true, half);
  }
  #undef SQ_STORE
}

template <int VB, bool KC, int G, int QG, typename QT>
static void launch_decode(torch::Tensor& query, torch::Tensor& cache,
                          torch::Tensor& block_table, torch::Tensor& q_to_req,
                          torch::Tensor& q_to_klen, torch::Tensor& mid_o,
                          torch::Tensor& k_signs, float sm_scale_log2, int ns,
                          hipStream_t stream) {
  const int num_q = query.size(0);
  dim3 grid(num_q, cache.size(1), ns);
  decode_kernel<VB, KC, G, QG, QT><<<grid, 128, 0, stream>>>(
      (const QT*)query.data_ptr(), (const uint8_t*)cache.data_ptr(),
      block_table.data_ptr<int>(), q_to_req.data_ptr<int>(),
      q_to_klen.data_ptr<int>(), mid_o.data_ptr<float>(),
      k_signs.data_ptr<int>(), sm_scale_log2, num_q, cache.size(2),
      block_table.size(1), block_table.size(0), cache.size(0), ns,
      query.stride(0), query.stride(1), cache.stride(0), cache.stride(1),
      cache.stride(2), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2));
}

template <int VB, bool KC, typename QT>
static void dispatch_decode(int g, int qg, torch::Tensor& query,
                            torch::Tensor& cache, torch::Tensor& block_table,
                            torch::Tensor& q_to_req, torch::Tensor& q_to_klen,
                            torch::Tensor& mid_o, torch::Tensor& k_signs,
                            float sl2, int ns, hipStream_t stream) {
  #define SQ_DEC(GG, QQ)                                                     \
    if (g == GG && qg == QQ) {                                               \
      launch_decode<VB, KC, GG, QQ, QT>(query, cache, block_table, q_to_req, \
                                        q_to_klen, mid_o, k_signs, sl2, ns,  \
                                        stream);                             \
      return;                                                                \
    }
  SQ_DEC(1, 1);
  SQ_DEC(1, 4);
  SQ_DEC(2, 1);
  SQ_DEC(2, 4);
  SQ_DEC(4, 1);
  SQ_DEC(4, 4);
  SQ_DEC(6, 1);
  SQ_DEC(6, 4);
  SQ_DEC(8, 1);
  SQ_DEC(8, 4);
  #undef SQ_DEC
  TORCH_CHECK(false, "octave_decode: unsupported GQA group ", g);
}

int octave_decode_wmma(torch::Tensor query, torch::Tensor cache,
                       torch::Tensor block_table, torch::Tensor q_to_req,
                       torch::Tensor q_to_klen, torch::Tensor mid_o,
                       torch::Tensor k_signs, double sm_scale,
                       int64_t num_kv_splits, int64_t fmt,
                       const std::optional<torch::Tensor>& indices,
                       const int* qpos, const int* wtags, int wmask,
                       const int* stags, int smask);

static void launch_reduce(torch::Tensor out, torch::Tensor mid_o,
                          torch::Tensor v_signs, int num_splits,
                          cudaStream_t stream) {
  dim3 grid(out.size(0), out.size(1));
  if (out.dtype() == at::kBFloat16)
    reduce_kernel<__hip_bfloat16><<<grid, 256, 0, stream>>>(
        mid_o.data_ptr<float>(), (__hip_bfloat16*)out.data_ptr(),
        v_signs.data_ptr<int>(), num_splits, mid_o.stride(0), mid_o.stride(1),
        mid_o.stride(2), out.stride(0), out.stride(1));
  else
    reduce_kernel<half><<<grid, 256, 0, stream>>>(
        mid_o.data_ptr<float>(), (half*)out.data_ptr(), v_signs.data_ptr<int>(),
        num_splits, mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        out.stride(0), out.stride(1));
}

void octave_decode(torch::Tensor out, torch::Tensor query, torch::Tensor cache,
                   torch::Tensor block_table, torch::Tensor q_to_req,
                   torch::Tensor q_to_klen, torch::Tensor mid_o,
                   torch::Tensor k_signs, torch::Tensor v_signs,
                   double sm_scale, int64_t num_kv_splits, int64_t fmt,
                   int64_t query_group, bool use_wmma) {
  const int num_q = query.size(0);
  if (num_q == 0) return;
  check_cache(cache, fmt);
  TORCH_CHECK(query.size(2) == D && query.stride(2) == 1);
  TORCH_CHECK(out.dtype() == query.dtype());
  TORCH_CHECK(mid_o.size(2) >= num_kv_splits && mid_o.size(3) >= D + 2);
  TORCH_CHECK(num_kv_splits <= 1024 && mid_o.stride(3) == 1);
  const int hq = query.size(1), hkv = cache.size(1);
  TORCH_CHECK(hq % hkv == 0);
  const int g = hq / hkv;
  const int qg = query_group > 1 ? 4 : 1;
  int ns = (int)num_kv_splits;
  const float sl2 = (float)sm_scale * 1.4426950408889634f;
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bool bf = query.dtype() == at::kBFloat16;
  // Fast tier when the device and shapes allow it, else the portable kernel.
  const int wmma_splits =
      use_wmma
          ? octave_decode_wmma(query, cache, block_table, q_to_req, q_to_klen,
                               mid_o, k_signs, sm_scale, num_kv_splits, fmt,
                               std::nullopt, nullptr, nullptr, 0, nullptr, 0)
          : 0;
  if (wmma_splits > 0) {
    ns = wmma_splits;
  } else {
  #define SQ_DISPATCH(B, KC)                                                   \
    if (bf)                                                                    \
      dispatch_decode<B, KC, __hip_bfloat16>(g, qg, query, cache, block_table, \
                                             q_to_req, q_to_klen, mid_o,       \
                                             k_signs, sl2, ns, stream);        \
    else                                                                       \
      dispatch_decode<B, KC, half>(g, qg, query, cache, block_table, q_to_req, \
                                   q_to_klen, mid_o, k_signs, sl2, ns, stream)
    if (fmt == 4) {
      SQ_DISPATCH(4, false);
    } else if (fmt == 3) {
      SQ_DISPATCH(3, false);
    } else {
      SQ_DISPATCH(3, true);
    }
  #undef SQ_DISPATCH
  }
  launch_reduce(out, mid_o, v_signs, ns, stream);
}

// Sparse decode: query row i attends to the logical positions indices[i, :]
// of request q_to_req[i] (negative entries are skipped). WMMA tier only.
void octave_decode_sparse(
    torch::Tensor out, torch::Tensor query, torch::Tensor cache,
    torch::Tensor block_table, torch::Tensor q_to_req, torch::Tensor indices,
    torch::Tensor mid_o, torch::Tensor k_signs, torch::Tensor v_signs,
    double sm_scale, int64_t num_kv_splits, int64_t fmt,
    const std::optional<torch::Tensor>& positions,
    const std::optional<torch::Tensor>& wtab,
    const std::optional<torch::Tensor>& wtags,
    const std::optional<torch::Tensor>& stab,
    const std::optional<torch::Tensor>& stags) {
  const int num_q = query.size(0);
  if (num_q == 0) return;
  check_cache(cache, fmt);
  TORCH_CHECK(query.size(2) == D && query.stride(2) == 1);
  TORCH_CHECK(out.dtype() == query.dtype());
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) >= num_q &&
              indices.stride(1) == 1 && indices.dtype() == at::kInt);
  TORCH_CHECK(mid_o.size(2) >= num_kv_splits && mid_o.size(3) >= D + 2);
  TORCH_CHECK(num_kv_splits <= 1024 && mid_o.stride(3) == 1);
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // With the exact window, the packed-cache splits leave the last split of
  // mid_o to octave_window_attn.
  const bool win = positions.has_value();
  TORCH_CHECK(!win || (wtab && wtags && stab && stags && num_kv_splits >= 2));
  const int wmask = win ? (int)wtags->size(0) - 1 : 0;
  const int smask = win ? (int)stags->size(0) - 1 : 0;
  TORCH_CHECK(!win || (((wmask + 1) & wmask) == 0 && ((smask + 1) & smask) == 0),
              "octave_decode_sparse: shadow tables must be powers of two");
  const int ns = octave_decode_wmma(
      query, cache, block_table, q_to_req, q_to_req, mid_o, k_signs, sm_scale,
      win ? num_kv_splits - 1 : num_kv_splits, fmt, indices,
      win ? positions->data_ptr<int>() : nullptr,
      win ? wtags->data_ptr<int>() : nullptr, wmask,
      win ? stags->data_ptr<int>() : nullptr, smask);
  TORCH_CHECK(ns > 0,
              "octave_decode_sparse: needs the WMMA tier (gfx11, head size "
              "256, GQA group <= 8)");
  if (!win) {
    launch_reduce(out, mid_o, v_signs, ns, stream);
    return;
  }
  const int hq = query.size(1), hkv = cache.size(1);
  TORCH_CHECK(hq / hkv <= 8 && wtab->size(1) == hkv && wtab->size(2) == 2 * D);
  dim3 grid(num_q, hkv);
  #define SQ_WIN(T)                                                          \
    window_attn_kernel<T><<<grid, 256, 0, stream>>>(                         \
        (const T*)query.data_ptr(), positions->data_ptr<int>(),              \
        q_to_req.data_ptr<int>(), block_table.data_ptr<int>(),               \
        indices.data_ptr<int>(), (const T*)wtab->data_ptr(),                 \
        wtags->data_ptr<int>(), wmask, (const T*)stab->data_ptr(),           \
        stags->data_ptr<int>(), smask, mid_o.data_ptr<float>(),              \
        v_signs.data_ptr<int>(), (float)sm_scale, ns, (int)indices.size(1),  \
        indices.stride(0), cache.size(2), block_table.size(1),               \
        block_table.size(0), cache.size(0), hq, hkv, query.stride(0),        \
        query.stride(1), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2))
  if (query.dtype() == at::kBFloat16)
    SQ_WIN(__hip_bfloat16);
  else
    SQ_WIN(half);
  #undef SQ_WIN
  launch_reduce(out, mid_o, v_signs, ns + 1, stream);
}

void octave_window_store(torch::Tensor key, torch::Tensor value,
                         torch::Tensor slot_mapping, torch::Tensor positions,
                         torch::Tensor wtab, torch::Tensor wtags,
                         torch::Tensor wlocks, torch::Tensor stab,
                         torch::Tensor stags, torch::Tensor slocks) {
  const int n = slot_mapping.size(0);
  if (n == 0) return;
  TORCH_CHECK(slot_mapping.dtype() == at::kLong && positions.dtype() == at::kInt);
  TORCH_CHECK(wtab.dtype() == key.dtype() && stab.dtype() == key.dtype());
  const int hkv = wtab.size(1);
  const int wmask = (int)wtags.size(0) - 1, smask = (int)stags.size(0) - 1;
  const at::cuda::OptionalCUDAGuard guard(device_of(key));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  #define SQ_WSTORE(T)                                                        \
    window_store_kernel<T><<<n, 32, 0, stream>>>(                             \
        (const T*)key.data_ptr(), (const T*)value.data_ptr(),                 \
        slot_mapping.data_ptr<int64_t>(), positions.data_ptr<int>(),          \
        (T*)wtab.data_ptr(), wtags.data_ptr<int>(), wlocks.data_ptr<int>(),   \
        wmask, (T*)stab.data_ptr(), stags.data_ptr<int>(),                    \
        slocks.data_ptr<int>(), smask, hkv, key.stride(0), key.stride(1),     \
        value.stride(0), value.stride(1))
  if (key.dtype() == at::kBFloat16)
    SQ_WSTORE(__hip_bfloat16);
  else
    SQ_WSTORE(half);
  #undef SQ_WSTORE
}

void octave_rotate(torch::Tensor x, torch::Tensor signs, bool k_layout,
                   bool inverse) {
  if (x.numel() == 0) return;
  TORCH_CHECK(x.dim() == 3 && x.size(2) == D && x.stride(2) == 1,
              "octave_rotate: x must be [T, H, 256] with contiguous last dim");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid(x.size(0), x.size(1));
  if (x.dtype() == at::kBFloat16)
    rotate_kernel<__hip_bfloat16><<<grid, 32, 0, stream>>>(
        (__hip_bfloat16*)x.data_ptr(), signs.data_ptr<int>(), k_layout, inverse,
        x.stride(0), x.stride(1));
  else
    rotate_kernel<half>
        <<<grid, 32, 0, stream>>>((half*)x.data_ptr(), signs.data_ptr<int>(),
                                  k_layout, inverse, x.stride(0), x.stride(1));
}

#else
void octave_cache_store(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, torch::Tensor, int64_t) {
  TORCH_CHECK(false, "octave requires ROCm");
}
void octave_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, double, int64_t, int64_t, int64_t, bool) {
  TORCH_CHECK(false, "octave requires ROCm");
}
void octave_decode_sparse(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor, double,
                          int64_t, int64_t,
                          const std::optional<torch::Tensor>&,
                          const std::optional<torch::Tensor>&,
                          const std::optional<torch::Tensor>&,
                          const std::optional<torch::Tensor>&,
                          const std::optional<torch::Tensor>&) {
  TORCH_CHECK(false, "octave requires ROCm");
}
void octave_window_store(torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor) {
  TORCH_CHECK(false, "octave requires ROCm");
}
void octave_rotate(torch::Tensor, torch::Tensor, bool, bool) {
  TORCH_CHECK(false, "octave requires ROCm");
}
#endif
