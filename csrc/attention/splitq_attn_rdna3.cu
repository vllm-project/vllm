// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SplitQ KV cache kernels for RDNA3 (gfx11), head_size 256 with 64 RoPE dims.
// Format: vllm/v1/attention/ops/rocm_splitq.py (the PyTorch reference there
// is the source of truth for the byte layout).
//
//   splitq_cache_store   quantize + pack K/V into slots (one wave per slot).
//   splitq_decode        split-KV attention per query group; K dot products
//                        with v_dot4_i32_iu8 on the packed codes, the query
//                        quantized to int8 once per block. Consecutive query
//                        tokens of one request (MTP verify) share the KV read.
//   splitq_to_int8       expand a request's cached prefix into int8
//                        per-token-head buffers for the WMMA prefill kernel.
//
// Everything is in the rotated space: NoPE K dims and V are sign-flipped and
// Hadamard-rotated at store time; the query NoPE part is rotated the same way
// and the attention output is rotated back in the reduce kernel.

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

namespace splitq {

constexpr int D = 256;       // head size
constexpr int R = 64;        // RoPE dims (int8)
constexpr int N = D - R;     // NoPE dims (rotated, packed)
constexpr int NB = N / 64;   // NoPE Hadamard blocks
constexpr int DPL = D / 32;  // dims per lane when a wave owns one slot

template <int BITS>
struct Layout {
  static constexpr int OFF_KN = R;
  static constexpr int OFF_V = OFF_KN + N * BITS / 8;
  static constexpr int OFF_SC = OFF_V + D * BITS / 8;
  static constexpr int NUM_SC = 2 + NB;
  static constexpr int SLOT = (OFF_SC + 2 * NUM_SC + 7) / 8 * 8;
  static constexpr int LEVELS = 1 << BITS;
  static constexpr float C0 = (LEVELS - 1) * 0.5f;
  static constexpr float STEP = BITS == 4 ? 0.3355f : 0.5865f;
};

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

// Rotate a lane's 8 dims: NoPE lanes (8..31) in 64-dim blocks, or all lanes
// as one 256-dim block. RoPE lanes are left untouched in the NoPE case.
__device__ __forceinline__ void rotate_nope(float (&x)[DPL], int lane,
                                            const int* nope_signs) {
  float y[DPL];
  #pragma unroll
  for (int i = 0; i < DPL; ++i)
    y[i] = lane >= R / DPL ? x[i] * sign_of(nope_signs, lane * DPL + i - R)
                           : x[i];
  fwht_wave<4>(y, lane);
  if (lane >= R / DPL) {
  #pragma unroll
    for (int i = 0; i < DPL; ++i) x[i] = y[i] * 0.125f;
  }
}

__device__ __forceinline__ void rotate_full(float (&x)[DPL], int lane,
                                            const int* signs) {
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= sign_of(signs, lane * DPL + i);
  fwht_wave<16>(x, lane);
  #pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= 0.0625f;
}

// Uniform midrise quantizer over the lanes of one reduction group.
// Returns the least-squares scale; codes in [0, LEVELS).
template <int BITS, int MASK>
__device__ __forceinline__ float quantize_group(const float (&x)[DPL],
                                                int (&codes)[DPL],
                                                float group_dims) {
  using L = Layout<BITS>;
  float ss = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) ss += x[i] * x[i];
  ss = xor_sum<MASK>(ss);
  const float step = fmaxf(sqrtf(ss / group_dims), 1e-12f) * L::STEP;
  const float inv_step = 1.0f / step;
  float num = 0.f, den = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    int c = (int)floorf(x[i] * inv_step + L::LEVELS * 0.5f);
    c = min(max(c, 0), L::LEVELS - 1);
    codes[i] = c;
    const float cc = c - L::C0;
    num += x[i] * cc;
    den += cc * cc;
  }
  num = xor_sum<MASK>(num);
  den = xor_sum<MASK>(den);
  return den > 0.f ? num / den : 0.f;
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
template <int BITS, typename KT>
__global__ void __launch_bounds__(32)
    store_kernel(const KT* __restrict__ key, const KT* __restrict__ value,
                 uint8_t* __restrict__ cache,
                 const int64_t* __restrict__ slot_mapping,
                 const int* __restrict__ nope_signs,
                 const int* __restrict__ v_signs, int block_size, int64_t sk0,
                 int64_t sk1, int64_t sv0, int64_t sv1, int64_t scb,
                 int64_t sch, int64_t sct) {
  using L = Layout<BITS>;
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  uint8_t* dst = cache + (slot / block_size) * scb + h * sch +
                 (slot % block_size) * sct;

  float k[DPL], v[DPL];
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    k[i] = to_f<KT>(key[t * sk0 + h * sk1 + lane * DPL + i]);
    v[i] = to_f<KT>(value[t * sv0 + h * sv1 + lane * DPL + i]);
  }
  const bool rope_lane = lane < R / DPL;

  // RoPE part: symmetric int8 per slot.
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) amax = fmaxf(amax, fabsf(k[i]));
  amax = xor_max<4>(amax);
  const float kr_scale = fmaxf(amax, 1e-12f) / 127.0f;
  if (rope_lane) {
    uint32_t w[2] = {0, 0};
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      int q = __float2int_rn(k[i] / kr_scale);
      q = min(max(q, -127), 127);
      w[i >> 2] |= (uint32_t)(q & 0xFF) << (8 * (i & 3));
    }
    *reinterpret_cast<uint2*>(dst + lane * DPL) = make_uint2(w[0], w[1]);
  }

  // NoPE part of K: rotate, then quantize per 64-dim block (8 lanes).
  rotate_nope(k, lane, nope_signs);
  int kc[DPL];
  const float kn_scale = quantize_group<BITS, 4>(k, kc, 64.f);
  const int nl = lane - R / DPL;  // NoPE lane index (valid when >= 0)
  if constexpr (BITS == 4) {
    if (!rope_lane)
      *reinterpret_cast<uint32_t*>(dst + L::OFF_KN + 4 * nl) = pack4(kc);
  } else {
    uint32_t lo, hi;
    pack3(kc, nl, lo, hi);
    lo = xor_or<1>(lo);
    hi = xor_or<2>(hi);
    if (!rope_lane && (nl & 1) == 0)
      *reinterpret_cast<uint32_t*>(dst + L::OFF_KN + 4 * (nl >> 1)) = lo;
    if (!rope_lane && (nl & 3) == 0)
      *reinterpret_cast<uint32_t*>(dst + L::OFF_KN + N / 4 + 4 * (nl >> 2)) =
          hi;
  }

  // V: rotate all 256 dims, one scale.
  rotate_full(v, lane, v_signs);
  int vc[DPL];
  const float v_scale = quantize_group<BITS, 16>(v, vc, (float)D);
  if constexpr (BITS == 4) {
    *reinterpret_cast<uint32_t*>(dst + L::OFF_V + 4 * lane) = pack4(vc);
  } else {
    uint32_t lo, hi;
    pack3(vc, lane, lo, hi);
    lo = xor_or<1>(lo);
    hi = xor_or<2>(hi);
    if ((lane & 1) == 0)
      *reinterpret_cast<uint32_t*>(dst + L::OFF_V + 4 * (lane >> 1)) = lo;
    if ((lane & 3) == 0)
      *reinterpret_cast<uint32_t*>(dst + L::OFF_V + D / 4 + 4 * (lane >> 2)) =
          hi;
  }

  uint16_t* sc = reinterpret_cast<uint16_t*>(dst + L::OFF_SC);
  if (lane == 0) {
    sc[0] = f_to_half_bits(kr_scale);
    sc[1 + NB] = f_to_half_bits(v_scale);
  }
  if (!rope_lane && (nl & 7) == 0) sc[1 + (nl >> 3)] = f_to_half_bits(kn_scale);
}

// ---------------------------------------------------------------------------
// Decode stage 1: grid (num_q, num_kv_heads, num_splits), block 128.
// A block serves up to QG consecutive query tokens of one request (only the
// first token of each group of a run launches work) and all G query heads of
// its KV head: ROWS = QG * G attention rows share every K/V load.
// ---------------------------------------------------------------------------
constexpr int WAVES = 4;
constexpr int CHUNK = 32 * WAVES;

template <int BITS, int G, int QG, typename QT>
__global__ void __launch_bounds__(128) decode_kernel(
    const QT* __restrict__ Q, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen, float* __restrict__ mid_o,
    const int* __restrict__ nope_signs, float sm_scale_log2, int num_q,
    int block_size, int max_blocks, int num_reqs, int num_phys_blocks,
    int num_splits, int64_t sq0, int64_t sq1, int64_t scb, int64_t sch,
    int64_t sct, int64_t smo, int64_t smh, int64_t sms) {
  using L = Layout<BITS>;
  constexpr int ROWS = QG * G;
  constexpr int QW = D / 4;  // int8 query words per row

  __shared__ int q8[ROWS][QW];
  __shared__ float q_rs[ROWS], q_ns[ROWS];
  __shared__ float q_sum[ROWS][NB];
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

  // Query: rotate the NoPE part, quantize RoPE and NoPE parts to int8.
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
    rotate_nope(x, lane, nope_signs);
    const bool rope_lane = lane < R / DPL;
    float amax = 0.f;
  #pragma unroll
    for (int i = 0; i < DPL; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float a_r = xor_max<16>(rope_lane ? amax : 0.f);
    const float a_n = xor_max<16>(rope_lane ? 0.f : amax);
    const float s = fmaxf(rope_lane ? a_r : a_n, 1e-20f) / 127.0f;
    uint32_t w[2] = {0, 0};
    int isum = 0;
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      int q = __float2int_rn(x[i] / s);
      q = min(max(q, -127), 127);
      isum += q;
      w[i >> 2] |= (uint32_t)(q & 0xFF) << (8 * (i & 3));
    }
    q8[r][2 * lane] = (int)w[0];
    q8[r][2 * lane + 1] = (int)w[1];
    const float bsum = xor_sum<4>((float)isum);
    if (!rope_lane && (lane & 7) == 0) q_sum[r][(lane - R / DPL) >> 3] = bsum;
    if (lane == 0) {
      q_rs[r] = fmaxf(a_r, 1e-20f) / 127.0f;
      q_ns[r] = fmaxf(a_n, 1e-20f) / 127.0f;
    }
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

      const uint2* p2 = reinterpret_cast<const uint2*>(p);
      uint32_t kr[R / 4];
  #pragma unroll
      for (int w = 0; w < R / 8; ++w) {
        uint2 u = p2[w];
        kr[2 * w] = u.x;
        kr[2 * w + 1] = u.y;
      }
      constexpr int KNW = N * BITS / 32;
      uint32_t kn[KNW];
      const uint2* pn = reinterpret_cast<const uint2*>(p + L::OFF_KN);
  #pragma unroll
      for (int w = 0; w < KNW / 2; ++w) {
        uint2 u = pn[w];
        kn[2 * w] = u.x;
        kn[2 * w + 1] = u.y;
      }
      const uint2 sc2 = *reinterpret_cast<const uint2*>(p + L::OFF_SC);
      const uint32_t sc_last =
          *reinterpret_cast<const uint16_t*>(p + L::OFF_SC + 8);
      const float kr_s = half_bits_to_f(sc2.x & 0xFFFF);
      float kn_s[NB];
      kn_s[0] = half_bits_to_f(sc2.x >> 16);
      kn_s[1] = half_bits_to_f(sc2.y & 0xFFFF);
      kn_s[2] = half_bits_to_f(sc2.y >> 16);
      tok_vs[wave * 32 + lane] = half_bits_to_f(sc_last);

  #pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int* qr = q8[r];
        int dr = 0;
  #pragma unroll
        for (int w = 0; w < R / 4; ++w)
          dr = __builtin_amdgcn_sudot4(true, qr[w], true, (int)kr[w], dr,
                                       false);
        float sn = 0.f;
  #pragma unroll
        for (int b = 0; b < NB; ++b) {
          int dn = 0;
          if constexpr (BITS == 4) {
  #pragma unroll
            for (int u = 0; u < 8; ++u) {
              const uint32_t w = kn[8 * b + u];
              const int qw = R / 4 + 16 * b + 2 * u;
              dn = __builtin_amdgcn_sudot4(true, qr[qw], true,
                                           (int)(w & 0x0F0F0F0Fu), dn, false);
              dn = __builtin_amdgcn_sudot4(true, qr[qw + 1], true,
                                           (int)((w >> 4) & 0x0F0F0F0Fu), dn,
                                           false);
            }
          } else {
            int dh = 0;
  #pragma unroll
            for (int v = 0; v < 4; ++v) {  // 2-bit plane, 16 dims per word
              const uint32_t w = kn[4 * b + v];
  #pragma unroll
              for (int i = 0; i < 4; ++i)
                dn = __builtin_amdgcn_sudot4(
                    true, qr[R / 4 + 16 * b + 4 * v + i], true,
                    (int)((w >> (2 * i)) & 0x03030303u), dn, false);
            }
  #pragma unroll
            for (int x = 0; x < 2; ++x) {  // 1-bit plane, 32 dims per word
              const uint32_t w = kn[N / 16 + 2 * b + x];
  #pragma unroll
              for (int i = 0; i < 8; ++i)
                dh = __builtin_amdgcn_sudot4(
                    true, qr[R / 4 + 16 * b + 8 * x + i], true,
                    (int)((w >> i) & 0x01010101u), dh, false);
            }
            dn += 4 * dh;
          }
          sn += kn_s[b] * ((float)dn - L::C0 * q_sum[r][b]);
        }
        const float sc =
            (q_rs[r] * kr_s * (float)dr + q_ns[r] * sn) * sm_scale_log2;
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
      float c0, c1;
      if constexpr (BITS == 4) {
        const uint32_t w =
            *reinterpret_cast<const uint32_t*>(p + L::OFF_V + 4 * (my_d >> 3));
        const int e = my_d & 7;  // even
        c0 = (float)((w >> (8 * (e & 3) + 4 * (e >> 2))) & 15);
        c1 = (float)((w >> (8 * ((e + 1) & 3) + 4 * ((e + 1) >> 2))) & 15);
      } else {
        const uint32_t lo = *reinterpret_cast<const uint32_t*>(
            p + L::OFF_V + 4 * (my_d >> 4));
        const uint32_t hi = *reinterpret_cast<const uint32_t*>(
            p + L::OFF_V + D / 4 + 4 * (my_d >> 5));
        const int e16 = my_d & 15, e32 = my_d & 31;
        c0 = (float)(((lo >> (8 * (e16 & 3) + 2 * (e16 >> 2))) & 3) |
                     (((hi >> (8 * (e32 & 3) + (e32 >> 2))) & 1) << 2));
        const int f16 = e16 + 1, f32 = e32 + 1;
        c1 = (float)(((lo >> (8 * (f16 & 3) + 2 * (f16 >> 2))) & 3) |
                     (((hi >> (8 * (f32 & 3) + (f32 >> 2))) & 1) << 2));
      }
      const float vs = tok_vs[t];
      c0 = (c0 - L::C0) * vs;
      c1 = (c1 - L::C0) * vs;
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
// block 256 (one thread per dim).
// ---------------------------------------------------------------------------
template <typename OT>
__global__ void __launch_bounds__(256)
    reduce_kernel(const float* __restrict__ mid_o, OT* __restrict__ out,
                  const int* __restrict__ v_signs, int num_splits, int64_t smo,
                  int64_t smh, int64_t sms, int64_t soo, int64_t soh) {
  __shared__ float x[D];
  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int d = threadIdx.x;
  const float* base = mid_o + qi * smo + hi * smh;

  float m = -INFINITY;
  for (int s = 0; s < num_splits; ++s) m = fmaxf(m, base[s * sms + D]);
  float o = 0.f, l = 0.f;
  if (m != -INFINITY) {
    for (int s = 0; s < num_splits; ++s) {
      const float* sp = base + s * sms;
      const float ms = sp[D];
      if (ms == -INFINITY) continue;
      const float a = exp2f(ms - m);
      o += sp[d] * a;
      l += sp[D + 1] * a;
    }
  }
  x[d] = o;
  __syncthreads();
  for (int h = 1; h < D; h <<= 1) {
    float a = 0.f, b = 0.f;
    const bool lo = (d & h) == 0;
    if (lo) {
      a = x[d];
      b = x[d + h];
    }
    __syncthreads();
    if (lo) {
      x[d] = a + b;
      x[d + h] = a - b;
    }
    __syncthreads();
  }
  const float inv_l = l > 0.f ? 1.0f / l : 0.f;
  out[qi * soo + hi * soh + d] =
      from_f<OT>(x[d] * 0.0625f * sign_of(v_signs, d) * inv_l);
}

// ---------------------------------------------------------------------------
// Expand the cached prefix (seq_len - query_len tokens) of each request into
// int8 per-token-head K/V (rotated space) for the WMMA prefill kernel.
// grid (max_ctx_pad, num_kv_heads, num_reqs), block 32.
// Output K/V: [num_reqs * max_ctx_pad, H, D] int8; scales [.., H].
// ---------------------------------------------------------------------------
template <int BITS>
__global__ void __launch_bounds__(32)
    to_int8_kernel(const uint8_t* __restrict__ cache,
                   const int* __restrict__ block_table, int64_t sbt,
                   const int* __restrict__ query_start_loc,
                   const int* __restrict__ seq_lens, int max_ctx_pad,
                   int block_size, int num_phys_blocks, int64_t scb,
                   int64_t sch, int64_t sct, int8_t* __restrict__ k_out,
                   int8_t* __restrict__ v_out, float* __restrict__ ks_out,
                   float* __restrict__ vs_out, int num_kv_heads) {
  using L = Layout<BITS>;
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int req = blockIdx.z;
  const int lane = threadIdx.x;
  const int ctx =
      seq_lens[req] - (query_start_loc[req + 1] - query_start_loc[req]);
  if (t >= ctx || t >= max_ctx_pad) return;
  const int lb = t / block_size;
  int pb = block_table[req * sbt + lb];
  if (pb < 0 || pb >= num_phys_blocks) pb = 0;
  const uint8_t* p = cache + pb * scb + h * sch + (int64_t)(t % block_size) * sct;
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(p + L::OFF_SC);

  float k[DPL], v[DPL];
  const bool rope_lane = lane < R / DPL;
  if (rope_lane) {
    const float kr_s = half_bits_to_f(sc[0]);
    const uint2 u = *reinterpret_cast<const uint2*>(p + lane * DPL);
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      const uint32_t w = i < 4 ? u.x : u.y;
      k[i] = (float)(int8_t)((w >> (8 * (i & 3))) & 0xFF) * kr_s;
    }
  } else {
    const int nl = lane - R / DPL;
    const float kn_s = half_bits_to_f(sc[1 + (nl >> 3)]);
    if constexpr (BITS == 4) {
      const uint32_t w =
          *reinterpret_cast<const uint32_t*>(p + L::OFF_KN + 4 * nl);
  #pragma unroll
      for (int i = 0; i < DPL; ++i)
        k[i] = ((float)((w >> (8 * (i & 3) + 4 * (i >> 2))) & 15) - L::C0) *
               kn_s;
    } else {
      const uint32_t lo =
          *reinterpret_cast<const uint32_t*>(p + L::OFF_KN + 4 * (nl >> 1));
      const uint32_t hi = *reinterpret_cast<const uint32_t*>(
          p + L::OFF_KN + N / 4 + 4 * (nl >> 2));
  #pragma unroll
      for (int i = 0; i < DPL; ++i) {
        const int e16 = (nl & 1) * DPL + i, e32 = (nl & 3) * DPL + i;
        const int c = ((lo >> (8 * (e16 & 3) + 2 * (e16 >> 2))) & 3) |
                      (((hi >> (8 * (e32 & 3) + (e32 >> 2))) & 1) << 2);
        k[i] = ((float)c - L::C0) * kn_s;
      }
    }
  }
  const float v_s = half_bits_to_f(sc[1 + NB]);
  if constexpr (BITS == 4) {
    const uint32_t w = *reinterpret_cast<const uint32_t*>(p + L::OFF_V + 4 * lane);
  #pragma unroll
    for (int i = 0; i < DPL; ++i)
      v[i] = ((float)((w >> (8 * (i & 3) + 4 * (i >> 2))) & 15) - L::C0) * v_s;
  } else {
    const uint32_t lo =
        *reinterpret_cast<const uint32_t*>(p + L::OFF_V + 4 * (lane >> 1));
    const uint32_t hi =
        *reinterpret_cast<const uint32_t*>(p + L::OFF_V + D / 4 + 4 * (lane >> 2));
  #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      const int e16 = (lane & 1) * DPL + i, e32 = (lane & 3) * DPL + i;
      const int c = ((lo >> (8 * (e16 & 3) + 2 * (e16 >> 2))) & 3) |
                    (((hi >> (8 * (e32 & 3) + (e32 >> 2))) & 1) << 2);
      v[i] = ((float)c - L::C0) * v_s;
    }
  }

  float ka = 0.f, va = 0.f;
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    ka = fmaxf(ka, fabsf(k[i]));
    va = fmaxf(va, fabsf(v[i]));
  }
  ka = xor_max<16>(ka);
  va = xor_max<16>(va);
  const float ks = fmaxf(ka, 1e-12f) / 127.0f;
  const float vs = fmaxf(va, 1e-12f) / 127.0f;
  uint32_t kw[2] = {0, 0}, vw[2] = {0, 0};
  #pragma unroll
  for (int i = 0; i < DPL; ++i) {
    const int kq = min(max(__float2int_rn(k[i] / ks), -127), 127);
    const int vq = min(max(__float2int_rn(v[i] / vs), -127), 127);
    kw[i >> 2] |= (uint32_t)(kq & 0xFF) << (8 * (i & 3));
    vw[i >> 2] |= (uint32_t)(vq & 0xFF) << (8 * (i & 3));
  }
  const int64_t row =
      ((int64_t)req * max_ctx_pad + t) * num_kv_heads + h;
  *reinterpret_cast<uint2*>(k_out + row * D + lane * DPL) = make_uint2(kw[0], kw[1]);
  *reinterpret_cast<uint2*>(v_out + row * D + lane * DPL) = make_uint2(vw[0], vw[1]);
  if (lane == 0) {
    ks_out[row] = ks;
    vs_out[row] = vs;
  }
}

}  // namespace splitq

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
using namespace splitq;

static void check_cache(const torch::Tensor& cache, int bits) {
  TORCH_CHECK(cache.dtype() == at::kByte && cache.dim() == 4,
              "splitq: cache must be uint8 [blocks, heads, block_size, slot]");
  const int slot = bits == 4 ? Layout<4>::SLOT : Layout<3>::SLOT;
  TORCH_CHECK(cache.size(3) == slot && cache.stride(3) == 1,
              "splitq: slot size mismatch, expected ", slot, " got ",
              cache.size(3));
}

void splitq_cache_store(torch::Tensor key, torch::Tensor value,
                        torch::Tensor cache, torch::Tensor slot_mapping,
                        torch::Tensor nope_signs, torch::Tensor v_signs,
                        int64_t bits) {
  const int n = slot_mapping.size(0);
  if (n == 0) return;
  check_cache(cache, bits);
  TORCH_CHECK(key.size(-1) == D && key.stride(-1) == 1 && value.stride(-1) == 1,
              "splitq: head_size must be 256 with contiguous last dim");
  TORCH_CHECK(slot_mapping.dtype() == at::kLong);
  const int hkv = cache.size(1);
  const at::cuda::OptionalCUDAGuard guard(device_of(key));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid(n, hkv);
  #define SQ_STORE(B, T)                                                     \
    store_kernel<B, T><<<grid, 32, 0, stream>>>(                             \
        (const T*)key.data_ptr(), (const T*)value.data_ptr(),                \
        (uint8_t*)cache.data_ptr(), slot_mapping.data_ptr<int64_t>(),        \
        nope_signs.data_ptr<int>(), v_signs.data_ptr<int>(), cache.size(2),  \
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),      \
        cache.stride(0), cache.stride(1), cache.stride(2))
  const bool bf = key.dtype() == at::kBFloat16;
  if (bits == 4) {
    if (bf) SQ_STORE(4, __hip_bfloat16); else SQ_STORE(4, half);
  } else {
    if (bf) SQ_STORE(3, __hip_bfloat16); else SQ_STORE(3, half);
  }
  #undef SQ_STORE
}

template <int BITS, int G, int QG, typename QT>
static void launch_decode(torch::Tensor& query, torch::Tensor& cache,
                          torch::Tensor& block_table, torch::Tensor& q_to_req,
                          torch::Tensor& q_to_klen, torch::Tensor& mid_o,
                          torch::Tensor& nope_signs, float sm_scale_log2,
                          int ns, hipStream_t stream) {
  const int num_q = query.size(0);
  dim3 grid(num_q, cache.size(1), ns);
  decode_kernel<BITS, G, QG, QT><<<grid, 128, 0, stream>>>(
      (const QT*)query.data_ptr(), (const uint8_t*)cache.data_ptr(),
      block_table.data_ptr<int>(), q_to_req.data_ptr<int>(),
      q_to_klen.data_ptr<int>(), mid_o.data_ptr<float>(),
      nope_signs.data_ptr<int>(), sm_scale_log2, num_q, cache.size(2),
      block_table.size(1), block_table.size(0), cache.size(0), ns,
      query.stride(0), query.stride(1), cache.stride(0), cache.stride(1),
      cache.stride(2), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2));
}

template <int BITS, typename QT>
static void dispatch_decode(int g, int qg, torch::Tensor& query,
                            torch::Tensor& cache, torch::Tensor& block_table,
                            torch::Tensor& q_to_req, torch::Tensor& q_to_klen,
                            torch::Tensor& mid_o, torch::Tensor& nope_signs,
                            float sl2, int ns, hipStream_t stream) {
  #define SQ_DEC(GG, QQ)                                                    \
    if (g == GG && qg == QQ) {                                              \
      launch_decode<BITS, GG, QQ, QT>(query, cache, block_table, q_to_req,  \
                                      q_to_klen, mid_o, nope_signs, sl2, ns, \
                                      stream);                              \
      return;                                                               \
    }
  SQ_DEC(1, 1) SQ_DEC(1, 4) SQ_DEC(2, 1) SQ_DEC(2, 4) SQ_DEC(4, 1)
  SQ_DEC(4, 4) SQ_DEC(6, 1) SQ_DEC(6, 4) SQ_DEC(8, 1) SQ_DEC(8, 4)
  #undef SQ_DEC
  TORCH_CHECK(false, "splitq_decode: unsupported GQA group ", g);
}

void splitq_decode(torch::Tensor out, torch::Tensor query, torch::Tensor cache,
                   torch::Tensor block_table, torch::Tensor q_to_req,
                   torch::Tensor q_to_klen, torch::Tensor mid_o,
                   torch::Tensor nope_signs, torch::Tensor v_signs,
                   double sm_scale, int64_t num_kv_splits, int64_t bits,
                   int64_t query_group) {
  const int num_q = query.size(0);
  if (num_q == 0) return;
  check_cache(cache, bits);
  TORCH_CHECK(query.size(2) == D && query.stride(2) == 1);
  TORCH_CHECK(out.dtype() == query.dtype());
  TORCH_CHECK(mid_o.size(2) >= num_kv_splits && mid_o.size(3) >= D + 2);
  const int hq = query.size(1), hkv = cache.size(1);
  TORCH_CHECK(hq % hkv == 0);
  const int g = hq / hkv;
  const int qg = query_group > 1 ? 4 : 1;
  const int ns = (int)num_kv_splits;
  const float sl2 = (float)sm_scale * 1.4426950408889634f;
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bool bf = query.dtype() == at::kBFloat16;
  if (bits == 4) {
    if (bf)
      dispatch_decode<4, __hip_bfloat16>(g, qg, query, cache, block_table,
                                         q_to_req, q_to_klen, mid_o,
                                         nope_signs, sl2, ns, stream);
    else
      dispatch_decode<4, half>(g, qg, query, cache, block_table, q_to_req,
                               q_to_klen, mid_o, nope_signs, sl2, ns, stream);
  } else {
    if (bf)
      dispatch_decode<3, __hip_bfloat16>(g, qg, query, cache, block_table,
                                         q_to_req, q_to_klen, mid_o,
                                         nope_signs, sl2, ns, stream);
    else
      dispatch_decode<3, half>(g, qg, query, cache, block_table, q_to_req,
                               q_to_klen, mid_o, nope_signs, sl2, ns, stream);
  }
  dim3 grid2(num_q, hq);
  if (bf)
    reduce_kernel<__hip_bfloat16><<<grid2, 256, 0, stream>>>(
        mid_o.data_ptr<float>(), (__hip_bfloat16*)out.data_ptr(),
        v_signs.data_ptr<int>(), ns, mid_o.stride(0), mid_o.stride(1),
        mid_o.stride(2), out.stride(0), out.stride(1));
  else
    reduce_kernel<half><<<grid2, 256, 0, stream>>>(
        mid_o.data_ptr<float>(), (half*)out.data_ptr(),
        v_signs.data_ptr<int>(), ns, mid_o.stride(0), mid_o.stride(1),
        mid_o.stride(2), out.stride(0), out.stride(1));
}

void splitq_to_int8(torch::Tensor cache, torch::Tensor block_table,
                    torch::Tensor query_start_loc, torch::Tensor seq_lens,
                    int64_t max_ctx_pad, torch::Tensor k_out,
                    torch::Tensor v_out, torch::Tensor k_scale_out,
                    torch::Tensor v_scale_out, int64_t bits) {
  const int num_reqs = seq_lens.size(0);
  if (num_reqs == 0 || max_ctx_pad == 0) return;
  check_cache(cache, bits);
  const int hkv = cache.size(1);
  TORCH_CHECK(k_out.dtype() == at::kChar && k_out.is_contiguous() &&
              v_out.is_contiguous() && k_scale_out.is_contiguous() &&
              v_scale_out.is_contiguous());
  TORCH_CHECK(k_out.numel() >= (int64_t)num_reqs * max_ctx_pad * hkv * D);
  TORCH_CHECK(block_table.stride(1) == 1);
  const at::cuda::OptionalCUDAGuard guard(device_of(cache));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid(max_ctx_pad, hkv, num_reqs);
  #define SQ_I8(B)                                                            \
    to_int8_kernel<B><<<grid, 32, 0, stream>>>(                               \
        (const uint8_t*)cache.data_ptr(), block_table.data_ptr<int>(),        \
        block_table.stride(0), query_start_loc.data_ptr<int>(),               \
        seq_lens.data_ptr<int>(), (int)max_ctx_pad, cache.size(2),            \
        cache.size(0), cache.stride(0), cache.stride(1), cache.stride(2),     \
        (int8_t*)k_out.data_ptr(), (int8_t*)v_out.data_ptr(),                 \
        k_scale_out.data_ptr<float>(), v_scale_out.data_ptr<float>(), hkv)
  if (bits == 4) SQ_I8(4); else SQ_I8(3);
  #undef SQ_I8
}

#else
void splitq_cache_store(torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, torch::Tensor, int64_t) {
  TORCH_CHECK(false, "splitq requires ROCm");
}
void splitq_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, double, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "splitq requires ROCm");
}
void splitq_to_int8(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    int64_t, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, int64_t) {
  TORCH_CHECK(false, "splitq requires ROCm");
}
#endif
