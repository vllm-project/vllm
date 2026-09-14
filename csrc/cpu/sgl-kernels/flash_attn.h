// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#pragma once
#include "common.h"
#include "vec.h"
#include "vec_pack.h"

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    // for COLS = 2, 4 use 512bit store
    if constexpr (col % 2 == 0) {
      auto [a_fvec0, a_fvec1] = load_float_vec2(input + col * 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    auto [a_fvec0, a_fvec1] = load_float_vec2(acc + d);
    a_fvec0 = a_fvec0 * s_fvec;
    a_fvec1 = a_fvec1 * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void copy_stub<at::BFloat16>(at::BFloat16* __restrict__ out, const float* __restrict__ acc, float s, int size) {
  const __m512 vscale = _mm512_set1_ps(s);
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - 32; d += 32) {
    __m512 va0 = _mm512_mul_ps(_mm512_loadu_ps(acc + d), vscale);
    __m512 va1 = _mm512_mul_ps(_mm512_loadu_ps(acc + d + 16), vscale);
    __m512i vb = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
    _mm512_storeu_si512(out + d, vb);
  }
  int remainder = size - d;
  if (remainder > 0) {
    if (remainder <= 16) {
      const __mmask16 vmask = (1ULL << remainder) - 1;
      __m512 va = _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask, acc + d), vscale);
      __m256i vb = (__m256i)(_mm512_cvtneps_pbh(va));
      _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(out + d), vmask, vb);
    } else {  // remainder > 16
      const __mmask16 vmask = (1ULL << (remainder - 16)) - 1;
      __m512 va0 = _mm512_mul_ps(_mm512_loadu_ps(acc + d), vscale);
      __m512 va1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask, acc + d + 16), vscale);
      __m512i vb = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
      const __mmask32 vmask2 = (1ULL << remainder) - 1;
      _mm512_mask_storeu_epi16(reinterpret_cast<__m512i*>(out + d), vmask2, vb);
    }
  }
}
#endif

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct flash_attn_softmax {
  static inline void apply(
      float* __restrict__ s_i,
      scalar_t* __restrict__ s_delta2,
      float* __restrict__ v_prime,
      float* __restrict__ s_prime,
      float* __restrict__ m_prime,
      int m_size,
      int n_size,
      int padded_n_size,
      int head_size_v,
      const float sm_scale,
      int row) {
    using Vec = at::vec::Vectorized<float>;
    const Vec scale_vec = Vec(sm_scale);
    float* s_delta = s_i;
    // s_i <- s_i * scale
    at::vec::map<float>([scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

    // m_i: max value per row
    float m_i =
        at::vec::reduce_all<float>([](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
    m_i = std::max(m_i, m_prime[row]);

    // m_delta <- exp(m' - m_i)
    float m_delta = std::exp(m_prime[row] - m_i);

    // s_delta <- exp(s_i - m_i)
    at::vec::map<float>(
        [m_i](Vec x) { return (x - Vec(m_i)).fexp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

    // s' <- s' * m_delta + sum(s_delta)
    s_prime[row] *= m_delta;
    s_prime[row] += at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

    m_prime[row] = m_i;

    // v' <- v' * m_delta
    at::vec::map<float>(
        [m_delta](Vec x) { return x * Vec(m_delta); },
        v_prime + row * head_size_v,
        v_prime + row * head_size_v,
        head_size_v);

    // Keep s_delta row-major for the following brgemm(P @ V), and only
    // convert the columns that brgemm will consume.
    fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
    copy_stub<scalar_t>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N, 1.f, padded_n_size);
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct flash_attn_softmax<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      float* __restrict__ s_i,
      at::BFloat16* __restrict__ s_delta2,
      float* __restrict__ v_prime,
      float* __restrict__ s_prime,
      float* __restrict__ m_prime,
      int m_size,
      int n_size,
      int padded_n_size,
      int head_size_v,
      const float sm_scale,
      int row) {
    float* s_delta = s_i;
    const __m512 vscale = _mm512_set1_ps(sm_scale);

    int n_remainder = n_size & 15;  // 0xF
    const __mmask16 vmask = (1ULL << n_remainder) - 1;

    int v_remainder = head_size_v & 15;  // 0xF
    const __mmask16 vmask1 = (1ULL << v_remainder) - 1;

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    __m512 va;
    __m256i vb;
    __m512 vmax;
    __m512 vsum;
    __m512 vmdelta;

    const __m512 vneg_inf = _mm512_set1_ps(NEG_INF);

    int m = row;
    vmax = vneg_inf;

    // s_i <- s_i * scale
    int n = 0;
    for (; n <= n_size - 16; n += 16) {
      va = _mm512_mul_ps(_mm512_loadu_ps(s_i + m * BLOCK_N + n), vscale);
      vmax = _mm512_max_ps(va, vmax);
    }
    if (n_remainder > 0) {
      va = _mm512_mul_ps(_mm512_mask_loadu_ps(vneg_inf, vmask, s_i + m * BLOCK_N + n), vscale);
      vmax = _mm512_max_ps(va, vmax);
    }

    // m_i: max value per row
    float m_i = _mm512_reduce_max_ps(vmax);
    m_i = std::max(m_i, m_prime[m]);
    vmax = _mm512_set1_ps(m_i);

    // m_delta <- exp(m' - m_i)
    float m_delta = std::exp(m_prime[m] - m_i);

    // s_delta <- exp(s_i - m_i)
    vsum = _mm512_setzero_ps();
    for (n = 0; n <= n_size - 16; n += 16) {
      va = _mm512_mul_ps(_mm512_loadu_ps(s_i + m * BLOCK_N + n), vscale);
      va = _mm512_fexp_u20_ps(_mm512_sub_ps(va, vmax));
      vsum = _mm512_add_ps(vsum, va);

      vb = (__m256i)(_mm512_cvtneps_pbh(va));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(s_delta2 + m * BLOCK_N + n), vb);
    }
    if (n_remainder > 0) {
      va = _mm512_mul_ps(_mm512_mask_loadu_ps(vneg_inf, vmask, s_i + m * BLOCK_N + n), vscale);
      va = _mm512_fexp_u20_ps(_mm512_sub_ps(va, vmax));
      vsum = _mm512_add_ps(vsum, va);

      vb = (__m256i)(_mm512_cvtneps_pbh(va));
      _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(s_delta2 + m * BLOCK_N + n), vmask, vb);
    }

    // s' <- s' * m_delta + sum(s_delta)
    s_prime[m] *= m_delta;
    s_prime[m] += _mm512_reduce_add_ps(vsum);

    m_prime[m] = m_i;

    // pad s_delta with 0, pad_size range from [0, 32)
    int pad_size = padded_n_size - n_size;
    if (pad_size > 0) {
      const __m512i vzero = _mm512_setzero_si512();
      __mmask32 vmask2 = (1ULL << pad_size) - 1;
      _mm512_mask_storeu_epi16(reinterpret_cast<__m512i*>(s_delta2 + m * BLOCK_N + n_size), vmask2, vzero);
    }

    // v' <- v' * m_delta
    vmdelta = _mm512_set1_ps(m_delta);
    int k = 0;
    for (; k <= head_size_v - 16; k += 16) {
      va = _mm512_mul_ps(_mm512_loadu_ps(v_prime + m * head_size_v + k), vmdelta);
      _mm512_storeu_ps(reinterpret_cast<__m512*>(v_prime + m * head_size_v + k), va);
    }
    if (v_remainder > 0) {
      va = _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask1, v_prime + m * head_size_v + k), vmdelta);
      _mm512_mask_storeu_ps(reinterpret_cast<__m512*>(v_prime + m * head_size_v + k), vmask1, va);
    }
  }
};
#endif
