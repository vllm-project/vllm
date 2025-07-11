// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

#include "common.h"
#include "vec.h"
#include "gemm.h"

// clang-format off

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  // no remainder
  #pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <>
inline void copy_stub<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ input, int64_t size) {
  // size might be 64x + 32
  std::memcpy(out, input, size * sizeof(uint8_t));
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int64_t d;
    #pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// out = input + input2 * scale
template <typename scalar_t>
inline void add_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input,
    const scalar_t* __restrict__ input2, float scale, int64_t size) {

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);
  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

/// gemm for w13
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B0, const int8_t* __restrict__ B1, scalar_t* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs0, const float* __restrict__ Bs1,
      const int32_t* __restrict__ Bcomp0, const int32_t* __restrict__ Bcomp1,
      int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B0, const int8_t* __restrict__ B1, at::BFloat16* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs0, const float* __restrict__ Bs1,
      const int32_t* __restrict__ Bcomp0, const int32_t* __restrict__ Bcomp1,
      int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512i va;
    __m512i vb0[COLS];
    __m512i vb1[COLS];
    __m512i vc0[ROWS * COLS];
    __m512i vc1[ROWS * COLS];
    __m512i vcomp0[COLS];
    __m512i vcomp1[COLS];
    __m512  was;
    __m512  vbs0[COLS];
    __m512  vbs1[COLS];

    auto loadc = [&](auto i) {
      vc0[i] = _mm512_set1_epi32(0);
      vc1[i] = _mm512_set1_epi32(0);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb; // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b0_ptr = reinterpret_cast<const int32_t*>(B0);
    const int32_t* b1_ptr = reinterpret_cast<const int32_t*>(B1);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb0[col] = _mm512_loadu_si512(b0_ptr + k * ldb4 + col * 16);
        vb1[col] = _mm512_loadu_si512(b1_ptr + k * ldb4 + col * 16);
      }
      vc0[i] = _mm512_dpbusd_epi32(vc0[i], va, vb0[col]);
      vc1[i] = _mm512_dpbusd_epi32(vc1[i], va, vb1[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto scalec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr(col == 0) {
        was = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp
      if constexpr (row == 0) {
        vbs0[col] = _mm512_loadu_ps(Bs0 + col * 16);
        vbs1[col] = _mm512_loadu_ps(Bs1 + col * 16);
        vcomp0[col] = _mm512_loadu_si512(Bcomp0 + col * 16);
        vcomp1[col] = _mm512_loadu_si512(Bcomp1 + col * 16);
      }
      __m512 c0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc0[i], vcomp0[col]));
      __m512 c1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc1[i], vcomp1[col]));
      vc0[i] = _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(c0, was), vbs0[col]));
      vc1[i] = _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(c1, was), vbs1[col]));
    };
    Unroll<ROWS * COLS>{}(scalec);

    using Vec = at::vec::Vectorized<float>;
    const Vec one = Vec(1.f);
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        Vec x0 = _mm512_castsi512_ps(vc0[row * COLS + col + 0]);
        Vec x1 = _mm512_castsi512_ps(vc0[row * COLS + col + 1]);
        Vec y0 = _mm512_castsi512_ps(vc1[row * COLS + col + 0]);
        Vec y1 = _mm512_castsi512_ps(vc1[row * COLS + col + 1]);
        // silu
        x0 = x0 / (one + x0.neg().exp_u20());
        x1 = x1 / (one + x1.neg().exp_u20());
        // mul
        x0 = x0 * y0;
        x1 = x1 * y1;

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(__m512(x1), __m512(x0))));
        }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_VNNI(MB_SIZE, NB_SIZE)                        \
    tinygemm_kernel_vnni<scalar_t, MB_SIZE, NB_SIZE>::apply(                 \
        A + mb_start * lda, B0 + nb_start * 4, B1 + nb_start * 4,            \
        C + mb_start * ldc + nb_start, As + mb_start,                        \
        Bs0 + nb_start, Bs1 + nb_start, Bcomp0 + nb_start, Bcomp1 + nb_start,\
        K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B0,
    const int8_t* __restrict__ B1,
    scalar_t* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {

  const int32_t* Bcomp0 = reinterpret_cast<const int32_t*>(B0 + block_size_n() * K);
  const int32_t* Bcomp1 = reinterpret_cast<const int32_t*>(B1 + block_size_n() * K);

  // pattern: 1-(2+2)-(8+8)
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        case 0x12: LAUNCH_TINYGEMM_KERNEL_VNNI(1, 32); break;
        case 0x22: LAUNCH_TINYGEMM_KERNEL_VNNI(2, 32); break;
        case 0x32: LAUNCH_TINYGEMM_KERNEL_VNNI(3, 32); break;
        case 0x42: LAUNCH_TINYGEMM_KERNEL_VNNI(4, 32); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

/// gemm for w2
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni2 {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B, float* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs, const int32_t* __restrict__ Bcomp,
      int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B, float* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs, const int32_t* __restrict__ Bcomp,
      int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512i va;
    __m512i vb[COLS];
    __m512i vc[ROWS * COLS];
    __m512i vcomp[COLS];
    __m512  was;
    __m512  vbs[COLS];

    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_epi32(0);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb; // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb[col] = _mm512_loadu_si512(b_ptr + k * ldb4 + col * 16);
      }
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr(col == 0) {
        was = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp per 2 vectors
      // also load bias if any
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          vbs[col + 0] = _mm512_loadu_ps(Bs + col * 16);
          vbs[col + 1] = _mm512_loadu_ps(Bs + col * 16 + 16);
          vcomp[col + 0] = _mm512_loadu_si512(Bcomp + col * 16);
          vcomp[col + 1] = _mm512_loadu_si512(Bcomp + col * 16 + 16);
        }
      }
      __m512 x = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[i], vcomp[col]));
      x = _mm512_mul_ps(_mm512_mul_ps(x, was), vbs[col]);
      _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), x);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_VNNI2(MB_SIZE, NB_SIZE)                       \
    tinygemm_kernel_vnni2<scalar_t, MB_SIZE, NB_SIZE>::apply(                \
        A + mb_start * lda, B + nb_start * 4, C + mb_start * ldc + nb_start, \
        As + mb_start, Bs + nb_start, Bcomp + nb_start,                      \
        K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {

  // B compensation
  const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * K);

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        case 0x12: LAUNCH_TINYGEMM_KERNEL_VNNI2(1, 32); break;
        case 0x22: LAUNCH_TINYGEMM_KERNEL_VNNI2(2, 32); break;
        case 0x32: LAUNCH_TINYGEMM_KERNEL_VNNI2(3, 32); break;
        case 0x42: LAUNCH_TINYGEMM_KERNEL_VNNI2(4, 32); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

} // anonymous namespace

template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {

  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(
          Aq_tmp + m * K,
          As_tmp[m],
          input + m * K,
          K);
    }
  });

  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // strides for w1: [E, 2N, K]
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);

  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;
  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = at::get_thread_num();
    uint8_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;

    alignas(64) float As[BLOCK_M];

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;

      // nb0 from top half and nb1 from bottom half
      int64_t nb0 = nb, nb1 = nb + NB;
      int64_t n_size = std::min(N - nb0 * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const int8_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
      const int8_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
      const float* __restrict__ Bs0 = w1s + expert_id * 2 * N + nb0 * BLOCK_N;
      const float* __restrict__ Bs1 = w1s + expert_id * 2 * N + nb1 * BLOCK_N;

      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, Aq_tmp + index * K, K);
        As[m] = As_tmp[index];
      }

      // fused 1.b: silu_and_mul(A @ B0, A @ B1)
      const int64_t offset = offsets[mb];
      tinygemm_kernel(
          /* A     */ A,
          /* B0    */ B0,
          /* B1    */ B1,
          /* C     */ ic1 + offset * N + nb * BLOCK_N,
          /* As    */ As,
          /* Bs0   */ Bs0,
          /* Bs1   */ Bs1,
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ K,
          /* lda   */ K,
          /* ldb   */ n_size,
          /* ldc   */ N);
    }
  });

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(
          Aq_tmp + m * N,
          As_tmp[m],
          ic1 + m * N,
          N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * packed_N;
  const int64_t stride_oc = packed_N;

  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = at::get_thread_num();
    // we won't be using C1 for gemm2
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;

      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const uint8_t* __restrict__ A = Aq_tmp + offsets[mb] * N;
      const float* __restrict__ As = As_tmp + offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const int8_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs = w2s + expert_id * K + nb * BLOCK_N;

      // 2.a gemm: C = A @ B
      tinygemm_kernel<scalar_t>(
          /* A     */ A,
          /* B     */ B,
          /* C     */ C,
          /* As    */ As,
          /* Bs    */ Bs,
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ IC,
          /* lda   */ IC,
          /* ldb   */ n_size,
          /* ldc   */ BLOCK_N);

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    }
  });

  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#define INSTANTIATE_MOE_INT8_TEMPLATE(TYPE)                                                  \
  template void fused_experts_int8_kernel_impl<TYPE> (                                       \
      TYPE* __restrict__ output, TYPE* __restrict__ ic1,                                     \
      TYPE* __restrict__ ic2, uint8_t* __restrict__ A_tmp,                                   \
      float* __restrict__ C_tmp, uint8_t* __restrict__ Aq_tmp,                               \
      float* __restrict__ As_tmp, const TYPE* __restrict__ input,                            \
      const int8_t* __restrict__ packed_w1, const int8_t* __restrict__ packed_w2,            \
      const float* __restrict__ w1s, const float* __restrict__ w2s,                          \
      const float* __restrict__ topk_weights, const int32_t* __restrict__ sorted_ids,        \
      const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ offsets,           \
      int64_t M, int64_t N, int64_t K, int64_t E, int64_t topk, int64_t num_tokens_post_pad)

INSTANTIATE_MOE_INT8_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_INT8_TEMPLATE(at::Half);

template <typename scalar_t>
void shared_expert_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {

  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(
          Aq_tmp + m * K,
          As_tmp[m],
          input + m * K,
          K);
    }
  });

   // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);
  const int64_t stride_n = packed_K;

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;

      // nb0 from top half and nb1 from bottom half
      int64_t nb0 = nb, nb1 = nb + NB;
      int64_t n_size = std::min(N - nb0 * BLOCK_N, BLOCK_N);
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);

      // A shape [m_size, K]
      const uint8_t* A = Aq_tmp + mb * BLOCK_M * K;
      const float* As = As_tmp + mb * BLOCK_M;

      // B shape [K, n_size] in vnni format
      const int8_t* __restrict__ B0 = packed_w1 + nb0 * BLOCK_N * stride_n;
      const int8_t* __restrict__ B1 = packed_w1 + nb1 * BLOCK_N * stride_n;
      const float* __restrict__ Bs0 = w1s + nb0 * BLOCK_N;
      const float* __restrict__ Bs1 = w1s + nb1 * BLOCK_N;

      // fused 1.b: silu_and_mul(A @ B0, A @ B1)
      tinygemm_kernel(
          /* A     */ A,
          /* B0    */ B0,
          /* B1    */ B1,
          /* C     */ ic1 + mb * BLOCK_M * N + nb * BLOCK_N,
          /* As    */ As,
          /* Bs0   */ Bs0,
          /* Bs1   */ Bs1,
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ K,
          /* lda   */ K,
          /* ldb   */ n_size,
          /* ldc   */ N);
    }
  });

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(
          Aq_tmp + m * N,
          As_tmp[m],
          ic1 + m * N,
          N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = packed_N;

  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = at::get_thread_num();
    // we won't be using C1 for gemm2
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;

      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A shape [m_size, IC]
      const uint8_t* __restrict__ A = Aq_tmp + mb * BLOCK_M * N;
      const float* __restrict__ As = As_tmp + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      const int8_t* __restrict__ B = packed_w2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs = w2s + nb * BLOCK_N;

      // 2.a gemm: C = A @ B
      tinygemm_kernel<scalar_t>(
          /* A     */ A,
          /* B     */ B,
          /* C     */ C,
          /* As    */ As,
          /* Bs    */ Bs,
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ IC,
          /* lda   */ IC,
          /* ldb   */ n_size,
          /* ldc   */ BLOCK_N);

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N;
      for (int64_t m = 0; m < m_size; ++m) {
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, n_size);
      }
    }
  });
}

#define INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(TYPE)                                        \
  template void shared_expert_int8_kernel_impl<TYPE> (                                       \
      TYPE* __restrict__ output, TYPE* __restrict__ ic1,                                     \
      float* __restrict__ C_tmp, uint8_t* __restrict__ Aq_tmp,                               \
      float* __restrict__ As_tmp, const TYPE* __restrict__ input,                            \
      const int8_t* __restrict__ packed_w1, const int8_t* __restrict__ packed_w2,            \
      const float* __restrict__ w1s, const float* __restrict__ w2s,                          \
      const TYPE* __restrict__ fused_experts_out, float routed_scaling_factor,               \
      int64_t M, int64_t N, int64_t K)

INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(at::BFloat16);
INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(at::Half);
