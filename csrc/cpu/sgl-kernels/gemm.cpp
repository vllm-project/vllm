// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "gemm.h"

#include "common.h"
#include "vec.h"

namespace {

// packed   layout:
//   quants {N, K}  int8_t
//   comp   {N}     int32_t
template <int BLOCK_N>
inline void s8s8_compensation(int8_t* __restrict__ packed, int K) {
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int COLS = BLOCK_N / 16;
  __m512i vcomp[COLS];

  for (int col = 0; col < COLS; ++col) {
    vcomp[col] = _mm512_setzero_si512();
  }

  const int64_t offset = BLOCK_N * K;
  const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
  for (int k = 0; k < K / 4; ++k) {
    for (int col = 0; col < COLS; ++col) {
      __m512i vb = _mm512_loadu_si512((const __m512i*)(packed + k * BLOCK_N * 4 + col * 64));
      vcomp[col] = _mm512_dpbusd_epi32(vcomp[col], off, vb);
    }
  }

  for (int col = 0; col < COLS; ++col) {
    _mm512_storeu_si512((__m512i*)(packed + offset + col * 64), vcomp[col]);
  }
#else
  TORCH_CHECK(false, "s8s8_compensation not implemented!");
#endif
}

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename packed_t>
inline void pack_vnni(packed_t* __restrict__ packed, const packed_t* __restrict__ weight, int N, int K) {
  const int VNNI_BLK = 2;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

template <>
inline void pack_vnni<int8_t>(int8_t* __restrict__ packed, const int8_t* __restrict__ weight, int N, int K) {
  constexpr int BLOCK_N = block_size_n();
  TORCH_CHECK(N == BLOCK_N);

  const int VNNI_BLK = 4;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
  s8s8_compensation<BLOCK_N>(packed, K);
}

// uint8_t: mxfp4 or int4
// pack to vnni2 format as they are computed with bfloat16
//
// from [N, K'/2, 2] to [K'/2, N, 2], view 2x int4 as unit8:
// from [N,    K   ] to [K,    N   ] where K = K'/2
//
template <>
inline void pack_vnni<uint8_t>(uint8_t* __restrict__ packed, const uint8_t* __restrict__ weight, int N, int K) {
  constexpr int BLOCK_N = block_size_n();

  uint8_t unpacked[2 * BLOCK_N];

  // 32-way pack (align with BLOCK_N), faster for avx512 unpacking
  //
  // for a range of (64):
  //   {0, 1, 2, ..., 63}
  //
  // original format:
  //   { 1|0,  3|2, ..., 63|62}
  //
  // packed format:
  //   {32|0, 31|1, ..., 63|31}
  //
  for (int k = 0; k < K; ++k) {
    // unpack first
    for (int n = 0; n < N; ++n) {
      uint8_t value = weight[n * K + k];
      unpacked[n * 2 + 0] = value & 0xF;  // lower 4 bits
      unpacked[n * 2 + 1] = value >> 4;   // higher 4 bits
    }
    // re-pack to 32-way
    for (int n = 0; n < N; ++n) {
      packed[k * N + n] = (unpacked[n + BLOCK_N] << 4) | unpacked[n];
    }
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_stub(float* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0, data1;
    bVec b_vec = bVec::loadu(input + d);
    std::tie(data0, data1) = at::vec::convert_to_float(b_vec);
    data0.store(out + d);
    data1.store(out + d + fVec::size());
  }
  for (; d < size; ++d) {
    out[d] = static_cast<float>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(
    scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

template <typename scalar_t, bool has_bias>
inline void scalar_sigmoid_and_mul(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ mul,
    int SIZE) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  // scalar sigmoid
  const fVec one = fVec(1.f);
  fVec X;
  if constexpr (has_bias) {
    assert(bias != nullptr);
    X = fVec(input[0] + bias[0]);
  } else {
    X = fVec(input[0]);
  }
  X = one / (one + X.neg().exp_u20());

  // vec mul
  constexpr int kVecSize = bVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    bVec m_bvec = bVec::loadu(mul + d);
    fVec m_fvec0, m_fvec1;
    std::tie(m_fvec0, m_fvec1) = at::vec::convert_to_float(m_bvec);
    m_fvec0 = m_fvec0 * X;
    m_fvec1 = m_fvec1 * X;

    bVec out_vec = convert_from_float_ext<scalar_t>(m_fvec0, m_fvec1);
    out_vec.store(out + d);
  }
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_set1_ps(0.f);
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      // for COLS = 1, 3 use 256bit store
      if constexpr (COLS % 2 == 0) {
        if constexpr (col % 2 == 0) {
          _mm512_storeu_si512(
              reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
              (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
        }
      } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + row * ldc + col * 16), (__m256i)(_mm512_cvtneps_pbh(vc[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * 2,                                            \
      C + mb_start * ldc + nb_start,                               \
      has_bias ? bias + nb_start : nullptr,                        \
      K,                                                           \
      lda,                                                         \
      ldb,                                                         \
      ldc);

template <typename scalar_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // copy from Ctmp to C
    for (int64_t m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
  static inline void apply(
      const float* __restrict__ A,
      const float* __restrict__ B,
      scalar_t* __restrict__ C,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);
  }
};

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  if (brg) {
    brgemm<scalar_t, has_bias>::apply(A, B, C, Ctmp, bias, M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-4-16, N = 16, 32, 48, 64
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 16);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 48);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 16);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 48);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 16);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 48);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, " x ", nb_size);
      }
    }
  }
}

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  TORCH_CHECK(brg, "Expected to use fp32 brgemm for small N GEMM");
  if (brg) {
    brgemm<scalar_t, has_bias>::apply(A, B, C, Ctmp, bias, M, N, K, lda, ldb, ldc);
    return;
  }
  // TODO : add intrinsic path
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      // for brgemm, use float32 for accumulate
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

      loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ mat1 + mb_start * mat1_strideM,
            /*   B */ mat2 + nb_start * K /* nb * BLOCK_N * K */,
            /*   C */ out + mb_start * out_strideM + nb_start,
            /* Ctmp*/ Ctmp,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ mat1_strideM,
            /* ldb */ nb_size,
            /* ldc */ out_strideM,
            /* brg */ use_brgemm);
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const float* __restrict__ mat2,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ post_mul_mat,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = true;  // TODO: add intrinsic path
  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      // for brgemm, use float32 for accumulate
      alignas(64) float Atmp[BLOCK_M * K];
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

      loop_2d<float>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);
        for (int64_t m = 0; m < mb_size; ++m) {
          copy_stub<scalar_t>(Atmp + m * K, mat1 + mb_start * mat1_strideM + m * K, K);
        }
        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ Atmp,
            /*   B */ mat2 + nb_start * K /* nb * BLOCK_N * K */,
            /*   C */ out + mb_start * out_strideM + nb_start,
            /* Ctmp*/ Ctmp,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ mat1_strideM,
            /* ldb */ nb_size,
            /* ldc */ out_strideM,
            /* brg */ use_brgemm);

        if (post_mul_mat != nullptr) {
          for (int64_t m = 0; m < mb_size; ++m) {
            scalar_sigmoid_and_mul<scalar_t, has_bias>(
                out + mb_start * out_strideM + nb_start + m * out_strideM,
                Ctmp + m * BLOCK_N,
                bias + nb_start,
                post_mul_mat + mb_start * out_strideM + m * out_strideM,
                out_strideM);
          }
        } else {
          for (int64_t m = 0; m < mb_size; ++m) {
            if constexpr (has_bias) {
              copy_add_stub(
                  out + mb_start * out_strideM + nb_start + m * out_strideM, Ctmp + m * BLOCK_N, bias + nb_start, N);
            } else {
              copy_stub(out + mb_start * out_strideM + nb_start + m * out_strideM, Ctmp + m * BLOCK_N, N);
            }
          }
        }
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

}  // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Ctmp, nullptr, M, N, K, lda, ldb, ldc, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      const TYPE* __restrict__ A,           \
      const TYPE* __restrict__ B,           \
      TYPE* __restrict__ C,                 \
      float* __restrict__ Ctmp,             \
      int64_t M,                            \
      int64_t N,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldb,                          \
      int64_t ldc,                          \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // for 3d moe weights
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_INPUT(weight);

  const int64_t ndim = weight.ndimension();
  TORCH_CHECK(ndim == 2 || ndim == 3, "expect weight to be 2d or 3d, got ", ndim, "d tensor.");

  if (ndim == 2 && weight.size(0) < TILE_N) {
    // for 2D weight and small OC shape, we use fma linear path, which needs transpose not pack
    return weight.to(at::kFloat).t().contiguous();
  }

  const auto st = weight.scalar_type();
  const int64_t E = ndim == 3 ? weight.size(0) : 1;
  const int64_t OC = ndim == 3 ? weight.size(1) : weight.size(0);
  const int64_t IC = ndim == 3 ? weight.size(2) : weight.size(1);

  // mxfp4 or int4 are packed with uint8
  const int64_t actual_IC = st == at::kByte ? IC * 2 : IC;

  // we handle 2 TILE_N at a time.
  TORCH_CHECK(OC % TILE_N == 0, "invalid weight out features ", OC);
  TORCH_CHECK(actual_IC % TILE_K == 0, "invalid weight input features ", actual_IC);

  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t NB = div_up(OC, BLOCK_N);

  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight = at::empty({}, weight.options());
  const int64_t stride = OC * IC;

  // Note: for `kByte` (uint8), it represents either `mxfp4` or `int4`.
  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf || st == at::kChar || st == at::kFloat8_e4m3fn || st == at::kByte,
      "expect weight to be bfloat16, float16, int8, fp8_e4m3 or uint8(mxfp4 or int4).");

  CPU_DISPATCH_PACKED_TYPES(st, [&] {
    // adjust most inner dimension size
    const int packed_row_size = get_row_size<packed_t>(actual_IC);
    auto sizes = weight.sizes().vec();
    sizes[ndim - 1] = packed_row_size;
    packed_weight.resize_(sizes);

    const packed_t* w_data = weight.data_ptr<packed_t>();
    packed_t* packed_data = packed_weight.data_ptr<packed_t>();

    // parallel on {E, NB}
    at::parallel_for(0, E * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t e{0}, nb{0};
      data_index_init(begin, e, E, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);

        int64_t n = nb * BLOCK_N;
        int64_t n_size = std::min(BLOCK_N, OC - n);
        pack_vnni<packed_t>(
            packed_data + e * OC * packed_row_size + n * packed_row_size, w_data + e * stride + n * IC, n_size, IC);

        // move to the next index
        data_index_step(e, E, nb, NB);
      }
    });
  });
  return packed_weight;
}

at::Tensor convert_scale_packed(at::Tensor& scale) {
  CHECK_INPUT(scale);

  const int64_t ndim = scale.ndimension();
  TORCH_CHECK(ndim == 2 || ndim == 3, "expect scale to be 2d or 3d, got ", ndim, "d tensor.");
  const auto st = scale.scalar_type();
  const int64_t E = ndim == 3 ? scale.size(0) : 1;
  const int64_t N = ndim == 3 ? scale.size(1) : scale.size(0);
  // number of groups, e.g. K/32
  const int64_t G = ndim == 3 ? scale.size(2) : scale.size(1);

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(N % BLOCK_N == 0, "invalid weight out features ", N);
  const int64_t NB = N / BLOCK_N;

  auto packed_scale = at::empty_like(scale);
  TORCH_CHECK(st == at::kByte, "expect scale to be uint8.");

  const uint8_t* s_data = scale.data_ptr<uint8_t>();
  uint8_t* packed_data = packed_scale.data_ptr<uint8_t>();

  // parallel on src {E, NB, BLOCK_N, G}, dst {E, NB, G, BLOCK_N}
  at::parallel_for(0, E * NB * BLOCK_N * G, 0, [&](int64_t begin, int64_t end) {
    int64_t e{0}, nb{0}, n{0}, g{0};
    data_index_init(begin, e, E, nb, NB, n, BLOCK_N, g, G);

    for (int64_t i = begin; i < end; ++i) {
      packed_data[e * N * G + nb * G * BLOCK_N + g * BLOCK_N + n] = s_data[i];
      // move to the next index
      data_index_step(e, E, nb, NB, n, BLOCK_N, g, G);
    }
  });
  return packed_scale;
}

// mat1 : [M, K]
// mat2 : [N, K] ([K, N] if use_fma_gemm)
// bias : [N]
// out  : [M, N]
//
at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);
  bool use_fma_gemm = false;
  if (packed_w.scalar_type() == at::kFloat) {
    use_fma_gemm = true;
  }

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = use_fma_gemm ? mat2.size(1) : mat2.size(0);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);
  if (!use_fma_gemm) {
    CHECK_EQ(mat1.size(1), K);
  }

  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, N}, mat1.options());
  // strides
  int64_t out_strideM = out.stride(0);
  int64_t mat1_strideM = mat1.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(dispatch_type, "weight_packed_linear_kernel_impl", [&] {
    if (use_fma_gemm) {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<float>(),
          bias_data,
          nullptr,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    } else {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          bias_data,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    }
  });

  return out;
}

// mat1         : [M, K]
// mat2         : [K, 1]
// post_mul_mat : [M, K]
// bias         : [N]
// out          : [M, N]
//
at::Tensor fused_linear_sigmoid_mul(
    at::Tensor& mat1,
    at::Tensor& mat2,
    const std::optional<at::Tensor>& bias,
    bool is_vnni,
    const at::Tensor& post_mul_mat) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);
  TORCH_CHECK(packed_w.scalar_type() == at::kFloat, "fused_linear_sigmoid_mul requires packed float weight")

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(1);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t out_strideM = post_mul_mat.size(1);
  int64_t mat1_strideM = mat1.stride(0);
  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, out_strideM}, mat1.options());

  TORCH_CHECK(
      N == 1 && out_strideM % 32 == 0,
      "post_mul_mat tensor size(1) should be 32 dividable, and the mat2 OC=1 (Mx1 as linear output shape)")

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(dispatch_type, "fused_linear_sigmoid_mul", [&] {
    weight_packed_linear_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<float>(),
        bias_data,
        post_mul_mat.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1_strideM,
        out_strideM);
  });

  return out;
}
