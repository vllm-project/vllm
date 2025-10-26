// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

#include "common.h"
#include "vec.h"
#include "gemm.h"

// clang-format off

// we use 4x32 for BLOCK_M
#define BLOCK_SIZE_M_SCALE 4

namespace {

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
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
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

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp,
    float scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb; // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);
  const __m512 vd = _mm512_set1_ps(scale);

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

#pragma GCC unroll 4
  for (int k = 0; k < K2; ++k) {
    __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2);
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }

    __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
    __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

    __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
    __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);

    // Apply scale
    __m512 f0_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 0));
    __m512 f0_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 1));
    __m512 f1_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 0));
    __m512 f1_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 1));

    f0_lo = _mm512_mul_ps(f0_lo, vd);
    f0_hi = _mm512_mul_ps(f0_hi, vd);
    f1_lo = _mm512_mul_ps(f1_lo, vd);
    f1_hi = _mm512_mul_ps(f1_hi, vd);

    bf16_0 = _mm512_cvtne2ps_pbh(f0_hi, f0_lo);
    bf16_1 = _mm512_cvtne2ps_pbh(f1_hi, f1_lo);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)bf16_0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)bf16_1);
  }
#else
  TORCH_CHECK(false, "unpack_B: scalar path not implemented!");
#endif
}

template <typename scalar_t, typename packed_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A, const packed_t* __restrict__ B, scalar_t* __restrict__ C,
      const float* __restrict__ bias, const float* __restrict__ scale, int K, int lda, int ldb, int ldc, int64_t block_size_K) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::Float8_e4m3fn* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ bias, const float* __restrict__ scale, int K, int lda, int ldb, int ldc, int64_t block_size_K) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    const int KB = div_up(K, BLOCK_K);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 64;
    constexpr int PREFETCH_SIZE_KB = 1;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vsum[ROWS * COLS];

    // block quant scale
    __m512 vscale;

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_setzero_ps();
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int lda2 = lda >> 1;
    const int ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(a_ptr + row * lda2 + k + PREFETCH_SIZE_K, _MM_HINT_T0);
        }
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          vb[col + 0] = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 0));
          vb[col + 1] = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 1));
        }
      }
      vsum[i] = _mm512_dpbf16_ps(vsum[i], va, vb[col]);
    };

    constexpr int BLOCK_K2 = BLOCK_K >> 1;
    for (int kb = 0; kb < KB; ++kb) {
      int kb_start = kb * BLOCK_K2;
      int kb_end = std::min(K, kb_start + BLOCK_K2);
      // 1. load scale vector
      vscale = _mm512_set1_ps(scale[kb]);
      if constexpr (PREFETCH_SIZE_KB > 0) {
        _mm_prefetch(scale + kb + PREFETCH_SIZE_KB, _MM_HINT_T0);
      }
      // 2. zero vsum for each block
      Unroll<ROWS * COLS>{}([&](auto i) {
        vsum[i] = _mm512_setzero_ps();
      });
      // 3. accumulate across each block
      for (int k = kb_start; k < kb_end; ++k) {
        Unroll<ROWS * COLS>{}(compute, k);
      }
      // 4. apply scale
      Unroll<ROWS * COLS>{}([&](auto i) {
        vc[i] = _mm512_fmadd_ps(vsum[i], vscale, vc[i]);
      });
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2,4 use 512bit store
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, at::Float8_e4m3fn, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, \
        has_bias ? bias + nb_start : nullptr, scale, K, lda, ldb, ldc, block_size_K);

template <typename scalar_t, typename packed_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};

template <bool has_bias>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {

    constexpr int BLOCK_N = block_size_n();

    // [K, BLOCK_N] -> [K / 2, BLOCK_N * 2]
    const int ldb_tmp = BLOCK_N;

    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);

      int idx = k >> 7; // k / BLOCK_K where BLOCK_K = 128
      unpack_B(Btmp + k * ldb_tmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);
    }

    at::native::cpublas::brgemm(
        M, N, K, lda, ldb_tmp, BLOCK_N, /* add_C */ false, A, Btmp, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K) {

  if (brg) {
    brgemm<scalar_t, at::Float8_e4m3fn, has_bias>::apply(
        A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-4-16
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

      switch(mb_size << 4 | nb_size >> 4) {
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void fp8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const at::Float8_e4m3fn* __restrict__ mat2,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    scalar_t* __restrict__ buffer,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    int64_t block_size_N,
    int64_t block_size_K,
    int64_t buffer_size_per_thread) {

  constexpr int64_t BLOCK_M = block_size_m() * BLOCK_SIZE_M_SCALE;
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const int64_t scale_size_K = div_up(K, block_size_K);
  const int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, MB, nb, NB);

      int tid = at::get_thread_num();
      scalar_t* __restrict__ Btmp = buffer + tid * buffer_size_per_thread;
      float* __restrict__ Ctmp = (float*)((void*)(Btmp + BLOCK_N * K));

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);
        const float* scale_ptr = scales2 + (nb / blocks_n_per_group) * scale_size_K;

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A            */ mat1 + mb_start * mat1_strideM,
            /*   B            */ mat2 + nb_start * K, // nb * BLOCK_N * K
            /*   C            */ out + mb_start * out_strideM + nb_start,
            /*   Btmp         */ Btmp,
            /*   Ctmp         */ Ctmp,
            /*   scale        */ scale_ptr,
            /*   bias         */ bias + nb_start,
            /*   M            */ mb_size,
            /*   N            */ nb_size,
            /*   K            */ K,
            /*   lda          */ mat1_strideM,
            /*   ldb          */ nb_size,
            /*   ldc          */ out_strideM,
            /*   brg          */ use_brgemm,
            /*   block_size_K */ block_size_K);

        // move to the next index
        data_index_step(mb, MB, nb, NB);
      }

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

} // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Btmp, Ctmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE)    \
  template void tinygemm_kernel<TYPE>(         \
      const TYPE* __restrict__ A,              \
      const at::Float8_e4m3fn* __restrict__ B, \
      TYPE* __restrict__ C,                    \
      TYPE* __restrict__ Btmp,                 \
      float* __restrict__ Ctmp,                \
      const float* __restrict__ scale,         \
      int64_t M,                               \
      int64_t N,                               \
      int64_t K,                               \
      int64_t lda,                             \
      int64_t ldb,                             \
      int64_t ldc,                             \
      bool brg,                                \
      int64_t block_size_K)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::fp8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, block_size, bias}));

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales2 to be float32.");

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1);

  CHECK_EQ(mat1.size(1), K);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  TORCH_CHECK(block_size.size() == 2,
      "fp8_scaled_mm_cpu: expect block_size.size() to be 2.");

  int64_t block_size_N = block_size[0];
  int64_t block_size_K = block_size[1];

  constexpr int64_t BLOCK_M = block_size_m() * BLOCK_SIZE_M_SCALE;
  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(block_size_N % BLOCK_N == 0, "fp8_scaled_mm_cpu: expect block_size_N to be multiples of BLOCK_N");
  TORCH_CHECK(block_size_K == BLOCK_K, "fp8_scaled_mm_cpu: expect block_size_K equals to BLOCK_K");
  CHECK_EQ(scales2.size(0), div_up(N, block_size_N));
  CHECK_EQ(scales2.size(1), div_up(K, block_size_K));

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "fp8_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype,
      "fp8_scaled_mm_cpu: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kFloat8_e4m3fn,
      "fp8_scaled_mm_cpu: expect mat2 to be fp8_e4m3.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales to be float32.");
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  // strides
  int64_t mat1_strideM = mat1.stride(0);
  int64_t out_strideM = out.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  // Btmp : [T, BLOCK_N * K]
  // Ctmp : [T, BLOCK_M * BLOCK_N]
  int num_threads = at::get_num_threads();
  int64_t size_per_thread = BLOCK_N * K + BLOCK_M * BLOCK_N * 2;
  auto buffer = at::empty({num_threads, size_per_thread}, mat1.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "fp8_scaled_mm_kernel_impl", [&] {
    fp8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<at::Float8_e4m3fn>(),
        scales2.data_ptr<float>(),
        bias_data,
        buffer.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1_strideM,
        out_strideM,
        block_size_N,
        block_size_K,
        size_per_thread);
  });

  return out;
}
