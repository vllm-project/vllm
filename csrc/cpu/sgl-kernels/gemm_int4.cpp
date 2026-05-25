// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include <torch/all.h>

#include "gemm.h"
#include "vec.h"

namespace {

#define BLOCK_N block_size_n()
#define BLOCK_M 128

template <bool sym_quant_act>
struct ActDtype;
template <>
struct ActDtype<true> {
  using type = int8_t;
};
template <>
struct ActDtype<false> {
  using type = uint8_t;
};

#if defined(CPU_CAPABILITY_AVX512)
struct alignas(32) m256i_wrapper {
  __m256i data;
};

inline std::array<m256i_wrapper, 2> load_zps_4vnni(const int8_t* __restrict__ zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask =
      _mm256_set_epi8(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  m256i_wrapper vzps_low_wp, vzps_high_wp;
  vzps_low_wp.data = vzps_low;
  vzps_high_wp.data = vzps_high;
  return {vzps_low_wp, vzps_high_wp};
}

inline std::array<m256i_wrapper, 2> load_uint4_as_int8(const uint8_t* __restrict__ qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i high = _mm256_srli_epi16(packed, 4);
  high = _mm256_and_si256(high, low_mask);
  __m256i low = _mm256_and_si256(packed, low_mask);
  m256i_wrapper low_wp, high_wp;
  low_wp.data = low;
  high_wp.data = high;
  return {low_wp, high_wp};
}

template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(const uint8_t* __restrict__ B, int8_t* dqB, const int8_t* __restrict__ qzeros, int64_t K) {
  // unpack weight int8 -> two int4
  // subtract zero point
  // B shape = [K, ldb] = [K, N / 2], actual shape = [K / 4, N / 2, 4]
  // dqB shape = [K, N], actual shape = [K / 4, N, 4]
#pragma GCC unroll 2
  for (int n = 0; n < N; n += 16) {
    auto [zps_low_wp, zps_high_wp] = load_zps_4vnni(&qzeros[n]);
    auto zps_low = zps_low_wp.data;
    auto zps_high = zps_high_wp.data;
    for (int k = 0; k < K; k += 4) {
      auto [vb_low_wp, vb_high_wp] = load_uint4_as_int8(B + ldb * k + n / 2 * 4);
      auto vb_low = vb_low_wp.data;
      auto vb_high = vb_high_wp.data;
      vb_high = _mm256_sub_epi8(vb_high, zps_high);
      vb_low = _mm256_sub_epi8(vb_low, zps_low);
      // store vb to B
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + n * 4), vb_low);
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + (n + 8) * 4), vb_high);
    }
  }
}

template <bool accum, int64_t N, bool sym_quant_act>
void _dequant_and_store(
    float* __restrict__ output,
    const int32_t* __restrict__ input,
    const float* __restrict__ scale_a,
    const int32_t* __restrict__ zp_a,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ comp_b,
    int M,
    int ldi,
    int ldo,
    int ldsa = 1) {
  for (int m = 0; m < M; ++m) {
    float a_scale = *(scale_a + m * ldsa);
    __m512 va_scale = _mm512_set1_ps(a_scale);
    int32_t a_zp;
    __m512i va_zp;
    if constexpr (!sym_quant_act) {
      a_zp = *(zp_a + m * ldsa);
      va_zp = _mm512_set1_epi32(a_zp);
    }
    int n = 0;
#pragma GCC unroll 2
    for (; n < N; n += 16) {
      __m512i vc = _mm512_loadu_si512(input + m * ldi + n);
      if constexpr (!sym_quant_act) {
        __m512i vb_comp = _mm512_loadu_si512(comp_b + n);
        vc = _mm512_sub_epi32(vc, _mm512_mullo_epi32(vb_comp, va_zp));
      }
      __m512 vc_f = _mm512_cvtepi32_ps(vc);
      __m512 vc_f_mul = _mm512_mul_ps(vc_f, va_scale);
      __m512 vb_s = _mm512_loadu_ps(scale_b + n);
      vc_f_mul = _mm512_mul_ps(vc_f_mul, vb_s);
      if constexpr (accum) {
        __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
        _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f_mul));
      } else {
        _mm512_storeu_ps(output + m * ldo + n, vc_f_mul);
      }
    }
    for (; n < N; ++n) {
      float dq_val;
      if constexpr (sym_quant_act) {
        dq_val = (float)input[m * ldi + n] * a_scale * scale_b[n];
      } else {
        dq_val = (float)(input[m * ldi + n] - a_zp * comp_b[n]) * a_scale * scale_b[n];
      }
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

#else
template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(const uint8_t* B, int8_t* dqB, const int8_t* qzeros, int64_t K) {
  // B shape = [K, N / 2]
  // dqB shape = [K, N]
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N / 2; ++n) {
      int32_t b = (int32_t)B[k * ldb + n];
      dqB[k * N + n * 2] = (b & 0xf) - qzeros[n];
      dqB[k * N + n * 2 + 1] = (b >> 4) - qzeros[n];
    }
  }
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
inline __m512i combine_m256i(__m256i a, __m256i b) {
  __m512i c = _mm512_castsi256_si512(a);
  return _mm512_inserti64x4(c, b, 1);
}

inline __m512i combine_m256i(std::array<m256i_wrapper, 2> two_256) {
  return combine_m256i(two_256[0].data, two_256[1].data);
}

// negate elements in a according to b's sign
static inline __m512i _mm512_sign_epi8(__m512i a, __m512i b) {
  __m512i zero = _mm512_setzero_si512();
  __mmask64 blt0 = _mm512_movepi8_mask(b);
  return _mm512_mask_sub_epi8(a, blt0, zero, a);
}

template <int64_t M, int64_t N, int64_t ldb, bool sym_quant_act>
void _dequant_gemm_accum_small_M(
    float* __restrict__ C,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // if sym_quant_act is true, A pointer type is passed in as uint8_t* but actually int8_t*.

  constexpr int COLS = N / 16;
  // Computing compensation is faster than loading it for small M
  // because it's memory bound.
  __m512i ones = _mm512_set1_epi8(1);  // used for computing compensation
  __m512i va;
  __m512i vb[COLS];
  __m512i vc[M * COLS];
  __m512 vscales[COLS];
  __m512i vzps[COLS];
  __m512i vcompensate[COLS];

  // Load scales and zps
  Unroll<COLS>{}([&](auto i) {
    vscales[i] = _mm512_loadu_ps(scales_b + i * 16);
    vzps[i] = combine_m256i(load_zps_4vnni(qzeros_b + i * 16));
    if constexpr (!sym_quant_act) {
      vcompensate[i] = _mm512_setzero_epi32();
    }
  });
  Unroll<M * COLS>{}([&](auto i) { vc[i] = _mm512_setzero_epi32(); });

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      va = _mm512_set1_epi32(*(int32_t*)(A + row * lda + k));
    }

    if constexpr (row == 0) {
      int B_offset = k * ldb + col * 16 * 2;
      vb[col] = combine_m256i(load_uint4_as_int8(B + B_offset));
      vb[col] = _mm512_sub_epi8(vb[col], vzps[col]);
      if constexpr (!sym_quant_act) {
        vcompensate[col] = _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
      }
      _mm_prefetch(B + B_offset + 128 * ldb, _MM_HINT_T0);
    }
    if constexpr (sym_quant_act) {
      auto vsb = _mm512_sign_epi8(vb[col], va);
      auto vabsa = _mm512_sign_epi8(va, va);
      vc[i] = _mm512_dpbusds_epi32(vc[i], vabsa, vsb);
    } else {
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    }
  };

  // Accumulate along k
  constexpr const int unroll = 4;
  int k = 0;
  for (; k < K / 4 / unroll; k++) {
    Unroll<unroll>{}([&](auto i) { Unroll<M * COLS>{}(compute, 4 * (k * unroll + i)); });
  }
  k *= 4 * unroll;
  for (; k < K; k += 4) {
    Unroll<M * COLS>{}(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;
    // compute (qC - compensate * zp_a) * scale_a * scale_b
    __m512 vc_float;
    if constexpr (!sym_quant_act) {
      vc[i] = _mm512_sub_epi32(vc[i], _mm512_mullo_epi32(vcompensate[col], _mm512_set1_epi32(*(qzeros_a + row))));
    }
    vc_float = _mm512_cvtepi32_ps(vc[i]);
    vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*(scales_a + row)));

    vc_float = _mm512_mul_ps(vc_float, vscales[col]);
    auto vc_old = _mm512_loadu_ps(C + row * ldc + col * 16);
    vc_float = _mm512_add_ps(vc_float, vc_old);
    _mm512_storeu_ps(C + row * ldc + col * 16, vc_float);
  };
  Unroll<M * COLS>{}(store);
}

#define CALL_DEQUANT_GEMM_ACCUM_SMALL_M(M) \
  _dequant_gemm_accum_small_M<M, N, ldb, sym_quant_act>(C, A, scales_a, qzeros_a, B, scales_b, qzeros_b, K, lda, ldc);
#endif

template <int64_t N, int64_t ldb, bool sym_quant_act>
void _dequant_gemm_accum(
    float* C,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int8_t* dqB,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc,
    bool use_brgemm) {
  // Compute GEMM int8 * int8 -> int32
  // dequant result to float by applying scales/qzeros
#if defined(CPU_CAPABILITY_AVX512)
  if (!use_brgemm) {
    switch (M) {
      case 1:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(1);
        break;
      case 2:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(2);
        break;
      case 3:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(3);
        break;
      case 4:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(4);
        break;
      default:
        TORCH_CHECK(false, "tinygemm_kernel: unexpected M for AVX path!");
    }
    return;
  }

  _dequant_weight_zp_only<N, ldb>(B, dqB, qzeros_b, K);
  using Tin = typename ActDtype<sym_quant_act>::type;
  Tin* A_ptr = (Tin*)A;
  if (use_brgemm) {
    int32_t C_i32[M * N];
    at::native::cpublas::brgemm(
        M, N, K, lda, N /*ldb*/, N /*ldc*/, false /* add_C */, A_ptr, dqB, C_i32, true /* is_vnni */);
    _mm_prefetch(B + N * K / 2, _MM_HINT_T0);
    _mm_prefetch(A + K, _MM_HINT_T0);
    _dequant_and_store<true, N, sym_quant_act>(
        C, C_i32, scales_a, qzeros_a, scales_b, compensation, M, N /*ldi*/, ldc, 1 /*ldsa*/);
  } else
#endif
  {
    TORCH_CHECK(false, "tinygemm_kernel: scalar path not implemented!");
  }
}

template <int64_t N>
inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m) {
  if (bias_ptr) {
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 bias_vec = _mm512_loadu_ps(bias_ptr + j);
        _mm512_storeu_ps(y_buf + i * N + j, bias_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = bias_ptr[j];
      }
    }
  } else {  // initialize to zero
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 zero_vec = _mm512_setzero_ps();
        _mm512_storeu_ps(y_buf + i * N + j, zero_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = 0;
      }
    }
  }
}

template <typename out_dtype, int64_t N>
inline void store_out(const float* y_buf, out_dtype* c_ptr, int64_t m, /* int64_t n, */ int64_t lda) {
  for (int i = 0; i < m; ++i) {
    int j = 0;
    if constexpr (std::is_same<out_dtype, float>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = y_buf[i * N + j];
      }
    } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::BFloat16(y_buf[i * N + j]);
      }
    } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::Half(y_buf[i * N + j]);
      }
    } else {
      TORCH_CHECK(false, "Unsupported output dtype");
    }
  }
}

void fill_val_stub(int32_t* __restrict__ output, int32_t value, int64_t size) {
  using iVec = at::vec::Vectorized<int32_t>;
  constexpr int VecSize = iVec::size();
  const iVec fill_val_vec = iVec(value);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - VecSize; d += VecSize) {
    fill_val_vec.store(output + d);
  }
  for (; d < size; ++d) {
    output[d] = value;
  }
}

template <typename act_dtype, typename out_dtype, bool sym_quant_act>
void _da8w4_linear_impl(
    act_dtype* __restrict__ input,
    const float* __restrict__ input_scales,
    const int32_t* __restrict__ input_qzeros,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ weight_qzeros,
    const float* __restrict__ bias,
    out_dtype* __restrict__ output,
    float* __restrict__ output_temp,
    int8_t* __restrict__ dequant_weight_temp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t num_groups) {
  // weight + compensation shape = [Nc, Kc, BLOCK_N * _block_k / 2 + BLOCK_N*sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, BLOCK_N]
  const bool use_brgemm = can_use_brgemm<int8_t>(M);
  int64_t block_m = [&]() -> long {
    if (M <= 48) {
      return M;
    } else if (M < 64) {
      return 32;
    } else if (M < 96) {
      return 64;
    } else {
      return 128;
    }
  }();
  int64_t Mc = div_up(M, block_m);
  bool parallel_on_M = M > 128;
  int64_t Nc = N / BLOCK_N;
  int64_t num_blocks = parallel_on_M ? Mc * Nc : Nc;
  int64_t group_size = div_up(K, num_groups);
  int64_t _block_k = get_4bit_block_k_size(group_size);
  int64_t Kc = K / _block_k;
  int64_t block_per_group = group_size / _block_k;

  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    int tid = get_thread_num();
    float* C_tmp = output_temp + tid * block_m * BLOCK_N;
    int8_t* dqB_tmp = dequant_weight_temp + tid * _block_k * BLOCK_N;
    for (const auto i : c10::irange(begin, end)) {
      int64_t mc = parallel_on_M ? i / Nc : 0;
      int64_t nc = parallel_on_M ? i % Nc : i;
      int64_t mc_end = parallel_on_M ? mc + 1 : Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
        // copy bias to y_buf if bias is not None
        auto bias_data = bias ? bias + nc * BLOCK_N : nullptr;
        copy_bias<BLOCK_N>(bias_data, C_tmp, m_size);
        for (int kci = 0; kci < Kc; ++kci) {
          int32_t* compensation_ptr =
              sym_quant_act
                  ? nullptr
                  : (int32_t*)(void*)(weight + (nc * Kc + kci) * (BLOCK_N * (_block_k / 2 + sizeof(int32_t))) +
                                      _block_k * BLOCK_N / 2) /*Bcomp*/;
          _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2, sym_quant_act>(
              /*C*/ C_tmp,
              /*A*/ (uint8_t*)input + mci * block_m * K + kci * _block_k,
              /*scales_a*/ input_scales + mci * block_m,
              /*qzeros_a*/ input_qzeros + mci * block_m,
              /*B*/ weight + (nc * Kc + kci) * (BLOCK_N * (_block_k / 2 + sizeof(int32_t))),
              /*scales_b*/ weight_scales + nc * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N,
              /*qzeros_b*/ weight_qzeros + nc * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N,
              /*Bcomp*/ compensation_ptr,
              /*dqB_tmp*/ dqB_tmp,
              /*M*/ m_size,
              /*K*/ _block_k,
              /*lda*/ K,
              /*ldc*/ BLOCK_N,
              /*use_brgemm*/ use_brgemm);
        }
        // store y_buf to output with dtype conversion
        store_out<out_dtype, BLOCK_N>(C_tmp, output + mci * block_m * N + nc * BLOCK_N, m_size, N /*lda*/);
      }
    }
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

/*
return: packed_weight, packed_scales, packed_qzeros
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor> convert_int4_weight_packed_with_compensation(
    const at::Tensor& weight, const at::Tensor& scales, const at::Tensor& qzeros) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  // qzeros shape = [N, G]
  TORCH_CHECK(weight.dim() == 2, "DA8W4 CPU: Weight should be a 2D tensor for packing");
  TORCH_CHECK(weight.size(1) % 2 == 0, "DA8W4 CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  auto new_qzeros = qzeros;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  if (new_qzeros.dim() == 1) {
    new_qzeros.unsqueeze_(1);
  }
  new_qzeros = new_qzeros.to(at::kChar);
  int64_t N = weight.size(0);
  int64_t K = weight.size(1);
  int64_t G = scales.size(1);
  int64_t group_size = K / G;
  int64_t _block_k = get_4bit_block_k_size(group_size);
  constexpr int block_n = block_size_n();
  int64_t Nc = N / block_n;
  int64_t Kc = K / _block_k;

  // Reorder weight to [N/block_n, K/_block_k, _block_k, block_n]
  // Reorder scales/qzeros to [N/block_n, G, block_n]
  // weight + compensation shape = [Nc, Kc, block_n * _block_k / 2 + block_n*sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, _block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  at::Tensor blocked_qzeros = new_qzeros.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  // Compensation = Σ(k)(W[k][n] - ZP[n]) for each block.
  auto weight_sub_qzero = weight.view({Nc, block_n, G, -1}).to(at::kInt) - new_qzeros.view({Nc, block_n, G, -1});
  weight_sub_qzero = weight_sub_qzero.view({Nc, block_n, Kc, _block_k});
  at::Tensor compensation = weight_sub_qzero.sum(-1);
  compensation = compensation.permute({0, 2, 1}).contiguous().to(at::kInt);
  int64_t buffer_size_nbytes = _block_k * block_n / 2 + block_n * sizeof(int32_t);
  blocked_weight = at::empty({Nc, Kc, buffer_size_nbytes}, weight.options());

  auto weight_ptr = weight_reordered.data_ptr<uint8_t>();
  auto compensation_ptr = compensation.data_ptr<int32_t>();
  auto blocked_weight_ptr = blocked_weight.data_ptr<uint8_t>();
  int64_t num_blocks = Nc * Kc;
  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      auto in_ptr = weight_ptr + i * _block_k * block_n;
      auto out_ptr = blocked_weight_ptr + i * block_n * (_block_k / 2 + sizeof(int32_t));
      int32_t* comp_in_prt = compensation_ptr + i * block_n;
      int32_t* comp_out_prt = (int32_t*)(void*)(blocked_weight_ptr + i * block_n * (_block_k / 2 + sizeof(int32_t)) +
                                                _block_k * block_n / 2);
      // Reorder weight block to VNNI4 and pack two lanes along N
      // N=16 viewed as two lanes: a0, ...a7, b0, ...b7
      // pack two lanes: [a0, b0], ..., [a7, b7]
      // plain shape = [_block_k, block_n]
      // packed shape = [_block_k / 4, block_n / 2, 4] viewed as [_block_k, block_n / 2]
      constexpr int n_group_size = 8;
      constexpr int vnni_size = 4;
      constexpr int n_group = block_n / n_group_size;  // 4
      for (int nb = 0; nb < n_group; nb += 2) {
        for (int k = 0; k < _block_k; k += vnni_size) {
          for (int ni = 0; ni < n_group_size; ++ni) {
            for (int ki = 0; ki < vnni_size; ++ki) {
              int src_idx_1 = nb * n_group_size + ni + (k + ki) * block_n;
              int src_idx_2 = (nb + 1) * n_group_size + ni + (k + ki) * block_n;
              int dst_idx = (nb / 2 * n_group_size + ni) * vnni_size + k * block_n / 2 + ki;
              uint8_t src_1 = *(in_ptr + src_idx_1);
              uint8_t src_2 = *(in_ptr + src_idx_2);
              uint8_t dst = (src_1 & 0x0f) | ((src_2 & 0x0f) << 4);
              *(out_ptr + dst_idx) = dst;
            }
          }
        }
      }
      // compensation [block_n]
      for (int nb = 0; nb < block_n; nb++) {
        *(comp_out_prt + nb) = *(comp_in_prt + nb);
      }
    }
  });

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales), std::move(blocked_qzeros));
}

std::tuple<at::Tensor, at::Tensor> unpack_4bit_to_32bit_signed(const at::Tensor& qweight, const at::Tensor& qzeros) {
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "qzeros must be int32");
  const auto W0 = qweight.size(0);
  const auto W1 = qweight.size(1);
  const auto Z0 = qzeros.size(0);
  const auto Z1 = qzeros.size(1);

  // unpacked_weights: (W0 * 8, W1), int8
  auto unpacked_weights = at::zeros({W0 * 8, W1}, at::TensorOptions().dtype(at::kChar));
  // unpacked_zeros: (Z0, Z1 * 8), int8
  auto unpacked_zeros = at::zeros({Z0, Z1 * 8}, at::TensorOptions().dtype(at::kChar));

  const int32_t* qw_ptr = qweight.data_ptr<int32_t>();
  const int32_t* qz_ptr = qzeros.data_ptr<int32_t>();
  int8_t* uw_ptr = unpacked_weights.data_ptr<int8_t>();
  int8_t* uz_ptr = unpacked_zeros.data_ptr<int8_t>();

  // ---- unpack qweight ----
  for (int64_t row = 0; row < W0 * 8; ++row) {
    const int i = row & 7;         // row % 8
    const int src_row = row >> 3;  // row // 8
    const int shift = 4 * i;
    for (int64_t col = 0; col < W1; ++col) {
      int32_t v = qw_ptr[src_row * W1 + col];
      uw_ptr[row * W1 + col] = static_cast<int8_t>((v >> shift) & 0xF);
    }
  }
  // ---- unpack qzeros ----
  for (int64_t col = 0; col < Z1 * 8; ++col) {
    const int i = col & 7;
    const int src_col = col >> 3;
    const int shift = 4 * i;

    for (int64_t row = 0; row < Z0; ++row) {
      int32_t v = qz_ptr[row * Z1 + src_col];
      uz_ptr[row * (Z1 * 8) + col] = static_cast<int8_t>((v >> shift) & 0xF);
    }
  }

  return std::make_tuple(unpacked_weights, unpacked_zeros + 1);
}

std::tuple<at::Tensor, at::Tensor>
autogptq_to_int4pack(const at::Tensor& qweight_tensor, const at::Tensor& qzeros_tensor) {
  TORCH_CHECK(qweight_tensor.scalar_type() == at::kInt, "qweight_tensor must be int32");
  TORCH_CHECK(qzeros_tensor.scalar_type() == at::kInt, "qzeros_tensor must be int32");
  TORCH_CHECK(qweight_tensor.is_cpu(), "CPU only implementation");
  if (qweight_tensor.dim() == 3) {
    const int64_t B = qweight_tensor.size(0);
    std::vector<at::Tensor> qweight_list;
    std::vector<at::Tensor> qzeros_list;
    qweight_list.reserve(B);
    qzeros_list.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
      auto outputs = unpack_4bit_to_32bit_signed(qweight_tensor[i], qzeros_tensor[i]);
      at::Tensor unpacked_qweight = std::get<0>(outputs);
      at::Tensor unpacked_qzeros = std::get<1>(outputs);
      qweight_list.push_back(unpacked_qweight.transpose(0, 1).contiguous().to(at::kByte));
      qzeros_list.push_back(unpacked_qzeros.contiguous().to(at::kByte));
    }
    return std::make_tuple(at::stack(qweight_list).detach(), at::stack(qzeros_list).detach());
  }
  auto outputs = unpack_4bit_to_32bit_signed(qweight_tensor, qzeros_tensor);
  at::Tensor unpacked_qweight = std::get<0>(outputs);
  at::Tensor unpacked_qzeros = std::get<1>(outputs);
  at::Tensor return_qweight = unpacked_qweight.transpose(0, 1).contiguous().to(at::kByte);
  at::Tensor return_qzeros = unpacked_qzeros.contiguous().to(at::kByte);
  return std::make_tuple(return_qweight, return_qzeros);
}

std::tuple<at::Tensor, at::Tensor> int4pack(at::Tensor qweight, at::Tensor qzeros, int64_t quant_method_4bit) {
  if (quant_method_4bit == CPUQuantAlgo::AWQ) {
    // autoawq unpacking
    qweight = qweight.contiguous();
    qzeros = qzeros.contiguous();
    // bitshifts: [0, 4, 1, 5, 2, 6, 3, 7] * 4
    auto bitshifts = at::tensor({0, 4, 1, 5, 2, 6, 3, 7}, at::kInt) * 4;
    auto qweight_unsq = qweight.unsqueeze(-1);  // [..., K, N/8, 1]
    auto unpacked = (at::bitwise_right_shift(qweight_unsq, bitshifts) & 0xF).contiguous();
    auto qweight_final = unpacked.flatten(-2).transpose(-1, -2).to(at::kByte).clone();
    auto qzeros_unsq = qzeros.unsqueeze(-1);
    auto qzeros_unpacked = (at::bitwise_right_shift(qzeros_unsq, bitshifts) & 0xF).contiguous();
    auto qzeros_final = qzeros_unpacked.flatten(-2).to(at::kByte).clone();
    return std::make_tuple(qweight_final, qzeros_final);
  } else if (quant_method_4bit == CPUQuantAlgo::GPTQ) {
    // autogptq unpacking
    auto outputs = autogptq_to_int4pack(qweight, qzeros);
    at::Tensor unpacked_qweight = std::get<0>(outputs);
    at::Tensor unpacked_qzeros = std::get<1>(outputs);
    return std::make_tuple(unpacked_qweight, unpacked_qzeros);
  } else {
    TORCH_CHECK(false, "CPU int4 pack only support AWQ or GPTQ...");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convert_weight_packed_scale_zp(
    at::Tensor qweight,  // awq: (*, K, N / 8)  ||  gptq: (*, K / 8, N) , int32
    at::Tensor qzeros,   // awq: (*, K / group_size, N / 8) ||  gptq: (*, K / group_size, N / 8) , int32
    at::Tensor scales,   // awq: (*, K / group_size, N) ||  gptq: (*, K / group_size, N) , bfloat16
    int64_t quant_method_4bit) {
  at::Tensor _qweight;
  at::Tensor _qzeros;

  auto res = int4pack(qweight, qzeros, quant_method_4bit);
  _qweight = std::get<0>(res);
  _qzeros = std::get<1>(res);

  auto _scales = scales;
  _qzeros = _qzeros.transpose(-2, -1).contiguous();  // .T
  _scales = _scales.transpose(-2, -1).contiguous();
  if (_qweight.dim() == 3) {  // Dim=3 for MOE packing, TODO: refine a unified loop
    int64_t E = _qweight.size(0);
    int64_t K = _qweight.size(2);
    int64_t G = _scales.size(2);
    int64_t group_size = K / G;
    int64_t _block_k = get_4bit_block_k_size(group_size);
    int64_t block_n = block_size_n();
    int64_t Nc = _qweight.size(1) / block_n;
    int64_t Kc = K / _block_k;
    int64_t buffer_size_nbytes = _block_k * block_n / 2 + block_n * sizeof(int32_t);
    auto blocked_weight = at::empty({E, Nc, Kc, buffer_size_nbytes}, _qweight.options());
    auto blocked_scales = at::empty({E, Nc, G, block_n}, _scales.options()).to(at::kFloat);
    auto blocked_qzeros = at::empty({E, Nc, G, block_n}, _qzeros.options()).to(at::kChar);
    for (int i = 0; i < _qweight.size(0); i++) {
      auto res_ = convert_int4_weight_packed_with_compensation(_qweight[i], _scales[i], _qzeros[i]);
      blocked_weight[i] = std::get<0>(res_);
      blocked_scales[i] = std::get<1>(res_);
      blocked_qzeros[i] = std::get<2>(res_);
    }
    _qweight = blocked_weight;
    _scales = blocked_scales;
    _qzeros = blocked_qzeros;
  } else {
    auto res_ = convert_int4_weight_packed_with_compensation(_qweight, _scales, _qzeros);
    _qweight = std::get<0>(res_);
    _scales = std::get<1>(res_);
    _qzeros = std::get<2>(res_);
  }

  return std::make_tuple(_qweight, _qzeros, _scales);
}

at::Tensor int4_scaled_mm_cpu_with_quant(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  int64_t M_a = input.size(0);
  int64_t K_a = input.size(1);
  int64_t lda = input.stride(0);

  const auto st = input.scalar_type();
  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf, "int4_scaled_mm_cpu_with_quant: expect A to be bfloat16 or half.");

  constexpr bool sym_quant_act = false;  // TODO: add sym quant path
  using Tin = typename ActDtype<sym_quant_act>::type;
  int64_t act_buffer_size = /* act quant */ M_a * K_a +
                            /* act scale */ M_a * sizeof(float) +
                            /* act zp */ M_a * sizeof(int32_t);
  auto act_buffer = at::empty({act_buffer_size}, input.options().dtype(at::kByte));
  // asym path, activation quants into uint8_t
  auto Aq_data = act_buffer.data_ptr<uint8_t>();
  auto As_data = reinterpret_cast<float*>(Aq_data + M_a * K_a);
  auto Azp_data = reinterpret_cast<int32_t*>(As_data + M_a);
  fill_val_stub(Azp_data, 128, M_a);  // sym_a s8s8 is unified to u8s8 with compensation (128)

  auto out_sizes = input.sizes().vec();
  int64_t N = weight_scales.size(0) * weight_scales.size(-1);
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options());
  // weight + compensation shape = [Nc, Kc, BLOCK_N * _block_k / 2 + BLOCK_N*sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, BLOCK_N]
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t _block_k = K_a / Kc;
  TORCH_CHECK(N == Nc * BLOCK_N, "DA8W4: weight and input shapes mismatch");
  // scales/qzeros shape = [Nc, G, BLOCK_N]
  int64_t num_groups = weight_scales.size(1);

  const uint8_t* b_ptr = weight.data_ptr<uint8_t>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  const int8_t* b_qzeros_ptr = weight_qzeros.data_ptr<int8_t>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
  int num_threads = at::get_num_threads();
  int64_t temp_buffer_size = /* output temp */ num_threads * BLOCK_M * BLOCK_N * sizeof(float) +
                             /*  weight dequant temp */ num_threads * _block_k * BLOCK_N;
  auto c_temp_buffer = at::empty({temp_buffer_size}, input.options().dtype(at::kChar));
  float* c_temp_ptr = (float*)((void*)(c_temp_buffer.data_ptr<int8_t>()));
  int8_t* dqB_temp_ptr = (int8_t*)((void*)(c_temp_ptr + num_threads * BLOCK_M * BLOCK_N));

#define LAUNCH_DA8W4_LINEAR_WITH_QUANT_IMPL(sym_quant_act)                                                 \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                                         \
      at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "int4_scaled_mm_cpu_with_quant", [&] { \
        const scalar_t* __restrict__ A_data = input.data_ptr<scalar_t>();                                  \
        scalar_t* __restrict__ c_ptr = output.data_ptr<scalar_t>();                                        \
        at::parallel_for(0, M_a, 0, [&](int64_t begin, int64_t end) {                                      \
          for (int64_t m = begin; m < end; ++m) {                                                          \
            quantize_row_int8<scalar_t>(Aq_data + m * K_a, As_data[m], A_data + m * lda, K_a);             \
          }                                                                                                \
        });                                                                                                \
        _da8w4_linear_impl<Tin, scalar_t, sym_quant_act>(                                                  \
            Aq_data,                                                                                       \
            As_data,                                                                                       \
            Azp_data,                                                                                      \
            b_ptr,                                                                                         \
            b_scales_ptr,                                                                                  \
            b_qzeros_ptr,                                                                                  \
            bias_ptr,                                                                                      \
            c_ptr,                                                                                         \
            c_temp_ptr,                                                                                    \
            dqB_temp_ptr,                                                                                  \
            M_a,                                                                                           \
            N,                                                                                             \
            K_a,                                                                                           \
            num_groups);                                                                                   \
      });

  LAUNCH_DA8W4_LINEAR_WITH_QUANT_IMPL(sym_quant_act);

  return output;
}
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());
    Vec res = convert_from_float_ext<scalar_t>(x0, x1);
    res.store(out + d);
  }
}

template <typename scalar_t>
void tinygemm_kernel(
    scalar_t* C,
    float* C_temp,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int8_t* dqB_tmp,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc_f,
    int64_t ldc_s,
    bool store_out,
    bool use_brgemm) {
  // TODO: add sym quant act, now only asym
  _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2, false>(
      C_temp, A, scales_a, qzeros_a, B, scales_b, qzeros_b, compensation, dqB_tmp, M, K, lda, ldc_f, use_brgemm);
  if (store_out) {
    // copy from Ctmp to C
    for (int64_t m = 0; m < M; ++m) {
      copy_stub<scalar_t>(C + m * ldc_s, C_temp + m * ldc_f, BLOCK_N);
    }
  }
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      TYPE * C,                             \
      float* C_temp,                        \
      const uint8_t* A,                     \
      const float* scales_a,                \
      const int32_t* qzeros_a,              \
      const uint8_t* B,                     \
      const float* scales_b,                \
      const int8_t* qzeros_b,               \
      const int32_t* compensation,          \
      int8_t* dqB_tmp,                      \
      int64_t M,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldc_f,                        \
      int64_t ldc_s,                        \
      bool store_out,                       \
      bool use_brgemm)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

// int4 gemm dispatch api register
at::Tensor int4_scaled_mm_cpu(
    at::Tensor& x, at::Tensor& w, at::Tensor& w_zeros, at::Tensor& w_scales, std::optional<at::Tensor> bias) {
  return int4_scaled_mm_cpu_with_quant(x, w, w_scales, w_zeros, bias, x.scalar_type());
}
