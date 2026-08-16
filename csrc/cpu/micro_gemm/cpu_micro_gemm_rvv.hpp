#ifndef CPU_MICRO_GEMM_RVV_HPP
#define CPU_MICRO_GEMM_RVV_HPP

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#if defined(__riscv_v)

namespace cpu_micro_gemm {
namespace {

constexpr int32_t RVV_MGEMM_N8 = 8;
constexpr int32_t RVV_MGEMM_B_GROUP_STRIDE = 16;

template <typename scalar_t>
FORCE_INLINE fixed_fp32x8_t load_row8_b_as_f32(const scalar_t* ptr);

template <>
FORCE_INLINE fixed_fp32x8_t load_row8_b_as_f32<float>(const float* ptr) {
  return RVVI(__riscv_vle32_v_f32, LMUL_256)(ptr, RVV_MGEMM_N8);
}

template <>
FORCE_INLINE fixed_fp32x8_t
load_row8_b_as_f32<c10::Half>(const c10::Half* ptr) {
  #if defined(__riscv_zvfh)
  fixed_fp16x8_t vec = RVVI(__riscv_vle16_v_f16, LMUL_128)(
      reinterpret_cast<const _Float16*>(ptr), RVV_MGEMM_N8);
  return RVVI(__riscv_vfwcvt_f_f_v_f32, LMUL_256)(vec, RVV_MGEMM_N8);
  #else
  alignas(32) float values[RVV_MGEMM_N8];
  for (int32_t i = 0; i < RVV_MGEMM_N8; ++i) {
    values[i] = static_cast<float>(ptr[i]);
  }
  return RVVI(__riscv_vle32_v_f32, LMUL_256)(values, RVV_MGEMM_N8);
  #endif
}

template <>
FORCE_INLINE fixed_fp32x8_t
load_row8_b_as_f32<c10::BFloat16>(const c10::BFloat16* ptr) {
  #if defined(__riscv_zvfbfmin)
  fixed_u16x8_t raw = RVVI(__riscv_vle16_v_u16, LMUL_128)(
      reinterpret_cast<const uint16_t*>(ptr), RVV_MGEMM_N8);
  fixed_bf16x8_t vec =
      RVVI4(__riscv_vreinterpret_v_u16, LMUL_128, _bf16, LMUL_128)(raw);
  return RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_256)(vec, RVV_MGEMM_N8);
  #else
  fixed_u16x8_t raw = RVVI(__riscv_vle16_v_u16, LMUL_128)(
      reinterpret_cast<const uint16_t*>(ptr), RVV_MGEMM_N8);
  auto wide = RVVI(__riscv_vzext_vf2_u32, LMUL_256)(raw, RVV_MGEMM_N8);
  auto shifted = RVVI(__riscv_vsll_vx_u32, LMUL_256)(wide, 16, RVV_MGEMM_N8);
  return RVVI4(__riscv_vreinterpret_v_u32, LMUL_256, _f32, LMUL_256)(shifted);
  #endif
}

// Mx8 RVV kernel. B points at one 8-channel half of a 16-channel packed group,
// with rows separated by RVV_MGEMM_B_GROUP_STRIDE scalar elements.
template <int32_t M, typename scalar_t>
FORCE_INLINE void gemm_micro_rvv_fma_mx8_ku4(const scalar_t* __restrict__ a_ptr,
                                             const scalar_t* __restrict__ b_ptr,
                                             float* __restrict__ c_ptr,
                                             const int64_t lda,
                                             const int64_t ldc, const int32_t k,
                                             const bool accum_c) {
  static_assert(0 < M && M <= 8);

  #define RVV_ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
  #define RVV_IF_M(i) if constexpr (M > (i))

  #define RVV_DECL_A(i) const scalar_t* __restrict__ a##i = a_ptr + (i) * lda;
  RVV_ROWS_APPLY(RVV_DECL_A)
  #undef RVV_DECL_A

  #define RVV_DECL_ACC(i) fixed_fp32x8_t acc##i;
  RVV_ROWS_APPLY(RVV_DECL_ACC)
  #undef RVV_DECL_ACC

  #define RVV_INIT_ACC(i)                                                  \
    RVV_IF_M(i) {                                                          \
      if (accum_c) {                                                       \
        acc##i = RVVI(__riscv_vle32_v_f32, LMUL_256)(c_ptr + (i) * ldc,    \
                                                     RVV_MGEMM_N8);        \
      } else {                                                             \
        acc##i = RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.0f, RVV_MGEMM_N8); \
      }                                                                    \
    }
  RVV_ROWS_APPLY(RVV_INIT_ACC)
  #undef RVV_INIT_ACC

  int32_t k_idx = 0;
  for (; k_idx + 3 < k; k_idx += 4) {
  #define RVV_FMA_ROW(i, K_OFFSET)                                     \
    RVV_IF_M(i) {                                                      \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(                  \
          acc##i, static_cast<float>(*(a##i + k_idx + (K_OFFSET))), b, \
          RVV_MGEMM_N8);                                               \
    }

  #define RVV_STEP_K(K_OFFSET)                                      \
    {                                                               \
      fixed_fp32x8_t b = load_row8_b_as_f32<scalar_t>(              \
          b_ptr + (k_idx + (K_OFFSET)) * RVV_MGEMM_B_GROUP_STRIDE); \
      RVV_FMA_ROW(0, K_OFFSET)                                      \
      RVV_FMA_ROW(1, K_OFFSET)                                      \
      RVV_FMA_ROW(2, K_OFFSET)                                      \
      RVV_FMA_ROW(3, K_OFFSET)                                      \
      RVV_FMA_ROW(4, K_OFFSET)                                      \
      RVV_FMA_ROW(5, K_OFFSET)                                      \
      RVV_FMA_ROW(6, K_OFFSET)                                      \
      RVV_FMA_ROW(7, K_OFFSET)                                      \
    }

    RVV_STEP_K(0)
    RVV_STEP_K(1)
    RVV_STEP_K(2)
    RVV_STEP_K(3)
  #undef RVV_STEP_K
  #undef RVV_FMA_ROW
  }

  for (; k_idx < k; ++k_idx) {
    fixed_fp32x8_t b =
        load_row8_b_as_f32<scalar_t>(b_ptr + k_idx * RVV_MGEMM_B_GROUP_STRIDE);
  #define RVV_TAIL_ROW(i)                                                \
    RVV_IF_M(i) {                                                        \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(                    \
          acc##i, static_cast<float>(*(a##i + k_idx)), b, RVV_MGEMM_N8); \
    }
    RVV_ROWS_APPLY(RVV_TAIL_ROW)
  #undef RVV_TAIL_ROW
  }

  #define RVV_STORE_ROW(i)                                           \
    RVV_IF_M(i) {                                                    \
      RVVI(__riscv_vse32_v_f32, LMUL_256)(c_ptr + (i) * ldc, acc##i, \
                                          RVV_MGEMM_N8);             \
    }
  RVV_ROWS_APPLY(RVV_STORE_ROW)
  #undef RVV_STORE_ROW

  #undef RVV_ROWS_APPLY
  #undef RVV_IF_M
}

template <int32_t M, typename scalar_t>
FORCE_INLINE void gemm_micro_rvv_mx32_ku4(DEFINE_CPU_MICRO_GEMM_PARAMS) {
  static_assert(0 < M && M <= 8);
  scalar_t* __restrict__ curr_b_0 = b_ptr;
  scalar_t* __restrict__ curr_b_1 = b_ptr + b_n_group_stride;

  gemm_micro_rvv_fma_mx8_ku4<M>(a_ptr, curr_b_0, c_ptr, lda, ldc, k, accum_c);
  gemm_micro_rvv_fma_mx8_ku4<M>(a_ptr, curr_b_0 + RVV_MGEMM_N8,
                                c_ptr + RVV_MGEMM_N8, lda, ldc, k, accum_c);
  gemm_micro_rvv_fma_mx8_ku4<M>(a_ptr, curr_b_1, c_ptr + 16, lda, ldc, k,
                                accum_c);
  gemm_micro_rvv_fma_mx8_ku4<M>(a_ptr, curr_b_1 + RVV_MGEMM_N8, c_ptr + 24, lda,
                                ldc, k, accum_c);
}

class TileGemmRVV {
 public:
  template <typename scalar_t>
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    switch (m) {
      case 1:
        gemm_micro_rvv_mx32_ku4<1>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 2:
        gemm_micro_rvv_mx32_ku4<2>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 3:
        gemm_micro_rvv_mx32_ku4<3>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 4:
        gemm_micro_rvv_mx32_ku4<4>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 5:
        gemm_micro_rvv_mx32_ku4<5>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 6:
        gemm_micro_rvv_mx32_ku4<6>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 7:
        gemm_micro_rvv_mx32_ku4<7>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 8:
        gemm_micro_rvv_mx32_ku4<8>(CPU_MICRO_GEMM_PARAMS);
        break;
    }
  }
};

}  // namespace

template <typename scalar_t>
class MicroGemm<cpu_utils::ISA::RVV, scalar_t> {
 public:
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 32;

 public:
  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TileGemmRVV::gemm<scalar_t>(CPU_MICRO_GEMM_PARAMS);
  }

  static void pack_weight(const scalar_t* __restrict__ weight,
                          scalar_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK_EQ(output_size % 16, 0);
    for (int32_t o_idx = 0; o_idx < output_size; ++o_idx) {
      const scalar_t* __restrict__ curr_weight = weight + o_idx * input_size;
      scalar_t* __restrict__ curr_packed_weight =
          packed_weight + (o_idx / 16) * (16 * input_size) + o_idx % 16;
      for (int32_t i_idx = 0; i_idx < input_size; ++i_idx) {
        *curr_packed_weight = *curr_weight;

        curr_packed_weight += 16;
        ++curr_weight;
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif  // defined(__riscv_v)

#endif  // CPU_MICRO_GEMM_RVV_HPP
