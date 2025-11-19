#ifndef CPU_MICRO_GEMM_VEC_HPP
#define CPU_MICRO_GEMM_VEC_HPP
#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

namespace cpu_micro_gemm {
namespace {
// 8-2-16 pattern, 8 regs for A, 2 regs for B, 16 regs for C, [8, K] @ [k, 32]
template <typename scalar_t>
class TileGemm82 {
 public:
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    switch (m) {
      case 1:
        gemm_micro<1>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 2:
        gemm_micro<2>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 3:
        gemm_micro<3>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 4:
        gemm_micro<4>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 5:
        gemm_micro<5>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 6:
        gemm_micro<6>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 7:
        gemm_micro<7>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 8:
        gemm_micro<8>(CPU_MICRO_GEMM_PARAMS);
        break;
    }
  }

  template <int32_t M>
  static void gemm_micro(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    static_assert(0 < M <= 8);
    using load_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;

    scalar_t* __restrict__ curr_b_0 = b_ptr;
    scalar_t* __restrict__ curr_b_1 = b_ptr + b_n_group_stride;
    float* __restrict__ curr_c_0 = c_ptr;
    float* __restrict__ curr_c_1 = c_ptr + 16;

    vec_op::FP32Vec16 c_regs[M * 2];
    if (accum_c) {
      float* __restrict__ curr_m_c_0 = curr_c_0;
      float* __restrict__ curr_m_c_1 = curr_c_1;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        c_regs[i * 2] = vec_op::FP32Vec16(curr_m_c_0);
        c_regs[i * 2 + 1] = vec_op::FP32Vec16(curr_m_c_1);

        // update
        curr_m_c_0 += ldc;
        curr_m_c_1 += ldc;
      });
    }

    scalar_t* __restrict__ curr_a = a_ptr;
    for (int32_t k_idx = 0; k_idx < k; ++k_idx) {
      load_vec_t b_0_reg(curr_b_0);
      vec_op::FP32Vec16 fp32_b_0_reg(b_0_reg);
      load_vec_t b_1_reg(curr_b_1);
      vec_op::FP32Vec16 fp32_b_1_reg(b_1_reg);

      scalar_t* __restrict__ curr_m_a = curr_a;
      vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
        scalar_t v = *curr_m_a;
        load_vec_t a_reg_original(v);
        vec_op::FP32Vec16 a_reg(a_reg_original);
        c_regs[i * 2] = c_regs[i * 2] + a_reg * fp32_b_0_reg;
        c_regs[i * 2 + 1] = c_regs[i * 2 + 1] + a_reg * fp32_b_1_reg;

        // update
        curr_m_a += lda;
      });

      // update
      curr_a += 1;
      curr_b_0 += 16;
      curr_b_1 += 16;
    }

    vec_op::unroll_loop<int32_t, M>([&](int32_t i) {
      c_regs[i * 2].save(curr_c_0);
      c_regs[i * 2 + 1].save(curr_c_1);

      // update
      curr_c_0 += ldc;
      curr_c_1 += ldc;
    });
  }
};
}  // namespace

// Gemm kernel uses vector instructions, requires B matrix to be packed
template <typename scalar_t>
class MicroGemm<cpu_utils::ISA::VEC, scalar_t> {
 public:
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 32;

 public:
  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TileGemm82<scalar_t>::gemm(CPU_MICRO_GEMM_PARAMS);
  }
};
}  // namespace cpu_micro_gemm

#endif
