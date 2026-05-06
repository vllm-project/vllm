#ifndef CPU_MICRO_GEMM_IMPL_HPP
#define CPU_MICRO_GEMM_IMPL_HPP
#include "cpu/utils.hpp"
#include "cpu/cpu_types.hpp"

namespace cpu_micro_gemm {
#define DEFINE_CPU_MICRO_GEMM_PARAMS                                        \
  scalar_t *__restrict__ a_ptr, scalar_t *__restrict__ b_ptr,               \
      float *__restrict__ c_ptr, const int32_t m, const int32_t k,          \
      const int64_t lda, const int64_t b_n_group_stride, const int64_t ldc, \
      const bool accum_c

#define CPU_MICRO_GEMM_PARAMS \
  a_ptr, b_ptr, c_ptr, m, k, lda, b_n_group_stride, ldc, accum_c

// Note: weights for MicroGemm should be packed as (output_size / 16) contiguous
// blocks, means the logical shape of blocks is [16, input_size]. And the actual
// layout of blocks can be ISA-specific.
template <cpu_utils::ISA isa, typename scalar_t>
class MicroGemm {
 public:
  static constexpr int32_t MaxMSize = 16;
  static constexpr int32_t NSize = 16;

 public:
  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TORCH_CHECK(false, "Unimplemented MicroGemm.");
  }
};

template <int32_t n_size, typename scalar_t>
FORCE_INLINE void default_epilogue(float* __restrict__ c_ptr,
                                   scalar_t* __restrict__ d_ptr,
                                   const int32_t m, const int64_t ldc,
                                   const int64_t ldd) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  static_assert(n_size % 16 == 0);

  float* __restrict__ curr_c = c_ptr;
  scalar_t* __restrict__ curr_d = d_ptr;
  for (int32_t i = 0; i < m; ++i) {
    float* __restrict__ curr_c_iter = curr_c;
    scalar_t* __restrict__ curr_d_iter = curr_d;
    vec_op::unroll_loop<int32_t, n_size / 16>([&](int32_t n_g_idx) {
      vec_op::FP32Vec16 c_vec_fp32(curr_c_iter);
      scalar_vec_t c_vec(c_vec_fp32);
      c_vec.save(curr_d_iter);
      curr_c_iter += 16;
      curr_d_iter += 16;
    });
    curr_c += ldc;
    curr_d += ldd;
  }
}

template <int32_t n_size, typename scalar_t>
FORCE_INLINE void bias_epilogue(float* __restrict__ c_ptr,
                                scalar_t* __restrict__ d_ptr,
                                scalar_t* __restrict__ bias_ptr,
                                const int32_t m, const int64_t ldc,
                                const int64_t ldd) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  static_assert(n_size % 16 == 0);
  constexpr int32_t n_group_num = n_size / 16;
  static_assert(n_group_num <= 16);

  vec_op::FP32Vec16 bias_vecs[n_group_num];
  scalar_t* __restrict__ curr_bias = bias_ptr;
  vec_op::unroll_loop<int32_t, n_group_num>([&](int32_t i) {
    scalar_vec_t vec(curr_bias);
    bias_vecs[i] = vec_op::FP32Vec16(vec);
    curr_bias += 16;
  });

  float* __restrict__ curr_c = c_ptr;
  scalar_t* __restrict__ curr_d = d_ptr;
  for (int32_t i = 0; i < m; ++i) {
    float* __restrict__ curr_c_iter = curr_c;
    scalar_t* __restrict__ curr_d_iter = curr_d;
    vec_op::unroll_loop<int32_t, n_group_num>([&](int32_t n_g_idx) {
      vec_op::FP32Vec16 c_vec_fp32(curr_c_iter);
      c_vec_fp32 = c_vec_fp32 + bias_vecs[n_g_idx];
      scalar_vec_t c_vec(c_vec_fp32);
      c_vec.save(curr_d_iter);
      curr_c_iter += 16;
      curr_d_iter += 16;
    });
    curr_c += ldc;
    curr_d += ldd;
  }
}

template <int32_t n_size, typename scalar_t>
FORCE_INLINE void add_bias_epilogue(float* c_ptr, float* d_ptr,
                                    scalar_t* __restrict__ bias_ptr,
                                    const int32_t m, const int64_t ldc,
                                    const int64_t ldd) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  static_assert(n_size % 16 == 0);
  constexpr int32_t n_group_num = n_size / 16;
  static_assert(n_group_num <= 16);

  vec_op::FP32Vec16 bias_vecs[n_group_num];
  scalar_t* __restrict__ curr_bias = bias_ptr;
  vec_op::unroll_loop<int32_t, n_group_num>([&](int32_t i) {
    scalar_vec_t vec(curr_bias);
    bias_vecs[i] = vec_op::FP32Vec16(vec);
    curr_bias += 16;
  });

  float* curr_c = c_ptr;
  float* curr_d = d_ptr;
  for (int32_t i = 0; i < m; ++i) {
    float* curr_c_iter = curr_c;
    float* curr_d_iter = curr_d;
    vec_op::unroll_loop<int32_t, n_group_num>([&](int32_t n_g_idx) {
      vec_op::FP32Vec16 c_vec_fp32(curr_c_iter);
      c_vec_fp32 = c_vec_fp32 + bias_vecs[n_g_idx];
      c_vec_fp32.save(curr_d_iter);
      curr_c_iter += 16;
      curr_d_iter += 16;
    });
    curr_c += ldc;
    curr_d += ldd;
  }
}
}  // namespace cpu_micro_gemm

#endif
