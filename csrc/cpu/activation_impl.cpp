#include "cpu_types.hpp"

namespace {
template <typename scalar_t>
void silu_and_mul_cpu_impl(int num_tokens, int d, scalar_t *__restrict__ input,
                           scalar_t *__restrict__ output) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  TORCH_CHECK(d % VEC_ELEM_NUM == 0);

  const vec_op::FP32Vec8 zeros(0.0);
  const vec_op::FP32Vec8 ones(1.0);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < d; j += VEC_ELEM_NUM) {
      const int start = i * 2 * d;
      const scalar_vec_t x(input + start + j);
      const scalar_vec_t y(input + start + d + j);

      const vec_op::FP32Vec8 f32_x(x.reg);
      const vec_op::FP32Vec8 f32_y(y.reg);

      const vec_op::FP32Vec8 f32_ans =
          f32_y * (f32_x / (ones + (zeros - f32_x).exp()));

      const scalar_vec_t ans(f32_ans.reg);
      ans.save(output + i * d + j);
    }
  }
}
}; // namespace

void silu_and_mul_cpu(torch::Tensor &out, torch::Tensor &input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_cpu_impl", [&] {
        CPU_KERNEL_GUARD_IN(silu_and_mul_cpu_impl)
        silu_and_mul_cpu_impl(num_tokens, d, input.data_ptr<scalar_t>(),
                              out.data_ptr<scalar_t>());
        CPU_KERNEL_GUARD_OUT(silu_and_mul_cpu_impl)
      });
}

void gelu_new_cpu(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(false, "gelu_new is unsupported on CPU.")
}

void gelu_fast_cpu(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(false, "gelu_fast is unsupported on CPU.")
}
