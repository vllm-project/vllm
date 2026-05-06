#include "cpu_types.hpp"

namespace {
template <typename scalar_t, vec_op::FP32Vec8 (*func)(const vec_op::FP32Vec8&),
          bool is_gated>
void activation_kernel(int num_tokens, int d, scalar_t* __restrict__ input,
                       scalar_t* __restrict__ output) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  TORCH_CHECK(d % VEC_ELEM_NUM == 0);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < d; j += VEC_ELEM_NUM) {
      int start = i * d;
      if constexpr (is_gated) {
        start *= 2;
      }

      const scalar_vec_t x(input + start + j);
      const vec_op::FP32Vec8 f32_x(x);
      vec_op::FP32Vec8 f32_ans = func(f32_x);

      if constexpr (is_gated) {
        const scalar_vec_t y(input + start + d + j);
        const vec_op::FP32Vec8 f32_y(y);
        f32_ans = f32_y * f32_ans;
      }

      const scalar_vec_t result(f32_ans);
      result.save(output + i * d + j);
    }
  }
}

FORCE_INLINE vec_op::FP32Vec8 silu_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 zeros(0.0);
  const vec_op::FP32Vec8 ones(1.0);
  return x / (ones + (zeros - x).exp());
}

FORCE_INLINE vec_op::FP32Vec8 gelu_new_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 w1(0.79788456f);
  const vec_op::FP32Vec8 w2(0.044715f);
  const vec_op::FP32Vec8 w3(0.5);
  const vec_op::FP32Vec8 x3 = x * x * x;
  const vec_op::FP32Vec8 t = (w1 * (x + w2 * x3)).tanh();
  return w3 * x * (ones + t);
}

FORCE_INLINE vec_op::FP32Vec8 gelu_fast_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 w1(0.79788456f);
  const vec_op::FP32Vec8 w2(0.044715f);
  const vec_op::FP32Vec8 w3(0.5);
  const vec_op::FP32Vec8 t = (x * w1 * (ones + x * w2 * x)).tanh();
  return w3 * x * (ones + t);
}

FORCE_INLINE vec_op::FP32Vec8 gelu_quick_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 zeros(0.0);
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 w1(1.702f);
  return x / (ones + (zeros - w1 * x).exp());
}

FORCE_INLINE vec_op::FP32Vec8 gelu_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 w1(M_SQRT1_2);
  const vec_op::FP32Vec8 w2(0.5);
  return x * w2 * (ones + (x * w1).er());
}

FORCE_INLINE vec_op::FP32Vec8 gelu_tanh_act(const vec_op::FP32Vec8& x) {
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 w1(M_SQRT2 * M_2_SQRTPI * 0.5);
  const vec_op::FP32Vec8 w2(0.5);
  const vec_op::FP32Vec8 w3(0.044715);
  const vec_op::FP32Vec8 x_3 = x * x * x;
  const vec_op::FP32Vec8 inner = w1 * (x + x_3 * w3);
  return x * w2 * (ones + inner.tanh());
}
};  // namespace

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_impl", [&] {
    CPU_KERNEL_GUARD_IN(silu_and_mul_impl)
    activation_kernel<scalar_t, silu_act, true>(
        num_tokens, d, input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    CPU_KERNEL_GUARD_OUT(silu_and_mul_impl)
  });
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul_impl", [&] {
    CPU_KERNEL_GUARD_IN(gelu_and_mul_impl)
    activation_kernel<scalar_t, gelu_act, true>(
        num_tokens, d, input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    CPU_KERNEL_GUARD_OUT(gelu_and_mul_impl)
  });
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "gelu_tanh_and_mul_impl", [&] {
        CPU_KERNEL_GUARD_IN(gelu_tanh_and_mul_impl)
        activation_kernel<scalar_t, gelu_tanh_act, true>(
            num_tokens, d, input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>());
        CPU_KERNEL_GUARD_OUT(gelu_tanh_and_mul_impl)
      });
}

void gelu_new(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1);

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_new_impl", [&] {
    CPU_KERNEL_GUARD_IN(gelu_new_impl)
    activation_kernel<scalar_t, gelu_new_act, false>(
        num_tokens, d, input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    CPU_KERNEL_GUARD_OUT(gelu_new_impl)
  });
}

void gelu_fast(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1);

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_fast_impl", [&] {
    CPU_KERNEL_GUARD_IN(gelu_fast_impl)
    activation_kernel<scalar_t, gelu_fast_act, false>(
        num_tokens, d, input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    CPU_KERNEL_GUARD_OUT(gelu_fast_impl)
  });
}

void gelu_quick(torch::Tensor& out, torch::Tensor& input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1);

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_quick_impl", [&] {
    CPU_KERNEL_GUARD_IN(gelu_quick_impl)
    activation_kernel<scalar_t, gelu_quick_act, false>(
        num_tokens, d, input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    CPU_KERNEL_GUARD_OUT(gelu_quick_impl)
  });
}
