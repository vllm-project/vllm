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

/*

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_and_mul(out, x)
        return out
*/

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

namespace {
template <typename scalar_t>
void gelu_and_mul_cpu_impl(int num_tokens, int d, scalar_t *__restrict__ input,
                           scalar_t *__restrict__ output) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  TORCH_CHECK(d % VEC_ELEM_NUM == 0);

  const vec_op::FP32Vec8 half(0.5);
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 sqrt_two_over_pi(sqrtf(2.0 / M_PI));
  const vec_op::FP32Vec8 gelu_const(0.044715);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < d; j += VEC_ELEM_NUM) {
      const int start = i * 2 * d;
      const scalar_vec_t x(input + start + j);
      const scalar_vec_t y(input + start + d + j);

      const vec_op::FP32Vec8 f32_x(x.reg);
      const vec_op::FP32Vec8 f32_y(y.reg);

      const vec_op::FP32Vec8 f32_ans =
          f32_y * half * f32_x * (ones + (sqrt_two_over_pi * (f32_x + gelu_const * f32_x * f32_x * f32_x)).tanh());

      const scalar_vec_t ans(f32_ans.reg);
      ans.save(output + i * d + j);
    }
  }
}
}

/*

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_and_mul(out, x)
        return out
*/

void gelu_and_mul_cpu(torch::Tensor &out, torch::Tensor &input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "gelu_and_mul_cpu_impl", [&] {
        CPU_KERNEL_GUARD_IN(gelu_and_mul_cpu_impl)
        gelu_and_mul_cpu_impl(num_tokens, d, input.data_ptr<scalar_t>(),
                              out.data_ptr<scalar_t>());
        CPU_KERNEL_GUARD_OUT(gelu_and_mul_cpu_impl)
      });
}

namespace {

template <typename scalar_t>
void gelu_new_cpu_impl(int num_tokens, int d, scalar_t *__restrict__ input,
                           scalar_t *__restrict__ output) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  TORCH_CHECK(d % VEC_ELEM_NUM == 0);

  const vec_op::FP32Vec8 half(0.5);
  const vec_op::FP32Vec8 ones(1.0);
  const vec_op::FP32Vec8 sqrt_two_over_pi(sqrtf(2.0 / M_PI));
  const vec_op::FP32Vec8 gelu_const(0.044715);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < d; j += VEC_ELEM_NUM) {
      const int start = i * d;
      const scalar_vec_t x(input + start + j);

      const vec_op::FP32Vec8 f32_x(x.reg);

      const vec_op::FP32Vec8 f32_ans =
          half * f32_x * (ones + (sqrt_two_over_pi * (f32_x + gelu_const * f32_x * f32_x * f32_x)).tanh());

      const scalar_vec_t ans(f32_ans.reg);
      ans.save(output + i * d + j);
    }
  }
}
}

/*
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c *
                                           (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.gelu_new(out, x)
        return out
*/

void gelu_new_cpu(torch::Tensor &out, torch::Tensor &input) {
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "gelu_new_cpu_impl", [&] {
        CPU_KERNEL_GUARD_IN(gelu_new_cpu_impl)
        gelu_new_cpu_impl(num_tokens, d, input.data_ptr<scalar_t>(),
                          out.data_ptr<scalar_t>());
        CPU_KERNEL_GUARD_OUT(gelu_new_cpu_impl)
      });
}

void gelu_fast_cpu(torch::Tensor &out, torch::Tensor &input) {
  gelu_new_cpu(out, input);
}
