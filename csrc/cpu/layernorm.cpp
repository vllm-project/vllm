#include "cpu_types.hpp"

namespace {
template <typename scalar_t>
void rms_norm_impl(scalar_t* __restrict__ out,
                   const scalar_t* __restrict__ input,
                   const scalar_t* __restrict__ weight, const bool has_weight,
                   const float epsilon, const int num_tokens,
                   const int hidden_size, const int64_t tokens_per_weight_row,
                   const int64_t weight_stride) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  TORCH_CHECK(hidden_size % VEC_ELEM_NUM == 0);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    vec_op::FP32Vec8 variance(0.0);
    auto input_p = input + i * hidden_size;
    auto output_p = out + i * hidden_size;
    for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM) {
      scalar_vec_t x(input_p + j);
      vec_op::FP32Vec8 fp32_x(x);
      variance = variance + fp32_x * fp32_x;
    }

    float s_variance =
        1.0f / sqrtf(variance.reduce_sum() / (float)hidden_size + epsilon);
    vec_op::FP32Vec8 fp32_s_variance(s_variance);

    for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM) {
      scalar_vec_t x(input_p + j);
      vec_op::FP32Vec8 fp32_x(x);
      vec_op::FP32Vec8 fp32_out;
      if (has_weight) {
        // 2D weight: row = outermost input index (weight_stride 0 for 1D).
        scalar_vec_t w(weight + (i / tokens_per_weight_row) * weight_stride +
                       j);
        vec_op::FP32Vec8 fp32_w(w);
        fp32_out = fp32_x * fp32_s_variance * fp32_w;
      } else {
        fp32_out = fp32_x * fp32_s_variance;
      }

      scalar_vec_t out(fp32_out);
      out.save(output_p + j);
    }
  }
}

template <typename scalar_t>
void fused_add_rms_norm_impl(scalar_t* __restrict__ input,
                             scalar_t* __restrict__ residual,
                             const scalar_t* __restrict__ weight,
                             const bool has_weight, const float epsilon,
                             const int num_tokens, const int hidden_size) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  TORCH_CHECK(hidden_size % VEC_ELEM_NUM == 0);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    vec_op::FP32Vec8 variance(0.0);
    auto input_p = input + i * hidden_size;
    auto residual_p = residual + i * hidden_size;
    for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM) {
      scalar_vec_t x(input_p + j);
      scalar_vec_t res(residual_p + j);
      vec_op::FP32Vec8 fp32_x(x);
      vec_op::FP32Vec8 fp32_res(res);

      fp32_x = fp32_x + fp32_res;
      variance = variance + fp32_x * fp32_x;
      scalar_vec_t out(fp32_x);
      out.save(residual_p + j);
    }

    float s_variance =
        1.0f / sqrtf(variance.reduce_sum() / (float)hidden_size + epsilon);
    vec_op::FP32Vec8 fp32_s_variance(s_variance);

    for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM) {
      vec_op::FP32Vec8 fp32_out;
      if (has_weight) {
        scalar_vec_t w(weight + j);
        scalar_vec_t res(residual_p + j);
        vec_op::FP32Vec8 fp32_w(w);
        vec_op::FP32Vec8 fp32_res(res);
        fp32_out = fp32_res * fp32_s_variance * fp32_w;
      } else {
        scalar_vec_t res(residual_p + j);
        vec_op::FP32Vec8 fp32_res(res);
        fp32_out = fp32_res * fp32_s_variance;
      }

      scalar_vec_t out(fp32_out);
      out.save(input_p + j);
    }
  }
}
}  // namespace

void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const bool has_weight = weight.has_value();
  int64_t weight_stride = 0;
  int64_t tokens_per_weight_row = num_tokens > 0 ? num_tokens : 1;
  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
    if (weight->dim() == 1) {
      TORCH_CHECK(weight->size(0) == hidden_size);
    } else if (weight->dim() == 2) {
      TORCH_CHECK(input.dim() >= 2 && weight->size(0) == input.size(0));
      TORCH_CHECK(weight->size(1) == hidden_size);
      weight_stride = weight->stride(0);
      tokens_per_weight_row =
          input.size(0) > 0 ? num_tokens / input.size(0) : 1;
    } else {
      TORCH_CHECK(false, "rms_norm weight must be 1D or 2D");
    }
  }

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_impl", [&] {
    CPU_KERNEL_GUARD_IN(rms_norm_impl)
    rms_norm_impl(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  has_weight ? weight->data_ptr<scalar_t>() : nullptr,
                  has_weight, epsilon, num_tokens, hidden_size,
                  tokens_per_weight_row, weight_stride);
    CPU_KERNEL_GUARD_OUT(rms_norm_impl)
  });
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        std::optional<torch::Tensor> weight, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  const bool has_weight = weight.has_value();
  if (has_weight) {
    TORCH_CHECK(weight->scalar_type() == input.scalar_type());
    TORCH_CHECK(weight->is_contiguous());
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "fused_add_rms_norm_impl", [&] {
        CPU_KERNEL_GUARD_IN(fused_add_rms_norm_impl)
        fused_add_rms_norm_impl(
            input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
            has_weight ? weight->data_ptr<scalar_t>() : nullptr, has_weight,
            epsilon, num_tokens, hidden_size);
        CPU_KERNEL_GUARD_OUT(fused_add_rms_norm_impl)
      });
}
