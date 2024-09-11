#include "cpu_types.hpp"
#include "dnnl_helper.hpp"

namespace {
template <typename scalar_t>
struct KernelVecType {
  using load_vec_type = void;
  using cvt_vec_type = void;
};

template <>
struct KernelVecType<float> {
  using load_vec_type = vec_op::FP32Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using load_vec_type = vec_op::BF16Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

#ifdef __AVX512F__
template <typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int num_tokens,
                                   const int hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  const cvt_vec_t inv_scale(1.0 / *scale);
  const cvt_vec_t i8_min_vec(i8_min);
  const cvt_vec_t i8_max_vec(i8_max);

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    int j = 0;
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      load_vec_t elems(input + i * hidden_size + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = (elems_fp32 * inv_scale).clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output + i * hidden_size + j);
    }

    load_vec_t elems(input + i * hidden_size + j);
    cvt_vec_t elems_fp32(elems);
    elems_fp32 = (elems_fp32 * inv_scale).clamp(i8_min_vec, i8_max_vec);
    vec_op::INT8Vec16 elems_int8(elems_fp32);

    if (j + vec_elem_num == hidden_size) {
      elems_int8.save(output + i * hidden_size + j);
    } else {
      elems_int8.save(output + i * hidden_size + j, hidden_size - j);
    }
  }
}

template <typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, const int num_tokens,
                                    const int hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    cvt_vec_t max_abs(0.0);
    {
      int j = 0;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input + i * hidden_size + j);
        cvt_vec_t elems_fp32(elems);
        max_abs = max_abs.max(elems_fp32.abs());
      }

      load_vec_t elems(input + i * hidden_size + j);
      cvt_vec_t elems_fp32(elems);

      if (j + vec_elem_num == hidden_size) {
        max_abs = max_abs.max(elems_fp32.abs());
      } else {
        max_abs = max_abs.max(elems_fp32.abs(), hidden_size - j);
      }
    }

    float scale_val = max_abs.reduce_max() / 127.0f;
    scale[i] = scale_val;
    const cvt_vec_t inv_scale(1.0 / scale_val);

    {
      int j = 0;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input + i * hidden_size + j);
        cvt_vec_t elems_fp32(elems);
        elems_fp32 = (elems_fp32 * inv_scale);
        vec_op::INT8Vec16 elems_int8(elems_fp32);
        elems_int8.save(output + i * hidden_size + j);
      }

      load_vec_t elems(input + i * hidden_size + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = (elems_fp32 * inv_scale);
      vec_op::INT8Vec16 elems_int8(elems_fp32);

      if (j + vec_elem_num == hidden_size) {
        elems_int8.save(output + i * hidden_size + j);
      } else {
        elems_int8.save(output + i * hidden_size + j, hidden_size - j);
      }
    }
  }
}

template <bool Bias, typename scalar_t>
void dynamic_output_scale_impl(const float* input, scalar_t* output,
                               const float* scale, const scalar_t* bias,
                               const int num_tokens, const int hidden_size) {
  CPU_KERNEL_GUARD_IN(dynamic_output_scale_impl)
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    int j = 0;
    cvt_vec_t token_scale_vec(scale[i]);
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      cvt_vec_t elems_fp32(input + i * hidden_size + j);
      elems_fp32 = elems_fp32 * token_scale_vec;

      if constexpr (Bias) {
        load_vec_t bias_vec(bias + j);
        cvt_vec_t bias_vec_fp32(bias_vec);
        elems_fp32 = elems_fp32 + bias_vec_fp32;
      }

      load_vec_t elems_out(elems_fp32);
      elems_out.save(output + i * hidden_size + j);
    }

    cvt_vec_t elems_fp32(input + i * hidden_size + j);
    elems_fp32 = elems_fp32 * token_scale_vec;

    if constexpr (Bias) {
      load_vec_t bias_vec(bias + j);
      cvt_vec_t bias_vec_fp32(bias_vec);
      elems_fp32 = elems_fp32 + bias_vec_fp32;
    }

    load_vec_t elems_out(elems_fp32);

    if (j + vec_elem_num == hidden_size) {
      elems_out.save(output + i * hidden_size + j);
    } else {
      elems_out.save(output + i * hidden_size + j, hidden_size - j);
    }
  }
}
#else
template <typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int num_tokens,
                                   const int hidden_size) {
  TORCH_CHECK(false, "static_scaled_int8_quant_impl requires AVX512 support.")
}

template <typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, const int num_tokens,
                                    const int hidden_size) {
  TORCH_CHECK(false, "dynamic_scaled_int8_quant_impl requires AVX512 support.")
}

template <typename scalar_t>
void dynamic_output_scale_impl() {
  TORCH_CHECK(false, "dynamic_output_scale_impl requires AVX512 support.")
}
#endif
}  // namespace

void int8_scaled_mm(torch::Tensor& c,               // [M, OC], row-major
                    const torch::Tensor& a,         // [M, IC], row-major
                    const torch::Tensor& b,         // [IC, OC], column-major
                    const torch::Tensor& a_scales,  // [1] or [M]
                    const torch::Tensor& b_scales,  // [1] or [OC]
                    const c10::optional<torch::Tensor>& bias  // [OC]
) {
  CPU_KERNEL_GUARD_IN(cutlass_scaled_mm)
  // Checks for conformality
  TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8,
              "int8_scaled_mm only supports INT8 inputs.")
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  VLLM_DISPATCH_FLOATING_TYPES(c.scalar_type(), "cutlass_scaled_mm", [&] {
    if (a_scales.numel() != 1) {
      // per-token
      // Note: oneDNN doesn't support per-token activation quantization
      torch::Tensor tmp_fp32_out =
          torch::empty_like(c, ::at::ScalarType::Float);
      DNNLPrimitiveHelper<true>::gemm_s8s8_jit(
          a.data_ptr<int8_t>(), b.data_ptr<int8_t>(),
          tmp_fp32_out.data_ptr<float>(), (void*)(0), a.size(0), b.size(1),
          a.size(1), (float*)(0), b_scales.data_ptr<float>(), 0,
          b_scales.numel());
      if (bias.has_value()) {
        dynamic_output_scale_impl<true>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            a_scales.data_ptr<float>(), bias->data_ptr<scalar_t>(), c.size(0),
            c.size(1));
      } else {
        dynamic_output_scale_impl<false>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            a_scales.data_ptr<float>(), (scalar_t*)(0), c.size(0), c.size(1));
      }
    } else {
      // per-tensor
      if (bias.has_value()) {
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<scalar_t>(),
            bias->data_ptr<scalar_t>(), a.size(0), b.size(1), a.size(1),
            a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            a_scales.numel(), b_scales.numel());
      } else {
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<scalar_t>(),
            (void*)(0), a.size(0), b.size(1), a.size(1),
            a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            a_scales.numel(), b_scales.numel());
      }
    }
  });
}

// static-per-tensor quantization.
void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              const torch::Tensor& input,  // [..., hidden_size]
                              const torch::Tensor& scale) {
  CPU_KERNEL_GUARD_IN(static_scaled_int8_quant)
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_impl", [&] {
        static_scaled_int8_quant_impl(
            input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
            scale.data_ptr<float>(), num_tokens, hidden_size);
      });
}

// dynamic-per-token quantization.
void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    const torch::Tensor& input,  // [..., hidden_size]
    torch::Tensor& scale         // [..., 1]
) {
  CPU_KERNEL_GUARD_IN(dynamic_scaled_int8_quant)
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_impl", [&] {
        dynamic_scaled_int8_quant_impl(
            input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
            scale.data_ptr<float>(), num_tokens, hidden_size);
      });
}
