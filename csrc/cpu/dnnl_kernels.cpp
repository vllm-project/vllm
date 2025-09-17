#include "cpu_types.hpp"
#include "dnnl_helper.h"

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

#if !defined(__aarch64__) || defined(ARM_BF16_SUPPORT)
template <>
struct KernelVecType<c10::BFloat16> {
  using load_vec_type = vec_op::BF16Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};
#endif

template <>
struct KernelVecType<c10::Half> {
#if defined(__powerpc64__) || defined(__s390x__)
  // Power architecture-specific vector type
  using load_vec_type = vec_op::FP32Vec16;
#else
  // Fallback for other architectures
  using load_vec_type = vec_op::FP16Vec16;
#endif
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <bool AZP, typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int32_t* azp,
                                   const int64_t num_tokens,
                                   const int64_t input_stride,
                                   const int64_t hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int64_t vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  const cvt_vec_t inv_scale(1.0 / *scale);
  const cvt_vec_t i8_min_vec(i8_min);
  const cvt_vec_t i8_max_vec(i8_max);

  cvt_vec_t zp_vec;
  if constexpr (AZP) {
    zp_vec = cvt_vec_t(static_cast<float>(*azp));
  }

#pragma omp parallel for
  for (int64_t i = 0; i < num_tokens; ++i) {
    int64_t j = 0;
    const scalar_t* input_ptr = input + i * input_stride;
    int8_t* output_ptr = output + i * hidden_size;
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = elems_fp32 * inv_scale;

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + zp_vec;
      }

      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output_ptr + j);
    }

    load_vec_t elems(input_ptr + j);
    cvt_vec_t elems_fp32(elems);
    elems_fp32 = elems_fp32 * inv_scale;

    if constexpr (AZP) {
      elems_fp32 = elems_fp32 + zp_vec;
    }

    elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
    vec_op::INT8Vec16 elems_int8(elems_fp32);
    elems_int8.save(output_ptr + j, hidden_size - j);
  }
}

template <bool AZP, typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, int32_t* azp,
                                    const int64_t num_tokens,
                                    const int64_t input_stride,
                                    const int64_t hidden_size) {
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  constexpr float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  const cvt_vec_t i8_min_vec(i8_min);
  const cvt_vec_t i8_max_vec(i8_max);

#pragma omp parallel for
  for (int64_t i = 0; i < num_tokens; ++i) {
    cvt_vec_t max_value(std::numeric_limits<float>::lowest());
    cvt_vec_t min_value(std::numeric_limits<float>::max());
    {
      int64_t j = 0;
      const scalar_t* input_ptr = input + i * input_stride;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input_ptr + j);
        cvt_vec_t elems_fp32(elems);
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32);
          min_value = min_value.min(elems_fp32);
        } else {
          max_value = max_value.max(elems_fp32.abs());
        }
      }

      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);

      if (j + vec_elem_num == hidden_size) {
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32);
          min_value = min_value.min(elems_fp32);
        } else {
          max_value = max_value.max(elems_fp32.abs());
        }
      } else {
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32, hidden_size - j);
          min_value = min_value.min(elems_fp32, hidden_size - j);
        } else {
          max_value = max_value.max(elems_fp32.abs(), hidden_size - j);
        }
      }
    }

    float scale_val;
    float azp_val = 0.0f;
    if constexpr (AZP) {
      float max_scalar = max_value.reduce_max();
      float min_scalar = min_value.reduce_min();
      scale_val = (max_scalar - min_scalar) / 255.0f;
      azp_val = std::nearbyint(-128.0f - min_scalar / scale_val);
      azp[i] = azp_val;
      scale[i] = scale_val;
    } else {
      scale_val = max_value.reduce_max() / 127.0f;
      scale[i] = scale_val;
    }

    const cvt_vec_t inv_scale(1.0 / scale_val);
    const cvt_vec_t azp_vec(azp_val);

    {
      int64_t j = 0;
      const scalar_t* input_ptr = input + i * input_stride;
      int8_t* output_ptr = output + i * hidden_size;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input_ptr + j);
        cvt_vec_t elems_fp32(elems);
        elems_fp32 = (elems_fp32 * inv_scale);

        if constexpr (AZP) {
          elems_fp32 = elems_fp32 + azp_vec;
        }
        elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
        vec_op::INT8Vec16 elems_int8(elems_fp32);
        elems_int8.save(output_ptr + j);
      }

      load_vec_t elems(input_ptr + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = (elems_fp32 * inv_scale);

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + azp_vec;
      }
      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output_ptr + j, hidden_size - j);
    }
  }
}

template <bool AZP, bool Bias, typename scalar_t>
void dynamic_quant_epilogue(const float* input, scalar_t* output,
                            const float* a_scale, const int32_t* azp,
                            const float* azp_adj, const scalar_t* bias,
                            const int64_t num_tokens,
                            const int64_t hidden_size) {
  CPU_KERNEL_GUARD_IN(dynamic_quant_epilogue)
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  const int64_t thread_num = omp_get_max_threads();
  if (num_tokens > thread_num) {
#pragma omp parallel for
    for (int64_t i = 0; i < num_tokens; ++i) {
      const float* input_ptr = input + i * hidden_size;
      scalar_t* output_ptr = output + i * hidden_size;
      int64_t j = 0;
      cvt_vec_t token_scale_vec(a_scale[i]);
      cvt_vec_t token_zp_scale_vec;
      if constexpr (AZP) {
        float zp_scale_val = a_scale[i] * static_cast<float>(azp[i]);
        token_zp_scale_vec = cvt_vec_t(zp_scale_val);
      }
      for (; j < hidden_size - vec_elem_num; ++j) {
        cvt_vec_t elems_fp32(input_ptr + j);
        elems_fp32 = elems_fp32 * token_scale_vec;
        if constexpr (AZP) {
          cvt_vec_t azp_adj_fp32(azp_adj + j);
          elems_fp32 = elems_fp32 - azp_adj_fp32 * token_zp_scale_vec;
        }
        if constexpr (Bias) {
          load_vec_t bias_vec(bias + j);
          cvt_vec_t bias_vec_fp32(bias_vec);
          elems_fp32 = elems_fp32 + bias_vec_fp32;
        }
        load_vec_t elems_out(elems_fp32);
        elems_out.save(output_ptr + j);
      }
      cvt_vec_t elems_fp32(input_ptr + j);
      elems_fp32 = elems_fp32 * token_scale_vec;
      if constexpr (AZP) {
        cvt_vec_t azp_adj_fp32(azp_adj + j);
        elems_fp32 = elems_fp32 - azp_adj_fp32 * token_zp_scale_vec;
      }
      if constexpr (Bias) {
        load_vec_t bias_vec(bias + j);
        cvt_vec_t bias_vec_fp32(bias_vec);
        elems_fp32 = elems_fp32 + bias_vec_fp32;
      }
      load_vec_t elems_out(elems_fp32);
      elems_out.save(output_ptr + j, hidden_size - j);
    }
  } else {
    const int64_t vec_iteration =
        (hidden_size + vec_elem_num - 1) / vec_elem_num;
    const int64_t vec_iteration_per_thread =
        (vec_iteration + thread_num - 1) / thread_num;
    const int64_t elem_num_per_thread = vec_iteration_per_thread * vec_elem_num;
#pragma omp parallel for schedule(static, 1)
    for (int64_t i = 0; i < thread_num; ++i) {
      const int64_t start = elem_num_per_thread * i;
      const int64_t end = std::min(hidden_size, elem_num_per_thread + start);
      for (int64_t j = 0; j < num_tokens; ++j) {
        cvt_vec_t token_scale_vec(a_scale[j]);
        cvt_vec_t token_zp_scale_vec;
        if constexpr (AZP) {
          float zp_scale_val = a_scale[j] * static_cast<float>(azp[j]);
          token_zp_scale_vec = cvt_vec_t(zp_scale_val);
        }
        int64_t k = start;
        const float* input_ptr = input + j * hidden_size;
        scalar_t* output_ptr = output + j * hidden_size;
        for (; k < end - vec_elem_num; k += vec_elem_num) {
          cvt_vec_t elems_fp32(input_ptr + k);
          elems_fp32 = elems_fp32 * token_scale_vec;
          if constexpr (AZP) {
            cvt_vec_t azp_adj_fp32(azp_adj + k);
            elems_fp32 = elems_fp32 - azp_adj_fp32 * token_zp_scale_vec;
          }
          if constexpr (Bias) {
            load_vec_t bias_vec(bias + k);
            cvt_vec_t bias_vec_fp32(bias_vec);
            elems_fp32 = elems_fp32 + bias_vec_fp32;
          }
          load_vec_t elems_out(elems_fp32);
          elems_out.save(output_ptr + k);
        }
        if (k < end) {
          cvt_vec_t elems_fp32(input_ptr + k);
          elems_fp32 = elems_fp32 * token_scale_vec;
          if constexpr (AZP) {
            cvt_vec_t azp_adj_fp32(azp_adj + k);
            elems_fp32 = elems_fp32 - azp_adj_fp32 * token_zp_scale_vec;
          }
          if constexpr (Bias) {
            load_vec_t bias_vec(bias + k);
            cvt_vec_t bias_vec_fp32(bias_vec);
            elems_fp32 = elems_fp32 + bias_vec_fp32;
          }
          load_vec_t elems_out(elems_fp32);
          elems_out.save(output_ptr + k, end - k);
        }
      }
    }
  }
}
}  // namespace

int64_t create_onednn_scaled_mm_handler(
    const torch::Tensor& b,         // [IC, OC], column-major
    const torch::Tensor& b_scales,  // [1] or [OC]
    at::ScalarType output_type, bool dynamic_act_quant, bool use_azp,
    int64_t primitive_cache_size) {
  TORCH_CHECK(b.dim() == 2);
  TORCH_CHECK(b.stride(0) == 1);  // Column-major
  TORCH_CHECK(b_scales.is_contiguous());

  W8A8MatMulPrimitiveHandler::Args args;
  args.primitive_cache_size = primitive_cache_size;

  if (b_scales.numel() == 1) {
    args.b_quantization_strategy =
        W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TENSOR;
  } else {
    TORCH_CHECK_EQ(b_scales.numel(), b.size(1));
    args.b_quantization_strategy =
        W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_OUTPUT_CHANNEL;
  }
  args.b_scales_ptr = b_scales.data_ptr<float>();
  args.b_k_size = b.size(0);
  args.b_k_stride = b.stride(0);
  args.b_n_size = b.size(1);
  args.b_n_stride = b.stride(1);
  args.b_ptr = b.data_ptr<int8_t>();

  if (dynamic_act_quant) {
    // dynamic per-token, bias, A scales and A zps will be applied in outside.
    args.a_quantization_strategy =
        W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TOKEN;
    args.use_a_zero_point = false;
  } else {
    // static per-tensor
    args.a_quantization_strategy =
        W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TENSOR;
    args.use_a_zero_point = use_azp;
  }

  VLLM_DISPATCH_FLOATING_TYPES(output_type, "create_onednn_scaled_mm_handler",
                               [&] {
                                 if (dynamic_act_quant) {
                                   args.c_type = get_dnnl_type<float>();
                                 } else {
                                   args.c_type = get_dnnl_type<scalar_t>();
                                 }
                               });

  return reinterpret_cast<int64_t>(new W8A8MatMulPrimitiveHandler(args));
}

void onednn_scaled_mm(
    torch::Tensor& c,                             // [M, OC], row-major
    const torch::Tensor& a,                       // [M, IC], row-major
    const torch::Tensor& a_scales,                // [M] or [1]
    const std::optional<torch::Tensor>& azp,      // [M] or [1]
    const std::optional<torch::Tensor>& azp_adj,  // [M] or [1]
    const std::optional<torch::Tensor>& bias,     // [N]
    int64_t handler) {
  CPU_KERNEL_GUARD_IN(onednn_scaled_mm)
  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(c.is_contiguous());
  W8A8MatMulPrimitiveHandler* ptr =
      reinterpret_cast<W8A8MatMulPrimitiveHandler*>(handler);
  const int32_t* azp_ptr = nullptr;
  if (azp.has_value()) {
    azp_ptr = azp->data_ptr<int32_t>();
  }
  if (ptr->get_input_scale_strategy() ==
      W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TENSOR) {
    TORCH_CHECK_EQ(a_scales.numel(), 1);
  }

  W8A8MatMulPrimitiveHandler::ExecArgs exec_args;
  exec_args.a_ptr = a.data_ptr<int8_t>();
  exec_args.a_m_size = a.size(0);
  exec_args.bias_ptr = nullptr;
  exec_args.bias_type = get_dnnl_type<void>();
  exec_args.use_bias = false;
  exec_args.a_scales_ptr = nullptr;
  exec_args.a_zero_points_ptr = nullptr;

  VLLM_DISPATCH_FLOATING_TYPES(c.scalar_type(), "onednn_scaled_mm", [&] {
    if (ptr->get_input_scale_strategy() ==
        W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TENSOR) {
      if (bias.has_value()) {
        exec_args.bias_ptr = bias->data_ptr<scalar_t>();
        exec_args.bias_type = get_dnnl_type<scalar_t>();
        exec_args.use_bias = true;
      }
      exec_args.a_scales_ptr = a_scales.data_ptr<float>();
      exec_args.a_zero_points_ptr = azp_ptr;
      exec_args.c_ptr = c.data_ptr<scalar_t>();
      ptr->execute(exec_args);
    } else if (ptr->get_input_scale_strategy() ==
               W8A8MatMulPrimitiveHandler::QuantizationStrategy::PER_TOKEN) {
      torch::Tensor tmp_fp32_out =
          torch::empty_like(c, ::at::ScalarType::Float);
      exec_args.c_ptr = tmp_fp32_out.data_ptr<float>();
      ptr->execute(exec_args);
      if (bias.has_value()) {
        if (azp.has_value()) {
          dynamic_quant_epilogue<true, true>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), azp_ptr, azp_adj->data_ptr<float>(),
              bias->data_ptr<scalar_t>(), c.size(0), c.size(1));
        } else {
          dynamic_quant_epilogue<false, true>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), azp_ptr, nullptr,
              bias->data_ptr<scalar_t>(), c.size(0), c.size(1));
        }
      } else {
        if (azp.has_value()) {
          dynamic_quant_epilogue<true, false>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), azp_ptr, azp_adj->data_ptr<float>(),
              (scalar_t*)nullptr, c.size(0), c.size(1));
        } else {
          dynamic_quant_epilogue<false, false>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), azp_ptr, nullptr, (scalar_t*)nullptr,
              c.size(0), c.size(1));
        }
      }
    } else {
      TORCH_CHECK(false, "invalid act quant type.");
    }
  });
}

// static-per-tensor quantization.
void static_scaled_int8_quant(
    torch::Tensor& out,          // [batch, hidden_size]
    const torch::Tensor& input,  // [batch, hidden_size]
    const torch::Tensor& scale, std::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(static_scaled_int8_quant)
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK_EQ(input.dim(), 2);
  TORCH_CHECK_EQ(input.stride(1), 1);
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp.has_value() || azp->numel() == 1);

  const int64_t stride = input.stride(0);
  const int64_t hidden_size = input.size(1);
  const int64_t num_tokens = input.size(0);
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          static_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              stride, hidden_size);
        } else {
          static_scaled_int8_quant_impl<false>(input.data_ptr<scalar_t>(),
                                               out.data_ptr<int8_t>(),
                                               scale.data_ptr<float>(), nullptr,
                                               num_tokens, stride, hidden_size);
        }
      });
}

// dynamic-per-token quantization.
void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [batch, hidden_size]
    const torch::Tensor& input,  // [batch, hidden_size]
    torch::Tensor& scale,        // [batch, 1]
    std::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(dynamic_scaled_int8_quant)
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK_EQ(input.dim(), 2);
  TORCH_CHECK_EQ(input.stride(1), 1);

  const int64_t hidden_size = input.size(1);
  const int64_t num_tokens = input.size(0);
  const int64_t stride = input.stride(0);
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          dynamic_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              stride, hidden_size);
        } else {
          dynamic_scaled_int8_quant_impl<false>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), nullptr, num_tokens, stride,
              hidden_size);
        }
      });
}

int64_t create_onednn_mm_handler(const torch::Tensor& b,
                                 int64_t primitive_cache_size) {
  TORCH_CHECK(b.dim() == 2);

  MatMulPrimitiveHandler::Args args;
  args.primitive_cache_size = primitive_cache_size;

  args.b_k_size = b.size(0);
  args.b_k_stride = b.stride(0);
  args.b_n_size = b.size(1);
  args.b_n_stride = b.stride(1);
  args.b_ptr = b.data_ptr();

  VLLM_DISPATCH_FLOATING_TYPES(b.scalar_type(), "create_onednn_mm_handler",
                               [&] {
                                 args.c_type = get_dnnl_type<scalar_t>();
                                 args.ab_type = get_dnnl_type<scalar_t>();
                               });

  return reinterpret_cast<int64_t>(new MatMulPrimitiveHandler(args));
}

void onednn_mm(torch::Tensor& c,        // [M, OC], row-major
               const torch::Tensor& a,  // [M, IC], row-major
               const std::optional<torch::Tensor>& bias, int64_t handler) {
  CPU_KERNEL_GUARD_IN(onednn_mm)
  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(a.stride(-1) == 1);
  TORCH_CHECK(c.stride(-1) == 1);
  MatMulPrimitiveHandler* ptr =
      reinterpret_cast<MatMulPrimitiveHandler*>(handler);

  MatMulPrimitiveHandler::ExecArgs exec_args;
  exec_args.a_m_size = a.size(0);
  exec_args.a_m_stride = a.stride(0);

  VLLM_DISPATCH_FLOATING_TYPES(a.scalar_type(), "onednn_mm", [&] {
    if (bias.has_value()) {
      exec_args.use_bias = true;
      exec_args.bias_type = get_dnnl_type<scalar_t>();
      exec_args.bias_ptr = bias->data_ptr<scalar_t>();
    } else {
      exec_args.use_bias = false;
      exec_args.bias_type = get_dnnl_type<void>();
      exec_args.bias_ptr = nullptr;
    }
    exec_args.a_ptr = a.data_ptr<scalar_t>();
    exec_args.c_ptr = c.data_ptr<scalar_t>();

    ptr->execute(exec_args);
  });
}
