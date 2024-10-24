#include "cpu_types.hpp"
#include "dnnl_helper.hpp"

namespace {
template <typename scalar_t>
struct KernelVecType {
  using load_vec_type = void;
  using azp_adj_load_vec_type = void;
  using cvt_vec_type = void;
};

template <>
struct KernelVecType<float> {
  using load_vec_type = vec_op::FP32Vec16;
  using azp_adj_load_vec_type = vec_op::INT32Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using load_vec_type = vec_op::BF16Vec16;
  using azp_adj_load_vec_type = vec_op::INT32Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::Half> {
  using load_vec_type = vec_op::FP16Vec16;
  using azp_adj_load_vec_type = vec_op::INT32Vec16;
  using cvt_vec_type = vec_op::FP32Vec16;
};

#ifdef __AVX512F__
template <bool AZP, typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int32_t* azp,
                                   const int num_tokens,
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

  cvt_vec_t zp_vec;
  if constexpr (AZP) {
    zp_vec = cvt_vec_t(static_cast<float>(*azp));
  }

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    int j = 0;
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      load_vec_t elems(input + i * hidden_size + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = elems_fp32 * inv_scale;

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + zp_vec;
      }

      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output + i * hidden_size + j);
    }

    load_vec_t elems(input + i * hidden_size + j);
    cvt_vec_t elems_fp32(elems);
    elems_fp32 = elems_fp32 * inv_scale;

    if constexpr (AZP) {
      elems_fp32 = elems_fp32 + zp_vec;
    }

    elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
    vec_op::INT8Vec16 elems_int8(elems_fp32);
    elems_int8.save(output + i * hidden_size + j, hidden_size - j);
  }
}

template <bool AZP, typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, int32_t* azp,
                                    const int num_tokens,
                                    const int hidden_size) {
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
  for (int i = 0; i < num_tokens; ++i) {
    cvt_vec_t max_value(std::numeric_limits<float>::lowest());
    cvt_vec_t min_value(std::numeric_limits<float>::max());
    {
      int j = 0;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input + i * hidden_size + j);
        cvt_vec_t elems_fp32(elems);
        if constexpr (AZP) {
          max_value = max_value.max(elems_fp32);
          min_value = min_value.min(elems_fp32);
        } else {
          max_value = max_value.max(elems_fp32.abs());
        }
      }

      load_vec_t elems(input + i * hidden_size + j);
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

    float scale_val, azp_val;
    if constexpr (AZP) {
      float max_scalar = max_value.reduce_max();
      float min_scalar = min_value.reduce_min();
      scale_val = (max_scalar - min_scalar) / 255.0f;
      azp_val = std::nearbyint(-128.0f - min_scalar / scale_val);
      azp[i] = static_cast<int32_t>(azp_val);
      scale[i] = scale_val;
    } else {
      scale_val = max_value.reduce_max() / 127.0f;
      scale[i] = scale_val;
    }

    const cvt_vec_t inv_scale(1.0 / scale_val);
    const cvt_vec_t azp_vec(azp_val);

    {
      int j = 0;
      for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
        load_vec_t elems(input + i * hidden_size + j);
        cvt_vec_t elems_fp32(elems);
        elems_fp32 = (elems_fp32 * inv_scale);

        if constexpr (AZP) {
          elems_fp32 = elems_fp32 + azp_vec;
        }
        elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
        vec_op::INT8Vec16 elems_int8(elems_fp32);
        elems_int8.save(output + i * hidden_size + j);
      }

      load_vec_t elems(input + i * hidden_size + j);
      cvt_vec_t elems_fp32(elems);
      elems_fp32 = (elems_fp32 * inv_scale);

      if constexpr (AZP) {
        elems_fp32 = elems_fp32 + azp_vec;
      }
      elems_fp32 = elems_fp32.clamp(i8_min_vec, i8_max_vec);
      vec_op::INT8Vec16 elems_int8(elems_fp32);
      elems_int8.save(output + i * hidden_size + j, hidden_size - j);
    }
  }
}

template <bool PerChannel, typename scalar_t>
void static_quant_epilogue(const float* input, scalar_t* output,
                           const float a_scale, const float* b_scale,
                           const int32_t* azp_with_adj, const int num_tokens,
                           const int hidden_size) {
  CPU_KERNEL_GUARD_IN(dynamic_output_scale_impl)
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using azp_adj_load_vec_t =
      typename KernelVecType<scalar_t>::azp_adj_load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    cvt_vec_t a_scale_vec(a_scale);
    cvt_vec_t b_scale_vec(*b_scale);
    cvt_vec_t scale_vec = a_scale_vec * b_scale_vec;

    int j = 0;
    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      cvt_vec_t elems_fp32(input + i * hidden_size + j);
      azp_adj_load_vec_t azp_adj_vec(azp_with_adj + j);
      cvt_vec_t azp_adj_fp32(azp_adj_vec);

      if constexpr (PerChannel) {
        b_scale_vec = cvt_vec_t(b_scale + j);
        scale_vec = b_scale_vec * a_scale_vec;
      }

      elems_fp32 = elems_fp32 - scale_vec * azp_adj_fp32;

      load_vec_t elems_out(elems_fp32);
      elems_out.save(output + i * hidden_size + j);
    }

    cvt_vec_t elems_fp32(input + i * hidden_size + j);
    azp_adj_load_vec_t azp_adj_vec(azp_with_adj + j);
    cvt_vec_t azp_adj_fp32(azp_adj_vec);

    if constexpr (PerChannel) {
      b_scale_vec = cvt_vec_t(b_scale + j);
      scale_vec = b_scale_vec * a_scale_vec;
    }

    elems_fp32 = elems_fp32 - scale_vec * azp_adj_fp32;

    load_vec_t elems_out(elems_fp32);
    elems_out.save(output + i * hidden_size + j, hidden_size - j);
  }
}

template <bool AZP, bool PerChannel, bool Bias, typename scalar_t>
void dynamic_quant_epilogue(const float* input, scalar_t* output,
                            const float* a_scale, const float* b_scale,
                            const int32_t* azp, const int32_t* azp_adj,
                            const scalar_t* bias, const int num_tokens,
                            const int hidden_size) {
  CPU_KERNEL_GUARD_IN(dynamic_quant_epilogue)
  using load_vec_t = typename KernelVecType<scalar_t>::load_vec_type;
  using azp_adj_load_vec_t =
      typename KernelVecType<scalar_t>::azp_adj_load_vec_type;
  using cvt_vec_t = typename KernelVecType<scalar_t>::cvt_vec_type;
  constexpr int vec_elem_num = load_vec_t::VEC_ELEM_NUM;

  #pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    int j = 0;
    cvt_vec_t token_scale_vec(a_scale[i]);
    cvt_vec_t token_zp_scale_vec;
    if constexpr (AZP) {
      float zp_scale_val = a_scale[i] * static_cast<float>(azp[i]);
      if constexpr (!PerChannel) {
        zp_scale_val *= *b_scale;
      }
      token_zp_scale_vec = cvt_vec_t(zp_scale_val);
    }

    for (; j < hidden_size - vec_elem_num; j += vec_elem_num) {
      cvt_vec_t elems_fp32(input + i * hidden_size + j);
      elems_fp32 = elems_fp32 * token_scale_vec;

      if constexpr (AZP) {
        azp_adj_load_vec_t azp_adj_vec(azp_adj + j);
        cvt_vec_t azp_adj_fp32(azp_adj_vec);
        azp_adj_fp32 = azp_adj_fp32 * token_zp_scale_vec;

        if constexpr (PerChannel) {
          cvt_vec_t b_scale_vec(b_scale + j);
          azp_adj_fp32 = azp_adj_fp32 * b_scale_vec;
        }

        elems_fp32 = elems_fp32 - azp_adj_fp32;
      }

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

    if constexpr (AZP) {
      azp_adj_load_vec_t azp_adj_vec(azp_adj + j);
      cvt_vec_t azp_adj_fp32(azp_adj_vec);
      azp_adj_fp32 = azp_adj_fp32 * token_zp_scale_vec;

      if constexpr (PerChannel) {
        cvt_vec_t b_scale_vec(b_scale + j);
        azp_adj_fp32 = azp_adj_fp32 * b_scale_vec;
      }

      elems_fp32 = elems_fp32 - azp_adj_fp32;
    }

    if constexpr (Bias) {
      load_vec_t bias_vec(bias + j);
      cvt_vec_t bias_vec_fp32(bias_vec);
      elems_fp32 = elems_fp32 + bias_vec_fp32;
    }

    load_vec_t elems_out(elems_fp32);
    elems_out.save(output + i * hidden_size + j, hidden_size - j);
  }
}
#else
template <typename scalar_t>
void static_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                   const float* scale, const int32_t* azp,
                                   const int num_tokens,
                                   const int hidden_size) {
  TORCH_CHECK(false, "static_scaled_int8_quant_impl requires AVX512 support.")
}

template <typename scalar_t>
void dynamic_scaled_int8_quant_impl(const scalar_t* input, int8_t* output,
                                    float* scale, int32_t* azp,
                                    const int num_tokens,
                                    const int hidden_size) {
  TORCH_CHECK(false, "dynamic_scaled_int8_quant_impl requires AVX512 support.")
}

template <bool PerChannel, typename scalar_t>
void static_quant_epilogue(const float* input, scalar_t* output,
                           const float a_scale, const float* b_scale,
                           const int32_t* azp_with_adj, const int num_tokens,
                           const int hidden_size) {
  TORCH_CHECK(false, "static_quant_epilogue requires AVX512 support.")
}

template <typename scalar_t>
void dynamic_quant_epilogue(const float* input, scalar_t* output,
                            const float* a_scale, const float* b_scale,
                            const int32_t* azp, const int32_t* azp_with_adj,
                            const scalar_t* bias, const int num_tokens,
                            const int hidden_size) {
  TORCH_CHECK(false, "dynamic_quant_epilogue requires AVX512 support.")
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

  VLLM_DISPATCH_FLOATING_TYPES(c.scalar_type(), "int8_scaled_mm", [&] {
    if (a_scales.numel() != 1) {
      // per-token
      // Note: oneDNN doesn't support per-token activation quantization
      // Ideally we want to fuse the GEMM and the scale procedure with oneDNN
      // JIT, the intermediate data is cached in registers or L1. But for now
      // the oneDNN GEMM code generation only supports two quantization
      // patterns: per-tensor or per-output-channel of weight.
      // So we have to apply the per-token scale with a 'epilogue'. In C=s_a *
      // s_b * (A@B) + bias, the C_inter = s_b * (A@B) is computed by oneDNN
      // GEMM, then the per-token scale (and bias) is applied with the epilogue
      // C=s_a * C_inter + bias.
      torch::Tensor tmp_fp32_out =
          torch::empty_like(c, ::at::ScalarType::Float);
      // Compute C_inter=s_b * (A@B)
      DNNLPrimitiveHelper<true>::gemm_s8s8_jit<float, void>(
          a.data_ptr<int8_t>(), b.data_ptr<int8_t>(),
          tmp_fp32_out.data_ptr<float>(), nullptr, a.size(0), b.size(1),
          a.size(1), nullptr, b_scales.data_ptr<float>(), 0, b_scales.numel());
      if (bias.has_value()) {
        // Compute C=s_a * C_inter + bias
        dynamic_quant_epilogue<false, true, true>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            a_scales.data_ptr<float>(), nullptr, nullptr, nullptr,
            bias->data_ptr<scalar_t>(), c.size(0), c.size(1));
      } else {
        // Compute C=s_a * C_inter
        dynamic_quant_epilogue<false, true, false, scalar_t>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            a_scales.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
            c.size(0), c.size(1));
      }
    } else {
      // per-tensor
      if (bias.has_value()) {
        // Compute C=s_a * s_b * (A@B) + bias
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<scalar_t>(),
            bias->data_ptr<scalar_t>(), a.size(0), b.size(1), a.size(1),
            a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            a_scales.numel(), b_scales.numel());
      } else {
        // Compute C=s_a * s_b * (A@B)
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit<scalar_t, void>(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<scalar_t>(),
            nullptr, a.size(0), b.size(1), a.size(1),
            a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            a_scales.numel(), b_scales.numel());
      }
    }
  });
}

void int8_scaled_mm_azp(torch::Tensor& c,        // [M, OC], row-major
                        const torch::Tensor& a,  // [M, IC], row-major
                        const torch::Tensor& b,  // [IC, OC], column-major
                        const torch::Tensor& a_scales,            // [1] or [M]
                        const torch::Tensor& b_scales,            // [1] or [OC]
                        const torch::Tensor& azp_adj,             // [OC]
                        const c10::optional<torch::Tensor>& azp,  // [1] or [M]
                        const c10::optional<torch::Tensor>& bias  // [OC]
) {
  CPU_KERNEL_GUARD_IN(cutlass_scaled_mm_azp)
  // Checks for conformality
  TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8,
              "int8_scaled_mm_azp only supports INT8 inputs.")
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
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  VLLM_DISPATCH_FLOATING_TYPES(c.scalar_type(), "int8_scaled_mm_azp", [&] {
    torch::Tensor tmp_fp32_out = torch::empty_like(c, ::at::ScalarType::Float);
    if (a_scales.numel() != 1) {
      // per-token
      // Note: oneDNN doesn't support per-token activation quantization
      // Compute C_inter=s_b * (A@B)
      DNNLPrimitiveHelper<true>::gemm_s8s8_jit<float, void>(
          a.data_ptr<int8_t>(), b.data_ptr<int8_t>(),
          tmp_fp32_out.data_ptr<float>(), nullptr, a.size(0), b.size(1),
          a.size(1), nullptr, b_scales.data_ptr<float>(), 0, b_scales.numel());
      if (bias.has_value()) {
        // Compute C=s_a * C_inter - s_a * s_b * azp * azp_adj + bias
        if (b_scales.numel() != 1) {
          // Per-Channel
          dynamic_quant_epilogue<true, true, true>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
              azp->data_ptr<int32_t>(), azp_adj.data_ptr<int32_t>(),
              bias->data_ptr<scalar_t>(), c.size(0), c.size(1));
        } else {
          // Per-Tensor
          dynamic_quant_epilogue<true, false, true>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
              azp->data_ptr<int32_t>(), azp_adj.data_ptr<int32_t>(),
              bias->data_ptr<scalar_t>(), c.size(0), c.size(1));
        }
      } else {
        // Compute C=s_a * C_inter - s_a * s_b * azp * azp_adj
        if (b_scales.numel() != 1) {
          // Per-Channel
          dynamic_quant_epilogue<true, true, false, scalar_t>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
              azp->data_ptr<int32_t>(), azp_adj.data_ptr<int32_t>(), nullptr,
              c.size(0), c.size(1));
        } else {
          // Per-Tensor
          dynamic_quant_epilogue<true, false, false, scalar_t>(
              tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
              a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
              azp->data_ptr<int32_t>(), azp_adj.data_ptr<int32_t>(), nullptr,
              c.size(0), c.size(1));
        }
      }
    } else {
      // per-tensor
      if (bias.has_value()) {
        // Compute C_inter=s_a * s_b * (A@B) + bias
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(),
            tmp_fp32_out.data_ptr<float>(), bias->data_ptr<scalar_t>(),
            a.size(0), b.size(1), a.size(1), a_scales.data_ptr<float>(),
            b_scales.data_ptr<float>(), a_scales.numel(), b_scales.numel());
      } else {
        // Compute C_inter=s_a * s_b * (A@B)
        DNNLPrimitiveHelper<false>::gemm_s8s8_jit<float, void>(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(),
            tmp_fp32_out.data_ptr<float>(), nullptr, a.size(0), b.size(1),
            a.size(1), a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            a_scales.numel(), b_scales.numel());
      }

      // Compute C=C_inter - s_a * s_b * azp_adj
      if (b_scales.numel() != 1) {
        // Per-Channel
        static_quant_epilogue<true>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            *a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            azp_adj.data_ptr<int32_t>(), a.size(0), b.size(1));
      } else {
        // Per-Tensor
        static_quant_epilogue<false>(
            tmp_fp32_out.data_ptr<float>(), c.data_ptr<scalar_t>(),
            *a_scales.data_ptr<float>(), b_scales.data_ptr<float>(),
            azp_adj.data_ptr<int32_t>(), a.size(0), b.size(1));
      }
    }
  });
}

// static-per-tensor quantization.
void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              const torch::Tensor& input,  // [..., hidden_size]
                              const torch::Tensor& scale,
                              c10::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(static_scaled_int8_quant)
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp.has_value() || azp->numel() == 1);

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          static_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              hidden_size);
        } else {
          static_scaled_int8_quant_impl<false>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), nullptr, num_tokens, hidden_size);
        }
      });
}

// dynamic-per-token quantization.
void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    const torch::Tensor& input,  // [..., hidden_size]
    torch::Tensor& scale,        // [..., 1]
    c10::optional<torch::Tensor> const& azp) {
  CPU_KERNEL_GUARD_IN(dynamic_scaled_int8_quant)
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_impl", [&] {
        if (azp.has_value()) {
          dynamic_scaled_int8_quant_impl<true>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), azp->data_ptr<int32_t>(), num_tokens,
              hidden_size);
        } else {
          dynamic_scaled_int8_quant_impl<false>(
              input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
              scale.data_ptr<float>(), nullptr, num_tokens, hidden_size);
        }
      });
}
