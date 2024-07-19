#include "cpu_types.hpp"

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
    }
    else {
      elems_int8.save_low(output + i * hidden_size + j, hidden_size - j);
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
#endif
}  // namespace

// static-per-tensor quantization.
void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              const torch::Tensor& input,  // [..., hidden_size]
                              const torch::Tensor& scale) {
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
    torch::Tensor& output,       // [..., hidden_size]
    const torch::Tensor& input,  // [..., hidden_size]
    torch::Tensor& scale) {}