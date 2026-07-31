// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_FUSED_MOE_ACTIVATIONS_HPP
#define CPU_FUSED_MOE_ACTIVATIONS_HPP

#include <cmath>
#include <cstdint>
#include <string>

#include "cpu/cpu_arch_macros.h"
#include "cpu/utils.hpp"

namespace cpu_fused_moe_utils {
enum class FusedMOEAct {
  SiluAndMul,
  SwigluOAIAndMul,
  GeluAndMul,
  GeluTanhAndMul,
};

inline FusedMOEAct get_act_type(const std::string& act) {
  if (act == "silu") {
    return FusedMOEAct::SiluAndMul;
  } else if (act == "swigluoai") {
    return FusedMOEAct::SwigluOAIAndMul;
  } else if (act == "gelu") {
    return FusedMOEAct::GeluAndMul;
  } else if (act == "gelu_tanh") {
    return FusedMOEAct::GeluTanhAndMul;
  } else {
    TORCH_CHECK(false, "Invalid act type: " + act);
  }
}

template <typename scalar_t>
void swigluoai_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                       const int32_t m_size, const int32_t n_size,
                       const int32_t input_stride,
                       const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
#if !defined(__aarch64__)
  // For GPT-OSS interleaved gate-up weights
  alignas(64) static int32_t index[16] = {0,  2,  4,  6,  8,  10, 12, 14,
                                          16, 18, 20, 22, 24, 26, 28, 30};
  vec_op::INT32Vec16 index_vec(index);
#endif
  vec_op::FP32Vec16 gate_up_max_vec(7.0);
  vec_op::FP32Vec16 up_min_vec(-7.0);
  vec_op::FP32Vec16 alpha_vec(1.702);
  vec_op::FP32Vec16 one_vec(1.0);

  DEFINE_FAST_EXP

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < n_size; n += 32) {
      // Note: AdvSIMD does not support gather loads
#if defined(__aarch64__)
      vec_op::FP32Vec16 gate_vec(vec_op::uninit);
      vec_op::FP32Vec16 up_vec(vec_op::uninit);
      vec_op::FP32Vec16::load_even_odd(input + n, gate_vec, up_vec);
#else
      vec_op::FP32Vec16 gate_vec(input + n, index_vec);
      vec_op::FP32Vec16 up_vec(input + n + 1, index_vec);
#endif
      gate_vec = gate_vec.min(gate_up_max_vec);
      up_vec = up_vec.clamp(up_min_vec, gate_up_max_vec);
      auto sigmoid_vec = one_vec / (one_vec + fast_exp(-gate_vec * alpha_vec));
      auto glu = gate_vec * sigmoid_vec;
      auto gated_output_fp32 = (one_vec + up_vec) * glu;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n / 2);
    }
    input += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
void silu_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                  const int32_t m_size, const int32_t n_size,
                  const int32_t input_stride, const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  const int32_t dim = n_size / 2;
  float* __restrict__ gate = input;
  float* __restrict__ up = input + dim;
  vec_op::FP32Vec16 one_vec(1.0);

  DEFINE_FAST_EXP

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < dim; n += 16) {
      vec_op::FP32Vec16 gate_vec(gate + n);
      vec_op::FP32Vec16 up_vec(up + n);
      auto sigmoid_vec = one_vec / (one_vec + fast_exp(-gate_vec));
      auto silu = gate_vec * sigmoid_vec;
      auto gated_output_fp32 = up_vec * silu;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n);
    }
    gate += input_stride;
    up += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
void gelu_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                  const int32_t m_size, const int32_t n_size,
                  const int32_t input_stride, const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  const int32_t dim = n_size / 2;
  float* __restrict__ gate = input;
  float* __restrict__ up = input + dim;
  vec_op::FP32Vec16 one_vec(1.0);
  vec_op::FP32Vec16 w1_vec(M_SQRT1_2);
  vec_op::FP32Vec16 w2_vec(0.5);
  alignas(64) float temp[16];

  DEFINE_FAST_EXP

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < dim; n += 16) {
      vec_op::FP32Vec16 gate_vec(gate + n);
      vec_op::FP32Vec16 up_vec(up + n);
      auto er_input_vec = gate_vec * w1_vec;

      er_input_vec.save(temp);
      for (int32_t i = 0; i < 16; ++i) {
        temp[i] = std::erf(temp[i]);
      }
      vec_op::FP32Vec16 er_vec(temp);
      auto gelu = gate_vec * w2_vec * (one_vec + er_vec);
      auto gated_output_fp32 = up_vec * gelu;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n);
    }
    gate += input_stride;
    up += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
void gelu_tanh_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                       const int32_t m_size, const int32_t n_size,
                       const int32_t input_stride,
                       const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  const int32_t dim = n_size / 2;
  float* __restrict__ gate = input;
  float* __restrict__ up = input + dim;
  vec_op::FP32Vec16 one_vec(1.0);
  vec_op::FP32Vec16 w1_vec(0.7978845608028654);
  vec_op::FP32Vec16 w2_vec(0.5);
  vec_op::FP32Vec16 w3_vec(0.044715);

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < dim; n += 16) {
      vec_op::FP32Vec16 gate_vec(gate + n);
      vec_op::FP32Vec16 up_vec(up + n);
      auto gate_pow3_vec = gate_vec * gate_vec * gate_vec;
      auto inner_vec = w1_vec * (gate_vec + w3_vec * gate_pow3_vec);
      // Note: can't use fast_exp form because diffusiongemma will generate
      // wrong results
      auto tanh_vec = inner_vec.tanh();
      auto gelu_tanh = gate_vec * w2_vec * (one_vec + tanh_vec);
      auto gated_output_fp32 = up_vec * gelu_tanh;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n);
    }
    gate += input_stride;
    up += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
FORCE_INLINE void apply_gated_act(const FusedMOEAct act,
                                  float* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  const int32_t m, const int32_t n,
                                  const int32_t input_stride,
                                  const int32_t output_stride) {
  switch (act) {
    case FusedMOEAct::SwigluOAIAndMul:
      swigluoai_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    case FusedMOEAct::SiluAndMul:
      silu_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    case FusedMOEAct::GeluAndMul:
      gelu_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    case FusedMOEAct::GeluTanhAndMul:
      gelu_tanh_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    default:
      TORCH_CHECK(false, "Unsupported act type.");
  }
}
}  // namespace cpu_fused_moe_utils

#endif
