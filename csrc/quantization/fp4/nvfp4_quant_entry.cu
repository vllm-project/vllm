/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include "nvfp4_utils.cuh"

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             bool is_sf_swizzled_layout);
#endif

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
void scaled_fp4_quant_sm103a(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf);
// PDL variant: launches quant with ProgrammaticStreamSerialization.
void scaled_fp4_quant_sm103a_pdl(torch::Tensor const& output,
                                 torch::Tensor const& input,
                                 torch::Tensor const& output_sf,
                                 torch::Tensor const& input_sf);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_experts_quant_sm1xxa(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_nvfp4_quant_sm1xxa(torch::Tensor& output,
                                     torch::Tensor& output_sf,
                                     torch::Tensor& input,
                                     torch::Tensor& input_sf);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);
#endif

void scaled_fp4_quant_out(torch::Tensor const& input,
                          torch::Tensor const& input_sf,
                          bool is_sf_swizzled_layout, torch::Tensor& output,
                          torch::Tensor& output_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_quant_sm1xxa(output, input, output_sf, input_sf,
                                 is_sf_swizzled_layout);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization kernel");
}

std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant_func(
    torch::Tensor const& input, torch::Tensor const& input_sf,
    bool is_sf_swizzled_layout) {
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto device = input.device();

  // Two fp4 values packed into a uint8
  auto output = torch::empty(
      {m, n / 2}, torch::TensorOptions().device(device).dtype(torch::kUInt8));

  torch::Tensor output_sf;
  if (is_sf_swizzled_layout) {
    auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
    output_sf = torch::empty(
        {sf_m, sf_n},
        torch::TensorOptions().device(device).dtype(torch::kInt32));
  } else {
    output_sf = torch::empty(
        {m, n / CVT_FP4_SF_VEC_SIZE},
        torch::TensorOptions().device(device).dtype(torch::kUInt8));
  }

  scaled_fp4_quant_out(input, input_sf, is_sf_swizzled_layout, output,
                       output_sf);
  return {output, output_sf};
}

void scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled nvfp4 experts quantization kernel");
}

void silu_and_mul_nvfp4_quant(torch::Tensor& output, torch::Tensor& output_sf,
                              torch::Tensor& input, torch::Tensor& input_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return silu_and_mul_nvfp4_quant_sm1xxa(output, output_sf, input, input_sf);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 quantization kernel");
}

void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 experts quantization kernel");
}

// SM103-native quantization: writes SM103-layout scale factors directly,
// eliminating the SM100->SM103 conversion step on the critical path.
std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant_sm103a_func(
    torch::Tensor const& input, torch::Tensor const& input_sf) {
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto device = input.device();

  auto output = torch::empty(
      {m, n / 2}, torch::TensorOptions().device(device).dtype(torch::kUInt8));

  auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
  auto output_sf = torch::empty(
      {sf_m, sf_n},
      torch::TensorOptions().device(device).dtype(torch::kInt32));

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  scaled_fp4_quant_sm103a(output, input, output_sf, input_sf);
  return {output, output_sf};
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled SM103 nvfp4 quantization kernel");
}

void scaled_fp4_quant_sm103a_out(torch::Tensor const& input,
                                 torch::Tensor const& input_sf,
                                 torch::Tensor& output,
                                 torch::Tensor& output_sf) {
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  scaled_fp4_quant_sm103a(output, input, output_sf, input_sf);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled SM103 nvfp4 quantization kernel");
}

// ============================================================================
// PDL-enabled SM103 quantization entry points.
//
// These launch the quant kernel with ProgrammaticStreamSerialization,
// allowing the subsequent GEMM to begin before quantization completes.
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant_sm103a_pdl_func(
    torch::Tensor const& input, torch::Tensor const& input_sf) {
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto device = input.device();

  auto output = torch::empty(
      {m, n / 2}, torch::TensorOptions().device(device).dtype(torch::kUInt8));

  auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
  auto output_sf = torch::empty(
      {sf_m, sf_n},
      torch::TensorOptions().device(device).dtype(torch::kInt32));

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  scaled_fp4_quant_sm103a_pdl(output, input, output_sf, input_sf);
  return {output, output_sf};
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled SM103 PDL nvfp4 quantization kernel");
}

void scaled_fp4_quant_sm103a_pdl_out(torch::Tensor const& input,
                                     torch::Tensor const& input_sf,
                                     torch::Tensor& output,
                                     torch::Tensor& output_sf) {
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  scaled_fp4_quant_sm103a_pdl(output, input, output_sf, input_sf);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled SM103 PDL nvfp4 quantization kernel");
}
