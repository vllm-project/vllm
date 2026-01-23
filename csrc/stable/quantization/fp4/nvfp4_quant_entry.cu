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

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "../../torch_utils.h"

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_quant_sm1xxa(torch::stable::Tensor const& output,
                             torch::stable::Tensor const& input,
                             torch::stable::Tensor const& output_sf,
                             torch::stable::Tensor const& input_sf,
                             bool is_sf_swizzled_layout);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_nvfp4_quant_sm1xxa(torch::stable::Tensor& output,
                                     torch::stable::Tensor& output_sf,
                                     torch::stable::Tensor& input,
                                     torch::stable::Tensor& input_sf);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);
#endif

void scaled_fp4_quant(torch::stable::Tensor& output,
                      torch::stable::Tensor const& input,
                      torch::stable::Tensor& output_sf,
                      torch::stable::Tensor const& input_sf,
                      bool is_sf_swizzled_layout) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_quant_sm1xxa(output, input, output_sf, input_sf,
                                 is_sf_swizzled_layout);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(false,
                                  "No compiled nvfp4 quantization kernel");
}

void scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled nvfp4 experts quantization kernel");
}

void silu_and_mul_nvfp4_quant(torch::stable::Tensor& output,
                              torch::stable::Tensor& output_sf,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& input_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return silu_and_mul_nvfp4_quant_sm1xxa(output, output_sf, input, input_sf);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 quantization kernel");
}

void silu_and_mul_scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 experts quantization kernel");
}
