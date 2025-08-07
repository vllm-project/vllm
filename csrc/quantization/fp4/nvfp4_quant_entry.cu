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

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf);
#endif

#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);
#endif

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  return scaled_fp4_quant_sm1xxa(output, input, output_sf, input_sf);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization kernel");
}

void scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts) {
#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
  return scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled nvfp4 experts quantization kernel");
}
