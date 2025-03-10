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

#if defined ENABLE_NVFP4 && ENABLE_NVFP4
void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha);
#endif

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A,
                           torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf,
                           torch::Tensor const& alpha) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  return cutlass_scaled_fp4_mm_sm100a(D, A, B, A_sf, B_sf, alpha);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No compiled nvfp4 mm kernel, vLLM should "
                              "be compiled using CUDA 12.8 and target "
                              "compute capability 100 or above.");
}
