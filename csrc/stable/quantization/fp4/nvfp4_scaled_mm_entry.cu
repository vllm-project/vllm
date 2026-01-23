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
#include <torch/csrc/stable/accelerator.h>
#include "../../torch_utils.h"
#include "stable/cutlass_extensions/common.hpp"

#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
void cutlass_scaled_fp4_mm_sm100a(torch::stable::Tensor& D,
                                  torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha);
#endif

#if defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120
void cutlass_scaled_fp4_mm_sm120a(torch::stable::Tensor& D,
                                  torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha);
#endif

void cutlass_scaled_fp4_mm(torch::stable::Tensor& D,
                           torch::stable::Tensor const& A,
                           torch::stable::Tensor const& B,
                           torch::stable::Tensor const& A_sf,
                           torch::stable::Tensor const& B_sf,
                           torch::stable::Tensor const& alpha) {
  // Make sure we're on A's device.
  torch::stable::accelerator::DeviceGuard device_guard(A.get_device_index());
  const int32_t sm = get_sm_version_num();

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  if (sm >= 100 && sm < 120) {
    cutlass_scaled_fp4_mm_sm100a(D, A, B, A_sf, B_sf, alpha);
    return;
  }
#endif

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) {
    cutlass_scaled_fp4_mm_sm120a(D, A, B, A_sf, B_sf, alpha);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled nvfp4 mm kernel for SM ", sm,
      ". Recompile with CUDA >= 12.8 and CC >= 100.");
}

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability) {
  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);
  return cuda_device_capability >= 100 && runtimeVersion >= 12080;
}
