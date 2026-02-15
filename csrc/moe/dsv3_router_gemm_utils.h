/*
 * Adapted from SGLang's sgl-kernel implementation, which was adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>

inline int getSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  int sm_major = 0;
  int sm_minor = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
  return sm_major * 10 + sm_minor;
}

inline bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      const char* env = std::getenv("TRTLLM_ENABLE_PDL");
      enablePDL = env && env[0] == '1' && env[1] == '\0';
    }
  });
  return enablePDL;
}
