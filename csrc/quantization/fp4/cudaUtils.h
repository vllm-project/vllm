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

#pragma once

#include "cutlass/cutlass.h"
#include <climits>

namespace vllm {
namespace common {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& file, int line, const std::string& message)
      : std::runtime_error("CUDA Error at " + file + ":" +
                           std::to_string(line) + " - " + message) {}
};

template <typename T>
void check(T result, const char* func, const char* file, int line) {
  if (result != cudaSuccess) {
    throw CudaException(
        file, line,
        std::string("[VLLM][ERROR] CUDA runtime error in ") + func + ": " +
            cudaGetErrorString(static_cast<cudaError_t>(result)));
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

inline int getMaxSharedMemoryPerBlockOptin() {
  int device_id;
  int max_shared_memory_per_block;
  check_cuda_error(cudaGetDevice(&device_id));
  check_cuda_error(cudaDeviceGetAttribute(
      &max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      device_id));
  return max_shared_memory_per_block;
}

inline int getSMVersion() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  check_cuda_error(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  check_cuda_error(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

}  // namespace common
}  // namespace vllm
