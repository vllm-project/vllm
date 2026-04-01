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

// cp.async helpers: async copy from global to shared memory (SM80+).
// On older architectures or ROCm these fall back to synchronous loads.

namespace vllm {
namespace cuda_async {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)

__device__ __forceinline__ void cp_async_shared_global_16_cg(void* smem_ptr,
                                                             const void* glob_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_shared_global_ca(void* smem_ptr,
                                                          const void* glob_ptr,
                                                          int size_bytes) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if (size_bytes == 4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else if (size_bytes == 8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem), "l"(glob_ptr));
  }
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(n));
}
#endif

}  // namespace cuda_async
}  // namespace vllm
