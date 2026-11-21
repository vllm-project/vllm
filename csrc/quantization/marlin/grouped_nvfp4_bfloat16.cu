// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "kernel.h"
#include "marlin_template.h"

namespace marlin {
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

MarlinFuncPtr get_grouped_nvfp4_marlin_kernel() {
  return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                vllm::kFE4M3fn.id(), 128, 4, 8, 4, false, 3, 1, false, 8>;
}
MarlinFuncPtr get_grouped_nvfp4_marlin_kernel_m32() {
  return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                vllm::kFE4M3fn.id(), 128, 2, 8, 4, false, 4, 1, false, 8>;
}
}  // namespace marlin
