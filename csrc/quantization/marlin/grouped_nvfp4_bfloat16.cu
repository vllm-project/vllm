// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "kernel.h"
#include "marlin_template.h"

namespace marlin {
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

MarlinFuncPtr get_grouped_nvfp4_marlin_kernel(GroupedNvfp4Kernel kernel) {
  switch (kernel) {
    case GroupedNvfp4Kernel::M32Whole:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 8, 4,
                    false, 4, 1, false, 8, true, false>;
    case GroupedNvfp4Kernel::M32Partial:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 8, 4,
                    false, 4, 1, false, 8, true, true>;
    case GroupedNvfp4Kernel::M64Prefix:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4,
                    false, 3, 1, false, 8, false, false>;
    case GroupedNvfp4Kernel::M64Whole:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4,
                    false, 3, 1, false, 8, true, false>;
    case GroupedNvfp4Kernel::M64Partial:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4,
                    false, 3, 1, false, 8, true, true>;
    case GroupedNvfp4Kernel::M64WholeG16:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4,
                    false, 3, 1, false, 16, true, false>;
    case GroupedNvfp4Kernel::M64N256Partial:
      return Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 256, 4, 16, 4,
                    false, 4, 1, false, 1, true, true>;
  }
  return nullptr;
}

}  // namespace marlin
