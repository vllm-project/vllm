// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Architecture-specific targets also define their family feature macro.
#if defined(VLLM_MARLIN_LDMATRIX_S4_ENABLED) && CUDART_VERSION >= 13040 && \
    ((defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 900) || \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                            \
      (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 ||                            \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030 ||                            \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1070 ||                            \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1100 ||                            \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200 ||                            \
       __CUDA_ARCH_FAMILY_SPECIFIC__ == 1210)))
  #define VLLM_MARLIN_LDMATRIX_S4_DEVICE_ENABLED 1
#else
  #define VLLM_MARLIN_LDMATRIX_S4_DEVICE_ENABLED 0
#endif
