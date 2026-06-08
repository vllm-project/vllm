#pragma once

// Shared TORCH_UTILS_CHECK across both libtorch stable and unstable source
// files. Keep this header free of CUTLASS/CUTE so attention/quant headers can
// use it.
//
// If TORCH_TARGET_VERSION is defined, we are building _C_stable_libtorch.so so
// use STD_TORCH_CHECK via header-only.
// Otherwise, use TORCH_CHECK via torch/all.h.

#ifdef TORCH_TARGET_VERSION
  #if __has_include(<torch/headeronly/util/Exception.h>)
    #include <torch/headeronly/util/Exception.h>
  #else
    #include <c10/util/Exception.h>
    #define STD_TORCH_CHECK TORCH_CHECK
  #endif
  #define TORCH_UTILS_CHECK STD_TORCH_CHECK
#else
  #include <torch/all.h>
  #define TORCH_UTILS_CHECK TORCH_CHECK
#endif
