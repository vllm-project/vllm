#pragma once

// Shared TORCH_UTILS_CHECK for legacy ATen TUs vs stable-extension TUs.
// Keep this header free of CUTLASS/CUTE so attention/quant headers can use it.
//
// - TORCH_TARGET_VERSION: _C_stable_libtorch (STD_TORCH_CHECK via header-only).
// - else: typical _C / ATen path (TORCH_CHECK via torch/all.h).

#ifdef TORCH_TARGET_VERSION
  #include <torch/headeronly/util/Exception.h>
  #define TORCH_UTILS_CHECK STD_TORCH_CHECK
#else
  #include <torch/all.h>
  #define TORCH_UTILS_CHECK TORCH_CHECK
#endif
