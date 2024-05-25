#include <torch/extension.h>

#include "punica_ops.h"

#define TORCH_LIBRARY_EXPAND(NAME, M) TORCH_LIBRARY(NAME, M)

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.impl("dispatch_bgmv", torch::kCUDA, &dispatch_bgmv);
  m.impl("dispatch_bgmv_low_level", torch::kCUDA, &dispatch_bgmv_low_level);
}
