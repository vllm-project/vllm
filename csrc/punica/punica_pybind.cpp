#include "punica_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  vllm::def(m, "dispatch_bgmv", &dispatch_bgmv, {0});
  m.impl("dispatch_bgmv", torch::kCUDA, &dispatch_bgmv);

  vllm::def(m, "dispatch_bgmv_low_level", &dispatch_bgmv_low_level, {0});
  m.impl("dispatch_bgmv_low_level", torch::kCUDA, &dispatch_bgmv_low_level);
}

// TODO: get rid of this
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
