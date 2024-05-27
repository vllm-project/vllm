#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  vllm::def(m, "topk_softmax", &topk_softmax, {0, 1});
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
}

// TODO: get rid of this?
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
