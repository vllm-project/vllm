#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices,Tensor "
      "token_expert_indices,Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
}

// TODO: get rid of this?
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
