#include "moe_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax, "Apply topk softmax to the gating outputs.");
  m.def("expand_and_permute", &expand_and_permute, "Expand and permute the input tokens.");
  m.def("moe_mlp", &moe_mlp, "Apply the MoE MLP.");
  m.def("unpermute_and_reduce", &unpermute_and_reduce, "Unpermute and reduce the MoE outputs.");
}
