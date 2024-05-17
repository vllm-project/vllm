#include <Python.h>

#include "moe_ops.h"
#include "marlin_moe_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax, "Apply topk softmax to the gating outputs.");
  m.def("marlin_gemm_moe", &marlin_gemm_moe, "Marlin gemm moe kernel.");
}

// // // should be enough for a unit test?
// TORCH_LIBRARY(nm_ops, m) {
//   // m.def("marlin_gemm_moe() -> Tensor");
//   // m.def("marlin_gemm_moe(Tensor a, Tensor b_q_weights, Tensor sorted_ids, Tensor b_scales, Tensor g_idx, Tensor perm, "
//   //       "Tensor workspace, int size_m, "
//   //       "int size_n, int size_k, bool is_k_full, int num_tokens_post_padded, int num_experts, int moe_block_size) -> Tensor");
// }
