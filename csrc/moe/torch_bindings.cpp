#include "registration.h"
#include "moe_ops.h"
#include "marlin_moe_ops.h"

#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Apply topk softmax to the gating outputs.
  ops.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  ops.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  ops.def(
      "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
      "g_idx, Tensor! perm, Tensor! workspace, int size_m, int size_n, int "
      "size_k, bool is_k_full, int num_experts, int topk, int moe_block_size, "
      "bool replicate_input, bool apply_weights) -> Tensor");

  ops.impl("marlin_gemm_moe", torch::kCUDA, &marlin_gemm_moe);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
