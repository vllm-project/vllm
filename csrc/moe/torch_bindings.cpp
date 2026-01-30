#include "core/registration.h"

#include <torch/all.h>

// Define the _moe_C library here. The stable extension (_moe_C_stable_libtorch)
// will add its ops using STABLE_TORCH_LIBRARY_FRAGMENT.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");

  m.def(
      "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
      "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
      "int b_q_type, SymInt size_m, "
      "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
      "topk, "
      "int moe_block_size, bool replicate_input, bool apply_weights)"
      " -> Tensor");
}

// Implementation is registered in marlin_moe_wna16/ops.cu

REGISTER_EXTENSION(_moe_C)
