#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor! input, Tensor output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // temporarily adapted from
  // https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
  m.def(
      "sgl_moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                         int block_size, Tensor! sorted_token_ids,"
      "                         Tensor! experts_ids,"
      "                         Tensor! num_tokens_post_pad) -> ()");
  m.impl("sgl_moe_align_block_size", torch::kCUDA, &sgl_moe_align_block_size);

#ifndef USE_ROCM
  m.def(
      "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
      "Tensor b_scales, Tensor? b_qzeros, "
      "Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int bit) -> Tensor");

  m.impl("moe_wna16_gemm", torch::kCUDA, &moe_wna16_gemm);

  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor! b_scales, Tensor? b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, bool is_ep, int b_q_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float) -> Tensor");

  // conditionally compiled so impl registration is in source file

#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
