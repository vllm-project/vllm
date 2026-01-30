#include "stable/moe/moe_ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Use STABLE_TORCH_LIBRARY_FRAGMENT since the _moe_C library is defined in the
// non-stable _moe_C extension via TORCH_LIBRARY.
STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");

  // Apply topk sigmoid to the gating outputs.
  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids,"
      "                     Tensor? maybe_expert_map) -> () ");

#ifndef USE_ROCM
  m.def(
      "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
      "Tensor b_scales, Tensor? b_qzeros, "
      "Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int bit) -> Tensor");

  m.def(
      "moe_permute(Tensor input, Tensor topk_ids,"
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert,"
      "int n_local_expert,"
      "int topk, int? align_block_size,Tensor! permuted_input, Tensor! "
      "expert_first_token_offset, Tensor! inv_permuted_idx, Tensor! "
      "permuted_idx, Tensor! m_indices)->()");

  m.def(
      "moe_unpermute(Tensor permuted_hidden_states, Tensor topk_weights,"
      "Tensor inv_permuted_idx, Tensor? expert_first_token_offset, "
      "int topk, Tensor! hidden_states)->()");

  m.def("moe_permute_unpermute_supported() -> bool");

  // Row shuffle for MoE
  m.def(
      "shuffle_rows(Tensor input_tensor, Tensor dst2src_map, Tensor! "
      "output_tensor) -> ()");

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("topk_softmax", TORCH_BOX(&topk_softmax));
  m.impl("topk_sigmoid", TORCH_BOX(&topk_sigmoid));
  m.impl("moe_sum", TORCH_BOX(&moe_sum));
  m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
  m.impl("batched_moe_align_block_size",
         TORCH_BOX(&batched_moe_align_block_size));
  m.impl("moe_lora_align_block_size", TORCH_BOX(&moe_lora_align_block_size));

#ifndef USE_ROCM
  m.impl("moe_wna16_gemm", TORCH_BOX(&moe_wna16_gemm));
  // moe_permute and moe_unpermute are registered in moe_permute_unpermute_op.cu
  m.impl("shuffle_rows", TORCH_BOX(&shuffle_rows));
  m.impl("grouped_topk", TORCH_BOX(&grouped_topk));
#endif  // USE_ROCM
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CompositeExplicitAutograd, m) {
  m.impl("moe_permute_unpermute_supported",
         TORCH_BOX(&moe_permute_unpermute_supported));
}

REGISTER_EXTENSION(_moe_C_stable_libtorch)
