#include "marlin_moe_kernel_ku4b8.h"

namespace marlin_moe {

// We return bool so we can create these different kernel calls as a sequence
// of if-elseif's.
bool call_marlin_moe_kernel_ku4b8(
    vllm::ScalarType const& q_type, int thread_n_blocks, int thread_k_blocks,
    bool has_act_order, int group_blocks, int num_threads, int blocks,
    int max_shared_mem, cudaStream_t stream, const int4* A_ptr,
    const int4* B_ptr, int4* C_ptr, const int* sorted_ids_ptr,
    const float* topk_weights_ptr, const int4* s_ptr, const int4* zp_ptr,
    const int* g_idx_ptr, int* expert_offsets_ptr, int num_groups,
    int expert_idx, int num_experts, int topk, int prob_m, int prob_n,
    int prob_k, int tot_m, int* locks, bool replicate_input, bool apply_weights,
    int m_block, int max_par, int cfg_max_m_blocks) {
  bool has_zp = false;

  if (false) {
  }
  GPTQ_CALL_IF_MOE(vllm::kU4B8, 16, 4, 256)
  GPTQ_CALL_IF_MOE(vllm::kU4B8, 8, 8, 256)
  GPTQ_CALL_IF_MOE(vllm::kU4B8, 8, 4, 128)
  GPTQ_CALL_IF_MOE(vllm::kU4B8, 4, 8, 128)
  else {
    return false;
  }
  return true;
}

}  // namespace marlin_moe
