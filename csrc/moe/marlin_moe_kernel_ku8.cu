#include "marlin_moe_kernel.cuh"

namespace marlin_moe {

#define __CALL_IF_MOE_8(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS,            \
                      THREAD_K_BLOCKS, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS,   \
                      NUM_THREADS)                                            \
  else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&          \
           thread_n_blocks == THREAD_N_BLOCKS &&                              \
           thread_k_blocks == THREAD_K_BLOCKS &&                              \
           has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&              \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {      \
    cudaFuncSetAttribute(MarlinMoE<W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, \
                                   THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES,  \
                                   HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>,      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                         max_shared_mem);                                     \
    MarlinMoE<W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,     \
              THREAD_K_BLOCKS, STAGES, HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS>   \
        <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                    \
            A_ptr, B_ptr, C_ptr, sorted_ids_ptr, topk_weights_ptr, s_ptr,     \
            zp_ptr, g_idx_ptr, expert_offsets_ptr, num_groups, expert_idx,    \
            num_experts, topk, prob_m, prob_n, prob_k, tot_m, locks,          \
            replicate_input, apply_weights, m_block, max_par,                 \
            cfg_max_m_blocks);                                                \
  }


#define AWQ_CALL_IF_MOE_8(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
  __CALL_IF_MOE_8(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF_MOE_8(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
  __CALL_IF_MOE_8(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF_MOE_8(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
  __CALL_IF_MOE_8(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF_MOE_8(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                             \
  __CALL_IF_MOE_8(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF_MOE_8(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF_MOE_8(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

// We return bool so we can create these different kernel calls as a sequence
// of if-elseif's.
bool call_marlin_moe_kernel_ku8(
        vllm::ScalarType const& q_type, int thread_m_blocks,
        int thread_n_blocks, int thread_k_blocks, bool has_act_order,
        bool has_zp, int group_blocks, int num_threads, int blocks,
        int max_shared_mem, cudaStream_t stream, const int4* A_ptr,
        const int4* B_ptr, int4* C_ptr, const int* sorted_ids_ptr,
        const float* topk_weights_ptr, const int4* s_ptr, const int4* zp_ptr,
        const int* g_idx_ptr, int* expert_offsets_ptr, int num_groups,
        int expert_idx, int num_experts, int topk, int prob_m, int prob_n,
        int prob_k, int tot_m, int* locks, bool replicate_input,
        bool apply_weights, int m_block, int max_par, int cfg_max_m_blocks) {
    if (false) {
    }
    AWQ_CALL_IF_MOE_8(vllm::kU8, 16, 4, 256)
    AWQ_CALL_IF_MOE_8(vllm::kU8, 8, 8, 256)
    AWQ_CALL_IF_MOE_8(vllm::kU8, 8, 4, 128)
    AWQ_CALL_IF_MOE_8(vllm::kU8, 4, 8, 128)
    else {
        return false;
    }
    return true;
}

}  // namespace marlin_moe
