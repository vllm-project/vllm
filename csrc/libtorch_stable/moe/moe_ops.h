#pragma once

#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <tuple>

void topk_softmax(torch::stable::Tensor& topk_weights,
                  torch::stable::Tensor& topk_indices,
                  torch::stable::Tensor& token_expert_indices,
                  torch::stable::Tensor& gating_output, bool renormalize,
                  std::optional<torch::stable::Tensor> bias);

void topk_sigmoid(torch::stable::Tensor& topk_weights,
                  torch::stable::Tensor& topk_indices,
                  torch::stable::Tensor& token_expert_indices,
                  torch::stable::Tensor& gating_output, bool renormalize,
                  std::optional<torch::stable::Tensor> bias);

void topk_softplus_sqrt(
    torch::stable::Tensor& topk_weights, torch::stable::Tensor& topk_indices,
    torch::stable::Tensor& token_expert_indices,
    torch::stable::Tensor& gating_output, bool renormalize,
    double routed_scaling_factor,
    const std::optional<torch::stable::Tensor>& correction_bias,
    const std::optional<torch::stable::Tensor>& input_ids,
    const std::optional<torch::stable::Tensor>& tid2eid);

void moe_sum(torch::stable::Tensor& input, torch::stable::Tensor& output);

void moe_align_block_size(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map);

void batched_moe_align_block_size(
    int64_t max_tokens_per_batch, int64_t block_size,
    const torch::stable::Tensor& expert_num_tokens,
    torch::stable::Tensor sorted_ids, torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::stable::Tensor topk_ids, torch::stable::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad,
    torch::stable::Tensor adapter_enabled, torch::stable::Tensor lora_ids,
    std::optional<torch::stable::Tensor> maybe_expert_map);
#ifndef USE_ROCM
torch::stable::Tensor moe_wna16_gemm(
    torch::stable::Tensor input, torch::stable::Tensor output,
    torch::stable::Tensor b_qweight, torch::stable::Tensor b_scales,
    std::optional<torch::stable::Tensor> b_qzeros,
    std::optional<torch::stable::Tensor> topk_weights,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad, int64_t top_k,
    int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N, int64_t BLOCK_SIZE_K,
    int64_t bit);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> grouped_topk(
    const torch::stable::Tensor& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    const torch::stable::Tensor& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t num_expert);

void shuffle_rows(const torch::stable::Tensor& input_tensor,
                  const torch::stable::Tensor& dst2src_map,
                  torch::stable::Tensor& output_tensor);

#ifndef USE_ROCM
// DeepSeek V3 optimized router GEMM kernel for SM90+
// Computes output = mat_a @ mat_b.T where:
//   mat_a: [num_tokens, hidden_dim] in bf16
//   mat_b: [num_experts, hidden_dim] in bf16
//   output: [num_tokens, num_experts] in bf16 or fp32
// Supports num_tokens in [1, 16], num_experts in {256, 384}, hidden_dim = 7168
void dsv3_router_gemm(torch::stable::Tensor& output,
                      const torch::stable::Tensor& mat_a,
                      const torch::stable::Tensor& mat_b);
#endif
