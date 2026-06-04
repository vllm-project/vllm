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

bool moe_permute_unpermute_supported();

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert);

void shuffle_rows(const torch::stable::Tensor& input_tensor,
                  const torch::stable::Tensor& dst2src_map,
                  torch::stable::Tensor& output_tensor);

void moe_permute(const torch::stable::Tensor& input,
                 const torch::stable::Tensor& topk_ids,
                 const torch::stable::Tensor& token_expert_indices,
                 const std::optional<torch::stable::Tensor>& expert_map,
                 int64_t n_expert, int64_t n_local_expert, int64_t topk,
                 torch::stable::Tensor& permuted_input,
                 torch::stable::Tensor& expert_first_token_offset,
                 torch::stable::Tensor& inv_permuted_idx,
                 torch::stable::Tensor& permuted_idx);

void moe_permute_with_scratch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& token_expert_indices,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk, torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& inv_permuted_idx,
    torch::stable::Tensor& permuted_idx, torch::stable::Tensor& sort_workspace,
    torch::stable::Tensor& permuted_experts_id,
    torch::stable::Tensor& sorted_row_idx,
    torch::stable::Tensor& topk_ids_for_sort);

void moe_unpermute(
    const torch::stable::Tensor& permuted_hidden_states,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& inv_permuted_idx,
    const std::optional<torch::stable::Tensor>& expert_first_token_offset,
    int64_t topk, torch::stable::Tensor& hidden_states);

#ifndef USE_ROCM
std::tuple<torch::stable::Tensor, torch::stable::Tensor> grouped_topk(
    const torch::stable::Tensor& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    const torch::stable::Tensor& bias, int64_t scoring_func = 0);

void mxfp8_experts_quant(const torch::stable::Tensor& input,
                         const torch::stable::Tensor& problem_sizes,
                         const torch::stable::Tensor& expert_offsets,
                         const torch::stable::Tensor& blockscale_offsets,
                         torch::stable::Tensor& quant_output,
                         torch::stable::Tensor& scale_factor);

void cutlass_mxfp8_grouped_mm(const torch::stable::Tensor& a,
                              const torch::stable::Tensor& b,
                              const torch::stable::Tensor& sfa,
                              const torch::stable::Tensor& sfb,
                              torch::stable::Tensor& d,
                              const torch::stable::Tensor& problem_sizes,
                              const torch::stable::Tensor& expert_offsets,
                              const torch::stable::Tensor& blockscale_offsets);
#endif
