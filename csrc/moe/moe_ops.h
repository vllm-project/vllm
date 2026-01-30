#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_sigmoid(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(torch::Tensor topk_ids, torch::Tensor lora_ids,
                               torch::Tensor adapter_enabled,
                               torch::Tensor token_lora_mapping,
                               int64_t num_experts_lora, int64_t max_loras,
                               int64_t block_size,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_pad,
                               std::optional<torch::Tensor> maybe_expert_map);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);