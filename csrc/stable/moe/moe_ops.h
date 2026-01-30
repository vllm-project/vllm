#pragma once

#include <torch/csrc/stable/library.h>
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

void moe_sum(torch::stable::Tensor& input, torch::stable::Tensor& output);

void moe_align_block_size(
    torch::stable::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::stable::Tensor sorted_token_ids, torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad,
    std::optional<torch::stable::Tensor> maybe_expert_map);

void batched_moe_align_block_size(
    int64_t max_tokens_per_batch, int64_t block_size,
    torch::stable::Tensor const& expert_num_tokens,
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
    torch::stable::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::stable::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(torch::stable::Tensor const& input_tensor,
                  torch::stable::Tensor const& dst2src_map,
                  torch::stable::Tensor& output_tensor);
