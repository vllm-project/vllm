#pragma once

#include <torch/all.h>

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

#ifndef USE_ROCM
// DeepSeek V3 optimized router GEMM kernel for SM90+
// Computes output = mat_a @ mat_b.T where:
//   mat_a: [num_tokens, hidden_dim] in bf16
//   mat_b: [num_experts, hidden_dim] in bf16
//   output: [num_tokens, num_experts] in bf16 or fp32
// Supports num_tokens in [1, 16], num_experts in {256, 384}, hidden_dim = 7168
void dsv3_router_gemm(torch::Tensor& output, const torch::Tensor& mat_a,
                      const torch::Tensor& mat_b);
#endif
