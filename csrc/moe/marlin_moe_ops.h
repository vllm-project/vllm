#pragma once

torch::Tensor marlin_gemm_moe(torch::Tensor& a, torch::Tensor& b_q_weights, torch::Tensor& sorted_ids,
                        torch::Tensor& b_scales, torch::Tensor& workspace,
                        int64_t size_m, int64_t size_n, int64_t size_k/*,
                        int64_t num_tokens_post_padded, int64_t num_experts, int64_t moe_block_size*/);
