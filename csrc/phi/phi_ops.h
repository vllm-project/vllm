#pragma once

#include <torch/all.h>

namespace phi_c
{
    using torch::Tensor;
    void grouped_gemm(Tensor activations,
                             Tensor weights,
                             Tensor weight_scales,
                             Tensor total_rows_before_expert,
                             Tensor out,
                             int64_t activation_type,
                             int64_t config_id);
    
    Tensor preprocess_weights_for_mixed_gemm(Tensor row_major_quantized_weight);

    void moe_align_block_size(
        Tensor topk_ids,
        int64_t num_experts,
        int64_t block_size,
        Tensor sorted_token_ids,
        Tensor experts_ids,
        Tensor num_tokens_post_pad,
        Tensor expert_offset,
        Tensor expert_length
    );
}