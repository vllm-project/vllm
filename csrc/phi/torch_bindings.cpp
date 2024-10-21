#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include "phi_ops.h"

#include <torch/library.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, phi_ops) {
    // Aligning the number of tokens to be processed by each expert such
    // that it is divisible by the block size.
    phi_ops.def(
        "moe_align_block_size(Tensor topk_ids,"
        "                     int num_experts,"
        "                     int block_size,"
        "                     Tensor! sorted_token_ids,"
        "                     Tensor! experts_ids,"
        "                     Tensor! num_tokens_post_pad,"
        "                     Tensor! expert_offset,"
        "                     Tensor! expert_length"
        ") -> ()");
    phi_ops.impl("moe_align_block_size", torch::kCUDA, &phi_c::moe_align_block_size);

    phi_ops.def(
        "grouped_gemm(Tensor activations,"
        "             Tensor weights,"
        "             Tensor weight_scales,"
        "             Tensor total_rows_before_expert,"
        "             Tensor! out,"
        "             int activation_type,"
        "             int config_id"
        ") -> ()");
    phi_ops.impl("grouped_gemm", torch::kCUDA, &phi_c::grouped_gemm);    

    phi_ops.def("preprocess_weights_for_mixed_gemm(Tensor row_major_quantized_weight) -> (Tensor processed_tensor)");
    phi_ops.impl("preprocess_weights_for_mixed_gemm", torch::kCPU, &phi_c::preprocess_weights_for_mixed_gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)