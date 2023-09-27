#ifndef FUSED_H
#define FUSED_H

#include <torch/types.h>

std::tuple<torch::Tensor,
           torch::Tensor> // (residual_output (FP), ln_output (INT8))
dq_add_layernorm_q(torch::Tensor input,          // INT32
                   float input_scale,            // FP
                   torch::Tensor residual_input, // FP
                   torch::Tensor gamma,          // FP
                   torch::Tensor beta,           // FP
                   float epsilon                 // FP
); // The output scale has already been fused into gamma and beta

#endif // FUSED_H