#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32(torch::Tensor input,  // INT8
                                   torch::Tensor weight, // INT8
                                   torch::Tensor bias    // INT32
);

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32_with_scaling(torch::Tensor input,  // INT8
                                                torch::Tensor weight, // INT8
                                                torch::Tensor bias,   // INT32
                                                float alpha,          // FP32
                                                float beta            // FP32
);

// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                       float alpha,          // FP32
                                       float beta            // FP32
);

// used by fc1, return INT8
torch::Tensor linear_relu_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
                                      torch::Tensor bias,   // INT8
                                      float alpha,          // FP32
                                      float beta            // FP32
);

// used by q_proj, k_proj, v_proj, return INT8
torch::Tensor linear_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                 torch::Tensor weight, // INT8
                                 torch::Tensor bias,   // INT8
                                 float alpha,          // FP32
                                 float beta            // FP32
);

#endif // LINEAR_HS