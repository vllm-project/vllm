#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

// Gated activation functions (input: [..., 2*d] -> output: [..., d])
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void mul_and_silu(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_tanh_and_mul(torch::stable::Tensor& out,
                       torch::stable::Tensor& input);
void fatrelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                       double alpha, double limit);

// Element-wise activation functions (input: [..., d] -> output: [..., d])
void gelu_new(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_fast(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_quick(torch::stable::Tensor& out, torch::stable::Tensor& input);

// Utility functions
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor);

#ifndef USE_ROCM
torch::stable::Tensor permute_cols(torch::stable::Tensor const& A,
                                   torch::stable::Tensor const& perm);
#endif

// Layernorm functions
void rms_norm(torch::stable::Tensor& out, torch::stable::Tensor input,
              const torch::stable::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::stable::Tensor& input,
                        torch::stable::Tensor& residual,
                        torch::stable::Tensor& weight, double epsilon);

// Layernorm + quantization functions
void rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                               const torch::stable::Tensor& input,
                               const torch::stable::Tensor& weight,
                               const torch::stable::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                                         const torch::stable::Tensor& input,
                                         torch::stable::Tensor& residual,
                                         const torch::stable::Tensor& weight,
                                         const torch::stable::Tensor& scale,
                                         double epsilon);

// Fused layernorm + dynamic per-token quantization
void rms_norm_dynamic_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor const& weight, torch::stable::Tensor& scales,
    double epsilon, std::optional<torch::stable::Tensor> scale_ub,
    std::optional<torch::stable::Tensor> residual);

// Fused layernorm + block quantization
void rms_norm_per_block_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor const& weight,
                              torch::stable::Tensor& scales,
                              double const epsilon,
                              std::optional<torch::stable::Tensor> scale_ub,
                              std::optional<torch::stable::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);
