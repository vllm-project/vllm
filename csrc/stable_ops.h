#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

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
