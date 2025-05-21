#pragma once

#include <torch/all.h>

namespace vllm {

void cutlass_scaled_mm_sm90_fp8(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm90_int8(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm90_int8(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales,
                                     torch::Tensor const& azp_adj,
                                     std::optional<torch::Tensor> const& azp,
                                     std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm90_fp8(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          torch::Tensor const& a_scales,
                                          torch::Tensor const& b_scales);

void cutlass_scaled_mm_sm100_fp8(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm100_fp8(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& a_scales,
                                           torch::Tensor const& b_scales);
}  // namespace vllm
