#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

#include "stable/torch_utils.h"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper).
*/

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90

void cutlass_scaled_mm_sm90(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias) {
  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                     vllm::cutlass_scaled_mm_sm90_fp8,
                     vllm::cutlass_scaled_mm_sm90_int8,
                     vllm::cutlass_scaled_mm_blockwise_sm90_fp8);
}

void cutlass_scaled_mm_azp_sm90(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

  vllm::cutlass_scaled_mm_azp_sm90_int8(out, a, b, a_scales, b_scales, azp_adj,
                                        azp, bias);
}

#endif
