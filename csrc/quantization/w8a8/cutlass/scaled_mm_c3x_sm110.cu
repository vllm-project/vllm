#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm110 (Blackwell).
*/

#if defined ENABLE_SCALED_MM_SM110 && ENABLE_SCALED_MM_SM110

void cutlass_scaled_mm_sm110(torch::stable::Tensor& c,
                             torch::stable::Tensor const& a,
                             torch::stable::Tensor const& b,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             std::optional<torch::stable::Tensor> const& bias) {
  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                     vllm::cutlass_scaled_mm_sm110_fp8,
                     nullptr,  // int8 not supported on SM110
                     vllm::cutlass_scaled_mm_blockwise_sm110_fp8);
}

#endif
