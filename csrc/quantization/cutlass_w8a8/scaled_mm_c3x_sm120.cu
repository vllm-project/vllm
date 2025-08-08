#include <cudaTypedefs.h>
#include "c3x/scaled_mm_kernels.hpp"

#include "cuda_utils.h"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm120 (Blackwell Geforce).
*/

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120

void cutlass_scaled_mm_sm120(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  int M = a.size(0), N = b.size(1), K = a.size(1);
  TORCH_CHECK(
      (a_scales.numel() == 1 || a_scales.numel() == a.size(0)) &&
          (b_scales.numel() == 1 || b_scales.numel() == b.size(1)),
      "Currently, block scaled fp8 gemm is not implemented for Blackwell");

  // Standard per-tensor/per-token/per-channel scaling
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn,
              "Currently, only fp8 gemm is implemented for Blackwell");
  vllm::cutlass_scaled_mm_sm120_fp8(c, a, b, a_scales, b_scales, bias);
}

#endif
