#include <cudaTypedefs.h>
#include "c3x/scaled_mm_kernels.hpp"

#include "cuda_utils.h"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper) or later.
*/

void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  int M = a.size(0), N = b.size(1), K = a.size(1);

  if ((a_scales.numel() == 1 || a_scales.numel() == a.size(0)) &&
      (b_scales.numel() == 1 || b_scales.numel() == b.size(1))) {
    // Standard per-tensor/per-token/per-channel scaling
    TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == torch::kFloat8_e4m3fn) {
      vllm::cutlass_scaled_mm_sm90_fp8(c, a, b, a_scales, b_scales, bias);
    } else {
      TORCH_CHECK(a.dtype() == torch::kInt8);
      vllm::cutlass_scaled_mm_sm90_int8(c, a, b, a_scales, b_scales, bias);
    }
  } else {
    using GroupShape = std::array<int64_t, 2>;
    auto make_group_shape = [](torch::Tensor const& x,
                               torch::Tensor const& s) -> GroupShape {
      TORCH_CHECK(s.dim() == 2, "cutlass_scaled_mm group scales must be 2D");
      return {cuda_utils::ceil_div(x.size(0), s.size(0)),
              cuda_utils::ceil_div(x.size(1), s.size(1))};
    };

    GroupShape a_scale_group_shape = make_group_shape(a, a_scales);
    GroupShape b_scale_group_shape = make_group_shape(b, b_scales);

    // 1x128 per-token group scales for activations
    // 128x128 blockwise scales for weights
    TORCH_CHECK((a_scale_group_shape == GroupShape{1, 128} &&
                 b_scale_group_shape == GroupShape{128, 128} &&
                 a.dtype() == torch::kFloat8_e4m3fn &&
                 b.dtype() == torch::kFloat8_e4m3fn),
                "cutlass_scaled_mm only supports datatype float8_e4m3fn.\n"
                "a_scale_group_shape must be [1, 128]. Got: [",
                a_scale_group_shape[0], ", ", a_scale_group_shape[1],
                "]\n"
                "b_scale_group_shape must be [128, 128]. Got: [",
                b_scale_group_shape[0], ", ", b_scale_group_shape[1], "]");
    TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");

    vllm::cutlass_scaled_mm_blockwise_sm90_fp8(c, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm90(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  vllm::cutlass_scaled_mm_azp_sm90_int8(out, a, b, a_scales, b_scales, azp_adj,
                                        azp, bias);
}
