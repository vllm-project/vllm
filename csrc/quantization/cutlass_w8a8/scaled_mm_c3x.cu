#include <cudaTypedefs.h>
#include "c3x/scaled_mm_c3x_kernels.hpp"

#include "core/math.hpp"

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

  using GroupShape = std::array<int64_t, 2>;

  int M = a.size(0), N = b.size(1), K = a.size(1);

  GroupShape scale_group_shape_a = [&, &s = a_scales]() -> GroupShape {
    if (s.numel() == 1) return {M, K};  // tensor-wise
    if (s.dim() == 2)
      return {ceil_div(a.size(0), s.size(0)), ceil_div(a.size(1), s.size(1))};
    TORCH_CHECK(false, "Unsupported scale shape for scale_a");
  }();

  GroupShape scale_group_shape_b = [&, &s = b_scales]() -> GroupShape {
    if (s.numel() == 1) return {K, N};  // tensor-wise
    if (s.dim() == 2)
      return {ceil_div(b.size(0), s.size(0)), ceil_div(b.size(1), s.size(1))};
    TORCH_CHECK(false, "Unsupported scale shape for scale_b");
  }();

  if ((scale_group_shape_a == GroupShape{M, K} ||
       scale_group_shape_a == GroupShape{1, K}) &&
      (scale_group_shape_b == GroupShape{K, N} ||
       scale_group_shape_b == GroupShape{K, 1})) {
    // "standard per-tensor/per-token/per-channel" scaling
    TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == torch::kFloat8_e4m3fn) {
      vllm::cutlass_scaled_mm_sm90_fp8(c, a, b, a_scales, b_scales, bias);
    } else {
      TORCH_CHECK(a.dtype() == torch::kInt8);
      vllm::cutlass_scaled_mm_sm90_int8(c, a, b, a_scales, b_scales, bias);
    }
  } else if (scale_group_shape_a == GroupShape{1, 128} &&
             scale_group_shape_b == GroupShape{128, 128}) {
    // 1x128 per-token group scales for activations
    // 128x128 blockwise scales for weights
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn &&
                    b.dtype() == torch::kFloat8_e4m3fn,
                "Currently only FP8 is supported for A group shape 1x128 and "
                "B group shape 128x128");
    TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");

    vllm::cutlass_scaled_mm_blockwise_sm90_fp8(c, a, b, a_scales, b_scales);
  } else {
    TORCH_CHECK(false, "Unsupported scale group shapes for CUTLASS 3.x GEMM");
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
