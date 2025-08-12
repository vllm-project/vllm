#include <cudaTypedefs.h>
#include "c3x/scaled_mm_kernels.hpp"

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

  GroupShape a_scale_group_shape = [&, &s = a_scales]() -> GroupShape {
    if (s.numel() == 1) return {M, K};  // tensor-wise
    if (s.dim() == 2)
      return {ceil_div(a.size(0), s.size(0)), ceil_div(a.size(1), s.size(1))};
    TORCH_CHECK(false, "Unsupported scale shape for scale_a");
  }();

  GroupShape b_scale_group_shape = [&, &s = b_scales]() -> GroupShape {
    if (s.numel() == 1) return {K, N};  // tensor-wise
    if (s.dim() == 2)
      return {ceil_div(b.size(0), s.size(0)), ceil_div(b.size(1), s.size(1))};
    TORCH_CHECK(false, "Unsupported scale shape for scale_b");
  }();

  if ((a_scale_group_shape == GroupShape{M, K} ||
       a_scale_group_shape == GroupShape{1, K}) &&
      (b_scale_group_shape == GroupShape{K, N} ||
       b_scale_group_shape == GroupShape{K, 1})) {
    // "standard per-tensor/per-token/per-channel" scaling
    TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == torch::kFloat8_e4m3fn) {
      vllm::cutlass_scaled_mm_sm90_fp8(c, a, b, a_scales, b_scales, bias);
    } else {
      TORCH_CHECK(a.dtype() == torch::kInt8);
      vllm::cutlass_scaled_mm_sm90_int8(c, a, b, a_scales, b_scales, bias);
    }
  } else if (a_scale_group_shape == GroupShape{1, 128} &&
             b_scale_group_shape == GroupShape{128, 128}) {
    // 1x128 per-token group scales for activations
    // 128x128 blockwise scales for weights
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn &&
                    b.dtype() == torch::kFloat8_e4m3fn,
                "Currently only FP8 is supported for A group shape 1x128 and "
                "B group shape 128x128");
    TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");

    vllm::cutlass_scaled_mm_blockwise_sm90_fp8(c, a, b, a_scales, b_scales);
  } else {
    TORCH_CHECK(false,
                "Unsupported scale group shapes for CUTLASS 3.x GEMM.\n "
                "a_scale_group_shape must be [1, 128], got: [",
                a_scale_group_shape[0], ", ", a_scale_group_shape[1],
                "]\n"
                "b_scale_group_shape must be [128, 128], got: [",
                b_scale_group_shape[0], ", ", b_scale_group_shape[1], "]");
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
