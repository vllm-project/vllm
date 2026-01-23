#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include "stable/torch_utils.h"
#include "cuda_utils.h"
#include "stable/cutlass_extensions/common.hpp"

template <typename Fp8Func, typename Int8Func, typename BlockwiseFunc>
void dispatch_scaled_mm(torch::stable::Tensor& c,
                        torch::stable::Tensor const& a,
                        torch::stable::Tensor const& b,
                        torch::stable::Tensor const& a_scales,
                        torch::stable::Tensor const& b_scales,
                        std::optional<torch::stable::Tensor> const& bias,
                        Fp8Func fp8_func, Int8Func int8_func,
                        BlockwiseFunc blockwise_func) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

  int M = a.size(0), N = b.size(1), K = a.size(1);

  if ((a_scales.numel() == 1 || a_scales.numel() == a.size(0)) &&
      (b_scales.numel() == 1 || b_scales.numel() == b.size(1))) {
    // Standard per-tensor/per-token/per-channel scaling
    STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
      fp8_func(c, a, b, a_scales, b_scales, bias);
    } else {
      STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
      if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
        int8_func(c, a, b, a_scales, b_scales, bias);
      } else {
        int32_t version_num = get_sm_version_num();
        STD_TORCH_CHECK(
            false, "Int8 not supported on SM", version_num,
            ". Use FP8 quantization instead, or run on older arch (SM < 100).");
      }
    }
  } else {
    STD_TORCH_CHECK(a_scales.dim() == 2, "a scale must be 2d tensor.");
    STD_TORCH_CHECK(b_scales.dim() == 2, "b scale must be 2d tensor.");
    int32_t version_num = get_sm_version_num();
    if (version_num >= 90) {
      STD_TORCH_CHECK(
          a.size(0) == a_scales.size(0) &&
              cuda_utils::ceil_div(a.size(1), int64_t(128)) == a_scales.size(1),
          "a_scale_group_shape must be [1, 128].");
      STD_TORCH_CHECK(
          cuda_utils::ceil_div(b.size(0), int64_t(128)) == b_scales.size(0) &&
              cuda_utils::ceil_div(b.size(1), int64_t(128)) == b_scales.size(1),
          "b_scale_group_shape must be [128, 128].");
    }

    STD_TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");
    blockwise_func(c, a, b, a_scales, b_scales);
  }
}
