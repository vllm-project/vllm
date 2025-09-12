#include <torch/all.h>
#include "cuda_utils.h"
#include "cutlass_extensions/common.hpp"

template <typename Fp8Func, typename Int8Func, typename BlockwiseFunc>
void dispatch_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                        torch::Tensor const& b, torch::Tensor const& a_scales,
                        torch::Tensor const& b_scales,
                        std::optional<torch::Tensor> const& bias,
                        Fp8Func fp8_func, Int8Func int8_func,
                        BlockwiseFunc blockwise_func) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  int M = a.size(0), N = b.size(1), K = a.size(1);

  if ((a_scales.numel() == 1 || a_scales.numel() == a.size(0)) &&
      (b_scales.numel() == 1 || b_scales.numel() == b.size(1))) {
    // Standard per-tensor/per-token/per-channel scaling
    TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == torch::kFloat8_e4m3fn) {
      fp8_func(c, a, b, a_scales, b_scales, bias);
    } else {
      TORCH_CHECK(a.dtype() == torch::kInt8);
      if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
        int8_func(c, a, b, a_scales, b_scales, bias);
      } else {
        TORCH_CHECK(false, "Int8 not supported for this architecture");
      }
    }
  } else {
    TORCH_CHECK(a_scales.dim() == 2, "a scale must be 2d tensor.");
    TORCH_CHECK(b_scales.dim() == 2, "b scale must be 2d tensor.");
    int32_t version_num = get_sm_version_num();
    if (version_num >= 90) {
      TORCH_CHECK(
          a.size(0) == a_scales.size(0) &&
              cuda_utils::ceil_div(a.size(1), int64_t(128)) == a_scales.size(1),
          "a_scale_group_shape must be [1, 128].");
      TORCH_CHECK(
          cuda_utils::ceil_div(b.size(0), int64_t(128)) == b_scales.size(0) &&
              cuda_utils::ceil_div(b.size(1), int64_t(128)) == b_scales.size(1),
          "b_scale_group_shape must be [128, 128].");
    }

    TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");
    blockwise_func(c, a, b, a_scales, b_scales);
  }
}
