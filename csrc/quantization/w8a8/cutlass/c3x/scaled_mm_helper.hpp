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
    if (version_num >= 100) {
      TORCH_CHECK(
          a.size(0) == a_scales.size(0) &&
              cuda_utils::ceil_div(a.size(1), int64_t(128)) == a_scales.size(1),
          "a_scale_group_shape must be [1, 128].");
      TORCH_CHECK(
          cuda_utils::ceil_div(b.size(0), int64_t(128)) == b_scales.size(0) &&
              cuda_utils::ceil_div(b.size(1), int64_t(128)) == b_scales.size(1),
          "b_scale_group_shape must be [128, 128].");
    } else {
      // TODO: Remove this after using cutlass sm90 blockwise scaling gemm
      // kernel, or introducing ceil_div to the load_init() of mainloop.
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
    }

    TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");
    blockwise_func(c, a, b, a_scales, b_scales);
  }
}
