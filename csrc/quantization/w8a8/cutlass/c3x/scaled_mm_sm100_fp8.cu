#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm100_fp8_dispatch.cuh"

namespace vllm {

void cutlass_scaled_mm_sm100_fp8(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm100_fp8_epilogue<true>(out, a, b, a_scales,
                                                      b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm100_fp8_epilogue<false>(out, a, b, a_scales,
                                                       b_scales);
  }
}

}  // namespace vllm
