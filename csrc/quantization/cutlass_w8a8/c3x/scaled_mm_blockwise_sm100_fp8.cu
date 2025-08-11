#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm100_fp8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

void cutlass_scaled_mm_blockwise_sm100_fp8(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& a_scales,
                                           torch::Tensor const& b_scales) {
  if (out.dtype() == torch::kBFloat16) {
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);

  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
