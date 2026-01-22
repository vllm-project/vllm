#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm100_fp8_dispatch.cuh"
#include "stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "stable/torch_utils.h"
#include <torch/headeronly/core/ScalarType.h>

namespace vllm {

void cutlass_scaled_mm_blockwise_sm100_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);

  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
