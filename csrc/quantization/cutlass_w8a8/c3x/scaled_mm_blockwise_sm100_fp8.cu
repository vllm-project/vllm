#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm100_fp8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

// Pads to a multiple of `alignment` rows.
inline torch::Tensor pad_tensor(const torch::Tensor& tensor,
                                int64_t alignment = 4,
                                bool is_column_major = false) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t pad_rows = (alignment - (rows % alignment)) % alignment;

  if (pad_rows == 0) {
    return tensor;
  }

  torch::Tensor padding = torch::zeros({pad_rows, cols}, tensor.options());
  torch::Tensor tensor_padded = torch::cat({tensor, padding}, 0);

  // Ensure column-major layout
  if (is_column_major) {
    return tensor_padded.t().contiguous().t();
  }
  return tensor_padded;
}

void cutlass_scaled_mm_blockwise_sm100_fp8(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& a_scales,
                                           torch::Tensor const& b_scales) {
  int64_t original_rows = a.size(0);

  torch::Tensor a_padded = pad_tensor(a, /*alignment=*/4);
  torch::Tensor a_scales_padded =
      pad_tensor(a_scales, /*alignment=*/4, /*col_major=*/true);
  torch::Tensor out_padded;
  if (a_padded.size(0) == a.size(0)) {
    out_padded = out;
  } else {
    out_padded =
        torch::zeros({a_padded.size(0), b.size(1)}, out.options()).contiguous();
  }

  if (out.dtype() == torch::kBFloat16) {
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::bfloat16_t>(
        out_padded, a_padded, b, a_scales_padded, b_scales);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::half_t>(
        out_padded, a_padded, b, a_scales_padded, b_scales);
  }
  if (a_padded.size(0) != a.size(0)) {
    out.copy_(out_padded.slice(0, 0, original_rows));
  }
}

}  // namespace vllm
