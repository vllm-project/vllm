#include <algorithm>

#include <torch/csrc/stable/ops.h>

#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm120 (Blackwell).
*/

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120

namespace {

// On GB10 (SM121) these FP8 GEMMs slow down 2-3x once A outgrows the L2: an
// 8192 x 16384 x 5120 GEMM (40 MiB of A) runs at 54 TFLOP/s against 170 when
// split. Rows are independent, so a problem with more than 12 MiB of A runs as
// row slices of at most 8 MiB of A, but at least 1024 rows to keep the SMs
// busy. The output is the same bit for bit.
constexpr int64_t kGb10SplitABytes = 12 << 20;
constexpr int64_t kGb10SliceABytes = 8 << 20;
constexpr int64_t kGb10MinSliceRows = 1024;

void cutlass_scaled_mm_sm120_fp8_rows(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias) {
  int64_t const m = a.size(0), row_bytes = a.size(1);  // FP8: 1 byte each
  int64_t const slice =
      std::max(kGb10MinSliceRows, kGb10SliceABytes / row_bytes / 128 * 128);
  bool const per_token = a_scales.numel() != 1;
  // Per-token scales are sliced with the rows, so they must be laid out by row.
  if (m * row_bytes <= kGb10SplitABytes || m <= slice ||
      get_sm_version_num() != 121 || (per_token && a_scales.size(0) != m)) {
    vllm::cutlass_scaled_mm_sm120_fp8(c, a, b, a_scales, b_scales, bias);
    return;
  }
  // narrow() takes a non-const handle; these copy the handle, not the data.
  torch::stable::Tensor a_all = a, a_scales_all = a_scales;
  for (int64_t start = 0; start < m; start += slice) {
    int64_t const rows = std::min(slice, m - start);
    torch::stable::Tensor c_rows = torch::stable::narrow(c, 0, start, rows);
    vllm::cutlass_scaled_mm_sm120_fp8(
        c_rows, torch::stable::narrow(a_all, 0, start, rows), b,
        per_token ? torch::stable::narrow(a_scales_all, 0, start, rows)
                  : a_scales,
        b_scales, bias);
  }
}

}  // namespace

void cutlass_scaled_mm_sm120(torch::stable::Tensor& c,
                             torch::stable::Tensor const& a,
                             torch::stable::Tensor const& b,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             std::optional<torch::stable::Tensor> const& bias) {
  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                     cutlass_scaled_mm_sm120_fp8_rows,
                     nullptr,  // int8 not supported on SM120
                     vllm::cutlass_scaled_mm_blockwise_sm120_fp8);
}

#endif
