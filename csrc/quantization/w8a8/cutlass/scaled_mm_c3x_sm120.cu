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

// On GB10 (SM121) these FP8 GEMMs slow down 2-3x once a problem has more than
// about 3072 rows (an 8192 x 5120 x 16384 GEMM runs at 55 TFLOP/s against 170
// for 2048 rows), whatever the scales. Rows are independent, so larger problems
// run as a sequence of 1024-row GEMMs, which keep the full rate and give the
// same output bit for bit.
constexpr int64_t kGb10MaxRows = 3072;
constexpr int64_t kGb10RowChunk = 1024;

void cutlass_scaled_mm_sm120_fp8_rows(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias) {
  int64_t const m = a.size(0);
  bool const per_token = a_scales.numel() != 1;
  // Per-token scales are sliced with the rows, so they must be laid out by row.
  if (m <= kGb10MaxRows || get_sm_version_num() != 121 ||
      (per_token && a_scales.size(0) != m)) {
    vllm::cutlass_scaled_mm_sm120_fp8(c, a, b, a_scales, b_scales, bias);
    return;
  }
  // narrow() takes a non-const handle; these copy the handle, not the data.
  torch::stable::Tensor a_all = a, a_scales_all = a_scales;
  for (int64_t start = 0; start < m; start += kGb10RowChunk) {
    int64_t const rows = std::min(kGb10RowChunk, m - start);
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
