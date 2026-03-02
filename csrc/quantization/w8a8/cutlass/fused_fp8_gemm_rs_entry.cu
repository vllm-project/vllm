#include <algorithm>

#include <ATen/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"
#include "ops.h"

namespace {

inline int64_t element_size_from_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      return 2;
    default:
      return 0;
  }
}

}  // namespace

torch::Tensor fused_bmm_fp8_reduce_scatter(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_scale, const torch::Tensor& b_scale,
    at::ScalarType out_dtype, fptr_t custom_ar_ptr, fptr_t reg_buffer,
    int64_t reg_buffer_sz_bytes, int64_t rank, int64_t world_size) {
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && a_scale.is_cuda() && b_scale.is_cuda(),
      "fused_bmm_fp8_reduce_scatter requires CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "fused_bmm_fp8_reduce_scatter expects 2D A/B");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "A must be float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "B must be float8_e4m3fn");
  TORCH_CHECK(a_scale.scalar_type() == at::ScalarType::Float &&
                  b_scale.scalar_type() == at::ScalarType::Float,
              "A_scale/B_scale must be float32");
  TORCH_CHECK(a_scale.numel() == 1 && b_scale.numel() == 1,
              "fused_bmm_fp8_reduce_scatter currently only supports per-tensor "
              "scaling");
  TORCH_CHECK(out_dtype == at::ScalarType::Half ||
                  out_dtype == at::ScalarType::BFloat16,
              "out_dtype must be float16 or bfloat16");
  TORCH_CHECK(a.size(1) == b.size(0), "A/B shape mismatch");
  TORCH_CHECK(world_size > 1, "world_size must be > 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(a.size(0) % world_size == 0,
              "M dimension must be divisible by world_size");
  TORCH_CHECK(custom_ar_ptr != 0, "invalid custom_ar_ptr");
  TORCH_CHECK(reg_buffer != 0, "invalid reg_buffer pointer");
  TORCH_CHECK(reg_buffer_sz_bytes > 0, "reg_buffer_sz_bytes must be > 0");

  int32_t sm = get_sm_version_num();
  TORCH_CHECK(
      sm >= 100 && sm < 110,
      "fused_bmm_fp8_reduce_scatter currently supports SM100 only, got SM", sm);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  auto m = a.size(0);
  auto n = b.size(1);
  auto out = torch::empty({m / world_size, n}, a.options().dtype(out_dtype));

  int64_t elem_size = element_size_from_dtype(out_dtype);
  TORCH_CHECK(elem_size > 0, "unsupported output dtype");
  int64_t bytes_per_row = n * elem_size;
  TORCH_CHECK(bytes_per_row > 0, "invalid bytes_per_row");
  int64_t max_rows_in_buffer = reg_buffer_sz_bytes / bytes_per_row;
  int64_t chunk_rows = (max_rows_in_buffer / world_size) * world_size;
  TORCH_CHECK(chunk_rows > 0,
              "reg_buffer is too small for fused_bmm_fp8_reduce_scatter");
  chunk_rows = std::min<int64_t>(chunk_rows, m);

  // Treat the already IPC-registered buffer as the GEMM staging area.
  // This avoids per-chunk temporary allocations and avoids an extra D2D copy
  // before reduce_scatter.
  auto staging =
      torch::from_blob(reinterpret_cast<void*>(reg_buffer), {chunk_rows, n},
                       a.options().dtype(out_dtype));

  for (int64_t row_start = 0; row_start < m; row_start += chunk_rows) {
    int64_t rows = std::min<int64_t>(chunk_rows, m - row_start);
    TORCH_CHECK(rows % world_size == 0,
                "chunk row count must be divisible by world_size");

    auto a_chunk = a.narrow(0, row_start, rows);
    auto staging_chunk = staging.narrow(0, 0, rows);
    cutlass_scaled_mm(staging_chunk, a_chunk, b, a_scale, b_scale,
                      std::nullopt);

    auto out_chunk = out.narrow(0, row_start / world_size, rows / world_size);
    reduce_scatter(custom_ar_ptr, staging_chunk, out_chunk,
                   /*reg_buffer=*/0, reg_buffer_sz_bytes);
  }

  return out;
}
