#include <algorithm>
#include <array>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"
#include "ops.h"

// Fake pointer type matching ops.h.
using fptr_t = int64_t;

namespace {

// Call cutlass_scaled_mm via PyTorch dispatcher.
// The implementation lives in _C_stable_libtorch.so (libtorch_stable path).
void call_cutlass_scaled_mm(torch::Tensor& out, const torch::Tensor& a,
                            const torch::Tensor& b,
                            const torch::Tensor& a_scale,
                            const torch::Tensor& b_scale) {
  static c10::OperatorHandle op =
      c10::Dispatcher::singleton().findSchemaOrThrow("_C::cutlass_scaled_mm",
                                                     "");
  op.typed<void(at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&,
                const c10::optional<at::Tensor>&)>()
      .call(out, a, b, a_scale, b_scale, c10::nullopt);
}

inline int64_t element_size_from_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      return 2;
    default:
      return 0;
  }
}

inline int64_t select_chunk_rows(int64_t buffer_bytes, int64_t bytes_per_row,
                                 int64_t world_size, int64_t max_rows) {
  int64_t slot_bytes = buffer_bytes / 2;
  int64_t rows = slot_bytes / bytes_per_row;
  rows = (rows / world_size) * world_size;
  return std::min(rows, max_rows);
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
  int64_t chunk_rows =
      select_chunk_rows(reg_buffer_sz_bytes, bytes_per_row, world_size, m);
  TORCH_CHECK(chunk_rows > 0,
              "reg_buffer is too small for fused_bmm_fp8_reduce_scatter");
  int64_t slot_bytes = chunk_rows * bytes_per_row;

  auto slot_options = a.options().dtype(out_dtype);
  std::array<torch::Tensor, 2> slots{
      torch::from_blob(reinterpret_cast<void*>(reg_buffer), {chunk_rows, n},
                       slot_options),
      torch::from_blob(reinterpret_cast<void*>(reg_buffer + slot_bytes),
                       {chunk_rows, n}, slot_options),
  };

  auto caller_stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  auto compute_stream = c10::cuda::getStreamFromPool(false, a.get_device());
  auto comm_stream = c10::cuda::getStreamFromPool(false, a.get_device());

  at::cuda::CUDAEvent caller_ready(cudaEventDisableTiming);
  caller_ready.record(caller_stream);
  caller_ready.block(compute_stream);
  caller_ready.block(comm_stream);

  std::array<at::cuda::CUDAEvent, 2> gemm_done{
      at::cuda::CUDAEvent(cudaEventDisableTiming),
      at::cuda::CUDAEvent(cudaEventDisableTiming),
  };
  std::array<at::cuda::CUDAEvent, 2> rs_done{
      at::cuda::CUDAEvent(cudaEventDisableTiming),
      at::cuda::CUDAEvent(cudaEventDisableTiming),
  };

  int64_t tile_idx = 0;
  for (int64_t row_start = 0; row_start < m;
       row_start += chunk_rows, ++tile_idx) {
    int slot = static_cast<int>(tile_idx % 2);
    int64_t rows = std::min<int64_t>(chunk_rows, m - row_start);
    TORCH_CHECK(rows % world_size == 0,
                "chunk row count must be divisible by world_size");

    if (tile_idx >= 2) {
      rs_done[slot].block(compute_stream);
    }

    auto a_chunk = a.narrow(0, row_start, rows);
    auto slot_chunk = slots[slot].narrow(0, 0, rows);
    {
      c10::cuda::CUDAStreamGuard guard(compute_stream);
      call_cutlass_scaled_mm(slot_chunk, a_chunk, b, a_scale, b_scale);
      gemm_done[slot].record(compute_stream);
    }

    if (tile_idx > 0) {
      int prev_slot = static_cast<int>((tile_idx - 1) % 2);
      int64_t prev_row_start = row_start - chunk_rows;
      int64_t prev_rows = std::min<int64_t>(chunk_rows, m - prev_row_start);
      auto prev_chunk = slots[prev_slot].narrow(0, 0, prev_rows);
      auto out_chunk =
          out.narrow(0, prev_row_start / world_size, prev_rows / world_size);
      {
        c10::cuda::CUDAStreamGuard guard(comm_stream);
        gemm_done[prev_slot].block(comm_stream);
        reduce_scatter(custom_ar_ptr, prev_chunk, out_chunk,
                       /*reg_buffer=*/0, reg_buffer_sz_bytes);
        rs_done[prev_slot].record(comm_stream);
      }
    }
  }

  int last_slot = static_cast<int>((tile_idx - 1) % 2);
  int64_t last_row_start = (tile_idx - 1) * chunk_rows;
  int64_t last_rows = std::min<int64_t>(chunk_rows, m - last_row_start);
  auto last_chunk = slots[last_slot].narrow(0, 0, last_rows);
  auto last_out =
      out.narrow(0, last_row_start / world_size, last_rows / world_size);
  {
    c10::cuda::CUDAStreamGuard guard(comm_stream);
    gemm_done[last_slot].block(comm_stream);
    reduce_scatter(custom_ar_ptr, last_chunk, last_out,
                   /*reg_buffer=*/0, reg_buffer_sz_bytes);
    rs_done[last_slot].record(comm_stream);
  }
  rs_done[last_slot].block(caller_stream);

  return out;
}
