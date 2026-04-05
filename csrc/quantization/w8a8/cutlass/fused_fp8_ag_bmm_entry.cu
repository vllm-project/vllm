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

inline int64_t select_chunk_rows(int64_t buffer_bytes, int64_t bytes_per_row,
                                 int64_t max_rows) {
  int64_t slot_bytes = buffer_bytes / 2;
  return std::min(slot_bytes / bytes_per_row, max_rows);
}

}  // namespace

torch::Tensor fused_all_gather_bmm_fp8(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_scale, const torch::Tensor& b_scale,
    at::ScalarType out_dtype, fptr_t custom_ar_ptr, fptr_t reg_buffer,
    int64_t reg_buffer_sz_bytes, int64_t rank, int64_t world_size) {
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && a_scale.is_cuda() && b_scale.is_cuda(),
      "fused_all_gather_bmm_fp8 requires CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "fused_all_gather_bmm_fp8 expects 2D A/B");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "A must be float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "B must be float8_e4m3fn");
  TORCH_CHECK(a_scale.scalar_type() == at::ScalarType::Float &&
                  b_scale.scalar_type() == at::ScalarType::Float,
              "A_scale/B_scale must be float32");
  TORCH_CHECK(a_scale.numel() == 1 && b_scale.numel() == 1,
              "fused_all_gather_bmm_fp8 currently only supports per-tensor "
              "scaling");
  TORCH_CHECK(out_dtype == at::ScalarType::Half ||
                  out_dtype == at::ScalarType::BFloat16,
              "out_dtype must be float16 or bfloat16");
  TORCH_CHECK(a.size(1) == b.size(0), "A/B shape mismatch");
  TORCH_CHECK(world_size > 1, "world_size must be > 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(custom_ar_ptr != 0, "invalid custom_ar_ptr");
  TORCH_CHECK(reg_buffer != 0, "invalid reg_buffer pointer");
  TORCH_CHECK(reg_buffer_sz_bytes > 0, "reg_buffer_sz_bytes must be > 0");
  TORCH_CHECK(a.element_size() == 1,
              "fused_all_gather_bmm_fp8 expects 1-byte fp8 input");

  int32_t sm = get_sm_version_num();
  TORCH_CHECK(sm >= 100 && sm < 110,
              "fused_all_gather_bmm_fp8 currently supports SM100 only, got SM",
              sm);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  auto m = a.size(0);
  auto k = a.size(1);
  auto n = b.size(1);

  int64_t bytes_per_input_row = k * a.element_size();
  TORCH_CHECK(bytes_per_input_row > 0, "invalid input row size");
  int64_t chunk_rows =
      select_chunk_rows(reg_buffer_sz_bytes, bytes_per_input_row, m);
  TORCH_CHECK(chunk_rows > 0,
              "reg_buffer is too small for fused_all_gather_bmm_fp8 staging");
  int64_t slot_bytes = chunk_rows * bytes_per_input_row;

  auto input_slot_options = a.options();
  std::array<torch::Tensor, 2> input_slots{
      torch::from_blob(reinterpret_cast<void*>(reg_buffer), {chunk_rows, k},
                       input_slot_options),
      torch::from_blob(reinterpret_cast<void*>(reg_buffer + slot_bytes),
                       {chunk_rows, k}, input_slot_options),
  };
  std::array<torch::Tensor, 2> gathered_slots{
      torch::empty({chunk_rows * world_size, k}, a.options()),
      torch::empty({chunk_rows * world_size, k}, a.options()),
  };
  std::array<torch::Tensor, 2> mm_slots{
      torch::empty({chunk_rows * world_size, n}, a.options().dtype(out_dtype)),
      torch::empty({chunk_rows * world_size, n}, a.options().dtype(out_dtype)),
  };

  auto out = torch::empty({m * world_size, n}, a.options().dtype(out_dtype));

  auto caller_stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  auto comm_stream = c10::cuda::getStreamFromPool(false, a.get_device());
  auto compute_stream = c10::cuda::getStreamFromPool(false, a.get_device());

  at::cuda::CUDAEvent caller_ready(cudaEventDisableTiming);
  caller_ready.record(caller_stream);
  caller_ready.block(comm_stream);
  caller_ready.block(compute_stream);

  std::array<at::cuda::CUDAEvent, 2> gather_done{
      at::cuda::CUDAEvent(cudaEventDisableTiming),
      at::cuda::CUDAEvent(cudaEventDisableTiming),
  };
  std::array<at::cuda::CUDAEvent, 2> compute_done{
      at::cuda::CUDAEvent(cudaEventDisableTiming),
      at::cuda::CUDAEvent(cudaEventDisableTiming),
  };

  auto enqueue_gather = [&](int64_t tile_idx) {
    int slot = static_cast<int>(tile_idx % 2);
    int64_t row_start = tile_idx * chunk_rows;
    int64_t rows = std::min<int64_t>(chunk_rows, m - row_start);

    if (tile_idx >= 2) {
      compute_done[slot].block(comm_stream);
    }

    auto a_chunk = a.narrow(0, row_start, rows);
    auto input_chunk = input_slots[slot].narrow(0, 0, rows);
    auto gathered_chunk = gathered_slots[slot].narrow(0, 0, rows * world_size);

    {
      c10::cuda::CUDAStreamGuard guard(comm_stream);
      AT_CUDA_CHECK(cudaMemcpyAsync(input_chunk.data_ptr(), a_chunk.data_ptr(),
                                    rows * bytes_per_input_row,
                                    cudaMemcpyDeviceToDevice,
                                    comm_stream.stream()));
      all_gather(custom_ar_ptr, input_chunk, gathered_chunk,
                 /*reg_buffer=*/0, reg_buffer_sz_bytes);
      gather_done[slot].record(comm_stream);
    }
  };

  int64_t num_tiles = (m + chunk_rows - 1) / chunk_rows;
  enqueue_gather(0);

  for (int64_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int slot = static_cast<int>(tile_idx % 2);
    int64_t row_start = tile_idx * chunk_rows;
    int64_t rows = std::min<int64_t>(chunk_rows, m - row_start);

    auto gathered_chunk = gathered_slots[slot].narrow(0, 0, rows * world_size);
    auto mm_chunk = mm_slots[slot].narrow(0, 0, rows * world_size);

    {
      c10::cuda::CUDAStreamGuard guard(compute_stream);
      gather_done[slot].block(compute_stream);
      call_cutlass_scaled_mm(mm_chunk, gathered_chunk, b, a_scale, b_scale);
      for (int64_t peer = 0; peer < world_size; ++peer) {
        auto src_rows = mm_chunk.narrow(0, peer * rows, rows);
        auto dst_rows = out.narrow(0, peer * m + row_start, rows);
        dst_rows.copy_(src_rows);
      }
      compute_done[slot].record(compute_stream);
    }

    if (tile_idx + 1 < num_tiles) {
      enqueue_gather(tile_idx + 1);
    }
  }

  compute_done[(num_tiles - 1) % 2].block(caller_stream);
  return out;
}
