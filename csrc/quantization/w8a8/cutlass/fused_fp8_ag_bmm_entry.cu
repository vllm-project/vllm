#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"
#include "ops.h"

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

  int64_t a_bytes = a.numel() * a.element_size();
  TORCH_CHECK(reg_buffer_sz_bytes >= a_bytes,
              "reg_buffer is too small for fused_all_gather_bmm_fp8 staging");

  auto gathered_a = torch::empty({m * world_size, k}, a.options());
  all_gather(custom_ar_ptr, const_cast<torch::Tensor&>(a), gathered_a,
             reg_buffer, reg_buffer_sz_bytes);

  auto out = torch::empty({m * world_size, n}, a.options().dtype(out_dtype));
  cutlass_scaled_mm(out, gathered_a, b, a_scale, b_scale, std::nullopt);
  return out;
}
