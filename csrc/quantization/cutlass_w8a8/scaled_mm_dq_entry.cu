#include <assert.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void cutlass_scaled_mm_dq_sm75(torch::Tensor &c, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

void cutlass_scaled_mm_dq_sm80(torch::Tensor &c, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

void cutlass_scaled_mm_dq_sm89(torch::Tensor &c, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

void cutlass_scaled_mm_dq_sm90(torch::Tensor &c, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

void cutlass_scaled_mm_dq(torch::Tensor &c, torch::Tensor const &a,
                          torch::Tensor const &b, torch::Tensor const &a_scales,
                          torch::Tensor const &b_scales) {
  int32_t major_capability;
  int32_t minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;

  // Checks for conformality
  assert(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  assert(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
         b.size(1) == c.size(1));
  assert(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  assert(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  assert(a.stride(1) == 1 && c.stride(1) == 1);           // Row-major
  assert(b.stride(0) == 1);                               // Column-major
  assert(c.stride(0) % 16 == 0 && b.stride(1) % 16 == 0); // 16 Byte Alignment
  assert(a_scales.is_contiguous() && b_scales.is_contiguous());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  if (version_num >= 90) {
    // Hopper
    cutlass_scaled_mm_dq_sm90(c, a, b, a_scales, b_scales);
  } else if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_dq_sm89(c, a, b, a_scales, b_scales);
  } else if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_dq_sm80(c, a, b, a_scales, b_scales);
  } else {
    // Turing
    TORCH_CHECK(version_num >= 75);
    cutlass_scaled_mm_dq_sm75(c, a, b, a_scales, b_scales);
  }
}
