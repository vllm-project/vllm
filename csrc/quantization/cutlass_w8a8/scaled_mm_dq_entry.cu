#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

void cutlass_scaled_mm_dq_sm75(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales);

void cutlass_scaled_mm_dq_sm80(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales);

void cutlass_scaled_mm_dq_sm89(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales);

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
void cutlass_scaled_mm_dq_sm90(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales);
#endif

void cutlass_scaled_mm_dq(torch::Tensor& c, torch::Tensor const& a,
                          torch::Tensor const& b, torch::Tensor const& a_scales,
                          torch::Tensor const& b_scales) {
  int32_t major_capability;
  int32_t minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;

  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  if (version_num >= 90) {
    // Hopper

    // Guard against compilation issues for sm90 kernels
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
    cutlass_scaled_mm_dq_sm90(c, a, b, a_scales, b_scales);
#else
    cutlass_scaled_mm_dq_sm80(c, a, b, a_scales, b_scales);
#endif
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
