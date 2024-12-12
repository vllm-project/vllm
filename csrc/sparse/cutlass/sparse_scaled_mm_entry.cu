#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
void cutlass_scaled_sparse_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                                   torch::Tensor const& e,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   c10::optional<torch::Tensor> const& bias);
#endif

int32_t test_get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

void cutlass_scaled_sparse_mm(torch::Tensor& c, torch::Tensor const& a,
                              torch::Tensor const& e, torch::Tensor const& b,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              c10::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) * 2 == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1);                      // Row-major
  TORCH_CHECK(b.stride(0) == 1 && c.stride(0) == 1);  // Column-major
  TORCH_CHECK(c.stride(1) % 16 == 0);                 // 16 Byte Alignment
  TORCH_CHECK(b.stride(1) % 16 == 0);                 // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = test_get_sm_version_num();
  // Hopper

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_scaled_sparse_mm_sm90(c, a, e, b, a_scales, b_scales, bias);
    return;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_sparse_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
