#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
void cutlass_semi_structured_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias);
#endif

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

void cutlass_semi_structured_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();
  // Hopper

  // TODO: Guard against compilation issues for sm90 kernels
// #if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_semi_structured_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
// #endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_semi_structured_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
