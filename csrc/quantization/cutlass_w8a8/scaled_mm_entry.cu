#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

void cutlass_scaled_mm_sm75(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm80(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm89(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);
#endif

void cutlass_scaled_mm_azp_sm75(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
void cutlass_scaled_mm_azp_sm90(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  // CUTLASS block-quantized FP8 kernels need at least CUDA 12.0
  // and at least SM90 (Hopper)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  }
#endif

  return false;
}

void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();
  // Hopper

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 75) {
    // Turing
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

void cutlass_scaled_mm_azp(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias) {
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

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  TORCH_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
