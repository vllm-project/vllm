#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability) {
  // sparse CUTLASS kernels need at least
  //   CUDA 12.2 and SM90 (Hopper)

#if defined CUDA_VERSION
  return CUDA_VERSION >= 12020 && cuda_device_capability >= 90;
#endif

  return false;
}

#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
void cutlass_scaled_sparse_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& e,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   std::optional<torch::Tensor> const& bias);

using CompressorResult = std::tuple<torch::Tensor, torch::Tensor>;
CompressorResult cutlass_sparse_compress_sm90(torch::Tensor const& a);
#endif

void cutlass_scaled_sparse_mm(torch::Tensor& c, torch::Tensor const& a,
                              torch::Tensor const& bt_nzs,
                              torch::Tensor const& bt_meta,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              std::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && bt_nzs.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(1) == bt_nzs.size(0) && bt_nzs.size(1) * 2 == a.size(1) &&
              a.size(0) == c.size(0));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == bt_nzs.size(0));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && bt_nzs.stride(1) == 1 &&
              c.stride(1) == 1);            // Row-major
  TORCH_CHECK(c.stride(0) % 16 == 0);       // 16 Byte Alignment
  TORCH_CHECK(bt_nzs.stride(0) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == bt_nzs.size(0) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_scaled_sparse_mm_sm90(c, a, bt_nzs, bt_meta, a_scales, b_scales,
                                  bias);
    return;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_sparse_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

std::vector<torch::Tensor> cutlass_sparse_compress(torch::Tensor const& a) {
  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1);      // Row-major
  TORCH_CHECK(a.stride(0) % 8 == 0);  // 8 Byte Alignment for Compression

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
  if (version_num >= 90) {
    std::vector<torch::Tensor> result_tensors;

    auto [a_meta, a_nzs] = cutlass_sparse_compress_sm90(a);
    result_tensors.push_back(std::move(a_nzs));
    result_tensors.push_back(std::move(a_meta));
    return result_tensors;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_sparse_compress for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
