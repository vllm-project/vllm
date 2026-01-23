#include <cudaTypedefs.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include "../../torch_utils.h"

#include "stable/cutlass_extensions/common.hpp"

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability) {
  // sparse CUTLASS kernels need at least
  //   CUDA 12.2 and SM90 (Hopper)

#if defined CUDA_VERSION
  return CUDA_VERSION >= 12020 && cuda_device_capability >= 90;
#endif

  return false;
}

#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
void cutlass_scaled_sparse_mm_sm90(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& e,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias);

using CompressorResult =
    std::tuple<torch::stable::Tensor, torch::stable::Tensor>;
CompressorResult cutlass_sparse_compress_sm90(torch::stable::Tensor const& a);
#endif

void cutlass_scaled_sparse_mm(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& bt_nzs, torch::stable::Tensor const& bt_meta,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias) {
  // Checks for conformality
  STD_TORCH_CHECK(a.dim() == 2 && bt_nzs.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(1) == bt_nzs.size(0) &&
                  bt_nzs.size(1) * 2 == a.size(1) && a.size(0) == c.size(0));
  STD_TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  STD_TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == bt_nzs.size(0));

  // Check for strides and alignment
  STD_TORCH_CHECK(a.stride(1) == 1 && bt_nzs.stride(1) == 1 &&
                  c.stride(1) == 1);            // Row-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0);       // 16 Byte Alignment
  STD_TORCH_CHECK(bt_nzs.stride(0) % 16 == 0);  // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    STD_TORCH_CHECK(bias->numel() == bt_nzs.size(0) && bias->is_contiguous() &&
                    bias->dim() == 1);
  }

  torch::stable::accelerator::DeviceGuard device_guard(a.get_device_index());
  int32_t version_num = get_sm_version_num();

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
  // We build for 9.0a which is not forward compatible, so restrict this to
  // Hopper only
  if (version_num == 90) {
    cutlass_scaled_sparse_mm_sm90(c, a, bt_nzs, bt_meta, a_scales, b_scales,
                                  bias);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_sparse_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

std::vector<torch::stable::Tensor> cutlass_sparse_compress(
    torch::stable::Tensor const& a) {
  // Check for strides and alignment
  STD_TORCH_CHECK(a.stride(1) == 1);      // Row-major
  STD_TORCH_CHECK(a.stride(0) % 8 == 0);  // 8 Byte Alignment for Compression

  torch::stable::accelerator::DeviceGuard device_guard(a.get_device_index());
  int32_t version_num = get_sm_version_num();

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
  // We build for 9.0a which is not forward compatible, so restrict this to
  // Hopper only
  if (version_num == 90) {
    std::vector<torch::stable::Tensor> result_tensors;

    auto [a_meta, a_nzs] = cutlass_sparse_compress_sm90(a);
    result_tensors.push_back(std::move(a_nzs));
    result_tensors.push_back(std::move(a_meta));
    return result_tensors;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_sparse_compress for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
