#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
bool cutlass_sparse_compress_sm90(torch::Tensor& a_nzs, torch::Tensor& a_meta,
                                  torch::Tensor const& a);
#endif

bool cutlass_sparse_compress_entry(torch::Tensor& a_nzs, torch::Tensor& a_meta,
                                   torch::Tensor const& a) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && a_meta.dim() == 2 && a_nzs.dim() == 2);
  TORCH_CHECK(a.size(0) == a_nzs.size(0) && a.size(0) == a_meta.size(0) &&
              a_nzs.size(1) * 2 == a.size(1) &&
              a_meta.size(1) * 2 * 4 == a.size(1));
  // Considering elemsPerMetaElem = 8b / 2b_per_nz = 4

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && a_nzs.stride(1) == 1 &&
              a_meta.stride(1) == 1);  // Row-major
  TORCH_CHECK(a.stride(0) % 8 == 0);   // 8 Byte Alignment for Compression

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SPARSE_SCALED_MM_C3X && ENABLE_SPARSE_SCALED_MM_C3X
  if (version_num >= 90) {
    return cutlass_sparse_compress_sm90(a_nzs, a_meta, a);
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_sparse_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
