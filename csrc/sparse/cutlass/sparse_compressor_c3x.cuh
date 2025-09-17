#pragma once

// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12020
#include "sparse_scaled_mm_c3x.cuh"

#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// clang-format on

using namespace cute;
using namespace vllm;

using CompressorResult = std::tuple<torch::Tensor, torch::Tensor>;
/// Make A structured sparse by replacing elements with 0 and compress it
template <typename Gemm>
CompressorResult cutlass_sparse_compress(torch::Tensor const& a) {
  // Checks for conformality
  TORCH_CHECK(a.dtype() == torch::kInt8 || a.dtype() == torch::kFloat8_e4m3fn ||
              a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16);
  TORCH_CHECK(a.dim() == 2)
  // Check for strides and alignment
  TORCH_CHECK(a.stride(0) % 4 == 0)  // Required for semi-structured sparsity
  TORCH_CHECK(a.stride(1) == 1)

  using GemmKernel = typename Gemm::KernelType;
  using ElementA = typename Gemm::ElementAB;
  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;

  int m = a.size(0);
  int k = a.size(1);
  using ProblemShape = typename GemmKernel::ProblemShape;
  ProblemShape prob_shape{m, 1, k, 1};

  int64_t lda = a.stride(0);
  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  StrideA a_stride{lda, Int<1>{}, 0};

  using CompressorUtility = typename Gemm::CompressorUtility;
  CompressorUtility compressor_utility(prob_shape, a_stride);

  // Allocate buffers for the metadata E and the compressed matrix A
  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int MC = compressor_utility.get_tensorA_m_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  auto const a_meta_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto const a_nzs_options =
      torch::TensorOptions().dtype(a.dtype()).device(a.device());

  auto a_meta = torch::zeros({ME, KE}, a_meta_options);
  auto a_nzs = torch::zeros({MC, KC}, a_nzs_options);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto a_nzs_ptr = static_cast<ElementA*>(a_nzs.data_ptr());
  auto a_meta_ptr = static_cast<ElementE*>(a_meta.data_ptr());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using Compressor = typename Gemm::Compressor;
  typename Compressor::Arguments arguments{
      prob_shape, {a_ptr, a_stride, a_nzs_ptr, a_meta_ptr}, {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.data_ptr()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return {a_meta, a_nzs};
}

#endif
