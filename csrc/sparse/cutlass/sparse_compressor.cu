#include <cudaTypedefs.h>

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/detail/dependent_false.hpp"

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "cutlass_extensions/common.hpp"

#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include <iostream>

#include "cutlass/cutlass.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "sparse_scaled_mm_c3x.cuh"

using namespace cute;
using namespace vllm;

/// Make A structured sparse by replacing elements with 0 and compress it
template <typename ElementA_>
bool sparsify_and_compress(torch::Tensor& a_compressed, torch::Tensor& e,
                           torch::Tensor const& a) {
  // Checks for conformality
  TORCH_CHECK(a.dtype() == torch::kInt8 || a.dtype() == torch::kFloat8_e4m3fn ||
              a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16);
  TORCH_CHECK(a.dim() == 2)
  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1)

  int m = a.size(0);
  int k = a.size(1);

  using ProblemShape = Shape<int, int, int, int>;
  using ElementA = ElementA_;
  using LayoutTagA = cutlass::layout::RowMajor;

  // Layouts for reference (non-sparse) tensors
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideE = StrideA;

  using Gemm = typename sm90_config_default<ElementA, cutlass::half_t,
                                            c3x::ScaledEpilogue>::Cutlass3xGemm;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  // Just a dummy value
  int32_t n = 128;

  int64_t lda = a.stride(0);

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, 0};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  using LayoutA = typename GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename GemmKernel::CollectiveMainloop::LayoutE;

  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;

  LayoutA a_layout = SparseConfig::fill_layoutA(prob_shape);
  LayoutE e_layout = SparseConfig::fill_layoutE(prob_shape);

  // Offline compressor kernel
  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          ProblemShape, ElementA, LayoutTagA, SparseConfig>;

  using CompressorKernel =
      cutlass::transform::kernel::StructuredSparseCompressor<
          ProblemShape, ElementA, LayoutTagA, SparseConfig,
          cutlass::arch::Sm90>;

  using Compressor =
      cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  auto [M, N, K, L] = prob_shape;

  StrideA stride_A;
  StrideA stride_A_compressed;
  StrideE stride_E;

  stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

  CompressorUtility compressor_utility(prob_shape, stride_A);

  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());

  auto a_compressed_ptr = static_cast<ElementA*>(a_compressed.data_ptr());
  auto e_ptr =
      static_cast<typename Gemm::CollectiveMainloop::ElementE*>(e.data_ptr());

  stride_A_compressed =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, KC, L));
  stride_E =
      cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Compressor::Arguments arguments{
      prob_shape, {a_ptr, stride_A, a_compressed_ptr, e_ptr}, {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return true;
}

bool cutlass_compress_entry(torch::Tensor& a_compressed, torch::Tensor& e,
                            torch::Tensor const& a) {
  if (a.dtype() == torch::kBFloat16) {
    return sparsify_and_compress<cutlass::bfloat16_t>(a_compressed, e, a);
  } else if (a.dtype() == torch::kFloat16) {
    return sparsify_and_compress<cutlass::half_t>(a_compressed, e, a);
  } else if (a.dtype() == torch::kFloat8_e4m3fn) {
    return sparsify_and_compress<cutlass::float_e4m3_t>(a_compressed, e, a);
  } else if (a.dtype() == torch::kInt8) {
    return sparsify_and_compress<int8_t>(a_compressed, e, a);
  }
  return false;
}