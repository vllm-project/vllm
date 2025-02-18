#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
// clang-format on

namespace vllm::c3x {

static inline cute::Shape<int, int, int, int> get_problem_shape(
    torch::Tensor const& a, torch::Tensor const& b) {
  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  return {m, n, k, 1};
}

template <typename GemmKernel>
void cutlass_gemm_caller(
    torch::Device device, cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info,
                                      scheduler};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                         torch::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, cute::Int<1>{}, 0};
  StrideB b_stride{ldb, cute::Int<1>{}, 0};
  StrideC c_stride{ldc, cute::Int<1>{}, cute::Int<0>{}};

  typename GemmKernel::ProblemShape prob_shape = get_problem_shape(a, b);

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                  epilogue_args);
}

}  // namespace vllm::c3x