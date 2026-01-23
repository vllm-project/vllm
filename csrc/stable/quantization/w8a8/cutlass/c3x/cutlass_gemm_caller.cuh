#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"
#include "stable/cutlass_extensions/common.hpp"
#include "stable/torch_utils.h"
// clang-format on

namespace vllm::c3x {

static inline cute::Shape<int, int, int, int> get_problem_shape(
    torch::stable::Tensor const& a, torch::stable::Tensor const& b) {
  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  return {m, n, k, 1};
}

template <typename GemmKernel>
void cutlass_gemm_caller(
    torch::stable::Device device, cute::Shape<int, int, int, int> prob_shape,
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
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt,  // layout
                           device);

  auto stream = get_current_cuda_stream(device.index());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(torch::stable::Tensor& out,
                         torch::stable::Tensor const& a,
                         torch::stable::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = StrideC;
  using StrideAux = StrideC;

  typename GemmKernel::ProblemShape prob_shape = get_problem_shape(a, b);
  auto [M, N, K, L] = prob_shape;

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  StrideD d_stride =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
  StrideAux aux_stride = d_stride;

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  // auto d_ptr = static_cast<ElementC*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, d_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                  epilogue_args);
}

}  // namespace vllm::c3x
