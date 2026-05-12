// SPDX-License-Identifier: Apache-2.0
// Local GEMM caller for MLA absorption BMM kernels.
// Uses regular PyTorch C++ API (torch::Tensor, ATen CUDA streams)
// instead of libtorch stable API, so these kernels can be compiled
// in the _C target (which doesn't set Py_LIMITED_API).
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cute/tensor.hpp"

namespace vllm::mla {

template <typename GemmKernel>
void cutlass_gemm_caller(
    at::Device device, cute::Shape<int, int, int, int> prob_shape,
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

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));

  auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace vllm::mla
