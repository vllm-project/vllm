#pragma once

#include <torch/all.h>
#include <Python.h>

#include "machete_mm_kernel.cuh"
#include "cutlass_extensions/torch_utils.hpp"
#include "core/scalar_type.hpp"

namespace machete {

struct MMArgs {
  torch::Tensor const& A;
  torch::Tensor const& B;
  vllm::ScalarType const& b_type;
  std::optional<at::ScalarType> const& maybe_out_type;
  std::optional<torch::Tensor> const& maybe_group_scales;
  std::optional<torch::Tensor> const& maybe_group_zeros;
  std::optional<int64_t> maybe_group_size;
  std::optional<torch::Tensor> const& maybe_channel_scales;
  std::optional<torch::Tensor> const& maybe_token_scales;
  std::optional<std::string> maybe_schedule;
};

struct SupportedSchedulesArgs {
  at::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<at::ScalarType> maybe_group_scales_type;
  std::optional<at::ScalarType> maybe_group_zeros_type;
  std::optional<at::ScalarType> maybe_channel_scales_type;
  std::optional<at::ScalarType> maybe_token_scales_type;
  std::optional<at::ScalarType> maybe_out_type;
};

torch::Tensor mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args);

template <typename MacheteKernel>
torch::Tensor run_impl(MMArgs args) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  auto device = args.A.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);

  // Allocate output
  torch::Tensor D = torch::empty(
      {M, N},
      torch::TensorOptions()
          .dtype(equivalent_scalar_type_v<typename MacheteKernel::ElementD>)
          .device(device));

  auto arguments = MacheteKernel::create_arguments(
      stream,  //
      args.A, args.B, D, args.maybe_group_scales, args.maybe_group_zeros,
      args.maybe_group_size, args.maybe_channel_scales,
      args.maybe_token_scales);
  TORCH_CHECK(MacheteKernel::can_implement(arguments),
              "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(
      workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));

  MacheteKernel::run(arguments, workspace.mutable_data_ptr(), stream);

  return D;
};

};  // namespace machete