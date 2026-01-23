#pragma once

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "machete_mm_kernel.cuh"
#include "stable/cutlass_extensions/torch_utils.hpp"
#include "stable/torch_utils.h"
#include "stable/core/scalar_type.hpp"

namespace machete {

struct MMArgs {
  torch::stable::Tensor const& A;
  torch::stable::Tensor const& B;
  vllm::ScalarType const& b_type;
  std::optional<torch::headeronly::ScalarType> const& maybe_out_type;
  std::optional<torch::stable::Tensor> const& maybe_group_scales;
  std::optional<torch::stable::Tensor> const& maybe_group_zeros;
  std::optional<int64_t> maybe_group_size;
  std::optional<torch::stable::Tensor> const& maybe_channel_scales;
  std::optional<torch::stable::Tensor> const& maybe_token_scales;
  std::optional<std::string> maybe_schedule;
};

struct SupportedSchedulesArgs {
  torch::headeronly::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<torch::headeronly::ScalarType> maybe_group_scales_type;
  std::optional<torch::headeronly::ScalarType> maybe_group_zeros_type;
  std::optional<torch::headeronly::ScalarType> maybe_channel_scales_type;
  std::optional<torch::headeronly::ScalarType> maybe_token_scales_type;
  std::optional<torch::headeronly::ScalarType> maybe_out_type;
};

torch::stable::Tensor mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args);

template <typename MacheteKernel>
torch::stable::Tensor run_impl(MMArgs args) {
  auto device = args.A.device();
  auto device_index = device.index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  auto stream = get_current_cuda_stream(device_index);

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);

  // Allocate output
  torch::stable::Tensor D = torch::stable::empty(
      {M, N}, equivalent_scalar_type_v<typename MacheteKernel::ElementD>,
      torch::headeronly::Layout::Strided, device);

  auto arguments = MacheteKernel::create_arguments(
      stream,  //
      args.A, args.B, D, args.maybe_group_scales, args.maybe_group_zeros,
      args.maybe_group_size, args.maybe_channel_scales,
      args.maybe_token_scales);
  STD_TORCH_CHECK(MacheteKernel::can_implement(arguments),
                  "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  torch::stable::Tensor workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           torch::headeronly::Layout::Strided, device);

  MacheteKernel::run(arguments, workspace.mutable_data_ptr(), stream);

  return D;
};

};  // namespace machete
