#pragma once

#include <torch/all.h>
#include <Python.h>

#include "machete_mm_kernel.cuh"
#include "cutlass_extensions/torch_utils.hpp"

namespace machete {

struct PyTorchArguments {
  torch::Tensor const& A;
  torch::Tensor const& B;
  c10::optional<torch::Tensor> const& scales;
  c10::optional<torch::Tensor> const& zeros;
  c10::optional<int64_t> group_size;
  c10::optional<torch::Tensor> const& C;
  c10::optional<double> alpha;
  c10::optional<double> beta;
  c10::optional<std::string> schedule;
};

template <typename MacheteKernel>
torch::Tensor run_impl(PyTorchArguments args) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  auto device = args.A.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  using EleA = typename MacheteKernel::ElementA;
  using EleB = typename MacheteKernel::ElementB;
  using EleC = typename MacheteKernel::ElementC;
  using EleD = typename MacheteKernel::ElementD;
  using EleScale = typename MacheteKernel::ElementS;
  using EleZero = typename MacheteKernel::ElementZ;

  using StrideA = typename MacheteKernel::StrideA;
  using StrideC = typename MacheteKernel::StrideC;
  using StrideD = typename MacheteKernel::StrideD;
  using StrideS = typename MacheteKernel::StrideS;
  using StrideZ = typename MacheteKernel::StrideZ;

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);

  // Allocate output
  torch::Tensor D =
      torch::empty({M, N}, torch::TensorOptions()
                               .dtype(equivalent_scalar_type_v<EleD>)
                               .device(device));

  auto const &A = args.A, &B = args.B;
  auto const &C = args.C, &scales = args.scales, &zeros = args.zeros;

  auto layout_A = make_cute_layout<StrideA>(A, "A");
  auto layout_D = make_cute_layout<StrideD>(D, "D");
  auto layout_C = maybe_make_cute_layout<StrideC>(C, "C");
  auto layout_S = maybe_make_cute_layout<StrideS>(scales, "scales");
  auto layout_Z = maybe_make_cute_layout<StrideZ>(zeros, "zeros");

  auto A_ptr = static_cast<EleA const*>(A.const_data_ptr());
  auto B_ptr = static_cast<EleB const*>(B.const_data_ptr());
  auto D_ptr = static_cast<EleD*>(D.mutable_data_ptr());
  auto C_ptr = static_cast<EleC const*>(C ? C->const_data_ptr() : nullptr);
  auto S_ptr =
      static_cast<EleScale const*>(scales ? scales->const_data_ptr() : nullptr);
  auto Z_ptr =
      static_cast<EleZero const*>(zeros ? zeros->const_data_ptr() : nullptr);

  auto arguments = MacheteKernel::create_arguments(
      stream, A_ptr, layout_A, B_ptr, D_ptr, layout_D, C_ptr, layout_C, S_ptr,
      layout_S, Z_ptr, layout_Z, args.alpha.value_or(1), args.beta.value_or(0),
      args.group_size.value_or(K));
  TORCH_CHECK(MacheteKernel::can_implement(arguments),
              "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(
      workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));

  MacheteKernel::run(arguments, workspace.mutable_data_ptr(), stream);

  return D;
};

template <typename ElementA, typename ElementB, typename ElementD = ElementA,
          typename AccumulatorT = float, typename ScaleT = ElementA,
          typename ZeroT = ElementA>
struct GemmDispatcher {
  static torch::Tensor dispatch(PyTorchArguments args);
  static std::vector<std::string> supported_schedules();
};

};  // namespace machete