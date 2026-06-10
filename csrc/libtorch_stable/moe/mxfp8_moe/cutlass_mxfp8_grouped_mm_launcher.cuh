// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_launcher.cuh

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass_mxfp8_grouped_mm_functor.cuh"
#include "cutlass_mxfp8_grouped_mm_traits.cuh"
#include "libtorch_stable/torch_utils.h"

namespace expert_specialization {

template <typename GemmTraits>
void cutlass_mxfp8_grouped_mm_pre_compute(
    torch::stable::Tensor& a_ptrs, torch::stable::Tensor& b_ptrs,
    torch::stable::Tensor& sfa_ptrs, torch::stable::Tensor& sfb_ptrs,
    torch::stable::Tensor& d_ptrs, torch::stable::Tensor& stride_a,
    torch::stable::Tensor& stride_b, torch::stable::Tensor& stride_d,
    torch::stable::Tensor& layout_sfa, torch::stable::Tensor& layout_sfb,
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& sfa, const torch::stable::Tensor& sfb,
    const torch::stable::Tensor& d, const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets, cudaStream_t stream) {
  using OffsetFunctor = CutlassMxfp8GroupedMmOffsetFunctor<GemmTraits>;
  using ElementA = typename OffsetFunctor::ElementA;
  using ElementB = typename OffsetFunctor::ElementB;
  using ElementSF = typename OffsetFunctor::ElementSF;
  using ElementD = typename OffsetFunctor::ElementD;

  using LayoutFunctor = CutlassMxfp8GroupedMmLayoutFunctor<GemmTraits>;
  using LayoutSFA = typename LayoutFunctor::LayoutSFA;
  using LayoutSFB = typename LayoutFunctor::LayoutSFB;

  using StrideFunctor = CutlassMxfp8GroupedMmStrideFunctor<GemmTraits>;
  using StrideA = typename StrideFunctor::StrideA;
  using StrideB = typename StrideFunctor::StrideB;
  using StrideD = typename StrideFunctor::StrideD;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  STD_TORCH_CHECK(num_experts <= 1024,
                  "Number of experts cannot exceed 1024, the maximum number of "
                  "threads per block.");

  OffsetFunctor offset_functor(
      reinterpret_cast<int*>(expert_offsets.data_ptr()),
      reinterpret_cast<int*>(blockscale_offsets.data_ptr()),
      reinterpret_cast<ElementA*>(a.data_ptr()),
      reinterpret_cast<ElementB*>(b.data_ptr()),
      reinterpret_cast<ElementSF*>(sfa.data_ptr()),
      reinterpret_cast<ElementSF*>(sfb.data_ptr()),
      reinterpret_cast<ElementD*>(d.data_ptr()),
      reinterpret_cast<ElementA**>(a_ptrs.data_ptr()),
      reinterpret_cast<ElementB**>(b_ptrs.data_ptr()),
      reinterpret_cast<ElementSF**>(sfa_ptrs.data_ptr()),
      reinterpret_cast<ElementSF**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<ElementD**>(d_ptrs.data_ptr()));
  LayoutFunctor layout_functor(
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()));
  StrideFunctor stride_functor(reinterpret_cast<StrideA*>(stride_a.data_ptr()),
                               reinterpret_cast<StrideB*>(stride_b.data_ptr()),
                               reinterpret_cast<StrideD*>(stride_d.data_ptr()));
  cutlassMxfp8GroupedMmPreComputeKernel<<<1, num_experts, 0, stream>>>(
      static_cast<int*>(problem_sizes.data_ptr()), offset_functor,
      layout_functor, stride_functor);
}

template <typename GemmTraits>
void cutlass_mxfp8_grouped_mm(const torch::stable::Tensor& a_ptrs,
                              const torch::stable::Tensor& b_ptrs,
                              const torch::stable::Tensor& sfa_ptrs,
                              const torch::stable::Tensor& sfb_ptrs,
                              const torch::stable::Tensor& d_ptrs,
                              const torch::stable::Tensor& stride_a,
                              const torch::stable::Tensor& stride_b,
                              const torch::stable::Tensor& stride_d,
                              const torch::stable::Tensor& layout_sfa,
                              const torch::stable::Tensor& layout_sfb,
                              const torch::stable::Tensor& problem_sizes,
                              cudaStream_t stream) {
  using Gemm = typename GemmTraits::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementOutput;
  using StrideA = typename GemmTraits::StrideA;
  using StrideB = typename GemmTraits::StrideB;
  using StrideD = typename GemmTraits::StrideD;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using UnderlyingProblemShape =
      typename GemmTraits::ProblemShape::UnderlyingProblemShape;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = d_ptrs.get_device_index();
  hw_info.sm_count = get_device_prop()->multiProcessorCount;
  hw_info.cluster_shape = GemmTraits::MMAConfig::preferred_cluster;
  hw_info.cluster_shape_fallback = GemmTraits::MMAConfig::fallback_cluster;

  int num_experts = static_cast<int>(problem_sizes.size(0));

  UnderlyingProblemShape* underlying_problem_shape =
      reinterpret_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  typename Gemm::Arguments arguments = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, underlying_problem_shape, nullptr},
      {reinterpret_cast<const ElementA**>(a_ptrs.data_ptr()),
       reinterpret_cast<StrideA*>(stride_a.data_ptr()),
       reinterpret_cast<const ElementB**>(b_ptrs.data_ptr()),
       reinterpret_cast<StrideB*>(stride_b.data_ptr()),
       reinterpret_cast<const ElementSF**>(sfa_ptrs.data_ptr()),
       reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
       reinterpret_cast<const ElementSF**>(sfb_ptrs.data_ptr()),
       reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())},
      {{},
       nullptr,
       nullptr,
       reinterpret_cast<ElementD**>(d_ptrs.data_ptr()),
       reinterpret_cast<StrideD*>(stride_d.data_ptr())},
      hw_info,
      {}  // Scheduler
  };

  Gemm gemm;

  auto can_implement_status = gemm.can_implement(arguments);
  STD_TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
                  "Failed to implement GEMM");

  size_t workspace_size = gemm.get_workspace_size(arguments);
  torch::stable::Tensor workspace = torch::stable::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::headeronly::ScalarType::Byte, std::nullopt, d_ptrs.device());

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "Failed to initialize GEMM");

  status = gemm.run(stream, nullptr, true);  // Enable PDL
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void cutlass_mxfp8_grouped_mm_dispatch_out_dtype(
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& sfa, const torch::stable::Tensor& sfb,
    torch::stable::Tensor& d, const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets, cudaStream_t stream) {
  int num_experts = static_cast<int>(problem_sizes.size(0));
  auto device = a.device();

  torch::stable::Tensor a_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor b_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor sfa_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor sfb_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor d_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);

  torch::stable::Tensor stride_a = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor stride_b = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor stride_d = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor layout_sfa =
      torch::stable::empty({num_experts, 5}, torch::headeronly::ScalarType::Int,
                           std::nullopt, device);
  torch::stable::Tensor layout_sfb =
      torch::stable::empty({num_experts, 5}, torch::headeronly::ScalarType::Int,
                           std::nullopt, device);

  using GemmTraits = CutlassMxfp8GroupedMmGemmTraits<MMA1SMConfig, OutType>;
  cutlass_mxfp8_grouped_mm_pre_compute<GemmTraits>(
      a_ptrs, b_ptrs, sfa_ptrs, sfb_ptrs, d_ptrs, stride_a, stride_b, stride_d,
      layout_sfa, layout_sfb, a, b, sfa, sfb, d, problem_sizes, expert_offsets,
      blockscale_offsets, stream);
  cutlass_mxfp8_grouped_mm<GemmTraits>(
      a_ptrs, b_ptrs, sfa_ptrs, sfb_ptrs, d_ptrs, stride_a, stride_b, stride_d,
      layout_sfa, layout_sfb, problem_sizes, stream);
}

}  // namespace expert_specialization
