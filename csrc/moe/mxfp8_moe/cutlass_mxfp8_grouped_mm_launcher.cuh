// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_launcher.cuh

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass_mxfp8_grouped_mm_functor.cuh"
#include "cutlass_mxfp8_grouped_mm_traits.cuh"

namespace expert_specialization {

template <typename GemmTraits>
void cutlass_mxfp8_grouped_mm_pre_compute(
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs, torch::Tensor& d_ptrs, torch::Tensor& stride_a,
    torch::Tensor& stride_b, torch::Tensor& stride_d, torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& sfa, const torch::Tensor& sfb, const torch::Tensor& d,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets, cudaStream_t stream) {
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

  int num_experts = (int)expert_offsets.size(0);
  TORCH_CHECK(num_experts <= 1024,
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
void cutlass_mxfp8_grouped_mm(
    const torch::Tensor& a_ptrs, const torch::Tensor& b_ptrs,
    const torch::Tensor& sfa_ptrs, const torch::Tensor& sfb_ptrs,
    const torch::Tensor& d_ptrs, const torch::Tensor& stride_a,
    const torch::Tensor& stride_b, const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes, cudaStream_t stream) {
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
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  hw_info.cluster_shape = GemmTraits::MMAConfig::preferred_cluster;
  hw_info.cluster_shape_fallback = GemmTraits::MMAConfig::fallback_cluster;

  int num_experts = (int)problem_sizes.size(0);

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
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM");

  torch::TensorOptions options_uint8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(d_ptrs.device());
  size_t workspace_size = gemm.get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(workspace_size, options_uint8);

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(stream, nullptr, true);  // Enable PDL
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void cutlass_mxfp8_grouped_mm_dispatch_out_dtype(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa,
    const torch::Tensor& sfb, torch::Tensor& d,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets, cudaStream_t stream) {
  int num_experts = (int)problem_sizes.size(0);
  torch::TensorOptions options_int64 =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::TensorOptions options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor sfa_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor sfb_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor d_ptrs = torch::empty(num_experts, options_int64);

  torch::Tensor stride_a = torch::empty(num_experts, options_int64);
  torch::Tensor stride_b = torch::empty(num_experts, options_int64);
  torch::Tensor stride_d = torch::empty(num_experts, options_int64);
  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int32);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int32);

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