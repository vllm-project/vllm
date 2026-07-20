/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from SGLang's W4A8 grouped-MoE implementation introduced in
 * https://github.com/sgl-project/sglang/pull/7772. Modified by the vLLM
 * project for stable libtorch and the GLM W4AFP8 operator contract.
 */

#pragma once

/**
 * @file w4afp8_grouped_mm_c3x.cuh
 * @brief Implementation of grouped GEMM operation with int4 and fp8 mixed
 * precision
 *
 * This file implements a grouped GEMM operation that multiplies FP8 matrices
 * (A) with quantized INT4 matrices (B), applying per-block scaling factors.
 * The implementation is optimized for NVIDIA Hopper GPUs, leveraging Tensor
 * Cores for mixed precision arithmetic.
 *
 * Key features:
 * - Supports grouped GEMM operations with multiple experts
 * - Uses FP8 (e4m3) for matrix A
 * - Uses INT4 quantization for matrix B with per-block scaling
 * - Implements preprocessing for INT4 encoding and scale packing
 * - Optimized for Hopper architecture with Tensor Core operations
 */

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tuple>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "w4afp8_get_group_starts.cuh"

using namespace cute;

namespace {

// Type definitions
using MmaType = cutlass::float_e4m3_t;     // FP8 e4m3 type
using QuantType = cutlass::int4b_t;        // 4-bit integer type
using ElementAccumulator = float;          // Accumulator type
using ElementScale = cutlass::bfloat16_t;  // Scale type
using ElementC = cutlass::bfloat16_t;      // Output type
using ElementD = ElementC;                 // Output type
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// Architecture-specific configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
// constexpr int TileShapeK = 512;
// using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
// using ClusterShape = Shape<_1, _1, _1>;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

// Transposed layouts
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// Alignments
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_w4afp8_group_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC_Transpose*, AlignmentC,
          ElementD, LayoutD_Transpose*, AlignmentD,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly =
      typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
          ArchTag, OperatorClass, cute::tuple<QuantType, ElementScalePacked>,
          LayoutB_Transpose*, AlignmentB, MmaType, LayoutA_Transpose*,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  // Define the final kernel and GEMM operation types
  using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;

  using GemmScaleOnly =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA =
      cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB =
      cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

/**
 * @brief Main function to run int4 * fp8 grouped GEMM from PyTorch
 *
 * This function performs multiple GEMM operations in parallel where each
 * operation multiplies an FP8 matrix (A) with a quantized INT4 matrix (B),
 * applying BF16 group scales. It's designed for efficient execution
 * on NVIDIA Hopper GPUs, leveraging Tensor Cores for optimal performance with
 * mixed precision arithmetic.
 *
 * @param d_tensors Output tensor D with shape [total_m, total_n]
 * @param a_tensors Tensor containing all A matrices (fp8_e4m3) with shape
 * [total_m, K]
 * @param b_tensors Tensor containing all B matrices (int4 packed as int8) with
 * shape [E, N, K/2]
 * @param a_scales Scalar FP32 activation scale
 * @param b_scales Tensor containing B matrix scale factors with shape [E,
 * K//512, N*4]
 * @param expert_offsets Tensor containing expert offsets for determining group
 * boundaries (int32)
 * @param problem_sizes Tensor containing problem sizes with shape [num_experts,
 * 3] (M, N, K for each group) (int32)
 * @param a_strides Stride information for A tensors
 * @param b_strides Stride information for B tensors
 * @param d_strides Stride information for D tensors
 * @param s_strides Stride information for scale tensors
 * @param chunk_size Weight quantization group size; must be 128
 */
// template <typename TileShape, typename ClusterShape, typename KernelSchedule,
// typename EpilogueSchedule>
template <typename Gemm>
void cutlass_w4afp8_group_gemm_caller(
    torch::stable::Tensor& d_tensors, torch::stable::Tensor const& a_tensors,
    torch::stable::Tensor const& b_tensors,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& problem_sizes,
    torch::stable::Tensor const& a_strides,
    torch::stable::Tensor const& b_strides,
    torch::stable::Tensor const& d_strides,
    torch::stable::Tensor const& s_strides, int64_t chunk_size) {
  //   using Gemm = cutlass_3x_w4afp8_group_gemm<TileShape, ClusterShape,
  //   KernelSchedule, EpilogueSchedule>;
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  STD_TORCH_CHECK(a_tensors.dim() == 2, "A tensor must be 2D [total_m, K]");
  STD_TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  STD_TORCH_CHECK(d_tensors.dim() == 2,
                  "Output tensor must be 2D [total_m, N]");
  STD_TORCH_CHECK(b_scales.dim() == 3, "Weight scales must be a 3D tensor");
  STD_TORCH_CHECK(a_scales.dim() == 1 && a_scales.numel() == 1,
                  "Activation scale must be a 1D scalar tensor");
  STD_TORCH_CHECK(expert_offsets.dim() == 1,
                  "expert_offsets must be a 1D tensor");
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  int num_experts = static_cast<int>(expert_offsets.size(0));
  STD_TORCH_CHECK(num_experts > 0 && num_experts <= 1024,
                  "Number of experts must be in [1, 1024]");
  STD_TORCH_CHECK(chunk_size == 128, "W4AFP8 requires weight group_size=128");
  STD_TORCH_CHECK(problem_sizes.size(0) == num_experts,
                  "problem_sizes must have num_experts rows");
  STD_TORCH_CHECK(problem_sizes.size(1) == 3,
                  "problem_sizes must have 3 columns (N, M, K)");
  STD_TORCH_CHECK(b_tensors.size(0) == num_experts,
                  "B tensor first dimension must match number of groups");
  STD_TORCH_CHECK(b_scales.size(0) == num_experts,
                  "Scale tensor first dimension must match number of groups");
  int64_t n = b_tensors.size(1);
  int64_t k = a_tensors.size(1);
  STD_TORCH_CHECK(b_tensors.size(2) * 2 == k,
                  "B tensor K/2 dimension must match A tensor K dimension");
  STD_TORCH_CHECK(
      d_tensors.size(0) == a_tensors.size(0) && d_tensors.size(1) == n,
      "Output shape must be [total_m, N]");
  STD_TORCH_CHECK(k % 128 == 0, "W4AFP8 K dimension must be divisible by 128");
  STD_TORCH_CHECK(b_scales.numel() == num_experts * n * k / 128,
                  "Weight scales must contain E*N*K/128 elements");

  STD_TORCH_CHECK(
      a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "A tensor must be fp8 (float_e4m3_t) type");
  STD_TORCH_CHECK(
      b_tensors.scalar_type() == torch::headeronly::ScalarType::Char,
      "B tensor must contain packed int4 values (stored as int8)");
  STD_TORCH_CHECK(
      a_scales.scalar_type() == torch::headeronly::ScalarType::Float,
      "Activation scale must be float32");
  STD_TORCH_CHECK(
      b_scales.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Weight scales must be bfloat16");
  STD_TORCH_CHECK(
      d_tensors.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Output tensor must be bfloat16");
  STD_TORCH_CHECK(
      expert_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "Expert offsets must be int32 type");
  STD_TORCH_CHECK(
      problem_sizes.scalar_type() == torch::headeronly::ScalarType::Int,
      "Problem sizes must be int32 type");
  STD_TORCH_CHECK(
      a_strides.scalar_type() == torch::headeronly::ScalarType::Long &&
          b_strides.scalar_type() == torch::headeronly::ScalarType::Long &&
          d_strides.scalar_type() == torch::headeronly::ScalarType::Long &&
          s_strides.scalar_type() == torch::headeronly::ScalarType::Long,
      "Stride tensors must be int64");

  auto device = a_tensors.device();
  STD_TORCH_CHECK(a_tensors.is_cuda() && b_tensors.is_cuda() &&
                      d_tensors.is_cuda() && a_scales.is_cuda() &&
                      b_scales.is_cuda() && expert_offsets.is_cuda() &&
                      problem_sizes.is_cuda() && a_strides.is_cuda() &&
                      b_strides.is_cuda() && d_strides.is_cuda() &&
                      s_strides.is_cuda(),
                  "All W4AFP8 tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      b_tensors.device() == device && d_tensors.device() == device &&
          a_scales.device() == device && b_scales.device() == device &&
          expert_offsets.device() == device &&
          problem_sizes.device() == device && a_strides.device() == device &&
          b_strides.device() == device && d_strides.device() == device &&
          s_strides.device() == device,
      "All W4AFP8 tensors must be on the same device");
  STD_TORCH_CHECK(a_tensors.is_contiguous() && b_tensors.is_contiguous() &&
                      d_tensors.is_contiguous() && a_scales.is_contiguous() &&
                      b_scales.is_contiguous() &&
                      expert_offsets.is_contiguous() &&
                      problem_sizes.is_contiguous() &&
                      a_strides.is_contiguous() && b_strides.is_contiguous() &&
                      d_strides.is_contiguous() && s_strides.is_contiguous(),
                  "All W4AFP8 tensors must be contiguous");

  const torch::stable::accelerator::DeviceGuard device_guard(
      a_tensors.get_device_index());
  auto* device_prop = get_device_prop();
  STD_TORCH_CHECK(device_prop->major == 9 && device_prop->minor == 0,
                  "W4AFP8 CUTLASS kernel requires SM90");

  auto stream = get_current_cuda_stream(a_tensors.get_device_index());
  torch::stable::Tensor a_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor b_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor out_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor a_scales_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);
  torch::stable::Tensor b_scales_ptrs = torch::stable::empty(
      num_experts, torch::headeronly::ScalarType::Long, std::nullopt, device);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.get_device_index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 0;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = static_cast<float*>(a_scales.data_ptr());
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape problem_shape;
  problem_shape.num_groups = num_experts;
  problem_shape.problem_shapes = problem_sizes_as_shapes;
  problem_shape.host_problem_shapes = nullptr;

  run_int4_fp8_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                                     a_scales_ptrs, b_scales_ptrs, a_tensors,
                                     b_tensors, d_tensors, a_scales, b_scales);

  arguments = Args{cutlass::gemm::GemmUniversalMode::kGrouped,
                   problem_shape,
                   {static_cast<const QuantType**>(b_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
                    static_cast<const MmaType**>(a_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
                    static_cast<const typename Gemm::ElementScalePacked**>(
                        b_scales_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
                    static_cast<int>(chunk_size)},
                   {fusion_args, nullptr, nullptr,
                    static_cast<ElementD**>(out_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
                   hw_info};

  // Instantiate and run GEMM
  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, device);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM implementation not supported");
  }

  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM initialization failed");
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM execution failed");
  }
}

}  // namespace
