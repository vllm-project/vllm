/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <cutlass/arch/arch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous.")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

template <typename ElementAB, typename ElementC, typename ElementSF,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts_sm120(
    ElementAB** a_offsets, ElementAB** b_offsets, ElementC** out_offsets,
    ElementSF** a_scales_offsets, ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int, ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int, const int32_t* expert_offsets,
    const int32_t* sf_offsets, const int32_t* problem_sizes_as_shapes,
    const int K, const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);

  assert((m >= 0 && n == N && k == K && k % 2 == 0) &&
         "unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / group_size);

  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;
  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;

  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) == 0 &&
         "TMA requires 128-byte alignment");
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) == 0 &&
         "TMA requires 128-byte alignment");

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), 1));
}

#define __CALL_GET_STARTS_KERNEL_BLOCKSCALE_SM120(ELEMENT_AB_TYPE, SF_TYPE,    \
                                                  TENSOR_C_TYPE, C_TYPE,      \
                                                  LayoutSFA, LayoutSFB,       \
                                                  ScaleConfig)                \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                            \
    __get_group_gemm_starts_sm120<ELEMENT_AB_TYPE, C_TYPE, SF_TYPE, float,     \
                                  LayoutSFA, LayoutSFB, ScaleConfig>          \
        <<<1, num_experts, 0, stream>>>(                                      \
            static_cast<ELEMENT_AB_TYPE**>(a_starts.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE**>(b_starts.data_ptr()),              \
            static_cast<C_TYPE**>(out_starts.data_ptr()),                     \
            static_cast<SF_TYPE**>(a_scales_starts.data_ptr()),               \
            static_cast<SF_TYPE**>(b_scales_starts.data_ptr()),               \
            static_cast<float**>(alpha_starts.data_ptr()),                    \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),              \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE*>(a_tensors.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE*>(b_tensors.data_ptr()),              \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                     \
            static_cast<SF_TYPE*>(a_scales.data_ptr()),                       \
            static_cast<SF_TYPE*>(b_scales.data_ptr()),                       \
            static_cast<float*>(alphas.data_ptr()),                           \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                 \
            static_cast<int32_t*>(sf_offsets.data_ptr()),                     \
            static_cast<int32_t*>(problem_sizes.data_ptr()), K, N);           \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
static inline void run_get_group_gemm_starts_sm120(
    const torch::Tensor& a_starts, const torch::Tensor& b_starts,
    const torch::Tensor& out_starts, const torch::Tensor& a_scales_starts,
    const torch::Tensor& b_scales_starts, const torch::Tensor& alpha_starts,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor const& out_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& alphas,
    torch::Tensor const& expert_offsets, torch::Tensor const& sf_offsets,
    torch::Tensor const& problem_sizes, int M, int N, int K) {
  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  TORCH_CHECK(out_tensors.size(1) == N,
              "Output tensor shape doesn't match expected shape");
  TORCH_CHECK(K / 2 == b_tensors.size(2),
              "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
              " dimension must match");
  if (false) {
  }
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE_SM120(
      cutlass::float_e2m1_t, cutlass::float_ue4m3_t, torch::kBFloat16,
      cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE_SM120(cutlass::float_e2m1_t,
                                            cutlass::float_ue4m3_t,
                                            torch::kFloat16, half, LayoutSFA,
                                            LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

template <typename OutType>
static inline void run_fp4_blockwise_scaled_group_mm_sm120(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int M,
    int N, int K) {
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using ClusterShape = Shape<_1, _1, _1>;

  struct MMA1SMConfig {
    using MmaTileShape = Shape<_128, _128, _128>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120;
    using EpilogueSchedule =
        cutlass::epilogue::collective::EpilogueScheduleAuto;
  };

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, EpilogueOperatorClass, typename MMA1SMConfig::MmaTileShape,
          ClusterShape, Shape<_128, _64>, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutD, AlignmentD,
          typename MMA1SMConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, MainloopOperatorClass, ElementA, LayoutA*, AlignmentA,
          ElementB, LayoutB*, AlignmentB, ElementAccumulator,
          typename MMA1SMConfig::MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename MMA1SMConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::CollectiveMainloop::InternalLayoutSFB;
}

#endif // ENABLE_NVFP4_SM120
