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
#include <cuda_runtime.h>
#include <unordered_map>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cassert>

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
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

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

static inline bool nvfp4_sm120_debug_enabled() {
  static int inited = 0;
  static bool enabled = false;
  if (!inited) {
    enabled = std::getenv("VLLM_DEBUG_NVFP4_MOE_SM120") != nullptr;
    inited = 1;
  }
  return enabled;
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
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  };

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, EpilogueOperatorClass, typename MMA1SMConfig::MmaTileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutC*, AlignmentC,
          ElementD, LayoutD, AlignmentD,
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
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int);
  torch::Tensor c_strides1 =
      torch::full({num_experts}, output.stride(0), options_int);
  torch::Tensor a_strides1 =
      torch::full({num_experts}, a.stride(0) * 2, options_int);
  torch::Tensor b_strides1 =
      torch::full({num_experts}, b.stride(1) * 2, options_int);

  run_get_group_gemm_starts_sm120<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, alpha_ptrs,
      layout_sfa, layout_sfb, a, b, output, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes, M, N, K);

  Gemm gemm_op;
  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  if (nvfp4_sm120_debug_enabled()) {
    std::fprintf(stderr,
                 "[nvfp4-sm120] preparing grouped GEMM: num_experts=%d M=%d N=%d K=%d out_dtype=%d\n",
                 num_experts, M, N, K, static_cast<int>(output.scalar_type()));
  }

  cutlass::KernelHardwareInfo hw_info;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  hw_info.device_id = a.get_device();
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);
  }
  hw_info.sm_count = std::min(cached_sm_counts[hw_info.device_id], INT_MAX);

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides1.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides1.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides1.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides1.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  if (nvfp4_sm120_debug_enabled()) {
    std::fprintf(stderr,
                 "[nvfp4-sm120] can_implement status=%d (0==success)\n",
                 static_cast<int>(can_implement_status));
  }
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM (SM120 NVFP4 MoE)");

  auto status = gemm_op.initialize(args, workspace.data_ptr());
  if (nvfp4_sm120_debug_enabled()) {
    std::fprintf(stderr,
                 "[nvfp4-sm120] initialize status=%d (0==success)\n",
                 static_cast<int>(status));
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to initialize GEMM (SM120 NVFP4 MoE)");

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  if (nvfp4_sm120_debug_enabled()) {
    std::fprintf(stderr, "[nvfp4-sm120] run status=%d (0==success)\n",
                 static_cast<int>(status));
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to run GEMM (SM120 NVFP4 MoE)");

  if (nvfp4_sm120_debug_enabled()) {
    cudaError_t errSync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
      std::fprintf(stderr, "[nvfp4-sm120] cudaDeviceSynchronize error: %s\n",
                   cudaGetErrorString(errSync));
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[nvfp4-sm120] CUDA last error after run: %s\n",
                   cudaGetErrorString(err));
    }
  }
}

void cutlass_fp4_group_mm_sm120(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
  CHECK_INPUT(a, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(b, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(a_blockscale, SF_DTYPE, "a_blockscale");
  CHECK_INPUT(b_blockscales, SF_DTYPE, "b_blockscales");
  CHECK_INPUT(alphas, at::ScalarType::Float, "alphas");

  TORCH_CHECK(a_blockscale.dim() == 2,
              "expected a_blockscale to be of shape [num_experts, rounded_m,"
              " k // group_size], observed rank: ",
              a_blockscale.dim())
  TORCH_CHECK(b_blockscales.dim() == 3,
              "expected b_blockscale to be of shape: "
              " [num_experts, n, k // group_size], observed rank: ",
              b_blockscales.dim())
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be  a 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3,
              "problem_sizes must have the shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32.");

  int M = static_cast<int>(a.size(0));
  int N = static_cast<int>(b.size(1));
  int E = static_cast<int>(b.size(0));
  int K = static_cast<int>(2 * b.size(2));

  if (nvfp4_sm120_debug_enabled()) {
    std::fprintf(stderr,
                 "[nvfp4-sm120] dispatch: E=%d M=%d N=%d K=%d out_dtype=%d\n",
                 E, M, N, K, static_cast<int>(output.scalar_type()));
  }

  if (output.scalar_type() == torch::kBFloat16) {
    run_fp4_blockwise_scaled_group_mm_sm120<cutlass::bfloat16_t>(
        output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
        expert_offsets, sf_offsets, M, N, K);
  } else {
    run_fp4_blockwise_scaled_group_mm_sm120<cutlass::half_t>(
        output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
        expert_offsets, sf_offsets, M, N, K);
  }
}

#endif // ENABLE_NVFP4_SM120
