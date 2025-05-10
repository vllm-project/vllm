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

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include <cassert>

using namespace cute;

template <typename ElementAB, typename ElementC, typename ElementSF,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts(
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
  // Originally int32_t but upcasting to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  // size for block in block scale.
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert((m >= 0 && n == N && k == K && k % 2 == 0) &&
         "unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / group_size);
  // Shape of A as uint8/byte = [M, K // 2]
  // Shape of B as uint8/byte = [E, N, K // 2]
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;

  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  // Shape of a_scale = [sum(sf_sizes), K // group_size]
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");

  // Shape of B scale = [E, N, K // group_size]
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");
  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

#define __CALL_GET_STARTS_KERNEL_BLOCKSCALE(ELEMENT_AB_TYPE, SF_TYPE,         \
                                            TENSOR_C_TYPE, C_TYPE, LayoutSFA, \
                                            LayoutSFB, ScaleConfig)           \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                            \
    __get_group_gemm_starts<ELEMENT_AB_TYPE, C_TYPE, SF_TYPE, float,          \
                            LayoutSFA, LayoutSFB, ScaleConfig>                \
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
void run_get_group_gemm_starts(
    const torch::Tensor& a_starts, const torch::Tensor& b_starts,
    const torch::Tensor& out_starts, const torch::Tensor& a_scales_starts,
    const torch::Tensor& b_scales_starts, const torch::Tensor& alpha_starts,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    /*these are used for their base addresses*/
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
  //(ELEMENT_AB_TYPE, BS_TYPE, TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB,
  // ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(
      cutlass::float_e2m1_t, cutlass::float_ue4m3_t, torch::kBFloat16,
      cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(cutlass::float_e2m1_t,
                                      cutlass::float_ue4m3_t, torch::kFloat16,
                                      half, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

template <typename OutType>
void run_fp4_blockwise_scaled_group_mm(
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
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass =
      cutlass::arch::OpClassTensorOp;  // Epilogue Operator class tag
  using MainloopOperatorClass =
      cutlass::arch::OpClassBlockScaledTensorOp;  // Mainloop Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based
                                                  // on the tile size

  using ClusterShape = Shape<_1, _1, _1>;
  struct MMA1SMConfig {
    using MmaTileShape = Shape<_128, _128, _128>;
    using KernelSchedule = cutlass::gemm::
        KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, EpilogueOperatorClass, typename MMA1SMConfig::MmaTileShape,
          ClusterShape, Shape<_128, _64>, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutC*, AlignmentD,
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

  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
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

  run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, alpha_ptrs,
      layout_sfa, layout_sfb, a, b, output, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes, M, N, K);

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Set the Scheduler info
  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm100GroupParams<
          typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.get_device();
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);
  }
  hw_info.sm_count = min(cached_sm_counts[hw_info.device_id], INT_MAX);

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides1.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides1.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  // Epilogue Arguments
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

  // Gemm Arguments
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
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM");

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

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

void cutlass_fp4_group_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // Input validation
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

  if (output.scalar_type() == torch::kBFloat16) {
    run_fp4_blockwise_scaled_group_mm<cutlass::bfloat16_t>(
        output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
        expert_offsets, sf_offsets, M, N, K);
  } else {
    run_fp4_blockwise_scaled_group_mm<cutlass::half_t>(
        output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
        expert_offsets, sf_offsets, M, N, K);
  }
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_fp4_group_mm kernel, vLLM must "
      "be compiled with ENABLE_NVFP4 for SM100+ and CUDA "
      "12.8 or above.");
#endif
}
