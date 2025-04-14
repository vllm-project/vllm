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

#define DEBUG
using namespace cute;
static int getSMVersion()
{
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  return deviceProp.major * 10 + deviceProp.minor;
}


template <typename ElementAB, typename ElementC, typename ElementSF, typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void get_group_gemm_starts(
    int32_t *expert_offsets, ElementAB **a_offsets, ElementAB **b_offsets,
    ElementC **out_offsets, ElementSF **a_scales_offsets,
    ElementSF **b_scales_offsets, ElementAccumulator **alpha_offsets,
    ElementAB *a_base_as_int, ElementAB *b_base_as_int, ElementC *out_base_as_int,
    ElementSF *a_scales_base_as_int, ElementSF *b_scales_base_as_int,
    ElementAccumulator *alphas_base_as_int, LayoutSFA *layout_sfa_base_as_int,
    LayoutSFB *layout_sfb_base_as_int, int *problem_sizes_as_shapes,
    int64_t k, int64_t n)
{
  int expert_id = threadIdx.x;

  if (expert_id >= gridDim.x * blockDim.x)
  {
    return;
  }

  int64_t expert_offset = expert_offsets[expert_id];
  // size for block in block scale.
  // int64_t group_size = 16;
  int m = problem_sizes_as_shapes[expert_id * 3];
  int n_ = static_cast<int>(n);
  int k_ = static_cast<int>(k);
  
  // Shape of A = [M, K] 
  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  // Shape of B = [E, N, K]
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n;
  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  // Shape of a_scale = [M, k // group_size]
  a_scales_offsets[expert_id] =
      a_scales_base_as_int + expert_offset * k;
  
  // Shape of B scale = [E, N, K // group_size]
  b_scales_offsets[expert_id] =
      b_scales_base_as_int + expert_id * n * k;
  
  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA *layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB *layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n_, k_, 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n_, k_, 1));
}

#define __CALL_GET_STARTS_KERNEL_BLOCKSCALE(ELEMENT_AB_TYPE, SF_TYPE, TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)  \
  else if (out_tensors.dtype() == TENSOR_C_TYPE)                                                      \
  {                                                                                                   \
    get_group_gemm_starts<ELEMENT_AB_TYPE, C_TYPE, SF_TYPE, float, LayoutSFA, LayoutSFB, ScaleConfig> \
        <<<1, num_experts, 0, stream>>>(                                                              \
            static_cast<int32_t *>(expert_offsets.data_ptr()),                                        \
            static_cast<ELEMENT_AB_TYPE **>(a_starts.data_ptr()),                                     \
            static_cast<ELEMENT_AB_TYPE **>(b_starts.data_ptr()),                                     \
            static_cast<C_TYPE **>(out_starts.data_ptr()),                                            \
            static_cast<SF_TYPE **>(a_scales_starts.data_ptr()),                                      \
            static_cast<SF_TYPE **>(b_scales_starts.data_ptr()),                                      \
            static_cast<float **>(alpha_starts.data_ptr()),                                           \
            static_cast<ELEMENT_AB_TYPE *>(a_tensors.data_ptr()),                                     \
            static_cast<ELEMENT_AB_TYPE *>(b_tensors.data_ptr()),                                     \
            static_cast<C_TYPE *>(out_tensors.data_ptr()),                                            \
            static_cast<SF_TYPE *>(a_scales.data_ptr()),                                              \
            static_cast<SF_TYPE *>(b_scales.data_ptr()),                                              \
            static_cast<float *>(alphas.data_ptr()),                                                  \
            reinterpret_cast<LayoutSFA *>(layout_sfa.data_ptr()),                                     \
            reinterpret_cast<LayoutSFB *>(layout_sfb.data_ptr()),                                     \
            static_cast<int *>(problem_sizes.data_ptr()),                                             \
            k, n);                                            \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
      torch::Tensor const &expert_offsets, torch::Tensor &a_starts,
      torch::Tensor &b_starts, torch::Tensor &out_starts,
      torch::Tensor &a_scales_starts, torch::Tensor &b_scales_starts,
      torch::Tensor &alpha_starts, torch::Tensor const &a_tensors,
      torch::Tensor const &b_tensors, torch::Tensor &out_tensors,
      torch::Tensor const &a_scales, torch::Tensor const &b_scales, 
      torch::Tensor const &alphas, torch::Tensor const &layout_sfa,
      torch::Tensor const &layout_sfb, torch::Tensor const &problem_sizes)
{
  TORCH_CHECK(a_tensors.dtype() == at::ScalarType::Byte);
  TORCH_CHECK(b_tensors.dtype() == at::ScalarType::Byte);
  TORCH_CHECK(a_scales.dtype() == at::ScalarType::Float8_e4m3fn)
  TORCH_CHECK(b_scales.dtype() == at::ScalarType::Float8_e4m3fn)
    
  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  using ElementSFType =cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  // multiply a: [m, k]; b[e, n, k]
  // note that when actual cutlass kernels are called: k must be multiplied by 2
  int m = a_tensors.size(0);
  int n = b_tensors.size(1);
  int k = a_tensors.size(1);
  TORCH_CHECK(out_tensors.size(1) == n, "Output tensor shape doesn't match expected shape");
  TORCH_CHECK(k == b_tensors.size(2), "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
                                      " dimension must match");

  if (false)
  {
  }
  //(ELEMENT_AB_TYPE, BS_TYPE, TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::float_ue4m3_t,torch::kBFloat16, cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::float_ue4m3_t, torch::kFloat16, half, LayoutSFA, LayoutSFB, ScaleConfig)
  else
  {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}


// #ifdef DEBUG
template <typename T>
void print(const torch::Tensor& tensor, std::string text){
   torch::Tensor cpu_tensor = tensor.cpu();
   std::cout << "Printing: " ;
   auto data = cpu_tensor.data_ptr<T>(); 
   auto num_elements = cpu_tensor.numel();
   std::cout << text << " [" <<num_elements <<"]:\n";
   for (decltype(num_elements) i = 0; i < num_elements; ++i)
    {
      std::cout << "  " << std::hex << std::showbase << reinterpret_cast<void *>(data[i]) << std::endl;
    }
    std::cout << std::dec; // Reset number formatting
}
// #endif

template <typename OutType>
void run_fp4_blockwise_scaled_group_mm(
    torch::Tensor &output,
    const torch::Tensor &a,
    const torch::Tensor &b,
    const torch::Tensor &a_blockscale,
    const torch::Tensor &b_blockscales,
    const torch::Tensor &alphas,
    const torch::Tensor &stride_a,
    const torch::Tensor &stride_b,
    const torch::Tensor &stride_c,
    const torch::Tensor &layout_sfa,
    const torch::Tensor &layout_sfb,
    const torch::Tensor &problem_sizes,
    const torch::Tensor &expert_offsets)
{ 
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType =cutlass::float_ue4m3_t;
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
  
  std::cout << "Reached 195\n" ;
  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;               // Epilogue Operator class tag
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;    // Mainloop Operator class tag
  using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size


  using ClusterShape = Shape<int32_t,int32_t,_1>; 
  struct MMA1SMConfig {
    using MmaTileShape     = Shape<_128,_256,_256>;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;   // Kernel to launch
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;              // Epilogue to launch
    using OutputTileShape  = decltype(shape_div(MmaTileShape{}, Shape<_1,_1,_1>{}));
  };
  
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, EpilogueOperatorClass,
    typename MMA1SMConfig::OutputTileShape, ClusterShape,
    Shape<_128,_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutC *, AlignmentD,
    typename MMA1SMConfig::EpilogueSchedule
    // , FusionOperation  // Enable for SF Output
  >::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, MainloopOperatorClass,
  ElementA, LayoutA *, AlignmentA,
  ElementB, LayoutB *, AlignmentB,
  ElementAccumulator,
    typename MMA1SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    typename MMA1SMConfig::KernelSchedule
  >::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
                    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  std::cout << "Reached 255\n" ;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  std::cout <<"Found number of experts: "<< num_experts << "\n";
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  std::cout << "Reached 261\n" << options_int ;
  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);

  run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
          expert_offsets, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs,
          b_scales_ptrs, alpha_ptrs, a, b, output, a_blockscale, b_blockscales,
          alphas, layout_sfa, layout_sfb, problem_sizes);
 
  auto problem_sizes_ptr = static_cast<const int32_t *>(problem_sizes.data_ptr());

  // Use the expert offsets tensor directly
  auto expert_offsets_ptr = static_cast<const int32_t *>(expert_offsets.data_ptr());
        
  std::cout << "Reached 265\n" ;
  #ifdef DEBUG
  {
    print<int32_t>(expert_offsets, "expert_offsets");
    print<int64_t>(a_ptrs, "a_ptrs");
    print<int64_t>(b_ptrs, "b_ptrs");
    print<int64_t>(out_ptrs, "out_ptrs");
    print<int64_t>(a_scales_ptrs, "a_scales_ptrs");
    print<int64_t>(b_scales_ptrs, "b_scales_ptrs");
  }
  #endif
  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape *problem_sizes_as_shapes = static_cast<UnderlyingProblemShape *>(problem_sizes.data_ptr());
  // using ElementType = cutlass::float_e2m1_t;
  // using ElementSFType =cutlass::float_ue4m3_t;
  // using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  // using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ArrayElementA = typename GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename GemmKernel::CollectiveMainloop::ArrayElementB;
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ArrayElementA **>(a_ptrs.data_ptr()), 
      static_cast<StrideA *>(stride_a.data_ptr()),
      static_cast<const ArrayElementB **>(b_ptrs.data_ptr()),
      static_cast<StrideB *>(stride_b.data_ptr()),
      static_cast<const ElementSFType **>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA *>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType **>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB *>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.get_device();
  hw_info.cluster_shape = 2;
  // hw_info.cluster_shape.y = 1;
  hw_info.cluster_shape_fallback = 1; 
  // hw_info.cluster_shape_fallback.y = 1; 
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, //epilogue.thread
      nullptr,
      static_cast<StrideC *>(stride_c.data_ptr()),
      static_cast<ElementD **>(out_ptrs.data_ptr()),
      static_cast<StrideC *>(stride_c.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array = static_cast<ElementAccumulator**>(alpha_ptrs.data_ptr()); 

  // Use prob_shape in the GEMM arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, 
      {num_experts, problem_sizes_as_shapes, nullptr}, 
      mainloop_args, epilogue_args, hw_info, scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

void cutlass_blockscaled_fp4_group_mm(
    torch::Tensor &output,
    const torch::Tensor &a,
    const torch::Tensor &b,
    const torch::Tensor &a_blockscale,
    const torch::Tensor &b_blockscales,
    const torch::Tensor &alpha,
    const torch::Tensor &stride_a,
    const torch::Tensor &stride_b,
    const torch::Tensor &stride_c,
    const torch::Tensor &layout_sfa,
    const torch::Tensor &layout_sfb,
    const torch::Tensor &problem_sizes,
    const torch::Tensor &expert_offsets)
{
  // Differentiate M, N, K for full sized dimensions and m, n , k for actual tensor shapes.
  int M = a.size(0);
  int N = b.size(1);
  int E = b.size(0);
  int K = 2 * b.size(2);
  // Input validation
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");

  // Get output shapes and create output tensors
  auto sm_version = getSMVersion();
  std::cout << "Entered cultass_blockscaled_fp4_group_mm";

// #if defined ENABLE_NVFP4 && ENABLE_NVFP4

  if (sm_version == 100)
  {
    if (output.scalar_type() == torch::kBFloat16)
    {
      run_fp4_blockwise_scaled_group_mm<cutlass::bfloat16_t>(
          output, a, b, a_blockscale, b_blockscales, alpha, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets);
    }
    else
    {
      run_fp4_blockwise_scaled_group_mm<cutlass::half_t>(
          output, a, b, a_blockscale, b_blockscales, alpha, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets);
    }
  }
//#endif
  // TORCH_CHECK_NOT_IMPLEMENTED(false,
  //                             "No implemented cutlass_blockwise_scaled_mm_fp4 for current compute capability: ", sm_version);
}