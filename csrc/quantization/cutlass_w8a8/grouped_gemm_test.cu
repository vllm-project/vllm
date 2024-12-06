#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"

// TODO let's see which of these we'll need

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

#include "common.hpp"

// get rid of these?
// #include "helper.h"
// using namespace cute;

using namespace cute;

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
#define ENABLE_SM90_KERNEL_LEVEL 1
#endif

namespace {

  // A wrapper for the GEMM kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
// into code that will be executed on the device where it is defined.
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
  #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #endif
  }
};

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>; // <M,N,K> per group
using ElementAB_Type = cutlass::float_e4m3_t;                                    // Element type for A matrix operand
// using ElementB = cutlass::float_e4m3_t;                                    // Element type for B matrix operand
using ElementC_Type = cutlass::half_t;

// // A matrix configuration
// using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
// constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

// // B matrix configuration
// using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
// constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Alignment of B matrix in units of elements (up to 16 bytes)

// // C/D matrix configuration
// using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
// constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
// using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size

// Different configs for pingpong/cooperative
// struct CooperativeConfig {
//   using KernelSchedule = cutlass::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
//   using EpilogueSchedule = cutlass::KernelPtrArrayTmaWarpSpecializedCooperative;
//   using TileShape           = cute::Shape<cute::_256,cute::_128,cute::_128>;
//   using ClusterShape        = cute::Shape<cute::_2,cute::_2,cute::_1>;
// };

using         LayoutA     = cutlass::layout::RowMajor;  
using         LayoutB     = cutlass::layout::ColumnMajor;  
using         LayoutC     = cutlass::layout::ColumnMajor;      

template <typename ElementAB_, typename ElementC_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_group_gemm {

  using ElementAB = ElementAB_;
  using ElementC = ElementC_;
  using ElementAccumulator = float;

  using EpilogueDescriptor =
      cutlass::epilogue::collective::detail::EpilogueDescriptor<
          TileShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementC,
          ElementC, EpilogueSchedule>;

  using Epilogue = Epilogue_<ElementAccumulator, ElementC, EpilogueDescriptor>;

  using StrideC = cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  const int AlignmentAB  = 128 / cutlass::sizeof_bits<ElementAB>::value; 
  const int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value; 

  using EVTCompute = typename Epilogue::EVTCompute;
  // the orig hat  cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, 4,
    ElementC, LayoutC*, 4,
    EpilogueSchedule, EVTCompute
  >::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementAB, LayoutA*, 16,
    ElementAB, LayoutB*, 16,
    ElementAccumulator,
    TileShape, ClusterShape,
    Stages, KernelSchedule
  >::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename T>
struct ItemDeleter {
  void operator()(T* ptr) {
    cudaFree(ptr); // noexcept
  }
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_group_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                         torch::Tensor const& b,
                         torch::Tensor const& problem_sizes,
                         torch::Tensor const& out_offsets,
                         torch::Tensor const& a_offsets,
                         torch::Tensor const& b_offsets,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  // using ElementC = typename Gemm::ElementC;
  using ElementC = typename Gemm::ElementC;
  using ElementAcc = float;

  int groups = problem_sizes.size(0);
  std::vector<ElementAB*> a_ptrs_host(groups);
  std::vector<ElementAB*> b_ptrs_host(groups);
  std::vector<ElementC*> c_ptrs_host(groups);
  std::vector<ElementC*> d_ptrs_host(groups);

  for (int g = 0; g < groups; ++g) {
    a_ptrs_host.at(g) = (ElementAB*)a.data_ptr();// + a_offsets[g].item<int32_t>();
    b_ptrs_host.at(g) = (ElementAB*)b.data_ptr();// + b_offsets[g].item<int32_t>();
    c_ptrs_host.at(g) = (ElementC*)out.data_ptr();// + out_offsets[g].item<int32_t>();
    d_ptrs_host.at(g) = (ElementC*)out.data_ptr();// + out_offsets[g].item<int32_t>();
  }

  // int32_t groups = a.size(0);
  // int32_t m = a.size(1);
  // int32_t n = b.size(2);
  // int32_t k = a.size(2);

  // int64_t lda = a.stride(1);
  // int64_t ldb = b.stride(2);
  // int64_t ldc = out.stride(1);

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  // StrideA stride_A{lda, cute::Int<1>{}, 0};
  // StrideB stride_B{ldb, cute::Int<1>{}, 0};
  // StrideC stride_C{ldc, cute::Int<1>{}, cute::Int<0>{}};

  // this should be vector of A ptrs
  // auto ptr_A = static_cast<ElementAB*>(a.data_ptr());
  // auto ptr_B = static_cast<ElementAB*>(b.data_ptr());
  // auto ptr_C = static_cast<ElementC*>(out.data_ptr());

  cutlass::platform::unique_ptr<StrideA> stride_A;
  cutlass::platform::unique_ptr<StrideB> stride_B;
  cutlass::platform::unique_ptr<StrideC> stride_C;
  cutlass::platform::unique_ptr<StrideD> stride_D;

  cutlass::platform::unique_ptr<const ElementAB*> ptr_A;
  cutlass::platform::unique_ptr<const ElementAB*> ptr_B;
  cutlass::platform::unique_ptr<const ElementC*> ptr_C;
  cutlass::platform::unique_ptr<ElementC*> ptr_D;

  using GemmKernel = typename Gemm::GemmKernel;
  
  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using SingleProblemShape = typename ProblemShape::UnderlyingProblemShape;

  std::vector<SingleProblemShape> problem_sizes_host;
  problem_sizes_host.reserve(groups);
  for (int32_t g = 0; g < groups; ++g) {
    int32_t m = problem_sizes[g][0].item<int32_t>();
    int32_t n = problem_sizes[g][1].item<int32_t>();
    int32_t k = problem_sizes[g][2].item<int32_t>();
    problem_sizes_host.push_back({m, n, k});
  }

  SingleProblemShape* problem_sizes_device;
  int32_t problem_sizes_size = groups * sizeof(SingleProblemShape);
  cudaMalloc(&problem_sizes_device, problem_sizes_size);
  cudaMemcpy(problem_sizes_device, problem_sizes_host.data(), groups,
       cudaMemcpyHostToDevice);
  cutlass::platform::unique_ptr<SingleProblemShape, ItemDeleter<SingleProblemShape>> problem_sizes_ptr(
    problem_sizes_device);
  ProblemShape prob_shape{groups, problem_sizes_ptr.get(), problem_sizes_host.data()};

  const ElementAB** a_ptrs_device;
  cudaMalloc(&a_ptrs_device, groups * sizeof(ElementAB*));
  cudaMemcpy(a_ptrs_device, a_ptrs_host.data(), groups,cudaMemcpyHostToDevice);
  cutlass::platform::unique_ptr<const ElementAB*, ItemDeleter<const ElementAB*>> a_ptrs_ptr(
    a_ptrs_device
  );

  const ElementAB** b_ptrs_device;
  cudaMalloc(&b_ptrs_device, groups * sizeof(ElementAB*));
  cudaMemcpy(b_ptrs_device, b_ptrs_host.data(), groups,cudaMemcpyHostToDevice);
  cutlass::platform::unique_ptr<const ElementAB*, ItemDeleter<const ElementAB*>> b_ptrs_ptr(
    b_ptrs_device
  );

  const ElementC** c_ptrs_device;
  cudaMalloc(&c_ptrs_device, groups * sizeof(ElementC*));
  cudaMemcpy(c_ptrs_device, c_ptrs_host.data(), groups,cudaMemcpyHostToDevice);
  cutlass::platform::unique_ptr<const ElementC*, ItemDeleter<const ElementC*>> c_ptrs_ptr(
    c_ptrs_device
  );

  ElementC** d_ptrs_device;
  cudaMalloc(&d_ptrs_device, groups * sizeof(ElementC*));
  cudaMemcpy(d_ptrs_device, d_ptrs_host.data(), groups,cudaMemcpyHostToDevice);
  cutlass::platform::unique_ptr<ElementC*, ItemDeleter<ElementC*>> d_ptrs_ptr(
    d_ptrs_device
  );

  typename GemmKernel::MainloopArguments mainloop_args{
    a_ptrs_ptr.get(), stride_A.get(), b_ptrs_ptr.get(), stride_B.get()};
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptrs_ptr.get(), stride_C.get(), d_ptrs_ptr.get(), stride_D.get()};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      prob_shape,
      mainloop_args,
      epilogue_args,
      hw_info
    };

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  // // auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.initialize(args, workspace.data_ptr()));

  // #if defined(ENABLE_SM90_KERNEL_LEVEL)
    // printf("did run through\n");
    cutlass::Status status = gemm_op.run();
    CUTLASS_CHECK(status);
  // #endif

}

// typedef InType = cutlass::float_e4m3_t;
// typedef OutType = torch::half;
// typedef Epilogue = ScaledEpilogueBias;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M64 {
  // M in [1, 64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

}

// TODO hardcode types here?
void cutlass_grouped_mm_sm90(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b, torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            torch::Tensor const& problem_sizes,
                            torch::Tensor const& out_offsets,
                            torch::Tensor const& a_offsets,
                            torch::Tensor const& b_offsets) {

  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);
  // int32_t m = a.size(1); 

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config_default<ElementAB_Type, ElementC_Type,
                                       vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;
  // using Cutlass3xGemmM64 =
  //     typename sm90_fp8_config_M64<ElementAB_Type, ElementC_Type, vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;
  // using Cutlass3xGemmM128 =
  //     typename sm90_fp8_config_M128<ElementAB_Type, ElementC_Type, vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;


  // // uint32_t const m = a.size(0);
  // uint32_t const mp2 =
  //     std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  // if (mp2 <= 64) {
  //   // m in [1, 64]
  //   cutlass_group_gemm_caller<Cutlass3xGemmM64>(out, a, b, a_scales, b_scales);
  // } else if (mp2 <= 128) {
  //   // m in (64, 128]
  //   cutlass_group_gemm_caller<Cutlass3xGemmM128>(out, a, b, a_scales, b_scales);
  // } else {
  //   // m in (128, inf)
    cutlass_group_gemm_caller<Cutlass3xGemmDefault>(out, a, b, problem_sizes,
                    out_offsets, a_offsets, b_offsets, a_scales, b_scales);
  // }

}
