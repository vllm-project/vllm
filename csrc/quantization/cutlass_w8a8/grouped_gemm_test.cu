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

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;  // <M,N,K>
                                                                   // per group
using ElementAB_Type =
    cutlass::float_e4m3_t;  // Element type for A matrix operand
// using ElementB = cutlass::float_e4m3_t;                                    //
// Element type for B matrix operand
using ElementC_Type = cutlass::half_t;

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

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

  using StrideC =
      cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  const int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  const int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using EVTCompute = typename Epilogue::EVTCompute;
  // the orig hat  cutlass::epilogue::fusion::LinearCombination<ElementC,
  // ElementAccumulator>

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, 4, ElementC, LayoutC*, 4,
          EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA*, 16, ElementAB, LayoutB*,
          16, ElementAccumulator, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename T>
struct ItemDeleter {
  void operator()(T* ptr) {
    cudaFree(ptr);  // noexcept
  }
};

template <typename T>
cutlass::platform::unique_ptr<T, ItemDeleter<T>> make_device_ptr(
    std::vector<T>& data_host) {
  T* data_device;
  int count = data_host.size();
  cudaMalloc(&data_device, count * sizeof(T));
  cudaMemcpy(data_device, data_host.data(), count * sizeof(T),
             cudaMemcpyHostToDevice);
  return cutlass::platform::unique_ptr<T, ItemDeleter<T>>(data_device);
}

///////////////
template <class TupType, size_t... I>
void print(const TupType& _tup, std::index_sequence<I...>) {
  std::cout << "(";
  (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
  std::cout << ")\n";
}

template <class... T>
void print(const std::tuple<T...>& _tup) {
  print(_tup, std::make_index_sequence<sizeof...(T)>());
}
////////////

template <typename Gemm, typename... EpilogueArgs>
void cutlass_group_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& problem_sizes,
                               torch::Tensor const& out_offsets,
                               torch::Tensor const& a_offsets,
                               torch::Tensor const& b_offsets,
                               EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;
  using ElementAcc = float;

  int groups = problem_sizes.size(0);
  std::vector<const ElementAB*> a_ptrs_host(groups);
  std::vector<const ElementAB*> b_ptrs_host(groups);
  std::vector<const ElementC*> c_ptrs_host(groups);
  std::vector<ElementC*> d_ptrs_host(groups);

  for (int g = 0; g < groups; ++g) {
    a_ptrs_host.at(g) = static_cast<const ElementAB*>(a.data_ptr()) +
                        a_offsets[g].item<int32_t>();
    b_ptrs_host.at(g) = static_cast<const ElementAB*>(b.data_ptr()) +
                        b_offsets[g].item<int32_t>();
    c_ptrs_host.at(g) = static_cast<const ElementC*>(out.data_ptr()) +
                        out_offsets[g].item<int32_t>();
    d_ptrs_host.at(g) =
        static_cast<ElementC*>(out.data_ptr()) + out_offsets[g].item<int32_t>();
    printf("off: %d %d %d\n", a_offsets[g].item<int32_t>(),
           b_offsets[g].item<int32_t>(), out_offsets[g].item<int32_t>());
  }

  using GemmKernel = typename Gemm::GemmKernel;

  // using StrideA = typename GemmKernel::InternalStrideA;
  // using StrideB = typename GemmKernel::InternalStrideB;
  // using StrideC = typename GemmKernel::InternalStrideC;
  // // using StrideD = typename GemmKernel::InternalStrideD;

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC =
      typename GemmKernel::InternalStrideC;  // typename Gemm::StrideC;

  // StrideA a_stride{lda, Int<1>{}, Int<0>{}};
  // StrideB b_stride{ldb, Int<1>{}, Int<0>{}};
  // StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  std::vector<StrideA> a_stride_host(groups, StrideA{lda, Int<1>{}, Int<0>{}});
  std::vector<StrideB> b_stride_host(groups, StrideB{ldb, Int<1>{}, Int<0>{}});
  std::vector<StrideC> c_stride_host(groups, StrideC{ldc, Int<1>{}, Int<0>{}});

  printf("a: ");
  print(a_stride_host[0]);
  printf("\nb: ");
  print(b_stride_host[0]);
  printf("\nc: ");
  print(c_stride_host[0]);
  printf("\n");

  // for (int g = 0; g < groups; ++g) {
  //   int32_t m = problem_sizes[g][0].item<int32_t>();
  //   int32_t n = problem_sizes[g][1].item<int32_t>();
  //   int32_t k = problem_sizes[g][2].item<int32_t>();
  //   a_stride_host[g] = StrideA{k, cute::Int<1>{}, cute::Int<0>{}};  // m x k,
  //                                                                   // row
  //   b_stride_host[g] = StrideB{k, cute::Int<1>{}, cute::Int<0>{}};  // k x n,
  //                                                                   // col
  //   c_stride_host[g] = StrideC{n, cute::Int<1>{}, cute::Int<0>{}};  // m x n,
  //                                                                   // row
  // }

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using SingleProblemShape = typename ProblemShape::UnderlyingProblemShape;

  std::vector<SingleProblemShape> problem_sizes_host;
  problem_sizes_host.reserve(groups);
  for (int32_t g = 0; g < groups; ++g) {
    int32_t m = problem_sizes[g][0].item<int32_t>();
    int32_t n = problem_sizes[g][1].item<int32_t>();
    int32_t k = problem_sizes[g][2].item<int32_t>();
    problem_sizes_host.push_back({m, n, k});
    printf("mnk: %d, %d, %d\n", m, n, k);
  }

  auto problem_sizes_ptr =
      make_device_ptr<SingleProblemShape>(problem_sizes_host);
  ProblemShape prob_shape{groups, problem_sizes_ptr.get(),
                          problem_sizes_host.data()};

  // ElementAB* a_host_print;
  // int numel = a.numel();
  // cudaMalloc(&a_host_print, groups * sizeof(ElementAB));
  // cudaMemcpy(a_host_print, static_cast<ElementAB*>(a.data_ptr()), numel*
  // sizeof(ElementAB), cudaMemcpyDeviceToHost);
  // cudaMemcpy(static_cast<ElementAB*>(a.data_ptr()), a_host_print, numel*
  // sizeof(ElementAB), cudaMemcpyHostToDevice); cudaFree(a_host_print);

  auto a_ptrs_ptr = make_device_ptr<const ElementAB*>(a_ptrs_host);
  auto b_ptrs_ptr = make_device_ptr<const ElementAB*>(b_ptrs_host);
  auto c_ptrs_ptr = make_device_ptr<const ElementC*>(c_ptrs_host);
  auto d_ptrs_ptr = make_device_ptr<ElementC*>(d_ptrs_host);

  auto a_stride_ptr = make_device_ptr<StrideA>(a_stride_host);
  auto b_stride_ptr = make_device_ptr<StrideB>(b_stride_host);
  auto c_stride_ptr = make_device_ptr<StrideC>(c_stride_host);

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptrs_ptr.get(), a_stride_ptr.get(), b_ptrs_ptr.get(),
      b_stride_ptr.get()};
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptrs_ptr.get(), c_stride_ptr.get(), d_ptrs_ptr.get(),
      c_stride_ptr.get()};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args, hw_info};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

// typedef InType = cutlass::float_e4m3_t;
// typedef OutType = torch::half;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
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
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
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
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

}  // namespace

// TODO hardcode types here?
void cutlass_grouped_mm_sm90(
    torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& a_scales, torch::Tensor const& b_scales,
    torch::Tensor const& problem_sizes, torch::Tensor const& out_offsets,
    torch::Tensor const& a_offsets, torch::Tensor const& b_offsets) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);
  // int32_t m = a.size(1);

  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      ElementAB_Type, ElementC_Type, vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;
  // using Cutlass3xGemmM64 =
  //     typename sm90_fp8_config_M64<ElementAB_Type, ElementC_Type,
  //     vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;
  // using Cutlass3xGemmM128 =
  //     typename sm90_fp8_config_M128<ElementAB_Type, ElementC_Type,
  //     vllm::c3x::ScaledEpilogue>::Cutlass3xGemm;

  // // uint32_t const m = a.size(0);
  // uint32_t const mp2 =
  //     std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  // if (mp2 <= 64) {
  //   // m in [1, 64]
  //   cutlass_group_gemm_caller<Cutlass3xGemmM64>(out, a, b, a_scales,
  //   b_scales);
  // } else if (mp2 <= 128) {
  //   // m in (64, 128]
  //   cutlass_group_gemm_caller<Cutlass3xGemmM128>(out, a, b, a_scales,
  //   b_scales);
  // } else {
  //   // m in (128, inf)
  cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
      out, a, b, problem_sizes, out_offsets, a_offsets, b_offsets, a_scales,
      b_scales);
  // }
}
