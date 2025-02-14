#include "cutlass/cutlass.h"

// TODO clean up the includes we no longer need

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

#include "cutlass_extensions/common.hpp"

using namespace cute;

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
  #define ENABLE_SM90_KERNEL_LEVEL 1
#endif

// for debugging
// __global__ void print_elements(int64_t* tensor, int64_t elements) {
//   if (threadIdx.x == 0) {
//     for (int64_t i = 0; i < elements; ++i) {
//       printf("%ld/%ld ", i, tensor[i]);
//     }
//     printf("\n---\n");
//   }
// }

namespace {

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
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementAB_Type = cutlass::float_e4m3_t;
using ElementC_Type = cutlass::half_t;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

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

template <typename T>
cutlass::platform::unique_ptr<T, ItemDeleter<T>> allocate_device_ptr(
    int count) {
  T* data_device;
  cudaMalloc(&data_device, count * sizeof(T));
  return cutlass::platform::unique_ptr<T, ItemDeleter<T>>(data_device);
}

template <typename Gemm>
void cutlass_group_gemm_caller(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;

  int groups = (int)expert_offsets.size(0);
  int k_size = a_tensors.size(1);
  int n_size = out_tensors.size(1);

  bool per_act_token = a_scales.numel() != groups;
  bool per_out_ch = b_scales.numel() != groups;

  int b_single_size = k_size * n_size * sizeof(ElementAB_Type);
  int b_scale_single_size =
      (per_out_ch ? out_tensors.size(1) : 1) * sizeof(ElementAccumulator);

  // TODO b and b scales pointers can be computed outside this function
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());
  torch::Tensor a_ptrs_base = torch::full(
      groups, reinterpret_cast<int64_t>(a_tensors.data_ptr()), options_int);
  torch::Tensor out_ptrs_base = torch::full(
      groups, reinterpret_cast<int64_t>(out_tensors.data_ptr()), options_int);
  torch::Tensor a_scales_base = torch::full(
      groups, reinterpret_cast<int64_t>(a_scales.data_ptr()), options_int);

  torch::Tensor a_scales_offsets = torch::arange(0, groups, options_int);

  torch::Tensor a_ptrs = a_ptrs_base.add(
      expert_offsets, sizeof(ElementAB_Type) * a_tensors.size(1));
  torch::Tensor out_ptrs = out_ptrs_base.add(
      expert_offsets, sizeof(ElementC_Type) * out_tensors.size(1));

  torch::Tensor a_scales_ptrs =
      a_scales_base.add(per_act_token ? expert_offsets : a_scales_offsets,
                        sizeof(ElementAccumulator));

  int64_t b_tensor_base_addr = reinterpret_cast<int64_t>(b_tensors.data_ptr());
  int64_t b_scales_base_addr = reinterpret_cast<int64_t>(b_scales.data_ptr());

  torch::Tensor b_ptrs = torch::arange(
      b_tensor_base_addr, b_tensor_base_addr + b_single_size * groups,
      b_single_size, options_int);
  torch::Tensor b_scales_ptrs = torch::arange(
      b_scales_base_addr, b_scales_base_addr + b_scale_single_size * groups,
      b_scale_single_size, options_int);

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape prob_shape{groups, problem_sizes_as_shapes, nullptr};

  typename GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<const ElementAB_Type**>(a_ptrs.data_ptr()),
      reinterpret_cast<StrideA*>(a_strides.data_ptr()),
      reinterpret_cast<const ElementAB_Type**>(b_ptrs.data_ptr()),
      reinterpret_cast<StrideB*>(b_strides.data_ptr())};

  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(reinterpret_cast<const ElementAccumulator**>(
                                       a_scales_ptrs.data_ptr()),
                                   reinterpret_cast<const ElementAccumulator**>(
                                       b_scales_ptrs.data_ptr()),
                                   per_act_token, per_out_ch),
      reinterpret_cast<const ElementC_Type**>(out_ptrs.data_ptr()),
      reinterpret_cast<StrideC*>(c_strides.data_ptr()),
      reinterpret_cast<ElementC_Type**>(out_ptrs.data_ptr()),
      reinterpret_cast<StrideC*>(c_strides.data_ptr())};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace
