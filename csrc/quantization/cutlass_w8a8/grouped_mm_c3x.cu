#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

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

template <typename Gemm>
void cutlass_group_gemm_caller(torch::Tensor& out_tensors,
                               torch::Tensor const& a_tensors,
                               torch::Tensor const& b_tensors,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales,
                               torch::Tensor const& expert_offsets,
                               torch::Tensor const& problem_sizes) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;

  int groups = (int)expert_offsets.size(0);
  int k_size = a_tensors.size(1);
  int n_size = out_tensors.size(1);

  bool per_act_token = a_scales.numel() != groups;
  bool per_out_ch = b_scales.numel() != groups;

  int b_single_size = k_size * n_size;
  int b_scale_single_size = per_out_ch ? out_tensors.size(1) : 1;

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());
  torch::Tensor a_ptrs_base = torch::full(
      groups, reinterpret_cast<int64_t>(a_tensors.data_ptr()), options_int);
  torch::Tensor out_ptrs_base = torch::full(
      groups, reinterpret_cast<int64_t>(out_tensors.data_ptr()), options_int);
  torch::Tensor b_ptrs_base = torch::full(
      groups, reinterpret_cast<int64_t>(b_tensors.data_ptr()), options_int);
  torch::Tensor a_scales_base = torch::full(
      groups, reinterpret_cast<int64_t>(a_scales.data_ptr()), options_int);
  torch::Tensor b_scales_base = torch::full(
      groups, reinterpret_cast<int64_t>(b_scales.data_ptr()), options_int);

  torch::Tensor b_offsets =
      torch::arange(0, b_single_size * groups, b_single_size, options_int);
  torch::Tensor a_scales_offsets = torch::arange(0, groups, options_int);
  torch::Tensor b_scales_offsets = torch::arange(
      0, b_scale_single_size * groups, b_scale_single_size, options_int);

  torch::Tensor a_ptrs = a_ptrs_base.add(
      expert_offsets, sizeof(ElementAB_Type) * a_tensors.size(1));
  torch::Tensor out_ptrs = out_ptrs_base.add(
      expert_offsets, sizeof(ElementC_Type) * out_tensors.size(1));
  torch::Tensor b_ptrs = b_ptrs_base.add(b_offsets, sizeof(ElementAB_Type));

  torch::Tensor a_scales_ptrs =
      a_scales_base.add(per_act_token ? expert_offsets : a_scales_offsets,
                        sizeof(ElementAccumulator));
  torch::Tensor b_scales_ptrs =
      b_scales_base.add(b_scales_offsets, sizeof(ElementAccumulator));

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  std::vector<StrideA> a_stride_host(groups);
  std::vector<StrideB> b_stride_host(groups);
  std::vector<StrideC> c_stride_host(groups);

  // TODO pass strides?
  for (int32_t g = 0; g < groups; ++g) {
    int64_t lda = a_tensors.stride(0);    // row-major (m x k)
    int64_t ldb = a_tensors.stride(0);    // column-major (k x n)
    int64_t ldc = out_tensors.stride(0);  // row-major (m x n)

    a_stride_host[g] = StrideA{lda, Int<1>{}, Int<0>{}};
    b_stride_host[g] = StrideB{ldb, Int<1>{}, Int<0>{}};
    c_stride_host[g] = StrideC{ldc, Int<1>{}, Int<0>{}};
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape prob_shape{groups, problem_sizes_as_shapes, nullptr};

  auto a_stride_ptr = make_device_ptr(a_stride_host);
  auto b_stride_ptr = make_device_ptr(b_stride_host);
  auto c_stride_ptr = make_device_ptr(c_stride_host);

  typename GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<const ElementAB_Type**>(a_ptrs.data_ptr()),
      a_stride_ptr.get(),
      reinterpret_cast<const ElementAB_Type**>(b_ptrs.data_ptr()),
      b_stride_ptr.get()};

  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(reinterpret_cast<const ElementAccumulator**>(
                                       a_scales_ptrs.data_ptr()),
                                   reinterpret_cast<const ElementAccumulator**>(
                                       b_scales_ptrs.data_ptr()),
                                   per_act_token, per_out_ch),
      reinterpret_cast<const ElementC_Type**>(out_ptrs.data_ptr()),
      c_stride_ptr.get(),
      reinterpret_cast<ElementC_Type**>(out_ptrs.data_ptr()),
      c_stride_ptr.get()};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args, hw_info};

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

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>);
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

// TODO
void cutlass_grouped_mm_sm90(torch::Tensor& out_tensors,
                             torch::Tensor const& a_tensors,
                             torch::Tensor const& b_tensors,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             torch::Tensor const& expert_offsets,
                             torch::Tensor const& problem_sizes) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      ElementAB_Type, ElementC_Type,
      vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes);
}

__global__ void get_a_expert_offsets(const int* __restrict__ topk_ids,
                                     int32_t* expert_offsets,
                                     int32_t* problem_sizes1,
                                     int32_t* problem_sizes2, int topk_length,
                                     int n, int k) {
  int expert_id = threadIdx.x;
  int num_experts = blockDim.x;

  int occurrences = 0;
  for (int i = 0; i < topk_length; ++i) {
    occurrences += (topk_ids[i] == expert_id);
  }
  problem_sizes1[expert_id * 3] = occurrences;
  problem_sizes1[expert_id * 3 + 1] = 2 * n;
  problem_sizes1[expert_id * 3 + 2] = k;
  problem_sizes2[expert_id * 3] = occurrences;
  problem_sizes2[expert_id * 3 + 1] = k;
  problem_sizes2[expert_id * 3 + 2] = n;
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      tot_offset += problem_sizes1[i * 3];
      expert_offsets[i + 1] = tot_offset;
    }
  }
}

// // For a given "a" of size [M,K] performs a permutation of the M rows based
// // on the given "perm" indices.
// __global__ void permute_fp8_rows_kernel(cutlass::float_e4m3_t const*
// __restrict__ a_ptr,
//                                     int const* __restrict__ perm_int_ptr,
//                                     cutlass::float_e4m3_t* __restrict__
//                                     out_ptr, int size_m, int size_k, int
//                                     block_rows) {
//   int start_row = block_rows * blockIdx.x;
//   int finish_row = start_row + block_rows;
//   if (finish_row > size_m) {
//     finish_row = size_m;
//   }
//   int cur_block_rows = finish_row - start_row;

//   int row_stride = size_k * sizeof(cutlass::float_e4m3_t) / 16;

//   auto permute_row = [&](int row) {
//     int iters = size_k / blockDim.x;
//     int rest = size_k % blockDim.x;

//     int a_offset = perm_int_ptr[row] * row_stride;
//     int out_offset = row * row_stride;

//     cutlass::float_e4m3_t const* a_row_fp8 = a_ptr + a_offset;
//     cutlass::float_e4m3_t* out_fp8 = out_ptr + out_offset;

//     int base_k = 0;

//     for (int i = 0; i < iters; i++) {
//       int cur_k = base_k + threadIdx.x;
//       out_fp8[cur_k] = a_row_fp8[cur_k];
//       base_k += blockDim.x;
//     }

//     if (rest) {
//       if (threadIdx.x < rest) {
//         int cur_k = base_k + threadIdx.x;
//         out_fp8[cur_k] = a_row_fp8[cur_k];
//       }
//     }
//   };
// }

void compute_expert_offsets_caller(const torch::Tensor& topk_ids,
                                   torch::Tensor& expert_offsets,
                                   torch::Tensor& problem_sizes1,
                                   torch::Tensor& problem_sizes2,
                                   const int64_t num_experts, const int64_t n,
                                   const int64_t k) {
  get_a_expert_offsets<<<1, num_experts>>>(
      (const int32_t*)topk_ids.data_ptr(), (int32_t*)expert_offsets.data_ptr(),
      (int32_t*)problem_sizes1.data_ptr(), (int32_t*)problem_sizes2.data_ptr(),
      topk_ids.numel(), n, k);
}

// void permute_fp8_rows(torch::Tensor& a_ptr,
//                       torch::Tensor& perm_ptr,
//                       torch::Tensor& out_ptr,
//                       int size_m, int size_k, int topk, int block_rows) {
//     permute_fp8_rows_kernel<<<blocks, num_threads, 0, stream>>>(
//           (cutlass::float_e4m3_t const*)a_ptr.data_ptr(),
//           (const int*)perm_ptr.data_ptr(),
//           (cutlass::float_e4m3_t const*)out_ptr.data_ptr(), size_m * topk,
//           size_k, block_rows);
// }
