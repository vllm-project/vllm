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

template <typename Gemm>
void cutlass_group_gemm_caller(c10::List<at::Tensor> const& out_tensors,
                               c10::List<at::Tensor> const& a_tensors,
                               c10::List<at::Tensor> const& b_tensors,
                               c10::List<at::Tensor> const& a_scales,
                               c10::List<at::Tensor> const& b_scales) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;

  int groups = (int)a_tensors.size();
  TORCH_CHECK((int)b_tensors.size() == groups,
              "Number of B tensors must match number of groups.");
  TORCH_CHECK((int)out_tensors.size() == groups,
              "Number of output tensors must match number of groups.");

  std::vector<const ElementAB*> a_ptrs_host(groups);
  std::vector<const ElementAB*> b_ptrs_host(groups);
  std::vector<const ElementC*> c_ptrs_host(groups);
  std::vector<ElementC*> d_ptrs_host(groups);
  std::vector<const ElementAccumulator*> a_scales_ptrs_host(groups);
  std::vector<const ElementAccumulator*> b_scales_ptrs_host(groups);

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  problem_sizes_host.reserve(groups);

  for (int g = 0; g < groups; ++g) {
    a_ptrs_host[g] =
        reinterpret_cast<const ElementAB*>(a_tensors[g].data_ptr());
    b_ptrs_host[g] =
        reinterpret_cast<const ElementAB*>(b_tensors[g].data_ptr());
    c_ptrs_host[g] =
        reinterpret_cast<const ElementC*>(out_tensors[g].data_ptr());
    d_ptrs_host[g] = reinterpret_cast<ElementC*>(out_tensors[g].data_ptr());
    a_scales_ptrs_host[g] =
        reinterpret_cast<const ElementAccumulator*>(a_scales[g].data_ptr());
    b_scales_ptrs_host[g] =
        reinterpret_cast<const ElementAccumulator*>(b_scales[g].data_ptr());

    // printf("%p %p %p %p %p %p %p\n", a_ptrs_host[g], b_ptrs_host[g],
    //        c_ptrs_host[g], d_ptrs_host[g],)
    int64_t m = a_tensors[g].size(0);
    int64_t k = a_tensors[g].size(1);

    int64_t k_b = b_tensors[g].size(0);
    int64_t n = b_tensors[g].size(1);

    TORCH_CHECK(k == k_b, "Dimension mismatch between A and B: A has k=", k,
                " while B has k=", k_b);

    // Optionally, verify output shape matches (m,n)
    TORCH_CHECK(out_tensors[g].size(0) == m && out_tensors[g].size(1) == n,
                "Output tensor shape does not match m,n from A,B: ", "Got ",
                out_tensors[g].sizes(), " expected (", m, ", ", n, ")");

    problem_sizes_host.push_back({(int)m, (int)n, (int)k});
  }

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  std::vector<StrideA> a_stride_host(groups);
  std::vector<StrideB> b_stride_host(groups);
  std::vector<StrideC> c_stride_host(groups);

  for (int32_t g = 0; g < groups; ++g) {
    int64_t lda = a_tensors[g].stride(0);    // row-major (m x k)
    int64_t ldb = b_tensors[g].stride(1);    // column-major (k x n)
    int64_t ldc = out_tensors[g].stride(0);  // row-major (m x n)

    a_stride_host[g] = StrideA{lda, Int<1>{}, Int<0>{}};
    b_stride_host[g] = StrideB{ldb, Int<1>{}, Int<0>{}};
    c_stride_host[g] = StrideC{ldc, Int<1>{}, Int<0>{}};
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  auto problem_sizes_ptr = make_device_ptr(problem_sizes_host);
  ProblemShape prob_shape{groups, problem_sizes_ptr.get(),
                          problem_sizes_host.data()};

  auto a_ptrs_ptr = make_device_ptr(a_ptrs_host);
  auto b_ptrs_ptr = make_device_ptr(b_ptrs_host);
  auto c_ptrs_ptr = make_device_ptr(c_ptrs_host);
  auto d_ptrs_ptr = make_device_ptr(d_ptrs_host);

  auto a_scales_ptrs_ptr = make_device_ptr(a_scales_ptrs_host);
  auto b_scales_ptrs_ptr = make_device_ptr(b_scales_ptrs_host);

  auto a_stride_ptr = make_device_ptr(a_stride_host);
  auto b_stride_ptr = make_device_ptr(b_stride_host);
  auto c_stride_ptr = make_device_ptr(c_stride_host);

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptrs_ptr.get(), a_stride_ptr.get(), b_ptrs_ptr.get(),
      b_stride_ptr.get()};
  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          a_scales_ptrs_ptr.get(), b_scales_ptrs_ptr.get(),
          a_scales[0].numel() != 1, b_scales[0].numel() != 1),
      c_ptrs_ptr.get(), c_stride_ptr.get(), d_ptrs_ptr.get(),
      c_stride_ptr.get()};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args, hw_info};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors[0].device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors[0].device().index());
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

void cutlass_grouped_mm_sm90(c10::List<at::Tensor> const& out_tensors,
                             c10::List<at::Tensor> const& a_tensors,
                             c10::List<at::Tensor> const& b_tensors,
                             c10::List<at::Tensor> const& a_scales,
                             c10::List<at::Tensor> const& b_scales) {
  TORCH_CHECK(a_tensors.size() > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size() > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size() > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors[0].dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors[0].dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      ElementAB_Type, ElementC_Type,
      vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales);
}

__global__ void get_a_expert_offsets(cutlass::float_e4m3_t** trg_a_ptrs,
                                     cutlass::float_e4m3_t* base_a_ptr,
                                     const int* __restrict__ topk_ids,
                                     int64_t* expert_offsets,
                                     int topk_length) {
  int expert_id = threadIdx.x;
  int num_experts = blockDim.x;

  int occurrences = 0;
  for (int i = 0; i < topk_length; ++i) {
    occurrences += (topk_ids[i] == expert_id);
  }
  expert_offsets[expert_id + 1] = occurrences;
  __syncthreads();

  if (threadIdx.x == 0) {
    int64_t tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      trg_a_ptrs[i] = base_a_ptr + tot_offset;
      tot_offset += expert_offsets[i + 1];
      expert_offsets[i + 1] = tot_offset;
    }
  }
}

// // For a given "a" of size [M,K] performs a permutation of the M rows based
// // on the given "perm" indices.
// __global__ void permute_fp8_rows_kernel(cutlass::float_e4m3_t const* __restrict__ a_ptr,
//                                     int const* __restrict__ perm_int_ptr,
//                                     cutlass::float_e4m3_t* __restrict__ out_ptr,
//                                     int size_m, int size_k, int block_rows) {
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

void compute_expert_offsets_caller(torch::Tensor& trg_a_ptrs,
                                   torch::Tensor& a,
                                   const torch::Tensor& topk_ids,
                                   torch::Tensor& expert_offsets,
                                   const int64_t num_experts) {
    get_a_expert_offsets<<<1, num_experts>>>(
      (cutlass::float_e4m3_t**)trg_a_ptrs.data_ptr(),
      (cutlass::float_e4m3_t*)a.data_ptr(),
      (const int*)topk_ids.data_ptr(),
      (int64_t*)expert_offsets.data_ptr(),
      topk_ids.numel());
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
