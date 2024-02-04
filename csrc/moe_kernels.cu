#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

//
#include <iostream>

#include "dispatch_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>

// #include "cutlass/platform/platform.h"
// #include "cutlass/bfloat16.h"
// #include "cutlass/complex.h"
// #include "cutlass/gemm/kernel/gemm_grouped.h"
// #include "cutlass/gemm/kernel/default_gemm_grouped.h"
// #include "cutlass/gemm/device/gemm_grouped.h"

#include "cutlass/cutlass.h"

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

using namespace cute;

namespace vllm {

#define CUDA_CALL(code)					                    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		        \
  } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;  // <M,N,K> per group
using ElementA = cutlass::bfloat16_t;                                       // Element type for A matrix operand
using ElementB = cutlass::bfloat16_t;                                       // Element type for B matrix operand
using ElementC = float;                                                     // Element type for C and D matrix operands

// A matrix configuration
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_256,_128,_64>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::KernelGroupTmaWarpSpecializedCooperative; // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecializedGroup;                     // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

std::vector<cutlass::gemm::GemmCoord> MakeProblemSizes(torch::Tensor b, torch::Tensor batch_sizes) {
  const size_t num_experts = batch_sizes.size(0);
  const size_t k = b.size(1), n = b.size(2);
  std::vector<cutlass::gemm::GemmCoord> problem_sizes(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    int64_t batch_size = batch_sizes.data_ptr<int64_t>()[i];
    problem_sizes[i] = cutlass::gemm::GemmCoord(batch_size, n, k);
  }
  return problem_sizes;
}

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
			    x.data(), bytes,
			    cudaMemcpyHostToDevice,
			    c10::cuda::getCurrentCUDAStream()));
  return out;
}

template <typename Gemm>
typename Gemm::Arguments MakeArguments(torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor batch_sizes) {
  auto problem_sizes_host = MakeProblemSizes(b, batch_sizes);

  // Calculate the number of threadblocks to use and validate the result.
  int64_t num_experts = problem_sizes_host.size();

  // NOTE: This is borrowed from FasterTransformer.
  int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);
  if (!threadblock_count) {
    TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
  }

  // Create the host arrays of leading dimension data and pointer data.
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
  std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
  std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
  int64_t elements_a = 0, elements_b = 0, elements_c = 0;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  std::vector<ElementA *> ptr_a_host(num_experts);
  std::vector<ElementB *> ptr_b_host(num_experts);
  std::vector<ElementC *> ptr_c_host(num_experts);

  for (int i = 0; i < num_experts; ++i) {
    auto problem = problem_sizes_host[i];
    lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    offsets_a[i] = elements_a;
    offsets_b[i] = elements_b;
    offsets_c[i] = elements_c;

    ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
    ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
    ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

    elements_a += problem.m() * problem.k();
    elements_b += problem.k() * problem.n();
    elements_c += problem.m() * problem.n();
  }

  // Copy the problem sizes, pointers and leading dimension data to the device.
  torch::Tensor lda = CopyToDevice(lda_host, a.device());
  torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
  torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
  torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
  torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
  torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
  torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());

  typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
  typename Gemm::Arguments arguments((cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
  				     (int)num_experts,
  				     (int)threadblock_count,
  				     epilogue_op,
  				     (ElementA**)ptr_a.data_ptr(),
  				     (ElementB**)ptr_b.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     (ElementC**)ptr_c.data_ptr(),
  				     /*lda=*/(int64_t*)lda.data_ptr(),
  				     /*ldb=*/(int64_t*)ldb.data_ptr(),
  				     /*ldc=*/(int64_t*)ldc.data_ptr(),
  				     /*ldd=*/(int64_t*)ldc.data_ptr(),
  				     (cutlass::gemm::GemmCoord*)problem_sizes_host.data());
  return arguments;
}

torch::Tensor CutlassGroupedGemm(torch::Tensor a,
				 torch::Tensor b,
				 torch::Tensor c,
				 torch::Tensor batch_sizes) {
  Gemm gemm;

  auto arguments = MakeArguments<Gemm>(a, b, c, batch_sizes);
  int64_t workspace_size = gemm.get_workspace_size(arguments);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  torch::Tensor workspace = torch::empty(workspace_size, options);

  // Initialize the kernel.
  if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
  if(gemm.run(c10::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }
  return c;
}

}

void fused_moe(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids //,
    // torch::Tensor sorted_token_ids,
    // torch::Tensor expert_ids,
    // torch::Tensor num_tokens_post_padded,
    // bool MUL_ROUTED_WEIGHT,
    // int top_k,
    // int parallelism
    ) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    vllm::CutlassGroupedGemm(A, B, C, topk_weights);
}