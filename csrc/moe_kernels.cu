#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

//
#include <iostream>

#include "dispatch_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/platform/platform.h"
#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

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

using DefaultConfig = ::cutlass::gemm::device::DefaultGemmConfiguration<::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80, ::cutlass::bfloat16_t, ::cutlass::bfloat16_t, ::cutlass::bfloat16_t, float>;


// TODO(tgale): Update this for SM90 when it's supported by CUTLASS.
using GroupedGemmKernelNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  // Non-transposed A operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // Non-transposed B operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // C operand.
  ::cutlass::bfloat16_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<32, 128, 64>,
  ::cutlass::gemm::GemmShape<32, 32, 64>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  // DefaultConfig::ThreadblockShape,
  // DefaultConfig::WarpShape,
  // DefaultConfig::InstructionShape,
  // ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
  DefaultConfig::EpilogueOutputOp,
  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  // TODO(tgale): Experiment with GroupScheduleMode.
  // TODO(tgale): Tune this for SM90.
  DefaultConfig::kStages>::GemmKernel;
using GemmGroupedNN = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernelNN>;

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
  using Gemm = GemmGroupedNN;
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