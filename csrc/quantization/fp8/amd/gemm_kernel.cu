#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#ifndef CHECK_HIP_ERROR
  #define CHECK_HIP_ERROR(error)                                    \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",             \
              hipGetErrorString(error), error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                           \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
  #define CHECK_HIPBLASLT_ERROR(error)                                  \
    if (error != HIPBLAS_STATUS_SUCCESS) {                              \
      fprintf(stderr, "hipBLASLt error: '%s'(%d) at %s:%d\n",           \
              hipblasStatusToString(error), error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                               \
    }
#endif

/* TODO(HaiShaw): Have hipblasLt support mixed precision, s.t. input Tensors
                  a and b can be float16 or bfloat16 type (performance A.I.)
		  Extend interface to be more generic, to include bias, etc.
 */
void fp8_mm(torch::Tensor& a, torch::Tensor& b, torch::Tensor& result,
            torch::Tensor& scale_a, torch::Tensor& scale_b,
            const c10::optional<torch::Tensor>& scale_result, int64_t solidx) {
  auto a_strides{a.strides()};
  auto b_strides{b.strides()};
  auto a_sizes{a.sizes()};
  auto b_sizes{b.sizes()};

  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fnuz &&
                  b.dtype() == torch::kFloat8_e4m3fnuz,
              "The input tensors type should be float8_e4m3fnuz.");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-D.");
  TORCH_CHECK(a_sizes[1] == b_sizes[0], "a dim 1 must match b dim 0.");

  auto out_dtype = result.dtype();
  TORCH_CHECK(out_dtype == torch::kFloat8_e4m3fnuz ||
                  out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
              "Only float16, bfloat16 or float8_e4m3fnuz are supported as the "
              "output dtype.");
  hipblasDatatype_t hipblas_out_type;
  if (out_dtype == torch::kFloat8_e4m3fnuz) {
    hipblas_out_type = HIP_R_8F_E4M3_FNUZ;
  } else if (out_dtype == torch::kBFloat16) {
    hipblas_out_type = HIP_R_16BF;
  } else {
    hipblas_out_type = HIP_R_16F;
  }

  constexpr bool transpose_result = true;
  bool transpose_a;
  bool transpose_b;
  if ((b_strides[0] == 1) &&
      (b_strides[1] >= std::max<int64_t>(1, b_sizes[0]))) {
    transpose_b = false;
  } else if ((b_strides[1] == 1) &&
             (b_strides[0] >= std::max<int64_t>(1, b_sizes[1]))) {
    transpose_b = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }
  if ((a_strides[0] == 1) &&
      (a_strides[1] >= std::max<int64_t>(1, a_sizes[0]))) {
    transpose_a = false;
  } else if ((a_strides[1] == 1) &&
             (a_strides[0] >= std::max<int64_t>(1, a_sizes[1]))) {
    transpose_a = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }

  if (transpose_result) {
    bool tmp = transpose_a;
    transpose_a = !transpose_b;
    transpose_b = !tmp;
    a_strides = b.strides();
    b_strides = a.strides();
    a_sizes = b.sizes();
    b_sizes = a.sizes();
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  int64_t m = a_sizes[transpose_result ? 1 : 0];
  int64_t k = a_sizes[transpose_result ? 0 : 1];
  int64_t n = b_sizes[transpose_result ? 0 : 1];

  void* d_a = static_cast<void*>((transpose_result ? b : a).data_ptr());
  void* d_b = static_cast<void*>((transpose_result ? a : b).data_ptr());
  void* d_d = static_cast<void*>(result.data_ptr());

  auto d_scale_a = transpose_result ? scale_a.data_ptr() : scale_a.data_ptr();
  auto d_scale_b = transpose_result ? scale_b.data_ptr() : scale_b.data_ptr();
  auto d_scale_d =
      scale_result.has_value() ? scale_result.value().data_ptr() : nullptr;

  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto stream = at::cuda::getCurrentCUDAStream();

  hipblaslt_ext::GemmPreference gemmPref;
  gemmPref.setMaxWorkspaceBytes(0);
  hipblaslt_ext::Gemm gemm(handle, transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ,
                           hipblas_out_type, hipblas_out_type,
                           HIPBLAS_COMPUTE_32F);

  // TODO(HaiShaw): Add Epilogue usage in cases to support Bias, etc.
  hipblaslt_ext::GemmEpilogue
      epilogue{};  // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT.
                   // (Gemm only)
  hipblaslt_ext::GemmInputs inputs;
  inputs.a = d_a;
  inputs.b = d_b;
  inputs.c = d_d;
  inputs.d = d_d;
  inputs.alpha = &alpha;
  inputs.beta = &beta;
  inputs.scaleA = d_scale_a;
  inputs.scaleB = d_scale_b;
  inputs.scaleD = d_scale_d;

  auto&& problem = gemm.getProblemTypes();
  auto lda = problem.op_a == HIPBLAS_OP_N ? m : k;
  auto ldb = problem.op_b == HIPBLAS_OP_N ? k : n;
  auto ldc = m;
  auto strideA = m * k;
  auto strideB = n * k;
  auto strideC = m * n;

  CHECK_HIPBLASLT_ERROR(gemm.setProblem(m, n, k, 1, lda, ldb, ldc, ldc, strideA,
                                        strideB, strideC, strideC, epilogue,
                                        inputs, problem));

  std::vector<int> algoIndex(1);
  algoIndex[0] = solidx;
  std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
  TORCH_CUDABLAS_CHECK(
      hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo));

  CHECK_HIPBLASLT_ERROR(gemm.initialize(tmpAlgo[0].algo, nullptr));
  CHECK_HIPBLASLT_ERROR(gemm.run(stream));
}
