// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below
// just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by
// default in hip_fp16.h #undef __HIP_NO_HALF_OPERATORS__ #undef
// __HIP_NO_HALF_CONVERSIONS__ #endif

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <ATen/cuda/CUDAEvent.h>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <iostream>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <assert.h>
#include "nvToolsExt.h"

// #include <rocblas/rocblas.h>

// #ifdef USE_ROCM
// #define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 +
// ROCBLAS_VERSION_MINOR) #define USE_GEMM_FLAGS_FP16_ALT_IMPL
// (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242) #endif

// #ifdef __HIP_PLATFORM_HCC__
// 	#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 +
// ROCBLAS_VERSION_MINOR) 	#define USE_GEMM_FLAGS_FP16_ALT_IMPL
// (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242) 	#if USE_GEMM_FLAGS_FP16_ALT_IMPL
// 	  #ifdef ROCM_BACKWARD_PASS_GUARD
// 		flag = at::BackwardPassGuard::is_backward_pass() ?
// rocblas_gemm_flags_fp16_alt_impl : 0; 	  #endif 	#endif #endif

#ifndef CHECK_HIP_ERROR
  #define CHECK_HIP_ERROR(error)                                    \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",             \
              hipGetErrorString(error), error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                           \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
  #define CHECK_HIPBLAS_ERROR(error)                                    \
    if (error != HIPBLAS_STATUS_SUCCESS) {                              \
      fprintf(stderr, "hipBLAS error: '%s'(%d) at %s:%d\n",             \
              hipblasStatusToString(error), error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                               \
    }
#endif

namespace {
/*thread_local*/ cudaStream_t weight_stream;
// BUG: DLM has event and stream on different devices error
// In multi-GPU scenerio, do names defined in this namespace exist on all
// devices? C++ keyword: thread_local <- maybe this can help?
/*thread_local*/ cudaEvent_t event;

// hipBLASLt
hipblasLtHandle_t hipblaslt_handle;
hipblasLtMatmulPreference_t preference;
size_t workspace_size = 2 * 128 * 1024 * 1024;
// uint64_t workspace_size = 0;
void* d_workspace;
int request_solutions = 1;
int returnedAlgoCount = 0;

struct MatMulConfig {
  hipblasOperation_t op_A;
  hipblasOperation_t op_B;
  int M;
  int N;
  int K;
  hipDataType dtype;

  friend auto operator<(const MatMulConfig& left,
                        const MatMulConfig& right) -> bool {
    return std::tie(left.op_A, left.op_B, left.M, left.N, left.K, left.dtype) <
           std::tie(right.op_A, right.op_B, right.M, right.N, right.K,
                    right.dtype);
  }
};

// std::map<std::tuple<int, int, int, int, int, int>,
// std::vector<hipblasLtMatmulHeuristicResult_t>> heuristic_map;
std::map<MatMulConfig, hipblasLtMatmulHeuristicResult_t> heuristic_map;

hipEvent_t start, stop;
int bench_iters{1};
int warmup_iters{1};

bool cout_print = false;

torch::Tensor dTensor;

std::map<at::ScalarType, hipDataType> dtype_map{
    {at::kHalf, HIP_R_16F},
    {at::kBFloat16, HIP_R_16BF},
    {at::kFloat, HIP_R_32F},
    {at::kFloat8_e4m3fnuz, HIP_R_8F_E4M3_FNUZ}};

// std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
}  // namespace

// find all hipblaslt solutions for given gemm problem
std::vector<int64_t> hipblasLtMatmul_findallsols_wrapper(
    hipblasLtHandle_t handle, hipblasOperation_t op_A, hipblasOperation_t op_B,
    int m, int n, int k, const void* alpha, const void* a, int lda,
    const void* b, int ldb, const void* beta, void* c, int ldc,
    const void* bias, hipDataType intype, hipDataType outtype,
    hipStream_t& stream) {
  int flag{0};
  hipblasLtMatrixLayout_t matA, matB, matC;
  hipblasLtMatmulDesc_t matmul;
  if (op_A == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, m, k, lda));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, k, m, lda));
  }
  if (op_B == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, k, n, ldb));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, n, k, ldb));
  }
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, outtype, m, n, ldc));
  CHECK_HIPBLAS_ERROR(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));

  if (bias) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
    auto epilogue = HIPBLASLT_EPILOGUE_BIAS;
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t)));
  }

  // std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(10);
  // CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
  //     handle, matmul, matA, matB, matC, matC,
  //     preference, 10, heuristicResult.data(), &returnedAlgoCount));
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
  CHECK_HIPBLAS_ERROR(hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, op_A, op_B, intype,
      intype, outtype, outtype, HIPBLAS_COMPUTE_32F, heuristicResult));

  std::vector<int64_t> algoIndex;
  int returned_algo_count = heuristicResult.size();
  // for (int i = 0; i < returnedAlgoCount; i++) {
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristicResult[i].algo;
    size_t ret_workspace_size = 0;
    auto status = hipblaslt_ext::matmulIsAlgoSupported(
        handle, matmul, alpha, matA, matB, beta, matC, matC, algo,
        ret_workspace_size);
    if (status == HIPBLAS_STATUS_SUCCESS) {
      if (ret_workspace_size < workspace_size) {
        algoIndex.push_back(hipblaslt_ext::getIndexFromAlgo(algo));
      }
    }
  }

  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  return algoIndex;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * hipBLASLt GEMM call
 */
hipblasStatus_t hipblasLtMatmul_sol_wrapper(
    hipblasLtHandle_t handle, hipblasOperation_t op_A, hipblasOperation_t op_B,
    int m, int n, int k, const void* alpha, const void* a, int lda,
    const void* scaleA, const void* b, int ldb, const void* scaleB,
    const void* beta, void* c, int ldc, const void* scaleC, const void* bias,
    hipDataType intype, hipDataType outtype, hipStream_t& stream,
    int solution_index = -1) {
  // TODO: flag is not supported for hipblasLt yet
  int flag{0};
  // if (dtype == HIPBLAS_R_16F) {
  //  use fp16 alt impl for MI200
  //  https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
  // flag = rocblas_gemm_flags_fp16_alt_impl;
  //}

  // nvtxRangePushA("hipBLASLt variables creation");
  hipblasLtMatrixLayout_t matA, matB, matC;
  hipblasLtMatmulDesc_t matmul;
  if (op_A == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, m, k, lda));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, k, m, lda));
  }
  if (op_B == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, k, n, ldb));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, n, k, ldb));
  }
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, outtype, m, n, ldc));
  CHECK_HIPBLAS_ERROR(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));
  if (scaleA != nullptr) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA,
        sizeof(scaleA)));
  }
  if (scaleB != nullptr) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB,
        sizeof(scaleB)));
  }
  if (scaleC != nullptr) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleC,
        sizeof(scaleC)));
  }
  if (bias) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
    auto epilogue = HIPBLASLT_EPILOGUE_BIAS;
    static_assert(sizeof(epilogue) == sizeof(int32_t));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int32_t)));
  }
  // nvtxRangePop();
  //  if heuristic does not exist in the map, do search and push into the map
  // auto gemm_key { MatMulConfig { op_A, op_B, m, n, k, dtype } };
  // if (heuristic_map.count(gemm_key) <= 0) {
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(1);
  if (solution_index < 0) {
    // nvtxRangePushA("hipblasLtMatmulAlgoGetHeuristic");
    std::cout
        << "Warning! HipbSolId Gemm Fallback Path used for solution index <0"
        << std::endl;
    if (cout_print) {
      std::cout << (op_A == HIPBLAS_OP_N ? "N" : "T")
                << (op_B == HIPBLAS_OP_N ? "N" : "T") << " (" << m << ", " << n
                << ", " << k << "), dtype: " << intype << ", (lda, ldb, ldc): ("
                << lda << ", " << ldb << ", " << ldc << "), " << std::endl;
    }
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matC, preference, request_solutions,
        heuristicResult.data(), &returnedAlgoCount));
    if ((returnedAlgoCount != request_solutions) && cout_print) {
      std::cout << "less solution found! request: " << request_solutions
                << ", found: " << returnedAlgoCount << std::endl;
    }
  } else {
    std::vector<int> algoIndex(1);
    algoIndex[0] = solution_index;
    CHECK_HIPBLAS_ERROR(
        hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResult));
  }

  hipblasStatus_t status = hipblasLtMatmul(
      handle, matmul, alpha, a, matA, b, matB, beta, c, matC, c, matC,
      &heuristicResult[0].algo, d_workspace, workspace_size, stream);

  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));

  return status;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
torch::Tensor hipb_mm(const torch::Tensor& mat1, const torch::Tensor& mat2,
                      const int64_t solution_index,
                      at::optional<torch::Tensor> bias,
                      at::optional<c10::ScalarType> out_dtype,
                      at::optional<torch::Tensor> scale1,
                      at::optional<torch::Tensor> scale2,
                      at::optional<torch::Tensor> scaleOut) {
  auto mat1_strides{mat1.strides()};
  auto mat2_strides{mat2.strides()};
  auto mat1_sizes{mat1.sizes()};
  auto mat2_sizes{mat2.sizes()};

  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected mat1 and mat2 to have the same dtype, but got: ",
              mat1.dtype(), " != ", mat2.dtype());
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0],
              "mat1 dim 1 must match mat2 dim 0");

  auto inDtype{mat1.options().dtype().toScalarType()};
  auto outDtype{out_dtype.has_value() ? out_dtype.value() : inDtype};
  auto options{at::TensorOptions().dtype(outDtype).device(at::kCUDA)};
  auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};

  bool transpose_result = true;
  bool transpose_mat1;
  bool transpose_mat2;
  if ((mat2_strides[0] == 1) &&
      (mat2_strides[1] >= std::max<int64_t>(1, mat2_sizes[0]))) {
    transpose_mat2 = false;
  } else if ((mat2_strides[1] == 1) &&
             (mat2_strides[0] >= std::max<int64_t>(1, mat2_sizes[1]))) {
    transpose_mat2 = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }
  if ((mat1_strides[0] == 1) &&
      (mat1_strides[1] >= std::max<int64_t>(1, mat1_sizes[0]))) {
    transpose_mat1 = false;
  } else if ((mat1_strides[1] == 1) &&
             (mat1_strides[0] >= std::max<int64_t>(1, mat1_sizes[1]))) {
    transpose_mat1 = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }

  if (transpose_result) {
    bool tmp = transpose_mat1;
    transpose_mat1 = !transpose_mat2;
    transpose_mat2 = !tmp;
    mat1_strides = mat2.strides();
    mat2_strides = mat1.strides();
    mat1_sizes = mat2.sizes();
    mat2_sizes = mat1.sizes();
  }

  float one{1.0f};
  float zero{0.0f};
  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
  int64_t mat2_ld = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
  int64_t result_ld = result.stride(transpose_result ? 0 : 1);

  void *d_scale1 = nullptr, *d_scale2 = nullptr, *d_scaleOut = nullptr;
  if (scale1.has_value()) {
    d_scale1 = static_cast<void*>(scale1.value().data_ptr());
  }
  if (scale2.has_value()) {
    d_scale2 = static_cast<void*>(scale2.value().data_ptr());
  }
  if (scaleOut.has_value()) {
    d_scaleOut = static_cast<void*>(scaleOut.value().data_ptr());
  }

  auto hipblasInType = dtype_map.at(inDtype);
  auto hipblasOutType = dtype_map.at(outDtype);

  void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
  void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
  void* ptrC{static_cast<void*>(result.data_ptr())};
  if (transpose_result) std::swap(d_scale1, d_scale2);
  auto current_stream{torch::hip::getCurrentHIPStream().stream()};
  void* bias_ptr =
      bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr;

  CHECK_HIPBLAS_ERROR(hipblasLtMatmul_sol_wrapper(
      hipblaslt_handle, transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one, ptrA,
      mat1_ld, d_scale1, ptrB, mat2_ld, d_scale2, &zero, ptrC, result_ld,
      d_scaleOut, bias_ptr, hipblasInType, hipblasOutType, current_stream,
      solution_index));

  return result;
}

// find all hipblas solutions and return them to python land
std::vector<int64_t> hipb_findallsols(
    const torch::Tensor& mat1, const torch::Tensor& mat2,
    at::optional<torch::Tensor> bias = at::nullopt,
    at::optional<c10::ScalarType> out_dtype = at::nullopt) {
  auto mat1_strides{mat1.strides()};
  auto mat2_strides{mat2.strides()};
  auto mat1_sizes{mat1.sizes()};
  auto mat2_sizes{mat2.sizes()};
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected mat1 and mat2 to have the same dtype, but got: ",
              mat1.dtype(), " != ", mat2.dtype());
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0],
              "mat1 dim 1 must match mat2 dim 0");

  auto inType{mat1.options().dtype().toScalarType()};
  auto outType{out_dtype.has_value() ? out_dtype.value() : inType};

  auto options{at::TensorOptions().dtype(outType).device(at::kCUDA)};
  auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};
  bool transpose_result = true;
  bool transpose_mat1;
  bool transpose_mat2;
  if ((mat2_strides[0] == 1) &&
      (mat2_strides[1] >= std::max<int64_t>(1, mat2_sizes[0]))) {
    transpose_mat2 = false;
  } else if ((mat2_strides[1] == 1) &&
             (mat2_strides[0] >= std::max<int64_t>(1, mat2_sizes[1]))) {
    transpose_mat2 = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }
  if ((mat1_strides[0] == 1) &&
      (mat1_strides[1] >= std::max<int64_t>(1, mat1_sizes[0]))) {
    transpose_mat1 = false;
  } else if ((mat1_strides[1] == 1) &&
             (mat1_strides[0] >= std::max<int64_t>(1, mat1_sizes[1]))) {
    transpose_mat1 = true;
  } else {
    assert(false &&
           "unusual strides detected, may need to clone a contiguous tensor");
  }
  if (transpose_result) {
    bool tmp = transpose_mat1;
    transpose_mat1 = !transpose_mat2;
    transpose_mat2 = !tmp;
    mat1_strides = mat2.strides();
    mat2_strides = mat1.strides();
    mat1_sizes = mat2.sizes();
    mat2_sizes = mat1.sizes();
  }
  float one{1.0f};
  float zero{0.0f};
  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
  int64_t mat2_ld = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
  int64_t result_ld = result.stride(transpose_result ? 0 : 1);
  hipDataType hipblasInType = dtype_map.at(inType);
  hipDataType hipblasOutType = dtype_map.at(outType);

  void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
  void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
  void* ptrC{static_cast<void*>(result.data_ptr())};
  auto current_stream{torch::hip::getCurrentHIPStream().stream()};

  auto bias_ptr =
      bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr;

  return hipblasLtMatmul_findallsols_wrapper(
      hipblaslt_handle, transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one, ptrA,
      mat1_ld, ptrB, mat2_ld, &zero, ptrC, result_ld, bias_ptr, hipblasInType,
      hipblasOutType, current_stream);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////

void hipb_create_extension() {
  // CHECK_HIP_ERROR(hipStreamCreate(&weight_stream));
  // CHECK_HIP_ERROR(hipEventCreateWithFlags(&event, cudaEventDisableTiming));

  // hipBLASLt
  CHECK_HIPBLAS_ERROR(hipblasLtCreate(&hipblaslt_handle));
  CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));

  // CHECK_HIP_ERROR(hipEventCreate(&start));
  // CHECK_HIP_ERROR(hipEventCreate(&stop));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void hipb_destroy_extension() {
  // CHECK_HIP_ERROR(hipStreamDestroy(weight_stream));
  // CHECK_HIP_ERROR(hipEventDestroy(event));

  // hipBLASLt
  CHECK_HIPBLAS_ERROR(hipblasLtDestroy(hipblaslt_handle));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIP_ERROR(hipFree(d_workspace));

  // CHECK_HIP_ERROR(hipEventDestroy(start));
  // CHECK_HIP_ERROR(hipEventDestroy(stop));
}
