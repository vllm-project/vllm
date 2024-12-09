// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below
// just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by
// default in hip_fp16.h #undef __HIP_NO_HALF_OPERATORS__ #undef
// __HIP_NO_HALF_CONVERSIONS__ #endif

#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API

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
// #include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <assert.h>
#include "nvToolsExt.h"

#include <rocblas/rocblas.h>

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
rocblas_handle r_handle;

/*thread_local*/ cudaStream_t weight_stream;
// BUG: DLM has event and stream on different devices error
// In multi-GPU scenerio, do names defined in this namespace exist on all
// devices? C++ keyword: thread_local <- maybe this can help?
/*thread_local*/ cudaEvent_t event;

// hipBLASLt
hipblasLtHandle_t hipblaslt_handle;
hipblasLtMatmulPreference_t preference;
uint64_t workspace_size = 32 * 1024 * 1024;
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
  hipblasDatatype_t dtype;

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

bool cout_print = true;
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * hipBLASLt GEMM call
 */
/*
hipblasStatus_t hipblasLtMatmul_wrapper(
    hipblasLtHandle_t handle,
    hipblasOperation_t op_A,
    hipblasOperation_t op_B,
    int m, int n, int k,
    const void *alpha,
    const void *a,
    int lda,
    const void *b,
    int ldb,
    const void *beta,
    void *c,
    int ldc,
    hipblasDatatype_t dtype,
    hipStream_t &stream)
{
  // TODO: flag is not supported for hipblasLt yet
  int flag { 0 };
  if (dtype == HIPBLAS_R_16F) {
    // use fp16 alt impl for MI200
    //
https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    flag = rocblas_gemm_flags_fp16_alt_impl;
  }

  nvtxRangePushA("hipBLASLt variables creation");
  hipblasLtMatrixLayout_t matA, matB, matC;
  hipblasLtMatmulDesc_t matmul;
  if (op_A == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype, m, k, lda));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype, k, m, lda));
  }
  if (op_B == HIPBLAS_OP_N) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype, k, n, ldb));
  } else {
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype, n, k, ldb));
  }
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype, m, n, ldc));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32,
HIPBLAS_R_32F)); CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute( matmul,
HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));
  nvtxRangePop();

  // if heuristic does not exist in the map, do search and push into the map
  auto gemm_key { MatMulConfig { op_A, op_B, m, n, k, dtype } };
  if (heuristic_map.count(gemm_key) <= 0) {
    nvtxRangePushA("hipblasLtMatmulAlgoGetHeuristic");
    if (cout_print) {
      std::cout << (op_A == HIPBLAS_OP_N ? "N" : "T") << (op_B == HIPBLAS_OP_N ?
"N" : "T")
                << " (" << m << ", " << n << ", " << k << "), dtype: " << dtype
                << ", (lda, ldb, ldc): (" << lda << ", " << ldb << ", " << ldc
<< "), " << std::endl;
    }
    std::vector<hipblasLtMatmulHeuristicResult_t>
heuristicResult(request_solutions);
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
      handle, matmul, matA, matB, matC, matC,
      preference, request_solutions, heuristicResult.data(),
&returnedAlgoCount)); if((returnedAlgoCount != request_solutions) && cout_print)
{ std::cout << "less solution found! request: " << request_solutions
                << ", found: " << returnedAlgoCount << std::endl;
    }

    if (returnedAlgoCount == 1) {
      heuristic_map[gemm_key] = heuristicResult[0];
    } else {
      // benchmark requested solutions and pick best one
      int bestIndex { -1 };
      double bestMs { std::numeric_limits<double>::max() };
      for (int sol { 0 }; sol < returnedAlgoCount; ++sol) {
        // warm up
        for (int iter { 0 }; iter < warmup_iters; ++iter) {
          CHECK_HIPBLAS_ERROR(hipblasLtMatmul(handle, matmul,
              alpha,
              a, matA,
              b, matB,
              beta,
              c, matC,
              c, matC, // In case beta != 0, these runs can overwrite the values
in c
                       // since c and d are the same
                       // TODO: allocates separate d memory for these runs
              &heuristicResult[sol].algo,
              d_workspace, workspace_size,
              stream));
        }
        // performance measuring
        double eventMs;
        CHECK_HIP_ERROR(hipEventRecord(start, stream));
        for (int iter { 0 }; iter < bench_iters; ++iter) {
          CHECK_HIPBLAS_ERROR(hipblasLtMatmul(handle, matmul,
              alpha,
              a, matA,
              b, matB,
              beta,
              c, matC,
              c, matC, // In case beta != 0, these runs can overwrite the values
in c
                       // since c and d are the same
                       // TODO: allocates separate d memory for these runs
              &heuristicResult[sol].algo,
              d_workspace, workspace_size,
              stream));
        }
        CHECK_HIP_ERROR(hipEventRecord(stop, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(stop));
        float temp;
        CHECK_HIP_ERROR(hipEventElapsedTime(&temp, start, stop));
        eventMs = double(temp);
        eventMs /= bench_iters;

        if (cout_print) {
          std::cout << "    Sol " << sol << ": average time per iter " <<
std::to_string(eventMs) << " ms";
        }
        if (bestMs > eventMs) {
          bestMs = eventMs;
          bestIndex = sol;
          if (cout_print) {
            std::cout << " *" << std::endl;
          }
        } else {
          if (cout_print) {
            std::cout << std::endl;
          }
        }
      }
      heuristic_map[gemm_key] = heuristicResult[bestIndex];
    }
    nvtxRangePop();
  }

  hipblasStatus_t status = hipblasLtMatmul(handle, matmul,
      alpha,
      a, matA,
      b, matB,
      beta,
      c, matC,
      c, matC,
      &heuristic_map[gemm_key].algo,
      d_workspace, workspace_size,
      stream);

  nvtxRangePushA("hipBLASLt variables deletion");
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  nvtxRangePop();

  return status;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int64_t> RocFindAllSolIdxBlas(const torch::Tensor& mat1,
                                          const torch::Tensor& mat2) {
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

  auto abcType{mat1.options().dtype()};
  auto options{at::TensorOptions().dtype(abcType).device(at::kCUDA)};
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

  void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
  void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
  void* ptrC{static_cast<void*>(result.data_ptr())};
  auto current_stream{torch::hip::getCurrentHIPStream().stream()};

  rocblas_set_stream(r_handle, current_stream);
  uint32_t flags{0};
  rocblas_datatype abcRtype;
  if (abcType == at::kHalf) {
    abcRtype = rocblas_datatype_f16_r;
  } else if (abcType == at::kBFloat16) {
    abcRtype = rocblas_datatype_bf16_r;
  } else if (abcType == at::kFloat) {
    abcRtype = rocblas_datatype_f32_r;
  } else {
    assert(false && "Wrong datatype!");
  }

#define GEMM_EX_ARGS                                                          \
  r_handle,                                                                   \
      transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,  \
      transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,  \
      m, n, k, &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, &zero, \
      ptrC, abcRtype, result_ld, ptrC, abcRtype, result_ld,                   \
      rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index

  rocblas_int sizeSolve;
  // CHECK_ROCBLAS_ERROR(
  rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL,
                                &sizeSolve);

  // Fill array with list of solutions that match type
  // Note: some of these may be invalid
  std::vector<rocblas_int> solutionsSolve(sizeSolve);
  // CHECK_ROCBLAS_ERROR(
  rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none,
                                solutionsSolve.data(), &sizeSolve);

  std::vector<int64_t> validSolutions;
  for (auto sol : solutionsSolve) {
    auto status = rocblas_gemm_ex(
        r_handle,
        transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
        transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none,
        m, n, k, &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, &zero,
        ptrC, abcRtype, result_ld, ptrC, abcRtype, result_ld,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, sol,
        rocblas_gemm_flags_none);
    if (status == rocblas_status_success) {
      validSolutions.push_back(sol);
    }
  }

  return validSolutions;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
torch::Tensor RocSolIdxBlas(const torch::Tensor& mat1,
                            const torch::Tensor& mat2,
                            const int64_t solution_index = 0) {
  auto mat1_strides{mat1.strides()};
  auto mat2_strides{mat2.strides()};
  auto mat1_sizes{mat1.sizes()};
  auto mat2_sizes{mat2.sizes()};
  // std::cout << " | mat1 info: size: " << mat1_sizes << " stride: " <<
  // mat1_strides << std::endl
  //           << " | mat2 info: size: " << mat2_sizes << " stride: " <<
  //           mat2_strides << std::endl;

  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected mat1 and mat2 to have the same dtype, but got: ",
              mat1.dtype(), " != ", mat2.dtype());
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0],
              "mat1 dim 1 must match mat2 dim 0");

  auto abcType{mat1.options().dtype()};
  auto options{at::TensorOptions().dtype(abcType).device(at::kCUDA)};
  auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};
  // std::cout << " | result info: size: " << result.sizes() << " stride: " <<
  // result.strides() << std::endl;

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
  // std::cout << " | transpose_result: " << (transpose_result ? "true" :
  // "false") << std::endl
  //           << " | transpose_A: " << (transpose_mat1 ? "true" : "false") <<
  //           std::endl
  //           << " | transpose_B: " << (transpose_mat2 ? "true" : "false") <<
  //           std::endl;
  // std::cout << " | A matrix: size: " << mat1_sizes << " stride: " <<
  // mat1_strides << std::endl
  //           << " | B matrix: size: " << mat2_sizes << " stride: " <<
  //           mat2_strides << std::endl;

  float one{1.0f};
  float zero{0.0f};
  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
  int64_t mat2_ld = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
  int64_t result_ld = result.stride(transpose_result ? 0 : 1);
  // std::cout << " | (m, n, k): " << m << ", " << n << ", " << k << std::endl
  //           << " | (lda, ldb, ldc): " << mat1_ld << ", " << mat2_ld << ", "
  //           << result_ld << std::endl;

  /*
  int flag { 0 };
  hipblasDatatype_t hipblasType;
  if (abcType == at::kHalf) {
    hipblasType = HIPBLAS_R_16F;
  } else if (abcType == at::kBFloat16) {
    hipblasType = HIPBLAS_R_16B;
  } else if (abcType == at::kFloat) {
    hipblasType = HIPBLAS_R_32F;
  } else {
    assert(false && "Wrong datatype!");
  }
  */
  void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
  void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
  void* ptrC{static_cast<void*>(result.data_ptr())};
  auto current_stream{torch::hip::getCurrentHIPStream().stream()};
  /*

  CHECK_HIPBLAS_ERROR(hipblasLtMatmul_wrapper(
      hipblaslt_handle,
      transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
      m, n, k,
      &one,
      ptrA, mat1_ld,
      ptrB, mat2_ld,
      &zero,
      ptrC, result_ld,
      hipblasType,
      current_stream));
  */
  rocblas_set_stream(r_handle, current_stream);
  uint32_t flags{0};
  // int32_t solution_index {0};
  rocblas_datatype abcRtype;
  if (abcType == at::kHalf) {
    abcRtype = rocblas_datatype_f16_r;
  } else if (abcType == at::kBFloat16) {
    abcRtype = rocblas_datatype_bf16_r;
  } else if (abcType == at::kFloat) {
    abcRtype = rocblas_datatype_f32_r;
  } else {
    assert(false && "Wrong datatype!");
  }

  // CHECK_ROCBLAS_ERROR(
  rocblas_gemm_ex(
      r_handle,
      transpose_mat1 ? rocblas_operation_transpose : rocblas_operation_none,
      transpose_mat2 ? rocblas_operation_transpose : rocblas_operation_none, m,
      n, k, &one, ptrA, abcRtype, mat1_ld, ptrB, abcRtype, mat2_ld, &zero, ptrC,
      abcRtype, result_ld, ptrC, abcRtype, result_ld, rocblas_datatype_f32_r,
      rocblas_gemm_algo_solution_index, solution_index, flags);
  //);

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void rocb_create_extension() {
  /*
  CHECK_HIP_ERROR(hipStreamCreate(&weight_stream));
  CHECK_HIP_ERROR(hipEventCreateWithFlags(&event, cudaEventDisableTiming));

  // hipBLASLt
  CHECK_HIPBLAS_ERROR(hipblasLtCreate(&hipblaslt_handle));
  CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
  sizeof(workspace_size)));

  CHECK_HIP_ERROR(hipEventCreate(&start));
  CHECK_HIP_ERROR(hipEventCreate(&stop)); */
  rocblas_create_handle(&r_handle);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void rocb_destroy_extension() {
  /*
  CHECK_HIP_ERROR(hipStreamDestroy(weight_stream));
  CHECK_HIP_ERROR(hipEventDestroy(event));

  // hipBLASLt
  CHECK_HIPBLAS_ERROR(hipblasLtDestroy(hipblaslt_handle));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIP_ERROR(hipFree(d_workspace));

  CHECK_HIP_ERROR(hipEventDestroy(start));
  CHECK_HIP_ERROR(hipEventDestroy(stop)); */
  rocblas_destroy_handle(r_handle);
}
