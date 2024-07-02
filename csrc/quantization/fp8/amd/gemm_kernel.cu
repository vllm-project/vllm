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

static void* workspace = nullptr;
static size_t workspace_size;

// Copied from
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/GemmHipblaslt.h
static size_t get_hipblaslt_workspace_size() {
  static const char* env = getenv("HIPBLASLT_WORKSPACE_SIZE");
  // 256MB is max workspace size allowed for hipblaslt
  // hipblaslt-bench uses 32MB
  // recommendation from hipblaslt author was 76MB
  size_t workspace_size = 32 * 1024;  // going with 32MB
  if (env) {
    try {
      workspace_size = std::stoi(env);
    } catch (std::invalid_argument const& e) {
      TORCH_WARN("invalid HIPBLASLT_WORKSPACE_SIZE,",
                 " using default workspace size of ", workspace_size, " KiB.");
    } catch (std::out_of_range const& e) {
      TORCH_WARN("HIPBLASLT_WORKSPACE_SIZE out of range,",
                 " using default workspace size of ", workspace_size, " KiB.");
    }
  }
  return workspace_size * 1024;
}

void create_workspace() {
  workspace_size = get_hipblaslt_workspace_size();
  if (workspace_size > 0)
    CHECK_HIP_ERROR(hipMalloc(&workspace, workspace_size));
}

torch::Tensor fp8_gemm(torch::Tensor& a, torch::Tensor& b,
                       torch::Tensor& scaleA, torch::Tensor& scaleB,
                       torch::Tensor& scaleD, int algo_idx) {
  auto a_strides{a.strides()};
  auto b_strides{b.strides()};
  auto a_sizes{a.sizes()};
  auto b_sizes{b.sizes()};

  // CHECK_INPUT(a);
  // CHECK_INPUT(b);
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fnuz &&
                  b.dtype() == torch::kFloat8_e4m3fnuz,
              "The input tensors should be in fp8.");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-D.");
  TORCH_CHECK(a_sizes[1] == b_sizes[0], "a dim 1 must match b dim 0.");

  auto options{
      at::TensorOptions().dtype(torch::kFloat8_e4m3fnuz).device(at::kCUDA)};
  auto result{torch::empty({a_sizes[0], b_sizes[1]}, options)};

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

  // void *d_scaleA, *d_scaleB, *d_workspace;
  // CHECK_HIP_ERROR(hipMalloc(&d_scaleA, sizeof(float)));
  // CHECK_HIP_ERROR(hipMalloc(&d_scaleB, sizeof(float)));
  // CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
  // CHECK_HIP_ERROR(hipMemcpy(d_scaleA, &(transpose_result ? scaleB : scaleA),
  // sizeof(float), hipMemcpyHostToDevice)); CHECK_HIP_ERROR(hipMemcpy(d_scaleB,
  // &(transpose_result ? scaleA : scaleB), sizeof(float),
  // hipMemcpyHostToDevice));
  auto d_scaleA = transpose_result ? scaleB.data_ptr() : scaleA.data_ptr();
  auto d_scaleB = transpose_result ? scaleA.data_ptr() : scaleB.data_ptr();
  auto d_scaleD = scaleD.data_ptr();

  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto stream = at::cuda::getCurrentCUDAStream();

  hipblaslt_ext::GemmPreference gemmPref;
  gemmPref.setMaxWorkspaceBytes(workspace_size);
  hipblaslt_ext::Gemm gemm(handle, transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ,
                           HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ,
                           HIPBLAS_COMPUTE_32F);

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
  inputs.scaleA = d_scaleA;
  inputs.scaleB = d_scaleB;
  inputs.scaleD = d_scaleD;

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

  if (algo_idx == 0) {
    constexpr int request_solutions = 1024;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    heuristicResult.reserve(request_solutions);
    CHECK_HIPBLASLT_ERROR(
        gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));
    static size_t solSize = 0;
    if (heuristicResult.size() != solSize) {
      std::cout << "fp8 sols: " << heuristicResult.size() << "\n";
      solSize = heuristicResult.size();
      for (auto& res : heuristicResult) {
        auto idx = hipblaslt_ext::getIndexFromAlgo(res.algo);
        std::cout << idx << "\n";
      }
    }
    TORCH_CHECK(!heuristicResult.empty(), "No valid solution found!");
    algo_idx = hipblaslt_ext::getIndexFromAlgo(heuristicResult[0].algo);
  }
  std::vector<int> algoIndex(1);
  algoIndex[0] = algo_idx;
  std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
  TORCH_CUDABLAS_CHECK(
      hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo));

  CHECK_HIPBLASLT_ERROR(gemm.initialize(tmpAlgo[0].algo, workspace));
  CHECK_HIPBLASLT_ERROR(gemm.run(stream));

  // hipFree(d_scaleA);
  // hipFree(d_scaleB);

  return result;
}

torch::Tensor fp8_gemm_16(torch::Tensor& a, torch::Tensor& b,
                          torch::Tensor& scaleA, torch::Tensor& scaleB,
                          int algo_idx) {
  auto a_strides{a.strides()};
  auto b_strides{b.strides()};
  auto a_sizes{a.sizes()};
  auto b_sizes{b.sizes()};

  // CHECK_INPUT(a);
  // CHECK_INPUT(b);
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fnuz &&
                  b.dtype() == torch::kFloat8_e4m3fnuz,
              "The input tensors should be in fp8.");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-D.");
  TORCH_CHECK(a_sizes[1] == b_sizes[0], "a dim 1 must match b dim 0.");

  auto options{at::TensorOptions().dtype(torch::kFloat16).device(at::kCUDA)};
  auto result{torch::empty({a_sizes[0], b_sizes[1]}, options)};

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

  // void *d_scaleA, *d_scaleB, *d_workspace;
  // CHECK_HIP_ERROR(hipMalloc(&d_scaleA, sizeof(float)));
  // CHECK_HIP_ERROR(hipMalloc(&d_scaleB, sizeof(float)));
  // CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
  // CHECK_HIP_ERROR(hipMemcpy(d_scaleA, &(transpose_result ? scaleB : scaleA),
  // sizeof(float), hipMemcpyHostToDevice)); CHECK_HIP_ERROR(hipMemcpy(d_scaleB,
  // &(transpose_result ? scaleA : scaleB), sizeof(float),
  // hipMemcpyHostToDevice));
  auto d_scaleA = transpose_result ? scaleB.data_ptr() : scaleA.data_ptr();
  auto d_scaleB = transpose_result ? scaleA.data_ptr() : scaleB.data_ptr();

  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto stream = at::cuda::getCurrentCUDAStream();

  hipblaslt_ext::GemmPreference gemmPref;
  gemmPref.setMaxWorkspaceBytes(workspace_size);
  hipblaslt_ext::Gemm gemm(handle, transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16F,
                           HIP_R_16F, HIPBLAS_COMPUTE_32F);

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
  inputs.scaleA = d_scaleA;
  inputs.scaleB = d_scaleB;

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
  if (algo_idx == 0) {
    constexpr int request_solutions = 1024;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    heuristicResult.reserve(request_solutions);
    CHECK_HIPBLASLT_ERROR(
        gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));
    static size_t solSize = 0;
    if (heuristicResult.size() != solSize) {
      std::cout << "fp16 sols: " << heuristicResult.size() << "\n";
      solSize = heuristicResult.size();
      for (auto& res : heuristicResult) {
        auto idx = hipblaslt_ext::getIndexFromAlgo(res.algo);
        std::cout << idx << "\n";
      }
    }
    algo_idx = hipblaslt_ext::getIndexFromAlgo(heuristicResult[0].algo);
    TORCH_CHECK(!heuristicResult.empty(), "No valid solution found!");
  }
  std::vector<int> algoIndex(1);
  algoIndex[0] = algo_idx;
  std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
  TORCH_CUDABLAS_CHECK(
      hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo));

  CHECK_HIPBLASLT_ERROR(gemm.initialize(tmpAlgo[0].algo, workspace));
  CHECK_HIPBLASLT_ERROR(gemm.run(stream));

  // hipFree(d_scaleA);
  // hipFree(d_scaleB);

  return result;
}