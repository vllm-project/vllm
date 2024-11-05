#include <cusparse.h>
#include <torch/all.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#define STUB_FUNC_IMPL()                                                     \
  torch::Tensor cslt_compress_fp8_semi_structured(                           \
      const torch::Tensor& input) {                                          \
  torch::Tensor cslt_compress_fp8_semi_structured(                           \
      const torch::Tensor& input) {                                          \
    TORCH_CHECK(false,                                                       \
                "cusparseLt is not found or "                                \
                "unsupported dtype for compressed matrix in current "        \
                "version of cuSPARSELt.");                                   \
  }                                                                          \
                                                                             \
  torch::Tensor cslt_mm_fp8_semi_structured(                                 \
      const torch::Tensor& compressed_A, const torch::Tensor& dense_B,       \
      const c10::optional<torch::Tensor>& scale_opt,                         \
      const c10::optional<torch::Tensor>& bias_opt) {                        \
    TORCH_CHECK(false,                                                       \
                "Unsupported dtype for compressed matrix multiplication in " \
                "current version of cuSPARSELt.");                           \
  }                                                                          \
  torch::Tensor cslt_mm_fp8_semi_structured2(                                \
      const torch::Tensor& compressed_A, const torch::Tensor& dense_B,       \
      const c10::optional<torch::Tensor>& scale_opt,                         \
      const c10::optional<torch::Tensor>& bias_opt) {                        \
    TORCH_CHECK(false,                                                       \
                "Unsupported dtype for compressed matrix multiplication in " \
                "current version of cuSPARSELt.");                           \
  }

#if defined(VLLM_CUSPARSELT_ENABLED)

  #include <cusparseLt.h>

  #if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 600

    #define CUDASPARSE_CHECK(EXPR)                                 \
      do {                                                         \
        cusparseStatus_t __err = EXPR;                             \
        TORCH_CHECK(__err == CUSPARSE_STATUS_SUCCESS,              \
                    "CUDA error: ", cusparseGetErrorString(__err), \
                    " when calling `" #EXPR "`");                  \
      } while (0)

namespace vllm {
namespace cusparseLt {

struct cusparseLtEntry {
  cusparseLtMatDescriptor_t* sparse_input_descriptor_p;
  cusparseLtMatDescriptor_t* dense_input_descriptor_p;
  cusparseLtMatDescriptor_t* res_descriptor_p;
  cusparseLtMatDescriptor_t* C_descriptor_p;

  cusparseLtMatmulDescriptor_t* matmul_p;
  cusparseLtMatmulPlan_t* plan_p;
  cusparseLtMatmulAlgSelection_t* alg_sel_p;

  void* workspace_ptr;

  ~cusparseLtEntry() {
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(sparse_input_descriptor_p));
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(dense_input_descriptor_p));
    TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(C_descriptor_p));
    TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(res_descriptor_p));
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(plan_p));

    // Destructor is called after the cuda cleanup so double free is done here.
    // AT_CUDA_CHECK(cudaFree(workspace_ptr));
    delete sparse_input_descriptor_p;
    delete dense_input_descriptor_p;
    delete res_descriptor_p;
    delete C_descriptor_p;
    delete plan_p;
    delete alg_sel_p;
    delete matmul_p;
  }
};

cusparseLtHandle_t handle;
bool handle_initialized = false;
using cacheID = std::tuple<int64_t, int64_t, int64_t, at::ScalarType>;

std::map<cacheID, cusparseLtEntry> cusparseLt_cache;

void prepare_mm_semi_structured(const cacheID& tuple_id,
                                at::ScalarType out_dtype,
                                bool is_B_contiguous) {
  auto m = std::get<0>(tuple_id);
  auto k = std::get<1>(tuple_id);
  auto n = std::get<2>(tuple_id);
  at::ScalarType input_dtype = std::get<3>(tuple_id);
  auto& entry = cusparseLt_cache[tuple_id];

  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type;

  switch (input_dtype) {
    case at::ScalarType::Char:
      input_type = CUDA_R_8I;
      output_type = CUDA_R_8I;
      C_type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
      break;
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::Float8_e4m3fn:
      input_type = CUDA_R_8F_E4M3;
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported dtype for cuSPARSELt compressed matrix multiplication.");
      break;
  }

  // cudaDataType input_type = CUDA_R_8F_E4M3;
  // cudaDataType output_type;
  // cudaDataType C_type;
  // cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
  // switch (out_dtype) {
  //   case at::ScalarType::Float8_e4m3fn:
  //     output_type = CUDA_R_8F_E4M3;
  //     C_type = CUDA_R_16F;
  //     break;
  //   case at::ScalarType::Half:
  //     output_type = CUDA_R_16F;
  //     C_type = CUDA_R_16F;
  //     break;
  //   case at::ScalarType::BFloat16:
  //     output_type = CUDA_R_16BF;
  //     C_type = CUDA_R_16BF;
  //     break;
  //   case at::ScalarType::Float:
  //     output_type = CUDA_R_32F;
  //     C_type = CUDA_R_32F;
  //     break;
  //   default:
  //     TORCH_CHECK(false,
  //                 "Unsupported out_dtype passed, must be one of {fp16, bf16,
  //                 " "float32} for fp8 inputs");
  //     break;
  // }
  entry.sparse_input_descriptor_p = new cusparseLtMatDescriptor_t();
  entry.dense_input_descriptor_p = new cusparseLtMatDescriptor_t();
  entry.res_descriptor_p = new cusparseLtMatDescriptor_t();
  entry.C_descriptor_p = new cusparseLtMatDescriptor_t();

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle, entry.sparse_input_descriptor_p, m, k, k, 16, input_type,
      CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));

  // initialize dense descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle, entry.dense_input_descriptor_p, (is_B_contiguous) ? k : n,
      (is_B_contiguous) ? n : k, (is_B_contiguous) ? n : k, 16, input_type,
      CUSPARSE_ORDER_ROW));

  // initialize result descriptor
  TORCH_CUDASPARSE_CHECK(
      cusparseLtDenseDescriptorInit(&handle, entry.res_descriptor_p, m, n, n,
                                    16, output_type, CUSPARSE_ORDER_ROW));
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle, entry.C_descriptor_p, m, n, n, 16, C_type, CUSPARSE_ORDER_ROW));

  entry.matmul_p = new cusparseLtMatmulDescriptor_t();
  entry.plan_p = new cusparseLtMatmulPlan_t();
  entry.alg_sel_p = new cusparseLtMatmulAlgSelection_t();

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle, entry.matmul_p, CUSPARSE_OPERATION_NON_TRANSPOSE,
      (is_B_contiguous) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                        : CUSPARSE_OPERATION_TRANSPOSE,
      entry.sparse_input_descriptor_p, entry.dense_input_descriptor_p,
      entry.C_descriptor_p, entry.res_descriptor_p, compute_type));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, entry.alg_sel_p, entry.matmul_p, CUSPARSELT_MATMUL_ALG_DEFAULT));
  int num_search_iters = 5;
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
      &handle, entry.alg_sel_p, CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
      &num_search_iters, sizeof(num_search_iters)));

  // TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanInit(
  //     &handle, &plan, &entry.matmul, &alg_sel));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanInit(
      &handle, entry.plan_p, entry.matmul_p, entry.alg_sel_p));

  size_t workspace_size;
  // TORCH_CUDASPARSE_CHECK(
  //     cusparseLtMatmulGetWorkspace(&handle, &global_plan, &workspace_size));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, entry.plan_p, &workspace_size));
  AT_CUDA_CHECK(cudaMalloc((void**)&entry.workspace_ptr, workspace_size));
}

}  // namespace cusparseLt
}  // namespace vllm

torch::Tensor cslt_compress_fp8_semi_structured(const torch::Tensor& input) {
  namespace vc = vllm::cusparseLt;
  if (!vc::handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&vc::handle));
    vc::handle_initialized = true;
  }

  cudaDataType type;
  auto compression_factor = 9;
  cusparseLtMatDescriptor_t input_descriptor;

  switch (input.scalar_type()) {
    case at::ScalarType::Char:
      type = CUDA_R_8I;
      compression_factor = 10;
      break;
    case at::ScalarType::Half:
      type = CUDA_R_16F;
      break;
    case at::ScalarType::BFloat16:
      type = CUDA_R_16BF;
      break;
    case at::ScalarType::Float:
      type = CUDA_R_32F;
      break;
    case at::ScalarType::Float8_e4m3fn:
      type = CUDA_R_8F_E4M3;
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt compressed matrix");
      break;
  }

  auto compressed_tensor =
      input.new_empty(input.numel() * compression_factor / 16);

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &vc::handle, &input_descriptor, input.size(0), input.size(1),
      input.size(1), 16, type, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  size_t compressed_size, compressed_buffer_size;
  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
      &vc::handle, &input_descriptor, &compressed_size,
      &compressed_buffer_size));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &vc::handle, &input_descriptor, true, CUSPARSE_OPERATION_NON_TRANSPOSE,
      input.data_ptr(), compressed_tensor.data_ptr(), compressedBufferPtr.get(),
      stream));
  return compressed_tensor;
}

torch::Tensor cslt_mm_fp8_semi_structured(
    const torch::Tensor& compressed_A, const torch::Tensor& dense_B,
    const c10::optional<double>& alpha_opt,
    const c10::optional<torch::Tensor>& bias_opt) {
  namespace vc = vllm::cusparseLt;
  if (!vc::handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&vc::handle));
    vc::handle_initialized = true;
  }

  auto input_dtype = compressed_A.scalar_type();
  auto out_dtype = dense_B.scalar_type();
  auto compression_factor = (input_dtype == at::ScalarType::Char) ? 10 : 9;

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor) / k;

  vc::cacheID tuple_id = std::make_tuple(m, k, n, input_dtype);
  bool found = vc::cusparseLt_cache.count(tuple_id);
  if (not found) {
    vc::prepare_mm_semi_structured(tuple_id, out_dtype,
                                   dense_B.is_contiguous());
  }
  auto& entry = vc::cusparseLt_cache[tuple_id];

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &vc::handle, entry.matmul_p, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias,
        sizeof(dBias)));
  }

  // float alpha = 1.0;
  float alpha = alpha_opt.has_value() ? static_cast<float>(*alpha_opt) : 1.0;
  float beta = 0.0;
  auto alpha_ptr = &alpha;

  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = at::empty({m, n}, res_tensor_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (found) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &vc::handle, entry.plan_p, alpha_ptr, compressed_A.data_ptr(),
        dense_B.data_ptr(), &beta, res.data_ptr(), res.data_ptr(),
        entry.workspace_ptr, &stream, 1));
  } else {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
        &vc::handle, entry.plan_p, alpha_ptr, compressed_A.data_ptr(),
        dense_B.data_ptr(), &beta, res.data_ptr(), res.data_ptr(),
        entry.workspace_ptr, &stream, 1));
  }
  return res;
}

torch::Tensor cslt_mm_fp8_semi_structured2(
    const torch::Tensor& compressed_A, const torch::Tensor& dense_B,
    const c10::optional<double>& alpha_opt,
    const c10::optional<torch::Tensor>& bias_opt) {
  TORCH_CHECK(compressed_A.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "Only float8 e4m3 is supported in vllm:cslt_compress");
  namespace vc = vllm::cusparseLt;
  if (!vc::handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&vc::handle));
    vc::handle_initialized = true;
  }

  // cusparseLt data structures
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  cudaDataType input_type = CUDA_R_8F_E4M3;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
  auto compression_factor = 9;
  auto out_dtype = dense_B.scalar_type();

  switch (out_dtype) {
    case at::ScalarType::Float8_e4m3fn:
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
      break;
    case at::ScalarType::Half:
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      break;
    case at::ScalarType::BFloat16:
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      break;
    case at::ScalarType::Float:
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported out_dtype passed, must be one of {fp16, bf16, "
                  "float32} for fp8 inputs");
      break;
  }

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor) / k;

  // initialize sparse descriptor
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &vc::handle, &sparse_input_descriptor, m, k, k, 16, input_type,
      CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));

  // initialize dense input descriptor
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &dense_input_descriptor, (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k, (dense_B.is_contiguous()) ? n : k, 16,
      input_type, CUSPARSE_ORDER_ROW));

  // create result tensor
  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = at::empty({m, n}, res_tensor_options);

  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtDenseDescriptorInit(&vc::handle, &res_descriptor, m, n, n, 16,
                                    output_type, CUSPARSE_ORDER_ROW));

  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &C_descriptor, m, n, n, 16, C_type, CUSPARSE_ORDER_ROW));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &vc::handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor, &dense_input_descriptor, &C_descriptor,
      &res_descriptor, compute_type));

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &vc::handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias,
        sizeof(dBias)));
  }

  float beta = 0.0;
  const float alpha =
      alpha_opt.has_value() ? static_cast<float>(*alpha_opt) : 1.0;
  auto alpha_ptr = &alpha;

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &vc::handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&vc::handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&vc::handle, &plan, &workspace_size));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto workspace_ptr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmul(&vc::handle, &plan, alpha_ptr, compressed_A.data_ptr(),
                       dense_B.data_ptr(), &beta, res.data_ptr(),
                       res.data_ptr(), workspace_ptr.get(), &stream, 1));

  // Destroy descriptors
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&C_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  // Destroy plan
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));
  return res;
}
  #else

STUB_FUNC_IMPL()

  #endif

#else

STUB_FUNC_IMPL()

#endif
