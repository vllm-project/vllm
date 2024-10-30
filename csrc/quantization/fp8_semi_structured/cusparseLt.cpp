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
      const c10::optional<torch::Tensor>& bias_opt) {                        \
    TORCH_CHECK(false,                                                       \
                "Unsupported dtype for compressed matrix multiplication in " \
                "current version of cuSPARSELt.");                           \
  }                                                                          \
                                                                             \
  int64_t cslt_prepare_mm_fp8_semi_structured(                               \
      const torch::Tensor& compressed_A, const torch::Tensor& dense_B) {     \
    TORCH_CHECK(false,                                                       \
                "cusparseLt is not found or "                                \
                "unsupported dtype for compressed matrix in current "        \
                "version of cuSPARSELt.");                                   \
  }                                                                          \
                                                                             \
  torch::Tensor cslt_mm_fp8_semi_structured_prepared(int64_t id) {           \
    TORCH_CHECK(false,                                                       \
                "cusparseLt is not found or "                                \
                "unsupported dtype for compressed matrix in current "        \
                "version of cuSPARSELt.");                                   \
  }                                                                          \
                                                                             \
  void cslt_fp8_semi_structured_destroy(int64_t id) {                        \
    TORCH_CHECK(false,                                                       \
                "cusparseLt is not found or "                                \
                "unsupported dtype for compressed matrix in current "        \
                "version of cuSPARSELt.");                                   \
  }                                                                          \

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
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cusparseLtMatDescriptor_t dense_input_descriptor;
  cusparseLtMatDescriptor_t res_descriptor;
  cusparseLtMatDescriptor_t C_descriptor;

  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;


  void* sparse_mat_ptr;
  void* dense_mat_ptr;

  torch::Device device = torch::kCUDA;
  torch::Dtype out_dtype;
  void* workspace_ptr;

  int m;
  int n;
  int k;
};

cusparseLtHandle_t handle;
bool handle_initialized = false;
using cacheID = int64_t;


std::map<cacheID, cusparseLtEntry> cusparseLt_cache;
}  // namespace cusparseLt
}  // namespace vllm

// vllm::cusparseLt::cusparseLtEntry entry;

torch::Tensor cslt_compress_fp8_semi_structured(const torch::Tensor& input) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "Only float8 e4m3 is supported in vllm:cslt_compress");
  namespace vc = vllm::cusparseLt;
  if (!vc::handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&vc::handle));
    vc::handle_initialized = true;
  }
  // create sparse descriptor, dtype
  auto compression_factor = 9;
  cusparseLtMatDescriptor_t input_descriptor;
  cudaDataType type = CUDA_R_8F_E4M3;
  auto compressed_tensor =
      input.new_empty(input.numel() * compression_factor / 16);

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &vc::handle, &input_descriptor, input.size(0), input.size(1), input.size(1),
      16, type, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));

  size_t compressed_size, compressed_buffer_size;
  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
      &vc::handle, &input_descriptor, &compressed_size, &compressed_buffer_size));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &vc::handle, &input_descriptor, true, CUSPARSE_OPERATION_NON_TRANSPOSE,
      input.data_ptr(), compressed_tensor.data_ptr(), compressedBufferPtr.get(),
      stream));
  return compressed_tensor;
}

vllm::cusparseLt::cacheID cslt_prepare_mm_fp8_semi_structured(const torch::Tensor& compressed_A,
                                            const torch::Tensor& dense_B, 
                                            const c10::optional<torch::Tensor>& bias_opt) {
  TORCH_CHECK(compressed_A.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "Only float8 e4m3 is supported in vllm:cslt_compress");
  namespace vc = vllm::cusparseLt;
  if (!vc::handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&vc::handle));
    vc::handle_initialized = true;
  }
  vc::cacheID id;
  if (vc::cusparseLt_cache.empty()) {
    id = 0;
  } else {
    id = vc::cusparseLt_cache.rbegin()->first + 1;
  }

  vc::cusparseLtEntry& entry = vc::cusparseLt_cache[id];

  float alpha = 1.0;
  float beta = 0.0;
  cudaDataType input_type = CUDA_R_8F_E4M3;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
  auto compression_factor = 9;
  auto out_dtype = dense_B.scalar_type();
  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor) / k;

  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cusparseLtMatDescriptor_t dense_input_descriptor;
  cusparseLtMatDescriptor_t res_descriptor;
  cusparseLtMatDescriptor_t C_descriptor;

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
  // initialize sparse descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &vc::handle, &sparse_input_descriptor, m, k, k, 16, input_type,
      CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));

  // initialize dense descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n, (dense_B.is_contiguous()) ? n : k,
      (dense_B.is_contiguous()) ? n : k, 16, input_type, CUSPARSE_ORDER_ROW));

  // initialize result descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &res_descriptor, m, n, n, 16,
      output_type,
      CUSPARSE_ORDER_ROW));
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &C_descriptor, m, n, n, 16, C_type,
      CUSPARSE_ORDER_ROW));

  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &vc::handle, &entry.matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor, &dense_input_descriptor,
      &C_descriptor, &res_descriptor, compute_type));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &vc::handle, &alg_sel, &entry.matmul,
      CUSPARSELT_MATMUL_ALG_DEFAULT));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanInit(
      &vc::handle, &plan, &entry.matmul, &alg_sel));
  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&vc::handle, &plan, &workspace_size));
  AT_CUDA_CHECK(cudaMalloc((void**) &entry.workspace_ptr, workspace_size));

  entry.device = dense_B.device();
  entry.out_dtype = out_dtype;
  entry.m = m;
  entry.n = n;
  entry.k = k;
  entry.sparse_mat_ptr = compressed_A.data_ptr();
  entry.dense_mat_ptr = dense_B.data_ptr();
  entry.plan = plan;
  entry.sparse_input_descriptor = sparse_input_descriptor;
  entry.dense_input_descriptor = dense_input_descriptor;
  entry.C_descriptor = C_descriptor;
  entry.res_descriptor = res_descriptor;

  return id;
}

torch::Tensor cslt_mm_fp8_semi_structured_prepared(vllm::cusparseLt::cacheID id) {
  namespace vc = vllm::cusparseLt;
  TORCH_CHECK(vc::handle_initialized,
              "Call of matmul with unintialized matmul");
  if (vc::cusparseLt_cache.count(id) == 0) {
    TORCH_CHECK(false, "cusparse matmul Id is not found");
  }
  const auto& entry = vc::cusparseLt_cache[id];

  auto res_tensor_options =
      c10::TensorOptions().dtype(entry.out_dtype).device(entry.device);
  at::Tensor res = at::empty({entry.m, entry.n}, res_tensor_options);
  float alpha = 1.0;
  float beta = 0.0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmul(&vc::handle, &entry.plan, &alpha, entry.sparse_mat_ptr,
                       entry.dense_mat_ptr, &beta, res.data_ptr(),
                       res.data_ptr(), entry.workspace_ptr, &stream, 1));

  return res;
}

void cslt_fp8_semi_structured_destroy(vllm::cusparseLt::cacheID id) {
  namespace vc = vllm::cusparseLt;
  TORCH_CHECK(vc::handle_initialized,
              "Call of destroy cusparseId with unintialized cusparseLt");
  if (vc::cusparseLt_cache.count(id) == 0) {
    TORCH_CHECK(false, "cusparse matmul Id is not found");
  }
  auto& entry = vc::cusparseLt_cache[id];

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&entry.sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&entry.dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&entry.C_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&entry.res_descriptor));
  // Destroy plan
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&entry.plan));
  AT_CUDA_CHECK(cudaFree(entry.workspace_ptr));
  vc::cusparseLt_cache.erase(id);
}

torch::Tensor cslt_mm_fp8_semi_structured(
    const torch::Tensor& compressed_A, const torch::Tensor& dense_B,
    const c10::optional<torch::Tensor>& alpha_opt,
    const c10::optional<torch::Tensor>& bias_opt, bool transpose_result) {
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

  int tensor_alpha_mode = 0;
  float alpha = 1.0;
  float beta = 0.0;
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
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &res_descriptor, m, n, n, 16,
      output_type,
      CUSPARSE_ORDER_ROW));

  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &vc::handle, &C_descriptor, m, n, n, 16, C_type,
      CUSPARSE_ORDER_ROW));

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

  const auto alpha_tensor =
      alpha_opt.has_value() ? *alpha_opt : torch::Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      tensor_alpha_mode = 1;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle, &matmul, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode, sizeof(tensor_alpha_mode)));
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

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

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
      &vc::handle, &plan, &alpha, compressed_A.data_ptr(), dense_B.data_ptr(),
      &beta, res.data_ptr(), res.data_ptr(), workspace_ptr.get(), &stream, 1));

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
