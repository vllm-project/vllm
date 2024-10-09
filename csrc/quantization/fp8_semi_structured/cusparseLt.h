#include <cusparse.h>
#include <torch/all.h>

#include <cusparseLt.h>
#include <cuda_fp8.h>

namespace vllm {


cusparseLtHandle_t handle;
bool handle_initialized = false;
#if not (defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602)

torch::Tensor cslt_compress_fp8_semi_structured(const torch::Tensor& input) {
    
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float8_e4m3fn, "Only float8 e4m3 is supported in vllm:cslt_compress")
    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }
    // create sparse descriptor, dtype
    auto compression_factor = 9;
    cusparseLtMatDescriptor_t input_descriptor;
    cudaDataType type = CUDA_R_8F_E4M3;
    auto compressed_tensor = input.new_empty(input.numel() * compression_factor / 16);

    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &input_descriptor,
        input.size(0),
        input.size(1),
        input.size(1),
        16,
        type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    size_t compressed_size, compressed_buffer_size;
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
        &handle,
        &input_descriptor,
        &compressed_size,
        &compressed_buffer_size));

    auto& allocator = ::c10::cuda::CUDACachingAllocator::get();
    auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
        &handle,
        &input_descriptor,
        true,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        input.data_ptr(),
        compressed_tensor.data_ptr(),
        compressedBufferPtr.get(),
        stream));
    return compressed_tensor;
}

torch::Tensor cslt_mm_fp8_semi_structured(
    const torch::Tensor& compressed_A,
    const torch::Tensor& dense_B,
    const c10::optional<torch::Tensor>& bias_opt,
    bool transpose_result
)
{
    TORCH_CHECK(compressed_A.scalar_type() == at::ScalarType::Float8_e4m3fn, "Only float8 e4m3 is supported in vllm:cslt_compress");
    
    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }
    // cusparseLt data structures
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulAlgSelection_t alg_sel;
    
    float alpha = 1.0;
    float beta = 0.0;
    cudaDataType input_type = CUDA_R_8F_E4M3;
    cudaDataType output_type;
    cudaDataType C_type;
    cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
    auto compression_factor = 9;
    ScalarType out_dtype = dense_B.scalar_type();

    switch (out_dtype)
    {
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
            TORCH_CHECK(false, "Unsupported out_dtype passed, must be one of {fp16, bf16, float32} for fp8 inputs");
            break;
    }

    int64_t k = dense_B.size(0);
    int64_t n = dense_B.size(1);
    int64_t m = (compressed_A.numel() * 16 / compression_factor  ) / k;


    //initialize sparse descriptor
    cusparseLtMatDescriptor_t sparse_input_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &sparse_input_descriptor,
        m,
        k,
        k,
        16,
        input_type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    // initialize dense input descriptor
    cusparseLtMatDescriptor_t dense_input_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &dense_input_descriptor,
        (dense_B.is_contiguous()) ? k : n,
        (dense_B.is_contiguous()) ? n : k,
        (dense_B.is_contiguous()) ? n : k,
        16,
        input_type,
        CUSPARSE_ORDER_ROW));
    
    // create result tensor
    auto res_tensor_options = c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
    at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                        : at::empty({m, n}, res_tensor_options);

    cusparseLtMatDescriptor_t res_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &res_descriptor,
        m,
        n,
        (transpose_result) ? m: n,
        16,
        output_type,
        (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

    cusparseLtMatDescriptor_t C_descriptor;
    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &C_descriptor,
        m,
        n,
        (transpose_result) ? m: n,
        16,
        C_type,
        (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &C_descriptor,
      &res_descriptor,
      compute_type));
    
    // set bias pointer for matmul, need to assign to get location
    if (bias_opt.has_value()) {
        auto& bias = bias_opt.value();
        void* dBias = bias.data_ptr();
        TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
            &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));
    }

    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
                                    CUSPARSELT_MATMUL_ALG_DEFAULT);
    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
    size_t workspace_size;
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));


    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto workspacePtr = allocator.allocate(workspace_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &plan,
        &alpha,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));

    // Destroy descriptors
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
    TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
    // Destroy plan
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));
    return res;
}
#else

torch::Tensor cslt_compress_fp8_semi_structured(const torch::Tensor& input) {
    TORCH_CHECK(false, "Unsupported dtype for compressed matrix in current version of cuSPARSELt.");
}

at::Tensor cslt_mm_fp8_semi_structured(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    bool transpose_result,
)
{
#if not (defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602)
    TORCH_CHECK(false, "Unsupported dtype for compressed matrix multiplication in current version of cuSPARSELt.");
#endif
}

#endif


} // namespace vllm