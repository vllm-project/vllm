

#include "moe_gemm_kernels.h"

#include "c10/cuda/CUDAStream.h"

#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_preprocessors.h"

#include <torch/all.h>

using torch::Tensor;

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)

namespace phi_c
{
    template <typename T>
    T* get_ptr(Tensor t)
    {
        return (T*)t.data_ptr();
    }

    template <typename T, typename WeightType>
    Tensor grouped_gemm_helper(Tensor activations,
                                    Tensor weights,
                                    Tensor weight_scales,
                                    Tensor total_rows_before_expert,
                                    Tensor res,
                                    int activation_type,
                                    int config_id
                                    )
    {
        const at::ScalarType _st = activations.scalar_type();
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const int num_rows = activations.size(0);
        const int64_t gemm_k = activations.size(1);
        const int64_t gemm_n = weights.size(-1);
        const int64_t experts = weights.size(0);

        assert(gemm_k >= 128 / cutlass::sizeof_bits<WeightType>::value);
        assert(gemm_k% (128 / cutlass::sizeof_bits<WeightType>::value) == 0);
        assert(gemm_n % (128 / cutlass::sizeof_bits<WeightType>::value) == 0);

        assert(activations.size(1) == weights.size(1));
        assert(experts == weight_scales.size(0));
        assert(total_rows_before_expert.dtype() == torch::kInt64);

        // auto res = torch::zeros({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        // auto res = torch::empty({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        bool fused_moe = false;

        if(activation_type == (int)tensorrt_llm::ActivationType::Identity)
        {
            assert(activations.scalar_type() == res.scalar_type());
            assert(activations.device() == res.device());
            assert(activations.size(0) == res.size(0));
            assert(activations.size(1) == res.size(1));
        }

        T *act_ptr = get_ptr<T>(activations);
        WeightType *wt_ptr = get_ptr<WeightType>(weights);
        T *weight_scale_ptr = get_ptr<T>(weight_scales);
        T *res_ptr = get_ptr<T>(res);
        int64_t *total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

        tensorrt_llm::MoeGemmRunner<T, WeightType> moe_gemm_runner;
        auto configs = moe_gemm_runner.getConfigs();
        assert(configs.size() > 1);
        // estimate_best_config_from_occupancies()
        moe_gemm_runner.setBestConfig(configs[config_id]);
        moe_gemm_runner.moeGemmBiasAct(
            act_ptr,
            wt_ptr,
            weight_scale_ptr,
            nullptr,
            res_ptr,
            total_rows_before_expert_ptr,
            tensorrt_llm::HopperGroupedGemmInput{},
            (int64_t)num_rows,
            (int64_t)gemm_n,
            (int64_t)gemm_k,
            experts,
            (tensorrt_llm::ActivationType)activation_type,
            fused_moe,
            stream);

        return res;
    }

    void grouped_gemm(Tensor activations,
                    Tensor weights,
                    Tensor weight_scales,
                    Tensor total_rows_before_expert,
                    Tensor out,
                    int64_t activation_type,
                    int64_t config_id = 0)
    {
        const at::ScalarType _st = activations.scalar_type();
        CHECK_INPUT(activations, _st);
        CHECK_INPUT(weight_scales, _st);
        CHECK_INPUT(total_rows_before_expert, torch::kInt64);

        switch (_st)
        {
        case at::ScalarType::Half:
        {
            if (weights.scalar_type() == torch::kInt8)
            {
                CHECK_INPUT(weights, torch::kInt8);
                grouped_gemm_helper<__half, uint8_t>(
                    activations, weights, weight_scales, total_rows_before_expert, out, activation_type, config_id);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
        case at::ScalarType::BFloat16:
        {
            if (weights.scalar_type() == torch::kInt8)
            {
                CHECK_INPUT(weights, torch::kInt8);
                grouped_gemm_helper<__nv_bfloat16, uint8_t>(
                    activations, weights, weight_scales, total_rows_before_expert, out, activation_type, config_id);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
        default:
            TORCH_CHECK(false, "Incompatible tensor type for grouped gemm bias");
        }
    }

    Tensor preprocess_weights_for_mixed_gemm(Tensor row_major_quantized_weight)
    {
        CHECK_CPU(row_major_quantized_weight);
        CHECK_CONTIGUOUS(row_major_quantized_weight);
        TORCH_CHECK(row_major_quantized_weight.dim() == 2 || row_major_quantized_weight.dim() == 3,
                    "Invalid dim. The dim of weight should be 2 or 3");

        const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
        const size_t num_rows    = row_major_quantized_weight.size(-2);
        const size_t num_cols    = row_major_quantized_weight.size(-1);

        Tensor  processed_tensor = torch::zeros_like(row_major_quantized_weight);
        int8_t* input_byte_ptr   = get_ptr<int8_t>(row_major_quantized_weight);
        int8_t* output_byte_ptr  = get_ptr<int8_t>(processed_tensor);

        auto quant_type = tensorrt_llm::kernels::cutlass_kernels::QuantType::W8_A16;
        tensorrt_llm::kernels::cutlass_kernels::preprocess_weights_for_mixed_gemm(
            output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, quant_type);

        return processed_tensor;
    }
}