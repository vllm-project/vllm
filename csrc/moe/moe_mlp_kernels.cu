/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../dispatch_utils.h"
#include "moe_gemm_kernels.h"

#include <cuda.h>
#include <math.h>
#include <sstream>

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

namespace tensorrt_llm {

// ============================== Gated Activation =================================
template <class T, class ActFn>
__global__ void doGatedActivationKernel(
    T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, size_t inter_size)
{
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    ActFn fn{};
    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;
    for (int i = tid; i < inter_size; i += blockDim.x)
    {
        T fc1_value = gemm_result[i];
        // BF16 isn't supported, use FP32 for activation function
        float gate_value = gemm_result[i + inter_size];
        T gate_act = fn(gate_value);
        output[i] = fc1_value * gate_act;
    }
}

template <class T>
void doGatedActivation(T* output, const T* gemm_result, const int64_t* num_valid_tokens_ptr, int inter_size,
    int num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    const int blocks = num_tokens;
    const int threads = std::min(inter_size, 1024);

    // TODO Instead of T use a vectored type if performance would benefit
    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>
        : &doGatedActivationKernel<T, cutlass::epilogue::thread::GELU<float>>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

template <typename T>
void run_moe_mlp(
    T* moe_output,
    T* fc1_output,
    T* glu_output,
    const T* input_tokens,
    int64_t* cum_num_tokens_per_expert,
    const T* fc1_expert_weights,
    const T* fc1_expert_biases,
    ActivationType fc1_activation_type,
    const T* fc2_expert_weights,
    const int64_t num_expanded_tokens,
    const int hidden_size,
    const int inter_size,
    const int num_experts,
    cudaStream_t stream)
{
    // FIXME(woosuk): The MoE GEMM runner is created for each call. This is inefficient.
    tensorrt_llm::MoeGemmRunner<T, T> moe_gemm_runner;
    // Compute FC1
    if (!tensorrt_llm::isGatedActivation(fc1_activation_type)) {
        moe_gemm_runner.moeGemmBiasAct(
            input_tokens, fc1_expert_weights, nullptr, fc1_expert_biases, fc1_output,
            cum_num_tokens_per_expert, num_expanded_tokens, inter_size, hidden_size, num_experts,
            fc1_activation_type, stream);
    } else {
        const size_t fc1_out_size = inter_size * 2;
        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        moe_gemm_runner.moeGemmBiasAct(
            input_tokens, fc1_expert_weights, nullptr, fc1_expert_biases, glu_output,
            cum_num_tokens_per_expert, num_expanded_tokens, fc1_out_size, hidden_size, num_experts,
            ActivationType::Identity, stream);
        doGatedActivation<T>(
            fc1_output, glu_output, nullptr, inter_size, num_expanded_tokens,
            fc1_activation_type, stream);
    }
    // Compute FC2
    moe_gemm_runner.moeGemm(
        fc1_output, fc2_expert_weights, nullptr, moe_output, cum_num_tokens_per_expert,
        num_expanded_tokens, hidden_size, inter_size, num_experts, stream);
}

} // namespace tensorrt_llm

// FIXME(woosuk)
#define LAUNCH_MOE_MLP(scalar_t, nv_t)                                                                    \
    tensorrt_llm::run_moe_mlp<nv_t>(                                                                      \
        (nv_t *) moe_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) fc1_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) glu_output.data_ptr<scalar_t>(),                                                            \
        (nv_t *) input_tokens.data_ptr<scalar_t>(),                                                          \
        cum_num_tokens_per_expert.data_ptr<int64_t>(),                                              \
        (nv_t *) fc1_expert_weights.data_ptr<scalar_t>(),                                                    \
        (nv_t *) (fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr<scalar_t>() : nullptr),   \
        fc1_activation_type_enum,                                                                   \
        (nv_t *) fc2_expert_weights.data_ptr<scalar_t>(),                                                    \
        num_expanded_tokens,                                                                        \
        hidden_size,                                                                                \
        inter_size,                                                                                 \
        num_experts,                                                                                \
        stream);

void moe_mlp(
    torch::Tensor& moe_output,                              // [num_tokens * topk, hidden_size]
    torch::Tensor& input_tokens,                            // [num_tokens * topk, hidden_size]
    torch::Tensor& cum_num_tokens_per_expert,               // [num_experts]
    torch::Tensor& fc1_expert_weights,                      // [num_experts, inter_size or 2 * inter_size, hidden_size]
    const c10::optional<torch::Tensor>& fc1_expert_biases,  // [num_experts, inter_size]
    int fc1_activation_type,
    torch::Tensor& fc2_expert_weights)                      // [num_experts, hidden_size, inter_size]
{
    const int64_t num_expanded_tokens = input_tokens.numel() / input_tokens.size(-1);
    const int num_experts = fc2_expert_weights.size(0);
    const int hidden_size = fc2_expert_weights.size(1);
    const int inter_size = fc2_expert_weights.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tokens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::ActivationType fc1_activation_type_enum = static_cast<tensorrt_llm::ActivationType>(fc1_activation_type);
    torch::Tensor fc1_output = torch::empty({num_expanded_tokens, inter_size}, input_tokens.options());
    const bool is_glu = tensorrt_llm::isGatedActivation(fc1_activation_type_enum);
    const int64_t glu_output_size = is_glu ? num_expanded_tokens * inter_size * 2 : 0;
    torch::Tensor glu_output = torch::empty({glu_output_size}, input_tokens.options());

    auto dtype = input_tokens.dtype();
    if (dtype == at::ScalarType::Float) {
        LAUNCH_MOE_MLP(float, float);
    } else if (dtype == at::ScalarType::Half) {
        LAUNCH_MOE_MLP(at::Half, half);
    } else if (dtype == at::ScalarType::BFloat16) {
        LAUNCH_MOE_MLP(at::BFloat16, __nv_bfloat16);
    } else {
        TORCH_CHECK(false, "Unsupported data type: ", dtype);
    }
}
