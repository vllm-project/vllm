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

#pragma once
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace tensorrt_llm
{

// Note update moe.py to match
enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    InvalidType
};

constexpr bool isGatedActivation(ActivationType activation_type)
{
    return activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu;
}

template <typename T, /*The type used for activations/scales/compute*/
    typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner
{
public:
    MoeGemmRunner();

    void setBestConfig(std::optional<cutlass_extensions::CutlassGemmConfig> best_config)
    {
        best_config_ = std::move(best_config);
    }

    void moeGemmBiasAct(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        ActivationType activation_type, cudaStream_t stream);

    void moeGemm(const T* A, const WeightType* B, const T* weight_scales, T* C, int64_t* total_rows_before_expert,
        int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream);

    std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs();

private:
    template <typename EpilogueTag>
    void dispatchToArch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, cudaStream_t stream, int* occupancy = nullptr);

    template <typename EpilogueTag>
    void runGemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cudaStream_t stream);

private:
    int sm_;
    int multi_processor_count_;
    std::optional<cutlass_extensions::CutlassGemmConfig> best_config_{};
};

} // namespace tensorrt_llm
