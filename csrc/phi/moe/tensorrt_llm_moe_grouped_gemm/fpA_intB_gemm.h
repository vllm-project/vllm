/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"
#include <cuda_runtime_api.h>
#include <vector>

namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// TRT Activation Type does not have Gelu or Silu
enum class ActivationType
{
    Gelu,
    Relu,
    Silu,
    Identity,
    InvalidType
};

/*
  This runner only supports:
  T in {half, __nv_bfloat} WeightType in {int8_t, cutlass::uint4b_t}

  Activations, biases, scales and outputs are all assumed to be row-major.

  However, it is assumed that B is in a special format governed by cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.
  In this case, B must be preprocessed using the cutlass weight only quant preprocessors. The weight preprocessor
  will instantiate the layout and preprocess based on the instantiation, so layout changes should only require
  modifications to mix_gemm_B_layout.h.
*/

class CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunnerInterface() {}

    virtual ~CutlassFpAIntBGemmRunnerInterface() {}

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n,
        int k, tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 16;
    static constexpr int MIN_N_TILE = 64;
};

template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp,
    typename ScaleZeroType = ActivationType, typename BiasType = ActivationType, typename OutputType = ActivationType>
class CutlassFpAIntBGemmRunner : public virtual CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunner();
    ~CutlassFpAIntBGemmRunner();

    void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    // Disabled since the fused GEMM, activation kernels will not be used in v1.

    // void gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C, int m, int n,
    //     int k, ActivationType activation_type, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t
    //     stream);

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(int const m, int const n, int const k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    template <typename EpilogueTag>
    void dispatch_to_arch(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
        ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
        int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
        const size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr);

private:
    int sm_;
    int multi_processor_count_;
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
