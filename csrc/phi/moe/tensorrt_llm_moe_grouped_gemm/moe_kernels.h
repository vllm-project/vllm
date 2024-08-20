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
#include "cutlass/gemm/gemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/quantization.h"
#include "moe_gemm_kernels.h"
//#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace tensorrt_llm::kernels
{

    static inline size_t pad_to_multiple_of_16(size_t const &input)
    {
        static constexpr int ALIGNMENT = 16;
        return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
    }

    class CubKeyValueSorter
    {
    public:
        CubKeyValueSorter();

        CubKeyValueSorter(int const num_experts);

        void updateNumExperts(int const num_experts);

        static size_t getWorkspaceSize(const size_t num_key_value_pairs, int const num_experts);

        void run(void *workspace, const size_t workspace_size, int const *keys_in, int *keys_out, int const *values_in,
                 int *values_out, const size_t num_key_value_pairs, cudaStream_t stream);

    private:
        int num_experts_;
        int num_bits_;
    };

    enum class MOEParallelismMode : int
    {
        NONE = 0,           //!< Ignore parallelism and duplicate the work across all nodes
        EXPERT_PARALLELISM, //!< Divide the experts between each node. The number of experts must be a multiple of
                            //!< parallelism
        TENSOR_PARALLELISM, //!< Divide the weight matrices between the nodes. The hidden dimension must be a multiple of
                            //!< parallelism
    };

    enum class MOEExpertScaleNormalizationMode : int
    {
        NONE = 0,    //!< Run the softmax on all scales and select the topk
        RENORMALIZE, //!< Renormalize the selected scales so they sum to one. This is equivalent to only running softmax on
                     //!< the topk selected experts
    };

    /**
     * \brief Describes what parallelism mode the MoE is using
     *
     * Tensor Parallelism refers to the mode where the weight matrices for each expert are sliced up between nodes.
     * Each node will handle part of each expert, the final result is achieved by summing the result.
     * The inter_size dimension should be divided by the number of nodes prior to passing it to the MoE plugin, only the
     * required slice of the weights should be provided to the plugin FC1 is a ColumnLinear and FC2 is a RowLinear, see
     * tensorrt_llm/mlp/mlp.py for an example of how this works for a single MLP
     *
     * NOTE: The bias for fc2 is only applied on rank 0. If we added it on all nodes the allreduce() would contain multiple
     * copies of the bias. The bias on other node will be ignored, and may be set to nullptr
     *
     * Expert Parallelism refers to the mode where experts are divided between the nodes. Each node will handle only the
     * tokens that are routed to the experts it is assigned to. Only the weights for the node's experts should be provided
     * to the plugin For example, with #experts = 8, expert parallelism = 2: Node 0 would handle experts 0-3, and node 1
     * would handle experts 4-7
     *
     * Regardless of parallelism mode:
     *  * The input routing values must be the complete routing for all tokens/experts (required for softmax)
     *  * An allreduce must be run on the result to combine the results from different nodes if parallelism > 1
     */
    struct MOEParallelismConfig
    {
        constexpr static MOEParallelismConfig TensorParallelism(int tp_size, int tp_rank)
        {
            return {tp_size, tp_rank, 1, 0};
        }

        constexpr static MOEParallelismConfig ExpertParallelism(int ep_size, int ep_rank)
        {
            return {1, 0, ep_size, ep_rank};
        }

        int const tp_size = 1;
        int const tp_rank = 0;
        int const ep_size = 1;
        int const ep_rank = 0;
    };

    struct QuantParams
    {
        // Int weight only quantization params
        void const *fc1_weight_scales = nullptr;
        void const *fc2_weight_scales = nullptr;

        // FP8 quantization params
        float const *dequant_fc1 = nullptr;
        float const *quant_fc2 = nullptr;
        float const *dequant_fc2 = nullptr;
        float const *quant_final = nullptr;

        static QuantParams FP8(
            float const *dequant_fc1, float const *quant_fc2, float const *dequant_fc2, float const *quant_final = nullptr)
        {
            return QuantParams{nullptr, nullptr, dequant_fc1, quant_fc2, dequant_fc2, quant_final};
        }

        static QuantParams Int(void const *fc1_weight_scales, void const *fc2_weight_scales)
        {
            return QuantParams{fc1_weight_scales, fc2_weight_scales, nullptr, nullptr, nullptr, nullptr};
        }
    };

    class CutlassMoeFCRunnerInterface
    {
    public:
        virtual ~CutlassMoeFCRunnerInterface() = default;
        virtual size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                        int const num_experts, int const k, ActivationType activation_type,
                                        MOEParallelismConfig parallelism_config) const = 0;
        virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) = 0;
        virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

        virtual void runMoe(void const *input_activations, float const *gating_output, void const *fc1_expert_weights,
                            void const *fc1_expert_biases, ActivationType fc1_activation_type, void const *fc2_expert_weights,
                            void const *fc2_expert_biases, QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
                            int64_t const inter_size, int const num_experts, int const k, char *workspace_ptr, void *final_output,
                            bool const *finished, int64_t const active_rows, void *expert_scales,
                            int *expanded_source_row_to_expanded_dest_row, int *expert_for_source_row,
                            MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
                            cudaStream_t stream) = 0;

        bool is_profiler = false;
    };

    // Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
    // Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
    // Avoid making several duplicates of this class.
    template <typename T,              /*The type used for activations/scales/compute*/
              typename WeightType,     /* The type for the MoE weights */
              typename OutputType = T, /* The type for the MoE weights */
              typename Enable = void>
    class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface
    {
    public:
        CutlassMoeFCRunner() = default;

        ~CutlassMoeFCRunner() override = default;

        static_assert(
            std::is_same_v<T, WeightType> || !std::is_same_v<T, float>, "Does not support float with quantized weights");

        size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const fc1_output_size,
                                int const num_experts, int const k, ActivationType activation_type,
                                MOEParallelismConfig parallelism_config) const override;

        void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm_config) override
        {
            moe_gemm_runner_.setBestConfig(std::move(gemm_config));
        }

        std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override
        {
            return moe_gemm_runner_.getConfigs();
        }

        static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm)
        {
            using RunnerType = decltype(moe_gemm_runner_);
            return RunnerType::getConfigs(sm);
        }

        void runMoe(void const *input_activations, float const *gating_output, void const *fc1_expert_weights,
                    void const *fc1_expert_biases, ActivationType fc1_activation_type, void const *fc2_expert_weights,
                    void const *fc2_expert_biases, QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
                    int64_t const inter_size, int const num_experts, int const k, char *workspace_ptr, void *final_output,
                    bool const *finished, int64_t const active_rows, void *expert_scales,
                    int *expanded_source_row_to_expanded_dest_row, int *expert_for_source_row,
                    MOEParallelismConfig parallelism_config, MOEExpertScaleNormalizationMode normalization_mode,
                    cudaStream_t stream) override;

    private:
        using HopperGemmOutputType = typename HopperGroupedGemmInput::OutputTypeAdaptor_t<T>;

        void computeTotalRowsBeforeExpert(int const *sorted_indices, int const total_indices, int const num_experts,
                                          int64_t *total_rows_before_expert, cudaStream_t stream);
        HopperGroupedGemmInput computeStridesHopper(int64_t const *total_rows_before_expert,
                                                    HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k, int const num_experts, T const *in,
                                                    WeightType const *weights, float const *fp8_dequant, T const *bias, HopperGemmOutputType *output,
                                                    cudaStream_t stream);
        std::vector<size_t> getWorkspaceBufferSizes(int64_t const num_rows, int64_t const hidden_size,
                                                    int64_t const inter_size, int const num_experts, int const num_experts_per_node, int const k,
                                                    ActivationType activation_type) const;
        void configureWsPtrs(char *ws_ptr, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                             int const num_experts, int const num_experts_per_node, int const k, ActivationType activation_type);

    private:
        bool mayHaveDifferentGEMMOutputType() const
        {
            // We just check if its supported because we need to know when calculating workspace size
            return moe_gemm_runner_.supportsHopperSpecialisation() && !std::is_same_v<T, HopperGemmOutputType>;
        }

        CubKeyValueSorter sorter_;
        MoeGemmRunner<T, WeightType> moe_gemm_runner_;

        // Pointers
        int *source_rows_{};
        int *permuted_rows_{};
        int *permuted_experts_{};
        char *sorter_ws_{};
        T *permuted_data_{};
        float *softmax_out_{};

        int64_t *total_rows_before_expert_{};

        void *glu_inter_result_{};
        void *fc2_result_{};
        T *fc1_result_{};

        HopperGroupedGemmInput hopper_grouped_gemm_input_;
    };

    // void makeLoadBalancedRoutingConfiguration(
    //    void *data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
