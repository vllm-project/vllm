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

namespace vllm {
namespace moe {

enum class MOEExpertScaleNormalizationMode : int
{
    NONE = 0,    //!< Run the softmax on all scales and select the topk
    RENORMALIZE, //!< Renormalize the selected scales so they sum to one. This is equivalent to only running softmax on
                 //!< the topk selected experts
};

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, int RESIDUAL_NUM, bool HAS_BIAS, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(const T* expanded_permuted_rows, T* reduced_unpermuted_output, const T* skip_1,
    const T* skip_2, const T* bias, const float* scales, const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row, const int cols, const int k, const int64_t* num_valid_ptr)
{
    const int original_row = blockIdx.x;
    const int num_rows = gridDim.x;
    const auto offset = original_row * cols;
    T* reduced_row_ptr = reduced_unpermuted_output + offset;
    const T* skip_1_row_ptr{};
    const T* skip_2_row_ptr{};

    if (RESIDUAL_NUM >= 1)
    {
        skip_1_row_ptr = skip_1 + offset;
    }

    if (RESIDUAL_NUM == 2)
    {
        skip_2_row_ptr = skip_2 + offset;
    }
    const int64_t num_valid = *num_valid_ptr;
    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
    {
        T thread_output{0.f};
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            const int expanded_original_row = original_row + k_idx * num_rows;
            const int expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            const int64_t k_offset = original_row * k + k_idx;
            const float row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row sum has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            const T* expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];

            const T* bias_ptr = bias + expert_idx * cols;
            const T bias_value = HAS_BIAS ? bias_ptr[tid] : T(0.f);

            thread_output = static_cast<float>(thread_output)
                + row_scale * static_cast<float>(expanded_permuted_rows_row_ptr[tid] + bias_value);
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || thread_output))
        {
            assert(row_rescale != 0.f);
            thread_output = static_cast<float>(thread_output) / row_rescale;
        }

        if (RESIDUAL_NUM == 1)
        {
            thread_output = thread_output + skip_1_row_ptr[tid];
        }
        else if (RESIDUAL_NUM == 2)
        {
            thread_output = thread_output + skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
        reduced_row_ptr[tid] = thread_output;
    }
}

template <typename T, int RESIDUAL_NUM>
void finalizeMoeRoutingKernelLauncherSelectBias(const T* expanded_permuted_rows, T* reduced_unpermuted_output,
    const T* skip_1, const T* skip_2, const T* bias, const float* scales,
    const int* expanded_source_row_to_expanded_dest_row, const int* expert_for_source_row, const int num_rows,
    const int cols, const int k, const int64_t* num_valid_ptr, const bool has_bias,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    const int blocks = num_rows;
    const int threads = std::min(cols, 1024);

    const bool check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3][2]
        = {{
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::NO_SCALE, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::NO_SCALE, false>},
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::DEFAULT, false>},
               {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::RENORM_SCALE, false>,
                   &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::RENORM_SCALE, false>},
           },
            {
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::NO_SCALE, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::NO_SCALE, true>},
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::DEFAULT, true>},
                {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::RENORM_SCALE, true>,
                    &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode::RENORM_SCALE, true>},
            }};
    auto* const func = func_map[check_finished][int(renorm_scales)][has_bias];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, skip_1, skip_2, bias,
        scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

template <typename T>
void finalizeMoeRoutingKernelLauncher(const T* expanded_permuted_rows, T* reduced_unpermuted_output,
    const float* topk_weights,  const int* expanded_source_row_to_expanded_dest_row, const int* expert_for_source_row,
    const int num_rows, const int cols, const int k, bool renormalize, cudaStream_t stream)
{
    const MOEExpertScaleNormalizationMode normalization_mode = renormalize ? MOEExpertScaleNormalizationMode::RENORMALIZE
                                                                         : MOEExpertScaleNormalizationMode::NONE;
    finalizeMoeRoutingKernelLauncherSelectBias<T, 0>(
        expanded_permuted_rows, reduced_unpermuted_output, nullptr, nullptr, nullptr,
        topk_weights, expanded_source_row_to_expanded_dest_row, expert_for_source_row,
        num_rows, cols, k, nullptr, false, normalization_mode, stream);
}

} // namespace moe
} // namespace vllm

void unpermute_and_reduce(
    torch::Tensor& output_tokens,               // [num_tokens, hidden_size]
    torch::Tensor& experts_output,              // [num_tokens * topk, hidden_size]
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& reverse_permutation_map,     // [num_tokens * topk]
    bool renormalize)
{
    const int hidden_size = output_tokens.size(-1);
    const int num_tokens = output_tokens.numel() / hidden_size;
    const int topk = topk_weights.size(-1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output_tokens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(
        experts_output.scalar_type(), "finalizeMoeRoutingKernelLauncher",
        [&] {
            vllm::moe::finalizeMoeRoutingKernelLauncher(
                experts_output.data_ptr<scalar_t>(),
                output_tokens.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                reverse_permutation_map.data_ptr<int>(),
                topk_indices.data_ptr<int>(),
                num_tokens,
                hidden_size,
                topk,
                renormalize,
                stream);
        });
}
