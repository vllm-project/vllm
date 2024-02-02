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

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

namespace vllm {
namespace moe {

// ========================== CUB Sorting things ====================================
size_t get_workspace_size_for_radix_sort(
    const size_t num_key_value_pairs,
    const int num_buckets)
{
    size_t num_bits = (int) log2(num_buckets) + 1;
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        NULL, required_storage, null_int, null_int, null_int, null_int,
        num_key_value_pairs, 0, num_bits);
    return required_storage;
}

void radix_sort(
    const int* keys_in,
    int* keys_out,
    const int* values_in,
    int* values_out,
    void* workspace,
    size_t workspace_size,
    const int num_buckets,
    const size_t num_key_value_pairs,
    cudaStream_t stream)
{
    size_t num_bits = (int) log2(num_buckets) + 1;
    cub::DeviceRadixSort::SortPairs(
        workspace, workspace_size, keys_in, keys_out, values_in, values_out,
        num_key_value_pairs, 0, num_bits, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
__device__ inline int findTotalEltsLeqTarget(const int* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Sets up the gemm assuming the inputs, experts and outputs are stored in row major order.
// Assumes we want to perform output = matmul(inputs, experts) + bias
//
// "total_rows_before_expert" contains the index one past the last occurrence of the corresponding expert.
// e.g. Index 0 is the start offset of expert 1, the final entry is the total number of active rows
__global__ void computeTotalRowsBeforeExpertKernel(const int* sorted_experts, const int sorted_experts_len,
    const int64_t num_experts, int64_t* total_rows_before_expert)
{
    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = findTotalEltsLeqTarget(sorted_experts, sorted_experts_len, expert);
}

void computeTotalRowsBeforeExpert(const int* sorted_indices, const int total_indices, const int num_experts,
    int64_t* total_rows_before_expert, cudaStream_t stream)
{
    const int threads = std::min(1024, num_experts);
    const int blocks = (num_experts + threads - 1) / threads;

    computeTotalRowsBeforeExpertKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}


// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(const T* unpermuted_input, T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    const int num_rows, const int64_t* num_dest_rows, const int cols)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    const int expanded_dest_row = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
    {
        // Duplicate and permute rows
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
        {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

template <typename T>
void expandInputRowsKernelLauncher(
    T* output,
    int* reverse_permutation_map,
    const T* input_tokens,
    const int* sorted_token_expert_indices,
    const int num_tokens,
    const int hidden_size,
    const int topk,
    cudaStream_t stream)
{
    const int64_t blocks = num_tokens * topk;
    const int threads = std::min(hidden_size, 1024);
    expandInputRowsKernel<T, false><<<blocks, threads, 0, stream>>>(
        input_tokens, output, sorted_token_expert_indices, reverse_permutation_map,
        num_tokens, nullptr, hidden_size);
}

} // namespace moe
} // namespace vllm

void expand_and_permute(
    torch::Tensor& permuted_tokens,             // [num_tokens * topk, hidden_size]
    torch::Tensor& cum_num_tokens_per_expert,   // [num_experts]
    torch::Tensor& reverse_permutation_map,     // [num_tokens * topk]
    torch::Tensor& input_tokens,                // [num_tokens, hidden_size]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& token_expert_indices)        // [num_tokens, topk]
{
    const int num_experts = cum_num_tokens_per_expert.size(0);
    const int topk = topk_indices.size(-1);
    const int num_tokens = topk_indices.numel() / topk;
    const int hidden_size = input_tokens.size(-1);

    const size_t num_expanded_tokens = num_tokens * topk;
    int64_t workspace_size_bytes = (int64_t) vllm::moe::get_workspace_size_for_radix_sort(
        num_expanded_tokens, num_experts);
    workspace_size_bytes = (workspace_size_bytes + 15) / 16 * 16;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tokens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor cub_workspace = torch::empty(
        {workspace_size_bytes / input_tokens.element_size()}, input_tokens.options());
    torch::Tensor sorted_topk_indices = torch::empty_like(topk_indices);
    torch::Tensor sorted_token_expert_indices = torch::empty_like(token_expert_indices);

    // Sort the token_expert_indices using topk_indices as the key
    vllm::moe::radix_sort(
        topk_indices.data_ptr<int>(),
        sorted_topk_indices.data_ptr<int>(),
        token_expert_indices.data_ptr<int>(),
        sorted_token_expert_indices.data_ptr<int>(),
        cub_workspace.data_ptr(),
        workspace_size_bytes,
        num_experts,
        num_expanded_tokens,
        stream);

    // Compute the cumulative number of tokens per expert
    vllm::moe::computeTotalRowsBeforeExpert(
        sorted_topk_indices.data_ptr<int>(),
        num_expanded_tokens,
        num_experts,
        cum_num_tokens_per_expert.data_ptr<int64_t>(),
        stream);

    // Expand and permute the input tokens
    VLLM_DISPATCH_FLOATING_TYPES(
        input_tokens.scalar_type(), "expandInputRowsKernelLauncher",
        [&] {
            vllm::moe::expandInputRowsKernelLauncher(
                permuted_tokens.data_ptr<scalar_t>(),
                reverse_permutation_map.data_ptr<int>(),
                input_tokens.data_ptr<scalar_t>(),
                sorted_token_expert_indices.data_ptr<int>(),
                num_tokens,
                hidden_size,
                topk,
                stream);
        });
}
