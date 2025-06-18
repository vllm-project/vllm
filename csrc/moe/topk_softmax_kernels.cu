/*
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
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
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"

#ifndef USE_ROCM
    #include <cub/util_type.cuh>
    #include <cub/cub.cuh>
#else
    #include <hipcub/util_type.hpp>
    #include <hipcub/hipcub.hpp>
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
    float data[N];
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const float* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    // NOTE(woosuk): The original implementation uses CUTLASS aligned array here.
    // We defined our own aligned array and use it here to avoid the dependency on CUTLASS.
    using AccessType = AlignedArray<float, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    float row_chunk[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, typename IndType>
void topkGatingSoftmaxLauncherHelper(const float* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert);
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)                       \
    topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB>(         \
        gating_output, nullptr, topk_weights, topk_indices,            \
        token_expert_indices, num_tokens, topk, 0, num_experts,         \
        stream);

template <typename IndType>
void topkGatingSoftmaxKernelLauncher(
    const float* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    cudaStream_t stream) {
    static constexpr int WARPS_PER_TB = 4;
    switch (num_experts) {
        case 1:
            LAUNCH_SOFTMAX(1, WARPS_PER_TB);
            break;
        case 2:
            LAUNCH_SOFTMAX(2, WARPS_PER_TB);
            break;
        case 4:
            LAUNCH_SOFTMAX(4, WARPS_PER_TB);
            break;
        case 8:
            LAUNCH_SOFTMAX(8, WARPS_PER_TB);
            break;
        case 16:
            LAUNCH_SOFTMAX(16, WARPS_PER_TB);
            break;
        case 32:
            LAUNCH_SOFTMAX(32, WARPS_PER_TB);
            break;
        case 64:
            LAUNCH_SOFTMAX(64, WARPS_PER_TB);
            break;
        case 128:
            LAUNCH_SOFTMAX(128, WARPS_PER_TB);
            break;
        case 256:
            LAUNCH_SOFTMAX(256, WARPS_PER_TB);
            break;
        default: {
            TORCH_CHECK(softmax_workspace != nullptr,
                "softmax_workspace must be provided for num_experts that are not a power of 2.");
            static constexpr int TPB = 256;
            moeSoftmax<TPB><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, softmax_workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, nullptr, topk_weights, topk_indices, token_expert_indices,
                num_experts, topk, 0, num_experts);
        }
    }
}

} // namespace moe
} // namespace vllm

void topk_softmax(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& token_expert_indices,        // [num_tokens, topk]
    torch::Tensor& gating_output)               // [num_tokens, num_experts]
{
    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, gating_output.options());

    if(topk_indices.scalar_type() == at::ScalarType::Int)
    {
        vllm::moe::topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens,
            num_experts,
            topk,
            stream);
    }
    else if (topk_indices.scalar_type() == at::ScalarType::UInt32)
    {
        vllm::moe::topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<uint32_t>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens,
            num_experts,
            topk,
            stream);
    }
    else {
        assert(topk_indices.scalar_type() == at::ScalarType::Int64);
        vllm::moe::topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int64_t>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens,
            num_experts,
            topk,
            stream);
    }
}
