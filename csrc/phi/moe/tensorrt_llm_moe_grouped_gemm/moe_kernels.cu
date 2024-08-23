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

// DISABLE Pytorch CUDAExtension Flags
#undef __CUDA_NO_HALF_CONVERSIONS__ 
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include "tensorrt_llm/common/workspace.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/cudaUtils.h"
#include "moe_kernels.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

    static constexpr int WARP_SIZE = 32;

    // ====================== Softmax things ===============================
    // We have our own implementation of softmax here so we can support transposing the output
    // in the softmax kernel when we extend this module to support expert-choice routing.
    template <int TPB>
    __launch_bounds__(TPB) __global__
        void moeSoftmax(float const *input, bool const *finished, float *output, int64_t const num_cols)
    {
        using BlockReduce = cub::BlockReduce<float, TPB>;
        __shared__ typename BlockReduce::TempStorage tmpStorage;

        __shared__ float normalizing_factor;
        __shared__ float float_max;

        int64_t const thread_row_offset = blockIdx.x * num_cols;

        cub::Sum sum;
        float threadData(-FLT_MAX);

        // Don't touch finished rows.
        if ((finished != nullptr) && finished[blockIdx.x])
        {
            return;
        }

        for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
        {
            int64_t const idx = thread_row_offset + ii;
            threadData = max(input[idx], threadData);
        }

        float const maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
        if (threadIdx.x == 0)
        {
            float_max = maxElem;
        }
        __syncthreads();

        threadData = 0;

        for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
        {
            int64_t const idx = thread_row_offset + ii;
            threadData += exp((static_cast<float>(input[idx]) - float_max));
        }

        auto const Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

        if (threadIdx.x == 0)
        {
            normalizing_factor = 1.f / Z;
        }
        __syncthreads();

        for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
        {
            int64_t const idx = thread_row_offset + ii;
            float const val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
            output[idx] = val;
        }
    }

    template <int TPB>
    __launch_bounds__(TPB) __global__ void moeTopK(float const *inputs_after_softmax, bool const *finished, float *output,
                                                   int *indices, int *source_rows, int const num_experts, int const k, int const start_expert, int const end_expert)
    {

        using cub_kvp = cub::KeyValuePair<int, float>;
        using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
        __shared__ typename BlockReduce::TempStorage tmpStorage;

        cub_kvp thread_kvp;
        cub::ArgMax arg_max;

        int64_t const num_rows = gridDim.x;
        int64_t const block_row = blockIdx.x;

        bool const row_is_active = finished ? !finished[block_row] : true;
        int64_t const thread_read_offset = blockIdx.x * num_experts;
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            thread_kvp.key = 0;
            thread_kvp.value = -1.f; // This is OK because inputs are probabilities

            cub_kvp inp_kvp;
            for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
            {
                int64_t const idx = thread_read_offset + expert;
                inp_kvp.key = expert;
                inp_kvp.value = inputs_after_softmax[idx];

                for (int prior_k = 0; prior_k < k_idx; ++prior_k)
                {
                    int const prior_winning_expert = indices[k * block_row + prior_k];

                    if (prior_winning_expert == expert)
                    {
                        inp_kvp = thread_kvp;
                    }
                }

                thread_kvp = arg_max(inp_kvp, thread_kvp);
            }

            cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
            if (threadIdx.x == 0)
            {
                // Ignore experts the node isn't responsible for with expert parallelism
                int const expert = result_kvp.key;
                bool const node_uses_expert = expert >= start_expert && expert < end_expert;
                bool const should_process_row = row_is_active && node_uses_expert;

                int64_t const idx = k * block_row + k_idx;
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

    template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
    __launch_bounds__(WARPS_PER_CTA *WARP_SIZE) __global__
        void topkGatingSoftmax(float const *input, bool const *finished, float *output, int64_t const num_rows,
                               int *indices, int *source_rows, int const k, int const start_expert, int const end_expert)
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
        int64_t const cta_base_row = blockIdx.x * ROWS_PER_CTA;

        // Now, using the base row per thread block, we compute the base row per warp.
        int64_t const warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

        // The threads in a warp are split into sub-groups that will work on a row.
        // We compute row offset for each thread sub-group
        int const thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
        int64_t const thread_row = warp_base_row + thread_row_in_warp;

        // Threads with indices out of bounds should early exit here.
        if (thread_row >= num_rows)
        {
            return;
        }
        bool const row_is_active = finished ? !finished[thread_row] : true;

        // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
        // row it will read.
        float const *thread_row_ptr = input + thread_row * ELTS_PER_ROW;

        // Now, we compute the group each thread belong to in order to determine the first column to start loads.
        int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
        int const first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
        float const *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

        // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
        // this can support all powers of 2 up to 16.
        using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

        // Finally, we pull in the data from global mem
        cutlass::Array<float, VPT> row_chunk;
        AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
        AccessType const *vec_thread_read_ptr = reinterpret_cast<AccessType const *>(thread_read_ptr);
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
            thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
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
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
        }

        // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
        // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
        // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
        // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
        // argmax after computing the softmax.
        float const reciprocal_row_sum = 1.f / row_sum;

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
                float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
                int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

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
                bool const node_uses_expert = expert >= start_expert && expert < end_expert;
                bool const should_process_row = row_is_active && node_uses_expert;

                // The lead thread from each sub-group will write out the final results to global memory. (This will be a
                // single) thread per row of the input/output matrices.
                int64_t const idx = k * thread_row + k_idx;
                output[idx] = max_val;
                indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
                source_rows[idx] = k_idx * num_rows + thread_row;
            }

            // Finally, we clear the value in the thread with the current max if there is another iteration to run.
            if (k_idx + 1 < k)
            {
                int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
                int const thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

                // Only the thread in the group which produced the max will reset the "winning" value to -inf.
                if (thread_group_idx == thread_to_clear_in_group)
                {
                    int const offset_for_expert = expert % ELTS_PER_LDG;
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
            static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);
            static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
            static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
            static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
            static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
        };
    } // namespace detail

    template <int EXPERTS, int WARPS_PER_TB>
    void topkGatingSoftmaxLauncherHelper(float const *input, bool const *finished, float *output, int *indices,
                                         int *source_row, int64_t const num_rows, int const k, int const start_expert, int const end_expert,
                                         cudaStream_t stream)
    {
        static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

        static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
        using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
        static constexpr int VPT = Constants::VPT;
        static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
        int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        int64_t const num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

        dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
        topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
            input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert);
    }

    void topkGatingSoftmaxKernelLauncher(float const *input, bool const *finished, float *output,
                                         float *softmax_temp_output, int *indices, int *source_row, int64_t const num_rows, int const num_experts,
                                         int const k, int const start_expert, int const end_expert, cudaStream_t stream)
    {
        static constexpr int WARPS_PER_TB = 4;

        switch (num_experts)
        {
        case 1:
        {
            topkGatingSoftmaxLauncherHelper<1, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 2:
        {
            topkGatingSoftmaxLauncherHelper<2, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 4:
        {
            topkGatingSoftmaxLauncherHelper<4, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 8:
        {
            topkGatingSoftmaxLauncherHelper<8, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 16:
        {
            topkGatingSoftmaxLauncherHelper<16, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 32:
        {
            topkGatingSoftmaxLauncherHelper<32, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 64:
        {
            topkGatingSoftmaxLauncherHelper<64, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 128:
        {
            topkGatingSoftmaxLauncherHelper<128, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        case 256:
        {
            topkGatingSoftmaxLauncherHelper<256, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, k, start_expert, end_expert, stream);
            break;
        }
        default:
        {
            static constexpr int TPB = 256;
            TLLM_CHECK(softmax_temp_output != nullptr);
            moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
            moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(
                softmax_temp_output, finished, output, indices, source_row, num_experts, k, start_expert, end_expert);
        }
        }
    }

    // ========================== CUB Sorting things ====================================
    CubKeyValueSorter::CubKeyValueSorter()
        : num_experts_(0), num_bits_(sizeof(int) * 8)
    {
    }

    CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
        : num_experts_(num_experts), num_bits_((int)log2(num_experts) + 1)
    {
    }

    void CubKeyValueSorter::updateNumExperts(int const num_experts)
    {
        num_experts_ = num_experts;
        num_bits_ = (int)log2(num_experts) + 1;
    }

    size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts)
    {
        int num_bits = static_cast<int>(log2(num_experts)) + 1;
        size_t required_storage = 0;
        int *null_int = nullptr;
        cub::DeviceRadixSort::SortPairs(
            nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);
        return required_storage;
    }

    void CubKeyValueSorter::run(void *workspace, size_t const workspace_size, int const *keys_in, int *keys_out,
                                int const *values_in, int *values_out, size_t const num_key_value_pairs, cudaStream_t stream)
    {
        size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
        size_t actual_ws_size = workspace_size;

        TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
                             "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
        cub::DeviceRadixSort::SortPairs(
            workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
    }

    // ============================== Infer GEMM sizes =================================
    // TODO Could linear search be better for small # experts
    template <class T>
    __device__ inline int64_t findTotalEltsLeqTarget(T const *sorted_indices, int64_t const arr_length, T const target)
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
    __global__ void computeTotalRowsBeforeExpertKernel(int const *sorted_experts, int64_t const sorted_experts_len,
                                                       int64_t const num_experts, int64_t *total_rows_before_expert)
    {
        // First, compute the global tid. We only need 1 thread per expert.
        int const expert = blockIdx.x * blockDim.x + threadIdx.x;
        if (expert >= num_experts)
        {
            return;
        }

        // This should construct the last index where each expert occurs.
        total_rows_before_expert[expert] = findTotalEltsLeqTarget(sorted_experts, sorted_experts_len, expert);
    }

    namespace detail
    {
        // TODO these are copied from CUTLASS because the cutlass version is missing __device__ decorator
        template <class StrideIntT>
        CUTLASS_HOST_DEVICE cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> make_cute_packed_stride(
            cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
        {
            static_assert(std::is_integral_v<StrideIntT>,
                          "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
            auto s_copy = s;
            cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
            return s_copy;
        }

        template <class StrideIntT>
        CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> make_cute_packed_stride(
            cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
        {
            static_assert(std::is_integral_v<StrideIntT>,
                          "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
            auto s_copy = s;
            cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
            return s_copy;
        }

    } // namespace detail

    __device__ void computeHopperInputStrides(
        HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx)
    {
        layout_info.stride_a[out_idx] = detail::make_cute_packed_stride(
            HopperGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, cute::Int<1>{}));
        layout_info.stride_b[out_idx] = detail::make_cute_packed_stride(
            HopperGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, cute::Int<1>{}));
        if (layout_info.stride_c)
        {
            assert(false && "CUTLASS does not support a 1xN bias");
            //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
            layout_info.stride_c[out_idx] = detail::make_cute_packed_stride(
                HopperGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, cute::Int<1>{}));
        }
        layout_info.stride_d[out_idx] = detail::make_cute_packed_stride(
            HopperGroupedGemmInput::StrideD{}, cute::make_shape(gemm_n, gemm_m, cute::Int<1>{}));
    }

    template <class T, class WeightType>
    __device__ void computeHopperInputPointers(HopperGroupedGemmInput layout_info, int64_t gemm_m, int64_t gemm_n,
                                               int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const *in, WeightType const *weights, T const *bias,
                                               HopperGroupedGemmInput::OutputTypeAdaptor_t<T> *output, int64_t const out_idx)
    {
        // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
        layout_info.ptr_a[out_idx] = in + num_tokens_before_expert * gemm_k;

        // Each expert's weight matrix is a constant size NxK, with `expert` experts
        layout_info.ptr_b[out_idx] = weights + expert * (gemm_n * gemm_k);

        if (bias)
        {
            // Each expert's bias is a constant size N, with `expert` experts
            layout_info.ptr_c[out_idx] = bias + expert * gemm_n;
        }

        // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
        layout_info.ptr_d[out_idx] = output + num_tokens_before_expert * gemm_n;
    }

    // TODO Some of this setup could be cached
    template <class T, class WeightType>
    __global__ void computeStridesHopperKernel(int64_t const *total_rows_before_expert, HopperGroupedGemmInput layout_info,
                                               int64_t gemm_n, int64_t gemm_k, int64_t const num_experts, T const *in, WeightType const *weights,
                                               float const *fp8_dequant, T const *bias, typename HopperGroupedGemmInput::OutputTypeAdaptor_t<T> *output)
    {
        // First, compute the global tid. We only need 1 thread per expert.
        int const expert = blockIdx.x * blockDim.x + threadIdx.x;
        if (expert >= num_experts)
        {
            return;
        }

        auto const num_tokens_including_expert = total_rows_before_expert[expert];
        auto const num_tokens_before_expert = expert > 0 ? total_rows_before_expert[expert - 1] : 0;
        auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
        auto const gemm_m = num_tokens_to_expert;

        // M and N transposed since we are using the #tokens as the N dimension
        layout_info.shape_info.problem_shapes[expert] = HopperGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm_n, gemm_m, gemm_k);

        if (fp8_dequant)
        {
            layout_info.alpha_scale_ptr_array[expert] = fp8_dequant + expert;
        }

        assert(gemm_m <= INT32_MAX);
        assert(gemm_n <= INT32_MAX);
        assert(gemm_k <= INT32_MAX);
        computeHopperInputStrides(layout_info, gemm_m, gemm_n, gemm_k, expert);

        computeHopperInputPointers(
            layout_info, gemm_m, gemm_n, gemm_k, num_tokens_before_expert, expert, in, weights, bias, output, expert);
    }

    // ========================== Permutation things =======================================

    template <class T, class U>
    __host__ __device__ constexpr static U arrayConvert(T const &input)
    {
        using Type = typename U::Element;
        static_assert(T::kElements == U::kElements);
        U u;
#pragma unroll
        for (int i = 0; i < U::kElements; i++)
        {
            u[i] = static_cast<Type>(input[i]);
        }
        return u;
    }

    // Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

    // "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
    // duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
    // experts in the end.

    // Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
    // k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
    // all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
    // of the expanded index.

    constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

    template <typename T, bool CHECK_SKIPPED>
    __global__ void expandInputRowsKernel(T const *unpermuted_input, T *permuted_output,
                                          int const *expanded_dest_row_to_expanded_source_row, int *expanded_source_row_to_expanded_dest_row,
                                          int64_t const num_rows, int64_t const *num_dest_rows, int64_t const cols)
    {

        // Reverse permutation map.
        // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
        // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
        // thread block will be responsible for all k summations.
        int64_t const expanded_dest_row = blockIdx.x;
        int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
        if (threadIdx.x == 0)
        {
            assert(expanded_dest_row <= INT32_MAX);
            expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
        }

        if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
        {
            // Load 128-bits per thread
            constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
            using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

            // Duplicate and permute rows
            int64_t const source_row = expanded_source_row % num_rows;

            auto const *source_row_ptr = reinterpret_cast<DataElem const *>(unpermuted_input + source_row * cols);
            auto *dest_row_ptr = reinterpret_cast<DataElem *>(permuted_output + expanded_dest_row * cols);

            int64_t const start_offset = threadIdx.x;
            int64_t const stride = EXPAND_THREADS_PER_BLOCK;
            int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

            for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
            {
                dest_row_ptr[elem_index] = source_row_ptr[elem_index];
            }
        }
    }

    template <typename T>
    void expandInputRowsKernelLauncher(T const *unpermuted_input, T *permuted_output,
                                       int const *expanded_dest_row_to_expanded_source_row, int *expanded_source_row_to_expanded_dest_row,
                                       int64_t const num_rows, int64_t const *num_valid_tokens_ptr, int64_t const cols, int const k, cudaStream_t stream)
    {
        int64_t const blocks = num_rows * k;
        int64_t const threads = EXPAND_THREADS_PER_BLOCK;
        auto func = (num_valid_tokens_ptr != nullptr) ? expandInputRowsKernel<T, true> : expandInputRowsKernel<T, false>;
        func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output, expanded_dest_row_to_expanded_source_row,
                                             expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, cols);
    }

    enum class ScaleMode : int
    {
        NO_SCALE = 0,
        DEFAULT = 1,
        RENORM_SCALE = 2,
    };

    constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

    // Final kernel to unpermute and scale
    // This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
    template <typename T, typename OutputType, class GemmOutputType, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
    __global__ void finalizeMoeRoutingKernel(GemmOutputType const *expanded_permuted_rows,
                                             OutputType *reduced_unpermuted_output, T const *bias, float const *scales,
                                             int const *expanded_source_row_to_expanded_dest_row, int const *expert_for_source_row, int64_t const orig_cols,
                                             int64_t const k, int64_t const *num_valid_ptr)
    {
        assert(orig_cols % 4 == 0);
        int64_t const original_row = blockIdx.x;
        int64_t const num_rows = gridDim.x;
        auto const offset = original_row * orig_cols;
        OutputType *reduced_row_ptr = reduced_unpermuted_output + offset;
        int64_t const num_valid = *num_valid_ptr;

        // Load 128-bits per thread, according to the smallest data type we read/write
        constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / std::min(cutlass::sizeof_bits<OutputType>::value, cutlass::sizeof_bits<GemmOutputType>::value);

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

        using BiasElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
        using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
        using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
        using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
        using a = cutlass::Array<half, FINALIZE_ELEM_PER_THREAD>;
        auto const *bias_v = reinterpret_cast<BiasElem const *>(bias);
        auto const *expanded_permuted_rows_v = reinterpret_cast<InputElem const *>(expanded_permuted_rows);
        auto *reduced_row_ptr_v = reinterpret_cast<OutputElem *>(reduced_row_ptr);

#pragma unroll
        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            bool has_valid = false;
            ComputeElem thread_output;
            thread_output.fill(0);
            float row_rescale{0.f};
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                int64_t const expanded_original_row = original_row + k_idx * num_rows;
                int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

                int64_t const k_offset = original_row * k + k_idx;
                float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
                if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
                {
                    row_rescale = row_rescale + row_scale;
                }

                // Check after row_rescale has accumulated
                if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
                {
                    continue;
                }

                auto const *expanded_permuted_rows_row_ptr = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

                int64_t const expert_idx = expert_for_source_row[k_offset];

                auto const *bias_ptr = bias_v + expert_idx * num_elems_in_col;
                ComputeElem bias_value;
                if (bias)
                {
                    bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
                }
                else
                {
                    bias_value.fill(0);
                }

                ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
                thread_output = thread_output + row_scale * (expert_result + bias_value);
                has_valid = true;
            }

            if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || has_valid))
            {
                assert(row_rescale != 0.f);
                for (auto &elem : thread_output)
                {
                    elem /= row_rescale;
                }
            }

            OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
            reduced_row_ptr_v[elem_index] = output_elem;
        }
    }

    template <class T, class OutputType, class GemmOutputType = T>
    void finalizeMoeRoutingKernelLauncher(GemmOutputType const *expanded_permuted_rows,
                                          OutputType *reduced_unpermuted_output, T const *bias, float const *scales,
                                          int const *expanded_source_row_to_expanded_dest_row, int const *expert_for_source_row, int64_t const num_rows,
                                          int64_t const cols, int64_t const k, int64_t const *num_valid_ptr, MOEParallelismConfig parallelism_config,
                                          MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
    {
        int64_t const blocks = num_rows;
        int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

        // Only add bias on rank 0 for tensor parallelism
        bool const is_rank_0 = parallelism_config.tp_rank == 0;
        T const *bias_ptr = is_rank_0 ? bias : nullptr;

        bool const check_finished = num_valid_ptr != nullptr;

        ScaleMode renorm_scales = ScaleMode::DEFAULT;
        if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
        {
            renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
        }

        using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, false>);
        FuncPtr func_map[2][3] = {
            {
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::NO_SCALE, false>,
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, false>,
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::RENORM_SCALE, false>,
            },
            {
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::NO_SCALE, true>,
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, true>,
                &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::RENORM_SCALE, true>,
            },
        };
        auto *const func = func_map[check_finished][int(renorm_scales)];
        func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, scales,
                                             expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
    }

    // ============================== Gated Activation =================================

    template <class T, class ActFn>
    __global__ void doGatedActivationKernel(
        T *output, T const *gemm_result, int64_t const *num_valid_tokens_ptr, int64_t inter_size)
    {
        int64_t const tid = threadIdx.x;
        int64_t const token = blockIdx.x;
        if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
        {
            return;
        }

        ActFn fn{};
        output = output + token * inter_size;
        gemm_result = gemm_result + token * inter_size * 2;
        for (int64_t i = tid; i < inter_size; i += blockDim.x)
        {
            auto fc1_value = static_cast<float>(gemm_result[i]);
            // BF16 isn't supported, use FP32 for activation function
            auto gate_value = static_cast<float>(gemm_result[i + inter_size]);
            float gate_act = fn(gate_value);
            output[i] = static_cast<T>(fc1_value * gate_act);
        }
    }

    template <class T>
    void doGatedActivation(T *output, T const *gemm_result, int64_t const *num_valid_tokens_ptr, int64_t inter_size,
                           int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
    {
        int64_t const blocks = num_tokens;
        int64_t const threads = std::min(inter_size, int64_t{1024});

        // TODO Instead of T use a vectored type if performance would benefit
        // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
        auto *fn = activation_type == ActivationType::Swiglu
                       ? &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>
                       : &doGatedActivationKernel<T, cutlass::epilogue::thread::GELU<float>>;
        fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
    }

    // ============================== Activation =================================

    template <class T, class ActFn>
    __global__ void doActivationKernel(T *output, HopperGroupedGemmInput::OutputTypeAdaptor_t<T> const *gemm_result,
                                       float const *fp8_quant, T const *bias_ptr, int64_t const *total_rows_before_expert_, int num_experts,
                                       int64_t inter_size, bool gated)
    {
        int64_t const tid = threadIdx.x;
        int64_t const token = blockIdx.x;
        if (token >= total_rows_before_expert_[num_experts - 1])
        {
            return;
        }

        size_t gated_mul = gated ? 2 : 1;
        size_t gated_off = gated ? inter_size : 0;

        ActFn fn{};
        gemm_result = gemm_result + token * inter_size * gated_mul;
        output = output + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases

        int64_t expert = 0;
        if (bias_ptr)
        {
            // TODO this is almost certainly faster as a linear scan
            expert = findTotalEltsLeqTarget(total_rows_before_expert_, num_experts, (int64_t)token);
        }

        float const quant_scale = fp8_quant ? *fp8_quant : 1.f;

        if (bias_ptr)
        {
            bias_ptr = bias_ptr + expert * inter_size * gated_mul;
        }
        for (int64_t i = tid; i < inter_size; i += blockDim.x)
        {
            auto fc1_value = static_cast<float>(gemm_result[i + gated_off]);
            if (bias_ptr)
            {
                fc1_value += static_cast<float>(bias_ptr[i + gated_off]);
            }

            float gate_act = fn(fc1_value);

            if (gated)
            {
                gate_act *= static_cast<float>(gemm_result[i]) + (bias_ptr ? static_cast<float>(bias_ptr[i]) : 0.0f);
            }

            output[i] = static_cast<T>(gate_act * quant_scale);
        }
    }

    template <class T>
    void doActivation(T *output, HopperGroupedGemmInput::OutputTypeAdaptor_t<T> const *gemm_result, float const *fp8_quant,
                      T const *bias, int64_t const *total_rows_before_expert_, int num_experts, int64_t inter_size, int64_t num_tokens,
                      ActivationType activation_type, cudaStream_t stream)
    {
        int64_t const blocks = num_tokens;
        int64_t const threads = std::min(inter_size, int64_t{1024});

        // TODO Instead of T use a vectored type if performance would benefit
        auto fn_list = std::array{
            &doActivationKernel<T, cutlass::epilogue::thread::GELU<float>>,    // Gelu
            &doActivationKernel<T, cutlass::epilogue::thread::ReLu<float>>,    // Relu
            &doActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>,    // Silu
            &doActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>,    // Swiglu
            &doActivationKernel<T, cutlass::epilogue::thread::GELU<float>>,    // Geglu
            &doActivationKernel<T, cutlass::epilogue::thread::Identity<float>> // Identity
        };
        auto fn = fn_list[static_cast<int>(activation_type)];
        fn<<<blocks, threads, 0, stream>>>(output, gemm_result, fp8_quant, bias, total_rows_before_expert_, num_experts,
                                           inter_size, isGatedActivation(activation_type));
    }

    template <class T, class WeightType, class OutputType, class Enable>
    std::vector<size_t> CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::getWorkspaceBufferSizes(
        int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
        int const num_experts_per_node, int const k, ActivationType activation_type) const
    {
        size_t const num_moe_inputs = k * num_rows;
        size_t const permuted_elems = num_moe_inputs * hidden_size;
        size_t const interbuf_elems = num_moe_inputs * inter_size;
        size_t glu_inter_elems = 0;
        bool is_gated_activation = isGatedActivation(activation_type);
        if (is_gated_activation)
        {
            glu_inter_elems = interbuf_elems * 2;
        }
        else if (mayHaveDifferentGEMMOutputType())
        {
            // In this case we are using activation quantization, and some intermediate buffers will be unquantized
            // We need to have separate memory for these as we can no longer alias the output buffer for reuse
            glu_inter_elems = interbuf_elems;
        }
        size_t num_softmax_outs = 0;

        bool using_hopper = moe_gemm_runner_.supportsHopperSpecialisation();
        size_t const gemm_output_dtype = using_hopper ? sizeof(HopperGemmOutputType) : sizeof(T);

        bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
        if (!is_pow_2 || num_experts > 256)
        {
            num_softmax_outs = num_rows * num_experts;
        }

        size_t const source_rows_size = num_moe_inputs * sizeof(int);
        size_t const permuted_rows_size = num_moe_inputs * sizeof(int);
        size_t const permuted_experts_size = num_moe_inputs * sizeof(int);
        size_t const permuted_data_size = permuted_elems * sizeof(T);
        size_t const total_rows_before_expert_size = num_experts_per_node * sizeof(int64_t);
        size_t const softmax_out_size = num_softmax_outs * sizeof(float);
        size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
        size_t const fc1_result_size = interbuf_elems * sizeof(T);         // Acitvation quantizes so back to sizeof(T)
        size_t const sorter_size = CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts);
        size_t const fc2_result_size = permuted_elems * gemm_output_dtype; // May be an intermediate type for quantization
        size_t const hopper_size = using_hopper ? HopperGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;
        size_t const gemm_workspace_size = moe_gemm_runner_.calcMaxWorkspaceSize(num_experts_per_node);

        std::vector<size_t> workspace{source_rows_size, permuted_rows_size, permuted_experts_size, permuted_data_size,
                                      total_rows_before_expert_size, softmax_out_size, glu_inter_size,
                                      // These pointers reuse the same memory
                                      std::max(fc1_result_size, sorter_size), fc2_result_size, hopper_size, gemm_workspace_size};
        return workspace;
    }

    template <class T, class WeightType, class OutputType, class Enable>
    size_t CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::getWorkspaceSize(int64_t const num_rows,
                                                                                   int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
                                                                                   ActivationType activation_type, MOEParallelismConfig parallelism_config) const
    {
        int const ep_size = parallelism_config.ep_size;
        TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of tp size");
        auto workspace = getWorkspaceBufferSizes(
            num_rows, hidden_size, inter_size, num_experts, num_experts / ep_size, k, activation_type);
        return tensorrt_llm::common::calculateTotalWorkspaceSize(workspace.data(), workspace.size());
    }

    template <class T, class WeightType, class OutputType, class Enable>
    void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::configureWsPtrs(char *ws_ptr, int64_t const num_rows,
                                                                                int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const num_experts_per_node,
                                                                                int const k, ActivationType activation_type)
    {
        auto ws_sizes = getWorkspaceBufferSizes(
            num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, activation_type);

        std::vector<int8_t *> ws_sliced{(int8_t *)ws_ptr};
        for (auto size : ws_sizes)
        {
            ws_sliced.push_back(nextWorkspacePtr(ws_sliced.back(), size));
        }
        ws_sliced.pop_back();

        source_rows_ = (int *)ws_sliced[0];
        permuted_rows_ = (int *)ws_sliced[1];
        permuted_experts_ = (int *)ws_sliced[2];
        permuted_data_ = (T *)ws_sliced[3];

        total_rows_before_expert_ = (int64_t *)ws_sliced[4];

        softmax_out_ = nullptr;
        bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
        if (!is_pow_2 || num_experts > 256)
        {
            softmax_out_ = (float *)ws_sliced[5];
        }

        glu_inter_result_ = (T *)ws_sliced[6];

        // These pointers are aliased. Since the sort ws can be overwritten after it is finished
        sorter_ws_ = (char *)ws_sliced[7];
        fc1_result_ = (T *)ws_sliced[7];

        fc2_result_ = (T *)ws_sliced[8];

        hopper_grouped_gemm_input_ = {};
        if (moe_gemm_runner_.isHopperSpecialised())
        {
            hopper_grouped_gemm_input_.configureWorkspace(ws_sliced[9], num_experts_per_node, ws_sliced[10], ws_sizes[10]);
        }
    }

    template <class T, class WeightType, class OutputType, class Enable>
    void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::runMoe(void const *input_activations_void,
                                                                       float const *gating_output, void const *fc1_expert_weights_void, void const *fc1_expert_biases_void,
                                                                       ActivationType fc1_activation_type, void const *fc2_expert_weights_void, void const *fc2_expert_biases_void,
                                                                       QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                                                       int const num_experts, int const k, char *workspace_ptr, void *final_output_void, bool const *finished,
                                                                       int64_t const active_rows, void *expert_scales_void, int *expanded_source_row_to_expanded_dest_row,
                                                                       int *expert_for_source_row, MOEParallelismConfig parallelism_config,
                                                                       MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
    {
        static constexpr bool int_scales_required = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
        static constexpr bool fp8_scales_required = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

        auto const *input_activations = static_cast<T const *>(input_activations_void);
        auto const *fc1_expert_weights = static_cast<WeightType const *>(fc1_expert_weights_void);
        auto const *fc1_expert_biases = static_cast<T const *>(fc1_expert_biases_void);
        auto const *fc2_expert_weights = static_cast<WeightType const *>(fc2_expert_weights_void);
        auto const *fc1_int_scales = static_cast<T const *>(quant_params.fc1_weight_scales);
        auto const *fc2_int_scales = static_cast<T const *>(quant_params.fc2_weight_scales);
        auto const *fc1_fp8_dequant = quant_params.dequant_fc1;
        auto const *fc2_fp8_quant = quant_params.quant_fc2;
        auto const *fc2_fp8_dequant = quant_params.dequant_fc2;
        auto const *fc2_expert_biases = static_cast<T const *>(fc2_expert_biases_void);
        auto *final_output = static_cast<OutputType *>(final_output_void);
        auto *expert_scales = static_cast<float *>(expert_scales_void);

        TLLM_CHECK(input_activations);
        TLLM_CHECK(gating_output);
        TLLM_CHECK(fc1_expert_weights);
        TLLM_CHECK(fc2_expert_weights);
        TLLM_CHECK(workspace_ptr);
        TLLM_CHECK(expert_scales);
        TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
        TLLM_CHECK(expert_for_source_row);
        TLLM_CHECK(num_experts % parallelism_config.ep_size == 0);
        TLLM_CHECK_WITH_INFO(hidden_size >= 128 / cutlass::sizeof_bits<WeightType>::value,
                             "Hidden size is too small to meet alignment requirements for MOE GEMM");
        TLLM_CHECK_WITH_INFO(hidden_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
                             "Hidden size does not meet minimum alignment requirements for MOE GEMM");
        TLLM_CHECK_WITH_INFO(inter_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
                             "Inter size does not meet minimum alignment requirements for MOE GEMM");

        // These values must fit into an int for building the source maps
        TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
        TLLM_CHECK_WITH_INFO(
            num_rows * num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
        TLLM_CHECK_WITH_INFO(k * num_experts <= std::numeric_limits<int>::max(), "k * num_experts is too large");

        if (int_scales_required)
        {
            TLLM_CHECK_WITH_INFO(
                fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
            TLLM_CHECK_WITH_INFO(
                fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");

            TLLM_CHECK_WITH_INFO(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
                                 "FP8 scales are provided for integer quantization");
        }
        else if (fp8_scales_required)
        {
            TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Bias is not supported with FP8");
            TLLM_CHECK_WITH_INFO(fc2_expert_biases == nullptr, "Bias is not supported with FP8");

            TLLM_CHECK_WITH_INFO(
                fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
            TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
            TLLM_CHECK_WITH_INFO(
                fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

            TLLM_CHECK_WITH_INFO(
                fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
        }
        else
        {
            TLLM_CHECK_WITH_INFO(
                fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
            TLLM_CHECK_WITH_INFO(
                fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
            TLLM_CHECK_WITH_INFO(
                fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
            TLLM_CHECK_WITH_INFO(
                fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
            TLLM_CHECK_WITH_INFO(
                fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
        }

        int const num_experts_per_node = num_experts / parallelism_config.ep_size;
        int const start_expert = num_experts_per_node * parallelism_config.ep_rank;
        int const end_expert = start_expert + num_experts_per_node;

        configureWsPtrs(
            workspace_ptr, num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, fc1_activation_type);
        topkGatingSoftmaxKernelLauncher(gating_output, finished, expert_scales, softmax_out_, expert_for_source_row,
                                        source_rows_, num_rows, num_experts, k, start_expert, end_expert, stream);

        sync_check_cuda_error();

        // We need to use the full num_experts because that is the sentinel value used by topk for disabled experts
        sorter_.updateNumExperts(num_experts);
        size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(k * num_rows, num_experts));
        sorter_.run((void *)sorter_ws_, sorter_ws_size_bytes, expert_for_source_row, permuted_experts_, source_rows_,
                    permuted_rows_, k * num_rows, stream);

        sync_check_cuda_error();

        bool const is_gated_activation = isGatedActivation(fc1_activation_type);
        bool const use_fused_moe = moe_gemm_runner_.isFusedGatedActivation(is_gated_activation, inter_size, hidden_size);
        size_t const fc1_out_size = ((!use_fused_moe) && is_gated_activation) ? inter_size * 2 : inter_size;

        // Upper bound on number of expanded rows
        int64_t const expanded_active_expert_rows = k * active_rows;
        computeTotalRowsBeforeExpert(
            permuted_experts_, expanded_active_expert_rows, num_experts_per_node, total_rows_before_expert_, stream);

        bool const needs_num_valid = finished || parallelism_config.ep_size > 1;
        int64_t const *num_valid_tokens_ptr = needs_num_valid ? total_rows_before_expert_ + num_experts_per_node - 1 : nullptr;
        expandInputRowsKernelLauncher(input_activations, permuted_data_, permuted_rows_,
                                      expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, hidden_size, k, stream);

        sync_check_cuda_error();

        bool const using_hopper = moe_gemm_runner_.isHopperSpecialised();
        HopperGroupedGemmInput hopper_input = hopper_grouped_gemm_input_;
        if (using_hopper)
        {
            bool has_different_gemm_output_type = using_hopper && mayHaveDifferentGEMMOutputType();
            auto *gemm_output = (has_different_gemm_output_type || is_gated_activation) ? glu_inter_result_
                                                                                        : static_cast<void *>(fc1_result_);

            hopper_input = computeStridesHopper(total_rows_before_expert_, hopper_input, fc1_out_size, hidden_size,
                                                num_experts_per_node, permuted_data_, fc1_expert_weights, fc1_fp8_dequant, nullptr,
                                                static_cast<HopperGemmOutputType *>(gemm_output), stream);
            sync_check_cuda_error();

            moe_gemm_runner_.moeGemm(permuted_data_, nullptr, nullptr, nullptr, total_rows_before_expert_, hopper_input,
                                     expanded_active_expert_rows, fc1_out_size, hidden_size, num_experts_per_node, false, stream);

            sync_check_cuda_error();

            doActivation<T>(fc1_result_, static_cast<HopperGemmOutputType const *>(gemm_output), fc2_fp8_quant,
                            fc1_expert_biases, total_rows_before_expert_, num_experts_per_node, inter_size, num_rows * k,
                            fc1_activation_type, stream);

            sync_check_cuda_error();
        }
        else if (!is_gated_activation)
        {
            moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_int_scales, fc1_expert_biases,
                                            fc1_result_, total_rows_before_expert_, HopperGroupedGemmInput{}, expanded_active_expert_rows, fc1_out_size,
                                            hidden_size, num_experts_per_node, fc1_activation_type, use_fused_moe, stream);

            sync_check_cuda_error();
        }
        else
        {
            // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
            ActivationType activation_type = (use_fused_moe) ? fc1_activation_type : ActivationType::Identity;
            T *gemm_result = (use_fused_moe) ? fc1_result_ : static_cast<T *>(glu_inter_result_);
            moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_int_scales, fc1_expert_biases,
                                            gemm_result, total_rows_before_expert_, HopperGroupedGemmInput{}, expanded_active_expert_rows, fc1_out_size,
                                            hidden_size, num_experts_per_node, activation_type, use_fused_moe, stream);

            sync_check_cuda_error();
            if (!use_fused_moe)
            {
                doGatedActivation<T>(fc1_result_, static_cast<T const *>(glu_inter_result_), num_valid_tokens_ptr,
                                     inter_size, num_rows * k, fc1_activation_type, stream);

                sync_check_cuda_error();
            }
        }

        sync_check_cuda_error();

        if (using_hopper)
        {
            hopper_input = computeStridesHopper(total_rows_before_expert_, hopper_input, hidden_size, inter_size,
                                                num_experts_per_node, fc1_result_, fc2_expert_weights, fc2_fp8_dequant, nullptr,
                                                static_cast<HopperGemmOutputType *>(fc2_result_), stream);
            sync_check_cuda_error();
        }

        moe_gemm_runner_.moeGemm(fc1_result_, fc2_expert_weights, fc2_int_scales, static_cast<T *>(fc2_result_),
                                 total_rows_before_expert_, hopper_input, expanded_active_expert_rows, hidden_size, inter_size,
                                 num_experts_per_node, false, stream);

        sync_check_cuda_error();

        if (using_hopper)
        {
            finalizeMoeRoutingKernelLauncher<T, OutputType, HopperGemmOutputType>(
                static_cast<HopperGemmOutputType const *>(fc2_result_), final_output, fc2_expert_biases, expert_scales,
                expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, hidden_size, k,
                num_valid_tokens_ptr, parallelism_config, normalization_mode, stream);
        }
        else
        {
            finalizeMoeRoutingKernelLauncher<T, OutputType>(static_cast<T const *>(fc2_result_), final_output,
                                                            fc2_expert_biases, expert_scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows,
                                                            hidden_size, k, num_valid_tokens_ptr, parallelism_config, normalization_mode, stream);
        }

        sync_check_cuda_error();
    }

    template <class T, class WeightType, class OutputType, class Enable>
    void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::computeTotalRowsBeforeExpert(int const *sorted_indices,
                                                                                             int const total_indices, int const num_experts, int64_t *total_rows_before_expert, cudaStream_t stream)
    {
        int const threads = std::min(1024, num_experts);
        int const blocks = (num_experts + threads - 1) / threads;

        computeTotalRowsBeforeExpertKernel<<<blocks, threads, 0, stream>>>(
            sorted_indices, total_indices, num_experts, total_rows_before_expert);
    }

    template <class T, class WeightType, class OutputType, class Enable>
    HopperGroupedGemmInput CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::computeStridesHopper(
        int64_t const *total_rows_before_expert, HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k,
        int const num_experts, T const *in, WeightType const *weights, float const *fp8_dequant, T const *bias,
        HopperGemmOutputType *output, cudaStream_t stream)
    {
        if (!bias)
        {
            layout_info.ptr_c = nullptr;
            layout_info.stride_c = nullptr;
        }

        if (!fp8_dequant)
        {
            layout_info.alpha_scale_ptr_array = nullptr;
        }

        int const threads = std::min(1024, num_experts);
        int const blocks = (num_experts + threads - 1) / threads;

        computeStridesHopperKernel<<<blocks, threads, 0, stream>>>(
            total_rows_before_expert, layout_info, gemm_n, gemm_k, num_experts, in, weights, fp8_dequant, bias, output);

        return layout_info;
    }

    // ==================== Helper for getting load balanced routing for profiling ==================================

    template <class T>
    __global__ void initRoutingKernelDiagonal(void *data_void, int num_experts, int num_tokens, int k, int stride)
    {
        assert(k == 1 || (stride % num_experts) != 0);
        int token = blockIdx.x * blockDim.x + threadIdx.x;
        if (token >= num_tokens)
        {
            return;
        }
        T *data = reinterpret_cast<T *>(data_void) + token * num_experts;
        int start = token % num_experts;
        for (int i = 0; i < k; i++)
        {
            data[start] = T{1.f};
            start += stride;
            if (start >= num_experts) // Wrap
                start -= num_experts;
        }
    }

    void makeLoadBalancedRoutingConfiguration(
        void *data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream)
    {
        TLLM_CHECK_WITH_INFO(type == nvinfer1::DataType::kFLOAT, "Routing configuration must be float");
        check_cuda_error(
            cudaMemsetAsync(data_void, 0x0, int64_t{num_experts} * int64_t{num_tokens} * sizeof(float), stream));

        int stride = tensorrt_llm::common::ceilDiv(num_experts, k);

        int blockDim = 256;
        int gridDim = tensorrt_llm::common::ceilDiv(num_tokens, blockDim);
        initRoutingKernelDiagonal<float><<<gridDim, blockDim, 0, stream>>>(data_void, num_experts, num_tokens, k, stride);

        sync_check_cuda_error();
    }

    // ==================== Variable batched GEMM specializations ==================================
    template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
    template class CutlassMoeFCRunner<half, uint8_t>;
} // namespace tensorrt_llm::kernels
