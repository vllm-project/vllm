/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <type_traits>

#include "cuda_compat.h"
#include "cub_helpers.h"
#include "stable/torch_utils.h"

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

/// Aligned array type
template <typename T,
          /// Number of elements in the array
          int N,
          /// Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  }
}

// Scoring function enums
enum ScoringFunc {
  SCORING_SOFTMAX = 0,  // apply softmax
  SCORING_SIGMOID = 1   // apply sigmoid
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const InputType* input, const bool* finished, float* output,
                    const int num_cols) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = toFloat(input[idx]);
    threadData = max(val, threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = toFloat(input[idx]);
    threadData += expf(val - float_max);
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = toFloat(input[idx]);
    const float softmax_val = expf(val - float_max) * normalizing_factor;
    output[idx] = softmax_val;
  }
}

template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSigmoid(const InputType* input, const bool* finished, float* output,
                    const int num_cols) {
  const int thread_row_offset = blockIdx.x * num_cols;

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = toFloat(input[idx]);
    const float sigmoid_val = 1.0f / (1.0f + __expf(-val));
    output[idx] = sigmoid_val;
  }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__
    void moeTopK(const float* inputs_after_softmax, const bool* finished,
                 float* output, IndType* indices, int* source_rows,
                 const int num_experts, const int k, const int start_expert,
                 const int end_expert, const bool renormalize,
                 const float* bias) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  float selected_sum = 0.f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;

      // Apply correction bias if provided
      if (bias != nullptr) {
        inp_kvp.value = inputs_after_softmax[idx] + bias[expert];
      } else {
        inp_kvp.value = inputs_after_softmax[idx];
      }

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      // Ignore experts the node isn't responsible for with expert parallelism
      const int expert = result_kvp.key;
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * block_row + k_idx;
      // Return the unbiased scores for output weights
      output[idx] = inputs_after_softmax[thread_read_offset + expert];
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_rows + block_row;
      if (renormalize) {
        selected_sum += inputs_after_softmax[thread_read_offset + expert];
      }
    }
    __syncthreads();
  }

  // Renormalize the k weights for this row to sum to 1, if requested.
  if (renormalize) {
    if (threadIdx.x == 0) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * block_row + k_idx;
        output[idx] = output[idx] / denom;
      }
    }
  }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small
  power of 2. Additionally it also supports when number of experts is multiple
  of 64 which is still faster than the computing softmax and topK separately
  (only tested on CUDA yet). 2) This implementation assumes k is small, but will
  work for any k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, typename IndType, typename InputType = float,
          ScoringFunc SF>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGating(const InputType* input, const bool* finished, float* output,
                    const int num_rows, IndType* indices, int* source_rows,
                    const int k, const int start_expert, const int end_expert,
                    const bool renormalize, const float* bias) {
  static_assert(std::is_same_v<InputType, float> ||
                    std::is_same_v<InputType, __nv_bfloat16> ||
                    std::is_same_v<InputType, __half>,
                "InputType must be float, __nv_bfloat16, or __half");

  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  if constexpr (std::is_same_v<InputType, __nv_bfloat16> ||
                std::is_same_v<InputType, __half>) {
    static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
                  "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
  }

  // Restrictions based on previous section.
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM,
                "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time
  // variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Finally, we pull in the data from global mem
  float row_chunk[VPT];

  // NOTE(zhuhaoran): dispatch different input types loading, BF16/FP16 convert
  // to float
  if constexpr (std::is_same_v<InputType, float>) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
    const VecType* vec_thread_read_ptr =
        reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
  } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr =
          reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2));
        }
      }
    } else {  // ELTS_PER_LDG == 1
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __nv_bfloat16* scalar_ptr =
            thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __bfloat162float(*scalar_ptr);
      }
    }
  } else if constexpr (std::is_same_v<InputType, __half>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__half, ELTS_PER_LDG>;
      float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
      const VecType* vec_thread_read_ptr =
          reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __half22float2(
              *reinterpret_cast<const __half2*>(vec.data + jj * 2));
        }
      }
    } else {  // ELTS_PER_LDG == 1
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __half2float(*scalar_ptr);
      }
    }
  }

  if constexpr (SF == SCORING_SOFTMAX) {
    // First, we perform a max reduce within the thread.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask,
                                                            THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the
    // row. Now, we subtract the max from each element in the thread and take
    // the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = expf(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in
    // the thread_max and thread_sum variables respectively. Finally, we can
    // scale the rows for the softmax. Technically, for top-k gating we don't
    // need to compute the entire softmax row. We can likely look at the maxes
    // and only compute for the top-k values in the row. However, this kernel
    // will likely not be a bottle neck and it seems better to closer match
    // torch and find the argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }
  } else if constexpr (SF == SCORING_SIGMOID) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
    }
  }

  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  // If bias is not null, use biased value for selection
  float row_chunk_for_choice[VPT];
  // Apply correction bias
  if (bias != nullptr) {
#pragma unroll
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const int expert =
            first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
        float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
        row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] =
            row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
      }
    }
  } else {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk_for_choice[ii] = row_chunk[ii];
    }
  }

  // Now, row_chunk contains the softmax / sigmoid of the row chunk. Now, I want
  // to find the topk elements in each row, along with the max index.
  int start_col = first_elt_read_by_thread;

  float selected_sum = 0.f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val_for_choice = row_chunk_for_choice[0];
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val_for_choice > max_val_for_choice) {
          max_val_for_choice = val_for_choice;
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max_for_choice =
          VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, THREADS_PER_ROW);
      float other_max =
          VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
      int other_expert =
          VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max_for_choice > max_val_for_choice ||
          (other_max_for_choice == max_val_for_choice &&
           other_expert < expert)) {
        max_val_for_choice = other_max_for_choice;
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // Add a guard to ignore experts not included by this node
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      const int idx = k * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;
      if (renormalize) {
        selected_sum += max_val;
      }
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG +
                             offset_for_expert] = -10000.f;
      }
    }
  }

  // Renormalize the k weights for this row to sum to 1, if requested.
  if (renormalize) {
    if (thread_group_idx == 0) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] = output[idx] / denom;
      }
    }
  }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM,
          typename InputType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM,
          int MAX_BYTES_PER_LDG, typename IndType, typename InputType,
          ScoringFunc SF>
void topkGatingLauncherHelper(const InputType* input, const bool* finished,
                              float* output, IndType* indices, int* source_row,
                              const int num_rows, const int k,
                              const int start_expert, const int end_expert,
                              const bool renormalize, const float* bias,
                              cudaStream_t stream) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants =
      detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
  topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM,
             IndType, InputType, SF><<<num_blocks, block_dim, 0, stream>>>(
      input, finished, output, num_rows, indices, source_row, k, start_expert,
      end_expert, renormalize, bias);
}

#ifndef USE_ROCM
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                   \
    static_assert(WARP_SIZE == 32,                                            \
                  "Unsupported warp size. Only 32 is supported for CUDA");    \
    topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, MAX_BYTES, \
                             IndType, InputType, SF>(                         \
        gating_output, nullptr, topk_weights, topk_indices,                   \
        token_expert_indices, num_tokens, topk, 0, num_experts, renormalize,  \
        bias, stream);
#else
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                    \
    if (WARP_SIZE == 64) {                                                     \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 64, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, stream);                                                       \
    } else if (WARP_SIZE == 32) {                                              \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, stream);                                                       \
    } else {                                                                   \
      assert(false &&                                                          \
             "Unsupported warp size. Only 32 and 64 are supported for ROCm");  \
    }
#endif

template <typename IndType, typename InputType, ScoringFunc SF>
void topkGatingKernelLauncher(const InputType* gating_output,
                              float* topk_weights, IndType* topk_indices,
                              int* token_expert_indices, float* workspace,
                              const int num_tokens, const int num_experts,
                              const int topk, const bool renormalize,
                              const float* bias, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
#ifndef USE_ROCM
  // for bfloat16 dtype, we need 4 bytes loading to make sure num_experts
  // elements can be loaded by a warp
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
      (std::is_same_v<InputType, __nv_bfloat16> ||
       std::is_same_v<InputType, __half>)
          ? 4
          : 8;
#endif
  switch (num_experts) {
    case 1:
      LAUNCH_TOPK(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 2:
      LAUNCH_TOPK(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 4:
      LAUNCH_TOPK(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 8:
      LAUNCH_TOPK(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 16:
      LAUNCH_TOPK(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 32:
      LAUNCH_TOPK(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 64:
      LAUNCH_TOPK(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 128:
      LAUNCH_TOPK(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 256:
      LAUNCH_TOPK(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 512:
      LAUNCH_TOPK(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
      // (CUDA only) support multiples of 64 when num_experts is not power of 2.
      // ROCm uses WARP_SIZE 64 so 8 bytes loading won't fit for some of
      // num_experts, alternatively we can test 4 bytes loading and enable it in
      // future.
#ifndef USE_ROCM
    case 192:
      LAUNCH_TOPK(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 320:
      LAUNCH_TOPK(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 384:
      LAUNCH_TOPK(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 448:
      LAUNCH_TOPK(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 576:
      LAUNCH_TOPK(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
#endif
    default: {
      STD_TORCH_CHECK(workspace != nullptr,
                      "workspace must be provided for num_experts that are not "
                      "a power of 2 or multiple of 64.");
      static constexpr int TPB = 256;
      if constexpr (SF == SCORING_SOFTMAX) {
        moeSoftmax<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
            gating_output, nullptr, workspace, num_experts);
      } else if constexpr (SF == SCORING_SIGMOID) {
        moeSigmoid<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
            gating_output, nullptr, workspace, num_experts);
      } else {
        STD_TORCH_CHECK(false, "Unsupported scoring func");
      }
      moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
          workspace, nullptr, topk_weights, topk_indices, token_expert_indices,
          num_experts, topk, 0, num_experts, renormalize, bias);
    }
  }
}

}  // namespace moe
}  // namespace vllm

template <typename ComputeType, vllm::moe::ScoringFunc SF>
void dispatch_topk_launch(torch::stable::Tensor& gating_output,
                          torch::stable::Tensor& topk_weights,
                          torch::stable::Tensor& topk_indices,
                          torch::stable::Tensor& token_expert_indices,
                          torch::stable::Tensor& softmax_workspace,
                          int num_tokens, int num_experts, int topk,
                          bool renormalize,
                          std::optional<torch::stable::Tensor> bias,
                          cudaStream_t stream) {
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    const torch::stable::Tensor& bias_tensor = bias.value();
    STD_TORCH_CHECK(bias_tensor.scalar_type() == at::ScalarType::Float,
                    "bias tensor must be float32");
    STD_TORCH_CHECK(bias_tensor.dim() == 1, "bias tensor must be 1D");
    STD_TORCH_CHECK(bias_tensor.size(0) == num_experts,
                    "bias size mismatch, expected: ", num_experts);
    STD_TORCH_CHECK(bias_tensor.is_contiguous(),
                    "bias tensor must be contiguous");
    bias_ptr = bias_tensor.const_data_ptr<float>();
  }

  if (topk_indices.scalar_type() == torch::headeronly::ScalarType::Int) {
    vllm::moe::topkGatingKernelLauncher<int, ComputeType, SF>(
        reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
        topk_weights.mutable_data_ptr<float>(),
        topk_indices.mutable_data_ptr<int>(),
        token_expert_indices.mutable_data_ptr<int>(),
        softmax_workspace.mutable_data_ptr<float>(), num_tokens, num_experts,
        topk, renormalize, bias_ptr, stream);
  } else if (topk_indices.scalar_type() ==
             torch::headeronly::ScalarType::UInt32) {
    vllm::moe::topkGatingKernelLauncher<uint32_t, ComputeType, SF>(
        reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
        topk_weights.mutable_data_ptr<float>(),
        topk_indices.mutable_data_ptr<uint32_t>(),
        token_expert_indices.mutable_data_ptr<int>(),
        softmax_workspace.mutable_data_ptr<float>(), num_tokens, num_experts,
        topk, renormalize, bias_ptr, stream);
  } else {
    STD_TORCH_CHECK(topk_indices.scalar_type() ==
                    torch::headeronly::ScalarType::Long);
    vllm::moe::topkGatingKernelLauncher<int64_t, ComputeType, SF>(
        reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
        topk_weights.mutable_data_ptr<float>(),
        topk_indices.mutable_data_ptr<int64_t>(),
        token_expert_indices.mutable_data_ptr<int>(),
        softmax_workspace.mutable_data_ptr<float>(), num_tokens, num_experts,
        topk, renormalize, bias_ptr, stream);
  }
}

void topk_softmax(
    torch::stable::Tensor& topk_weights,          // [num_tokens, topk]
    torch::stable::Tensor& topk_indices,          // [num_tokens, topk]
    torch::stable::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::stable::Tensor& gating_output,         // [num_tokens, num_experts]
    bool renormalize, std::optional<torch::stable::Tensor> bias) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  torch::stable::accelerator::DeviceGuard device_guard(
      gating_output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(gating_output.get_device_index());
  torch::stable::Tensor softmax_workspace = torch::stable::empty(
      {workspace_size}, torch::headeronly::ScalarType::Float, std::nullopt,
      gating_output.device());

  if (gating_output.scalar_type() == torch::headeronly::ScalarType::Float) {
    dispatch_topk_launch<float, vllm::moe::SCORING_SOFTMAX>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        softmax_workspace, num_tokens, num_experts, topk, renormalize, bias,
        stream);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::Half) {
    dispatch_topk_launch<__half, vllm::moe::SCORING_SOFTMAX>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        softmax_workspace, num_tokens, num_experts, topk, renormalize, bias,
        stream);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::BFloat16) {
    dispatch_topk_launch<__nv_bfloat16, vllm::moe::SCORING_SOFTMAX>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        softmax_workspace, num_tokens, num_experts, topk, renormalize, bias,
        stream);
  } else {
    STD_TORCH_CHECK(false, "Unsupported gating_output data type: ",
                    gating_output.scalar_type());
  }
}

void topk_sigmoid(
    torch::stable::Tensor& topk_weights,          // [num_tokens, topk]
    torch::stable::Tensor& topk_indices,          // [num_tokens, topk]
    torch::stable::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::stable::Tensor& gating_output,         // [num_tokens, num_experts]
    bool renormalize, std::optional<torch::stable::Tensor> bias) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  torch::stable::accelerator::DeviceGuard device_guard(
      gating_output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(gating_output.get_device_index());
  torch::stable::Tensor workspace = torch::stable::empty(
      {workspace_size}, torch::headeronly::ScalarType::Float, std::nullopt,
      gating_output.device());

  if (gating_output.scalar_type() == torch::headeronly::ScalarType::Float) {
    dispatch_topk_launch<float, vllm::moe::SCORING_SIGMOID>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        workspace, num_tokens, num_experts, topk, renormalize, bias, stream);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::Half) {
    dispatch_topk_launch<__half, vllm::moe::SCORING_SIGMOID>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        workspace, num_tokens, num_experts, topk, renormalize, bias, stream);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::BFloat16) {
    dispatch_topk_launch<__nv_bfloat16, vllm::moe::SCORING_SIGMOID>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        workspace, num_tokens, num_experts, topk, renormalize, bias, stream);
  } else {
    STD_TORCH_CHECK(false, "Unsupported gating_output data type: ",
                    gating_output.scalar_type());
  }
}
