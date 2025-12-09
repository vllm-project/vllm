/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.21.0/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

namespace warp_topk {

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, int64_t k) {
  int64_t cache_topk = (sizeof(T) + sizeof(idxT)) * num_of_warp * k;
  int64_t n = std::max<int>(num_of_warp / 2 * k, num_of_warp * WARP_SIZE);
  return max(cache_topk,
             round_up_to_multiple_of<256>(n * sizeof(T)) + n * sizeof(idxT));
}

template <int size, bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge {
  // input should be a bitonic sequence, and sort it to be a monotonic sequence
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(val, other_val, idx_arr[i],
                                              idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;

    // ascending doesn't matter before merging since all we need is a bitonic
    // sequence
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          is_better = (*val_arr != other &&
                       (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr,
                                                                      idx_arr);
  }
};

template <bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);

      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);  // for min
        } else {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);  // for max
        }
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  // load and merge k sorted values
  __device__ void load_sorted(T const* __restrict__ in,
                              idxT const* __restrict__ in_idx, idxT start) {
    idxT idx = start + WARP_SIZE - 1 - lane_;
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
      if (idx < start + k_) {
        T t = in[idx];
        bool is_better;
        if constexpr (is_stable) {
          is_better =
              is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
        } else {
          is_better = is_better_than<greater>(t, val_arr_[i]);
        }
        if (is_better) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);
  }

  __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  __device__ void dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;

};  // end class WarpSort

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy),
        k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];  // extern __shared__ T smem_buf[];

    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  __device__ void add(T const* in, idxT start, idxT end) {
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) {
      return;
    }

    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }

    // after done(), smem is used for merging results among warps
    __syncthreads();
  }

 private:
  __device__ void set_k_th_() {
    k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          __shfl_sync(FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  __device__ void merge_buf_(T val, idxT idx) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);

    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);

    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};  // end class WarpSelect
}  // namespace warp_topk

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__device__ inline T neg_inf() {
  // cuda::std::numeric_limits<T>::infinity() returns `0` for [T=bf16 or fp16]
  // so we need to cast from fp32
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ inline bool is_finite(const T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

// Scoring function enums
enum ScoringFunc {
  SCORING_NONE = 0,    // no activation function
  SCORING_SIGMOID = 1  // apply sigmoid
};

// Efficient sigmoid approximation from TensorRT-LLM
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

template <typename T>
__device__ inline T apply_sigmoid(T val) {
  float f = cuda_cast<float, T>(val);
  return cuda_cast<T, float>(sigmoid_accurate(f));
}

template <ScoringFunc SF, typename T>
__device__ inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  } else {
    return val;
  }
}

template <typename T, ScoringFunc SF>
__device__ void topk_with_k2(T* output, T const* input, T const* bias,
                             cg::thread_block_tile<32> const& tile,
                             int32_t const lane_id,
                             int const num_experts_per_group) {
  // Get the top2 per thread
  T largest = neg_inf<T>();
  T second_largest = neg_inf<T>();

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + bias[i];

      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + bias[i];
      largest = value;
    }
  }

  __syncwarp();  // Ensure all threads have valid data before reduction
  // Get the top2 warpwise
  T max1 = cg::reduce(tile, largest, cg::greater<T>());

  T max2 = max1;
  bool equal_to_max1 = (max1 == largest);

  int count_max1 = __popc(__ballot_sync(FULL_WARP_MASK, equal_to_max1));

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = cg::reduce(tile, largest, cg::greater<T>());
  }

  if (lane_id == 0) {
    *output = max1 + max2;
  }
}

template <typename T, ScoringFunc SF>
__global__ void topk_with_k2_kernel(T* output, T* input, T const* bias,
                                    int64_t const num_tokens,
                                    int64_t const num_cases,
                                    int64_t const n_group,
                                    int64_t const num_experts_per_group) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;

  int32_t case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
  if (case_id < num_cases) {
    input += case_id * num_experts_per_group;
    // bias is per expert group, offset to current group
    int32_t group_id = case_id % n_group;
    T const* group_bias = bias + group_id * num_experts_per_group;
    output += case_id;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    topk_with_k2<T, SF>(output, input, group_bias, tile, lane_id,
                        num_experts_per_group);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename IdxT, ScoringFunc SF, int NGroup = -1>
__global__ void group_idx_and_topk_idx_kernel(
    T* scores, T const* group_scores, float* topk_values, IdxT* topk_indices,
    T const* bias, int64_t const num_tokens, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, int64_t const num_experts,
    int64_t const num_experts_per_group, bool renormalize,
    double routed_scaling_factor) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;
  int32_t case_id =
      blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;  // one per token
  scores += case_id * num_experts;
  group_scores += case_id * n_group;
  topk_values += case_id * topk;
  topk_indices += case_id * topk;

  constexpr bool kUseStaticNGroup = (NGroup > 0);
  // use int32 to avoid implicit conversion
  int32_t const n_group_i32 =
      kUseStaticNGroup ? NGroup : static_cast<int32_t>(n_group);

  int32_t align_num_experts_per_group =
      warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];  // NOTE: reuse the shared memory here to
                                      // store the target topk idx
  int32_t* s_topk_idx = reinterpret_cast<int32_t*>(smem_buf);
  T* s_topk_value =
      reinterpret_cast<T*>(s_topk_idx + NUM_WARPS_PER_BLOCK * topk) +
      warp_id * topk;
  s_topk_idx += warp_id * topk;

  T value = neg_inf<T>();
  T topk_group_value = neg_inf<T>();
  int32_t num_equalto_topkth_group;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // I think all prolog can be put before
                                         // acqbulk because it's ptr arithmetic
#endif

  if (case_id < num_tokens) {
    // calculate group_idx
    int32_t target_num_min =
        WARP_SIZE - n_group_i32 + static_cast<int32_t>(topk_group);
    // The check is necessary to avoid abnormal input
    if (lane_id < n_group_i32 && is_finite(group_scores[lane_id])) {
      value = group_scores[lane_id];
    }

    int count_equal_to_top_value = WARP_SIZE - n_group_i32;
    int pre_count_equal_to_top_value = 0;
    // Use loop to find the largset top_group
    while (count_equal_to_top_value < target_num_min) {
      __syncwarp();  // Ensure all threads have valid data before reduction
      topk_group_value = cg::reduce(tile, value, cg::greater<T>());
      if (value == topk_group_value) {
        value = neg_inf<T>();
      }
      pre_count_equal_to_top_value = count_equal_to_top_value;
      count_equal_to_top_value =
          __popc(__ballot_sync(FULL_WARP_MASK, (value == neg_inf<T>())));
    }
    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value;
  }
  __syncthreads();

  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      queue((int32_t)topk, neg_inf<T>());

  int count_equalto_topkth_group = 0;
  bool if_proceed_next_topk = topk_group_value != neg_inf<T>();
  if (case_id < num_tokens && if_proceed_next_topk) {
    auto process_group = [&](int i_group) {
      if ((group_scores[i_group] > topk_group_value) ||
          ((group_scores[i_group] == topk_group_value) &&
           (count_equalto_topkth_group < num_equalto_topkth_group))) {
        int32_t offset = i_group * num_experts_per_group;
        for (int32_t i = lane_id; i < align_num_experts_per_group;
             i += WARP_SIZE) {
          T candidates = neg_inf<T>();
          if (i < num_experts_per_group) {
            // apply scoring function (if any) and add bias
            T input = scores[offset + i];
            if (is_finite(input)) {
              T score = apply_scoring<SF>(input);
              candidates = score + bias[offset + i];
            }
          }
          queue.add(candidates, offset + i);
        }
        if (group_scores[i_group] == topk_group_value) {
          count_equalto_topkth_group++;
        }
      }
    };

    if constexpr (kUseStaticNGroup) {
#pragma unroll
      for (int i_group = 0; i_group < NGroup; ++i_group) {
        process_group(i_group);
      }
    } else {
      for (int i_group = 0; i_group < n_group_i32; ++i_group) {
        process_group(i_group);
      }
    }
    queue.done();
    __syncwarp();
    // Get the topk_idx
    queue.dumpIdx(s_topk_idx);
    __syncwarp();
  }

  // Load the valid score value
  // Calculate the summation
  float topk_sum = 1e-20;
  if (case_id < num_tokens && if_proceed_next_topk) {
    for (int i = lane_id;
         i < warp_topk::round_up_to_multiple_of<WARP_SIZE>(topk);
         i += WARP_SIZE) {
      T value = cuda_cast<T, float>(0.0f);
      if (i < topk) {
        // Load the score value (without bias) for normalization
        T input = scores[s_topk_idx[i]];
        value = apply_scoring<SF>(input);
        s_topk_value[i] = value;
      }
      if (renormalize) {
        topk_sum +=
            cg::reduce(tile, cuda_cast<float, T>(value), cg::plus<float>());
      }
    }
  }

  __syncthreads();

  if (case_id < num_tokens) {
    if (if_proceed_next_topk) {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        float base = cuda_cast<float, T>(s_topk_value[i]);
        float value = renormalize ? (base / topk_sum * routed_scaling_factor)
                                  : (base * routed_scaling_factor);
        topk_indices[i] = s_topk_idx[i];
        topk_values[i] = value;
      }
    } else {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        topk_indices[i] = i;
        topk_values[i] = 1.0f / topk;
      }
    }
    // Note: when if_proceed_next_topk==false, choose the first 8 experts as the
    // default result.
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename IdxT, ScoringFunc SF>
inline void launch_group_idx_and_topk_kernel(
    cudaLaunchConfig_t const& config, T* scores, T* group_scores,
    float* topk_values, IdxT* topk_indices, T const* bias,
    int64_t const num_tokens, int64_t const n_group, int64_t const topk_group,
    int64_t const topk, int64_t const num_experts,
    int64_t const num_experts_per_group, bool const renormalize,
    double const routed_scaling_factor) {
  auto launch = [&](auto* kernel_instance2) {
    cudaLaunchKernelEx(&config, kernel_instance2, scores, group_scores,
                       topk_values, topk_indices, bias, num_tokens, n_group,
                       topk_group, topk, num_experts, num_experts_per_group,
                       renormalize, routed_scaling_factor);
  };

  switch (n_group) {
    case 4: {
      launch(&group_idx_and_topk_idx_kernel<T, IdxT, SF, 4>);
      break;
    }
    case 8: {
      launch(&group_idx_and_topk_idx_kernel<T, IdxT, SF, 8>);
      break;
    }
    case 16: {
      launch(&group_idx_and_topk_idx_kernel<T, IdxT, SF, 16>);
      break;
    }
    case 32: {
      launch(&group_idx_and_topk_idx_kernel<T, IdxT, SF, 32>);
      break;
    }
    default: {
      launch(&group_idx_and_topk_idx_kernel<T, IdxT, SF>);
      break;
    }
  }
}

template <typename T, typename IdxT>
void invokeNoAuxTc(T* scores, T* group_scores, float* topk_values,
                   IdxT* topk_indices, T const* bias, int64_t const num_tokens,
                   int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk,
                   bool const renormalize, double const routed_scaling_factor,
                   int const scoring_func, bool enable_pdl = false,
                   cudaStream_t const stream = 0) {
  int64_t num_cases = num_tokens * n_group;
  int64_t topk_with_k2_num_blocks = (num_cases - 1) / NUM_WARPS_PER_BLOCK + 1;
  cudaLaunchConfig_t config;
  config.gridDim = topk_with_k2_num_blocks;
  config.blockDim = BLOCK_SIZE;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto const sf = static_cast<ScoringFunc>(scoring_func);
  int64_t const num_experts_per_group = num_experts / n_group;
  auto launch_topk_with_k2 = [&](auto* kernel_instance1) {
    cudaLaunchKernelEx(&config, kernel_instance1, group_scores, scores, bias,
                       num_tokens, num_cases, n_group, num_experts_per_group);
  };
  switch (sf) {
    case SCORING_NONE: {
      auto* kernel_instance1 = &topk_with_k2_kernel<T, SCORING_NONE>;
      launch_topk_with_k2(kernel_instance1);
      break;
    }
    case SCORING_SIGMOID: {
      auto* kernel_instance1 = &topk_with_k2_kernel<T, SCORING_SIGMOID>;
      launch_topk_with_k2(kernel_instance1);
      break;
    }
    default:
      // should be guarded by higher level checks.
      TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }

  int64_t topk_with_k_group_num_blocks =
      (num_tokens - 1) / NUM_WARPS_PER_BLOCK + 1;
  size_t dynamic_smem_in_bytes =
      warp_topk::calc_smem_size_for_block_wide<T, int32_t>(NUM_WARPS_PER_BLOCK,
                                                           topk);
  config.gridDim = topk_with_k_group_num_blocks;
  config.blockDim = BLOCK_SIZE;
  config.dynamicSmemBytes = dynamic_smem_in_bytes;
  config.stream = stream;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  switch (sf) {
    case SCORING_NONE: {
      launch_group_idx_and_topk_kernel<T, IdxT, SCORING_NONE>(
          config, scores, group_scores, topk_values, topk_indices, bias,
          num_tokens, n_group, topk_group, topk, num_experts,
          num_experts_per_group, renormalize, routed_scaling_factor);
      break;
    }
    case SCORING_SIGMOID: {
      launch_group_idx_and_topk_kernel<T, IdxT, SCORING_SIGMOID>(
          config, scores, group_scores, topk_values, topk_indices, bias,
          num_tokens, n_group, topk_group, topk, num_experts,
          num_experts_per_group, renormalize, routed_scaling_factor);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }
}

#define INSTANTIATE_NOAUX_TC(T, IdxT)                                       \
  template void invokeNoAuxTc<T, IdxT>(                                     \
      T * scores, T * group_scores, float* topk_values, IdxT* topk_indices, \
      T const* bias, int64_t const num_tokens, int64_t const num_experts,   \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,  \
      bool const renormalize, double const routed_scaling_factor,           \
      int const scoring_func, bool enable_pdl, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, int32_t);
INSTANTIATE_NOAUX_TC(half, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, int32_t);
}  // end namespace moe
}  // namespace vllm

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func = 0) {
  auto data_type = scores.scalar_type();
  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
  TORCH_CHECK(num_experts % n_group == 0,
              "num_experts should be divisible by n_group");
  TORCH_CHECK(n_group <= 32,
              "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(scoring_func == vllm::moe::SCORING_NONE ||
                  scoring_func == vllm::moe::SCORING_SIGMOID,
              "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  torch::Tensor group_scores = torch::empty(
      {num_tokens, n_group}, torch::dtype(data_type).device(torch::kCUDA));
  // Always output float32 for topk_values (eliminates Python-side conversion)
  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());

  switch (data_type) {
    case torch::kFloat16:
      // Handle Float16
      vllm::moe::invokeNoAuxTc<half, int32_t>(
          reinterpret_cast<half*>(scores.mutable_data_ptr()),
          reinterpret_cast<half*>(group_scores.mutable_data_ptr()),
          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<half const*>(bias.data_ptr()), num_tokens,
          num_experts, n_group, topk_group, topk, renormalize,
          routed_scaling_factor, static_cast<int>(scoring_func), false, stream);
      break;
    case torch::kFloat32:
      // Handle Float32
      vllm::moe::invokeNoAuxTc<float, int32_t>(
          reinterpret_cast<float*>(scores.mutable_data_ptr()),
          reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<float const*>(bias.data_ptr()), num_tokens,
          num_experts, n_group, topk_group, topk, renormalize,
          routed_scaling_factor, static_cast<int>(scoring_func), false, stream);
      break;
    case torch::kBFloat16:
      // Handle BFloat16
      vllm::moe::invokeNoAuxTc<__nv_bfloat16, int32_t>(
          reinterpret_cast<__nv_bfloat16*>(scores.mutable_data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(group_scores.mutable_data_ptr()),
          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()), num_tokens,
          num_experts, n_group, topk_group, topk, renormalize,
          routed_scaling_factor, static_cast<int>(scoring_func), false, stream);
      break;
    default:
      // Handle other data types
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
  return {topk_values, topk_indices};
}
