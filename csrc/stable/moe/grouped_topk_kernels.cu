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
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include "stable/torch_utils.h"
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

  // Accessors for per-lane selected value/index.
  // NOTE: For the common case `capacity == WARP_SIZE`, `max_arr_len_ == 1`
  // and callers should use `i == 0`.
  __device__ __forceinline__ idxT get_idx(int i = 0) const {
    return idx_arr_[i];
  }

  __device__ __forceinline__ T get_val(int i = 0) const { return val_arr_[i]; }

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
        k_th_idx_(0),
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
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  } else {
    static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID,
                  "Unsupported ScoringFunc in apply_scoring");
    return val;
  }
}

template <typename T, typename BiasT, ScoringFunc SF>
__device__ void topk_with_k2(T* output, T const* input, BiasT const* bias,
                             cg::thread_block_tile<32> const& tile,
                             int32_t const lane_id,
                             int const num_experts_per_group) {
  // Get the top2 per thread
  T largest = neg_inf<T>();
  T second_largest = neg_inf<T>();

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);

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
      value = value + static_cast<T>(bias[i]);
      largest = value;
    }
  }
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

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
__global__ void grouped_topk_fused_kernel(
    T* scores, float* topk_values, IdxT* topk_indices, BiasT const* bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, bool renormalize,
    double routed_scaling_factor) {
  int32_t const token_id = static_cast<int32_t>(blockIdx.x);
  if (token_id >= num_tokens) {
    return;
  }

  int32_t const warp_id = threadIdx.x / WARP_SIZE;
  int32_t const lane_id = threadIdx.x % WARP_SIZE;

  int32_t const n_group_i32 = static_cast<int32_t>(n_group);
  int32_t const topk_group_i32 = static_cast<int32_t>(topk_group);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);

  int32_t const num_warps = blockDim.x / WARP_SIZE;
  if (warp_id >= n_group_i32 || num_warps < n_group_i32) {
    return;
  }

  int32_t const num_experts_per_group = num_experts_i32 / n_group_i32;

  T* scores_token = scores + static_cast<int64_t>(token_id) * num_experts;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];
  // warpSelect internal staging buffer layout
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;

  // user-managed shared memory starts after warpSelect internal staging.
  uintptr_t ptr_u = reinterpret_cast<uintptr_t>(smem_buf + internal_bytes);
  ptr_u = (ptr_u + 15) & ~static_cast<uintptr_t>(15);  // align to 16B
  T* s_group_scores = reinterpret_cast<T*>(ptr_u);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // I think all prolog can be put before
                                         // acqbulk because it's ptr arithmetic
#endif

  // phase 1: per-group scan
  int32_t const group_offset = warp_id * num_experts_per_group;
  topk_with_k2<T, BiasT, SF>(s_group_scores + warp_id,
                             scores_token + group_offset, bias + group_offset,
                             tile, lane_id, num_experts_per_group);

  __syncthreads();

  // phase 2: warp0 selects groups + merges candidates to final topk
  if (warp_id != 0) {
    return;
  }

  topk_values += static_cast<int64_t>(token_id) * topk;
  topk_indices += static_cast<int64_t>(token_id) * topk;

  // select topk_group groups by group score
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      group_sel(static_cast<int32_t>(topk_group_i32), neg_inf<T>());

  // all lanes must participate in WarpSelect::add().
  T gscore = (lane_id < n_group_i32) ? s_group_scores[lane_id] : neg_inf<T>();
  group_sel.add(gscore, lane_id);
  group_sel.done();

  // proceed only if the k-th selected group score is not -inf
  bool proceed = false;
  if (topk_group_i32 > 0) {
    int const kth_lane = topk_group_i32 - 1;
    // broadcast the k-th selected group score to all lanes
    T kth_val = __shfl_sync(FULL_WARP_MASK, group_sel.get_val(0), kth_lane);
    proceed = (kth_val != neg_inf<T>());
  }

  if (!proceed) {
    for (int i = lane_id; i < topk_i32; i += WARP_SIZE) {
      topk_indices[i] = static_cast<IdxT>(i);
      topk_values[i] = 1.0f / static_cast<float>(topk_i32);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

  // merge per-group topk candidates for selected groups, then select topk
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      expert_sel(static_cast<int32_t>(topk_i32), neg_inf<T>());

  // selected group ids reside in lanes [0, topk_group)
  int32_t sel_gid_lane = (lane_id < topk_group_i32) ? group_sel.get_idx(0) : 0;

  // add candidates from selected groups to expert_sel
  for (int32_t g = 0; g < topk_group_i32; ++g) {
    int32_t gid = __shfl_sync(FULL_WARP_MASK, sel_gid_lane, g);
    int32_t const offset = gid * num_experts_per_group;
    int32_t const align_num_experts_per_group =
        warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);
    for (int32_t i = lane_id; i < align_num_experts_per_group; i += WARP_SIZE) {
      // all lanes must call `add()` the same number of times.
      T cand = neg_inf<T>();
      int32_t idx = 0;
      if (i < num_experts_per_group) {
        idx = offset + i;
        T input = scores_token[idx];
        if (is_finite(input)) {
          T score = apply_scoring<SF>(input);
          cand = score + static_cast<T>(bias[idx]);
        }
      }
      expert_sel.add(cand, idx);
    }
  }
  expert_sel.done();

  // compute unbiased routing weights + optional renorm.
  float lane_unbiased = 0.0f;
  IdxT lane_idx = 0;
  if (lane_id < topk_i32) {
    lane_idx = static_cast<IdxT>(expert_sel.get_idx(0));
    T in = scores_token[static_cast<int32_t>(lane_idx)];
    lane_unbiased = cuda_cast<float, T>(apply_scoring<SF>(in));
  }

  float topk_sum = 1e-20f;
  if (renormalize) {
    topk_sum += cg::reduce(tile, lane_unbiased, cg::plus<float>());
  }

  float scale = static_cast<float>(routed_scaling_factor);
  if (renormalize) {
    scale /= topk_sum;
  }

  if (lane_id < topk_i32) {
    topk_indices[lane_id] = lane_idx;
    topk_values[lane_id] = lane_unbiased * scale;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename BiasT, typename IdxT>
void invokeNoAuxTc(T* scores, float* topk_values, IdxT* topk_indices,
                   BiasT const* bias, int64_t const num_tokens,
                   int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk,
                   bool const renormalize, double const routed_scaling_factor,
                   int const scoring_func, bool enable_pdl = false,
                   cudaStream_t const stream = 0) {
  cudaLaunchConfig_t config;
  // One block per token; one warp per group.
  config.gridDim = static_cast<uint32_t>(num_tokens);
  config.blockDim = static_cast<uint32_t>(n_group) * WARP_SIZE;
  // Dynamic shared memory: WarpSelect staging + per-group topk buffers.
  int32_t const num_warps = static_cast<int32_t>(n_group);
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;
  size_t const extra_bytes = 16 + static_cast<size_t>(n_group) * sizeof(T);
  config.dynamicSmemBytes = internal_bytes + extra_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto const sf = static_cast<ScoringFunc>(scoring_func);
  switch (sf) {
    case SCORING_NONE: {
      auto* kernel_instance =
          &grouped_topk_fused_kernel<T, BiasT, IdxT, SCORING_NONE>;
      cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                         topk_indices, bias, num_tokens, num_experts, n_group,
                         topk_group, topk, renormalize, routed_scaling_factor);
      return;
    }
    case SCORING_SIGMOID: {
      auto* kernel_instance =
          &grouped_topk_fused_kernel<T, BiasT, IdxT, SCORING_SIGMOID>;
      cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                         topk_indices, bias, num_tokens, num_experts, n_group,
                         topk_group, topk, renormalize, routed_scaling_factor);
      return;
    }
    default:
      // should be guarded by higher level checks.
      STD_TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT)                                 \
  template void invokeNoAuxTc<T, BiasT, IdxT>(                               \
      T * scores, float* topk_values, IdxT* topk_indices, BiasT const* bias, \
      int64_t const num_tokens, int64_t const num_experts,                   \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,   \
      bool const renormalize, double const routed_scaling_factor,            \
      int const scoring_func, bool enable_pdl, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, int32_t);
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(half, float, int32_t);
INSTANTIATE_NOAUX_TC(half, half, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, int32_t);
}  // end namespace moe
}  // namespace vllm

std::tuple<torch::stable::Tensor, torch::stable::Tensor> grouped_topk(
    torch::stable::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::stable::Tensor const& bias, int64_t scoring_func = 0) {
  auto data_type = scores.scalar_type();
  auto bias_type = bias.scalar_type();
  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  STD_TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
  STD_TORCH_CHECK(n_group > 0, "n_group must be positive");
  STD_TORCH_CHECK(topk > 0, "topk must be positive");
  STD_TORCH_CHECK(topk_group > 0, "topk_group must be positive");
  STD_TORCH_CHECK(topk_group <= n_group, "topk_group must be <= n_group");
  STD_TORCH_CHECK(num_experts % n_group == 0,
                  "num_experts should be divisible by n_group");
  STD_TORCH_CHECK(n_group <= 32,
                  "n_group should be smaller than or equal to 32 for now");
  STD_TORCH_CHECK(topk <= 32,
                  "topk should be smaller than or equal to 32 for now");
  STD_TORCH_CHECK(topk <= topk_group * (num_experts / n_group),
                  "topk must be <= topk_group * (num_experts / n_group)");
  STD_TORCH_CHECK(
      scoring_func == vllm::moe::SCORING_NONE ||
          scoring_func == vllm::moe::SCORING_SIGMOID,
      "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  // Always output float32 for topk_values (eliminates Python-side conversion)
  torch::stable::Tensor topk_values = torch::stable::empty(
      {num_tokens, topk}, torch::headeronly::ScalarType::Float, std::nullopt,
      scores.device());
  torch::stable::Tensor topk_indices = torch::stable::empty(
      {num_tokens, topk}, torch::headeronly::ScalarType::Int, std::nullopt,
      scores.device());

  auto stream = get_current_cuda_stream(scores.get_device_index());

#define LAUNCH_KERNEL(T, IdxT)                                                 \
  do {                                                                         \
    switch (bias_type) {                                                       \
      case torch::headeronly::ScalarType::Half:                                \
        vllm::moe::invokeNoAuxTc<T, half, IdxT>(                               \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                   \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),          \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),          \
            reinterpret_cast<half const*>(bias.const_data_ptr()), num_tokens,  \
            num_experts, n_group, topk_group, topk, renormalize,               \
            routed_scaling_factor, static_cast<int>(scoring_func), false,      \
            stream);                                                           \
        break;                                                                 \
      case torch::headeronly::ScalarType::Float:                               \
        vllm::moe::invokeNoAuxTc<T, float, IdxT>(                              \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                   \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),          \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),          \
            reinterpret_cast<float const*>(bias.const_data_ptr()), num_tokens, \
            num_experts, n_group, topk_group, topk, renormalize,               \
            routed_scaling_factor, static_cast<int>(scoring_func), false,      \
            stream);                                                           \
        break;                                                                 \
      case torch::headeronly::ScalarType::BFloat16:                            \
        vllm::moe::invokeNoAuxTc<T, __nv_bfloat16, IdxT>(                      \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                   \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),          \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),          \
            reinterpret_cast<__nv_bfloat16 const*>(bias.const_data_ptr()),     \
            num_tokens, num_experts, n_group, topk_group, topk, renormalize,   \
            routed_scaling_factor, static_cast<int>(scoring_func), false,      \
            stream);                                                           \
        break;                                                                 \
      default:                                                                 \
        throw std::invalid_argument(                                           \
            "Invalid bias dtype, only supports float16, float32, and "         \
            "bfloat16");                                                       \
        break;                                                                 \
    }                                                                          \
  } while (0)

  switch (data_type) {
    case torch::headeronly::ScalarType::Half:
      // Handle Float16
      LAUNCH_KERNEL(half, int32_t);
      break;
    case torch::headeronly::ScalarType::Float:
      // Handle Float32
      LAUNCH_KERNEL(float, int32_t);
      break;
    case torch::headeronly::ScalarType::BFloat16:
      // Handle BFloat16
      LAUNCH_KERNEL(__nv_bfloat16, int32_t);
      break;
    default:
      // Handle other data types
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_KERNEL
  return {topk_values, topk_indices};
}
