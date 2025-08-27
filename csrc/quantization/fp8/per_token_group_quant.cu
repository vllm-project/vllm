#include <ATen/cuda/CUDAContext.h>

#include "../per_token_group_quant_8bit.h"

#include <cmath>

#include <cuda_fp8.h>

#include <torch/all.h>

#include "../vectorization.cuh"
#include "../vectorization_utils.cuh"
#include "../../dispatch_utils.h"

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s, const int group_size,
    const int num_groups, const int groups_per_block, const float eps,
    const float min_8bit, const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  using scale_element_t = float;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  constexpr int vec_size = 16 / sizeof(T);
  using vec_t = vllm::vec_n_t<T, vec_size>;

  // copy global -> shared & compute absmax
  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(
      group_input,        // in
      smem_group,         // out (shared)
      group_size,         // elements per group
      lane_id,            // thread id
      threads_per_group,  // stride in group
      scalar_op_cache);   // scalar handler

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  scale_element_t y_s_quant = y_s;

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  __syncthreads();

  // quantize shared -> global 8-bit
  auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = DST_DTYPE(q);
  };

  vllm::vectorize_with_alignment<vec_size>(
      smem_group,         // in (shared)
      group_output,       // out (global quant tensor)
      group_size,         // elements
      lane_id,            // tid
      threads_per_group,  // stride
      scalar_op_quant);   // scalar handler
}

template <int num_threads, int groups_per_block, bool REORDER, typename T,
          typename DST_DTYPE, bool SCALE_UE8M0 = false,
          typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel_fused(
    int32_t num_experts, const T* __restrict__ input,
    void* __restrict__ output_q, scale_packed_t* __restrict__ output_s,
    const int group_size, const float eps, const float min_8bit,
    const float max_8bit, int32_t* expert_offsets, int32_t* c_map,
    int scale_num_rows, int topk, int a_cols) {
  static constexpr int threads_per_group = 16;
  const int32_t local_group_id = threadIdx.x / threads_per_group;
  const int half_lane_id = threadIdx.x % threads_per_group;

  const int32_t block_group_id = blockIdx.x * groups_per_block;
  const int32_t global_group_id = block_group_id + local_group_id;
  int32_t scale_id = blockIdx.x * (num_threads / threads_per_group) +
                     (threadIdx.x / threads_per_group);
  const int32_t block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  using scale_element_t = float;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  int32_t* s_expert_offsets_scaled = reinterpret_cast<int32_t*>(
      smem + (static_cast<size_t>(groups_per_block) * group_size));

  auto k_scaled = scale_num_rows;

  for (int i = threadIdx.x; i < num_experts; i += num_threads) {
    s_expert_offsets_scaled[i] = expert_offsets[i] * k_scaled;
  }

  constexpr int vec_size = 16 / sizeof(T);

  // copy global -> shared & compute absmax
  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(
      group_input,        // in
      smem_group,         // out (shared)
      group_size,         // elements per group
      half_lane_id,       // thread id
      threads_per_group,  // stride in group
      scalar_op_cache);   // scalar handler

  __syncthreads();

  local_absmax = GroupReduceMax(local_absmax, half_lane_id);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  // quantize shared -> global 8-bit
  auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = DST_DTYPE(q);
  };

  // Here we find the expert matching elem_id.
  static_assert(threads_per_group == 16);

  auto parallel_search = [&](int32_t scale_id, int32_t col_id) {
    int32_t _expert_idx = half_lane_id;
    int32_t next_expert_offset{};

    // Let's not touch any memory if we don't need to.
    for (; _expert_idx < num_experts - 1 &&
           (next_expert_offset = s_expert_offsets_scaled[_expert_idx + 1]) <=
               scale_id;
         _expert_idx += threads_per_group) {
    }

    int32_t current_expert_offset = s_expert_offsets_scaled[_expert_idx];

    bool pred =
        (_expert_idx < num_experts - 1) && current_expert_offset <= scale_id;

    auto predicate_mask = __ballot_sync(0xffffffffu, pred);

    predicate_mask =
        (predicate_mask >> ((local_group_id & 0b1u) * 16u)) & 0xffffu;
    auto expert_idx = __ffs(predicate_mask) - 1;
    if (half_lane_id == expert_idx && predicate_mask) {
      _expert_idx =
          (_expert_idx / threads_per_group) * threads_per_group + expert_idx;
      auto num_tokens =
          (next_expert_offset - current_expert_offset) / scale_num_rows;
      int32_t local_id = scale_id - current_expert_offset;
      auto t = local_id / scale_num_rows;  // Untransposed row.
      static_cast<float*>(
          output_s)[current_expert_offset + col_id * num_tokens + t] = y_s;
    }
  };
  auto col_id = scale_id % scale_num_rows;

  if constexpr (REORDER) {
    auto _row_id = block_group_offset / a_cols;
    auto c_map_ptr = c_map + topk * _row_id;

    for (int i = 0; i < topk; i++) {
      auto row_id = c_map_ptr[i];

      DST_DTYPE* group_output =
          static_cast<DST_DTYPE*>(output_q) +
          (row_id * a_cols + (block_group_offset % a_cols));
      vllm::vectorize_with_alignment<vec_size>(
          smem_group,         // in (shared)
          group_output,       // out (global quant tensor)
          group_size,         // elements
          half_lane_id,       // tid
          threads_per_group,  // stride
          scalar_op_quant);   // scalar handler

      scale_id = row_id * scale_num_rows + col_id;
      parallel_search(scale_id, col_id);
    }
  } else {
    DST_DTYPE* group_output =
        static_cast<DST_DTYPE*>(output_q) + block_group_offset;
    vllm::vectorize_with_alignment<vec_size>(
        smem_group,         // in (shared)
        group_output,       // out (global quant tensor)
        group_size,         // elements
        half_lane_id,       // tid
        threads_per_group,  // stride
        scalar_op_quant);   // scalar handler
    parallel_search(scale_id, col_id);
  }
}

void per_token_group_quant_8bit(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double min_8bit, double max_8bit,
                                bool scale_ue8m0) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());

  const int num_groups = input.numel() / group_size;

  TORCH_CHECK(input.numel() % group_size == 0);
  TORCH_CHECK(output_s.dim() == 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                         \
  do {                                                                      \
    dim3 grid(num_blocks);                                                  \
    dim3 block(num_threads);                                                \
    size_t smem_bytes =                                                     \
        4 * static_cast<size_t>(groups_per_block) * group_size * sizeof(T); \
    if (is_column_major) {                                                  \
      if (scale_ue8m0) {                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>         \
            <<<grid, block, smem_bytes, stream>>>(                          \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),     \
                static_cast<float*>(output_s.data_ptr()), group_size,       \
                num_groups, groups_per_block, (float)eps, (float)min_8bit,  \
                (float)max_8bit, scale_num_rows, scale_stride);             \
      } else {                                                              \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>        \
            <<<grid, block, smem_bytes, stream>>>(                          \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),     \
                static_cast<float*>(output_s.data_ptr()), group_size,       \
                num_groups, groups_per_block, (float)eps, (float)min_8bit,  \
                (float)max_8bit, scale_num_rows, scale_stride);             \
      }                                                                     \
    } else {                                                                \
      if (scale_ue8m0) {                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, true>        \
            <<<grid, block, smem_bytes, stream>>>(                          \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),     \
                static_cast<float*>(output_s.data_ptr()), group_size,       \
                num_groups, groups_per_block, (float)eps, (float)min_8bit,  \
                (float)max_8bit);                                           \
      } else {                                                              \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, false>       \
            <<<grid, block, smem_bytes, stream>>>(                          \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),     \
                static_cast<float*>(output_s.data_ptr()), group_size,       \
                num_groups, groups_per_block, (float)eps, (float)min_8bit,  \
                (float)max_8bit);                                           \
      }                                                                     \
    }                                                                       \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == at::ScalarType::Char) {
          LAUNCH_KERNEL(scalar_t, int8_t);
        }
      }));

#undef LAUNCH_KERNEL
}

void per_token_group_quant_8bit_fused(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double min_8bit,
    double max_8bit, bool fused, const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes, bool reorder,
    const torch::Tensor& c_map, bool scale_ue8m0) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());

  const int num_groups = input.numel() / group_size;

  TORCH_CHECK(input.numel() % group_size == 0);
  TORCH_CHECK(output_s.dim() == 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;
  auto dst_type = output_q.scalar_type();

#define LAUNCH_KERNEL(T, DST_DTYPE)                                            \
  do {                                                                         \
    const int num_blocks = num_groups / groups_per_block;                      \
    static constexpr int num_threads = groups_per_block * THREADS_PER_GROUP;   \
    const int scale_num_rows = output_s.size(1);                               \
    const int64_t num_experts = expert_offsets.size(0);                        \
    int topk = c_map.size(0) / input.size(0);                                  \
    dim3 grid(num_blocks);                                                     \
    dim3 block(num_threads);                                                   \
    size_t smem_bytes =                                                        \
        (static_cast<size_t>(groups_per_block) * group_size) * sizeof(T) +     \
        num_experts * sizeof(int32_t);                                         \
    if (reorder) {                                                             \
      if (scale_ue8m0) {                                                       \
        per_token_group_quant_8bit_kernel_fused<num_threads, groups_per_block, \
                                                true, T, DST_DTYPE, true>      \
            <<<grid, block, smem_bytes, stream>>>(                             \
                num_experts, static_cast<T*>(input.data_ptr()),                \
                output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()), \
                group_size, (float)eps, (float)min_8bit, (float)max_8bit,      \
                (int32_t*)expert_offsets.data_ptr(),                           \
                reorder ? (int32_t*)c_map.data_ptr() : nullptr,                \
                scale_num_rows, topk, (int32_t)output_q.size(1));              \
      } else {                                                                 \
        per_token_group_quant_8bit_kernel_fused<num_threads, groups_per_block, \
                                                true, T, DST_DTYPE, false>     \
            <<<grid, block, smem_bytes, stream>>>(                             \
                num_experts, static_cast<T*>(input.data_ptr()),                \
                output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()), \
                group_size, (float)eps, (float)min_8bit, (float)max_8bit,      \
                (int32_t*)expert_offsets.data_ptr(),                           \
                reorder ? (int32_t*)c_map.data_ptr() : nullptr,                \
                scale_num_rows, topk, (int32_t)output_q.size(1));              \
      }                                                                        \
    } else {                                                                   \
      if (scale_ue8m0) {                                                       \
        per_token_group_quant_8bit_kernel_fused<num_threads, groups_per_block, \
                                                false, T, DST_DTYPE, true>     \
            <<<grid, block, smem_bytes, stream>>>(                             \
                num_experts, static_cast<T*>(input.data_ptr()),                \
                output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()), \
                group_size, (float)eps, (float)min_8bit, (float)max_8bit,      \
                (int32_t*)expert_offsets.data_ptr(),                           \
                reorder ? (int32_t*)c_map.data_ptr() : nullptr,                \
                scale_num_rows, topk, (int32_t)output_q.size(1));              \
      } else {                                                                 \
        per_token_group_quant_8bit_kernel_fused<num_threads, groups_per_block, \
                                                false, T, DST_DTYPE, false>    \
            <<<grid, block, smem_bytes, stream>>>(                             \
                num_experts, static_cast<T*>(input.data_ptr()),                \
                output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()), \
                group_size, (float)eps, (float)min_8bit, (float)max_8bit,      \
                (int32_t*)expert_offsets.data_ptr(),                           \
                reorder ? (int32_t*)c_map.data_ptr() : nullptr,                \
                scale_num_rows, topk, (int32_t)output_q.size(1));              \
      }                                                                        \
    }                                                                          \
  } while (0)

  if (num_groups % 16 == 0) {
    static constexpr int groups_per_block = 16;
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit", ([&] {
          if (dst_type == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          } else if (dst_type == at::ScalarType::Char) {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }));
  } else if (num_groups % 8 == 0) {
    static constexpr int groups_per_block = 8;
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit", ([&] {
          if (dst_type == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          } else if (dst_type == at::ScalarType::Char) {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }));
  } else if (num_groups % 4 == 0) {
    static constexpr int groups_per_block = 4;
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit", ([&] {
          if (dst_type == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          } else if (dst_type == at::ScalarType::Char) {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }));
  } else if (num_groups % 2 == 0) {
    static constexpr int groups_per_block = 2;
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit", ([&] {
          if (dst_type == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          } else if (dst_type == at::ScalarType::Char) {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }));
  } else {
    static constexpr int groups_per_block = 1;
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "per_token_group_quant_8bit", ([&] {
          if (dst_type == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
          } else if (dst_type == at::ScalarType::Char) {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }));
  }

#undef LAUNCH_KERNEL
}

void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q, torch::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0, bool fused,
                               const torch::Tensor& expert_offsets,
                               const torch::Tensor& problem_sizes, bool reorder,
                               const torch::Tensor& c_map) {
  if (fused) {
    per_token_group_quant_8bit_fused(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, fused,
        expert_offsets, problem_sizes, reorder, c_map, scale_ue8m0);
  } else {
    per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                               fp8_min, fp8_max, scale_ue8m0);
  }
}
