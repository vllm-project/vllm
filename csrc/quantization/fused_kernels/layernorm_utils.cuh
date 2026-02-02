#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"
#include "quant_conversions.cuh"

#include "../../cub_helpers.h"
#include "../../cuda_compat.h"

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  // sum of squares
  float ss = 0.0f;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, CubAddOp{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

__device__ float warpReduceMaxSpecialized(volatile float* val, int64_t tid,
                                          int64_t thread_in_warp,
                                          int64_t reduced_elems) {
  static_assert(WARP_SIZE == 32 || WARP_SIZE == 64);
  if constexpr (WARP_SIZE == 64) {
    if (thread_in_warp + 64 < reduced_elems)
      val[tid] = fmaxf(val[tid], val[tid + 64]);
  }
  if (thread_in_warp + 32 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 32]);
  if (thread_in_warp + 16 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 16]);
  if (thread_in_warp + 8 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 8]);
  if (thread_in_warp + 4 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 4]);
  if (thread_in_warp + 2 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 2]);
  if (thread_in_warp + 1 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 1]);
  return val[tid];
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false,
          bool is_scale_transposed = false, int64_t tma_scale_alignment = 0>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size, scalar_t const* __restrict__ residual = nullptr,
    int32_t const group_size = 0) {
  float block_absmax_val_maybe = 0.0f;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};
  __syncthreads();
  if (group_size > 0) {
    __shared__ float s_max_vals[1024];
    int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
    int64_t num_groups = hidden_size / group_size;
    int64_t const threads_per_group = blockDim.x / num_groups;
    int64_t const thread_in_group = threadIdx.x % threads_per_group;
    int64_t const group_offset = threadIdx.x / threads_per_group * group_size;
    int64_t const thread_offset = group_offset + thread_in_group;
    int64_t const thread_end =
        min(group_offset + group_size, static_cast<int64_t>(hidden_size));
    for (auto i = thread_offset; i < thread_end; i += threads_per_group) {
      float x = static_cast<float>(input[token_offset + i]);
      if constexpr (has_residual) {
        x += static_cast<float>(residual[token_offset + i]);
      }
      x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
      block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
    }
    s_max_vals[threadIdx.x] = block_absmax_val_maybe;
    __syncthreads();

    int64_t const warp_size = WARP_SIZE;
    int64_t const num_warps = blockDim.x / warp_size;
    int64_t const warp_id = threadIdx.x / warp_size;
    int64_t const thread_in_warp = threadIdx.x % warp_size;
    int64_t const groups_per_warp = (num_groups + num_warps - 1) / num_warps;
    for (auto i = 0; i < groups_per_warp; ++i) {
      int64_t const group_id = i * num_warps + warp_id;
      if (group_id < num_groups) {
        int64_t warp_start = group_id * threads_per_group;
        int64_t const start = warp_start + thread_in_warp;
        int64_t const warp_end = min(warp_start + threads_per_group,
                                     static_cast<int64_t>(hidden_size));
        for (auto j = start; j + warp_size < warp_end; j += warp_size) {
          s_max_vals[start] =
              fmaxf(s_max_vals[start], s_max_vals[j + warp_size]);
        }
        warpReduceMaxSpecialized(s_max_vals, start, thread_in_warp,
                                 min(warp_end - warp_start, warp_size));
      }
    }
    __syncthreads();

    if (thread_in_group == 0 && thread_offset < thread_end) {
      block_absmax_val_maybe = s_max_vals[threadIdx.x];
      float scale = 0.0f;
      if (scale_ub) {
        scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        scale = block_absmax_val_maybe;
      }
      // token scale computation
      scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
      // Global output store
      if constexpr (is_scale_transposed) {
        int64_t const scale_rows =
            tma_scale_alignment > 0 ? (gridDim.x + tma_scale_alignment - 1) / tma_scale_alignment * tma_scale_alignment : gridDim.x;
        all_token_scales[(threadIdx.x / threads_per_group) * scale_rows +
                         blockIdx.x] = scale;
      } else {
        all_token_scales[blockIdx.x * num_groups +
                         threadIdx.x / threads_per_group] = scale;
      }
    }
    __syncthreads();
  } else {
    int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

    for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      float x = static_cast<float>(input[token_offset + i]);
      if constexpr (has_residual) {
        x += static_cast<float>(residual[token_offset + i]);
      }

      x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
      block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    block_absmax_val_maybe =
        BlockReduce(reduceStore)
            .Reduce(block_absmax_val_maybe, CubMaxOp{}, blockDim.x);

    __shared__ float s_token_scale;
    if (threadIdx.x == 0) {
      float scale = 0.0f;
      if (scale_ub) {
        scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        scale = block_absmax_val_maybe;
      }
      // token scale computation
      scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
      s_token_scale = scale;                 // Shared memory store
      all_token_scales[blockIdx.x] = scale;  // Global output store
    }
    __syncthreads();

    *token_scale = s_token_scale;
  }
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false, bool is_scale_transposed = false,
          int64_t tma_scale_alignment = 0>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float* const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr,
                               int32_t const group_size = 0) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    // If groupwise is_scale_inverted is true, so we invert the scale here.
    int64_t scale_idx = 0;
    if (group_size > 0) {
      if constexpr (is_scale_transposed) {
        int64_t const scale_rows =
            tma_scale_alignment > 0 ? (gridDim.x + tma_scale_alignment - 1) / tma_scale_alignment * tma_scale_alignment : gridDim.x;
        scale_idx = (i / group_size) * scale_rows + blockIdx.x;
      } else {
        scale_idx = blockIdx.x * (hidden_size / group_size) + i / group_size;
      }
    }
    auto scale_val =
        (group_size > 0
             ? (is_scale_inverted ? 1.0f / scale[scale_idx] : scale[scale_idx])
             : *scale);
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale_val);
  }
}

namespace vectorized {

// Compute 1.0/rms(input)
// hidden_size must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];

    vec4_t<float> x;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      x.val[j] = static_cast<float>(in.val[j]);
    }

    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] += static_cast<float>(r.val[j]);
      }
    }

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      ss += x.val[j] * x.val[j];
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, CubAddOp{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

// Vectorized version of vllm::compute_dynamic_per_token_scales
// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool has_residual = false,
          bool is_scale_transposed = false, int64_t tma_scale_alignment = 0,
          int32_t group_size = 0>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  const int VEC_SIZE = 4;
  float block_absmax_val_maybe = 0.0f;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input = nullptr;
  vec4_t<scalar_t> const* vec_weight = nullptr;
  vec4_t<scalar_t> const* vec_residual = nullptr;

  if constexpr (group_size > 0) {
    __shared__ float s_max_vals[1024];

    int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
    int64_t const num_groups = hidden_size / group_size;
    int64_t const threads_per_group = blockDim.x / num_groups;
    int64_t const thread_in_group = threadIdx.x % threads_per_group;
    int64_t const group_offset =
        threadIdx.x / threads_per_group * (group_size >> 2);
    int64_t const thread_offset = group_offset + thread_in_group;
    int64_t const thread_end = min(group_offset + (group_size >> 2),
                                   static_cast<int64_t>(hidden_size >> 2));
    vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
    vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
    if constexpr (has_residual) {
      vec_residual =
          reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
    }
    int32_t const num_vec_elems = thread_end;

#pragma unroll 4
    for (auto i = thread_offset; i < num_vec_elems; i += threads_per_group) {
      vec4_t<scalar_t> in = vec_input[i];
      vec4_t<scalar_t> const w = vec_weight[i];

      vec4_t<float> x;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] = static_cast<float>(in.val[j]);
      }

      if constexpr (has_residual) {
        vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          x.val[j] += static_cast<float>(r.val[j]);
        }
      }

#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        block_absmax_val_maybe =
            fmaxf(block_absmax_val_maybe,
                  fabs(static_cast<scalar_t>(x.val[j] * rms) * w.val[j]));
      }
    }

    s_max_vals[threadIdx.x] = block_absmax_val_maybe;
    __syncthreads();

    int64_t const warp_size = WARP_SIZE;
    int64_t const num_warps = blockDim.x / warp_size;
    int64_t const warp_id = threadIdx.x / warp_size;
    int64_t const thread_in_warp = threadIdx.x % warp_size;
    int64_t const groups_per_warp = (num_groups + num_warps - 1) / num_warps;
    for (auto i = 0; i < groups_per_warp; ++i) {
      int64_t const group_id = i * num_warps + warp_id;
      if (group_id < num_groups) {
        int64_t warp_start = group_id * threads_per_group;
        int64_t const start = warp_start + thread_in_warp;
        int64_t const warp_end = min(warp_start + threads_per_group,
                                     static_cast<int64_t>(hidden_size));
        for (auto j = start; j + warp_size < warp_end; j += warp_size) {
          s_max_vals[start] =
              fmaxf(s_max_vals[start], s_max_vals[j + warp_size]);
        }
        warpReduceMaxSpecialized(s_max_vals, start, thread_in_warp,
                                 min(warp_end - warp_start, warp_size));
      }
    }
    __syncthreads();

    if (thread_in_group == 0 && thread_offset < thread_end) {
      block_absmax_val_maybe = s_max_vals[threadIdx.x];
      float scale = 0.0f;
      if (scale_ub) {
        scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        scale = block_absmax_val_maybe;
      }
      // token scale computation
      scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
      // Global output store
      if constexpr (is_scale_transposed) {
        int64_t const scale_rows =
            tma_scale_alignment > 0 ? (gridDim.x + tma_scale_alignment - 1) / tma_scale_alignment * tma_scale_alignment : gridDim.x;
        all_token_scales[(threadIdx.x / threads_per_group) * scale_rows +
                         blockIdx.x] = scale;
      } else {
        all_token_scales[blockIdx.x * num_groups +
                         threadIdx.x / threads_per_group] = scale;
      }
    }
    __syncthreads();

  } else {
    int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
    vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
    vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
    if constexpr (has_residual) {
      vec_residual =
          reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
    }

    int32_t const num_vec_elems = (hidden_size >> 2);

#pragma unroll 4
    for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
      vec4_t<scalar_t> in = vec_input[i];
      vec4_t<scalar_t> const w = vec_weight[i];

      vec4_t<float> x;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] = static_cast<float>(in.val[j]);
      }

      if constexpr (has_residual) {
        vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          x.val[j] += static_cast<float>(r.val[j]);
        }
      }

#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        block_absmax_val_maybe =
            fmaxf(block_absmax_val_maybe,
                  fabs(static_cast<scalar_t>(x.val[j] * rms) * w.val[j]));
      }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    block_absmax_val_maybe =
        BlockReduce(reduceStore)
            .Reduce(block_absmax_val_maybe, CubMaxOp{}, blockDim.x);

    __shared__ float s_token_scale;
    if (threadIdx.x == 0) {
      float scale = 0.0f;
      if (scale_ub) {
        scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        scale = block_absmax_val_maybe;
      }
      // token scale computation
      scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
      s_token_scale = scale;                 // shared memory store
      all_token_scales[blockIdx.x] = scale;  // global output store
    }
    __syncthreads();

    *token_scale = s_token_scale;
  }
}

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false, bool is_scale_transposed = false,
          int64_t tma_scale_alignment = 0, int32_t group_size = 0>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float* const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      x.val[j] = static_cast<float>(in.val[j]);
    }

    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] += static_cast<float>(r.val[j]);
      }
// Update residual
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        r.val[j] = static_cast<scalar_t>(x.val[j]);
      }
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;

    float scale_val;

    if constexpr (group_size > 0) {
      int64_t const num_groups = hidden_size / group_size;
      int64_t scale_idx = 0;
      if constexpr (is_scale_transposed) {
        int64_t const scale_rows =
            tma_scale_alignment > 0 ? (gridDim.x + tma_scale_alignment - 1) / tma_scale_alignment * tma_scale_alignment : gridDim.x;
        scale_idx = (i * VEC_SIZE / group_size) * scale_rows + blockIdx.x;
      } else {
        scale_idx = blockIdx.x * num_groups + i * VEC_SIZE / group_size;
      }
      scale_val =
          is_scale_inverted ? 1.0f / scale[scale_idx] : scale[scale_idx];
    } else {
      scale_val = *scale;
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out.val[j] = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
          static_cast<scalar_t>(x.val[j] * rms) * w.val[j], scale_val);
    }
    vec_output[i] = out;
  }
}

}  // namespace vectorized

}  // namespace vllm
