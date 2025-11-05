#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"
#include "quant_conversions.cuh"

#include "../../cub_helpers.h"

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

// TODO replace 32 with WARP_SIZE
__device__ float warpReduceMax(volatile float* val, int tid) {
  val[tid] = fmaxf(val[tid], val[tid + 32]);
  // printf("s_max vals red 32: %f (%d)\n", val[tid], tid);
  val[tid] = fmaxf(val[tid], val[tid + 16]);
  // printf("s_max vals red 16: %f (%d)\n", val[tid], tid);
  val[tid] = fmaxf(val[tid], val[tid + 8]);
  // printf("s_max vals red 8: %f (%d)\n", val[tid], tid);
  val[tid] = fmaxf(val[tid], val[tid + 4]);
  // printf("s_max vals red 4: %f (%d)\n", val[tid], tid);
  val[tid] = fmaxf(val[tid], val[tid + 2]);
  // printf("s_max vals red 2: %f (%d)\n", val[tid], tid);
  val[tid] = fmaxf(val[tid], val[tid + 1]);
  return val[tid];
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size, scalar_t const* __restrict__ residual = nullptr,
    int32_t const group_size = 0) {
  float block_absmax_val_maybe = 0.0f;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};
  if (group_size > 0) {
    // if (threadIdx.x == 0) {
    //   printf("block size: %d\n", blockDim.x);
    // }

   __shared__ float s_max_vals[1024];
    int32_t const token_offset =
        blockIdx.x * static_cast<int64_t>(hidden_size);
    int32_t num_groups = hidden_size / group_size;
    int32_t const threads_per_group = blockDim.x / num_groups;
    int32_t const thread_in_group = threadIdx.x % threads_per_group;
    int32_t const thread_offset = threadIdx.x / threads_per_group * group_size +
        thread_in_group;
    // printf("%d %d %d %d\n", threadIdx.x, threads_per_group, thread_in_group, thread_offset);
    // int64_t const hidden_element_offset = token_block_offset % hidden_size;
    for (auto i = thread_offset; i < thread_offset + group_size; i += threads_per_group) {
      float x = static_cast<float>(input[token_offset + i]);
      if constexpr (has_residual) {
        x += static_cast<float>(residual[token_offset + i]);
      }
      x = static_cast<float>(static_cast<scalar_t>(x * rms) *
                             weight[i]);
      block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
    }
    s_max_vals[threadIdx.x] = block_absmax_val_maybe;
    // printf("s_max_vals 0: %f (%d)\n", block_absmax_val_maybe, threadIdx.x);
    __syncthreads();

    int step_size = threads_per_group;
    int ctr = 1;
    while (step_size > 32 * 2) {
      step_size /= 2;
      if (thread_in_group < step_size) {
        s_max_vals[threadIdx.x] = fmaxf(s_max_vals[threadIdx.x], s_max_vals[threadIdx.x + step_size]);
        // printf("s_max_vals %d: %f (%d)\n", ctr, s_max_vals[threadIdx.x], threadIdx.x);
        ++ctr;
      }
      __syncthreads();
    }
    float reduced_local = 0.0f;
    if (thread_in_group < 32) {
      reduced_local = warpReduceMax(s_max_vals, threadIdx.x);
    }
    if (thread_in_group == 0) {
      block_absmax_val_maybe = reduced_local;
      // printf("s_max_vals end: %f (%d)\n", block_absmax_val_maybe, threadIdx.x);
    }
    __syncthreads();

    if (thread_in_group == 0) {
      // printf("block_absmax_val_maybe: %f (%d)\n", block_absmax_val_maybe, threadIdx.x);
      float scale = 0.0f;
      if (scale_ub) {
        scale = min(block_absmax_val_maybe, *scale_ub);
      } else {
        scale = block_absmax_val_maybe;
      }
      // token scale computation
      scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
      all_token_scales[blockIdx.x * num_groups + threadIdx.x / threads_per_group] = scale;  // Global output store
      token_scale[blockIdx.x * num_groups + threadIdx.x / threads_per_group] = scale;
      __syncthreads();
    }

    // using BlockReduce = cub::BlockReduce<float, 1024>;
    // __shared__ typename BlockReduce::TempStorage reduceStore;
    // block_absmax_val_maybe =
    //     BlockReduce(reduceStore)
    //         .Reduce(block_absmax_val_maybe, CubMaxOp{}, blockDim.x);
  
    // __shared__ float s_token_scale;
    // if (threadIdx.x == 0) {
    //   float scale = 0.0f;
    //   if (scale_ub) {
    //     scale = min(block_absmax_val_maybe, *scale_ub);
    //   } else {
    //     scale = block_absmax_val_maybe;
    //   }
    //   // token scale computation
    //   scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    //   s_token_scale = scale;                 // Shared memory store
    //   all_token_scales[blockIdx.x] = scale;  // Global output store
    // }
    // __syncthreads();
  
    // *token_scale = s_token_scale;

      // for each first warp of the group, do the for-loop reduction
      // then, call warpReduceMax and sync

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
          bool has_residual = false>
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
    auto scale_val =
        (group_size > 0 ? (is_scale_inverted ? 1.0f / scale[i / group_size]
                                             : scale[i / group_size])
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
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size, scalar_t const* __restrict__ residual = nullptr,
    int32_t const group_size = 0) {
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems =
      (group_size > 0 ? group_size : hidden_size) >> 2;
  float block_absmax_val_maybe = 0.0f;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input = nullptr;
  vec4_t<scalar_t> const* vec_weight = nullptr;
  vec4_t<scalar_t> const* vec_residual = nullptr;

  if (group_size > 0) {
    int64_t const token_block_offset =
        blockIdx.x * static_cast<int64_t>(group_size);
    int64_t const hidden_element_offset = token_block_offset % hidden_size;
    vec_input =
        reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_block_offset]);
    vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(
        &weight[hidden_element_offset]);
    if constexpr (has_residual) {
      vec_residual = reinterpret_cast<vec4_t<scalar_t> const*>(
          &residual[token_block_offset]);
    }
  } else {
    int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
    vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
    vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
    if constexpr (has_residual) {
      vec_residual =
          reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
    }
  }

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

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float* const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr,
                               int32_t const group_size = 0) {
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

    auto scale_val =
        (group_size > 0
             ? (is_scale_inverted ? 1.0f / scale[i * VEC_SIZE / group_size]
                                  : scale[i * VEC_SIZE / group_size])
             : *scale);
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
