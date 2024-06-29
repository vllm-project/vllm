#pragma once

#include <ATen/cuda/CUDAContext.h>
#ifndef USE_ROCM
#include <cooperative_groups.h>
#else
#include <hip/hip_cooperative_groups.h>
#endif
#ifndef USE_ROCM
#include <cuda/pipeline>
#endif
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "vec_dtypes.cuh"

namespace cg = cooperative_groups;

#ifdef USE_ROCM
template <size_t len>
__host__ __device__
inline void* memcpy_blocking(void *dst, const void *src) {
  // Does not handle the case of long datatypes
  char *d = reinterpret_cast<char *>(dst);
  const char *s = reinterpret_cast<const char *>(src);
  size_t i = 0;
#pragma unroll
  for (i = 0; i < len; ++i) {
    d[i] = s[i];
  }
  return dst;
}
#endif

#ifndef USE_ROCM

// nthrs = (32, 4)
template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty, int tz, typename in_T,
          typename out_T, typename W_T>
__global__ void
bgmv_shrink_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                   const W_T *__restrict__ W,
                   const int64_t *__restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t num_layers, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;
  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t j = blockIdx.x;
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ W_T W_shared[num_pipeline_stages * tile_size];
  __shared__ in_T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  // pipeline load W/X and compute WX;
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W + (idx * feat_out + j) * feat_in +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
  cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     X + (batch_idx * feat_in) +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
  pipe.producer_commit();
  size_t copy_idx, compute_idx;
  float y = 0.f;
  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size> w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size;
       ++tile_idx) {
    copy_idx = tile_idx % num_pipeline_stages;
    // pipeline stage: async copy W fragment
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         W + (idx * feat_out + j) * feat_in +
                             tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X + (batch_idx * feat_in) + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
    }
    pipe.producer_commit();

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // pipeline stage: compute WX
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] = sum;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) {
      y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  // final pipeline stage
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  y_warpwise[threadIdx.y] =
      ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
          ? sum
          : 0.f;
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y += y_warpwise[i];
  }

  block.sync();
  pipe.consumer_release();

  // write Y;
  if (block.thread_rank() == 0) {
    Y[batch_idx * full_y_size + y_offset + j] += static_cast<out_T>(y);
  }
}

#else

template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty, int tz, typename in_T,
          typename out_T, typename W_T>
__global__ void
bgmv_shrink_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                   const W_T *__restrict__ W,
                   const int64_t *__restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t num_layers, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;
  if (idx < 0) {
    return;
  }

  size_t j = blockIdx.x;
  constexpr size_t tile_size = tx * ty * vec_size;
  constexpr size_t num_tiles = (feat_in + tile_size - 1) / tile_size;
  __shared__ float y_warpwise[ty];

  float y = 0;
  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size> w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    if (tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x + 1) * vec_size - 1 < feat_in) {
      x_vec.load(X + (batch_idx * feat_in) +
                     tile_idx * tile_size +
                     (threadIdx.y * tx + threadIdx.x) * vec_size);
      w_vec.load(W + (idx * feat_out + j) * feat_in +
                     tile_idx * tile_size +
                     (threadIdx.y * tx + threadIdx.x) * vec_size);
    }

    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += convert_type<W_T, float>(w_vec[i]) * convert_type<in_T, float>(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += VLLM_SHFL_DOWN_SYNC(sum, offset);
    }

    __syncthreads();

    if (tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x + 1) * vec_size - 1 < feat_in) {
      y += sum;
    }
  }

  if (threadIdx.x == 0) {
    y_warpwise[threadIdx.y] = y;
  }
  __syncthreads();

  float y_write = 0.f;
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y_write += y_warpwise[i];
  }
 
  // write Y;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t y_idx = batch_idx * full_y_size + y_offset + j;
    Y[y_idx] = vllm_add<out_T>(Y[y_idx], convert_type<float, out_T>(y_write));
  }
}

#endif

// nthrs = (2, 16, 4)
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz,
          typename in_T, typename out_T, typename W_T>
__global__ void
bgmv_expand_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                   const W_T *__restrict__ W,
                   const int64_t *__restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t num_layers, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;

  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t tile_idx = blockIdx.x;

  // load X;
  vec_t<in_T, vec_size> x_vec;
  x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size);

  // load W;
  vec_t<W_T, vec_size> w_vec;
  w_vec.load(W + (idx * feat_out + tile_idx * tz * ty) * feat_in +
             block.thread_rank() * vec_size);

  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
#ifndef USE_ROCM
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
#else
    sum += convert_type<W_T, float>(w_vec[i]) * convert_type<in_T, float>(x_vec[i]) * scale;
#endif
  }

  cg::thread_block_tile g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += g.shfl_down(sum, offset);
  }
  sum = g.shfl(sum, 0);

  if (threadIdx.x == 0) {
#ifndef USE_ROCM
    Y[batch_idx * full_y_size + y_offset + tile_idx * (tz * ty) +
      threadIdx.z * ty + threadIdx.y] += static_cast<out_T>(sum);
#else
    size_t y_idx = batch_idx * full_y_size + y_offset + tile_idx * (tz * ty) +
                   threadIdx.z * ty + threadIdx.y;
    Y[y_idx] = vllm_add<out_T>(Y[y_idx], convert_type<float, out_T>(sum));
#endif
  }
}

template <int feat_in, int feat_out, typename in_T, typename out_T,
          typename W_T>
void bgmv_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                 const W_T *__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size, int64_t num_layers,
                 int64_t layer_idx, float scale) {
  constexpr size_t vec_size = 8;
  constexpr int tz = 4;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if constexpr (feat_in <= feat_out) {
    static_assert(feat_in % vec_size == 0);
    constexpr int tx = feat_in / vec_size;

    static_assert((32 % tx == 0 && feat_out % (32 / tx * tz) == 0) ||
                  (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) ||
                  (8 % tx == 0 && feat_out % (8 / tx * tz) == 0));

    if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
      constexpr int ty = 32 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else if (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
      constexpr int ty = 16 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else {
      constexpr int ty = 8 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    }
  } else {
#ifndef USE_ROCM
    static_assert(feat_in % (vec_size * 32) == 0 ||
                  feat_in % (vec_size * 16) == 0 ||
                  feat_in % (vec_size * 8) == 0);

    if constexpr (feat_in % (vec_size * 32) == 0) {
      constexpr int tx = 32;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size, vec_size * sizeof(in_T),
                         vec_size * sizeof(W_T), tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else if constexpr (feat_in % (vec_size / 2 * 32) == 0) {
      constexpr int tx = 32;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size / 2,
                         vec_size * sizeof(in_T) / 2,
                         vec_size * sizeof(W_T) / 2, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else if constexpr (feat_in % (vec_size / 2 * 16) == 0) {
      constexpr int tx = 16;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size / 2,
                         vec_size * sizeof(in_T) / 2,
                         vec_size * sizeof(W_T) / 2, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    }
#else
    constexpr size_t rocm_warp_size = warpSize;

#define CHECK_INPUT_TILEABLE_BY(vec_size_) \
    feat_in % (rocm_warp_size * vec_size_) == 0

#define LAUNCH_BGMV_SHRINK_KERNELS_ROCM(factor_, vec_size_, tx_, ty_)       \
    if constexpr (CHECK_INPUT_TILEABLE_BY(factor_)) {                       \
      constexpr size_t vec_size_shrink = vec_size_;                         \
      constexpr int tx = tx_;                                               \
      constexpr int ty = ty_;                                               \
      dim3 nblks(feat_out, batch_size);                                     \
      dim3 nthrs(tx, ty);                                                   \
      bgmv_shrink_kernel<feat_in, feat_out, vec_size_shrink,                \
                          vec_size_shrink * sizeof(in_T),                   \
                          vec_size_shrink * sizeof(W_T),                    \
                          tx, ty, tz>                                       \
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,        \
                                        full_y_size, num_layers, layer_idx, \
                                        scale);                             \
    }

    static_assert(CHECK_INPUT_TILEABLE_BY(32) ||
                  CHECK_INPUT_TILEABLE_BY(16) ||
                  CHECK_INPUT_TILEABLE_BY( 8) ||
                  CHECK_INPUT_TILEABLE_BY( 4) ||
                  CHECK_INPUT_TILEABLE_BY( 2) ||
                  CHECK_INPUT_TILEABLE_BY( 1));
    
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM(32, vec_size, rocm_warp_size, 32/vec_size)
    else
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM(16, vec_size, rocm_warp_size, 16/vec_size)
    else
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM( 8, vec_size, rocm_warp_size,  8/vec_size)
    else
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM( 4, vec_size, rocm_warp_size/(vec_size/4), vec_size/4)
    else
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM( 2, vec_size, rocm_warp_size/(vec_size/2), vec_size/2)
    else
    LAUNCH_BGMV_SHRINK_KERNELS_ROCM( 1, vec_size, rocm_warp_size/(vec_size/1), vec_size/1)

#undef CHECK_INPUT_TILEABLE_BY
#undef LAUNCH_BGMV_SHRINK_KERNELS_ROCM
#endif
  }
}

#define INST_BGMV(feat_in, feat_out, in_T, out_T, W_T)                         \
  template void bgmv_kernel<feat_in, feat_out>(                                \
      out_T * __restrict__ Y, const in_T *__restrict__ X,                      \
      const W_T *__restrict__ W, const int64_t *__restrict__ indicies,         \
      int64_t y_offset, int64_t full_y_size, int64_t batch_size,               \
      int64_t num_layers, int64_t layer_idx, float scale);

#define INST_BGMV_ONESIDE(in_T, out_T, W_T, feat_in, feat_out)                 \
  INST_BGMV(feat_in, feat_out, in_T, out_T, W_T)

#define INST_BGMV_TWOSIDE(in_T, out_T, W_T, narrow, wide)                      \
  INST_BGMV(narrow, wide, in_T, out_T, W_T)                                    \
  INST_BGMV(wide, narrow, in_T, out_T, W_T)
