/*
 * Adapted from https://github.com/microsoft/VPTQ
 * Copyright (C) 2024 wejoncy
 * Copyright (C) 2024 YangWang92
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <cstdlib>
#if defined(USE_ROCM)
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#endif
namespace vllm {
namespace vptq {
#if defined(USE_ROCM)
  #define VPTQ_LDG(arg) __ldg(arg)
  #define SHFL_DOWN(val, offset) __shfl_down(val, offset)
  #define WARP_SIZE warpSize
typedef __hip_bfloat162 __bfloat162;
typedef __hip_bfloat16 __bfloat16;
#else
  #define WARP_SIZE 32
  #define VPTQ_LDG(arg) *(arg)
  #define SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)
typedef __nv_bfloat162 __bfloat162;
typedef __nv_bfloat16 __bfloat16;
#endif

namespace cuda {

constexpr int kBlockSize = 256;

template <typename T>
struct TypeVec2 {
  typedef __half2 type;
};

template <>
struct TypeVec2<__bfloat16> {
  typedef __bfloat162 type;
};

template <>
struct TypeVec2<float> {
  typedef float2 type;
};

template <typename T>
T __device__ __forceinline__ ConvertFromFloat(float v, T vv) {
  (void)(vv);
  if constexpr (std::is_same<T, __bfloat16>::value) {
    return vv = __float2bfloat16(v);
  } else if constexpr (std::is_same<T, float>::value) {
    return vv = v;
  } else {
    static_assert(std::is_same<T, __half>::value);
    return vv = __float2half(v);
  }
}

template <typename T>
float __device__ __forceinline__ ConvertToFloat(T v) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
    return __bfloat162float(v);
  } else if constexpr (std::is_same<T, float>::value) {
    return v;
  } else {
    static_assert(std::is_same<T, __half>::value);
    return __half2float(v);
  }
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if constexpr (WarpSize >= 64)
    sum += SHFL_DOWN(sum, 32);  // 0-16, 1-17, 2-18, etc.
  if constexpr (WarpSize >= 32)
    sum += SHFL_DOWN(sum, 16);  // 0-16, 1-17, 2-18, etc.
  if constexpr (WarpSize >= 16)
    sum += SHFL_DOWN(sum, 8);  // 0-8, 1-9, 2-10, etc.
  if constexpr (WarpSize >= 8) sum += SHFL_DOWN(sum, 4);  // 0-4, 1-5, 2-6, etc.
  if constexpr (WarpSize >= 4)
    sum += SHFL_DOWN(sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if constexpr (WarpSize >= 2) sum += SHFL_DOWN(sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}
template <int GROUPSIZE, typename T>
__device__ __forceinline__ void ldg_vec_x(
    T* __restrict__ dst_t32, const uint32_t* __restrict__ src_u32) {
  uint32_t* dst_u32 = (uint32_t*)dst_t32;
  if constexpr (std::is_same<T, float>::value ||
                std::is_same<T, float2>::value) {
    return ldg_vec_x<GROUPSIZE * 2>(dst_u32, src_u32);
  }
  int2* dst = (int2*)dst_u32;
  const int2* src = (const int2*)src_u32;
  if constexpr (GROUPSIZE == 2) {
    *dst_u32 = VPTQ_LDG(src_u32);

  } else if constexpr (GROUPSIZE == 4) {
    *dst = VPTQ_LDG(src);
  } else if constexpr (GROUPSIZE == 6) {
    dst_u32[0] = VPTQ_LDG(src_u32);
    dst_u32[1] = VPTQ_LDG(src_u32 + 1);
    dst_u32[2] = VPTQ_LDG(src_u32 + 2);
  } else if constexpr (GROUPSIZE == 8) {
    *(int4*)dst = VPTQ_LDG((const int4*)src);
  } else if constexpr (GROUPSIZE == 16) {
    *(int4*)dst = VPTQ_LDG((const int4*)src);
    *(int4*)(dst + 2) = VPTQ_LDG((const int4*)(src + 2));
  } else if constexpr (GROUPSIZE == 12) {
    if (uint64_t(src) % 16) {
      dst[0] = VPTQ_LDG(src);
      int4 b = VPTQ_LDG((const int4*)(src + 1));
      dst[1] = *((const int2*)&b);
      dst[2] = *((const int2*)&b + 1);
    } else {
      *(int4*)dst = VPTQ_LDG((int4*)(src));
      dst[2] = VPTQ_LDG((src + 2));
    }

  } else if constexpr (GROUPSIZE == 24) {
    *((int4*)(dst)) = VPTQ_LDG((const int4*)(src));
    *(((int4*)(dst)) + 1) = VPTQ_LDG(((const int4*)(src)) + 1);
    *(((int4*)(dst)) + 2) = VPTQ_LDG(((const int4*)(src)) + 2);
  } else if constexpr (GROUPSIZE == 32) {
    *((int4*)(dst)) = VPTQ_LDG((const int4*)(src));
    *(((int4*)(dst)) + 1) = VPTQ_LDG(((const int4*)(src)) + 1);
    *(((int4*)(dst)) + 2) = VPTQ_LDG(((const int4*)(src)) + 2);
    *(((int4*)(dst)) + 3) = VPTQ_LDG(((const int4*)(src)) + 3);
  } else {
    assert(false);
  }
}

template <int WBITS>
__device__ __forceinline__ uint32_t iterator_packed_tensor(const uint32_t* ptr,
                                                           int idx) {
  if constexpr (WBITS == 32) {
    return ptr[idx];
  } else if constexpr (WBITS == 16) {
    return ((const uint16_t*)ptr)[idx];
  } else {
    int start_bits = idx * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;
    uint32_t v = (ptr[first] >> (start_bits)) & (uint32_t(1 << WBITS) - 1);
    if (first == second || end_bits == 0) {
      return v;
    } else {
      // second position might be out of bound
      uint32_t sec_v = ptr[second];
      v |= ((sec_v) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
      return v;
    }
  }
}

template <typename T>
__forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace cuda

template <typename T>
T __device__ __forceinline__ FMA2(T a, T b, T c) {
  if constexpr (std::is_same<T, __bfloat162>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    float x =
        __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(c.x);
    float y =
        __bfloat162float(a.y) * __bfloat162float(b.y) + __bfloat162float(c.y);
    return __bfloat162{__float2bfloat16(x), __float2bfloat16(y)};
#else
    return __hfma2(a, b, c);
#endif
  } else if constexpr (std::is_same<T, float2>::value) {
    return float2{a.x * b.x + c.x, a.y * b.y + c.y};
  } else {
    return __hfma2(a, b, c);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
T __device__ __forceinline__ FMA(T a, T b, T c) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    float x = __bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c);
    return __bfloat16{__float2bfloat16(x)};
#else
    return __hfma(a, b, c);
#endif
  } else if constexpr (std::is_same<T, float>::value) {
    return float{a.x * b.x + c.x};
  } else {
    return __hfma(a, b, c);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
T __device__ __forceinline__ ADD2(T a, T b) {
  if constexpr (std::is_same<T, __bfloat162>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(USE_ROCM)
    float x = __bfloat162float(a.x) + __bfloat162float(b.x);
    float y = __bfloat162float(a.y) + __bfloat162float(b.y);
    return __bfloat162{__float2bfloat16(x), __float2bfloat16(y)};
#else
    return __hadd2(a, b);
#endif
  } else if constexpr (std::is_same<T, float2>::value) {
    return float2{a.x + b.x, a.y + b.y};
  } else {
    return __hadd2(a, b);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
T __device__ __forceinline__ ZERO_VALUE(T a) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
#if defined(USE_ROCM)
    return __float2bfloat16(0.0f);
#else
    return __float2bfloat16_rn(0.0f);
#endif
  } else if constexpr (std::is_same<T, float>::value) {
    return 0.0f;
  } else {
    return __float2half(0.0f);
  }
}

#if defined(USE_ROCM)
__device__ __half operator+(const __half& a, const __half& b) {
  return __hadd(a,
                b);  // Use HIP's intrinsic __hadd for half-precision addition
}

// Overload the * operator for __half
__device__ __half operator*(const __half& a, const __half& b) {
  return __hmul(
      a, b);  // Use HIP's intrinsic __hmul for half-precision multiplication
}

#endif

template <typename T>
struct C10ToNvType {
  typedef __bfloat16 type;
};

template <>
struct C10ToNvType<c10::Half> {
  typedef __half type;
};

template <>
struct C10ToNvType<float> {
  typedef float type;
};

template <typename scalar_t, int IDXBITS, int ResidualBits, int GROUPSIZE,
          int OL_GroupSize, int Do_Reduce>
__global__ void WqA16WithOutliers_PackIndice(
    scalar_t* out, const scalar_t* input_data, const int32_t* q_indice,
    const uint16_t* q_indice_outliers, const scalar_t* __restrict__ centroids,
    const scalar_t* __restrict__ residual_centroids,
    const scalar_t* outliers_centroids, const uint16_t* invert_perm,
    const scalar_t* weight_scale, const scalar_t* weight_bias,
    const scalar_t* bias, int out_features, int in_features,
    int outliers_infeatures, const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  static_assert((GROUPSIZE & 1) == 0, "GROUPSIZE must be even ");
  int bidx = blockIdx.x;  // out_features//base_groupsize
  int bidy = blockIdx.y;  // batch
  int bidz = blockIdx.z;  // segment in_features
  int tidx = threadIdx.x;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;
  if constexpr (Do_Reduce > 0) {
    tidx += bidz * cuda::kBlockSize * Do_Reduce;
  }
  int in_y = bidx;
  __shared__ float shared_output[GROUPSIZE][cuda::kBlockSize / WARP_SIZE + 1];
  scalar_t tmp_output[GROUPSIZE];
  const scalar_t zero_value = ZERO_VALUE(scalar_t());
#pragma unroll
  for (int i = 0; i < GROUPSIZE; i++) {
    tmp_output[i] = zero_value;
  }
  input_data = input_data + in_features * bidy;
  out = out + out_features * bidy * gridDim.z;
  if (tidx >= in_features) {
    return;
  }
  scalar_t base[GROUPSIZE];
  const int inliers_infeatures_in_group =
      (in_features - outliers_infeatures) / group_nums;
  const int col_end =
      Do_Reduce ? min((bidz + 1) * cuda::kBlockSize * Do_Reduce, in_features)
                : in_features;
  for (int col = tidx; col < col_end; col += cuda::kBlockSize) {
    // const scalar_t scale = shared_w_scales[col];
    const int w_col = Do_Reduce ? (invert_perm ? invert_perm[col] : col) : 0;
    const scalar_t input_col_v = input_data[w_col];
    const scalar_t bias = input_col_v * weight_bias[w_col];
    scalar_t input_v = input_col_v * weight_scale[w_col];
    VecType input_v2 = VecType{input_v, input_v};
    VecType bias2 = VecType{bias, bias};

    int32_t mapped_index_x = col;
    if (mapped_index_x < outliers_infeatures) {
      // outliers
      constexpr int n_outlisers_groups_in_normalgroup =
          GROUPSIZE / OL_GroupSize;
#pragma unroll
      for (int i = 0; i < n_outlisers_groups_in_normalgroup; i++) {
        if (in_y * n_outlisers_groups_in_normalgroup + i >=
            out_features / OL_GroupSize)
          continue;
        const uint16_t outliers_ind =
            q_indice_outliers[(in_y * n_outlisers_groups_in_normalgroup + i) *
                                  outliers_infeatures +
                              mapped_index_x];
        const scalar_t* outliers_centroids_start =
            (outliers_centroids) + outliers_ind * OL_GroupSize;
        const int gi = i * OL_GroupSize;
        const int out_y = in_y * GROUPSIZE + gi;
        scalar_t* tmp_output_off_p = tmp_output + gi;
        scalar_t scalar_weight[OL_GroupSize];
        if (out_y < out_features) {
          cuda::ldg_vec_x<OL_GroupSize>(
              (scalar_weight), (const uint32_t*)outliers_centroids_start);
          VecType* weight_h2 = (VecType*)scalar_weight;
          VecType* tmp_output_off_h2 = (VecType*)tmp_output_off_p;
          tmp_output_off_h2[0] =
              FMA2(weight_h2[0], input_v2, tmp_output_off_h2[0]);
          tmp_output_off_h2[1] =
              FMA2(weight_h2[1], input_v2, tmp_output_off_h2[1]);
          tmp_output_off_h2[0] = ADD2(tmp_output_off_h2[0], bias2);
          tmp_output_off_h2[1] = ADD2(tmp_output_off_h2[1], bias2);
        }
      }
    } else {
      const int mapped_inliers_inx = (mapped_index_x - outliers_infeatures);
      int mappped_inx_in_a_codebook = mapped_inliers_inx;
      const scalar_t* centroids_cb = centroids;
      const scalar_t* residual_centroids_cb = residual_centroids;
      const uint32_t* q_indice_cb = (const uint32_t*)q_indice;
      if (group_nums > 1) {  // has multi-ple codebooks
        mappped_inx_in_a_codebook =
            mapped_inliers_inx % inliers_infeatures_in_group;
        const int code_books_id =
            mapped_inliers_inx / inliers_infeatures_in_group;
        q_indice_cb += code_books_id * index_stride_0;
        centroids_cb += code_books_id * centroids_stride_0;
        residual_centroids_cb += code_books_id * centroids_stride_0;
      }

      uint32_t merged_ind =
          cuda::iterator_packed_tensor<IDXBITS + ResidualBits>(
              q_indice_cb + in_y * index_stride_1, mappped_inx_in_a_codebook);
      const uint32_t base_ind = merged_ind & ((1 << IDXBITS) - 1);

      const scalar_t* centroids_start = (centroids_cb) + base_ind * GROUPSIZE;
      cuda::ldg_vec_x<GROUPSIZE>((base), (const uint32_t*)(centroids_start));

      VecType* hres_ptr = nullptr;
      if constexpr (ResidualBits > 0) {
        scalar_t residual[GROUPSIZE];
        const uint32_t res_ind =
            (merged_ind >> IDXBITS) & ((1 << ResidualBits) - 1);
        const scalar_t* residual_centroids_start =
            (residual_centroids_cb) + res_ind * GROUPSIZE;
        cuda::ldg_vec_x<GROUPSIZE>((residual),
                                   (const uint32_t*)(residual_centroids_start));

        VecType hres[GROUPSIZE / 2];
        hres_ptr = hres;
#pragma unroll
        for (int i = 0; i < GROUPSIZE / 2; i++) {
          hres[i] = ADD2(*(((VecType*)base) + i), *(((VecType*)residual) + i));
          // hres[i] = FMA2(hres[i], scale2, bias2);
        }
      } else {
        hres_ptr = (VecType*)base;
      }
      // scalar_t* res = (scalar_t*)hres;
      // #pragma unroll
      // for (int gi=0;gi<GROUPSIZE;gi++){
      //   tmp_output[gi] = __hfma(res[gi], input_v, tmp_output[gi]);
      //   tmp_output[gi] += bias;
      // }
      VecType* h2_tmp_output = (VecType*)tmp_output;
#pragma unroll
      for (int gi = 0; gi < GROUPSIZE / 2; gi++) {
        h2_tmp_output[gi] = FMA2(hres_ptr[gi], input_v2, h2_tmp_output[gi]);
        h2_tmp_output[gi] = ADD2(h2_tmp_output[gi], bias2);
      }
    }
  }

  // warp_size = WARP_SIZE
  int warpid = threadIdx.x / WARP_SIZE;  // at most 8 warp= 256/WARP_SIZE
  int landid = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int gi = 0; gi < GROUPSIZE; gi++) {
    float reduce_out = 0.f;
    reduce_out = cuda::ConvertToFloat(tmp_output[gi]);
    reduce_out = cuda::warpReduceSum<WARP_SIZE>(reduce_out);
    if (landid == 0) {
      shared_output[gi][warpid] = reduce_out;
    }
  }

  if constexpr (Do_Reduce > 0) {
    out += (in_y * GROUPSIZE) * gridDim.z + bidz;
    bias += (bias == nullptr ? 0 : (in_y * GROUPSIZE) + bidz);
  } else {
    out += in_y * GROUPSIZE;
    bias += (bias == nullptr ? 0 : GROUPSIZE);
  }

  __syncthreads();
  if (landid < cuda::kBlockSize / WARP_SIZE) {
#pragma unroll
    for (int wid = warpid; wid < GROUPSIZE;
         wid += cuda::kBlockSize / WARP_SIZE) {
      float reduce_out = shared_output[wid][landid];
      reduce_out =
          cuda::warpReduceSum<cuda::kBlockSize / WARP_SIZE>(reduce_out);
      if (landid == 0 && (in_y * GROUPSIZE + wid) < out_features) {
        if constexpr (Do_Reduce) {
          out[(wid)*gridDim.z] =
              cuda::ConvertFromFloat<scalar_t>(reduce_out, zero_value) +
              ((bidz == 0 && bias != 0) ? bias[wid] : zero_value);
        } else {
          out[wid] = cuda::ConvertFromFloat<scalar_t>(reduce_out, zero_value) +
                     ((bias != 0) ? bias[wid] : zero_value);
        }
      }
    }
  }
}

template <typename scalar_t, int IDXBITS, int ResidualBits, int GROUPSIZE,
          bool Return_OUF_x_INF>
__global__ void DequantizeWithOutliers_PackIndice(
    scalar_t* out, const int32_t* q_indice, const int16_t* q_indice_outliers,
    const scalar_t* centroids, const scalar_t* residual_centroids,
    const scalar_t* outliers_centroids, const uint16_t* invert_perm,
    const scalar_t* weight_scale, const scalar_t* weight_bias, int out_features,
    int in_features, int outliers_infeatures, int OL_GroupSize,
    const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  int bid = blockIdx.x;
  int tid = (bid * cuda::kBlockSize + threadIdx.x);
  int in_x = tid % in_features;
  int in_y = tid / in_features;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;

  uint16_t mapped_index_x = invert_perm ? invert_perm[in_x] : in_x;
  const scalar_t scale = weight_scale[in_x];
  const scalar_t bias = weight_bias[in_x];

  if (mapped_index_x < outliers_infeatures) {
    const int n_outlisers_groups_in_normalgroup = GROUPSIZE / OL_GroupSize;
    q_indice_outliers +=
        in_y * n_outlisers_groups_in_normalgroup * outliers_infeatures +
        mapped_index_x;
#pragma unroll(3)
    for (int i = 0; i < n_outlisers_groups_in_normalgroup; i++) {
      if (in_y * n_outlisers_groups_in_normalgroup + i >=
          out_features / OL_GroupSize)
        return;
      // const uint16_t outliers_ind =
      // q_indice_outliers[(in_y*n_outlisers_groups_in_normalgroup+i)*outliers_infeatures+mapped_index_x];
      const uint16_t outliers_ind = q_indice_outliers[(i)*outliers_infeatures];
      const scalar_t* outliers_centroids_start =
          outliers_centroids + outliers_ind * OL_GroupSize;
      const int gi = in_y * GROUPSIZE + i * OL_GroupSize;
#pragma unroll(4)
      for (int j = 0; j < OL_GroupSize; j++) {
        if ((gi + j) >= out_features) {
          return;
        }
        out[(gi + j) * in_features + in_x] =
            FMA(outliers_centroids_start[j], scale, bias);
      }
    }
    return;
  }

  const int inliers_infeatures_in_group =
      (in_features - outliers_infeatures) / group_nums;

  const int mapped_inliers_inx = (mapped_index_x - outliers_infeatures);
  const int code_books_id = mapped_inliers_inx / inliers_infeatures_in_group;
  const int mappped_inx_in_a_codebook =
      mapped_inliers_inx % inliers_infeatures_in_group;

  if (group_nums > 1) {  // has multi-ple codebooks
    q_indice += code_books_id * index_stride_0;
    centroids += code_books_id * centroids_stride_0;
    residual_centroids += code_books_id * centroids_stride_0;
  }
  q_indice += in_y * index_stride_1;
  uint32_t merged_ind = cuda::iterator_packed_tensor<IDXBITS + ResidualBits>(
      (const uint32_t*)q_indice, mappped_inx_in_a_codebook);

  const uint16_t base_ind = merged_ind & ((1 << IDXBITS) - 1);
  VecType base[GROUPSIZE / 2];
  const scalar_t* centroids_start = centroids + base_ind * GROUPSIZE;
  cuda::ldg_vec_x<GROUPSIZE>((base), (const uint32_t*)(centroids_start));

  if constexpr (ResidualBits > 0) {
    VecType residual[GROUPSIZE / 2];
    merged_ind >>= IDXBITS;
    const uint16_t res_ind = merged_ind & ((1 << ResidualBits) - 1);
    const scalar_t* residual_centroids_start =
        residual_centroids + res_ind * GROUPSIZE;
    cuda::ldg_vec_x<GROUPSIZE>((residual),
                               (const uint32_t*)(residual_centroids_start));
#pragma unroll
    for (int i = 0; i < GROUPSIZE / 2; i++) {
      base[i] = ADD2(*(((VecType*)base) + i), *(((VecType*)residual) + i));
    }
  }

  VecType hres[GROUPSIZE / 2];
  VecType scale2 = VecType{scale, scale};
  VecType bias2 = VecType{bias, bias};
#pragma unroll
  for (int i = 0; i < GROUPSIZE / 2; i++) {
    hres[i] = FMA2(base[i], scale2, bias2);
  }
  scalar_t* res = (scalar_t*)hres;
  const int group_step = in_y * GROUPSIZE;
  if constexpr (!Return_OUF_x_INF) {
    out += in_x * out_features + group_step;
  } else {
    out += (group_step)*in_features + in_x;
  }
#pragma unroll
  for (int i = 0; i < GROUPSIZE; i++) {
    if ((group_step + i) < out_features) {
      if constexpr (Return_OUF_x_INF) {
        out[i * in_features] = res[i];
      } else {
        out[i] = res[i];
      }
    }
  }
}

torch::Tensor lauch_deqantize_outliers_cuda_packkernel(
    const int* outf_x_inf,           //[out_f, in_f]
    const torch::Tensor& q_indice,   //[num_cen, o_c_size, in_inf]
    const torch::Tensor& centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>&
        q_indice_residual,  //[num_cen, o_c_size, in_inf]
    const c10::optional<torch::Tensor>&
        residual_centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>&
        outliers_indices,  //[num_cen, c_size, ol_in_f]
    const c10::optional<torch::Tensor>&
        outliers_centroids,  //[num_c, c_size, out_vec_len]
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_indice));

  int base_groupsize = centroids.size(-1);  // how many elements in a vector
  int res_groupsize =
      residual_centroids.has_value() ? residual_centroids.value().size(-1) : 0;
  TORCH_CHECK(((res_groupsize == base_groupsize) || (res_groupsize == 0)),
              "res_groupsize==base_groupsize is false, must be true");
  int index_bits =
      log2(centroids.size(1));  // how many bits to index quantization vector
  int res_index_bits = residual_centroids.has_value()
                           ? log2(residual_centroids.value().size(1))
                           : 0;
  auto out_size = outf_x_inf;
  dim3 blocks(cuda::ceil_div<int>(
      cuda::ceil_div<int>(out_size[0], base_groupsize) * out_size[1],
      cuda::kBlockSize));
  dim3 threads(cuda::kBlockSize);
  torch::Tensor output;
  constexpr bool out_ouf_inf = true;  // why =false is 10 times slow?
  if (out_ouf_inf) {                  // out_ouf_inf
    output = at::empty({out_size[0], out_size[1]}, centroids.options());
  } else {
    output = at::empty({out_size[1], out_size[0]}, centroids.options());
  }
  int outliers_indices_size_n1 =
      outliers_indices.has_value() ? outliers_indices.value().size(-1) : 0;
  int outliers_centroids_size_n1 =
      outliers_centroids.has_value() ? outliers_centroids.value().size(-1) : 1;

  const uint16_t* perm_ptr =
      perm.has_value() ? (const uint16_t*)(perm.value().data_ptr<int16_t>())
                       : nullptr;
  const int16_t* outliers_indices_ptr =
      outliers_indices.has_value()
          ? outliers_indices.value().data_ptr<int16_t>()
          : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
#define callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF,    \
                                ResidualBits)                                 \
  {                                                                           \
    using nv_type = typename C10ToNvType<scalar_t>::type;                     \
    DequantizeWithOutliers_PackIndice<nv_type, IDXBITS, ResidualBits,         \
                                      BASEGROUP, OUT_OUF_INF>                 \
        <<<blocks, threads, 0, stream>>>(                                     \
            reinterpret_cast<nv_type*>(output.data_ptr<scalar_t>()),          \
            q_indice.data_ptr<int32_t>(), outliers_indices_ptr,               \
            reinterpret_cast<const nv_type*>(centroids.data_ptr<scalar_t>()), \
            residual_centroids.has_value()                                    \
                ? reinterpret_cast<const nv_type*>(                           \
                      residual_centroids.value().data_ptr<scalar_t>())        \
                : nullptr,                                                    \
            outliers_centroids.has_value()                                    \
                ? reinterpret_cast<const nv_type*>(                           \
                      outliers_centroids.value().data_ptr<scalar_t>())        \
                : nullptr,                                                    \
            perm_ptr,                                                         \
            reinterpret_cast<const nv_type*>(                                 \
                weight_scale.data_ptr<scalar_t>()),                           \
            reinterpret_cast<const nv_type*>(                                 \
                weight_bias.data_ptr<scalar_t>()),                            \
            out_size[0], out_size[1], outliers_indices_size_n1,               \
            outliers_centroids_size_n1, q_indice.stride(0),                   \
            q_indice.stride(1), centroids.stride(0), q_indice.size(0));       \
  }

#define callDequantWithOutliers_dtype(IDXBITS, BASEGROUP, OUT_OUF_INF, \
                                      ResidualBits)                    \
  if (centroids.dtype() == at::ScalarType::Half) {                     \
    using scalar_t = c10::Half;                                        \
    callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF, \
                            ResidualBits);                             \
  } else {                                                             \
    using scalar_t = c10::BFloat16;                                    \
    callDequantWithOutliers(scalar_t, IDXBITS, BASEGROUP, OUT_OUF_INF, \
                            ResidualBits);                             \
  }

#define callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, ResidualBits)     \
  switch (index_bits) {                                                        \
    case 16:                                                                   \
      callDequantWithOutliers_dtype(16, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 15:                                                                   \
      callDequantWithOutliers_dtype(15, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 14:                                                                   \
      callDequantWithOutliers_dtype(14, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 13:                                                                   \
      callDequantWithOutliers_dtype(13, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 12:                                                                   \
      callDequantWithOutliers_dtype(12, BASEGROUP, OUT_OUF_INF, ResidualBits); \
      break;                                                                   \
    case 8:                                                                    \
      callDequantWithOutliers_dtype(8, BASEGROUP, OUT_OUF_INF, ResidualBits);  \
      break;                                                                   \
    case 4:                                                                    \
      callDequantWithOutliers_dtype(4, BASEGROUP, OUT_OUF_INF, ResidualBits);  \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "un-supported index_bits:" + std::to_string(index_bits));    \
  }
#define CASE_callDequantWithOutliers_bits(rib)                 \
  case rib: {                                                  \
    callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, rib); \
    break;                                                     \
  }
#define DispatchDequantWithOutliers(BASEGROUP, OUT_OUF_INF)     \
  switch (res_index_bits) {                                     \
    case 16:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 16); \
      break;                                                    \
    case 15:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 15); \
      break;                                                    \
    case 12:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 12); \
      break;                                                    \
    case 11:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 11); \
      break;                                                    \
    case 10:                                                    \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 10); \
      break;                                                    \
    case 9:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 9);  \
      break;                                                    \
    case 8:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 8);  \
      break;                                                    \
    case 7:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 7);  \
      break;                                                    \
    case 6:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 6);  \
      break;                                                    \
    case 5:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 5);  \
      break;                                                    \
    case 4:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 4);  \
      break;                                                    \
    case 3:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 3);  \
      break;                                                    \
    case 2:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 2);  \
      break;                                                    \
    case 0:                                                     \
      callDequantWithOutliers_bits(BASEGROUP, OUT_OUF_INF, 0);  \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "un-supported res_index_bits:" +       \
                             std::to_string(res_index_bits));   \
  }

#define CASE_DispatchDequantWithOutliers(bgsize)      \
  case bgsize: {                                      \
    DispatchDequantWithOutliers(bgsize, out_ouf_inf); \
    break;                                            \
  }
  switch (base_groupsize) {
    CASE_DispatchDequantWithOutliers(16);
    CASE_DispatchDequantWithOutliers(12);
    CASE_DispatchDequantWithOutliers(8);
    CASE_DispatchDequantWithOutliers(6);
    CASE_DispatchDequantWithOutliers(4);
    CASE_DispatchDequantWithOutliers(2);
    default:
      TORCH_CHECK(false, "un-supported base_groupsize:" +
                             std::to_string(base_groupsize));
  }
#undef CASE_DispatchDequantWithOutliers
  if (out_ouf_inf) {
    return output;
  } else {
    return output.t();
  }
}

torch::Tensor lauch_gemv_outliers_cuda_packkernel(
    const int out_features, const torch::Tensor& input,
    const torch::Tensor& q_indice,   //[num_cen, o_c_size, in_inf]
    const torch::Tensor& centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>&
        q_indice_residual,  //[num_cen, o_c_size, in_inf]
    const c10::optional<torch::Tensor>&
        residual_centroids,  //[num_c, c_size, vec_len]
    const c10::optional<torch::Tensor>&
        outliers_indices,  //[num_cen, c_size, ol_in_f]
    const c10::optional<torch::Tensor>&
        outliers_centroids,  //[num_c, c_size, out_vec_len]
    const c10::optional<torch::Tensor>& perm, const torch::Tensor& weight_scale,
    const torch::Tensor& weight_bias,
    const c10::optional<torch::Tensor>& bias) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  const int base_groupsize = centroids.size(-1);
  int index_bits = log2(centroids.size(1));
  int res_index_bits = residual_centroids.has_value()
                           ? log2(residual_centroids.value().size(1))
                           : 0;

  const int in_features = input.size(-1);
  // const int out_features = output.size(-1);
  auto output_shape = input.sizes().vec();
  output_shape[input.dim() - 1] = out_features;
  torch::Tensor output;
  //  blocks = (out_features, batch)
  dim3 blocks(cuda::ceil_div(out_features, base_groupsize),
              input.numel() / in_features);
  dim3 threads(cuda::kBlockSize);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // using scalar_t = c10::Half;
  // c10::BFloat16
  int shared_memory_size = 2 * in_features * 2;
  const int outliers_indices_size_n1 =
      outliers_indices.has_value() ? outliers_indices.value().size(-1) : 0;

  if (outliers_centroids.has_value()) {
    TORCH_CHECK(outliers_centroids.value().size(-1) == 4,
                "only support 4 out_vec_len");
  }
  const uint16_t* outliers_indices_ptr =
      (const uint16_t*)(outliers_indices.has_value()
                            ? outliers_indices.value().data_ptr<int16_t>()
                            : nullptr);
  const uint16_t* perm_ptr =
      perm.has_value() ? (const uint16_t*)(perm.value().data_ptr<int16_t>())
                       : nullptr;
#define CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce,      \
                        ResidualBits)                                          \
  {                                                                            \
    using nv_type = typename C10ToNvType<scalar_t>::type;                      \
    WqA16WithOutliers_PackIndice<nv_type, IDXBITS, ResidualBits, BASEGROUP, 4, \
                                 Do_Reduce>                                    \
        <<<blocks, threads, shared_memory_size, stream>>>(                     \
            reinterpret_cast<nv_type*>(out_buf.data_ptr<scalar_t>()),          \
            reinterpret_cast<const nv_type*>(input.data_ptr<scalar_t>()),      \
            q_indice.data_ptr<int32_t>(), outliers_indices_ptr,                \
            reinterpret_cast<const nv_type*>(centroids.data_ptr<scalar_t>()),  \
            residual_centroids.has_value()                                     \
                ? reinterpret_cast<const nv_type*>(                            \
                      residual_centroids.value().data_ptr<scalar_t>())         \
                : nullptr,                                                     \
            outliers_centroids.has_value()                                     \
                ? reinterpret_cast<const nv_type*>(                            \
                      outliers_centroids.value().data_ptr<scalar_t>())         \
                : nullptr,                                                     \
            perm_ptr,                                                          \
            reinterpret_cast<const nv_type*>(                                  \
                weight_scale.data_ptr<scalar_t>()),                            \
            reinterpret_cast<const nv_type*>(                                  \
                weight_bias.data_ptr<scalar_t>()),                             \
            bias.has_value() ? reinterpret_cast<const nv_type*>(               \
                                   bias.value().data_ptr<scalar_t>())          \
                             : nullptr,                                        \
            out_features, in_features, outliers_indices_size_n1,               \
            q_indice.stride(0), q_indice.stride(1), centroids.stride(0),       \
            q_indice.size(0));                                                 \
  }

#define CallWqA16kernel_dtype(out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                              ResidualBits)                           \
  if (input.dtype() == at::ScalarType::Half) {                        \
    using scalar_t = c10::Half;                                       \
    CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                    ResidualBits);                                    \
  } else {                                                            \
    using scalar_t = c10::BFloat16;                                   \
    CallWqA16kernel(scalar_t, out_buf, IDXBITS, BASEGROUP, Do_Reduce, \
                    ResidualBits);                                    \
  }

#define CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, ResidualBits)     \
  switch (index_bits) {                                                       \
    case 16:                                                                  \
      CallWqA16kernel_dtype(out_buf, 16, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 15:                                                                  \
      CallWqA16kernel_dtype(out_buf, 15, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 14:                                                                  \
      CallWqA16kernel_dtype(out_buf, 14, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 13:                                                                  \
      CallWqA16kernel_dtype(out_buf, 13, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 12:                                                                  \
      CallWqA16kernel_dtype(out_buf, 12, BASEGROUP, Do_Reduce, ResidualBits); \
      break;                                                                  \
    case 8:                                                                   \
      CallWqA16kernel_dtype(out_buf, 8, BASEGROUP, Do_Reduce, ResidualBits);  \
      break;                                                                  \
    case 4:                                                                   \
      CallWqA16kernel_dtype(out_buf, 4, BASEGROUP, Do_Reduce, ResidualBits);  \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false,                                                      \
                  "un-supported index_bits:" + std::to_string(index_bits));   \
  }

#define DispatchWqA16Kernel(out_buf, BASEGROUP, Do_Reduce)     \
  switch (res_index_bits) {                                    \
    case 16:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 16); \
      break;                                                   \
    case 15:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 15); \
      break;                                                   \
    case 12:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 12); \
      break;                                                   \
    case 11:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 11); \
      break;                                                   \
    case 10:                                                   \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 10); \
      break;                                                   \
    case 9:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 9);  \
      break;                                                   \
    case 8:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 8);  \
      break;                                                   \
    case 7:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 7);  \
      break;                                                   \
    case 6:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 6);  \
      break;                                                   \
    case 5:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 5);  \
      break;                                                   \
    case 4:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 4);  \
      break;                                                   \
    case 3:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 3);  \
      break;                                                   \
    case 2:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 2);  \
      break;                                                   \
    case 0:                                                    \
      CallWqA16kernel_bits(out_buf, BASEGROUP, Do_Reduce, 0);  \
      break;                                                   \
    default:                                                   \
      TORCH_CHECK(false, "un-supported res_index_bits:" +      \
                             std::to_string(res_index_bits));  \
  }

  if (in_features <= cuda::kBlockSize) {
    // output = at::empty(output_shape, centroids.options());
    // switch (base_groupsize){
    //   case 16:
    //       gpuErrchk(cudaFuncSetAttribute(WqA16WithOutliers<scalar_t, 16, 4,
    //       false>,
    //                 cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                 shared_memory_size));
    //       DispatchWqA16Kernel(output, 16, false);
    //   break;
    //   case 12:
    //       gpuErrchk(cudaFuncSetAttribute(WqA16WithOutliers<scalar_t, 12, 4,
    //       false>,
    //                 cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                 shared_memory_size));
    //       DispatchWqA16Kernel(output, 12, false);
    //   break;
    //   default:
    //       TORCH_CHECK(false, "un-supported
    //       base_groupsize:"+std::to_string(base_groupsize));
    // }
    TORCH_CHECK(false, "un-supported yet");
  } else {
    constexpr int do_reduce = 4;
    shared_memory_size = 0;
    auto tmp_output_shape = output_shape;
    tmp_output_shape.push_back(
        cuda::ceil_div(in_features, cuda::kBlockSize * do_reduce));
    torch::Tensor tmp_output = at::empty(tmp_output_shape, centroids.options());
    blocks.z = tmp_output_shape.back();
    switch (base_groupsize) {
      case 16:
        DispatchWqA16Kernel(tmp_output, 16, do_reduce);
        break;
      case 12:
        DispatchWqA16Kernel(tmp_output, 12, do_reduce);
        break;
      case 8:
        DispatchWqA16Kernel(tmp_output, 8, do_reduce);
        break;
      case 6:
        DispatchWqA16Kernel(tmp_output, 6, do_reduce);
        break;
      case 4:
        DispatchWqA16Kernel(tmp_output, 4, do_reduce);
        break;
      case 2:
        DispatchWqA16Kernel(tmp_output, 2, do_reduce);
        break;
      default:
        TORCH_CHECK(false,
                    "un-supported groupsize:" + std::to_string(base_groupsize));
    }
    output = tmp_output.sum(-1);
  }
  return output;
}

}  // namespace vptq
}  // namespace vllm

#define gpuErrchk(ret) gpuAssert((ret), __FILE__, __LINE__);

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    TORCH_CHECK(false, cudaGetErrorString(code));
  }
}

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor vptq_gemm(const torch::Tensor& input,
                        const torch::Tensor& q_indice,
                        const torch::Tensor& centroids,
                        const torch::Tensor& weight_scale,
                        const torch::Tensor& weight_bias,
                        const std::vector<int64_t>& g_i_o,
                        const c10::optional<torch::Tensor>& q_indice_residual,
                        const c10::optional<torch::Tensor>& residual_centroids,
                        const c10::optional<torch::Tensor>& q_indice_outliers,
                        const c10::optional<torch::Tensor>& outliers_centroids,
                        const c10::optional<torch::Tensor>& invperm,
                        const c10::optional<torch::Tensor>& bias) {
  int groupsize = g_i_o[0];
  int in_features = g_i_o[1];
  int out_features = g_i_o[2];
  CHECK_CONTIGUOUS(q_indice);
  CHECK_CONTIGUOUS(input);
  if (q_indice_residual.has_value()) {
    CHECK_CONTIGUOUS(q_indice_residual.value());
    CHECK_CONTIGUOUS(residual_centroids.value());
  }
  TORCH_CHECK(q_indice.dtype() == torch::kInt, "q_indice must be int");

  CHECK_CONTIGUOUS(centroids);
  auto dev_index = q_indice.device().index();
  auto inv_perm_device_index = dev_index;
  if (invperm.has_value()) {
    CHECK_CONTIGUOUS(invperm.value());
    inv_perm_device_index = invperm.value().device().index();
  }
  TORCH_CHECK(groupsize >= 2, "groupsize must be >= 2");
  if (q_indice_residual.has_value()) {
    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index &&
                    centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index &&
                    inv_perm_device_index == dev_index,
                "all input tensors must be on the same device");
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor output;

  output = vllm::vptq::lauch_gemv_outliers_cuda_packkernel(
      out_features, input, q_indice, centroids, q_indice_residual,
      residual_centroids, q_indice_outliers, outliers_centroids, invperm,
      weight_scale, weight_bias, bias);

  gpuErrchk(cudaPeekAtLastError());

  return output;
}

torch::Tensor vptq_dequant(
    const torch::Tensor& q_indice, const torch::Tensor& centroids,
    const torch::Tensor& weight_scale, const torch::Tensor& weight_bias,
    const std::vector<int64_t>& g_i_o,
    const c10::optional<torch::Tensor>& q_indice_residual,
    const c10::optional<torch::Tensor>& residual_centroids,
    const c10::optional<torch::Tensor>& q_indice_outliers,
    const c10::optional<torch::Tensor>& outliers_centroids,
    const c10::optional<torch::Tensor>& invperm) {
  int groupsize = g_i_o[0];
  int in_features = g_i_o[1];
  int out_features = g_i_o[2];
  auto dev_index = q_indice.device().index();

  CHECK_CONTIGUOUS(q_indice);
  if (q_indice_residual.has_value()) {
    CHECK_CONTIGUOUS(q_indice_residual.value());
    CHECK_CONTIGUOUS(residual_centroids.value());
  }

  CHECK_CONTIGUOUS(centroids);
  auto perm_dev_index = dev_index;
  if (invperm.has_value()) {
    CHECK_CONTIGUOUS(invperm.value());
    perm_dev_index = invperm.value().device().index();
  }
  CHECK_CONTIGUOUS(weight_scale);
  CHECK_CONTIGUOUS(weight_bias);
  TORCH_CHECK(q_indice.dtype() == torch::kInt, "q_indice must be int");
  TORCH_CHECK(groupsize >= 2, "groupsize must be >= 4");
  TORCH_CHECK(q_indice.dim() == 3, "must be 3D tensor");

  if (q_indice_residual.has_value()) {
    TORCH_CHECK(q_indice.size(0) == centroids.size(0) &&
                    centroids.sizes() == residual_centroids.value().sizes(),
                "size must be same");
    TORCH_CHECK(q_indice_residual.value().device().index() == dev_index &&
                    centroids.device().index() == dev_index &&
                    residual_centroids.value().device().index() == dev_index &&
                    perm_dev_index == dev_index,
                "all input tensors must be on the same device");
  }

  at::cuda::OptionalCUDAGuard guard(q_indice.device());
  torch::Tensor
      output;  // = at::empty({out_features, in_features}, centroids.options());
  const int out_f_x_in_f[2] = {out_features, in_features};

  output = vllm::vptq::lauch_deqantize_outliers_cuda_packkernel(
      out_f_x_in_f, q_indice, centroids, q_indice_residual, residual_centroids,
      q_indice_outliers, outliers_centroids, invperm, weight_scale,
      weight_bias);

  gpuErrchk(cudaPeekAtLastError());
  return output;
}
