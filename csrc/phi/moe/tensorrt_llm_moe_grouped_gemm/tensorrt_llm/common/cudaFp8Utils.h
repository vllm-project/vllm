/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FP8_MHA
#define FUSE_GEMM_ACT
#define FP8_GEMM_OUTPUT_QUANT_DISABLE

#ifdef FUSE_GEMM_ACT
#define USE_QGMMA
#endif

namespace tensorrt_llm
{
namespace common
{

constexpr float FP8_E4M3_MAX = 448.0f;

enum QuantizeMode
{
    PER_CHANNEL,
    PER_TENSOR,
    PER_CHANNEL_WEIGHT_PER_TENSOR_ACT,
    PER_TOKEN,
};

// Packed Data Type
typedef struct __CUDA_ALIGN__(32)
{
    float array[8];
} float8;

typedef struct __CUDA_ALIGN__(16)
{
    half array[8];
} half8;

typedef struct __CUDA_ALIGN__(8)
{
    half2 array[2];
} half2_2;

typedef struct __CUDA_ALIGN__(8)
{
    half array[4];
} half_4;

#ifdef ENABLE_BF16
typedef struct __CUDA_ALIGN__(4)
{
    __nv_bfloat16 array[2];
} __nv_bfloat16_2;

typedef struct __CUDA_ALIGN__(8)
{
    __nv_bfloat162 x, y;
} __nv_bfloat162_2_xy;

typedef struct __CUDA_ALIGN__(8)
{
    __nv_bfloat16 array[4];
} __nv_bfloat164;

typedef struct __CUDA_ALIGN__(8)
{
    __nv_bfloat162 array[2];
} __nv_bfloat162_2;

typedef struct __CUDA_ALIGN__(16)
{
    __nv_bfloat16 array[8];
} __nv_bfloat168;

typedef struct __CUDA_ALIGN__(16)
{
    __nv_bfloat162 array[4];
} __nv_bfloat162_4;

typedef struct __CUDA_ALIGN__(32)
{
    __nv_bfloat16 array[16];
} __nv_bfloat1616;
#endif

#ifdef ENABLE_FP8
typedef struct __CUDA_ALIGN__(2)
{
    __nv_fp8_e4m3 array[2];
} __nv_fp8_2_e4m3;

typedef struct __CUDA_ALIGN__(4)
{
    __nv_fp8_e4m3 array[4];
} __nv_fp8_4_e4m3;

typedef struct __CUDA_ALIGN__(4)
{
    __nv_fp8x2_e4m3 array[2];
} __nv_fp8x2_x2_e4m3;

typedef struct __CUDA_ALIGN__(8)
{
    __nv_fp8_e4m3 array[8];
} __nv_fp8_8_e4m3;

typedef struct __CUDA_ALIGN__(8)
{
    __nv_fp8x2_e4m3 array[4];
} __nv_fp8x2_x4_e4m3;

typedef struct __CUDA_ALIGN__(16)
{
    __nv_fp8_e4m3 array[16];
} __nv_fp8x16_e4m3;
#endif

// only BF16 and FP8
template <typename T, int PACK_SIZE>
struct PackType
{
    using type = float;
};

#ifdef ENABLE_BF16
template <>
struct PackType<__nv_bfloat16, 2>
{
    using type = __nv_bfloat16_2;
};

template <>
struct PackType<__nv_bfloat16, 4>
{
    using type = __nv_bfloat164;
};

template <>
struct PackType<__nv_bfloat16, 8>
{
    using type = __nv_bfloat168;
};
#endif

#ifdef ENABLE_FP8
template <>
struct PackType<__nv_fp8_e4m3, 2>
{
    using type = __nv_fp8_2_e4m3;
};

template <>
struct PackType<__nv_fp8_e4m3, 4>
{
    using type = __nv_fp8_4_e4m3;
};

template <>
struct PackType<__nv_fp8_e4m3, 8>
{
    using type = __nv_fp8_8_e4m3;
};
#endif

__inline__ __device__ void fp8x4_e4m3_to_bfloat2(__nv_bfloat162* out1, __nv_bfloat162* out2, __nv_fp8x4_e4m3 const* in)
{
    const char4 tmp_val = reinterpret_cast<char4 const*>(in)[0];
    *out1 = __nv_bfloat162((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    *out2 = __nv_bfloat162((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.z)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.w)[0]);
}

__inline__ __device__ __nv_bfloat162 fp8x2_e4m3_to_bfloat2(__nv_fp8x2_e4m3 const* in)
{
    const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
    __nv_bfloat162 out = __nv_bfloat162((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    return out;
}

__inline__ __device__ void fp8x4_e4m3_to_half2(half2* out1, half2* out2, __nv_fp8x4_e4m3 const* in)
{
    const char4 tmp_val = reinterpret_cast<char4 const*>(in)[0];
    *out1 = half2((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    *out2 = half2((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.z)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.w)[0]);
}

__inline__ __device__ half2 fp8x2_e4m3_to_half2(__nv_fp8x2_e4m3 const* in)
{
    const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
    half2 out = half2((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    return out;
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(T_OUT* output, T_S const* input_qua_amax_ptr, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream);

template <typename T_OUT, typename T_S, typename T_IN>
void invokeDequantizeMatrix(T_OUT* output, T_S const* input_qua_amax_ptr, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream);

template <typename T_FAKE, typename T_OUT, typename T_IN>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel, cudaStream_t stream);

template <typename T_S, typename T_W>
void invokeComputeFP8QuantizeScale(T_S* quant_ptr, const T_W* weights, const int64_t k, const int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream);

template <typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrix(T_OUT* output, T_S* quant_ptr, const T_IN* weights, const int64_t numel,
    const int64_t lda, QuantizeMode quantize_mode, cudaStream_t stream);

} // namespace common
} // namespace tensorrt_llm
#endif // ENABLE_FP8
