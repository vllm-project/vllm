// Adated from FasterTransformer, https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>

static inline __device__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

#if ENABLE_BF16
template<>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}

template<>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16