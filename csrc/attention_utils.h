#pragma once

#include "cuda_primitives.h"

#include <float.h>
#include <type_traits>

#define MMHA_USE_FP32_ACUM_FOR_FMA
#define MMHA_USE_FP32_ACUM_FOR_OUT

namespace cacheflow {

// A vector type to store Q, K, V elements.
template<typename T, int VEC_SIZE>
struct Vec {};
template<>
struct Vec<float, 1> {
    using Type = float;
};
template<>
struct Vec<float, 2> {
    using Type = float2;
};
template<>
struct Vec<float, 4> {
    using Type = float4;
};
template<>
struct Vec<uint16_t, 1> {
    using Type = uint16_t;
};
template<>
struct Vec<uint16_t, 2> {
    using Type = uint32_t;
};
template<>
struct Vec<uint16_t, 4> {
    using Type = uint2;
};
template<>
struct Vec<uint16_t, 8> {
    using Type = uint4;
};

template<typename T>
struct FloatVec {};
template<>
struct FloatVec<float> {
    using Type = float;
};
template<>
struct FloatVec<float2> {
    using Type = float2;
};
template<>
struct FloatVec<float4> {
    using Type = float4;
};
template<>
struct FloatVec<uint16_t> {
    using Type = float;
};
template<>
struct FloatVec<uint32_t> {
    using Type = float2;
};
template<>
struct FloatVec<uint2> {
    using Type = Float4_;
};
template<>
struct FloatVec<uint4> {
    using Type = Float8_;
};

template<int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
    using K_vec_acum = typename FloatVec<K_vec>::Type;
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct Qk_dot {
    template<typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b)
{
    float4 c;
    float zero = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
                 "    {%0, %1, %2, %3}, \n"
                 "    {%4, %5}, \n"
                 "    {%6}, \n"
                 "    {%7, %7, %7, %7}; \n"

                 : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
                 : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using K_vec_acum = typename FloatVec<uint32_t>::Type;
    K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
    return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Qk_dot<uint16_t, 4> {
    template<int N>
    static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
        return qk_hmma_dot_(q, k);
#else
        return qk_dot_<4>(q, k);
#endif  // defined MMHA_USE_HMMA_FOR_REDUCTION
    }
};

} // namespace cacheflow

#undef MMHA_USE_FP32_ACUM_FOR_FMA
#undef MMHA_USE_FP32_ACUM_FOR_OUT
