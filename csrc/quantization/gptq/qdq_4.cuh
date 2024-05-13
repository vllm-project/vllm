/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_4_cuh
#define _qdq_4_cuh

#include "qdq_util.cuh"

namespace vllm {
namespace gptq {
// Permutation:
//
// 77775555 33331111  66664444 22220000

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t qa0 = qa & 0x0f;
        uint32_t qa1 = (qa & 0xf0) >> 4;
        qa >>= 8;
        qb |= (qa1 << (i * 4 + 16));
        qb |= (qa0 << (i * 4));
    }
    q[0] = qb;
}

template <class D>
__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t q_0,
    typename D::T2 (&dq)[4],
    int stride,
    const uint32_t zero
);



template <class D>
__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale
(
    const uint32_t zero,
    const typename D::T scale,
    typename D::T2 (&z1z16)[2],
    typename D::T2 (&y1y16)[2]
);


template <class D>
__forceinline__ __device__ void dequant_4bit_8_prep_zero
(
    const uint32_t zero,
    typename D::T2(&z1z16)[2],
    typename D::T2(&y1y16)[2]
);




template <class D>
__forceinline__ __device__ void dequant_4bit_8_gptq
(
    const uint32_t q_0,
    typename D::T2 (&dq)[4],
    typename D::T2 (&z1z16)[2],
    typename D::T2 (&y1y16)[2],
    int stride,
    bool scaled
);



template <>
__forceinline__ __device__ void dequant_4bit_8<FP16TYPE>
(
    const uint32_t q_0,
    half2 (&dq)[4],
    int stride,
    const uint32_t zero
)
{
    using D = FP16TYPE;
    const typename D::T y16_ = D::float2num_rn(1.0f / 16.0f);
    const typename D::T2 y16 = D::nums2num2(y16_, y16_);
    const half_uint16<D> z1_(0xe400 | zero); // half(-1024.0f - zero);

    const typename D::T z16_ = D::num_sub(D::int2num_rn(-64), D::int2num_rn(zero));
    const typename D::T2 z1 = D::num2num2(z1_.as_half);
    const typename D::T2 z16 = D::num2num2(z16_);

    const uint32_t c0 = 0x64006400;
    uint32_t qa = q_0;
    half2_uint32<D> q0((qa & 0x000f000f) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32<D> q1((qa & 0x00f000f0) | c0); // half2(q[ 2], q[ 3]) * 16 + 1024
    qa >>= 8;
    half2_uint32<D> q2((qa & 0x000f000f) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32<D> q3((qa & 0x00f000f0) | c0); // half2(q[ 6], q[ 7]) * 16 + 1024

    dq[0] = D::num2_add(q0.as_half2, z1);
    dq[1] = D::num2_fma(q1.as_half2, y16, z16);
    dq[2] = D::num2_add(q2.as_half2, z1);
    dq[3] = D::num2_fma(q3.as_half2, y16, z16);
}




template <>
__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale<FP16TYPE>
(
    const uint32_t zero,
    const half scale,
    half2 (&z1z16)[2],
    half2 (&y1y16)[2]
)
{
    using D = FP16TYPE;
    half_uint16<D> z1(0xe400 | zero); // half(-1024.0f - zero);
    typename D::T z16 = D::num_sub(D::int2num_rn(-64), D::int2num_rn(zero));

    typename D::T2 scale2 = D::num2num2(scale);

    z1z16[0] = D::num2_mul(scale2, D::num2num2(z1.as_half));
    z1z16[1] = D::num2_mul(scale2, D::num2num2(z16));

    const typename D::T y1 = D::float2num_rn(1.0f);
    const typename D::T y16 = D::float2num_rn(1.0f / 16.0f);

    y1y16[0] = D::num2_mul(scale2, D::num2num2(y1));
    y1y16[1] = D::num2_mul(scale2, D::num2num2(y16));
}





template <>
__forceinline__ __device__ void dequant_4bit_8_prep_zero<FP16TYPE>
(
    const uint32_t zero,
    half2(&z1z16)[2],
    half2(&y1y16)[2]
)
{
    using D = FP16TYPE;
    half_uint16<D> z1(0xe400 | zero); // half(-1024.0f - zero);
    typename D::T z16 = D::num_sub(D::int2num_rn(-64), D::int2num_rn(zero));

    z1z16[0] = D::num2num2(z1.as_half);
    z1z16[1] = D::num2num2(z16);

    const typename D::T y1 = D::float2num_rn(1.0f);
    const typename D::T y16 = D::float2num_rn(1.0f / 16.0f);

    y1y16[0] = D::num2num2(y1);
    y1y16[1] = D::num2num2(y16);
}




template <>
__forceinline__ __device__ void dequant_4bit_8_gptq<FP16TYPE>
(
    const uint32_t q_0,
    half2 (&dq)[4],
    half2 (&z1z16)[2],
    half2 (&y1y16)[2],
    int stride,
    bool scaled
)
{   
    using D = FP16TYPE;
    const uint32_t c0 = 0x64006400;

    uint32_t qa = q_0;
    half2_uint32<D> q0((qa & 0x000f000f) | c0); // half2( q[0]      + 1024, q[1]      + 1024 )
    half2_uint32<D> q1((qa & 0x00f000f0) | c0); // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
    qa >>= 8;
    half2_uint32<D> q2((qa & 0x000f000f) | c0); // half2( q[4]      + 1024, q[5]      + 1024 )
    half2_uint32<D> q3((qa & 0x00f000f0) | c0); // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

    if (scaled)
    {
        dq[0] = D::num2_fma(q0.as_half2, y1y16[0], z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
        dq[1] = D::num2_fma(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
        dq[2] = D::num2_fma(q2.as_half2, y1y16[0], z1z16[0]);
        dq[3] = D::num2_fma(q3.as_half2, y1y16[1], z1z16[1]);
    }
    else
    {
        dq[0] = D::num2_add(q0.as_half2,           z1z16[0]);  // half2( q[0] - z, q[1] - z )
        dq[1] = D::num2_fma(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] - z, q[3] - z )
        dq[2] = D::num2_add(q2.as_half2,           z1z16[0]);  // half2( q[4] - z, q[5] - z )
        dq[3] = D::num2_fma(q3.as_half2, y1y16[1], z1z16[1]);  // half2( q[6] - z, q[7] - z )
    }
}


#if ((__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))) && !defined(USE_ROCM)

template <>
__forceinline__ __device__ void dequant_4bit_8<BF16TYPE>
(
    const uint32_t q_0,
    nv_bfloat162 (&dq)[4],
    int stride,
    const uint32_t zero
)
{
    using D = BF16TYPE;
    const half_uint16<D> z1_(0xc300 | zero); // half(-128.0f - zero);
    const typename D::T2 z1 = D::num2num2(z1_.as_half);

    const uint32_t c0 = 0x43004300;
    uint32_t qa = q_0;

    half2_uint32<D> q0((qa & 0x000f000f) | c0); // half2( q[0] + 128, q[1] + 128 )
    dq[0] = D::num2_add(q0.as_half2, z1);

    for (int i = 1; i < 4; i++) {
        qa >>= 4;
        half2_uint32<D> q1((qa & 0x000f000f) | c0); // half2( q[i * 2] + 128, q[i * 2 + 1] + 128 )
        dq[i] = D::num2_add(q0.as_half2, z1);
    }

}



template <>
__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale<BF16TYPE>
(
    const uint32_t zero,
    const nv_bfloat16 scale,
    nv_bfloat162 (&z1z16)[2],
    nv_bfloat162 (&y1y16)[2]
)
{
    using D = BF16TYPE;
    half_uint16<D> z1(0xc300 | zero); // half(-128.0f - zero);

    typename D::T2 scale2 = D::num2num2(scale);

    z1z16[0] = D::num2_mul(scale2, D::num2num2(z1.as_half));
    z1z16[1] = z1z16[0];

    const typename D::T y1 = D::float2num_rn(1.0f);

    y1y16[0] = D::num2_mul(scale2, D::num2num2(y1));
    y1y16[1] = y1y16[0];
}

template <>
__forceinline__ __device__ void dequant_4bit_8_prep_zero<BF16TYPE>
(
    const uint32_t zero,
    nv_bfloat162(&z1z16)[2],
    nv_bfloat162(&y1y16)[2]
)
{
    using D = BF16TYPE;
    half_uint16<D> z1(0xc300 | zero); // half(-128.0f - zero);

    z1z16[0] = D::num2num2(z1.as_half);
    z1z16[1] = z1z16[0];

    const typename D::T y1 = D::float2num_rn(1.0f);

    y1y16[0] = D::num2num2(y1);
    y1y16[1] = y1y16[0];
}


template <>
__forceinline__ __device__ void dequant_4bit_8_gptq<BF16TYPE>
(
    const uint32_t q_0,
    nv_bfloat162 (&dq)[4],
    nv_bfloat162 (&z1z16)[2],
    nv_bfloat162 (&y1y16)[2],
    int stride,
    bool scaled
)
{
    using D = BF16TYPE;
    const uint32_t c0 = 0x43004300;

    uint32_t qa = q_0;
    half2_uint32<D> q0((qa & 0x000f000f) | c0); // half2( q[0] + 128, q[1] + 128 )
    qa >>= 4;
    half2_uint32<D> q1((qa & 0x000f000f) | c0); // half2( q[2] + 128, q[3] + 128 )
    qa >>= 4;
    half2_uint32<D> q2((qa & 0x000f000f) | c0); // half2( q[4] + 128, q[5] + 128 )
    qa >>= 4;
    half2_uint32<D> q3((qa & 0x000f000f) | c0); // half2( q[6] + 128, q[7] + 128 )

    if (scaled)
    {
        dq[0] = D::num2_fma(q0.as_half2, y1y16[0], z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
        dq[1] = D::num2_fma(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
        dq[2] = D::num2_fma(q2.as_half2, y1y16[0], z1z16[0]);
        dq[3] = D::num2_fma(q3.as_half2, y1y16[1], z1z16[1]);
    }
    else
    {
        dq[0] = D::num2_add(q0.as_half2,           z1z16[0]);  // half2( q[0] - z, q[1] - z )
        dq[1] = D::num2_fma(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] - z, q[3] - z )
        dq[2] = D::num2_add(q2.as_half2,           z1z16[0]);  // half2( q[4] - z, q[5] - z )
        dq[3] = D::num2_fma(q3.as_half2, y1y16[1], z1z16[1]);  // half2( q[6] - z, q[7] - z )
    }
}

#endif
}  // namespace gptq
}  // namespace vllm

#endif
