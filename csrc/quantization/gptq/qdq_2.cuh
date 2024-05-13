/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_2_cuh
#define _qdq_2_cuh

#include "qdq_util.cuh"

namespace vllm {
namespace gptq {

// Permutation:
//
// ffddbb99 77553311  eeccaa88 66442200

__forceinline__ __device__ void shuffle_2bit_16
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint32_t qa0 = qa & 0x03;
        uint32_t qa1 = (qa & 0x0c) >> 2;
        qa >>= 4;
        qb |= (qa1 << (i * 2 + 16));
        qb |= (qa0 << (i * 2));
    }
    q[0] = qb;
}

template <class D>
__forceinline__ __device__ void dequant_2bit_16
(
    const uint32_t q_0,
    typename D::T2 (&dq)[8],
    int stride,
    const uint32_t zero
);



template <>
__forceinline__ __device__ void dequant_2bit_16<FP16TYPE>
(
    const uint32_t q_0,
    half2 (&dq)[8],
    int stride,
    const uint32_t zero
)
{
    using D = FP16TYPE;
    const uint32_t c0 = 0x64006400;
    const typename D::T y4_  = D::float2num_rn(1.0f /  4.0f);
    const typename D::T y16_ = D::float2num_rn(1.0f / 16.0f);
    const typename D::T y64_ = D::float2num_rn(1.0f / 64.0f);
    const typename D::T2 y4  = D::nums2num2(y4_,  y4_);
    const typename D::T2 y16 = D::nums2num2(y16_, y16_);
    const typename D::T2 y64 = D::nums2num2(y64_, y64_);

    const half_uint16<D> z1_(0xe400 | zero); // half(-1024.0f - zero);
    const typename D::T z4_ = D::num_sub(D::int2num_rn(-256), D::int2num_rn(zero));
    const typename D::T z16_ = D::num_sub(D::int2num_rn(-64), D::int2num_rn(zero));
    const typename D::T z64_ = D::num_sub(D::int2num_rn(-16), D::int2num_rn(zero));
    const typename D::T2 z1 = D::num2num2(z1_.as_half);
    const typename D::T2 z4 = D::num2num2(z4_);
    const typename D::T2 z16 = D::num2num2(z16_);
    const typename D::T2 z64 = D::num2num2(z64_);

    uint32_t qa = q_0;
    half2_uint32<D> q0((qa & 0x00030003) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32<D> q1((qa & 0x000c000c) | c0); // half2(q[ 2], q[ 3]) *  4 + 1024
    half2_uint32<D> q2((qa & 0x00300030) | c0); // half2(q[ 4], q[ 5]) * 16 + 1024
    half2_uint32<D> q3((qa & 0x00c000c0) | c0); // half2(q[ 6], q[ 7]) * 64 + 1024
    qa >>= 8;
    half2_uint32<D> q4((qa & 0x00030003) | c0); // half2(q[ 8], q[ 9])      + 1024
    half2_uint32<D> q5((qa & 0x000c000c) | c0); // half2(q[10], q[11]) *  4 + 1024
    half2_uint32<D> q6((qa & 0x00300030) | c0); // half2(q[12], q[13]) * 16 + 1024
    half2_uint32<D> q7((qa & 0x00c000c0) | c0); // half2(q[14], q[15]) * 64 + 1024

    dq[0] = D::num2_add(q0.as_half2, z1);
    dq[1] = D::num2_fma(q1.as_half2, y4,  z4);
    dq[2] = D::num2_fma(q2.as_half2, y16, z16);
    dq[3] = D::num2_fma(q3.as_half2, y64, z64);
    dq[4] = D::num2_add(q4.as_half2, z1);
    dq[5] = D::num2_fma(q5.as_half2, y4,  z4);
    dq[6] = D::num2_fma(q6.as_half2, y16, z16);
    dq[7] = D::num2_fma(q7.as_half2, y64, z64);
}


#if ((__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))) && !defined(USE_ROCM)
template <>
__forceinline__ __device__ void dequant_2bit_16<BF16TYPE>
(
    const uint32_t q_0,
    nv_bfloat162 (&dq)[8],
    int stride,
    const uint32_t zero
)
{
    using D = BF16TYPE;
    const uint32_t c0 = 0x43004300;
    const typename D::T y4_  = D::float2num_rn(1.0f /  4.0f);
    const typename D::T y16_ = D::float2num_rn(1.0f / 16.0f);
    const typename D::T2 y4  = D::nums2num2(y4_,  y4_);
    const typename D::T2 y16 = D::nums2num2(y16_, y16_);

    const half_uint16<D> z1_(0xc300 | zero); // half(-128.0f - zero);
    const typename D::T z4_ = D::num_sub(D::int2num_rn(-32), D::int2num_rn(zero));
    const typename D::T z16_ = D::num_sub(D::int2num_rn(-8), D::int2num_rn(zero));
    const typename D::T2 z1 = D::num2num2(z1_.as_half);
    const typename D::T2 z4 = D::num2num2(z4_);
    const typename D::T2 z16 = D::num2num2(z16_);

    uint32_t qa = q_0;
    half2_uint32<D> q0((qa & 0x00030003) | c0); // half2(q[ 0], q[ 1])      + 128
    half2_uint32<D> q1((qa & 0x000c000c) | c0); // half2(q[ 2], q[ 3]) *  4 + 128
    half2_uint32<D> q2((qa & 0x00300030) | c0); // half2(q[ 4], q[ 5]) * 16 + 128
    qa >>= 6;
    half2_uint32<D> q3((qa & 0x00030003) | c0); // half2(q[ 6], q[ 7])      + 128
    half2_uint32<D> q4((qa & 0x000c000c) | c0); // half2(q[ 8], q[ 9]) *  4 + 128
    half2_uint32<D> q5((qa & 0x00300030) | c0); // half2(q[10], q[11]) * 16 + 128
    qa >>= 6;
    half2_uint32<D> q6((qa & 0x00030003) | c0); // half2(q[12], q[13])      + 128
    half2_uint32<D> q7((qa & 0x000c000c) | c0); // half2(q[14], q[15]) *  4 + 128

    dq[0] = D::num2_add(q0.as_half2, z1);
    dq[1] = D::num2_fma(q1.as_half2, y4,  z4);
    dq[2] = D::num2_fma(q2.as_half2, y16, z16);
    dq[3] = D::num2_add(q3.as_half2, z1);
    dq[4] = D::num2_fma(q4.as_half2, y4,  z4);
    dq[5] = D::num2_fma(q5.as_half2, y16, z16);
    dq[6] = D::num2_add(q6.as_half2, z1);
    dq[7] = D::num2_fma(q7.as_half2, y4,  z4);
}

#endif


}  // namespace gptq
}  // namespace vllm

#endif
