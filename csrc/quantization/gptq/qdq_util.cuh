/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_util_cuh
#define _qdq_util_cuh

namespace vllm {
namespace gptq {

template <class D>
union half2_uint32
{
    uint32_t as_uint32;
    typename D::T2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(typename D::T2 val) : as_half2(val) {}
};

template <class D>
union half_uint16
{
    uint16_t as_uint16;
    typename D::T as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(typename D::T val) : as_half(val) {}
};

// Max_scale premultiplied by 1/256

template <class D>
__forceinline__ __device__ typename D::T dq_scale(const int qs, const typename D::T max_scale)
{
    int qs_i = qs + 1;
    typename D::T qs_h = D::int2num_rn(qs_i * qs_i);
    qs_h = D::num_mul(qs_h, max_scale);
    return qs_h;
}

template <class D>
__forceinline__ __device__ typename D::T dq(const int q, const int qzero, const typename D::T scale)
{
    return D::num_mul(D::int2num_rn(q - qzero), scale);
}

template <class D>
__forceinline__ __device__ typename D::T dq_ns(const int q, const int qzero)
{
    //return __hsub(__int2half_rn(q), __int2half_rn(qzero));
    return D::int2num_rn(q - qzero);
}

__forceinline__ __device__ int exb(const uint32_t q, const int shift, const int mask)
{
    return (int)((q >> shift) & mask);
}

__forceinline__ __device__ int exb(const uint32_t q1, const uint32_t q0, const int shift, const int mask)
{
    return (int)(__funnelshift_rc(q0, q1, shift) & mask);
}

}  // namespace gptq
}  // namespace vllm
#endif
