
#ifndef _data_types_cuh
#define _data_types_cuh
#include <cuda_fp16.h>


#if ((__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))) && !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif


class FP16TYPE {
public:
    using T = half;
    using T2 = half2;

    static __device__ float inline num2float(const half x) { return __half2float(x); }

    static __device__ half2 inline num2num2(const half x) { return __half2half2(x); }

    static __device__ half inline low2num(const half2 x) { return __low2half(x); }

    static __device__ half inline high2num(const half2 x) { return __high2half(x); }

    static __device__ float inline low2float(const half2 x) { return __low2float(x); }

    static __device__ float inline high2float(const half2 x) { return __high2float(x); }

    static __device__ half2 inline nums2num2(const half x1, const half x2) { return __halves2half2(x1, x2); }

    static __device__ half2 inline num2_fma(const half2 x1, const half2 x2, const half2 x3) { return __hfma2(x1, x2, x3); }

    static __device__ half inline high2float(const half x1, const half x2, const half x3) { return __hfma(x1, x2, x3); }

    static __device__ half2 inline num2_add(const half2 x1, const half2 x2) { return __hadd2(x1, x2); }

    static __device__ half inline num_add(const half x1, const half x2) { return __hadd(x1, x2); }

    static __device__ half2 inline num2_mul(const half2 x1, const half2 x2) { return __hmul2(x1, x2); }

    static __device__ half inline num_mul(const half x1, const half x2) { return __hmul(x1, x2); }

    static __device__ half2 inline num2_sub(const half2 x1, const half2 x2) { return __hsub2(x1, x2); }

    static __device__ half inline num_sub(const half x1, const half x2) { return __hsub(x1, x2); }

    static __device__ ushort inline num_as_ushort(const half x) { return __half_as_ushort(x); }

    static __device__ half inline int2num_rn(unsigned int x) { return __int2half_rn(x); }

    static __device__ half inline int2num_rn(int x) { return __int2half_rn(x); }

    static __device__ half inline float2num_rn(const float x) { return __float2half_rn(x); }

    static __host__ __device__ half inline float2num(const float x) { return __float2half(x); }
};

#if ((__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))) && !defined(USE_ROCM)

class BF16TYPE {
public:
    using T = nv_bfloat16;
    using T2 = nv_bfloat162;

    static __device__ float inline num2float(const nv_bfloat16 x) { return __bfloat162float(x); }

    static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) { return __bfloat162bfloat162(x); }

    static __device__ nv_bfloat16 inline low2num(const nv_bfloat162 x) { return __low2bfloat16(x); }

    static __device__ nv_bfloat16 inline high2num(const nv_bfloat162 x) { return __high2bfloat16(x); }

    static __device__ float inline low2float(const nv_bfloat162 x) { return __low2float(x); }

    static __device__ float inline high2float(const nv_bfloat162 x) { return __high2float(x); }

    static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1, const nv_bfloat16 x2) { return __halves2bfloat162(x1, x2); }

    static __device__ nv_bfloat162 inline num2_fma(const nv_bfloat162 x1, const nv_bfloat162 x2, const nv_bfloat162 x3) { return __hfma2(x1, x2, x3); }

    static __device__ nv_bfloat16 inline high2float(const nv_bfloat16 x1, const nv_bfloat16 x2, const nv_bfloat16 x3) { return __hfma(x1, x2, x3); }

    static __device__ nv_bfloat162 inline num2_add(const nv_bfloat162 x1, const nv_bfloat162 x2) { return __hadd2(x1, x2); }

    static __device__ nv_bfloat16 inline num_add(const nv_bfloat16 x1, const nv_bfloat16 x2) { return __hadd(x1, x2); }

    static __device__ nv_bfloat162 inline num2_mul(const nv_bfloat162 x1, const nv_bfloat162 x2) { return __hmul2(x1, x2); }

    static __device__ nv_bfloat16 inline num_mul(const nv_bfloat16 x1, const nv_bfloat16 x2) { return __hmul(x1, x2); }

    static __device__ nv_bfloat162 inline num2_sub(const nv_bfloat162 x1, const nv_bfloat162 x2) { return __hsub2(x1, x2); }

    static __device__ nv_bfloat16 inline num_sub(const nv_bfloat16 x1, const nv_bfloat16 x2) { return __hsub(x1, x2); }

    static __device__ ushort inline num_as_ushort(const nv_bfloat16 x) { return __bfloat16_as_ushort(x); }

    static __device__ nv_bfloat16 inline int2num_rn(unsigned int x) { return __int2bfloat16_rn(x); }

    static __device__ nv_bfloat16 inline int2num_rn(int x) { return __int2bfloat16_rn(x); }

    static __device__ nv_bfloat16 inline float2num_rn(const float x) { return __float2bfloat16_rn(x); }

    static __host__ __device__ nv_bfloat16 inline float2num(const float x) { return __float2bfloat16(x); }
};

#endif

#endif
