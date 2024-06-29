#ifndef CSRC__PUNICA__TYPE_CONVERT_H__
#define CSRC__PUNICA__TYPE_CONVERT_H__

#ifndef USE_ROCM

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#else

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define __TYPE_CONVERT__HOST_DEVICE__ __host__ __device__

typedef __half nv_half;
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

__TYPE_CONVERT__HOST_DEVICE__
inline __hip_bfloat162 make_bfloat162(__hip_bfloat16 val) {
  return __hip_bfloat162{val, val};
}

__TYPE_CONVERT__HOST_DEVICE__
inline __hip_bfloat162 make_bfloat162(__hip_bfloat16 vall, __hip_bfloat16 valr) {
  return __hip_bfloat162{vall, valr};
}

template <typename T_src, typename T_dst>
__TYPE_CONVERT__HOST_DEVICE__
inline T_dst convert_type(T_src val) {
  return static_cast<T_dst>(val);
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline float convert_type<__half, float>(__half val) {
  return __half2float(val);
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline __half convert_type<float, __half>(float val) {
  return __float2half(val);
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline float convert_type<__hip_bfloat16, float>(__hip_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline __hip_bfloat16 convert_type<float, __hip_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <typename T>
__TYPE_CONVERT__HOST_DEVICE__
inline T vllm_add(T a, T b) {
  return a + b;
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline __half vllm_add<__half>(__half a, __half b) {
  return __hadd(a, b);
}

template <>
__TYPE_CONVERT__HOST_DEVICE__
inline __hip_bfloat16 vllm_add<__hip_bfloat16>(__hip_bfloat16 a, __hip_bfloat16 b) {
  return __hadd(a, b);
}

#undef __TYPE_CONVERT__HOST_DEVICE__

#endif // USE_ROCM

#endif // CSRC__PUNICA__TYPE_CONVERT_H__
