#pragma once

#include <cutlass/detail/helper_macros.hpp>

#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
#define DG_IN_CUDA_COMPILATION
#endif

#if defined(__NVCC__) || (defined(__clang__) and defined(__CUDA__))
#define CUTLASS_HOST_DEVICE_NOINLINE  __device__ __host__
#define CUTLASS_DEVICE_NOINLINE __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE_NOINLINE __device__
#define CUTLASS_DEVICE_NOINLINE __device__
#else
#define CUTLASS_HOST_DEVICE_NOINLINE
#define CUTLASS_DEVICE_NOINLINE
#endif
