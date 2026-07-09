#pragma once
#define _USE_MATH_DEFINES
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDevice hipGetDevice
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceProp hipDeviceProp_t
#define cudaStream_t hipStream_t
#define cudaError hipError_t
using cudaStream_t = hipStream_t;
