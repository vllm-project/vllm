/*
Common data types and macros that are used across the kerutils library.
*/
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/config.hpp>  // For CUTE_DEVICE

namespace kerutils {

// Cache hints
enum class CacheHint {
  EVICT_FIRST,
  EVICT_NORMAL,
  EVICT_LAST,
  EVICT_UNCHANGED,
  NO_ALLOCATE
};

// Prefetch size
enum class PrefetchSize { B64, B128, B256 };

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;
using nve4m3 = __nv_fp8_e4m3;
using nve4m3x2 = __nv_fp8x2_e4m3;
using nve4m3x4 = __nv_fp8x4_e4m3;

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

}  // namespace kerutils

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  #define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  #define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
  #define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  #define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
  #define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  #define KERUTILS_ENABLE_SM80
  #define KERUTILS_ENABLE_SM90
  #define KERUTILS_ENABLE_SM90A
  #define KERUTILS_ENABLE_SM100
  #define KERUTILS_ENABLE_SM100A
#endif
