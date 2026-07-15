#pragma once

#include <cuda/std/cstdint>
#include <deep_gemm/common/compile.cuh>

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif
