/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

namespace vllm {
namespace gptq {
// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val) {
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    half2 old_val = *((half2*)&old);
    half2 new_val = __hadd2(old_val, val);
    old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
  } while (assumed != old);
}

//
// gfx11 (RDNA3/3.5) does not expose native atomicAdd overloads for
// half/half2 with TheRock ROCm 7.13. The CAS fallback must be visible
// on the host pass (for template instantiation) and on gfx11 device
// passes. CDNA (gfx9xx) on ROCm >= 7.13 has native overloads and is
// unaffected (the __HIP_DEVICE_COMPILE__ gate excludes it).
#if defined(__CUDA_ARCH__) || \
    (defined(USE_ROCM) && \
     (!defined(__HIP_DEVICE_COMPILE__) || \
      (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713 || \
      defined(__gfx1100__) || defined(__gfx1101__) || \
      defined(__gfx1102__) || defined(__gfx1103__) || \
      defined(__gfx1150__) || defined(__gfx1151__) || \
      defined(__gfx1152__) || defined(__gfx1153__)))
  #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)

__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}

    #if __CUDA_ARCH__ < 600 || defined(USE_ROCM)
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) {
  atomicAdd_half2(address, val);
}
    #endif

  #endif
#endif

}  // namespace gptq
}  // namespace vllm
#endif
