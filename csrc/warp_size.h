#pragma once

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#endif

struct Utils {
  static __host__ int get_warp_size() {
#if defined(USE_ROCM)
    static bool is_cached = false;
    static int result;

    if (!is_cached) {
      int device_id;
      hipDeviceProp_t deviceProp;
      hipGetDevice(&device_id);
      hipGetDeviceProperties(&deviceProp, device_id);

      result = deviceProp.warpSize;
      is_cached = true;
    }

    return result;
#else
    return 32;
#endif
  }

  static __device__ constexpr int get_warp_size() {
#if defined(USE_ROCM) && defined(__GFX9__)
    return 64;
#else
    return 32;
#endif
  }
};

#if defined(USE_ROCM)
  #define WARP_SIZE Utils::get_warp_size()
#else
  #define WARP_SIZE 32
#endif
