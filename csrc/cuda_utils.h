#pragma once

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __forceinline__ __host__ __device__
  #define DEVICE_INLINE __forceinline__ __device__
  #define HOST_INLINE __forceinline__ __host__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);
