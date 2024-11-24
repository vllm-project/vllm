#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif
int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  int device, value;
  if (device_id < 0) {
    cudaGetDevice(&device);
  } else {
    device = device_id;
  }
  cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attribute),
                         device);
  return value;
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  int64_t attribute;
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
  // cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74

#ifdef USE_ROCM
  attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
  attribute = cudaDevAttrMaxSharedMemoryPerBlockOptin;
#endif

  return get_device_attribute(attribute, device_id);
}
