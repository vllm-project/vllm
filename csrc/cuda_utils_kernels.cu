#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#endif
int get_device_attribute(
    int attribute,
    int device_id)
{
    int device, value;
    if (device_id < 0) {
        cudaGetDevice(&device);
    }
    else {
        device = device_id;
    }
    cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attribute), device);
    return value;
}
