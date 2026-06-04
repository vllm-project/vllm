#include "cutlass_extensions/common.hpp"

int32_t get_sm_version_num() {
  // Cached: the prior implementation queried device 0 on every call, which
  // added a couple of cudaDeviceGetAttribute driver calls (µs-scale) to the
  // hot path of every cutlass_scaled_mm / nvfp4 / moe dispatch. The existing
  // semantics already assumed a homogeneous device set (the device index was
  // hard-coded to 0), so we preserve that by caching a single value.
  static const int32_t cached = [] {
    int32_t major_capability = 0;
    int32_t minor_capability = 0;

    cudaError_t err_major =
        cudaDeviceGetAttribute(&major_capability,
                               cudaDevAttrComputeCapabilityMajor, 0);
    TORCH_CHECK(
        err_major == cudaSuccess,
        "cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor) failed: ",
        cudaGetErrorString(err_major));

    cudaError_t err_minor =
        cudaDeviceGetAttribute(&minor_capability,
                               cudaDevAttrComputeCapabilityMinor, 0);
    TORCH_CHECK(
        err_minor == cudaSuccess,
        "cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMinor) failed: ",
        cudaGetErrorString(err_minor));

    return major_capability * 10 + minor_capability;
  }();
  return cached;
}
