#pragma once

#ifdef USE_ROCM
////////////////////////////////////////
// For compatibility with CUDA and ROCm
////////////////////////////////////////
  #include <hip/hip_runtime_api.h>

extern "C" {
  #ifndef CUDA_SUCCESS
    #define CUDA_SUCCESS hipSuccess
  #endif  // CUDA_SUCCESS

// https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
typedef unsigned long long CUdevice;
typedef hipDeviceptr_t CUdeviceptr;
typedef hipError_t CUresult;
typedef hipCtx_t CUcontext;
typedef hipStream_t CUstream;
typedef hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle;
typedef hipMemAllocationGranularity_flags CUmemAllocationGranularity_flags;
typedef hipMemAllocationProp CUmemAllocationProp;
typedef hipMemAccessDesc CUmemAccessDesc;

  #define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
  #define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
  #define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
  #define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
  #define CU_MEM_ALLOCATION_COMP_NONE 0x0

// Error Handling
// https://docs.nvidia.com/cuda/archive/11.4.4/cuda-driver-api/group__CUDA__ERROR.html
CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
  *pStr = hipGetErrorString(hipError);
  return CUDA_SUCCESS;
}

// Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
CUresult cuCtxGetCurrent(CUcontext* ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxGetCurrent(ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxSetCurrent(ctx);
}

// Primary Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice dev) {
  return hipDevicePrimaryCtxRetain(ctx, dev);
}

// Virtual Memory Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  return hipMemAddressFree(ptr, size);
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
  return hipMemAddressReserve(ptr, size, alignment, addr, flags);
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                     const CUmemAllocationProp* prop,
                     unsigned long long flags) {
  return hipMemCreate(handle, size, prop, flags);
}

CUresult cuMemGetAllocationGranularity(
    size_t* granularity, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  return hipMemGetAllocationGranularity(granularity, prop, option);
}

CUresult cuMemMap(CUdeviceptr dptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  return hipMemMap(dptr, size, offset, handle, flags);
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  return hipMemRelease(handle);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc* desc, size_t count) {
  return hipMemSetAccess(ptr, size, desc, count);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  return hipMemUnmap(ptr, size);
}
}  // extern "C"

#else
////////////////////////////////////////
// Import CUDA headers for NVIDIA GPUs
////////////////////////////////////////
  #include <cuda_runtime_api.h>
  #include <cuda.h>
#endif
