#include "vmm.h"

#include <c10/core/ScalarType.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>


/*
** CacheDevicePtr functions implementation
*/

CacheDevicePtr::CacheDevicePtr()
    : dptr(0), reservedPageNum(0), allocatedPageNum(0) {}

CacheDevicePtr::~CacheDevicePtr() {
  if (dptr != 0) {
    auto status = cuMemUnmap(dptr, reservedPageNum * pageSize);

    for (int i = 0; i < handles.size(); i++) {
      auto status = cuMemRelease(handles[i]);
    }

    status = cuMemAddressFree(dptr, reservedPageNum * pageSize);
  }
}

void CacheDevicePtr::setPageSize(int64_t num) { pageSize = num * 2 * _MB; }

// get CUdeviceptr dptr
CUdeviceptr CacheDevicePtr::get_dptr() { return dptr; }

// get void * type pointer
void* CacheDevicePtr::get_void_ptr() { return reinterpret_cast<void *>(dptr); }



/*
** CacheAllocator functions implementation
*/

CacheAllocator::CacheAllocator() {
  // get current device gpu id
  int currentDevice;
  auto cudaStatus = cudaGetDevice(&currentDevice);
  TORCH_CHECK(cudaStatus == cudaSuccess, "cudaGetDevice failed!");

  // set memory allocation property struct CUmemAllocationProp, 
  // which is used to control the specific behavior of memory allocation
  prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = currentDevice;

  // set memory access descriptor struct CUmemAccessDesc,
  // which is used to control the access permission of memory
  accessDescr = {};
  accessDescr.location.id = prop.location.id;
  accessDescr.location.type = prop.location.type;
  accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // get the granularity of device memory allocation
  // getGranurality();

  // set the page size of memory allocation, default is equal to granurality
  // pageSize = granurality;
}

int64_t CacheAllocator::getGranurality() {
  cuMemGetAllocationGranularity(&granurality, &prop,
                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  printf("Granularity: %ld Bytes\n", granurality);
  return granurality;
}

void CacheAllocator::setPageSize(int64_t num) { pageSize = num * granurality; }

// reserve function, reserve virtual address space
int64_t CacheAllocator::reserveCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr, int64_t pageNum) {
  if (pageNum == 0) {
    return CUDA_SUCCESS;
  }
  size_t size = pageNum * pageSize;
  auto status = cuMemAddressReserve(&(ptr->dptr), size, 0, 0, 0);

  if (status != CUDA_SUCCESS) {
    printf("cuMemAddressReserve failed! error-code: %d\n", status);
  } else {
    ptr->reservedPageNum += pageNum;
  }

  return status;
}

// alloc function, allocate physical memory, map to the reserved virtual address
// space of dptr, and set access permission
int64_t CacheAllocator::allocCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr,
                                      int64_t pageNum, int64_t offset) {
  if (pageNum == 0) {
    return CUDA_SUCCESS;
  }
  // size = ((size - 1) / pageSize + 1) * pageSize;
  size_t size = pageNum * pageSize;
  auto start_dptr = ptr->dptr + offset;

  CUresult status = CUDA_SUCCESS;
  CUmemGenericAllocationHandle allocationHandle;
  if ((status = cuMemCreate(&allocationHandle, size, &prop, 0)) ==
      CUDA_SUCCESS) {
    if ((status = cuMemMap(start_dptr, size, 0, allocationHandle, 0)) ==
        CUDA_SUCCESS) {
      if ((status = cuMemSetAccess(start_dptr, size, &accessDescr, 1)) ==
          CUDA_SUCCESS) {
        // ptr.handles.push_back(allocationHandle);  // handles is unused now
        ptr->allocatedPageNum += pageNum;
      } else {
        printf("cuMemMap success,but cuMemSetAccess failed!, err code: %d\n",
               status);
        cuMemUnmap(start_dptr, size);
      }
    }
    // if (status != CUDA_SUCCESS) {
    //   printf("cuMemMap or cuMemsetAccess failed!, err code: %d\n", status);
    //   cuMemRelease(allocationHandle);
    // }
    cuMemRelease(
        allocationHandle);  // always release the handle, but the memory is
                            // still can access util cuMemUnmap
  } else {
    printf("cuMemCreate failed!, err code: %d\n", status);
  }
  return status;
}

// free function, unmap the virtual address space，release physical memory
// handles and free virtual address space
int64_t CacheAllocator::freeCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr) {
  CUresult status = CUDA_SUCCESS;
  if (ptr->dptr != 0) {
    status = cuMemUnmap(ptr->dptr, ptr->reservedPageNum * pageSize);
    // status = cuMemUnmap(ptr.dptr, ptr.allocatedPageNum * pageSize);
    if (status != CUDA_SUCCESS) {
      printf("cuMemUnmap failed! error-code: %d\n", status);
    } else {
      for (int i = 0; i < ptr->handles.size(); i++) {
        status = cuMemRelease(ptr->handles[i]);
        if (status != CUDA_SUCCESS) {
          printf("cuMemRelease failed! error-code: %d\n", status);
          return status;
        }
      }
      ptr->handles.clear();

      status = cuMemAddressFree(ptr->dptr, ptr->reservedPageNum * pageSize);
      if (status != CUDA_SUCCESS) {
        printf("cuMemAddressFree failed! error-code: %d\n", status);
      }
    }
  }
  return status;
}

// releaseCachePtrPages function, unmap the virtual address space，release
// physical memory handles but not free virtual address space
int64_t CacheAllocator::releaseCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr,
                                        int64_t pageNum, int64_t offset) {
  if (pageNum == 0) {
    return CUDA_SUCCESS;
  }
  auto start_dptr = ptr->dptr + offset;
  CUresult status = CUDA_SUCCESS;
  if (ptr->dptr != 0) {
    status = cuMemUnmap(start_dptr, pageNum * pageSize);
    // status = cuMemUnmap(ptr.dptr, ptr.allocatedPageNum * pageSize);
    if (status != CUDA_SUCCESS) {
      printf("cuMemUnmap failed! error-code: %d\n", status);
    } else {
      for (int i = 0; i < ptr->handles.size(); i++) {
        status = cuMemRelease(ptr->handles[i]);
        if (status != CUDA_SUCCESS) {
          printf("cuMemRelease failed! error-code: %d\n", status);
          return status;
        }
      }
      ptr->handles.clear();
    }
  }
  return status;
}



/*
** vmm other util functions implementation
*/

torch::Tensor wrap_dptr_to_tensor(CUdeviceptr d_ptr, const std::string dtype,
                                  at::ArrayRef<int64_t> shape) {
  // get current device gpu id
  int currentDevice;
  auto cudaStatus = cudaGetDevice(&currentDevice);
  TORCH_CHECK(cudaStatus == cudaSuccess, "cudaGetDevice failed!");

  auto _type = c10::kFloat;

  const std::unordered_map<std::string, c10::ScalarType> typeMap = {
      // float data type
      {"float64", c10::kDouble},    
      {"float32", c10::kFloat},
      {"float16", c10::kHalf},      
      {"float", c10::kFloat},
      {"double", c10::kDouble},     
      {"half", c10::kHalf},
      {"bfloat16", c10::kBFloat16}, 
      // integer data type
      {"int64", c10::kLong},
      {"int32", c10::kInt},         
      {"int16", c10::kShort},
      {"int8", c10::kChar},         
      {"int", c10::kInt},
      {"uint8", c10::kByte}};

  _type = typeMap.at(dtype);

  // set the data type and device of the Tensor
  auto options =
      torch::TensorOptions().dtype(_type).device(torch::kCUDA, currentDevice);

  // create a Tensor from the CUdeviceptr
  torch::Tensor tensor =
      torch::from_blob(reinterpret_cast<void*>(d_ptr), shape, options);

  return tensor;
}

torch::Tensor wrap_cache_ptr_to_tensor(const c10::intrusive_ptr<CacheDevicePtr>& ptr, 
                                       const std::string dtype,
                                       at::ArrayRef<int64_t> shape) {
  return wrap_dptr_to_tensor(ptr->dptr, dtype, shape);
}
