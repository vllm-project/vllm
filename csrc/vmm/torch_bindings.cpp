#include "registration.h"
#include "vmm.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // CacheDevicePtr class bind
  m.class_<CacheDevicePtr>("CacheDevicePtr")
      .def(torch::init<>())
      // .def_readwrite("dptr", &CacheDevicePtr::dptr);
      // dptr cann't bind success because of the type of dptr is
      // CUdeviceptr(=unsigned long long), which is not supported in torch
      .def_readwrite("reservedPageNum", &CacheDevicePtr::reservedPageNum)
      .def_readwrite("allocatedPageNum", &CacheDevicePtr::allocatedPageNum);

  // CacheAllocator class bind
  m.class_<CacheAllocator>("CacheAllocator")
      .def(torch::init<>())
      .def("setPageSize", &CacheAllocator::setPageSize)
      .def("reserveCachePtr", &CacheAllocator::reserveCachePtr)
      .def("allocCachePtr", &CacheAllocator::allocCachePtr)
      .def("freeCachePtr", &CacheAllocator::freeCachePtr)
      .def("releaseCachePtr", &CacheAllocator::releaseCachePtr);

  // other util functions bind
  m.def("wrap_cache_ptr_to_tensor", &wrap_cache_ptr_to_tensor);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
