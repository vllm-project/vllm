import torch
from vllm.logger import init_logger
from typing import List, Optional, Tuple, Type

logger = init_logger(__name__)

try:
    import vllm._vmm_C
except ImportError as e:
    logger.warning("Import vmm error msg: %s", e.msg)


# cache device ptr, used for kv cache tensor
class CacheDevicePtr:

    def __init__(self):
        self._ptr = torch.classes._vmm_C.CacheDevicePtr()

    @property
    def reserved_page_num(self):
        return self._ptr.revervedPageNum

    @reserved_page_num.setter
    def reserved_page_num(self, value: int):
        self._ptr.reservedPageNum = value

    @property
    def allocated_page_num(self):
        return self._ptr.allocatedPageNum

    @allocated_page_num.setter
    def allocated_page_num(self, value: int):
        self._ptr.allocatedPageNum = value


# cache allocator based vmm, used to manage kv cache tensor
class CacheAllocator:

    def __init__(self):
        self._allocator = torch.classes._vmm_C.CacheAllocator()

    def set_page_size(self, page_size: int):
        return self._allocator.setPageSize(page_size)

    def reserve_cache_ptr(self, ptr: CacheDevicePtr, page_num: int = 1):
        return self._allocator.reserveCachePtr(ptr._ptr, page_num)

    def alloc_cache_ptr(self,
                        ptr: CacheDevicePtr,
                        page_num: int = 1,
                        offset: int = 0):
        return self._allocator.allocCachePtr(ptr._ptr, page_num, offset)

    def free_cache_ptr(self, ptr: CacheDevicePtr):
        return self._allocator.freeCachePtr(ptr._ptr)

    def release_cache_ptr(self,
                          ptr: CacheDevicePtr,
                          page_num: int = 0,
                          offset: int = 0):
        return self._allocator.releaseCachePtr(ptr._ptr, page_num, offset)


# other utils
def wrap_cache_ptr_to_tensor(ptr: CacheDevicePtr, dtype_str: str,
                             shape: Tuple[int, ...]):
    return torch.ops._vmm_C.wrap_cache_ptr_to_tensor(ptr._ptr, dtype_str,
                                                     shape)
