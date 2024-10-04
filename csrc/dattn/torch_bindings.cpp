/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
 */ 
#include "core/registration.h"
#include "dattn.h"


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // kvCacheAllocator class bind
  m.class_<kvCacheAllocator>("kvCacheAllocator")
    //.def(torch::init<>())
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>())
    .def("reserveRegion", &kvCacheAllocator::reserveRegion)
    .def("releaseRegions", &kvCacheAllocator::releaseRegions)
    .def("allocCacheBlocks", &kvCacheAllocator::allocCacheBlocks)
    .def("getAllocPhyPages", &kvCacheAllocator::getAllocPhyPages)
    .def("collectPhyPages", &kvCacheAllocator::collectPhyPages);
    //.def("freeCacheBlock", &kvCacheAllocator::freeCacheBlock)
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
