#pragma once
/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
 */ 
//#include <torch/script.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <deque>
#include <unordered_map>
#include <torch/custom_class.h>
#include <c10/util/intrusive_ptr.h>

#define _MB (1 << 20)

using namespace std;

static inline void
checkRtError(cudaError_t res, const char *tok, const char *file, unsigned line) {
    if (res != cudaSuccess) {
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << cudaGetErrorString(res) << std::endl;
        abort();
    }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__);

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line) {
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);


// kvCacheRegion class, to warp CUdeviceptr, used to kv-cache tensor
// record the reserved virtual address size and allocated physical memory size.
// TODO: we may avoid expose this class externally in the future. 
class kvCacheRegion : public torch::CustomClassHolder{
private:
  char * dptr;

  // the number of bytes for the request's virtual address space (region)
  uint64_t region_size; 
  
  // the size of a kv cache block in bytes, which is NOT the number tokens inside a block
  uint64_t block_size; 

  // The real page_size supported by the hardware, which could be larger or smaller than block_size
  uint64_t page_size; 

  // The number of allocated physical pages for the current region.
  uint64_t total_pages;

  // Note that total_pages can be larger than use_pages, as it may inherit a 
  // live region that already has many allocated physical pages. 
  uint64_t used_pages;
 
  // virtual address of the next page that needs to be mapped. 
  // Typically, (nextUnmapedAddr - dptr)/page_size == total_pagees 
  char * nextUnmapedAddr; 

  // The difference between the used address (the end of invoking allocBlocks) and the starting pointer of the region
  uint64_t offset;   

public:

  kvCacheRegion(uint64_t region_size, uint64_t block_size, uint64_t page_size, CUdeviceptr ptr);

  ~kvCacheRegion();

  // get CUdeviceptr dptr
  CUdeviceptr getStartDptr();

  // get the number of physical pages
  uint64_t getAllocPhyPages(void); 
  uint64_t getUsedPhysicalPages(void);
  int64_t allocCacheBlocks(uint64_t blocks, uint64_t * used_pages);
  int freeUnusedPages(void);
  void freeAllPhyMemory(void);
};


// kvCacheAllocator class, used for memory allocation of kv-cachemanager, memory allocation is based on page granularity,
class kvCacheAllocator : public torch::CustomClassHolder{
private:

  /*
    The following information are about physical blocks. 
       
      total_pages is the total number of physical pages that have been assigned from the allocator. 
      used_pages can be less than total_pages, as used_pages will be incremented only when allocCacheBlock is invoked. 
   */
  uint64_t total_pages; 
  uint64_t used_pages; 

  // How many regions (requests) in this allocator
  uint64_t active_regions;

  // If total_pages is larger than the watermark, then we will start to garbage collect physical pages
  // More specifically, we will reclaim pages from cached regions at first, and then from active regions 
  uint64_t watermark_pages; 
  
  uint64_t region_size; 
  uint64_t block_size;
  uint64_t page_size;
  CUdevice device;
  std::mutex mutex;

  // the hashtable to record the relationship between regions and ptrs
  unordered_map<uint64_t, kvCacheRegion*> active_regions_map;
  std::deque<kvCacheRegion *> cached_regions; 

  // Internal functions
  void _cacheReleasedRegion(kvCacheRegion * region);
  kvCacheRegion * _getLastCachedRegion(void); 
  void _gcPhyPages(int64_t toCollectPages);
  void _initializeAllocHandles(void);
  // Release the virtual address space for a region that is related to one request
  void _releaseRegion(int64_t region_id);
  // Allocate physical memory, map to the reserved virtual address space of dptr, and set access permission
  int64_t _allocCacheBlocksForRequest(int64_t region_id, int64_t blocks = 1);


public:

  //kvCacheAllocator(); 
  // The default contructor. Otherwise, torch bindings will complain it. 
  kvCacheAllocator(int64_t max_seq_length, int64_t layers_num, int64_t heads_num, int64_t head_size, int64_t tokens_per_block, int64_t dtype_size);
  // {
    // Nothing to do
  //}

  ~kvCacheAllocator() = default;

  //void initialization(int64_t max_seq_length, int64_t layers_num, int64_t heads_num, int64_t head_size, int64_t tokens_per_block, int64_t dtype_size);
  

  // get the granularity of the physical memory allocation
  int64_t getPageSize(void);

  // Reserve the virtual address space for a region that is related to one request
  // In particular, the regionSize == 2 * max_seq_length * layers_num * heads_num * head_size * dtype_size
  // "2" here is to allocate Key and Value cache together, which helps to reduce the fragmentation 
  int64_t reserveRegion(int64_t region_id);

  void releaseRegions(std::vector<int64_t> regions);

  int64_t allocCacheBlocks(std::vector<std::vector<int64_t>> reqs_blocks);

  // Allow the python code to know the physical memory used for the whole 
  // kv cache or the memory for the specified request (when region_id is not 0). 
  int64_t getAllocPhyPages(int64_t region_id = 0); 
  void collectPhyPages(int64_t pages = 0); 
};

