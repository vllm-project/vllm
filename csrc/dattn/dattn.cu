/*
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
 */ 
 
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include "dattn.h"


#define KV_UTILIZATION_RATE (0.9)

static CUmemAllocationProp _prop = {};
static CUmemAccessDesc _accessDescr = {}; 
/* 
  In this allocator, we only have the following concepts, but without the concept of tokens.
  The python portion should convert the number of tokens to tokens depending on their block_size (e.g., 16)
  Region: virtual address space for a request. Currently, we support the space for max_seq_len.
 */
static uint64_t roundup(uint64_t size, uint64_t align_size) {
  return ((size + align_size - 1)/align_size) * align_size; 
}

static int allocatePhyPages(void * ptr, uint64_t size) {
  CUdeviceptr dptr = (CUdeviceptr)ptr;

  CUdevice dev; // device
  CHECK_DRV(cuCtxGetDevice(&dev));
  _prop.location.id = dev;
  _accessDescr.location = _prop.location;

  CUresult status = CUDA_SUCCESS;
  CUmemGenericAllocationHandle allocationHandle;
  if ((status = cuMemCreate(&allocationHandle, size, &_prop, 0)) == CUDA_SUCCESS) {
    if ((status = cuMemMap(dptr, size, 0ULL, allocationHandle, 0ULL)) == CUDA_SUCCESS) {
      if ((status = cuMemSetAccess(dptr, size, &_accessDescr, 1)) != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemMap success,but cuMemSetAccess failed!, err code: %d\n", status);
        cuMemUnmap(dptr, size);
      }
    }
    // always release the handle, but the memory is accessible util cuMemUnmap
    if((status = cuMemRelease(allocationHandle)) != CUDA_SUCCESS) {
      fprintf(stderr, "cuMemRelease failed, err code: %d\n", status);
    } 
  } else {
    fprintf(stderr, "cuMemCreate failed!, err code: %d\n", status);
  }
  return status == CUDA_SUCCESS ? 0 : -1;
}

// Free the physical memory [ptr, ptr + size]
static void freePhysicalMemory(void* ptr, size_t size) {
  CUdeviceptr dptr = (CUdeviceptr)ptr;
  CHECK_DRV(cuMemUnmap(dptr, size));
  CHECK_DRV(cuMemAddressFree(dptr, size));
}

/*
** kvCacheRegion functions implementation
*/
kvCacheRegion::kvCacheRegion(uint64_t region_size, uint64_t block_size, uint64_t page_size, CUdeviceptr ptr) {
  this->region_size = region_size;
  this->block_size = block_size;
  this->page_size = page_size; 
  this->dptr = reinterpret_cast<char*>(ptr);  
  this->nextUnmapedAddr = reinterpret_cast<char*>(ptr); 

  this->offset = 0; 
  this->total_pages = 0;
  this->used_pages = 0; 
}

// Decontructor: release all physical pages of this region
kvCacheRegion::~kvCacheRegion() {
  uint64_t size = this->total_pages * this->page_size; 
  freePhysicalMemory(this->dptr, size); 

  // Note that since the region is detroyed, 
  // no need to clear other counters. 
}

/*
// get CUdeviceptr dptr
CUdeviceptr kvCacheRegion::getDeviceDptr(void) { 
  return reinterpret_cast<CUdeviceptr>(this->dptr); 
}

// get void * type pointer
void* kvCacheRegion::getVoidDptr(void) { 
  return reinterpret_cast<void *>(this->dptr); 
}
*/

uint64_t kvCacheRegion::getAllocPhyPages(void) {
  return this->total_pages;
} 

uint64_t kvCacheRegion::getUsedPhysicalPages(void) {
  return this->used_pages; 
}

/*
  kvCacheRegion function: allocate cached blocks  
    if the return value > 0, then it is succesful. 
 */ 
int64_t kvCacheRegion::allocCacheBlocks(uint64_t blocks, uint64_t * used_pages) {
  uint64_t size = blocks * this->block_size;

  int64_t toallocPages = -1; 

  // Align the new offset to page_size
  uint64_t alignedOffset = roundup(this->offset + size, this->page_size); 

  // Check how many pages should we allocated this time
  char * alignedAddr = this->dptr + alignedOffset; 
  if( alignedAddr > this->nextUnmapedAddr) {

    // Check whether alignedAddr is actually aligned well
    assert((alignedAddr - this->nextUnmapedAddr)%this->page_size == 0);

    toallocPages = (alignedAddr - this->nextUnmapedAddr)/this->page_size; 

    assert(toallocPages >= 0);

    uint64_t allocSize = toallocPages * this->page_size;

    // Allocate physical pages, which will exit if can't allocate successfully
    if (toallocPages > 0 && allocatePhyPages(this->nextUnmapedAddr, allocSize) == 0) {
      this->nextUnmapedAddr = alignedAddr;
        
      // Update the used pages correspondingly. The statement works even when this->offset is not aligned to page_size
      *used_pages += toallocPages; 

      // Update the offset after allocating these blocks. 
      this->offset += size; 
      assert(this->offset <= alignedOffset);
    }
  }
 
  return toallocPages; 
}

// freeUnusedPages from a region, and return freed pages
int kvCacheRegion::freeUnusedPages(void) {
  int freedPages = 0;

  // Free pages only when total_pages is larger than used_pages
  if(this->total_pages > this->used_pages) {
    assert(this->nextUnmapedAddr > (this->dptr + offset));

    // Get the offset of next page, since we can't collect a page if its partialy used
    uint64_t alignedOffset = roundup(offset, this->page_size);
    
    // startAddr points to the beginning of the next page
    char * startAddr = this->dptr + alignedOffset; 

    uint64_t size = this->nextUnmapedAddr - startAddr; 
    assert((size % this->page_size) == 0); 

    freedPages = size/this->page_size; 
    // free all unused pages of this region. 
    // If a page is partially used, then it cannot be freed 
    if(size > 0) {
      freePhysicalMemory(startAddr, size);
      this->total_pages -= freedPages;
      this->nextUnmapedAddr = startAddr;  
      // No need to change offset here. 
    } 
  }

  return freedPages; 
}

/*
** kvCacheAllocator functions implementation
*/
kvCacheAllocator::kvCacheAllocator(int64_t max_seq_length, int64_t layers_num, int64_t heads_num, int64_t head_size, int64_t tokens_per_block, int64_t dtype_size) {
  uint64_t key_cache_block_per_layer =  tokens_per_block * heads_num * head_size * dtype_size; 
  uint64_t value_cache_block_per_layer = key_cache_block_per_layer;
  uint64_t cache_block_size = (key_cache_block_per_layer + value_cache_block_per_layer) * layers_num; 

  //fprintf(stderr, "kvCacheAllocator initialization: key_cache_block_per_layer-%d, cache_block_size-%d\n", key_cache_block_per_layer, cache_block_size); 
  // Getting the cuda device and force the initialization
  CUdevice dev; // device
  CHECK_RT(cudaFree(0));  // Force and check the initialization of the runtime
  CHECK_DRV(cuCtxGetDevice(&dev));
  
  size_t aligned_sz; 
  _prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  _prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _prop.location.id = dev;
  _accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  _accessDescr.location = _prop.location;

  CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  
  uint64_t max_blocks = roundup(max_seq_length, tokens_per_block)/tokens_per_block; 
  uint64_t region_size = max_blocks * cache_block_size; 

  this->page_size = aligned_sz;
  this->region_size = ((region_size + aligned_sz - 1) / aligned_sz) * aligned_sz;
  this->block_size = cache_block_size;

  //printf("kvCacheAllocator: page_size-%ld, region_size-%ld, block_size-%ld\n", this->page_size, this->region_size, this->block_size);

  // TODO: finding out how much physical blocks it includes. This is just for the reference or watermark, as 
  // there is no need to rely on pre-assigned values if physical blocks are allocated on-demand
  size_t freeMem, totalMem;
  CHECK_RT(cudaMemGetInfo(&freeMem, &totalMem)); 

  this->watermark_pages = (((uint64_t)(freeMem * KV_UTILIZATION_RATE))/this->page_size);  
   
  // Doing other initialization
  this->total_pages = 0;
  this->used_pages = 0;
  this->active_regions = 0;
}

int64_t kvCacheAllocator::getPageSize() {
  return this->page_size;
}


// reserve function, reserve virtual address space for a request, and also allocate the first physical block
int64_t kvCacheAllocator::reserveRegion(int64_t req_id) {
  CUdeviceptr ptr;
  kvCacheRegion * region = nullptr;

  // Check whether there are some cached regions 
  if(this->cached_regions.size()) {
    // Pop the latest region from cached vector, which is more efficient and therefore it is the default method
    region = _getLastCachedRegion();  
  }
  else {
    //printf("region_size == %d bytes == %d MB\n", this->region_size, this->region_size%2097152); 
    // The expensive way to get a new region. Only invoked when no cached regions
    // Allocate the virtual address for this region
    CHECK_DRV(cuMemAddressReserve(&ptr, this->region_size, 0ULL, 0ULL, 0ULL));

    // Create a new region from the scratch
    region = new kvCacheRegion(this->region_size, this->block_size, this->page_size, ptr); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);
  
  // Record the region information
  this->active_regions += 1; 
  this->active_regions_map[req_id] = region; 

  return static_cast<int64_t>(ptr);
}

// Release the region with the given req_id
void kvCacheAllocator::_releaseRegion(int64_t req_id) {
  // Find the region corresponding to the given req_id
  if(this->active_regions_map.count(req_id) == 0) {
    fprintf(stderr, "ERROR: req_id-%ld at does not exist at all.!\n", req_id);
    exit(-1); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);

  kvCacheRegion * region = this->active_regions_map[req_id];
  // Delete this region from active_regions_map that only keep 
  this->active_regions_map.erase(req_id);
  this->active_regions--; 
  // Note that as we don't actually release physical cache blocks. 
  // Therefore, we don't need to change the active_blocks here. 

  // Cache the given region, as it can be used for the future ideally. 
  // In order to reduce the overhead of memory management, we did not 
  // reclaim physical blocks until necessary.
  _cacheReleasedRegion(region); 
}

// Cache the released region. Don't release the virtual address and physical cache blocks
void kvCacheAllocator::_cacheReleasedRegion(kvCacheRegion * region) {
  this->cached_regions.push_back(region);
}

// Get the lastly-released region. If the region has some physical blocks, 
// they will be re-utilized as well.
// Note that using cached regions is way more efficient than allocating a new region
kvCacheRegion * kvCacheAllocator::_getLastCachedRegion(void) {
  assert(!this->cached_regions.empty());

  kvCacheRegion * region = this->cached_regions.back(); 
  this->cached_regions.pop_back(); 

  return region; 
} 

// This function is invoked when the number of physical pages is above 
// the preset threshold. It performs the garbage collecton of physical pages
void kvCacheAllocator::_gcPhyPages(int64_t toCollectPages) {

  assert(toCollectPages > 0); 

  // first, collect the pages in cached regions. 
  kvCacheRegion * region; 

  // First, collect pages from cached_regions as it won't affect active requests. 
  while(!this->cached_regions.empty() && toCollectPages > 0) {
    // Release Least-Recently-Used regions at first
    region = this->cached_regions.front();
    this->cached_regions.pop_front();

    int pages = region->getAllocPhyPages();
    if(pages > 0) {
      this->total_pages -= pages; 
      toCollectPages -= pages; 
    }

    // deconstruct this region, which will collect all physical pages inside
    delete region;
  }

  // Check active regions if necessary
  while(toCollectPages > 0) {
    // Collect pages from active regions
    for(auto it = this->active_regions_map.begin(); it != this->active_regions_map.end(); it++) {
      // it->second points to the region
      region = it->second; 

      int pages = region->freeUnusedPages(); 
      if(pages > 0) {
        // Update the total_pages for the allocator
        this->total_pages -= pages; 

        toCollectPages -= pages; 
      }

      // Exit the loop if we collect enough pages
      if(toCollectPages <= 0) {
        break; 
      }
    }
  }
  
}

// alloc function, allocate physical memory, map to the reserved virtual address
// This function is designed for both prefill and decoding phase, where prefill may 
// require to save KV cache of multiple tokens, which should not invoke this function multiple times. 
// Similarly, the python code may get the physical blocks for multiple tokens during the decoding phase
// Note that the allocator doesn't care about tokens (which should be handled by the python code), but only blocks here.
int64_t kvCacheAllocator::_allocCacheBlocksForRequest(int64_t req_id, int64_t blocks) {
  int64_t pages = -1;

  // Find the region corresponding to the given req_id, which should reserveRegion before
  // If the req_id doesn't exist at all, it is the bug that should be fixed.  
  if(this->active_regions_map.count(req_id) == 0) {
    fprintf(stderr, "ERROR: req_id %ld does not exist at all.!\n", req_id);
    exit(-1); 
  }

  std::lock_guard<std::mutex> lock(this->mutex);

  kvCacheRegion * region = this->active_regions_map[req_id]; 

  pages = region->allocCacheBlocks(blocks, &this->used_pages);

  if(pages > 0) { 
    this->total_pages += pages;

    // check whether we need to purge physical memory
    if(this->total_pages >= this->watermark_pages && this->total_pages > this->used_pages) {
      int toCollectPages = std::min(this->total_pages - this->used_pages, this->total_pages - this->watermark_pages); 

      // Garbage collection for physical pages. 
      _gcPhyPages(toCollectPages);
    } 
  }

  return pages;
}

// Allocate cache blocks for a range of requests. Each request information will be an vector, with
// the request id as the first, and then number of blocks as the second. 
int64_t kvCacheAllocator::allocCacheBlocks(std::vector<std::vector<int64_t>> req_blocks) {
  int64_t pages = 0; 

  for(auto row : req_blocks) {
    uint64_t req_id = row[0]; 
    uint64_t blocks = row[1]; 

    //fprintf(stderr, "allocating req_id-%d and blocks-%d\n", req_id, blocks);
    pages += _allocCacheBlocksForRequest(req_id, blocks); 
  }

  return pages; 
}
// Release regions specified in the vector
void kvCacheAllocator::releaseRegions(std::vector<int64_t> regions) {
  for(auto region : regions) {
    _releaseRegion(region);
  }
}


int64_t kvCacheAllocator::getAllocPhyPages(int64_t req_id) {
  int64_t pages = 0; 

  if(req_id == 0) {
    pages = this->total_pages; 
  }
  else {
    // Find the region corresponding to the given req_id, which should reserveRegion before
    // If the req_id doesn't exist at all, it is the bug that should be fixed.  
    if(this->active_regions_map.count(req_id) == 0) {
      fprintf(stderr, "ERROR: req_id does not exist at all.!");
      exit(-1); 
    }

    std::lock_guard<std::mutex> lock(this->mutex);

    kvCacheRegion * region = this->active_regions_map[req_id]; 
    pages = region->getAllocPhyPages(); 
  }

  return pages;
}

void kvCacheAllocator::collectPhyPages(int64_t pages) {
  if(pages == 0) {
    // Collect pages defined by watermark
    pages = std::min(this->total_pages - this->used_pages, this->total_pages - this->watermark_pages); 
  }
  
  _gcPhyPages(pages);
  return; 
}

#if 0
// TODO: we need to delete this function!!!
// free function, unmap the virtual address space，release physical memory
// handles and free virtual address space
int64_t kvCacheAllocator::freeCacheBlock(const c10::intrusive_ptr<kvCacheRegion>& ptr) {
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

#endif
