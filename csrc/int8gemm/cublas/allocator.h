/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Memory Allocator
 **/

#pragma once

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

#if defined(CUDART_VERSION) && CUDART_VERSION < 11020
#define CUDA_MEMORY_POOL_DISABLED
#endif

enum class AllocatorType { CUDA, TF, TH };

enum class ReallocType {
  INCREASE,
  REUSE,
  DECREASE,
};

class IAllocator {
public:
  virtual ~IAllocator(){};

  virtual void *malloc(size_t size, const bool is_set_zero = true,
                       bool is_host = false) = 0;
  virtual void free(void **ptr, bool is_host = false) const = 0;
  virtual void setStream(cudaStream_t stream) = 0;
  virtual cudaStream_t returnStream() = 0;
  virtual void memSet(void *ptr, const int val, const size_t size) = 0;

  template <typename T>
  void *reMalloc(T *ptr, size_t size, const bool is_set_zero = true,
                 bool is_host = false) {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size = ((size + 31) / 32) * 32; // make the buffer align with 32 bytes
    void *void_ptr = (void *)ptr;
    void *ptr_address = getAddress(void_ptr);
    if (isExist(ptr_address)) {
      ReallocType realloc_type = isReMalloc(ptr_address, size);
      if (realloc_type == ReallocType::INCREASE) {
        // FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.",
        // void_ptr);
        free((void **)(&void_ptr), is_host);
        return malloc(size, is_set_zero, is_host);
      }
#if !defined(CUDA_MEMORY_POOL_DISABLED)
      else if (realloc_type == ReallocType::DECREASE) {
        // FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to
        // memory pools.", void_ptr);
        free((void **)(&void_ptr), is_host);
        return malloc(size, is_set_zero, is_host);
      }
#endif
      else {
        // FT_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing
        // for reMalloc.", void_ptr, size);
        if (is_set_zero) {
          memSet(void_ptr, 0, size);
        }
        return void_ptr;
      }
    } else {
      // FT_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
      return malloc(size, is_set_zero, is_host);
    }
  }

protected:
  virtual bool isExist(void *address) const = 0;
  virtual ReallocType isReMalloc(void *address, size_t size) const = 0;

  void *getAddress(void *ptr) const { return ptr; }
};

template <AllocatorType AllocType_> class Allocator;

template <> class Allocator<AllocatorType::CUDA> : public IAllocator {
private:
  const int device_id_;
  cudaStream_t stream_ = 0; // initialize as default stream
  std::unordered_map<void *, size_t> *pointer_mapping_;

  bool isExist(void *address) const {
    return pointer_mapping_->count(address) > 0;
  }
  ReallocType isReMalloc(void *address, size_t size) const {
    FT_CHECK(isExist(address));
    if (pointer_mapping_->at(address) < size) {
      return ReallocType::INCREASE;
    } else if (pointer_mapping_->at(address) == size) {
      return ReallocType::REUSE;
    } else {
      return ReallocType::DECREASE;
    }
  }

public:
  Allocator(int device_id) : device_id_(device_id) {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    pointer_mapping_ = new std::unordered_map<void *, size_t>();
#if defined(CUDA_MEMORY_POOL_DISABLED)
    // FT_LOG_WARNING(
    //     "Async cudaMalloc/Free is not supported before CUDA 11.2. Using Sync
    //     cudaMalloc/Free." "Note this may lead to hang with NCCL kernels
    //     launched in parallel; if so, try NCCL_LAUNCH_MODE=GROUP");
#else
    int device_count = 1;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    cudaMemPool_t mempool;
    check_cuda_error(cudaDeviceGetDefaultMemPool(&mempool, device_id));
    cudaMemAccessDesc desc = {};
    int peer_access_available = 0;
    for (int i = 0; i < device_count; i++) {
      if (i == device_id) {
        continue;
      }
      check_cuda_error(
          cudaDeviceCanAccessPeer(&peer_access_available, device_id, i));
      if (!peer_access_available) {
        // FT_LOG_WARNING("Device " + std::to_string(device_id) + " peer access
        // Device " + std::to_string(i)
        //                + " is not available.");
        continue;
      }
      desc.location.type = cudaMemLocationTypeDevice;
      desc.location.id = i;
      desc.flags = cudaMemAccessFlagsProtReadWrite;
      check_cuda_error(cudaMemPoolSetAccess(mempool, &desc, 1));
    }
    // set memory pool threshold to avoid shrinking the pool
    uint64_t setVal = UINT64_MAX;
    check_cuda_error(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolAttrReleaseThreshold, &setVal));
#endif
  }

  virtual ~Allocator() {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
      free((void **)(&pointer_mapping_->begin()->first));
    }
    delete pointer_mapping_;
  }

  void setStream(cudaStream_t stream) { stream_ = stream; }

  cudaStream_t returnStream() { return stream_; };

  void *malloc(size_t size, const bool is_set_zero = true,
               bool is_host = false) {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (size == 0) {
      return nullptr;
    }
    void *ptr = nullptr;
    int o_device = 0;

    check_cuda_error(getSetDevice(device_id_, &o_device));
    if (is_host) {
      check_cuda_error(cudaMallocHost(&ptr, (size_t)(ceil(size / 32.)) * 32));
    } else {
#if defined(CUDA_MEMORY_POOL_DISABLED)
      check_cuda_error(cudaMalloc(&ptr, (size_t)(ceil(size / 32.)) * 32));
#else
      check_cuda_error(
          cudaMallocAsync(&ptr, (size_t)(ceil(size / 32.)) * 32, stream_));
#endif
    }
    if (is_set_zero) {
      check_cuda_error(
          cudaMemsetAsync(ptr, 0, (size_t)(ceil(size / 32.)) * 32, stream_));
    }
    check_cuda_error(getSetDevice(o_device));
    // FT_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);

    pointer_mapping_->insert({getAddress(ptr), size});

    return ptr;
  }

  void free(void **ptr, bool is_host = false) const {
    // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    void *address = getAddress(*ptr);
    if (*ptr != nullptr) {
      int o_device = 0;
      if (pointer_mapping_->count(address)) {
        // FT_LOG_DEBUG("Free buffer %p", address);
        check_cuda_error(getSetDevice(device_id_, &o_device));
        if (is_host) {
          check_cuda_error(cudaFreeHost(*ptr));
        } else {
#if defined(CUDA_MEMORY_POOL_DISABLED)
          check_cuda_error(cudaFree(*ptr));
#else
          check_cuda_error(cudaFreeAsync(*ptr, stream_));
          cudaStreamSynchronize(stream_);
#endif
        }
        check_cuda_error(getSetDevice(o_device));
        pointer_mapping_->erase(address);
      } else {
        // FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at
        // %p.", address);
      }
    }
    *ptr = nullptr;
    return;
  }

  void memSet(void *ptr, const int val, const size_t size) {
    check_cuda_error(cudaMemsetAsync(ptr, val, size, stream_));
  }
};