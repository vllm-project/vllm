/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * Common utils to be shared between Precompiled and JIT implementation.
 */
#pragma once
// NOTE: we use int32_t sequence lengths as gpt attention plugins use int32_t
// for that. XQA kernels assume all length should use uint32_t.

#include "xqa_params.h"
// #include "decoder_xqa_common.h"
#include <cassert>

// void syncAndCheck(char const* const file, int const line)
// {
//     if (true)
//     {
//         cudaGetLastError();
//         cudaDeviceSynchronize();
//     }
// }

// #define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

inline void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorName(err));
    throw std::runtime_error(cudaGetErrorName(err));
  }
}
inline int getMultiProcessorCount() {
  int device_id;
  int multi_processor_count;
  checkCuda(cudaGetDevice(&device_id));
  checkCuda(cudaDeviceGetAttribute(&multi_processor_count,
                                   cudaDevAttrMultiProcessorCount, device_id));
  return multi_processor_count;
}

template <typename T>
HOST_DEVICE_FUNC constexpr inline T divUp(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
HOST_DEVICE_FUNC constexpr inline T roundUp(T a, T b) {
  return divUp(a, b) * b;
}

constexpr inline uint32_t exactDiv(uint32_t a, uint32_t b) {
  assert(a % b == 0);
  return a / b;
}

using KVCachePageIndex = int32_t;
using SeqLenDataType = uint32_t;
struct KVCacheListParams {
  void const* pool = nullptr;
  KVCachePageIndex const* block_indices =
      nullptr;  // shape: [batchSize][beamWidth][2][maxNbPagesPerSeq].
  SeqLenDataType const* sequence_lengths =
      nullptr;  // shape: [batchSize][beamWidth] (for compatibility)
  // NOTE: max_num_blocks_per_sequence for paged kv cache.
  uint32_t capacity = 0;

  KVCacheListParams(void const* _pool, KVCachePageIndex const* _block_indices,
                    SeqLenDataType const* _sequence_lengths, uint32_t _capacity)
      : pool(_pool),
        block_indices(_block_indices),
        sequence_lengths(_sequence_lengths),
        capacity(_capacity) {}

  KVCacheListParams() = default;
};

struct XQALaunchParam {
  uint32_t num_k_heads;
  void* output;
  // void const* qkv;
  KVCacheListParams kvCacheParams;
  uint32_t batch_size;
  float const* kv_scale_quant_orig = nullptr;
  int* cu_seq_lens = nullptr;
  uint32_t* semaphores = nullptr;
  void* scratch = nullptr;
};

struct XQAKernelLoadHashKey {
  Data_type data_type;
  unsigned int sm;

  bool operator==(XQAKernelLoadHashKey const& other) const {
    return data_type == other.data_type && sm == other.sm;
  }
};

struct XQAKernelLoadHasher {
  size_t operator()(XQAKernelLoadHashKey const& s) const {
    size_t key = s.data_type;
    key <<= 16;
    key ^= s.sm;
    return key;
  }
};

struct XQAKernelRuntimeHashKey {
  Data_type kv_data_type;
  unsigned int head_size;
  unsigned int beam_size;
  unsigned int num_q_heads_per_kv;
  unsigned int m_tilesize;
  unsigned int tokens_per_page;
  bool paged_kv_cache;
  bool multi_query_tokens;

  bool operator==(XQAKernelRuntimeHashKey const& other) const {
    return kv_data_type == other.kv_data_type && head_size == other.head_size &&
           num_q_heads_per_kv == other.num_q_heads_per_kv &&
           beam_size == other.beam_size &&
           multi_query_tokens == other.multi_query_tokens &&
           m_tilesize == other.m_tilesize &&
           tokens_per_page == other.tokens_per_page &&
           paged_kv_cache == other.paged_kv_cache;
  }
};
std::ostream& operator<<(std::ostream& os, const XQAKernelRuntimeHashKey& key);

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(
    XQAParams const& xqaParams);

void buildXQALaunchParams(XQALaunchParam& launchParams, XQAParams const& params,
                          KVCacheListParams kv_cache_buffer);

struct XQAKernelRuntimeHasher {
  size_t operator()(XQAKernelRuntimeHashKey const& s) const {
    size_t key = s.kv_data_type;
    key <<= 16;
    key ^= s.head_size;
    key <<= 8;
    key ^= s.num_q_heads_per_kv;
    key <<= 8;
    key ^= s.beam_size;
    key <<= 6;
    key ^= s.m_tilesize;
    key <<= 10;
    key ^= s.tokens_per_page;
    key <<= 1;
    key ^= s.paged_kv_cache;
    key <<= 1;
    key ^= s.multi_query_tokens;
    return key;
  }
};

// XQA kernel can be uniquely identified by (LoadHashKey, RuntimeHashKey).
struct XQAKernelFullHashKey {
  XQAKernelLoadHashKey load_key;
  XQAKernelRuntimeHashKey runtime_key;

  XQAKernelFullHashKey() = default;

  XQAKernelFullHashKey(XQAKernelLoadHashKey const& load_key,
                       XQAKernelRuntimeHashKey const& runtime_key)
      : load_key(load_key), runtime_key(runtime_key) {}

  XQAKernelFullHashKey(void const* buffer, size_t buffer_size) {
    TORCH_CHECK(sizeof(*this) <= buffer_size);
    memcpy(this, buffer, sizeof(*this));
  }

  bool operator==(XQAKernelFullHashKey const& other) const {
    return load_key == other.load_key && runtime_key == other.runtime_key;
  }

  size_t getSerializationSize() const { return sizeof(*this); }

  void serialize(void* buffer, size_t buffer_size) const {
    TORCH_CHECK(sizeof(*this) <= buffer_size);
    memcpy(buffer, this, sizeof(*this));
  }
};

struct XQAKernelFullHasher {
  size_t operator()(XQAKernelFullHashKey const& s) const {
    return XQAKernelLoadHasher()(s.load_key) ^
           XQAKernelRuntimeHasher()(s.runtime_key);
  }
};

std::uintptr_t constexpr kCudaMemAlign = 128;

inline int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t*)addr;
}

inline int8_t* nextWorkspacePtrCommon(int8_t* ptr,
                                      uintptr_t previousWorkspaceSize,
                                      uintptr_t const alignment) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t*)addr, alignment);
}

inline int8_t* nextWorkspacePtrWithAlignment(
    int8_t* ptr, uintptr_t previousWorkspaceSize,
    uintptr_t const alignment = kCudaMemAlign) {
  return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, alignment);
}

template <typename T>
std::optional<T> getGlobalVar(CUmodule hmod, char const* const name,
                              bool required = false) {
  T* pVar = nullptr;
  size_t size = 0;
  auto const error = cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&pVar),
                                       &size, hmod, name);
  T ret;
  switch (error) {
    case CUDA_SUCCESS:
      TORCH_CHECK(size == sizeof(T));
      CUDACHECK(cudaMemcpy(&ret, pVar, size, cudaMemcpyDeviceToHost));
      break;
    case CUDA_ERROR_NOT_FOUND:
      if (!required) {
        return std::nullopt;
      }
      [[fallthrough]];
    default:
      cuErrCheck(("Failed to retrieve global variable from cubin.", error));
  }
  return std::optional<T>{std::move(ret)};
}

inline int computeMultiBlockCount(XQAParams const& xqaParams, int batch_size,
                                  int multiprocessor_count) {
  int multi_block_count = 1;
  int num_kv_heads = xqaParams.num_kv_heads;
  int history_length = xqaParams.timestep;

  int32_t const maxNbSubSeq = kXQA_MAX_NUM_SUB_SEQ;

  multi_block_count = history_length / kMinHistoryTokensPerBlock;
  // avoid using too many blocks for one sequence, otherwise the final reduction
  // may dominate.
  multi_block_count = std::min(
      multi_block_count,
      static_cast<int>(std::round(std::sqrt(multi_block_count * 8.F))));
  multi_block_count = std::max(multi_block_count, 1);
  // adjust to kTargetWaveFactor, as already initialized using
  // kMinHistoryTokensPerBlock, only need to decrease.
  double wave_count = (double)batch_size * num_kv_heads * multi_block_count /
                      (double)multiprocessor_count;
  double adj_factor = wave_count / (double)kTargetWaveFactor;
  if (adj_factor > 1.0) {
    multi_block_count = floor(multi_block_count / adj_factor);
  }
  multi_block_count = std::max(multi_block_count, 1);

  // Add limitation due to reserved workspace size.
  // When batch_size is large, multi-block is useless anyway. So large workspace
  // is not useful and we can set a hard limit for workspace size (computed from
  // maxNbSubSeq).
  multi_block_count =
      std::max(std::min(multi_block_count, maxNbSubSeq / batch_size), 1);

  TORCH_CHECK(multi_block_count >= 1,
              "MultiBlock count should be larger than 1");
  TORCH_CHECK(
      multi_block_count == 1 || batch_size * multi_block_count <= maxNbSubSeq,
      "Insufficient workspace");
  return multi_block_count;
}
