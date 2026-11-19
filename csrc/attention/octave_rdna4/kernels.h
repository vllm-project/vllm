// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace octave_rdna4 {

struct Activation {
  const void* data;
  int64_t token_stride, head_stride;
};

struct PagedCache {
  uint8_t* data;
  int64_t page_stride, head_stride, token_stride;
  int page_size, heads;

  __device__ uint8_t* at(int page, int head, int offset) const {
    return data + page * page_stride + head * head_stride +
           offset * token_stride;
  }
};

struct Shadow {
  void* data;
  int32_t* tags;
  int32_t* locks;
  int entries;
};

int cache_slot_bytes(int64_t format);

int launch_store(Activation key, Activation value, PagedCache cache,
                 const int64_t* slots, const uint32_t* signs, int tokens,
                 int dtype, int format, hipStream_t stream);

int launch_decode(float* out, float* partials, Activation query,
                  PagedCache cache, const int32_t* table, int64_t table_stride,
                  const int32_t* q_to_req, const int32_t* q_to_klen,
                  const uint32_t* signs, int tokens, int heads, int splits,
                  float scale, int dtype, int format, hipStream_t stream);

int launch_prefill(float* out, float* partials, Activation query,
                   Activation key, Activation value, PagedCache cache,
                   const int32_t* table, int64_t table_stride,
                   const int32_t* query_starts, const int32_t* seq_lens,
                   int requests, const uint32_t* signs, int tokens, int heads,
                   int splits, float scale, int dtype, int format,
                   hipStream_t stream);

int launch_prefill_expanded(float* out, void* keys, void* values,
                            const void* query_fp16, Activation query,
                            Activation key, Activation value, PagedCache cache,
                            const int32_t* table, int64_t table_stride,
                            const int32_t* starts, const int32_t* lengths,
                            int requests, int capacity, const uint32_t* signs,
                            int tokens, int heads, float scale, int dtype,
                            int format, hipStream_t stream);

namespace geometry {

decltype(octave_rdna4::launch_store) launch_store;
decltype(octave_rdna4::launch_decode) launch_decode;
decltype(octave_rdna4::launch_prefill) launch_prefill;

int launch_window_store(Activation key, Activation value, const int64_t* slots,
                        const int32_t* positions, Shadow recent, Shadow sinks,
                        int tokens, int heads, int dtype, hipStream_t stream);

int launch_decode_sparse(float* out, float* partial, Activation q,
                         PagedCache cache, const int32_t* table,
                         int64_t table_stride, int columns, int requests,
                         int pages, const int32_t* mapping,
                         const int32_t* indices, int64_t index_stride, int topk,
                         const uint32_t* signs, const int32_t* positions,
                         Shadow recent, Shadow sinks, int tokens, int heads,
                         int splits, float scale, int dtype, int format,
                         hipStream_t stream);

}  // namespace geometry
}  // namespace octave_rdna4
