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

__host__ __device__ inline int prefix_splits(int ctx, int splits) {
  const int used = ctx <= 0 ? 0 : 1 + (ctx - 1) / 256;
  return used < splits ? used : splits;
}

__host__ __device__ inline int bounded_prefix(int length, int chunk,
                                              int64_t capacity) {
  const int64_t prefix = int64_t(length) - chunk;
  return prefix <= 0 ? 0 : int(prefix < capacity ? prefix : capacity);
}

int launch_prefix_tiles(const int8_t* query, const float* query_scale,
                        PagedCache cache, const int32_t* table,
                        int64_t table_stride, int64_t prefix_capacity,
                        const int32_t* starts, const int32_t* lengths,
                        float* partials, float scale, int requests, int heads,
                        int tokens, int splits, int format, hipStream_t stream);
int launch_dense_prefill(float* out, const _Float16* query,
                         const _Float16* keys, const _Float16* residual,
                         const float* errors, const _Float16* values,
                         const int32_t* starts, const int32_t* lengths,
                         int requests, int tokens, int kv_heads, int group,
                         int capacity, float scale, uint8_t* repair,
                         hipStream_t stream);
int launch_chunk_only(float* out, Activation query, Activation keys,
                      Activation values, void* packed_chunk, float* chunk_lse,
                      const int32_t* starts, int requests, int tokens,
                      int kv_heads, int group, float scale, int dtype,
                      hipStream_t stream);
int launch_prefill_streamed(float* out, Activation query, Activation key,
                            Activation value, PagedCache cache,
                            const int32_t* table, int64_t table_stride,
                            int64_t prefix_capacity, const int32_t* starts,
                            const int32_t* lengths, int requests,
                            const uint32_t* signs, int tokens, int heads,
                            float scale, int dtype, int format, int splits,
                            float* partials, int8_t* staging, float* chunk_lse,
                            uint8_t* repair, hipStream_t stream);

int launch_prefill_expanded(float* out, void* keys, void* values,
                            const void* query_fp16, Activation query,
                            Activation key, Activation value, PagedCache cache,
                            const int32_t* table, int64_t table_stride,
                            const int32_t* starts, const int32_t* lengths,
                            int requests, int capacity, const uint32_t* signs,
                            int tokens, int heads, float scale, int dtype,
                            int format, void* correction, uint8_t* repair,
                            hipStream_t stream);

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
