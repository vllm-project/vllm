// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// C ABI for the CUDA compression path.
//
// Python builds one flat metadata blob (int64 tensor) per (slot, kv_part)
// once per execution plan.  During serving every chunk performs exactly one
// call into this module and all per-layer work happens in C++.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// Blob layout:
//   header: 16 int64 fields (see KvsMetaHeader)
//   parts:  num_parts rows of KvsPartMeta
//   sections: total_sections rows of KvsSectionMeta
enum KvsMetaField : int64_t {
  KVS_META_NUM_PARTS = 0,
  KVS_META_ARENA_A = 1,
  KVS_META_ARENA_B = 2,
  KVS_META_AUX = 3,
  KVS_META_RAW_TAIL = 4,
  KVS_META_U8_BASE = 5,
  KVS_META_AUX_ARENA_BYTES = 6,
  KVS_META_HEADER_FIELDS = 16,
};

enum KvsPartField : int64_t {
  KVS_PART_CACHE_BASE = 0,
  KVS_PART_BLOCK_STRIDE = 1,
  KVS_PART_HEAD_STRIDE = 2,
  KVS_PART_TOKEN_STRIDE = 3,
  KVS_PART_SIGNS = 4,
  KVS_PART_HEAD_ORDER = 5,
  KVS_PART_NUM_HEADS = 6,
  KVS_PART_HEAD_SIZE = 7,
  KVS_PART_KERNEL_TOKENS = 8,
  KVS_PART_RAW_OFFSET = 9,
  KVS_PART_SCRATCH_OFFSET = 10,
  KVS_PART_U8_OFFSET = 11,
  KVS_PART_RAW_BYTES = 12,
  KVS_PART_USES_TRANSFORMER = 13,
  KVS_PART_QUANTIZED = 14,
  KVS_PART_NUM_SECTIONS = 15,
  KVS_PART_SECTION_INDEX = 16,
  KVS_PART_BLOCK_SIZE = 17,
  KVS_PART_BLOCKS_PER_LOGICAL = 18,
  KVS_PART_FIELDS = 24,
};

enum KvsSectionField : int64_t {
  KVS_SECTION_HEAD_START = 0,
  KVS_SECTION_HEAD_COUNT = 1,
  KVS_SECTION_NUM_LEVELS = 2,
  KVS_SECTION_AXIS = 3,
  KVS_SECTION_AUX_MIN_OFFSET = 4,
  KVS_SECTION_AUX_SCALE_OFFSET = 5,
  KVS_SECTION_FIELDS = 8,
};

enum KvsQuantAxis : int64_t {
  KVS_AXIS_CHANNEL = 0,
  KVS_AXIS_TOKEN = 1,
  KVS_AXIS_TENSOR = 2,
};

// Encode one chunk (one kv_part across all layers of the blob).  The caller
// has already zeroed the u8/aux regions and uploaded the block indices.
int kvs_encode_flat(int64_t meta_ptr, int64_t indices_ptr, int64_t num_blocks,
                    int64_t valid_tokens, int64_t stream_ptr);

// Decode one chunk.  The u8 stream must already hold the decoded bytes.
int kvs_decode_flat(int64_t meta_ptr, int64_t indices_ptr, int64_t num_blocks,
                    int64_t valid_tokens, int64_t u8_bytes, int64_t stream_ptr);

// Diagnostics / validation.
int kvs_ping(int value);
int kvs_test_gather_fwht(int64_t cache_base, int64_t cache_block_stride,
                         int64_t cache_head_stride, int64_t cache_token_stride,
                         int64_t indices_ptr, int64_t signs_ptr, int num_heads,
                         int head_size, int kernel_tokens, int num_blocks,
                         int64_t arena_a_ptr, int64_t stream_ptr);

#ifdef __cplusplus
}
#endif
