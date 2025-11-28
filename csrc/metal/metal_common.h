#pragma once

#include <cstdint>

// Metal kernel argument structures for paged attention
// These structures are shared between C++ and Metal Shading Language

namespace vllm {
namespace metal {

// Arguments for paged attention V1 kernel
struct PagedAttentionArgsV1 {
    // Query dimensions
    int32_t num_seqs;
    int32_t num_heads;
    int32_t head_size;
    int32_t max_num_blocks_per_seq;

    // KV cache dimensions
    int32_t num_kv_heads;
    int32_t block_size;

    // Attention parameters
    float scale;  // 1.0 / sqrt(head_size)

    // ALiBi parameters (optional)
    float alibi_slope;

    // Quantization parameters
    float kv_scale;

    // Strides for query tensor [num_seqs, num_heads, head_size]
    int64_t query_stride_0;  // stride for num_seqs dimension
    int64_t query_stride_1;  // stride for num_heads dimension

    // Strides for output tensor (same as query)
    int64_t output_stride_0;
    int64_t output_stride_1;

    // K cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    int32_t x;  // vectorization factor (typically 16)
    int64_t k_cache_stride_block;
    int64_t k_cache_stride_head;
    int64_t k_cache_stride_dim;
    int64_t k_cache_stride_token;

    // V cache layout: [num_blocks, num_kv_heads, head_size, block_size]
    int64_t v_cache_stride_block;
    int64_t v_cache_stride_head;
    int64_t v_cache_stride_dim;

    // Block tables stride
    int64_t block_tables_stride;
};

// Arguments for paged attention V2 kernel (with partitioning)
struct PagedAttentionArgsV2 {
    // All V1 arguments
    PagedAttentionArgsV1 base;

    // V2-specific parameters
    int32_t max_num_partitions;
    int32_t partition_size;  // Typically 512

    // Temporary buffers for reduction
    int64_t exp_sums_stride;
    int64_t max_logits_stride;
};

// Arguments for cache reshape kernel
struct ReshapeAndCacheArgs {
    int32_t num_tokens;
    int32_t num_heads;
    int32_t head_size;
    int32_t block_size;
    int32_t x;  // vectorization factor

    // Strides for key tensor [num_tokens, num_heads, head_size]
    int64_t key_stride_token;
    int64_t key_stride_head;

    // Strides for value tensor
    int64_t value_stride_token;
    int64_t value_stride_head;

    // K cache layout strides
    int64_t k_cache_stride_block;
    int64_t k_cache_stride_head;
    int64_t k_cache_stride_dim;
    int64_t k_cache_stride_token;

    // V cache layout strides
    int64_t v_cache_stride_block;
    int64_t v_cache_stride_head;
    int64_t v_cache_stride_dim;
};

// Arguments for block swap/copy operations
struct BlockOperationArgs {
    int32_t num_pairs;
    int32_t block_size_in_bytes;
};

// Data type enumeration
enum class DataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8_E4M3 = 4,
    FP8_E5M2 = 5
};

// Thread configuration for kernels
struct ThreadConfig {
    uint32_t threads_per_threadgroup;
    uint32_t threadgroups_x;
    uint32_t threadgroups_y;
    uint32_t threadgroups_z;
};

} // namespace metal
} // namespace vllm
