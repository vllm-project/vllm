#pragma once

#include <torch/extension.h>
#include "metal_common.h"

namespace vllm {
namespace metal {

// Paged attention V1 launcher
void paged_attention_v1(
    torch::Tensor& out,              // [num_seqs, num_heads, head_size]
    torch::Tensor& query,            // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache,      // [num_blocks, num_kv_heads, head_size, block_size]
    torch::Tensor& block_tables,     // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,         // [num_seqs]
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& kv_cache_scales);

// Paged attention V2 launcher (with partitioning)
void paged_attention_v2(
    torch::Tensor& out,              // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,         // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,       // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,          // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,            // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int num_kv_heads,
    float scale,
    int block_size,
    int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& kv_cache_scales);

// Reshape and cache launcher
void reshape_and_cache(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache,      // [num_blocks, num_kv_heads, head_size, block_size]
    torch::Tensor& slot_mapping);    // [num_tokens]

// Copy blocks launcher
void copy_blocks(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& src_to_dst);      // [num_pairs, 2]

// Swap blocks launcher
void swap_blocks(
    torch::Tensor& src_cache,
    torch::Tensor& dst_cache,
    torch::Tensor& src_to_dst);      // [num_pairs, 2]

// Helper function to get kernel name based on parameters
std::string get_paged_attention_kernel_name(
    const std::string& base_name,
    torch::ScalarType query_type,
    torch::ScalarType cache_type,
    int head_size,
    int block_size,
    int num_threads);

} // namespace metal
} // namespace vllm
