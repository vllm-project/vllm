#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

void swap_blocks(torch::stable::Tensor& src, torch::stable::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const torch::stable::Tensor& block_mapping);

void reshape_and_cache(torch::stable::Tensor& key, torch::stable::Tensor& value,
                       torch::stable::Tensor& key_cache,
                       torch::stable::Tensor& value_cache,
                       torch::stable::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::stable::Tensor& k_scale,
                       torch::stable::Tensor& v_scale);

void reshape_and_cache_flash(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, const std::string& kv_cache_dtype,
    torch::stable::Tensor& k_scale, torch::stable::Tensor& v_scale);

void concat_and_cache_mla(torch::stable::Tensor& kv_c,
                          torch::stable::Tensor& k_pe,
                          torch::stable::Tensor& kv_cache,
                          torch::stable::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          torch::stable::Tensor& scale);

// NOTE: k_pe and kv_c order is flipped compared to concat_and_cache_mla
void concat_and_cache_mla_rope_fused(
    torch::stable::Tensor& positions, torch::stable::Tensor& q_pe,
    torch::stable::Tensor& k_pe, torch::stable::Tensor& kv_c,
    torch::stable::Tensor& rope_cos_sin_cache, bool rope_is_neox,
    torch::stable::Tensor& kv_cache_slot_mapping,
    torch::stable::Tensor& kv_cache, const std::string& kv_cache_dtype,
    torch::stable::Tensor& kv_cache_quant_scale);

// Just for unittest
void convert_fp8(torch::stable::Tensor& dst_cache,
                 torch::stable::Tensor& src_cache, double scale,
                 const std::string& kv_cache_dtype);

void gather_and_maybe_dequant_cache(
    torch::stable::Tensor const&
        src_cache,                     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::stable::Tensor const& dst,  // [TOT_TOKENS, ENTRIES...]
    torch::stable::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::stable::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::stable::Tensor const& scale,
    std::optional<torch::stable::Tensor> seq_starts = std::nullopt);

// TODO(hc): cp_gather_cache need support scaled kvcahe in the future.
void cp_gather_cache(
    torch::stable::Tensor const&
        src_cache,                     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::stable::Tensor const& dst,  // [TOT_TOKENS, ENTRIES...]
    torch::stable::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::stable::Tensor> seq_starts = std::nullopt);

// Gather and upconvert FP8 KV cache to BF16 workspace
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::stable::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::stable::Tensor const& dst,          // [TOT_TOKENS, 576]
    torch::stable::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& seq_lens,     // [BATCH]
    torch::stable::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size);

// Indexer K quantization and cache function
void indexer_k_quant_and_cache(
    torch::stable::Tensor& k,         // [num_tokens, head_dim]
    torch::stable::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::stable::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,             // quantization block size
    const std::string& scale_fmt);

// Extract function to gather quantized K cache
void cp_gather_indexer_k_quant_cache(
    const torch::stable::Tensor&
        kv_cache,                  // [num_blocks, block_size, cache_stride]
    torch::stable::Tensor& dst_k,  // [num_tokens, head_dim]
    torch::stable::Tensor&
        dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::stable::Tensor& block_table,   // [batch_size, num_blocks]
    const torch::stable::Tensor& cu_seq_lens);  // [batch_size + 1]
