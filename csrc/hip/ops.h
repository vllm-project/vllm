#pragma once

#include <optional>
#include <torch/library.h>

void paged_attention_custom(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale, 
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size, 
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype)