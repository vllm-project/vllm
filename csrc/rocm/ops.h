#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

// META3-2: bf16/fp16 wvSplitK that fuses a silu_and_mul preamble.
// in_b is laid out as [N=1, 2*K] = [gate(K) | up(K)]; the kernel writes
// silu(gate)*up into LDS before the GEMM.  Output is [N=1, M].
torch::Tensor wvSplitK_fused_silu_mul(const at::Tensor& in_a,
                                      const at::Tensor& in_b,
                                      const std::optional<at::Tensor>& in_bias,
                                      const int64_t CuCount);

// META3-2 Phase 2: bf16/fp16 wvSplitK that fuses BOTH the silu_and_mul
// preamble and a per-token scalar (gate) mul epilogue.
// in_b is [N=1, 2*K] = [gate(K) | up(K)].
// in_gate is [N=1, 1] (or [N]); the kernel multiplies each output element
// by in_gate[n] before writing to C.  Output is [N=1, M].
torch::Tensor wvSplitK_fused_silu_gate_mul(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_gate,
    const std::optional<at::Tensor>& in_bias, const int64_t CuCount);

torch::Tensor wvSplitK_int8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitK_w8a8(const at::Tensor& in_a, const at::Tensor& in_b,
                            const at::Tensor& in_w_scale,
                            const std::optional<at::Tensor>& in_a_scale,
                            const std::optional<at::Tensor>& in_bias,
                            const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

void fused_moe_wvSplitK_int4_gemm(torch::Tensor a, torch::Tensor w,
                                  torch::Tensor scales, torch::Tensor c,
                                  torch::Tensor expert_ids,
                                  int64_t block_size_m, int64_t CuCount,
                                  int64_t group_size, torch::Tensor zero_points,
                                  torch::Tensor sorted_token_ids,
                                  int64_t top_k);

#ifdef VLLM_SKINNY_GEMM_SWEEP_BF16
torch::Tensor wvSplitK_sweep(const at::Tensor& in_a, const at::Tensor& in_b,
                             const std::optional<at::Tensor>& in_bias,
                             const int64_t CuCount, const int64_t ytile,
                             const int64_t unrl, const int64_t achunk,
                             const int64_t wvprgrp);
#endif

#ifdef VLLM_SKINNY_GEMM_SWEEP
torch::Tensor wvSplitK_int8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp);

torch::Tensor wvSplitK_int4g_sweep(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const int64_t CuCount, const int64_t group_size, const int64_t ytile,
    const int64_t unrl, const int64_t achunk, const int64_t wvprgrp);

torch::Tensor wvSplitK_int4g_hf_sweep(
    const at::Tensor& in_a, const at::Tensor& in_b, const at::Tensor& in_scale,
    const int64_t CuCount, const int64_t group_size, const int64_t ytile,
    const int64_t unrl, const int64_t achunk, const int64_t wvprgrp);

void fused_moe_wvSplitK_int4_gemm_sweep(
    torch::Tensor a, torch::Tensor w, torch::Tensor scales, torch::Tensor c,
    torch::Tensor expert_ids, int64_t block_size_m, int64_t CuCount,
    int64_t group_size, torch::Tensor zero_points,
    torch::Tensor sorted_token_ids, int64_t top_k, bool fuse_silu_mul,
    int64_t ytile, int64_t unrl, int64_t achunk, int64_t wvprgrp);

torch::Tensor wvSplitK_w8a8_sweep(const at::Tensor& in_a,
                                  const at::Tensor& in_b,
                                  const at::Tensor& in_w_scale,
                                  const std::optional<at::Tensor>& in_a_scale,
                                  const std::optional<at::Tensor>& in_bias,
                                  const int64_t CuCount, const int64_t ytile,
                                  const int64_t unrl, const int64_t achunk,
                                  const int64_t wvprgrp);
#endif

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
