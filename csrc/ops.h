#pragma once

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

// rms_norm and fused_add_rms_norm declarations also exist in
// csrc/libtorch_stable/ops.h (torch::stable ABI for CUDA). They remain here
// because the CPU build still uses these torch::Tensor declarations.
void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        std::optional<torch::Tensor> weight, double epsilon);

// rotary_embedding also exist in csrc/libtorch_stable/ops.h (torch::stable
// ABI for CUDA). It remains here because the CPU build still uses these
// torch::Tensor declarations.
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit,
                        double alpha = 1.0, double beta = 0.0);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void relu_squared(torch::Tensor& out, torch::Tensor& input);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

// RDNA3 INT8 per-token-head paged prefill attention (gfx1100).
void paged_prefill_attn_rdna3_int8(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal);

// RDNA3 INT4 per-token-head paged prefill attention (gfx1100).
void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor rht_signs,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal);

// Fused RHT + INT4 quantize + nibble pack for RDNA3 reshape_and_cache.
void reshape_cache_int4_rdna3(torch::Tensor key, torch::Tensor value,
                              torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor k_scale_cache,
                              torch::Tensor v_scale_cache,
                              torch::Tensor rht_signs,
                              torch::Tensor slot_mapping);

// Inplace RHT butterfly for decode Q rotation / output unrotation.
void rht_rotate_inplace_rdna3(torch::Tensor data, torch::Tensor rht_signs,
                              bool inverse, double post_scale);

// HIP split-KV decode attention for INT4 per-token-head (RDNA3).
void pth_decode_int4_rdna3(torch::Tensor out, torch::Tensor query,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor k_scale_cache,
                           torch::Tensor v_scale_cache, torch::Tensor rht_signs,
                           torch::Tensor block_table, torch::Tensor q_to_req,
                           torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                           double sm_scale, int64_t num_kv_splits);

void pth_decode_int8_rdna3(torch::Tensor out, torch::Tensor query,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor k_scale_cache,
                           torch::Tensor v_scale_cache,
                           torch::Tensor block_table, torch::Tensor q_to_req,
                           torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                           double sm_scale, int64_t num_kv_splits);

// Octave KV cache (ROCm). fmt: V bits (3 or 4), plus 16 for compact K.
void octave_cache_store(torch::Tensor key, torch::Tensor value,
                        torch::Tensor cache, torch::Tensor slot_mapping,
                        torch::Tensor k_signs, torch::Tensor v_signs,
                        int64_t fmt);

void octave_decode(torch::Tensor out, torch::Tensor query, torch::Tensor cache,
                   torch::Tensor block_table, torch::Tensor q_to_req,
                   torch::Tensor q_to_klen, torch::Tensor mid_o,
                   torch::Tensor k_signs, torch::Tensor v_signs,
                   double sm_scale, int64_t num_kv_splits, int64_t fmt,
                   int64_t query_group, bool use_wmma);

void octave_decode_sparse(torch::Tensor out, torch::Tensor query,
                          torch::Tensor cache, torch::Tensor block_table,
                          torch::Tensor q_to_req, torch::Tensor indices,
                          torch::Tensor mid_o, torch::Tensor k_signs,
                          torch::Tensor v_signs, double sm_scale,
                          int64_t num_kv_splits, int64_t fmt,
                          const std::optional<torch::Tensor>& positions,
                          const std::optional<torch::Tensor>& wtab,
                          const std::optional<torch::Tensor>& wtags,
                          const std::optional<torch::Tensor>& stab,
                          const std::optional<torch::Tensor>& stags,
                          int64_t window);

void octave_window_store(torch::Tensor key, torch::Tensor value,
                         torch::Tensor slot_mapping, torch::Tensor positions,
                         torch::Tensor wtab, torch::Tensor wtags,
                         torch::Tensor wlocks, torch::Tensor stab,
                         torch::Tensor stags, torch::Tensor slocks);

void octave_rotate(torch::Tensor x, torch::Tensor signs, bool k_layout,
                   bool inverse);

void octave_prefill(torch::Tensor out, torch::Tensor q, torch::Tensor k,
                    torch::Tensor v, torch::Tensor cache,
                    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                    torch::Tensor seq_lens, int64_t max_query_len,
                    double sm_scale, int64_t fmt);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t hidden_size,
    int64_t intermediate_size, int64_t group_size,
    bool apply_router_weight_on_input, int64_t activation_kind);

using fptr_t = int64_t;
#ifdef USE_ROCM
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);
int64_t qr_max_size();
#endif
