#pragma once

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>
#include <vector>

void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        std::optional<torch::Tensor> weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t hidden_size,
    int64_t intermediate_size, int64_t group_size,
    bool apply_router_weight_on_input, int64_t activation_kind);
