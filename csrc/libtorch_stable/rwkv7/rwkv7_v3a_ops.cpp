// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <torch/all.h>
#include <torch/library.h>
#include <vector>

#define RWKV7_LAYER_NORM_EPS_SCHEMA "1e-5"

torch::Tensor layer_norm_f16_cuda(torch::Tensor x, torch::Tensor weight,
                                  torch::Tensor bias, double eps);
torch::Tensor emb_ln0_bf16_to_f16_cuda(torch::Tensor emb, torch::Tensor weight,
                                       torch::Tensor bias, double eps);
torch::Tensor linear_f16_cuda(torch::Tensor x, torch::Tensor weight,
                              bool allow_fp16_accumulation);
torch::Tensor linear_f16_m1_splitk_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor linear_t_f16_cuda(torch::Tensor x, torch::Tensor weight_t);
torch::Tensor linear_t_act_f16_cuda(torch::Tensor x, torch::Tensor weight_t,
                                    int64_t act);
torch::Tensor linear_t_vres_f16_cuda(torch::Tensor x, torch::Tensor weight_t,
                                     torch::Tensor v, torch::Tensor v_first,
                                     torch::Tensor v0);
std::vector<torch::Tensor> linear_wag_rank_in_f16_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor w1_t,
    torch::Tensor a1_t, torch::Tensor g1_t);
std::vector<torch::Tensor> linear_wagv_rank_in_f16_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t,
    torch::Tensor v1_t);
std::vector<torch::Tensor> linear_wag_rank_out_f16_cuda(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor w2_t,
    torch::Tensor a2_t, torch::Tensor g2_t);
std::vector<torch::Tensor> linear_wagv_rank_out_f16_cuda(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t,
    torch::Tensor v2_t, torch::Tensor v, torch::Tensor v_first,
    torch::Tensor v0);
torch::Tensor add_f16_cuda(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> add_layer_norm_f16_cuda(torch::Tensor x,
                                                   torch::Tensor residual,
                                                   torch::Tensor weight,
                                                   torch::Tensor bias,
                                                   double eps);
torch::Tensor add_last_layer_norm_f16_cuda(torch::Tensor x,
                                           torch::Tensor residual,
                                           torch::Tensor weight,
                                           torch::Tensor bias, double eps);
std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps);
std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_slots_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k,
    torch::Tensor slot_indices, double eps);
std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, double eps);
std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_slots_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, torch::Tensor slot_indices, double eps);
void advance_i32_cuda(torch::Tensor x, int64_t amount);
void advance_i32_slots_cuda(torch::Tensor x, torch::Tensor slot_indices,
                            int64_t amount);
void advance_i32_varlen_cuda(torch::Tensor x, torch::Tensor query_start_loc,
                             torch::Tensor slot_indices);

namespace {

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_i32_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
}

void check_bf16_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

torch::Tensor layer_norm_f16(torch::Tensor x, torch::Tensor weight,
                             torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return layer_norm_f16_cuda(x, weight, bias, eps);
}

torch::Tensor emb_ln0_bf16_to_f16(torch::Tensor emb, torch::Tensor weight,
                                  torch::Tensor bias, double eps) {
  check_bf16_cuda_contig(emb, "emb");
  check_bf16_cuda_contig(weight, "weight");
  check_bf16_cuda_contig(bias, "bias");
  TORCH_CHECK(emb.dim() == 2, "emb must have shape [V, C]");
  const int64_t c = emb.size(1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return emb_ln0_bf16_to_f16_cuda(emb, weight, bias, eps);
}

torch::Tensor linear_f16(torch::Tensor x, torch::Tensor weight,
                         bool allow_fp16_accumulation) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16 shape mismatch");
  return linear_f16_cuda(x, weight, allow_fp16_accumulation);
}

torch::Tensor linear_f16_m1_splitk(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0),
              "linear_f16_m1_splitk shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk requires M=1");
  return linear_f16_m1_splitk_cuda(x, weight);
}

torch::Tensor linear_t_f16(torch::Tensor x, torch::Tensor weight_t) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_f16 shape mismatch");
  return linear_t_f16_cuda(x, weight_t);
}

torch::Tensor linear_t_act_f16(torch::Tensor x, torch::Tensor weight_t,
                               int64_t act) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1),
              "linear_t_act_f16 shape mismatch");
  TORCH_CHECK(act == 1 || act == 2, "act must be 1=tanh or 2=sigmoid");
  return linear_t_act_f16_cuda(x, weight_t, act);
}

torch::Tensor linear_t_vres_f16(torch::Tensor x, torch::Tensor weight_t,
                                torch::Tensor v, torch::Tensor v_first,
                                torch::Tensor v0) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1),
              "linear_t_vres_f16 shape mismatch");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == weight_t.size(0),
              "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == weight_t.size(0),
              "v0 shape mismatch");
  return linear_t_vres_f16_cuda(x, weight_t, v, v_first, v0);
}

std::vector<torch::Tensor> linear_wag_rank_in_f16(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor w1_t,
    torch::Tensor a1_t, torch::Tensor g1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes(),
              "xw/xa/xg shape mismatch");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) &&
                  xw.size(-1) == g1_t.size(1),
              "rank-in K mismatch");
  return linear_wag_rank_in_f16_cuda(xw, xa, xg, w1_t, a1_t, g1_t);
}

std::vector<torch::Tensor> linear_wagv_rank_in_f16(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t,
    torch::Tensor v1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(xv, "xv");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  check_half_cuda_contig(v1_t, "v1_t");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes() &&
                  xw.sizes() == xv.sizes(),
              "xw/xa/xg/xv shape mismatch");
  TORCH_CHECK(
      w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2 && v1_t.dim() == 2,
      "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) &&
                  xw.size(-1) == g1_t.size(1) && xw.size(-1) == v1_t.size(1),
              "rank-in K mismatch");
  return linear_wagv_rank_in_f16_cuda(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t);
}

std::vector<torch::Tensor> linear_wag_rank_out_f16(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor w2_t,
    torch::Tensor a2_t, torch::Tensor g2_t) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim(),
              "w1/a1/g1 dim mismatch");
  TORCH_CHECK(
      w1.sizes().slice(0, w1.dim() - 1) == a1.sizes().slice(0, a1.dim() - 1),
      "w1/a1 batch mismatch");
  TORCH_CHECK(
      w1.sizes().slice(0, w1.dim() - 1) == g1.sizes().slice(0, g1.dim() - 1),
      "w1/g1 batch mismatch");
  TORCH_CHECK(w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0),
              "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1), "w rank mismatch");
  TORCH_CHECK(a1.size(-1) == a2_t.size(1), "a rank mismatch");
  TORCH_CHECK(g1.size(-1) == g2_t.size(1), "g rank mismatch");
  return linear_wag_rank_out_f16_cuda(w1, a1, g1, w2_t, a2_t, g2_t);
}

std::vector<torch::Tensor> linear_wagv_rank_out_f16(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t,
    torch::Tensor v2_t, torch::Tensor v, torch::Tensor v_first,
    torch::Tensor v0) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(v1, "v1");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  check_half_cuda_contig(v2_t, "v2_t");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim() &&
                  v1.dim() == w1.dim(),
              "rank dim mismatch");
  TORCH_CHECK(
      w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2 && v2_t.dim() == 2,
      "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0) &&
                  w2_t.size(0) == v2_t.size(0),
              "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1) && a1.size(-1) == a2_t.size(1) &&
                  g1.size(-1) == g2_t.size(1) && v1.size(-1) == v2_t.size(1),
              "rank mismatch");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == w2_t.size(0), "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == w2_t.size(0), "v0 shape mismatch");
  return linear_wagv_rank_out_f16_cuda(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t,
                                       v, v_first, v0);
}

torch::Tensor add_f16(torch::Tensor x, torch::Tensor y) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(x.sizes() == y.sizes(), "add_f16 shape mismatch");
  return add_f16_cuda(x, y);
}

std::vector<torch::Tensor> add_layer_norm_f16(torch::Tensor x,
                                              torch::Tensor residual,
                                              torch::Tensor weight,
                                              torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

torch::Tensor add_last_layer_norm_f16(torch::Tensor x, torch::Tensor residual,
                                      torch::Tensor weight, torch::Tensor bias,
                                      double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_last_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,T,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_last_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_cmix_mix_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "add_layer_norm_cmix_mix_f16 requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == c, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_cuda(x, residual, shift_state, weight,
                                          bias, x_k, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_slots(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k,
    torch::Tensor slot_indices, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  check_i32_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_cmix_mix_f16_slots x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "add_layer_norm_cmix_mix_f16_slots requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) > 0 &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.size(0) == x.size(0),
              "slot_indices shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == c, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_slots_cuda(
      x, residual, shift_state, weight, bias, x_k, slot_indices, eps);
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_tmix_mix6_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "add_layer_norm_tmix_mix6_f16 requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == c && x_w.numel() == c && x_k.numel() == c &&
                  x_v.numel() == c && x_a.numel() == c && x_g.numel() == c,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_cuda(x, residual, shift_state, weight,
                                           bias, x_r, x_w, x_k, x_v, x_a, x_g,
                                           eps);
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_slots(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, torch::Tensor slot_indices, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  check_i32_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_tmix_mix6_f16_slots x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "add_layer_norm_tmix_mix6_f16_slots requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) > 0 &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.size(0) == x.size(0),
              "slot_indices shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == c && x_w.numel() == c && x_k.numel() == c &&
                  x_v.numel() == c && x_a.numel() == c && x_g.numel() == c,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_slots_cuda(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g,
      slot_indices, eps);
}

void advance_i32(torch::Tensor x, int64_t amount) {
  check_i32_cuda_contig(x, "x");
  TORCH_CHECK(x.dim() == 1, "x must have shape [B]");
  advance_i32_cuda(x, amount);
}

void advance_i32_slots(torch::Tensor x, torch::Tensor slot_indices,
                       int64_t amount) {
  check_i32_cuda_contig(x, "x");
  check_i32_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(x.dim() == 1, "x must have shape [slots]");
  TORCH_CHECK(slot_indices.dim() == 1, "slot_indices must have shape [B]");
  advance_i32_slots_cuda(x, slot_indices, amount);
}

void advance_i32_varlen(torch::Tensor x, torch::Tensor query_start_loc,
                        torch::Tensor slot_indices) {
  check_i32_cuda_contig(x, "x");
  check_i32_cuda_contig(query_start_loc, "query_start_loc");
  check_i32_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(x.dim() == 1, "x must have shape [slots]");
  TORCH_CHECK(slot_indices.dim() == 1, "slot_indices must have shape [B]");
  TORCH_CHECK(query_start_loc.dim() == 1 &&
                  query_start_loc.size(0) == slot_indices.size(0) + 1,
              "query_start_loc must have shape [B+1]");
  advance_i32_varlen_cuda(x, query_start_loc, slot_indices);
}

}  // namespace

TORCH_LIBRARY(rwkv7_v3a_ops, m) {
  m.def(
      "layer_norm_f16(Tensor x, Tensor weight, Tensor bias, float "
      "eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor");
  m.def(
      "emb_ln0_bf16_to_f16(Tensor emb, Tensor weight, Tensor bias, float "
      "eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor");
  m.def(
      "linear_f16(Tensor x, Tensor weight, bool allow_fp16_accumulation=False) "
      "-> Tensor");
  m.def("linear_f16_m1_splitk(Tensor x, Tensor weight) -> Tensor");
  m.def("linear_t_f16(Tensor x, Tensor weight_t) -> Tensor");
  m.def("linear_t_act_f16(Tensor x, Tensor weight_t, int act) -> Tensor");
  m.def(
      "linear_t_vres_f16(Tensor x, Tensor weight_t, Tensor v, Tensor v_first, "
      "Tensor v0) -> Tensor");
  m.def(
      "linear_wag_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor w1_t, "
      "Tensor a1_t, Tensor g1_t) -> Tensor[]");
  m.def(
      "linear_wagv_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor xv, "
      "Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]");
  m.def(
      "linear_wag_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor w2_t, "
      "Tensor a2_t, Tensor g2_t) -> Tensor[]");
  m.def(
      "linear_wagv_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor v1, "
      "Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, Tensor v, Tensor "
      "v_first, Tensor v0) -> Tensor[]");
  m.def("add_f16(Tensor x, Tensor y) -> Tensor");
  m.def(
      "add_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor "
      "bias, float eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor[]");
  m.def(
      "add_last_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, "
      "Tensor bias, float eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor");
  m.def(
      "add_layer_norm_cmix_mix_f16(Tensor x, Tensor residual, Tensor(a!) "
      "shift_state, Tensor weight, Tensor bias, Tensor x_k, float "
      "eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor[]");
  m.def(
      "add_layer_norm_cmix_mix_f16_slots(Tensor x, Tensor residual, Tensor(a!) "
      "shift_state, Tensor weight, Tensor bias, Tensor x_k, Tensor "
      "slot_indices, float eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor[]");
  m.def(
      "add_layer_norm_tmix_mix6_f16(Tensor x, Tensor residual, Tensor(a!) "
      "shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor "
      "x_k, Tensor x_v, Tensor x_a, Tensor x_g, float "
      "eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor[]");
  m.def(
      "add_layer_norm_tmix_mix6_f16_slots(Tensor x, Tensor residual, "
      "Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor "
      "x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, Tensor "
      "slot_indices, float eps=" RWKV7_LAYER_NORM_EPS_SCHEMA ") -> Tensor[]");
  m.def("advance_i32(Tensor(a!) x, int amount) -> ()");
  m.def(
      "advance_i32_slots(Tensor(a!) x, Tensor slot_indices, int amount) -> ()");
  m.def(
      "advance_i32_varlen(Tensor(a!) x, Tensor query_start_loc, Tensor "
      "slot_indices) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_v3a_ops, CUDA, m) {
  m.impl("layer_norm_f16", &layer_norm_f16);
  m.impl("emb_ln0_bf16_to_f16", &emb_ln0_bf16_to_f16);
  m.impl("linear_f16", &linear_f16);
  m.impl("linear_f16_m1_splitk", &linear_f16_m1_splitk);
  m.impl("linear_t_f16", &linear_t_f16);
  m.impl("linear_t_act_f16", &linear_t_act_f16);
  m.impl("linear_t_vres_f16", &linear_t_vres_f16);
  m.impl("linear_wag_rank_in_f16", &linear_wag_rank_in_f16);
  m.impl("linear_wagv_rank_in_f16", &linear_wagv_rank_in_f16);
  m.impl("linear_wag_rank_out_f16", &linear_wag_rank_out_f16);
  m.impl("linear_wagv_rank_out_f16", &linear_wagv_rank_out_f16);
  m.impl("add_f16", &add_f16);
  m.impl("add_layer_norm_f16", &add_layer_norm_f16);
  m.impl("add_last_layer_norm_f16", &add_last_layer_norm_f16);
  m.impl("add_layer_norm_cmix_mix_f16", &add_layer_norm_cmix_mix_f16);
  m.impl("add_layer_norm_cmix_mix_f16_slots",
         &add_layer_norm_cmix_mix_f16_slots);
  m.impl("add_layer_norm_tmix_mix6_f16", &add_layer_norm_tmix_mix6_f16);
  m.impl("add_layer_norm_tmix_mix6_f16_slots",
         &add_layer_norm_tmix_mix6_f16_slots);
  m.impl("advance_i32", &advance_i32);
  m.impl("advance_i32_slots", &advance_i32_slots);
  m.impl("advance_i32_varlen", &advance_i32_varlen);
}
