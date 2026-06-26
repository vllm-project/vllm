// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_cuda(int B, int T, int C, torch::Tensor x,
                                          torch::Tensor shift_state,
                                          torch::Tensor x_r, torch::Tensor x_w,
                                          torch::Tensor x_k, torch::Tensor x_v,
                                          torch::Tensor x_a, torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_cfg_cuda(
    int B, int T, int C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g, int threads);
std::vector<torch::Tensor> tmix_mix6_t1_c4096_cuda(
    int B, torch::Tensor x, torch::Tensor shift_state, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, int threads, int vec, bool half_math);

std::vector<torch::Tensor> tmix_kk_a_gate_cuda(
    int B, int T, int C, int H, torch::Tensor k, torch::Tensor k_k,
    torch::Tensor a0, torch::Tensor a12, torch::Tensor k_a, torch::Tensor x,
    torch::Tensor shift_state, bool update_shift);

torch::Tensor tmix_lnx_rkvres_xg_cuda(int B, int T, int C, int H,
                                      torch::Tensor x, torch::Tensor r,
                                      torch::Tensor k, torch::Tensor v,
                                      torch::Tensor r_k, torch::Tensor weight,
                                      torch::Tensor bias, torch::Tensor g);

torch::Tensor tmix_vres_gate_cuda(int B, int T, int C, torch::Tensor v,
                                  torch::Tensor v_first, torch::Tensor v0,
                                  torch::Tensor v12);

torch::Tensor cmix_sparse_one_cuda(int C, int F, torch::Tensor x,
                                   torch::Tensor shift_state, torch::Tensor x_k,
                                   torch::Tensor key_fc,
                                   torch::Tensor value_fc);

torch::Tensor cmix_sparse_rows_cuda(int B, int T, int C, int F, torch::Tensor x,
                                    torch::Tensor shift_state,
                                    torch::Tensor x_k, torch::Tensor key_fc,
                                    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_one_cuda(int C, int F, torch::Tensor act,
                                        torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_rows_cuda(int B, int T, int C, int F,
                                         torch::Tensor act,
                                         torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_one_cuda(int C, int F, torch::Tensor preact,
                                             torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_cuda(int B, int T, int C, int F,
                                              torch::Tensor preact,
                                              torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_t512_cuda(int B, int T, int C, int F,
                                                   torch::Tensor preact,
                                                   torch::Tensor value_fc);

torch::Tensor cmix_mix_cuda(int B, int T, int C, torch::Tensor x,
                            torch::Tensor shift_state, torch::Tensor x_k);
torch::Tensor cmix_mix_cfg_cuda(int B, int T, int C, torch::Tensor x,
                                torch::Tensor shift_state, torch::Tensor x_k,
                                int threads);

torch::Tensor relu_square_cuda(torch::Tensor x);

torch::Tensor act_tanh_cuda(torch::Tensor x);

torch::Tensor act_sigmoid_cuda(torch::Tensor x);

torch::Tensor add_vec_cuda(int C, torch::Tensor x, torch::Tensor vec);

namespace {

int checked_int_arg(int64_t x, const char* name) {
  TORCH_CHECK(x > 0 && x <= std::numeric_limits<int>::max(), name,
              " must be positive and fit in int");
  return static_cast<int>(x);
}

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_3d(const torch::Tensor& x, int64_t B, int64_t T, int64_t C,
              const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 3, name, " must have shape [B,T,C]");
  TORCH_CHECK(x.size(0) == B && x.size(1) == T && x.size(2) == C, name,
              " shape mismatch");
}

void check_vec(const torch::Tensor& x, int64_t C, const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 1 && x.size(0) == C, name, " must have shape [C]");
}

}  // namespace

std::vector<torch::Tensor> tmix_mix6(int64_t B, int64_t T, int64_t C,
                                     torch::Tensor x, torch::Tensor shift_state,
                                     torch::Tensor x_r, torch::Tensor x_w,
                                     torch::Tensor x_k, torch::Tensor x_v,
                                     torch::Tensor x_a, torch::Tensor x_g) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                        checked_int_arg(C, "C"), x, shift_state, x_r, x_w, x_k,
                        x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_cfg(
    int64_t B, int64_t T, int64_t C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g, int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(
      threads == 128 || threads == 256 || threads == 512 || threads == 1024,
      "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_cfg_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                            checked_int_arg(C, "C"), x, shift_state, x_r, x_w,
                            x_k, x_v, x_a, x_g,
                            checked_int_arg(threads, "threads"));
}

std::vector<torch::Tensor> tmix_mix6_t1_c4096(
    int64_t B, torch::Tensor x, torch::Tensor shift_state, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, int64_t threads, int64_t vec, bool half_math) {
  TORCH_CHECK(
      threads == 128 || threads == 256 || threads == 512 || threads == 1024,
      "unsupported threads");
  TORCH_CHECK(vec == 1 || vec == 2 || vec == 4 || vec == 8, "unsupported vec");
  check_3d(x, B, 1, 4096, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == 4096,
              "shift_state must have shape [B,4096]");
  check_vec(x_r, 4096, "x_r");
  check_vec(x_w, 4096, "x_w");
  check_vec(x_k, 4096, "x_k");
  check_vec(x_v, 4096, "x_v");
  check_vec(x_a, 4096, "x_a");
  check_vec(x_g, 4096, "x_g");
  return tmix_mix6_t1_c4096_cuda(checked_int_arg(B, "B"), x, shift_state, x_r,
                                 x_w, x_k, x_v, x_a, x_g,
                                 checked_int_arg(threads, "threads"),
                                 checked_int_arg(vec, "vec"), half_math);
}

std::vector<torch::Tensor> tmix_kk_a_gate(int64_t B, int64_t T, int64_t C,
                                          int64_t H, torch::Tensor k,
                                          torch::Tensor k_k, torch::Tensor a0,
                                          torch::Tensor a12,
                                          torch::Tensor k_a) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  return tmix_kk_a_gate_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), checked_int_arg(H, "H"),
                             k, k_k, a0, a12, k_a, k, k, false);
}

std::vector<torch::Tensor> tmix_kk_a_gate_update_shift(
    int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor k,
    torch::Tensor k_k, torch::Tensor a0, torch::Tensor a12, torch::Tensor k_a,
    torch::Tensor x, torch::Tensor shift_state) {
  TORCH_CHECK(T == 1, "tmix_kk_a_gate_update_shift currently requires T=1");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  return tmix_kk_a_gate_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), checked_int_arg(H, "H"),
                             k, k_k, a0, a12, k_a, x, shift_state, true);
}

torch::Tensor tmix_lnx_rkvres_xg(int64_t B, int64_t T, int64_t C, int64_t H,
                                 torch::Tensor x, torch::Tensor r,
                                 torch::Tensor k, torch::Tensor v,
                                 torch::Tensor r_k, torch::Tensor weight,
                                 torch::Tensor bias, torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  return tmix_lnx_rkvres_xg_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(H, "H"), x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_vres_gate(int64_t B, int64_t T, int64_t C, torch::Tensor v,
                             torch::Tensor v_first, torch::Tensor v0,
                             torch::Tensor v12) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  return tmix_vres_gate_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), v, v_first, v0, v12);
}

torch::Tensor cmix_sparse_one(int64_t C, int64_t F, torch::Tensor x,
                              torch::Tensor shift_state, torch::Tensor x_k,
                              torch::Tensor key_fc, torch::Tensor value_fc) {
  check_3d(x, 1, 1, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == 1 &&
                  shift_state.size(1) == C,
              "shift_state must have shape [1,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_one_cuda(checked_int_arg(C, "C"), checked_int_arg(F, "F"),
                              x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_rows(int64_t B, int64_t T, int64_t C, int64_t F,
                               torch::Tensor x, torch::Tensor shift_state,
                               torch::Tensor x_k, torch::Tensor key_fc,
                               torch::Tensor value_fc) {
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_rows_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                               checked_int_arg(C, "C"), checked_int_arg(F, "F"),
                               x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_down_one(int64_t C, int64_t F, torch::Tensor act,
                                   torch::Tensor value_fc) {
  check_half_cuda_contig(act, "act");
  TORCH_CHECK(act.dim() == 1 && act.size(0) == F, "act must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_one_cuda(checked_int_arg(C, "C"),
                                   checked_int_arg(F, "F"), act, value_fc);
}

torch::Tensor cmix_sparse_down_rows(int64_t B, int64_t T, int64_t C, int64_t F,
                                    torch::Tensor act, torch::Tensor value_fc) {
  check_3d(act, B, T, F, "act");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_rows_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(F, "F"), act, value_fc);
}

torch::Tensor cmix_sparse_down_relu_one(int64_t C, int64_t F,
                                        torch::Tensor preact,
                                        torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F,
              "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_one_cuda(
      checked_int_arg(C, "C"), checked_int_arg(F, "F"), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows(int64_t B, int64_t T, int64_t C,
                                         int64_t F, torch::Tensor preact,
                                         torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(F, "F"), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512(int64_t B, int64_t T, int64_t C,
                                              int64_t F, torch::Tensor preact,
                                              torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  return cmix_sparse_down_relu_rows_t512_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(F, "F"), preact, value_fc);
}

torch::Tensor cmix_mix(int64_t B, int64_t T, int64_t C, torch::Tensor x,
                       torch::Tensor shift_state, torch::Tensor x_k) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  return cmix_mix_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                       checked_int_arg(C, "C"), x, shift_state, x_k);
}

torch::Tensor cmix_mix_cfg(int64_t B, int64_t T, int64_t C, torch::Tensor x,
                           torch::Tensor shift_state, torch::Tensor x_k,
                           int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(
      threads == 128 || threads == 256 || threads == 512 || threads == 1024,
      "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  return cmix_mix_cfg_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                           checked_int_arg(C, "C"), x, shift_state, x_k,
                           checked_int_arg(threads, "threads"));
}

torch::Tensor relu_square(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return relu_square_cuda(x);
}

torch::Tensor act_tanh(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_tanh_cuda(x);
}

torch::Tensor act_sigmoid(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_sigmoid_cuda(x);
}

torch::Tensor add_vec(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_half_cuda_contig(x, "x");
  check_vec(vec, C, "vec");
  TORCH_CHECK(x.numel() > 0 && (x.numel() % 2) == 0,
              "x.numel() must be positive and even");
  TORCH_CHECK(x.size(-1) == C, "x last dim must equal C");
  return add_vec_cuda(checked_int_arg(C, "C"), x, vec);
}

TORCH_LIBRARY(rwkv7_fast_ops_fp16, m) {
  m.def(
      "tmix_mix6(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) "
      "-> Tensor[]");
  m.def(
      "tmix_mix6_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "int threads) -> Tensor[]");
  m.def(
      "tmix_mix6_t1_c4096(int B, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "int threads, int vec, bool half_math=False) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor "
      "a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_update_shift(int B, int T, int C, int H, Tensor k, "
      "Tensor k_k, Tensor a0, Tensor a12, Tensor k_a, Tensor x, Tensor(a!) "
      "shift_state) -> Tensor[]");
  m.def(
      "tmix_lnx_rkvres_xg(int B, int T, int C, int H, Tensor x, Tensor r, "
      "Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_vres_gate(int B, int T, int C, Tensor v, Tensor v_first, Tensor "
      "v0, Tensor v12) -> Tensor");
  m.def(
      "cmix_sparse_one(int C, int F, Tensor x, Tensor(a!) shift_state, Tensor "
      "x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_rows(int B, int T, int C, int F, Tensor x, Tensor(a!) "
      "shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_down_one(int C, int F, Tensor act, Tensor value_fc) -> "
      "Tensor");
  m.def(
      "cmix_sparse_down_rows(int B, int T, int C, int F, Tensor act, Tensor "
      "value_fc) -> Tensor");
  m.def(
      "cmix_sparse_down_relu_one(int C, int F, Tensor preact, Tensor value_fc) "
      "-> Tensor");
  m.def(
      "cmix_sparse_down_relu_rows(int B, int T, int C, int F, Tensor preact, "
      "Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_down_relu_rows_t512(int B, int T, int C, int F, Tensor "
      "preact, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_mix(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor "
      "x_k) -> Tensor");
  m.def(
      "cmix_mix_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_k, int threads) -> Tensor");
  m.def("relu_square(Tensor x) -> Tensor");
  m.def("act_tanh(Tensor x) -> Tensor");
  m.def("act_sigmoid(Tensor x) -> Tensor");
  m.def("add_vec(int C, Tensor x, Tensor vec) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("tmix_mix6", &tmix_mix6);
  m.impl("tmix_mix6_cfg", &tmix_mix6_cfg);
  m.impl("tmix_mix6_t1_c4096", &tmix_mix6_t1_c4096);
  m.impl("tmix_kk_a_gate", &tmix_kk_a_gate);
  m.impl("tmix_kk_a_gate_update_shift", &tmix_kk_a_gate_update_shift);
  m.impl("tmix_lnx_rkvres_xg", &tmix_lnx_rkvres_xg);
  m.impl("tmix_vres_gate", &tmix_vres_gate);
  m.impl("cmix_sparse_one", &cmix_sparse_one);
  m.impl("cmix_sparse_rows", &cmix_sparse_rows);
  m.impl("cmix_sparse_down_one", &cmix_sparse_down_one);
  m.impl("cmix_sparse_down_rows", &cmix_sparse_down_rows);
  m.impl("cmix_sparse_down_relu_one", &cmix_sparse_down_relu_one);
  m.impl("cmix_sparse_down_relu_rows", &cmix_sparse_down_relu_rows);
  m.impl("cmix_sparse_down_relu_rows_t512", &cmix_sparse_down_relu_rows_t512);
  m.impl("cmix_mix", &cmix_mix);
  m.impl("cmix_mix_cfg", &cmix_mix_cfg);
  m.impl("relu_square", &relu_square);
  m.impl("act_tanh", &act_tanh);
  m.impl("act_sigmoid", &act_sigmoid);
  m.impl("add_vec", &add_vec);
}
