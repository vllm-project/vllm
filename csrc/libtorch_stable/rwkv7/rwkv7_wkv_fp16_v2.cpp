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

void wkv_seq_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                     torch::Tensor r, torch::Tensor w, torch::Tensor k,
                     torch::Tensor v, torch::Tensor a, torch::Tensor b,
                     torch::Tensor y, torch::Tensor elapsed_t);

void wkv_seq_slot_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                          torch::Tensor r, torch::Tensor w, torch::Tensor k,
                          torch::Tensor v, torch::Tensor a, torch::Tensor b,
                          torch::Tensor y, torch::Tensor slot_indices,
                          torch::Tensor elapsed_t);

void wkv_seq_w0_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                        torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                        torch::Tensor k, torch::Tensor v, torch::Tensor a,
                        torch::Tensor b, torch::Tensor y,
                        torch::Tensor elapsed_t);

void wkv_seq_w0_slot_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                             torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                             torch::Tensor k, torch::Tensor v, torch::Tensor a,
                             torch::Tensor b, torch::Tensor y,
                             torch::Tensor slot_indices,
                             torch::Tensor elapsed_t);

void wkv_seq_varlen_v2_cuda(int B, int max_t, int C, int H,
                            torch::Tensor query_start_loc,
                            torch::Tensor slot_indices, torch::Tensor state,
                            torch::Tensor r, torch::Tensor w, torch::Tensor k,
                            torch::Tensor v, torch::Tensor a, torch::Tensor b,
                            torch::Tensor y, torch::Tensor elapsed_t);

void wkv_seq_w0_varlen_v2_cuda(
    int B, int max_t, int C, int H, torch::Tensor query_start_loc,
    torch::Tensor slot_indices, torch::Tensor state, torch::Tensor r,
    torch::Tensor w, torch::Tensor w0, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor y, torch::Tensor elapsed_t);

namespace {

constexpr int64_t HEAD_SIZE = 64;

void check_positive_int_arg(int64_t x, const char* name) {
  TORCH_CHECK(x > 0 && x <= std::numeric_limits<int>::max(), name,
              " must be positive and fit in int");
}

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

void check_common_inputs(int64_t B, int64_t C, int64_t H, torch::Tensor state,
                         torch::Tensor elapsed_t) {
  check_positive_int_arg(B, "B");
  check_positive_int_arg(C, "C");
  check_positive_int_arg(H, "H");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_half_cuda_contig(state, "state");
  check_i32_cuda_contig(elapsed_t, "elapsed_t");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H &&
                  state.size(2) == HEAD_SIZE && state.size(3) == HEAD_SIZE,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
}

void check_slot_indices(const torch::Tensor& x, int64_t B) {
  check_i32_cuda_contig(x, "slot_indices");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == B,
              "slot_indices must have shape [B]");
}

void check_query_start_loc(const torch::Tensor& x, int64_t B) {
  check_i32_cuda_contig(x, "query_start_loc");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
}

void check_slot_common_inputs(int64_t B, int64_t C, int64_t H,
                              torch::Tensor state, torch::Tensor slot_indices,
                              torch::Tensor elapsed_t) {
  check_positive_int_arg(B, "B");
  check_positive_int_arg(C, "C");
  check_positive_int_arg(H, "H");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_half_cuda_contig(state, "state");
  check_i32_cuda_contig(elapsed_t, "elapsed_t");
  check_slot_indices(slot_indices, B);
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) == H &&
                  state.size(2) == HEAD_SIZE && state.size(3) == HEAD_SIZE,
              "state must have shape [slots,H,64,64]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == state.size(0),
              "elapsed_t must have shape [slots]");
}

void check_w0(const torch::Tensor& w0, int64_t C) {
  check_half_cuda_contig(w0, "w0");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
}

void check_seq_payload(int64_t B, int64_t T, int64_t C, torch::Tensor r,
                       torch::Tensor w, torch::Tensor k, torch::Tensor v,
                       torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  check_half_cuda_contig(r, "r");
  check_half_cuda_contig(w, "w");
  check_half_cuda_contig(k, "k");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(a, "a");
  check_half_cuda_contig(b, "b");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
      "r must have shape [B,T,C]");
}

void check_seq_inputs(int64_t B, int64_t T, int64_t C, int64_t H,
                      torch::Tensor state, torch::Tensor r, torch::Tensor w,
                      torch::Tensor k, torch::Tensor v, torch::Tensor a,
                      torch::Tensor b, torch::Tensor y,
                      torch::Tensor elapsed_t) {
  check_positive_int_arg(T, "T");
  check_common_inputs(B, C, H, state, elapsed_t);
  check_seq_payload(B, T, C, r, w, k, v, a, b, y);
}

void check_seq_slot_inputs(int64_t B, int64_t T, int64_t C, int64_t H,
                           torch::Tensor state, torch::Tensor r,
                           torch::Tensor w, torch::Tensor k, torch::Tensor v,
                           torch::Tensor a, torch::Tensor b, torch::Tensor y,
                           torch::Tensor slot_indices,
                           torch::Tensor elapsed_t) {
  check_positive_int_arg(T, "T");
  check_slot_common_inputs(B, C, H, state, slot_indices, elapsed_t);
  check_seq_payload(B, T, C, r, w, k, v, a, b, y);
}

void check_varlen_payload(int64_t total_tokens, int64_t C, torch::Tensor r,
                          torch::Tensor w, torch::Tensor k, torch::Tensor v,
                          torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  check_half_cuda_contig(r, "r");
  check_half_cuda_contig(w, "w");
  check_half_cuda_contig(k, "k");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(a, "a");
  check_half_cuda_contig(b, "b");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.dim() == 2 && r.size(0) == total_tokens && r.size(1) == C,
              "r must have shape [total_tokens,C]");
}

void check_seq_varlen_inputs(int64_t B, int64_t total_tokens, int64_t max_t,
                             int64_t C, int64_t H,
                             torch::Tensor query_start_loc,
                             torch::Tensor slot_indices, torch::Tensor state,
                             torch::Tensor r, torch::Tensor w, torch::Tensor k,
                             torch::Tensor v, torch::Tensor a, torch::Tensor b,
                             torch::Tensor y, torch::Tensor elapsed_t) {
  check_positive_int_arg(max_t, "max_t");
  check_positive_int_arg(total_tokens, "total_tokens");
  check_slot_common_inputs(B, C, H, state, slot_indices, elapsed_t);
  check_query_start_loc(query_start_loc, B);
  check_varlen_payload(total_tokens, C, r, w, k, v, a, b, y);
}

}  // namespace

void wkv_seq(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor state,
             torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
             torch::Tensor a, torch::Tensor b, torch::Tensor y,
             torch::Tensor elapsed_t) {
  check_seq_inputs(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
  wkv_seq_v2_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
                  static_cast<int>(H), state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_seq_slot(int64_t B, int64_t T, int64_t C, int64_t H,
                  torch::Tensor state, torch::Tensor r, torch::Tensor w,
                  torch::Tensor k, torch::Tensor v, torch::Tensor a,
                  torch::Tensor b, torch::Tensor y, torch::Tensor slot_indices,
                  torch::Tensor elapsed_t) {
  check_seq_slot_inputs(B, T, C, H, state, r, w, k, v, a, b, y, slot_indices,
                        elapsed_t);
  wkv_seq_slot_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                       static_cast<int>(C), static_cast<int>(H), state, r, w, k,
                       v, a, b, y, slot_indices, elapsed_t);
}

void wkv_seq_w0(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor state,
                torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                torch::Tensor k, torch::Tensor v, torch::Tensor a,
                torch::Tensor b, torch::Tensor y, torch::Tensor elapsed_t) {
  check_seq_inputs(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
  check_w0(w0, C);
  wkv_seq_w0_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                     static_cast<int>(C), static_cast<int>(H), state, r, w, w0,
                     k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_slot(int64_t B, int64_t T, int64_t C, int64_t H,
                     torch::Tensor state, torch::Tensor r, torch::Tensor w,
                     torch::Tensor w0, torch::Tensor k, torch::Tensor v,
                     torch::Tensor a, torch::Tensor b, torch::Tensor y,
                     torch::Tensor slot_indices, torch::Tensor elapsed_t) {
  check_seq_slot_inputs(B, T, C, H, state, r, w, k, v, a, b, y, slot_indices,
                        elapsed_t);
  check_w0(w0, C);
  wkv_seq_w0_slot_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                          static_cast<int>(C), static_cast<int>(H), state, r, w,
                          w0, k, v, a, b, y, slot_indices, elapsed_t);
}

void wkv_seq_varlen(int64_t B, int64_t total_tokens, int64_t max_t, int64_t C,
                    int64_t H, torch::Tensor query_start_loc,
                    torch::Tensor slot_indices, torch::Tensor state,
                    torch::Tensor r, torch::Tensor w, torch::Tensor k,
                    torch::Tensor v, torch::Tensor a, torch::Tensor b,
                    torch::Tensor y, torch::Tensor elapsed_t) {
  check_seq_varlen_inputs(B, total_tokens, max_t, C, H, query_start_loc,
                          slot_indices, state, r, w, k, v, a, b, y, elapsed_t);
  wkv_seq_varlen_v2_cuda(static_cast<int>(B), static_cast<int>(max_t),
                         static_cast<int>(C), static_cast<int>(H),
                         query_start_loc, slot_indices, state, r, w, k, v, a, b,
                         y, elapsed_t);
}

void wkv_seq_w0_varlen(int64_t B, int64_t total_tokens, int64_t max_t,
                       int64_t C, int64_t H, torch::Tensor query_start_loc,
                       torch::Tensor slot_indices, torch::Tensor state,
                       torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                       torch::Tensor k, torch::Tensor v, torch::Tensor a,
                       torch::Tensor b, torch::Tensor y,
                       torch::Tensor elapsed_t) {
  check_seq_varlen_inputs(B, total_tokens, max_t, C, H, query_start_loc,
                          slot_indices, state, r, w, k, v, a, b, y, elapsed_t);
  check_w0(w0, C);
  wkv_seq_w0_varlen_v2_cuda(static_cast<int>(B), static_cast<int>(max_t),
                            static_cast<int>(C), static_cast<int>(H),
                            query_start_loc, slot_indices, state, r, w, w0, k,
                            v, a, b, y, elapsed_t);
}

TORCH_LIBRARY(rwkv7_wkv_fp16_v2, m) {
  m.def(
      "wkv_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor "
      "w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor "
      "elapsed_t) -> ()");
  m.def(
      "wkv_seq_slot(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, "
      "Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.def(
      "wkv_seq_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.def(
      "wkv_seq_w0_slot(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor(a!) y, Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.def(
      "wkv_seq_varlen(int B, int total_tokens, int max_t, int C, int H, "
      "Tensor query_start_loc, Tensor slot_indices, Tensor(a!) state, Tensor "
      "r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, "
      "Tensor elapsed_t) -> ()");
  m.def(
      "wkv_seq_w0_varlen(int B, int total_tokens, int max_t, int C, int H, "
      "Tensor query_start_loc, Tensor slot_indices, Tensor(a!) state, Tensor "
      "r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor(a!) y, Tensor elapsed_t) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp16_v2, CUDA, m) {
  m.impl("wkv_seq", &wkv_seq);
  m.impl("wkv_seq_slot", &wkv_seq_slot);
  m.impl("wkv_seq_w0", &wkv_seq_w0);
  m.impl("wkv_seq_w0_slot", &wkv_seq_w0_slot);
  m.impl("wkv_seq_varlen", &wkv_seq_varlen);
  m.impl("wkv_seq_w0_varlen", &wkv_seq_w0_varlen);
}
