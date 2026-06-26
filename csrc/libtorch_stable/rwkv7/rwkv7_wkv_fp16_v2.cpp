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

void wkv_seq_w0_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                        torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                        torch::Tensor k, torch::Tensor v, torch::Tensor a,
                        torch::Tensor b, torch::Tensor y,
                        torch::Tensor elapsed_t);

void wkv_one_v2_cuda(int B, int C, int H, torch::Tensor state, torch::Tensor r,
                     torch::Tensor w, torch::Tensor k, torch::Tensor v,
                     torch::Tensor a, torch::Tensor b, torch::Tensor y,
                     torch::Tensor elapsed_t);

void wkv_one_w0_v2_cuda(int B, int C, int H, torch::Tensor state,
                        torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                        torch::Tensor k, torch::Tensor v, torch::Tensor a,
                        torch::Tensor b, torch::Tensor y,
                        torch::Tensor elapsed_t);

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

void check_w0(const torch::Tensor& w0, int64_t C) {
  check_half_cuda_contig(w0, "w0");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
}

void check_seq_inputs(int64_t B, int64_t T, int64_t C, int64_t H,
                      torch::Tensor state, torch::Tensor r, torch::Tensor w,
                      torch::Tensor k, torch::Tensor v, torch::Tensor a,
                      torch::Tensor b, torch::Tensor y,
                      torch::Tensor elapsed_t) {
  check_positive_int_arg(T, "T");
  check_common_inputs(B, C, H, state, elapsed_t);
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

void check_one_inputs(int64_t B, int64_t C, int64_t H, torch::Tensor state,
                      torch::Tensor r, torch::Tensor w, torch::Tensor k,
                      torch::Tensor v, torch::Tensor a, torch::Tensor b,
                      torch::Tensor y, torch::Tensor elapsed_t) {
  check_common_inputs(B, C, H, state, elapsed_t);
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
  TORCH_CHECK(r.dim() == 2 && r.size(0) == B && r.size(1) == C,
              "r must have shape [B,C]");
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

void wkv_one(int64_t B, int64_t C, int64_t H, torch::Tensor state,
             torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
             torch::Tensor a, torch::Tensor b, torch::Tensor y,
             torch::Tensor elapsed_t) {
  check_one_inputs(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
  wkv_one_v2_cuda(static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
                  state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_one_w0(int64_t B, int64_t C, int64_t H, torch::Tensor state,
                torch::Tensor r, torch::Tensor w, torch::Tensor w0,
                torch::Tensor k, torch::Tensor v, torch::Tensor a,
                torch::Tensor b, torch::Tensor y, torch::Tensor elapsed_t) {
  check_one_inputs(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
  check_w0(w0, C);
  wkv_one_w0_v2_cuda(static_cast<int>(B), static_cast<int>(C),
                     static_cast<int>(H), state, r, w, w0, k, v, a, b, y,
                     elapsed_t);
}

TORCH_LIBRARY(rwkv7_wkv_fp16_v2, m) {
  m.def(
      "wkv_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor "
      "w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor "
      "elapsed_t) -> ()");
  m.def(
      "wkv_seq_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.def(
      "wkv_one(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, "
      "Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor "
      "elapsed_t) -> ()");
  m.def(
      "wkv_one_w0(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, "
      "Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, "
      "Tensor elapsed_t) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp16_v2, CUDA, m) {
  m.impl("wkv_seq", &wkv_seq);
  m.impl("wkv_seq_w0", &wkv_seq_w0);
  m.impl("wkv_one", &wkv_one);
  m.impl("wkv_one_w0", &wkv_one_w0);
}
