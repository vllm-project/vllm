// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <torch/all.h>
#include <torch/library.h>

void wkv_fp32_v2_cuda(int B, int T, int C, int H, torch::Tensor state,
                      torch::Tensor r, torch::Tensor w, torch::Tensor k,
                      torch::Tensor v, torch::Tensor a, torch::Tensor b,
                      torch::Tensor y, torch::Tensor slot_indices);

void wkv_fp32_v2_cuda_varlen(int B, int max_t, int C, int H,
                             torch::Tensor query_start_loc,
                             torch::Tensor slot_indices, torch::Tensor state,
                             torch::Tensor r, torch::Tensor w, torch::Tensor k,
                             torch::Tensor v, torch::Tensor a, torch::Tensor b,
                             torch::Tensor y);

namespace {

#ifdef _IO_FP16_
constexpr auto IO_DTYPE = torch::kFloat16;
constexpr const char* IO_DTYPE_NAME = "fp16";
#else
constexpr auto IO_DTYPE = torch::kFloat32;
constexpr const char* IO_DTYPE_NAME = "fp32";
#endif

void check_float_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be fp32");
}

void check_io_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == IO_DTYPE, name, " must be ", IO_DTYPE_NAME);
}

void check_i32_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
}

void check_inputs(int64_t B, int64_t T, int64_t C, int64_t H,
                  torch::Tensor state, torch::Tensor r, torch::Tensor w,
                  torch::Tensor k, torch::Tensor v, torch::Tensor a,
                  torch::Tensor b, torch::Tensor y) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a, "a");
  check_io_cuda_contig(b, "b");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H &&
                  state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
      "r must have shape [B,T,C]");
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

void check_slot_inputs(int64_t B, int64_t T, int64_t C, int64_t H,
                       torch::Tensor state, torch::Tensor r, torch::Tensor w,
                       torch::Tensor k, torch::Tensor v, torch::Tensor a,
                       torch::Tensor b, torch::Tensor y,
                       torch::Tensor slot_indices) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_float_cuda_contig(state, "state");
  check_slot_indices(slot_indices, B);
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a, "a");
  check_io_cuda_contig(b, "b");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) == H &&
                  state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [slots,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
      "r must have shape [B,T,C]");
}

void check_varlen_inputs(int64_t B, int64_t total_tokens, int64_t max_t,
                         int64_t C, int64_t H, torch::Tensor query_start_loc,
                         torch::Tensor slot_indices, torch::Tensor state,
                         torch::Tensor r, torch::Tensor w, torch::Tensor k,
                         torch::Tensor v, torch::Tensor a, torch::Tensor b,
                         torch::Tensor y) {
  TORCH_CHECK(B > 0 && max_t > 0 && total_tokens > 0,
              "B, max_t, and total_tokens must be positive");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_float_cuda_contig(state, "state");
  check_query_start_loc(query_start_loc, B);
  check_slot_indices(slot_indices, B);
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a, "a");
  check_io_cuda_contig(b, "b");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) == H &&
                  state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [slots,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.dim() == 2 && r.size(0) == total_tokens && r.size(1) == C,
              "r must have shape [total_tokens,C]");
}

}  // namespace

void forward_impl(int64_t B, int64_t T, int64_t C, int64_t H,
                  torch::Tensor state, torch::Tensor r, torch::Tensor w,
                  torch::Tensor k, torch::Tensor v, torch::Tensor a,
                  torch::Tensor b, torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  torch::Tensor slot_indices;
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), state, r, w, k, v,
                   a, b, y, slot_indices);
}

void forward_slot_impl(int64_t B, int64_t T, int64_t C, int64_t H,
                       torch::Tensor state, torch::Tensor r, torch::Tensor w,
                       torch::Tensor k, torch::Tensor v, torch::Tensor a,
                       torch::Tensor b, torch::Tensor y,
                       torch::Tensor slot_indices) {
  check_slot_inputs(B, T, C, H, state, r, w, k, v, a, b, y, slot_indices);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), state, r, w, k, v,
                   a, b, y, slot_indices);
}

void forward_varlen_impl(int64_t B, int64_t total_tokens, int64_t max_t,
                         int64_t C, int64_t H,
                         torch::Tensor query_start_loc,
                         torch::Tensor slot_indices, torch::Tensor state,
                         torch::Tensor r, torch::Tensor w, torch::Tensor k,
                         torch::Tensor v, torch::Tensor a, torch::Tensor b,
                         torch::Tensor y) {
  check_varlen_inputs(B, total_tokens, max_t, C, H, query_start_loc,
                      slot_indices, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t),
                          static_cast<int>(C), static_cast<int>(H),
                          query_start_loc, slot_indices, state, r, w, k, v, a, b,
                          y);
}

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor state,
             torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
             torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  forward_impl(B, T, C, H, state, r, w, k, v, a, b, y);
}

void forward_slot(int64_t B, int64_t T, int64_t C, int64_t H,
                  torch::Tensor state, torch::Tensor r, torch::Tensor w,
                  torch::Tensor k, torch::Tensor v, torch::Tensor a,
                  torch::Tensor b, torch::Tensor y,
                  torch::Tensor slot_indices) {
  forward_slot_impl(B, T, C, H, state, r, w, k, v, a, b, y, slot_indices);
}

void forward_varlen(int64_t B, int64_t total_tokens, int64_t max_t, int64_t C,
                    int64_t H, torch::Tensor query_start_loc,
                    torch::Tensor slot_indices, torch::Tensor state,
                    torch::Tensor r, torch::Tensor w, torch::Tensor k,
                    torch::Tensor v, torch::Tensor a, torch::Tensor b,
                    torch::Tensor y) {
  forward_varlen_impl(B, total_tokens, max_t, C, H, query_start_loc,
                      slot_indices, state, r, w, k, v, a, b, y);
}

TORCH_LIBRARY(rwkv7_wkv_fp32_v2, m) {
  m.def(
      "forward(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor "
      "w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def(
      "forward_slot(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, "
      "Tensor slot_indices) -> ()");
  m.def(
      "forward_varlen(int B, int total_tokens, int max_t, int C, int H, "
      "Tensor query_start_loc, Tensor slot_indices, Tensor(a!) state, Tensor "
      "r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> "
      "()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp32_v2, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("forward_slot", &forward_slot);
  m.impl("forward_varlen", &forward_varlen);
}
