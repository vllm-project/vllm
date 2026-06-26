// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <torch/all.h>
#include <torch/library.h>

void wkv_fp32_v2_cuda(int B, int T, int C, int H, int mode, torch::Tensor state,
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

}  // namespace

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor state,
             torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
             torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), 0, state, r, w, k,
                   v, a, b, y);
}

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H,
                 torch::Tensor state, torch::Tensor r, torch::Tensor w,
                 torch::Tensor k, torch::Tensor v, torch::Tensor a,
                 torch::Tensor b, torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), 1, state, r, w, k,
                   v, a, b, y);
}

void forward_small(int64_t B, int64_t T, int64_t C, int64_t H,
                   torch::Tensor state, torch::Tensor r, torch::Tensor w,
                   torch::Tensor k, torch::Tensor v, torch::Tensor a,
                   torch::Tensor b, torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), 2, state, r, w, k,
                   v, a, b, y);
}

void forward_block(int64_t B, int64_t T, int64_t C, int64_t H,
                   torch::Tensor state, torch::Tensor r, torch::Tensor w,
                   torch::Tensor k, torch::Tensor v, torch::Tensor a,
                   torch::Tensor b, torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(static_cast<int>(B), static_cast<int>(T),
                   static_cast<int>(C), static_cast<int>(H), 3, state, r, w, k,
                   v, a, b, y);
}

TORCH_LIBRARY(rwkv7_wkv_fp32_v2, m) {
  m.def(
      "forward(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor "
      "w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def(
      "forward_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def(
      "forward_small(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def(
      "forward_block(int B, int T, int C, int H, Tensor(a!) state, Tensor r, "
      "Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp32_v2, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("forward_seq", &forward_seq);
  m.impl("forward_small", &forward_small);
  m.impl("forward_block", &forward_block);
}
