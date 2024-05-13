/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#include "gptq_marlin.cuh"

template <typename T> inline std::string str(T x) { return std::to_string(x); }

namespace gptq_marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__global__ void permute_cols_kernel(int4 const *__restrict__ a_int4_ptr,
                                    int const *__restrict__ perm_int_ptr,
                                    int4 *__restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {}

template <const int num_bits,        // number of bits used for weights
          const int threads,         // number of threads in a threadblock
          const int thread_m_blocks, // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the threadblock
          const int thread_n_blocks, // same for n dimension (output)
          const int thread_k_blocks, // same for k dimension (reduction)
          const int stages, // number of stages for the async global->shared
                            // fetch pipeline
          const bool has_act_order,   // whether act_order is enabled
          const int group_blocks = -1 // number of consecutive 16x16 blocks with
                                      // a separate quantization scale
          >
__global__ void
Marlin(const int4 *__restrict__ A, // fp16 input matrix of shape mxk
       const int4 *__restrict__ B, // 4bit quantized weight matrix of shape kxn
       int4 *__restrict__ C,       // fp16 output buffer of shape mxn
       const int4 *__restrict__ scales_ptr, // fp16 quantization scales of shape
                                            // (k/groupsize)xn
       const int *__restrict__ g_idx,       // int32 group indices of shape k
       int num_groups, // number of scale groups per output channel
       int prob_m,     // batch dimension m
       int prob_n,     // output dimension n
       int prob_k,     // reduction dimension k
       int *locks      // extra global storage for barrier synchronization
) {}

} // namespace gptq_marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                               torch::Tensor &b_scales, torch::Tensor &g_idx,
                               torch::Tensor &perm, torch::Tensor &workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else


  namespace fp16 {
#include "gptq_marlin_part.cu"
  } // namespace fp16


  namespace bf16 {
#define INCLUDE_GPT_MARGIN_BFLOAT16 1
#include "gptq_marlin_part.cu"
#undef INCLUDE_GPT_MARGIN_BFLOAT16
  } // namespace bf16

} // namespace gptq_marlin


torch::Tensor gptq_marlin_gemm(torch::Tensor &a, torch::Tensor &b_q_weight,
                               torch::Tensor &b_scales, torch::Tensor &g_idx,
                               torch::Tensor &perm, torch::Tensor &workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full) {
  // Verify num_bits
  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int pack_factor = 32 / num_bits;

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % gptq_marlin::tile_size == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK((size_k / gptq_marlin::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % gptq_marlin::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / gptq_marlin::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
  TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);
  torch::Tensor a_tmp = torch::empty({size_m, size_k}, options);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Verify g_idx and perm
  TORCH_CHECK((g_idx.size(0) == 0 && perm.size(0) == 0) ||
                  (g_idx.size(0) == size_k && perm.size(0) == size_k),
              "Unexpected g_idx.size(0) = ", g_idx.size(0),
              " and perm.size(0) = ", perm.size(0),
              ", where size_k = ", size_k);

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(0) != 0;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 2, "b_scales rank = ", b_rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  // Verify workspace size
  TORCH_CHECK(
      size_n % gptq_marlin::min_thread_n == 0, "size_n = ", size_n,
      ", is not divisible by min_thread_n = ", gptq_marlin::min_thread_n);
  int min_workspace_size =
      (size_n / gptq_marlin::min_thread_n) * gptq_marlin::max_par;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    gptq_marlin::fp16::marlin_mm_f16i4(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(), b_scales.data_ptr<at::Half>(),
        g_idx.data_ptr(), perm.data_ptr(), a_tmp.data_ptr<at::Half>(), size_m, size_n,
        size_k, workspace.data_ptr(), num_bits, has_act_order, is_k_full,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, gptq_marlin::max_par);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    gptq_marlin::bf16::marlin_mm_f16i4(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(), c.data_ptr<at::BFloat16>(), b_scales.data_ptr<at::BFloat16>(),
        g_idx.data_ptr(), perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(), size_m, size_n,
        size_k, workspace.data_ptr(), num_bits, has_act_order, is_k_full,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, gptq_marlin::max_par);
  } else {
    throw std::runtime_error("gpt_marlin_gemm only supports bfloat16 and float16");
  }

  return c;
}

#endif
