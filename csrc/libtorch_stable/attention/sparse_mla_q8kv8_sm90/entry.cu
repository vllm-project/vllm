/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"
#include "kernel.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

using torch::headeronly::ScalarType;

void check_cuda_contiguous(torch::stable::Tensor const& tensor,
                           char const* name, int device_index) {
  STD_TORCH_CHECK(tensor.device().is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.get_device_index() == device_index, name,
                  " must be on the same CUDA device as q");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void sparse_mla_q8kv8_prefill_sm90(
    torch::stable::Tensor const& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor const& indices, torch::stable::Tensor const& q_scale,
    torch::stable::Tensor const& kv_scale,
    torch::stable::Tensor const& attn_sink,
    torch::stable::Tensor const& topk_length, torch::stable::Tensor& out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& lse,
    double sm_scale) {
  STD_TORCH_CHECK(q.device().is_cuda() && q.is_contiguous(),
                  "q must be a contiguous CUDA tensor");
  int const device_index = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  STD_TORCH_CHECK(
      get_device_prop()->major == 9 && get_device_prop()->minor == 0,
      "sparse_mla_q8kv8_prefill_sm90 requires SM90");

  check_cuda_contiguous(kv, "kv", device_index);
  check_cuda_contiguous(indices, "indices", device_index);
  check_cuda_contiguous(q_scale, "q_scale", device_index);
  check_cuda_contiguous(kv_scale, "kv_scale", device_index);
  check_cuda_contiguous(attn_sink, "attn_sink", device_index);
  check_cuda_contiguous(topk_length, "topk_length", device_index);
  check_cuda_contiguous(out, "out", device_index);
  check_cuda_contiguous(max_logits, "max_logits", device_index);
  check_cuda_contiguous(lse, "lse", device_index);

  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 512,
                  "q must have shape [s_q, h_q, 512]");
  int const s_q = q.size(0);
  int const h_q = q.size(1);
  STD_TORCH_CHECK(s_q > 0, "q must contain at least one query");
  STD_TORCH_CHECK(h_q == 64 || h_q == 128,
                  "q head count must be padded to 64 or 128");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::Float8_e4m3fn,
                  "q must be float8_e4m3fn");

  STD_TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 512,
                  "kv must have shape [s_kv, 1, 512]");
  STD_TORCH_CHECK(kv.size(0) > 0, "kv must contain at least one row");
  STD_TORCH_CHECK(kv.scalar_type() == ScalarType::Float8_e4m3fn,
                  "kv must be float8_e4m3fn");

  STD_TORCH_CHECK(
      indices.dim() == 3 && indices.size(0) == s_q && indices.size(1) == 1,
      "indices must have shape [s_q, 1, topk]");
  int const topk = indices.size(2);
  STD_TORCH_CHECK(topk > 0 && topk % 128 == 0,
                  "indices topk must be a positive multiple of 128");
  STD_TORCH_CHECK(indices.scalar_type() == ScalarType::Int,
                  "indices must be int32");

  STD_TORCH_CHECK(
      q_scale.scalar_type() == ScalarType::Float && q_scale.numel() == 1,
      "q_scale must be a float32 scalar tensor");
  STD_TORCH_CHECK(
      kv_scale.scalar_type() == ScalarType::Float && kv_scale.numel() == 1,
      "kv_scale must be a float32 scalar tensor");
  STD_TORCH_CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == h_q &&
                      attn_sink.scalar_type() == ScalarType::Float,
                  "attn_sink must be float32 with shape [h_q]");
  STD_TORCH_CHECK(topk_length.dim() == 1 && topk_length.size(0) == s_q &&
                      topk_length.scalar_type() == ScalarType::Int,
                  "topk_length must be int32 with shape [s_q]");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == s_q && out.size(1) == h_q &&
                      out.size(2) == 512 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "out must be bfloat16 with shape [s_q, h_q, 512]");
  STD_TORCH_CHECK(max_logits.dim() == 2 && max_logits.size(0) == s_q &&
                      max_logits.size(1) == h_q &&
                      max_logits.scalar_type() == ScalarType::Float,
                  "max_logits must be float32 with shape [s_q, h_q]");
  STD_TORCH_CHECK(lse.dim() == 2 && lse.size(0) == s_q && lse.size(1) == h_q &&
                      lse.scalar_type() == ScalarType::Float,
                  "lse must be float32 with shape [s_q, h_q]");
  STD_TORCH_CHECK(out.mutable_data_ptr() != max_logits.mutable_data_ptr() &&
                      out.mutable_data_ptr() != lse.mutable_data_ptr() &&
                      max_logits.mutable_data_ptr() != lse.mutable_data_ptr(),
                  "out, max_logits, and lse must not alias");

  SparseMlaQ8Kv8PrefillParams params{};
  params.s_q = s_q;
  params.s_kv = kv.size(0);
  params.h_q = h_q;
  params.h_kv = 1;
  params.d_qk = 512;
  params.d_v = 512;
  params.topk = topk;
  params.sm_scale_div_log2 = static_cast<float>(sm_scale * M_LOG2E);
  params.q = reinterpret_cast<const uint8_t*>(q.const_data_ptr());
  params.kv = reinterpret_cast<const uint8_t*>(kv.const_data_ptr());
  params.indices = indices.const_data_ptr<int32_t>();
  params.attn_sink = attn_sink.const_data_ptr<float>();
  params.topk_length = topk_length.const_data_ptr<int32_t>();
  params.q_scale_ptr = q_scale.const_data_ptr<float>();
  params.kv_scale_ptr = kv_scale.const_data_ptr<float>();
  params.stride_q_s_q = h_q * 512;
  params.stride_q_h_q = 512;
  params.stride_kv_s_kv = 512;
  params.stride_kv_h_kv = 512;
  params.stride_indices_s_q = topk;
  params.stride_indices_h_kv = topk;
  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.mutable_data_ptr());
  params.max_logits = max_logits.mutable_data_ptr<float>();
  params.lse = lse.mutable_data_ptr<float>();
  params.stream = get_current_cuda_stream(device_index);

  sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<512, true, true>(params);
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, q8kv8_sparse_prefill_ops) {
  q8kv8_sparse_prefill_ops.def(
      "sparse_mla_q8kv8_prefill_sm90("
      "Tensor q, Tensor kv, Tensor indices, Tensor q_scale, Tensor kv_scale, "
      "Tensor attn_sink, Tensor topk_length, Tensor! out, "
      "Tensor! max_logits, Tensor! lse, float sm_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, q8kv8_sparse_prefill_ops) {
  q8kv8_sparse_prefill_ops.impl("sparse_mla_q8kv8_prefill_sm90",
                                TORCH_BOX(&sparse_mla_q8kv8_prefill_sm90));
}
