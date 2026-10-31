// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "core/registration.h"
#include "prefill.h"

#include <cuda_runtime.h>
#include <initializer_list>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <kerutils/supplemental/cuda_stream.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

void c128_prefill(const Tensor& q, const Tensor& packed, const Tensor& kv,
                  const Tensor& sink, const Tensor& lengths,
                  const Tensor& seq_lens, const Tensor& gather_lens,
                  Tensor& out, Tensor& raw_o, Tensor& raw_m, Tensor& raw_l,
                  int64_t compressed_width, double scale) {
  STD_TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 512,
                  "C128 requires padded [tokens, 64, 512] queries");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 &&
                      packed.scalar_type() == ScalarType::BFloat16 &&
                      kv.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "C128 query, packed query, KV and output must be BF16");
  STD_TORCH_CHECK(q.stride(1) == 512 && q.stride(2) == 1 &&
                      out.sizes() == q.sizes() && out.stride(1) == 512 &&
                      out.stride(2) == 1,
                  "invalid C128 query/output layout");
  int64_t padded = ((q.size(0) + 7) / 8) * 8;
  STD_TORCH_CHECK(
      packed.numel() == padded * 8 * 512 && packed.is_contiguous() &&
          raw_o.numel() == padded * 8 * 512 && raw_o.is_contiguous() &&
          raw_m.numel() == padded * 8 && raw_m.is_contiguous() &&
          raw_l.numel() == padded * 8 * 8 && raw_l.is_contiguous(),
      "invalid C128 workspace");
  STD_TORCH_CHECK(raw_o.scalar_type() == ScalarType::Float &&
                      raw_m.scalar_type() == ScalarType::Float &&
                      raw_l.scalar_type() == ScalarType::Float &&
                      sink.scalar_type() == ScalarType::Float &&
                      sink.numel() >= 8 && sink.is_contiguous(),
                  "invalid C128 floating-point buffers");
  STD_TORCH_CHECK(lengths.numel() == padded / 8 && lengths.is_contiguous() &&
                      lengths.scalar_type() == ScalarType::Int &&
                      seq_lens.numel() == 1 && gather_lens.numel() == 1 &&
                      seq_lens.scalar_type() == ScalarType::Int &&
                      gather_lens.scalar_type() == ScalarType::Int,
                  "invalid C128 metadata");
  STD_TORCH_CHECK(kv.dim() == 3 && kv.size(0) == 1 && kv.size(2) == 512 &&
                      kv.stride(1) == 512 && kv.stride(2) == 1 &&
                      compressed_width >= 0 && compressed_width <= kv.size(1),
                  "invalid C128 KV layout");
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &q, &packed, &kv, &sink, &lengths, &seq_lens, &gather_lens, &out,
           &raw_o, &raw_m, &raw_l}) {
    STD_TORCH_CHECK(t->device().is_cuda() && t->device() == q.device(),
                    "all C128 tensors must be on the same CUDA device");
  }
  STD_TORCH_CHECK(q.size(0) <= INT32_MAX && q.stride(0) <= INT32_MAX &&
                      out.stride(0) <= INT32_MAX && kv.size(1) <= INT32_MAX &&
                      q.size(0) * q.stride(0) <= INT32_MAX &&
                      q.size(0) * out.stride(0) <= INT32_MAX,
                  "C128 dimensions exceed int32");
  torch::stable::accelerator::DeviceGuard guard(q.get_device_index());
  c128_run(q.data_ptr(), packed.data_ptr(), kv.data_ptr(), sink.data_ptr(),
           lengths.data_ptr(), out.mutable_data_ptr(), raw_o.mutable_data_ptr(),
           raw_m.mutable_data_ptr(), raw_l.mutable_data_ptr(),
           seq_lens.data_ptr(), gather_lens.data_ptr(), q.size(0), kv.size(1),
           compressed_width, q.stride(0), out.stride(0),
           static_cast<float>(scale), kerutils::get_current_cuda_stream(q));
}

STABLE_TORCH_LIBRARY(_c128_C, m) {
  m.def(
      "prefill(Tensor q, Tensor packed, Tensor kv, Tensor sink, Tensor "
      "lengths, "
      "Tensor seq_lens, Tensor gather_lens, Tensor(a!) out, Tensor(b!) raw_o, "
      "Tensor(c!) raw_m, Tensor(d!) raw_l, int compressed_width, float scale) "
      "-> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_c128_C, CUDA, m) {
  m.impl("prefill", TORCH_BOX(&c128_prefill));
}

REGISTER_EXTENSION(_c128_C)
