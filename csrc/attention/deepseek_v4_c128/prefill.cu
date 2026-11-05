// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "prefill.cuh"
#include "prefill.h"

void c128_run(void* q, void* packed_q, void* kv, void* sink, void* lengths,
              void* out, void* raw_o, void* raw_m, void* raw_l, void* seq_lens,
              void* gather_lens, int num_query_tokens, int num_kv_rows,
              int compressed_width, int q_stride, int out_stride, float scale,
              cudaStream_t stream) {
  vllm::c128::Params p{};
  p.q = static_cast<bf16*>(packed_q);
  p.kv = static_cast<bf16*>(kv);
  p.out = static_cast<bf16*>(out);
  p.attn_sink = static_cast<float*>(sink);
  p.raw_o = static_cast<float*>(raw_o);
  p.raw_m = static_cast<float*>(raw_m);
  p.raw_l = static_cast<float*>(raw_l);
  p.seq_lens = static_cast<int*>(seq_lens);
  p.gather_lens = static_cast<int*>(gather_lens);
  p.query_tokens = num_query_tokens;
  p.compressed_width = compressed_width;
  p.topk_length = static_cast<int*>(lengths);
  p.s_q = (num_query_tokens + 7) / 8;
  p.h_q = 64;
  p.h_kv = 1;
  p.d_qk = 512;
  p.d_v = 512;
  p.s_kv = num_kv_rows;
  p.topk = ((num_kv_rows + 127) / 128) * 128;
  p.stride_q_s_q = 64 * 512;
  p.stride_q_h_q = 512;
  p.stride_kv_s_kv = 512;
  p.stride_out_s_q = out_stride;
  p.sm_scale_div_log2 = scale;
  p.stream = stream;
  vllm::c128::Kernel<false>::run(p);
  p.q = static_cast<bf16*>(q);
  p.s_q = num_query_tokens;
  p.stride_q_s_q = q_stride;
  vllm::c128::Kernel<true>::run(p);
  KU_CHECK_KERNEL_LAUNCH();
}
