// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI binding for the DeepSeek-V4 packed sparse prefill attention kernel
// on SM100 (bf16 operands, d_qk == d_v == 512).
//
// DSv4 at TP8 has 8 real heads per rank, but the stock FlashMLA sparse prefill
// pads q to h_q = 64, so 56/64 of every 64-row tile is zeros.  This entry
// drives the packed kernel: 16 query tokens x 8 real heads fill one 128-row
// tile, and per-row masking restores each token's own attend set over the
// group's ascending union list.
//
// DSv4's per-token attend set on that union list is TWO contiguous ranges:
//     query g attends union entries [0, pref_g)  UNION  [win_lo_g, win_hi_g)
// (a shared compressed prefix plus its own sliding window).  Both bounds are
// in union-RANK space, not key ids.
//
// Kept deliberately separate from litedsa.cu / litedsa_jit_binding.cu: those
// build the GLM fp8 kernel, and the two amalgamations cannot share a TU.

#include <cuda_runtime.h>

#include "litedsa_attention_sm100_dsv4.cuh"

#include "tvm_ffi_utils.h"

namespace {
constexpr float kLog2e = 1.4426950408889634f;
inline int dsv4_num_sms_of(TensorView t) {
  int dev = t.device().device_id;
  static thread_local int cached_dev = -1;
  static thread_local int cached_sms = 0;
  if (dev == cached_dev) return cached_sms;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cached_dev = dev;
  cached_sms = sms;
  return cached_sms;
}
}  // namespace

// q        [ng, 128, 512] bf16   row r = query r/h_per_q in the group, head r%h_per_q
// kv       [s_kv, 1, 512] bf16   flat dequantised workspace
// indices  [ng, 1, topk] int32   ascending union list, -1 invalid, topk % 64 == 0
// pref     [ng, G] int32         prefix bound per query (union-rank space)
// win_lo   [ng, G] int32         window start per query (union-rank space)
// win_hi   [ng, G] int32         window end   per query (union-rank space)
// counts   [ng] int32            union length per group (block-loop bound)
// attn_sink[128] fp32            per PACKED ROW (row r wants sink of head r%h_per_q)
// out      [ng, 128, 512] bf16
// lse      [ng, 128] fp32
void litedsa_masked_mla_dsv4_bf16(TensorView q, TensorView kv, TensorView indices,
                                  TensorView pref, TensorView win_lo, TensorView win_hi,
                                  TensorView counts, double sm_scale, int64_t h_per_q,
                                  TensorView out, TensorView lse, TensorView attn_sink) {
  using bf16 = cutlass::bfloat16_t;
  TVM_FFI_ICHECK(q.ndim() == 3 && kv.ndim() == 3 && indices.ndim() == 3)
      << "q [ng, 128, 512], kv [s_kv, 1, 512], indices [ng, 1, topk]";
  const int s_q = static_cast<int>(indices.size(0));  // == ng (packed groups)
  const int s_kv = static_cast<int>(kv.size(0));
  const int h_q = static_cast<int>(q.size(1));
  const int h_kv = static_cast<int>(kv.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  const int topk = static_cast<int>(indices.size(2));
  TVM_FFI_ICHECK(q.size(0) == s_q) << "q rows must match indices rows";
  TVM_FFI_ICHECK(h_q == 128) << "packed path requires h_q == 128";
  TVM_FFI_ICHECK(d_qk == 512 && kv.size(2) == 512) << "d_qk must be 512";
  TVM_FFI_ICHECK(topk % 64 == 0) << "topk must be a multiple of B_TOPK = 64";
  TVM_FFI_ICHECK(h_per_q > 0 && 128 % h_per_q == 0 && 128 / h_per_q <= 16)
      << "h_per_q must divide 128 with <= 16 queries per row";
  const int G = 128 / static_cast<int>(h_per_q);
  TVM_FFI_ICHECK(counts.size(0) == s_q) << "counts must be [ng]";
  TVM_FFI_ICHECK(pref.numel() == (int64_t)s_q * G && win_lo.numel() == (int64_t)s_q * G &&
                 win_hi.numel() == (int64_t)s_q * G)
      << "pref/win_lo/win_hi must be [ng, 128/h_per_q]";
  TVM_FFI_ICHECK(attn_sink.numel() == 128) << "attn_sink must be [128] (per packed row)";
  TVM_FFI_ICHECK(out.numel() == (int64_t)s_q * 128 * 512) << "out must be [ng, 128, 512]";
  TVM_FFI_ICHECK(lse.numel() == (int64_t)s_q * 128) << "lse must be [ng, 128]";

  cudaSetDevice(q.device().device_id);
  SparseAttnFwdParams params = {
      s_q,
      s_kv,
      h_q,
      h_kv,
      d_qk,
      512,
      topk,
      (float)sm_scale,
      (float)sm_scale * kLog2e,
      static_cast<bf16*>(q.data_ptr()),
      static_cast<bf16*>(kv.data_ptr()),
      static_cast<int32_t*>(indices.data_ptr()),
      static_cast<float*>(attn_sink.data_ptr()),
      static_cast<int32_t*>(counts.data_ptr()),  // topk_length: union length per group
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(kv.stride(0)),
      static_cast<int>(kv.stride(1)),
      static_cast<int>(indices.stride(0)),
      static_cast<int>(indices.stride(1)),
      static_cast<bf16*>(out.data_ptr()),
      nullptr,  // max_logits (unused by the DSv4 entry)
      static_cast<float*>(lse.data_ptr()),
      dsv4_num_sms_of(q),
      get_stream(q.device())};
  params.membership = nullptr;
  params.membership_qm = nullptr;
  params.h_per_q = static_cast<int>(h_per_q);
  params.out_scale = 1.0f;  // bf16 V: no dequant
  params.q_group_div = 1;
  params.topk_length_per_q = static_cast<const int*>(pref.data_ptr());
  params.win_lo_per_q = static_cast<const int*>(win_lo.data_ptr());
  params.win_hi_per_q = static_cast<const int*>(win_hi.data_ptr());

  sm100::fwd::head128_dsv4::run_fwd_phase1_kernel<512>(params);
}
