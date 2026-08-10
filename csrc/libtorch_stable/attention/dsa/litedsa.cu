// SPDX-License-Identifier: Apache-2.0
//
// Grouped (packed) sparse prefill attention for DSA models on SM100:
// G adjacent queries' heads are packed into one 128-row attention call over
// the UNION of their top-k lists (dedup'd KV work), and a per-query
// query-major membership bitmask restores EXACT per-query attend-sets
// inside the kernel (LSE-equal to the per-query path). Two primitives:
//   litedsa_union_qm  — union + membership + fused logical->physical
//                         conversion, overwrite-write into caller buffers
//                         (see include/flashinfer/litedsa/*.cuh);
//   litedsa_masked_mla_fp8  — SM100 head128 fp8 (e4m3) sparse prefill forward
//                         with the qm membership mask (vendored FlashMLA
//                         derivative under vendor_fmla/).
// Python orchestration (version cache keyed on the shared-indexer buffer,
// persistent buffers, group packing) lives in
// vllm/model_executor/layers/litedsa.py.
//
// The vLLM CMake target compiles this TU for SM100a with CUDA 12.8; keep the
// standalone TVM-FFI override available for rapid kernel A/B iteration.

#include <cuda_runtime.h>
#include <array>
#include <mutex>
#include <optional>

#include <flashinfer/litedsa/litedsa_union.cuh>

#include "sm100/prefill/sparse/fwd/head128_fp8/phase1.cuh"

#include "../../torch_utils.h"

using torch::stable::Tensor;

namespace {
void check_cuda(cudaError_t error, const char* operation) {
  STD_TORCH_CHECK(error == cudaSuccess, operation, " failed: ",
                  cudaGetErrorString(error));
}

inline int32_t* iptr(const Tensor& t) {
  return reinterpret_cast<int32_t*>(const_cast<void*>(t.const_data_ptr()));
}
inline uint32_t* uptr(const Tensor& t) {
  return reinterpret_cast<uint32_t*>(const_cast<void*>(t.const_data_ptr()));
}
inline int num_sms_of(const Tensor& t) {
  int dev = t.get_device_index();
  int sms = 0;
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev),
             "cudaDeviceGetAttribute");
  return sms;
}
constexpr float kLog2e = 1.4426950408889634f;

void check_cuda_int32(const Tensor& t, const char* name, int device) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(
      t.scalar_type() == torch::headeronly::ScalarType::Int,
      name, " must have dtype int32");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(t.get_device_index() == device,
                  name, " must be on the same CUDA device");
}
}  // namespace

void litedsa_union_qm(const Tensor& topk_indices, Tensor& u_phys,
                        Tensor& counts, Tensor& memb_qm, const Tensor& req_pg,
                        const Tensor& block_table, int64_t block_size,
                        int64_t seq_space) {
  using namespace flashinfer::litedsa;
  STD_TORCH_CHECK(topk_indices.is_cuda(),
                  "topk_indices must be a CUDA tensor");
  const int device = topk_indices.get_device_index();
  check_cuda_int32(topk_indices, "topk_indices", device);
  check_cuda_int32(u_phys, "u_phys", device);
  check_cuda_int32(counts, "counts", device);
  check_cuda_int32(memb_qm, "memb_qm", device);
  check_cuda_int32(req_pg, "req_pg", device);
  check_cuda_int32(block_table, "block_table", device);
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  STD_TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [T, k]");
  STD_TORCH_CHECK(u_phys.dim() == 2 && memb_qm.dim() == 3,
                  "u_phys [ng, cap], memb_qm [ng, G, cap/32]");
  const int T = static_cast<int>(topk_indices.size(0));
  const int k = static_cast<int>(topk_indices.size(1));
  const int ng = static_cast<int>(u_phys.size(0));
  const int cap = static_cast<int>(u_phys.size(1));
  const int G = static_cast<int>(memb_qm.size(1));
  STD_TORCH_CHECK(ng > 0 && T == ng * G, "T must equal ng * G");
  STD_TORCH_CHECK(k > 0 && block_size > 0, "k and block_size must be positive");
  STD_TORCH_CHECK(cap == G * k && cap % 128 == 0,
                  "cap must be G*k (constructive union bound), 128-aligned");
  STD_TORCH_CHECK(cap <= 32768,
                  "cap must be <= 32768 for the uint16 rank representation");
  STD_TORCH_CHECK(memb_qm.size(0) == ng && G <= 16 &&
                      memb_qm.size(2) == cap / 32,
                  "memb_qm must be [ng, G, cap/32], G <= 16");
  STD_TORCH_CHECK((int64_t)G * (cap / 32) <= 16 * 1024,
                  "qm smem accumulator overflow (G*cap/32 > 16K words)");
  STD_TORCH_CHECK(counts.dim() == 1 && counts.numel() == ng &&
                      req_pg.dim() == 1 && req_pg.numel() == ng,
                  "counts/req_pg must be [ng]");
  STD_TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  STD_TORCH_CHECK(seq_space > 0 && seq_space <= (1 << 20),
                  "seq_space must be in (0, 1M]");
  const int s_words = static_cast<int>((seq_space + 31) / 32);
  STD_TORCH_CHECK(s_words % 32 == 0,
                  "seq_space must be 1024-position aligned");
  const int smem = litedsa_union_qm_smem_bytes(s_words);
  STD_TORCH_CHECK(smem <= 224 * 1024, "seq_space too large for smem bitmap");

  static std::array<std::once_flag, 64> attr_once;
  STD_TORCH_CHECK(device >= 0 && device < static_cast<int>(attr_once.size()),
                  "unsupported CUDA device index ", device);
  std::call_once(attr_once[device], [] {
    check_cuda(cudaFuncSetAttribute((void*)litedsa_union_qm_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    224 * 1024),
               "cudaFuncSetAttribute");
  });
  const cudaStream_t stream = get_current_cuda_stream();
  litedsa_union_qm_kernel<<<ng, kLiteDSAUnionThreads, smem, stream>>>(
      iptr(topk_indices), iptr(u_phys), iptr(counts), uptr(memb_qm),
      iptr(req_pg), iptr(block_table),
      static_cast<int>(block_table.size(1)),
      static_cast<int>(block_table.size(1)), static_cast<int>(block_size), k,
      G, cap, s_words);
  check_cuda(cudaGetLastError(), "litedsa_union_qm_kernel launch");
}

void litedsa_masked_mla_fp8(const Tensor& q, const Tensor& kv,
                        const Tensor& indices, double sm_scale,
                        double out_scale, const Tensor& topk_length,
                        const Tensor& memb_qm, int64_t h_per_q, Tensor& out,
                        Tensor& max_logits, Tensor& lse) {
  using bf16 = cutlass::bfloat16_t;
  STD_TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  const int device = q.get_device_index();
  const auto check_cuda = [device](const Tensor& t, const char* name) {
    STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    STD_TORCH_CHECK(t.get_device_index() == device,
                    name, " must be on the same CUDA device");
    STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  };
  check_cuda(q, "q");
  check_cuda(kv, "kv");
  check_cuda(indices, "indices");
  check_cuda(topk_length, "topk_length");
  check_cuda(memb_qm, "memb_qm");
  check_cuda(out, "out");
  check_cuda(max_logits, "max_logits");
  check_cuda(lse, "lse");
  STD_TORCH_CHECK(
      q.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
          kv.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "q and kv must have dtype float8_e4m3fn");
  STD_TORCH_CHECK(
      indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_length.scalar_type() == torch::headeronly::ScalarType::Int &&
          memb_qm.scalar_type() == torch::headeronly::ScalarType::Int,
      "indices, topk_length, and memb_qm must have dtype int32");
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "out must have dtype bfloat16");
  STD_TORCH_CHECK(
      max_logits.scalar_type() == torch::headeronly::ScalarType::Float &&
          lse.scalar_type() == torch::headeronly::ScalarType::Float,
      "max_logits and lse must have dtype float32");
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  STD_TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3,
                  "q [rows, 128, 576], kv [s_kv, 1, 576], "
                  "indices [rows, 1, topk]");
  const int s_q = static_cast<int>(indices.size(0));
  const int s_kv = static_cast<int>(kv.size(0));
  const int h_q = static_cast<int>(q.size(1));
  const int h_kv = static_cast<int>(kv.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  const int topk = static_cast<int>(indices.size(2));
  STD_TORCH_CHECK(s_q > 0 && s_kv > 0, "q and kv must be non-empty");
  STD_TORCH_CHECK(q.size(0) == s_q, "q rows must match indices rows");
  STD_TORCH_CHECK(h_q == 128, "packed path requires h_q == 128");
  STD_TORCH_CHECK(h_kv == 1 && indices.size(1) == 1,
                  "kv and indices must have one head");
  STD_TORCH_CHECK(d_qk == 576 && kv.size(2) == 576, "d_qk must be 576");
  STD_TORCH_CHECK(topk > 0 && topk % 128 == 0,
                  "topk must be a positive multiple of 128");
  STD_TORCH_CHECK(h_per_q > 0 && 128 % h_per_q == 0 && 128 / h_per_q <= 16,
                  "h_per_q must divide 128 with <= 16 queries per row");
  const int G = 128 / h_per_q;
  STD_TORCH_CHECK(topk_length.dim() == 1 && topk_length.numel() == s_q,
                  "topk_length must be [rows]");
  STD_TORCH_CHECK(memb_qm.dim() == 3 && memb_qm.size(0) == s_q &&
                      memb_qm.size(1) == G && memb_qm.size(2) == topk / 32,
                  "membership_qm must be [rows, G, topk/32]");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == s_q &&
                      out.size(1) == 128 && out.size(2) == 512,
                  "out must be [rows, 128, 512]");
  STD_TORCH_CHECK(max_logits.dim() == 2 && max_logits.size(0) == s_q &&
                      max_logits.size(1) == 128 && lse.dim() == 2 &&
                      lse.size(0) == s_q && lse.size(1) == 128,
                  "max_logits and lse must be [rows, 128]");

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
      (bf16*)const_cast<void*>(q.const_data_ptr()),
      (bf16*)const_cast<void*>(kv.const_data_ptr()),
      iptr(indices),
      nullptr,  // attn_sink
      iptr(topk_length),
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(kv.stride(0)),
      static_cast<int>(kv.stride(1)),
      static_cast<int>(indices.stride(0)),
      static_cast<int>(indices.stride(1)),
      (bf16*)const_cast<void*>(out.const_data_ptr()),
      reinterpret_cast<float*>(const_cast<void*>(max_logits.const_data_ptr())),
      reinterpret_cast<float*>(const_cast<void*>(lse.const_data_ptr())),
      num_sms_of(q),
      get_current_cuda_stream()};
  params.membership = nullptr;
  params.h_per_q = static_cast<int>(h_per_q);
  params.out_scale = (float)out_scale;
  params.topk_length_per_q = nullptr;
  params.q_group_div = 1;
  params.membership_qm = uptr(memb_qm);

  sm100::fwd::head128_fp8::run_fwd_phase1_kernel<576>(params);
}
