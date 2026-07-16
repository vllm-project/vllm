// SPDX-License-Identifier: Apache-2.0
//
// Fused DSA (DeepSeek Sparse Attention) indexer top-k for SM100 (Blackwell).
// Streams KV in tiles and fuses fp8 MQA scoring (tcgen05 UMMA) + an online
// bucketed gate + compact top-k, so the [num_q, seq_len] logit matrix is never
// materialized. Three primitives (seed_prep / scan / select); the Python helper
// in vllm/model_executor/layers/dsa_litetopk.py orchestrates and allocates.
//
// Kernels are framework-agnostic (include/flashinfer/dsa_indexer/*.cuh, raw
// pointers). This file is the only torch-facing layer. Ported 1:1 from the
// FlashInfer TVM-FFI launcher; see that PR for provenance.
//
// NOTE: compile-unverified inside vLLM's build as of this draft; validate with
//   `uv pip install -e . --torch-backend=auto` on an SM100 host.

#include <cuda_runtime.h>
#include <optional>

#include <flashinfer/dsa_indexer/dsa_indexer.cuh>

#include "../../torch_utils.h"

using torch::stable::Tensor;

namespace {
inline void* ptr(const Tensor& t) {
  return const_cast<void*>(t.const_data_ptr());
}
inline float* fptr(const Tensor& t) {
  return reinterpret_cast<float*>(const_cast<void*>(t.const_data_ptr()));
}
inline int32_t* iptr(const Tensor& t) {
  return reinterpret_cast<int32_t*>(const_cast<void*>(t.const_data_ptr()));
}
}  // namespace

void dsa_litetopk_seed_prep(const Tensor& slog, int64_t num_buckets, int64_t topk,
                            int64_t cand_cap, int64_t emit_limit, double headroom,
                            int64_t probe_stride_tok, int64_t hist_stride, Tensor& origin,
                            Tensor& inv_delta, Tensor& th_bucket, Tensor& bcount,
                            Tensor& cand_val, Tensor& cand_idx, Tensor& cand_cnt) {
  STD_TORCH_CHECK(slog.dim() == 2, "slog must be 2D [Q, head]");
  const int Q = static_cast<int>(slog.size(0));
  const int head = static_cast<int>(slog.size(1));
  const int NB = static_cast<int>(num_buckets);
  const int K = static_cast<int>(topk);
  const int cap = static_cast<int>(cand_cap);
  STD_TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");
  STD_TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");

  cudaSetDevice(slog.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  if (seed_smem > 48 * 1024) {
    static bool attr_set = false;
    if (!attr_set) {
      cudaFuncSetAttribute((void*)seed_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           4 * 4096 * static_cast<int>(sizeof(int)));
      attr_set = true;
    }
  }
  const int emit_lim = emit_limit == 0 ? 0 : (emit_limit > 0 ? static_cast<int>(emit_limit) : head);
  const int pst = static_cast<int>(probe_stride_tok);
  const int hst = hist_stride > 1 ? static_cast<int>(hist_stride) : 1;

  seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
      fptr(slog), static_cast<int>(slog.stride(0)), head, NB, K, cap, emit_lim, pst, hst,
      static_cast<float>(headroom), fptr(origin), fptr(inv_delta), iptr(th_bucket),
      iptr(bcount), fptr(cand_val), iptr(cand_idx), iptr(cand_cnt));
}

void dsa_litetopk_scan(const Tensor& q, const Tensor& kv, const Tensor& kv_scales,
                       const Tensor& weights, const Tensor& cu_start, const Tensor& cu_end,
                       Tensor& origin, Tensor& inv_delta, Tensor& th_bucket, Tensor& cand_val,
                       Tensor& cand_idx, Tensor& cand_cnt, Tensor& bcount, int64_t num_buckets,
                       int64_t topk, int64_t refresh_every, int64_t num_kv_splits_override,
                       int64_t probe_group, int64_t probe_add_max) {
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  const int cand_cap = static_cast<int>(cand_val.size(1));
  const int num_buckets_i = static_cast<int>(num_buckets);
  const int topk_i = static_cast<int>(topk);
  STD_TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM,
                  "only GLM DSA H=32 D=128 is supported");
  const bool external_refresh = (refresh_every < 0);
  const int refresh_every_i = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every);

  cudaSetDevice(q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const int esz_fp8 = 1, esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  auto tm_q = make_2d(ptr(q), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM, seq_len * NUM_HEADS,
                      HEAD_DIM, BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  auto tm_kv = make_2d(ptr(kv), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM, seq_len_kv,
                       HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(ptr(kv_scales), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32, ks_aligned, 1,
                       BLOCK_KV, 1, 0, 0);
  auto tm_w = make_2d(ptr(weights), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32, NUM_HEADS, seq_len,
                      NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int smem = compute_smem_bytes();
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
                                                  NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
                                                  SPEC_THREADS, MATH_THREADS>;
  cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
  int num_kv_splits;
  if (num_kv_splits_override > 0) {
    num_kv_splits = static_cast<int>(num_kv_splits_override);
  } else {
    constexpr int kWaves = 4;
    const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
    num_kv_splits = (kWaves * NUM_SMS + qb - 1) / qb;
    const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
    if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
  }
  if (num_kv_splits < 1) num_kv_splits = 1;
  if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
  dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      (uint32_t)seq_len, (uint32_t)seq_len_kv, reinterpret_cast<uint32_t*>(iptr(cu_start)),
      reinterpret_cast<uint32_t*>(iptr(cu_end)), fptr(origin), fptr(inv_delta), iptr(th_bucket),
      iptr(bcount), (uint32_t)num_buckets_i, (uint32_t)topk_i, (uint32_t)refresh_every_i,
      (uint32_t)num_kv_splits, (uint32_t)probe_group,
      probe_group > 0 ? (((1ULL << 42) + (uint64_t)probe_group - 1) / (uint64_t)probe_group) : 0ULL,
      (uint32_t)probe_add_max, fptr(cand_val), iptr(cand_idx), iptr(cand_cnt), (uint32_t)cand_cap,
      tm_q, tm_kv, tm_ks, tm_w);

  if (external_refresh) {
    int block = 128;
    int grid_r = (seq_len + block - 1) / block;
    refresh_threshold_from_bcount_kernel<<<grid_r, block, 0, stream>>>(
        iptr(th_bucket), iptr(bcount), seq_len, num_buckets_i, topk_i);
  }
}

void dsa_litetopk_select(const Tensor& cand_val, const Tensor& cand_idx, const Tensor& cand_cnt,
                         const Tensor& origin, const Tensor& inv_delta, const Tensor& th_bucket,
                         int64_t num_buckets, int64_t topk, Tensor& out_val, Tensor& out_idx) {
  STD_TORCH_CHECK(cand_val.dim() == 2, "cand_val must be 2D [R, CAP]");
  const int R = static_cast<int>(cand_val.size(0));
  const int CAP = static_cast<int>(cand_val.size(1));
  const int K = static_cast<int>(topk);
  const int NB = static_cast<int>(num_buckets);
  STD_TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1, CAP]");
  STD_TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");

  cudaSetDevice(cand_val.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  compact_topk_min_thr_litetopk_kernel<<<R, 256, 0, stream>>>(
      fptr(cand_val), iptr(cand_idx), iptr(cand_cnt), fptr(origin), fptr(inv_delta), iptr(th_bucket),
      R, CAP, K, NB, fptr(out_val), iptr(out_idx), /*probe_group=*/0u, 0ULL, /*probe_add_max=*/0u,
      nullptr);
}
