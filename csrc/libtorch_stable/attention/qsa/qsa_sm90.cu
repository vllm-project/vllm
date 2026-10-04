// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SM90 BF16 QSA sparse paged GQA attention for prefill.
//
// Same semantics as `_qsa_sparse_paged_gqa_splitk_kernel` with NUM_SPLITS == 1
// (vllm/models/qwen4_exp/nvidia/ops/qsa.py): one CTA per (query row, KV head);
// the row attends to exactly its own selected tokens, and the epilogue applies
// the output gate after rounding the attention output to BF16.
//
// Both GEMMs are transposed so the selected keys take the wgmma M dimension and
// the (padded) query heads of one KV group take N = 16:
//   S^T[64 keys x 16 heads]  = K_tile[64 x D]  * Q^T[D x 16]    (A, B K-major)
//   O^T[D x 16 heads]       += V_tile^T[D x 64] * P^T[64 x 16]  (A MN-major)
// A per-token gather is one 512-byte K (or V) row per KV head, copied with
// 16-byte cp.async straight into the swizzled layouts the wgmma descriptors
// expect.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

namespace vllm::qsa_sm90 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

constexpr int kHeadDim = 256;
constexpr int kBlockN = 64;
constexpr int kHeads = 16;
constexpr int kThreads = 128;
// Token metadata is resolved two tiles ahead, so three tiles' worth is live.
constexpr int kMetaSlots = 3;
constexpr uint32_t kInvalidSlot = 0xffffffffu;

using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<kHeads>, Int<kHeadDim>>{}));
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutVt = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
using SmemLayoutP = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},
                                           Shape<Int<kHeads>, Int<kBlockN>>{}));
using MmaQK = decltype(make_tiled_mma(
    SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using MmaPV = decltype(make_tiled_mma(
    SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::K>{}));

// 75 KB, so three CTAs fit on one SM.
struct SharedStorage {
  cute::array_aligned<bf16, cosize_v<SmemLayoutQ>, 1024> q;
  cute::array_aligned<bf16, cosize_v<SmemLayoutK>, 1024> k;
  cute::array_aligned<bf16, cosize_v<SmemLayoutVt>, 1024> vt;
  cute::array_aligned<bf16, cosize_v<SmemLayoutP>, 1024> p;
  float reduce[kThreads / 32][kHeads];
  // Byte offset / 16 of each token's K (and V) row of this KV head, or
  // kInvalidSlot.
  uint32_t slot[kMetaSlots][kBlockN];
};

struct Params {
  const bf16* q;
  const bf16* k_cache;
  const bf16* v_cache;
  const int32_t* indices;
  const int32_t* block_table;
  const int32_t* token_to_req;
  const bf16* output_gate;
  bf16* out;
  int64_t stride_q_row, stride_q_head;
  int64_t stride_k_block, stride_k_token, stride_k_head;
  int64_t stride_indices_row, stride_table_req;
  int64_t stride_gate_row, stride_gate_head;
  int64_t stride_out_row, stride_out_head;
  int topk, page_size, page_table_width, num_cache_blocks, num_requests,
      group_size;
  float softmax_scale_log2;
};

__device__ __forceinline__ void cp_async_16(void* smem_dst,
                                            const void* gmem_src, bool valid) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(gmem_src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// A thread owns 4 head columns of a 64x16 / 256x16 wgmma accumulator:
// col = 2 * (lane % 4) + (i % 2) + 8 * ((i / 4) % 2) for fragment value i, so
// the slot of value i is known at compile time and per-column state stays in
// registers.
__device__ __forceinline__ constexpr int frag_slot(int i) {
  return (i & 1) + 2 * ((i >> 2) & 1);
}
__device__ __forceinline__ int slot_col(int slot, int lane) {
  return 2 * (lane & 3) + (slot & 1) + 8 * (slot >> 1);
}

__global__ void __launch_bounds__(kThreads, 3)
    qsa_sparse_gqa_sm90_kernel(const Params p) {
  extern __shared__ __align__(1024) char smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  const int request = p.token_to_req[row];
  const bool request_ok = request >= 0 && request < p.num_requests;
  const int safe_request = min(max(request, 0), p.num_requests - 1);
  const int32_t* index_row = p.indices + row * p.stride_indices_row;
  // Trailing count column of the packed selection buffer.
  const int valid_count = index_row[p.topk];
  const int num_cols = max(min(valid_count, p.topk), 0);
  const int num_tiles = (num_cols + kBlockN - 1) / kBlockN;
  const int first_head = kv_head * p.group_size;

  Tensor sQ = make_tensor(make_smem_ptr(smem.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(smem.k.data()), SmemLayoutK{});
  Tensor sVt = make_tensor(make_smem_ptr(smem.vt.data()), SmemLayoutVt{});
  Tensor sP = make_tensor(make_smem_ptr(smem.p.data()), SmemLayoutP{});

  // Q^T operand: [16 heads, D], zero rows for padded heads.
  for (int idx = tid; idx < kHeads * kHeadDim / 8; idx += kThreads) {
    const int h = idx / (kHeadDim / 8);
    const int c = (idx % (kHeadDim / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (h < p.group_size) {
      v = *reinterpret_cast<const uint4*>(
          p.q + row * p.stride_q_row + (first_head + h) * p.stride_q_head + c);
    }
    *reinterpret_cast<uint4*>(&sQ(h, c)) = v;
  }

  // Token metadata of a tile is resolved once by the first kBlockN threads;
  // the copies are then coalesced: a warp moves whole 512-byte K or V rows,
  // one 16-byte chunk per lane.
  auto resolve_tile = [&](int tile, int slot) {
    if (tid < kBlockN) {
      const int col = tile * kBlockN + tid;
      const int token = col < p.topk ? index_row[col] : -1;
      const int safe_token = max(token, 0);
      const int logical_page = safe_token / p.page_size;
      const int page_offset = safe_token % p.page_size;
      bool valid =
          request_ok && token >= 0 && logical_page < p.page_table_width;
      int physical_page = -1;
      if (valid) {
        physical_page =
            p.block_table[safe_request * p.stride_table_req + logical_page];
      }
      valid = valid && physical_page >= 0 && physical_page < p.num_cache_blocks;
      const int64_t element_offset =
          static_cast<int64_t>(physical_page) * p.stride_k_block +
          static_cast<int64_t>(page_offset) * p.stride_k_token +
          kv_head * p.stride_k_head;
      smem.slot[slot][tid] =
          valid ? static_cast<uint32_t>(element_offset >> 3) : kInvalidSlot;
    }
  };
  static_assert(kHeadDim / 8 == 32,
                "one 16-byte chunk per lane covers a 256-dim row");
  // Warp w copies rows 16w .. 16w+15, so a row's position inside its 8-row
  // swizzle atom is a compile-time constant and the swizzled addresses fold.
  constexpr int kRowsPerWarp = kBlockN / (kThreads / 32);
  auto load_rows = [&](int slot, uint32_t (&rows)[kRowsPerWarp]) {
    const uint4* src =
        reinterpret_cast<const uint4*>(&smem.slot[slot][warp * kRowsPerWarp]);
#pragma unroll
    for (int v = 0; v < kRowsPerWarp / 4; ++v) {
      const uint4 r = src[v];
      rows[4 * v + 0] = r.x;
      rows[4 * v + 1] = r.y;
      rows[4 * v + 2] = r.z;
      rows[4 * v + 3] = r.w;
    }
  };
  auto issue_k = [&](int slot) {
    uint32_t rows[kRowsPerWarp];
    load_rows(slot, rows);
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      const int j = warp * kRowsPerWarp + i;
      const bool valid = rows[i] != kInvalidSlot;
      const uint64_t offset = valid ? static_cast<uint64_t>(rows[i]) * 8 : 0;
      cp_async_16(&sK(j, lane * 8), p.k_cache + offset + lane * 8, valid);
    }
  };
  auto issue_v = [&](int slot) {
    uint32_t rows[kRowsPerWarp];
    load_rows(slot, rows);
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      const int j = warp * kRowsPerWarp + i;
      const bool valid = rows[i] != kInvalidSlot;
      const uint64_t offset = valid ? static_cast<uint64_t>(rows[i]) * 8 : 0;
      cp_async_16(&sVt(lane * 8, j), p.v_cache + offset + lane * 8, valid);
    }
  };

  MmaQK mma_qk;
  MmaPV mma_pv;
  auto thr_qk = mma_qk.get_thread_slice(tid);
  auto thr_pv = mma_pv.get_thread_slice(tid);
  Tensor tSrK = thr_qk.partition_fragment_A(sK);  // (MMA, MMA_M, MMA_K)
  Tensor tSrQ = thr_qk.partition_fragment_B(sQ);  // (MMA, MMA_N, MMA_K)
  Tensor tOrV = thr_pv.partition_fragment_A(sVt);
  Tensor tOrP = thr_pv.partition_fragment_B(sP);
  Tensor tSrS =
      partition_fragment_C(mma_qk, Shape<Int<kBlockN>, Int<kHeads>>{});
  Tensor tOrO =
      partition_fragment_C(mma_pv, Shape<Int<kHeadDim>, Int<kHeads>>{});
  Tensor tScS = thr_qk.partition_C(
      make_identity_tensor(Shape<Int<kBlockN>, Int<kHeads>>{}));
  Tensor tOcO = thr_pv.partition_C(
      make_identity_tensor(Shape<Int<kHeadDim>, Int<kHeads>>{}));
  clear(tOrO);

  float running_max[4], running_sum[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    running_max[s] = -1.0e20f;
    running_sum[s] = 0.0f;
  }

  auto qk_issue = [&]() {
    warpgroup_fence_operand(tSrS);
    warpgroup_arrive();
    mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int kb = 0; kb < size<2>(tSrK); ++kb) {
      cute::gemm(mma_qk, tSrK(_, _, kb), tSrQ(_, _, kb), tSrS);
      mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
  };
  auto qk_wait = [&]() {
    warpgroup_wait<0>();
    warpgroup_fence_operand(tSrS);
  };
  auto pv_gemm = [&]() {
    warpgroup_fence_operand(tOrO);
    warpgroup_arrive();
    CUTE_UNROLL
    for (int kb = 0; kb < size<2>(tOrV); ++kb) {
      cute::gemm(mma_pv, tOrV(_, _, kb), tOrP(_, _, kb), tOrO);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tOrO);
  };
  // Online softmax per head column over the keys of a tile; writes P^T and
  // rescales O. after_max_barrier runs once every warp's QK wgmma is done.
  auto softmax_tile = [&](int slot, auto&& after_max_barrier) {
    constexpr int kFragS = decltype(size(tSrS))::value;
    float tile_max[4] = {-1.0e20f, -1.0e20f, -1.0e20f, -1.0e20f};
    bool key_valid[kFragS];
    CUTE_UNROLL
    for (int i = 0; i < kFragS; ++i) {
      key_valid[i] = smem.slot[slot][get<0>(tScS(i))] != kInvalidSlot;
      const float x = key_valid[i] ? tSrS(i) * p.softmax_scale_log2 : -1.0e20f;
      tSrS(i) = x;
      tile_max[frag_slot(i)] = fmaxf(tile_max[frag_slot(i)], x);
    }
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
      tile_max[s] =
          fmaxf(tile_max[s], __shfl_xor_sync(0xffffffff, tile_max[s], 4));
      tile_max[s] =
          fmaxf(tile_max[s], __shfl_xor_sync(0xffffffff, tile_max[s], 8));
      tile_max[s] =
          fmaxf(tile_max[s], __shfl_xor_sync(0xffffffff, tile_max[s], 16));
    }
    if (lane < 4) {
      CUTE_UNROLL
      for (int s = 0; s < 4; ++s) {
        smem.reduce[warp][slot_col(s, lane)] = tile_max[s];
      }
    }
    __syncthreads();
    after_max_barrier();
    float alpha[4];
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
      const int col = slot_col(s, lane);
      float m = fmaxf(fmaxf(smem.reduce[0][col], smem.reduce[1][col]),
                      fmaxf(smem.reduce[2][col], smem.reduce[3][col]));
      m = fmaxf(running_max[s], m);
      alpha[s] = exp2f(running_max[s] - m);
      running_max[s] = m;
      running_sum[s] *= alpha[s];
    }
    CUTE_UNROLL
    for (int i = 0; i < kFragS; ++i) {
      const float prob =
          key_valid[i] ? exp2f(tSrS(i) - running_max[frag_slot(i)]) : 0.0f;
      running_sum[frag_slot(i)] += prob;
      sP(get<1>(tScS(i)), get<0>(tScS(i))) = bf16(prob);
    }
    CUTE_UNROLL
    for (int e = 0; e < size(tOrO); ++e) tOrO(e) *= alpha[frag_slot(e)];
  };

  // Single K and V buffers. A buffer is refilled only after a CTA-wide barrier
  // that every warp reaches after its wgmma on that buffer has completed: V(t)
  // right after the barrier that opens tile t (all PV(t-1) done), K(t+1) right
  // after the softmax max barrier (all QK(t) done). Metadata is resolved two
  // tiles ahead with its global loads overlapping QK's wgmma, so these copies
  // need no barrier of their own.
  if (num_tiles > 0) {
    resolve_tile(0, 0);
    if (num_tiles > 1) resolve_tile(1, 1);
    __syncthreads();
    issue_k(0);
  }
  cp_async_commit();

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int slot = tile % kMetaSlots;
    const bool has_next = tile + 1 < num_tiles;
    cp_async_wait<0>();  // K(tile) landed
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    issue_v(slot);
    cp_async_commit();
    qk_issue();
    if (tile + 2 < num_tiles) resolve_tile(tile + 2, (tile + 2) % kMetaSlots);
    qk_wait();
    softmax_tile(slot, [&] {
      if (has_next) issue_k((tile + 1) % kMetaSlots);
      cp_async_commit();
    });
    cp_async_wait<1>();  // V(tile) landed, K(tile + 1) may be in flight
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    pv_gemm();
  }
  cp_async_wait<0>();

  // All gate loads are issued here, ahead of the barriers below; loading each
  // next to its output store would serialize them, since the gate and output
  // pointers may alias.
  constexpr int kFragO = decltype(size(tOrO))::value;
  bf16 gate[kFragO];
  CUTE_UNROLL
  for (int e = 0; e < kFragO; ++e) {
    const int dim = get<0>(tOcO(e));
    const int head = get<1>(tOcO(e));
    gate[e] =
        head < p.group_size
            ? p.output_gate[row * p.stride_gate_row +
                            (first_head + head) * p.stride_gate_head + dim]
            : bf16(0.0f);
  }

  // Column sums across the lanes and warps that share a head column.
  CUTE_UNROLL
  for (int s = 0; s < 4; ++s) {
    running_sum[s] += __shfl_xor_sync(0xffffffff, running_sum[s], 4);
    running_sum[s] += __shfl_xor_sync(0xffffffff, running_sum[s], 8);
    running_sum[s] += __shfl_xor_sync(0xffffffff, running_sum[s], 16);
  }
  __syncthreads();
  if (lane < 4) {
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
      smem.reduce[warp][slot_col(s, lane)] = running_sum[s];
    }
  }
  __syncthreads();
  float inv_sum[4];
  bool has_values[4];
  CUTE_UNROLL
  for (int s = 0; s < 4; ++s) {
    const int col = slot_col(s, lane);
    const float total = smem.reduce[0][col] + smem.reduce[1][col] +
                        smem.reduce[2][col] + smem.reduce[3][col];
    has_values[s] = total > 0.0f;
    inv_sum[s] = 1.0f / fmaxf(total, 1.0e-20f);
  }
  CUTE_UNROLL
  for (int e = 0; e < kFragO; ++e) {
    const int dim = get<0>(tOcO(e));
    const int head = get<1>(tOcO(e));
    if (head < p.group_size) {
      // Round the attention output to BF16 before gating in FP32, matching
      // the Triton kernel's rounding boundary.
      const float attn = static_cast<float>(bf16(
          has_values[frag_slot(e)] ? tOrO(e) * inv_sum[frag_slot(e)] : 0.0f));
      const float g = static_cast<float>(gate[e]);
      p.out[row * p.stride_out_row + (first_head + head) * p.stride_out_head +
            dim] = bf16(attn / (1.0f + __expf(-g)));
    }
  }
}

}  // namespace vllm::qsa_sm90

namespace {

using torch::headeronly::ScalarType;

bool aligned16(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

// Prefill-only BF16 path; the caller keeps FP8 caches, decode and non-SM90
// devices on the Triton kernel.
void qsa_sparse_prefill_sm90(torch::stable::Tensor const& q,
                             torch::stable::Tensor const& k_cache,
                             torch::stable::Tensor const& v_cache,
                             torch::stable::Tensor const& indices,
                             torch::stable::Tensor const& block_table,
                             torch::stable::Tensor const& token_to_req,
                             torch::stable::Tensor const& output_gate,
                             torch::stable::Tensor& out) {
  namespace qsa = vllm::qsa_sm90;
  STD_TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                      indices.dim() == 2 && block_table.dim() == 2 &&
                      token_to_req.dim() == 1 && output_gate.dim() == 3 &&
                      out.dim() == 3,
                  "qsa_sparse_prefill_sm90: invalid tensor ranks");
  for (int d = 0; d < 4; ++d) {
    STD_TORCH_CHECK(v_cache.size(d) == k_cache.size(d) &&
                        v_cache.stride(d) == k_cache.stride(d),
                    "qsa_sparse_prefill_sm90: K and V views must share shape "
                    "and strides");
  }
  for (int d = 0; d < 3; ++d) {
    STD_TORCH_CHECK(
        out.size(d) == q.size(d) && output_gate.size(d) == q.size(d),
        "qsa_sparse_prefill_sm90: output and gate must match the query");
  }
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 &&
                      k_cache.scalar_type() == ScalarType::BFloat16 &&
                      output_gate.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "qsa_sparse_prefill_sm90: Q, K/V, gate and output must be "
                  "BF16");
  STD_TORCH_CHECK(indices.scalar_type() == ScalarType::Int &&
                      block_table.scalar_type() == ScalarType::Int &&
                      token_to_req.scalar_type() == ScalarType::Int,
                  "qsa_sparse_prefill_sm90: metadata must be int32");
  const int32_t device = q.get_device_index();
  STD_TORCH_CHECK(k_cache.get_device_index() == device &&
                      v_cache.get_device_index() == device &&
                      indices.get_device_index() == device &&
                      block_table.get_device_index() == device &&
                      token_to_req.get_device_index() == device &&
                      output_gate.get_device_index() == device &&
                      out.get_device_index() == device,
                  "qsa_sparse_prefill_sm90: tensors must share a device");
  STD_TORCH_CHECK(
      q.size(2) == qsa::kHeadDim && k_cache.size(3) == qsa::kHeadDim,
      "qsa_sparse_prefill_sm90: head_dim must be 256");
  const int64_t num_kv_heads = k_cache.size(2);
  STD_TORCH_CHECK(
      q.size(1) % num_kv_heads == 0 && q.size(1) / num_kv_heads <= qsa::kHeads,
      "qsa_sparse_prefill_sm90: GQA group must be at most 16");
  STD_TORCH_CHECK(indices.size(0) == q.size(0) && indices.size(1) >= 2 &&
                      token_to_req.size(0) == q.size(0),
                  "qsa_sparse_prefill_sm90: metadata must have one row per "
                  "query");
  STD_TORCH_CHECK(q.stride(2) == 1 && k_cache.stride(3) == 1 &&
                      output_gate.stride(2) == 1 && out.stride(2) == 1 &&
                      indices.stride(1) == 1 && block_table.stride(1) == 1 &&
                      token_to_req.stride(0) == 1,
                  "qsa_sparse_prefill_sm90: innermost dims must be contiguous");
  // Q rows are read and K/V rows gathered in 16-byte chunks; a token row is
  // addressed by one uint32 (byte offset / 16) relative to the K or V base.
  STD_TORCH_CHECK(
      q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && aligned16(q.data_ptr()),
      "qsa_sparse_prefill_sm90: query rows must be 16-byte "
      "aligned");
  STD_TORCH_CHECK(k_cache.stride(0) % 8 == 0 && k_cache.stride(1) % 8 == 0 &&
                      k_cache.stride(2) % 8 == 0 &&
                      aligned16(k_cache.data_ptr()) &&
                      aligned16(v_cache.data_ptr()),
                  "qsa_sparse_prefill_sm90: K/V rows must be 16-byte aligned");
  const int64_t max_element = (k_cache.size(0) - 1) * k_cache.stride(0) +
                              (k_cache.size(1) - 1) * k_cache.stride(1) +
                              (k_cache.size(2) - 1) * k_cache.stride(2);
  STD_TORCH_CHECK((max_element >> 3) < int64_t{0xffffffff},
                  "qsa_sparse_prefill_sm90: K/V cache span must stay below "
                  "64 GiB");
  const int64_t num_rows = q.size(0);
  if (num_rows == 0) return;

  qsa::Params params;
  params.q = reinterpret_cast<const qsa::bf16*>(q.data_ptr());
  params.k_cache = reinterpret_cast<const qsa::bf16*>(k_cache.data_ptr());
  params.v_cache = reinterpret_cast<const qsa::bf16*>(v_cache.data_ptr());
  params.indices = indices.const_data_ptr<int32_t>();
  params.block_table = block_table.const_data_ptr<int32_t>();
  params.token_to_req = token_to_req.const_data_ptr<int32_t>();
  params.output_gate =
      reinterpret_cast<const qsa::bf16*>(output_gate.data_ptr());
  params.out = reinterpret_cast<qsa::bf16*>(out.mutable_data_ptr());
  params.stride_q_row = q.stride(0);
  params.stride_q_head = q.stride(1);
  params.stride_k_block = k_cache.stride(0);
  params.stride_k_token = k_cache.stride(1);
  params.stride_k_head = k_cache.stride(2);
  params.stride_indices_row = indices.stride(0);
  params.stride_table_req = block_table.stride(0);
  params.stride_gate_row = output_gate.stride(0);
  params.stride_gate_head = output_gate.stride(1);
  params.stride_out_row = out.stride(0);
  params.stride_out_head = out.stride(1);
  params.topk = static_cast<int>(indices.size(1) - 1);
  params.page_size = static_cast<int>(k_cache.size(1));
  params.page_table_width = static_cast<int>(block_table.size(1));
  params.num_cache_blocks = static_cast<int>(k_cache.size(0));
  params.num_requests = static_cast<int>(block_table.size(0));
  params.group_size = static_cast<int>(q.size(1) / num_kv_heads);
  params.softmax_scale_log2 =
      (1.0f / sqrtf(static_cast<float>(qsa::kHeadDim))) * 1.4426950408889634f;

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  cudaStream_t stream = get_current_cuda_stream(device);
  constexpr int smem_size = sizeof(qsa::SharedStorage);
  auto kernel = qsa::qsa_sparse_gqa_sm90_kernel;
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "qsa_sparse_prefill_sm90: ", cudaGetErrorString(err));
  kernel<<<dim3(static_cast<unsigned>(num_rows),
                static_cast<unsigned>(num_kv_heads)),
           qsa::kThreads, smem_size, stream>>>(params);
  err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "qsa_sparse_prefill_sm90: ", cudaGetErrorString(err));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, qsa_sm90_ops) {
  // Output-gated QSA sparse attention over paged BF16 K/V (SM90, prefill).
  qsa_sm90_ops.def(
      "qsa_sparse_prefill_sm90(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor indices, Tensor block_table, Tensor token_to_req, "
      "Tensor output_gate, Tensor! out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, qsa_sm90_ops) {
  qsa_sm90_ops.impl("qsa_sparse_prefill_sm90",
                    TORCH_BOX(&qsa_sparse_prefill_sm90));
}
