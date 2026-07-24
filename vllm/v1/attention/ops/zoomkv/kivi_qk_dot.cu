// KIVI partial-chunk dense/sparse rerank CUDA kernel.
//
// Drop-in CUDA replacement for
// ``cache_hub.kivi.qk_dot.partial_chunk_kivi_qk_ref``.  Mirrors
// ``partial_rerank_chunk_dense_sparse_kernel`` in
// ``cache_hub/rerank/rerank_topk.cu`` (same grid/block/sort topology), but
// with the inner per-token dot replaced by KIVI per-(quest-)chunk
// per-channel 4-bit dequant + raw-q dot.
//
// Per-thread work:
//   1. Read this thread's token index = chunk_id * CHUNK_SIZE + lane_in_chunk.
//   2. Load 16 int32 (== 128 dims of 4-bit codes) for this token.
//   3. For each of the 16 dim-blocks of 8 dims:
//        unpack 8 codes (0..15), dequant: K_d = code_d * scale_c[d] + mn_c[d],
//        accumulate K_d * q[d].
//   4. Bitonic-sort scores across the 16 lanes of one chunk.
//   5. Write dense_topk or sparse_topk scores/indices to out buffers.
//
// Layouts (A-1 chunk-major packed K):
//   chunk_ids       : [bs, kv_heads, nk]                              int64
//   dense_mask      : [bs, kv_heads, nk]                              bool
//   packed_K        : [bs, kv_heads, n_chunks, D/8, CHUNK_SIZE]       int32
//                     (chunk-major: 16 lanes of one chunk for the same
//                      dim-block sit at consecutive int32 offsets, so a
//                      half-warp's read collapses to one 64-byte sector.)
//   chunk_min       : [bs, kv_heads, n_chunks_max, D]                 bf16
//   chunk_max       : [bs, kv_heads, n_chunks_max, D]                 bf16
//   raw_q           : [bs, kv_heads, D]                               bf16
//   out_scores      : [bs, kv_heads, nk * CHUNK_SIZE]                 fp32
//   out_indices     : [bs, kv_heads, nk * CHUNK_SIZE]                 int64
//
// out_indices semantics: still a chunk-internal token id of the form
//   real_chunk_id * CHUNK_SIZE + lane_in_chunk
// (same as the previous token-major layout) so all downstream global
// top-K consumers are unchanged.
//
// Fixed compile-time constants (production):
//   CHUNK_SIZE = 16   (group_size = quest_chunk_size = 16)
//   BITS       = 4    (levels = 15, feat_per_int = 8)
//   D_MAX      = 128  (head_dim; loop over D/8 = 16 dim-blocks)

#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

namespace {

struct KiviChunkDenseSparseParams {
  const void* __restrict__ chunk_ids_ptr;   // [bs, kv, nk] int64
  const void* __restrict__ dense_mask_ptr;  // [bs, kv, nk] bool
  const void* __restrict__ packed_K_ptr;  // [bs, kv, n_chunks, D/8, CHUNK_SIZE]
                                          // int32
  const void* __restrict__ chunk_min_ptr;  // [bs, kv, n_chunks_max, D] bf16
  const void* __restrict__ chunk_max_ptr;  // [bs, kv, n_chunks_max, D] bf16
  const void* __restrict__ raw_q_ptr;      // [bs, kv, D] bf16
  void* __restrict__ out_scores_ptr;       // [bs, kv, nk*CHUNK_SIZE] fp32
  void* __restrict__ out_indices_ptr;      // [bs, kv, nk*CHUNK_SIZE] int64

  int num_chunks;       // == nk
  int n_chunks_packed;  // packed_K chunk-axis size (chunk capacity)
  int n_chunks_max;     // chunk_min/max chunk capacity
  int head_dim;         // 128 (D)
  int n_pack;           // D / 8  (== 16 for D=128)
  int dense_topk;
  int sparse_topk;

  // packed_K strides in int32 elements (chunk-major):
  //   index = bs*pK_bs + head*pK_head + chunk*pK_chunk + dimblock*pK_dimblock +
  //   lane
  int pK_bs_stride;
  int pK_head_stride;
  int pK_chunk_stride;     // n_pack * CHUNK_SIZE
  int pK_dimblock_stride;  // CHUNK_SIZE

  // chunk_min/max strides in bf16 elements
  int cm_bs_stride;
  int cm_head_stride;
  int cm_chunk_stride;  // head_dim

  // raw_q strides in bf16 elements
  int q_bs_stride;
  int q_head_stride;

  // chunk_ids strides in int64 elements
  int ci_bs_stride;
  int ci_head_stride;

  // dense_mask strides in bytes (bool == uint8 in CUDA)
  int dm_bs_stride;
  int dm_head_stride;

  // out_scores / out_indices strides in fp32/int64 elements
  int os_bs_stride;
  int os_head_stride;
};

template <int D_MAX, int CHUNK_SIZE>
__global__ void partial_chunk_kivi_qk_dense_sparse_kernel(
    const KiviChunkDenseSparseParams params) {
  constexpr int CHUNKS_PER_WARP = WARP_SIZE / CHUNK_SIZE;  // 16->2, 8->4

  const int bidx = blockIdx.x;  // bs
  const int bidy = blockIdx.y;  // kv_head
  const int bidz = blockIdx.z;  // chunk-block index
  const int tidx = threadIdx.x;

  const int num_chunks = params.num_chunks;
  const int n_chunks_packed = params.n_chunks_packed;
  const int dense_topk = params.dense_topk;
  const int sparse_topk = params.sparse_topk;
  const int head_dim = params.head_dim;
  const int n_pack = params.n_pack;
  const int dim_blocks = n_pack;  // == D / 8

  const int warp_id = tidx / WARP_SIZE;
  const int lane_id = tidx % WARP_SIZE;
  const int sub_id = lane_id / CHUNK_SIZE;         // 0..CHUNKS_PER_WARP-1
  const int lane_in_chunk = lane_id % CHUNK_SIZE;  // 0..CHUNK_SIZE-1

  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int chunks_per_block = warps_per_block * CHUNKS_PER_WARP;
  const int chunk_slot =
      bidz * chunks_per_block + warp_id * CHUNKS_PER_WARP + sub_id;

  const bool valid_slot = (chunk_slot < num_chunks);

  // ---- chunk id + dense flag ----
  const int64_t* chunk_ids_ptr =
      reinterpret_cast<const int64_t*>(params.chunk_ids_ptr) +
      bidx * params.ci_bs_stride + bidy * params.ci_head_stride;
  const uint8_t* dense_mask_ptr =
      reinterpret_cast<const uint8_t*>(params.dense_mask_ptr) +
      bidx * params.dm_bs_stride + bidy * params.dm_head_stride;

  int64_t real_chunk_id = 0;
  bool is_dense = false;
  if (valid_slot) {
    real_chunk_id = __ldg(chunk_ids_ptr + chunk_slot);
    is_dense = (dense_mask_ptr[chunk_slot] != 0);
  }
  // Bounds-check the chunk id against the chunk-major buffer (instead of
  // checking a per-token kv_len).  Invalid lanes write -inf scores so any
  // downstream global top-k won't pick this slot; token_idx is just used
  // for the bitonic shuffle / output index, so it's still safe to compute
  // unconditionally.
  const bool chunk_in_range =
      (real_chunk_id >= 0) && (real_chunk_id < n_chunks_packed);
  int token_idx = static_cast<int>(real_chunk_id) * CHUNK_SIZE + lane_in_chunk;

  // ---- pointer setup ----
  const int32_t* packed_K_ptr =
      reinterpret_cast<const int32_t*>(params.packed_K_ptr) +
      bidx * params.pK_bs_stride + bidy * params.pK_head_stride;
  const __nv_bfloat16* chunk_min_ptr =
      reinterpret_cast<const __nv_bfloat16*>(params.chunk_min_ptr) +
      bidx * params.cm_bs_stride + bidy * params.cm_head_stride;
  const __nv_bfloat16* chunk_max_ptr =
      reinterpret_cast<const __nv_bfloat16*>(params.chunk_max_ptr) +
      bidx * params.cm_bs_stride + bidy * params.cm_head_stride;
  const __nv_bfloat16* raw_q_ptr =
      reinterpret_cast<const __nv_bfloat16*>(params.raw_q_ptr) +
      bidx * params.q_bs_stride + bidy * params.q_head_stride;

  // Per-chunk shared memory for raw_q[D] + chunk_min[D] + chunk_scale[D].
  // For D=128 bf16 that is 3*128*2 = 768 bytes / chunk; with
  // chunks_per_block=8 we use 6 KiB shared per block.  We size for the
  // template instantiation D_MAX (== 128 in production).
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  const int per_chunk_floats = D_MAX * 3;
  const int local_chunk = warp_id * CHUNKS_PER_WARP + sub_id;
  __nv_bfloat16* sm_q = smem + local_chunk * per_chunk_floats + 0 * D_MAX;
  __nv_bfloat16* sm_min = smem + local_chunk * per_chunk_floats + 1 * D_MAX;
  __nv_bfloat16* sm_scale = smem + local_chunk * per_chunk_floats + 2 * D_MAX;

  // Cooperative load: CHUNK_SIZE lanes per chunk × n_pack dim_blocks total,
  // 8 dims per dim_block.  When CHUNK_SIZE >= dim_blocks (16 lanes per
  // chunk, D=128 → 16 blocks) each lane loads 1 block; with CHUNK_SIZE=8
  // each lane loads 2 blocks (stride = CHUNK_SIZE).  uint4 vec loads keep
  // it coalesced regardless of stride.
  if (valid_slot) {
    const int chunk_offset = static_cast<int>(real_chunk_id) * head_dim;
    const __nv_bfloat16 levels_bf = __float2bfloat16(15.0f);
    for (int dim_block = lane_in_chunk; dim_block < dim_blocks;
         dim_block += CHUNK_SIZE) {
      const uint4 q_v =
          __ldg(reinterpret_cast<const uint4*>(raw_q_ptr) + dim_block);
      *reinterpret_cast<uint4*>(&sm_q[dim_block * 8]) = q_v;

      const uint4 mn_v =
          __ldg(reinterpret_cast<const uint4*>(chunk_min_ptr + chunk_offset) +
                dim_block);
      const uint4 mx_v =
          __ldg(reinterpret_cast<const uint4*>(chunk_max_ptr + chunk_offset) +
                dim_block);
      __nv_bfloat16 mn_arr[8];
      __nv_bfloat16 mx_arr[8];
      *reinterpret_cast<uint4*>(&mn_arr[0]) = mn_v;
      *reinterpret_cast<uint4*>(&mx_arr[0]) = mx_v;
      __nv_bfloat16 sc_arr[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const __nv_bfloat16 diff = __hsub(mx_arr[j], mn_arr[j]);
        sc_arr[j] = __hdiv(diff, levels_bf);
      }
      *reinterpret_cast<uint4*>(&sm_min[dim_block * 8]) = mn_v;
      *reinterpret_cast<uint4*>(&sm_scale[dim_block * 8]) =
          *reinterpret_cast<uint4*>(&sc_arr[0]);
    }
  }
  __syncwarp();

  // ---- per-token dequant + dot ----
  // Use fp32 accumulator to match torch's default bf16 reduction
  // (torch.sum(bf16) accumulates in fp32 then casts to bf16).
  //
  // The inner loop is bf162-vectorized (HFMA2-class throughput): we pack
  // adjacent (j, j+1) lanes into a __nv_bfloat162 pair and run one __hmul2
  // / __hadd2 per pair.  Each element of the pair is computed with the
  // exact same bf16 rounding as the original scalar __hmul / __hadd, so
  // the per-dim K*q products are bit-identical to the previous version.
  // The fp32 accumulation order is preserved (.x of pair p, then .y of
  // pair p, ...) so the final score is also bit-identical.
  float score_f = -1e30f;
  if (valid_slot && chunk_in_range) {
    float acc = 0.0f;
    // Chunk-major addressing: packed_K[chunk, dim_block, lane].
    // For a fixed dim_block, the 16 lanes of one chunk read 16
    // consecutive int32 elements -> a single 64-byte coalesced load
    // per half-warp (vs. 16 strided loads in the old token-major
    // layout).
    const int32_t* chunk_pack_ptr =
        packed_K_ptr +
        static_cast<int64_t>(real_chunk_id) * params.pK_chunk_stride +
        lane_in_chunk;

#pragma unroll(4)
    for (int i = 0; i < dim_blocks; ++i) {
      // 8 bf16s per dim_block, viewed as 4 bf162 pairs.
      __nv_bfloat162 q_blk[4];
      __nv_bfloat162 mn_blk[4];
      __nv_bfloat162 sc_blk[4];
      *reinterpret_cast<uint4*>(&q_blk[0]) =
          *reinterpret_cast<const uint4*>(&sm_q[i * 8]);
      *reinterpret_cast<uint4*>(&mn_blk[0]) =
          *reinterpret_cast<const uint4*>(&sm_min[i * 8]);
      *reinterpret_cast<uint4*>(&sc_blk[0]) =
          *reinterpret_cast<const uint4*>(&sm_scale[i * 8]);

      const uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(
          chunk_pack_ptr + i * params.pK_dimblock_stride));
// KIVI per-channel dequant: K_bf16 = (code_bf * scale_bf) + mn_bf
// (two bf16 ops, NOT FMA — matches the Python reference exactly).
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t code_lo = (packed >> (p * 8 + 0)) & 0xFu;
        const uint32_t code_hi = (packed >> (p * 8 + 4)) & 0xFu;
        const __nv_bfloat16 code_lo_bf =
            __int2bfloat16_rn(static_cast<int>(code_lo));
        const __nv_bfloat16 code_hi_bf =
            __int2bfloat16_rn(static_cast<int>(code_hi));
        const __nv_bfloat162 code_pair =
            __halves2bfloat162(code_lo_bf, code_hi_bf);
        const __nv_bfloat162 mul_pair = __hmul2(code_pair, sc_blk[p]);
        const __nv_bfloat162 k_pair = __hadd2(mul_pair, mn_blk[p]);
        const __nv_bfloat162 prod_pair = __hmul2(k_pair, q_blk[p]);
        // Preserve scalar accumulation order (j=0..7) by adding
        // .x (== old j=2p) first, then .y (== old j=2p+1).
        acc += __low2float(prod_pair);
        acc += __high2float(prod_pair);
      }
    }
    score_f = acc;
  }

// ---- bitonic sort within each CHUNK_SIZE-lane half-warp (descending) ----
#pragma unroll
  for (int k = 2; k <= CHUNK_SIZE; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j >= 1; j >>= 1) {
      float other_score = __shfl_xor_sync(0xFFFFFFFF, score_f, j);
      int other_idx = __shfl_xor_sync(0xFFFFFFFF, token_idx, j);
      bool desc = ((lane_in_chunk & k) == 0);
      bool is_lower = ((lane_in_chunk & j) == 0);
      bool want_larger = (desc == is_lower);
      bool do_swap =
          want_larger ? (other_score > score_f) : (other_score < score_f);
      if (do_swap) {
        score_f = other_score;
        token_idx = other_idx;
      }
    }
  }

  // ---- write output (uniform nk * CHUNK_SIZE layout) ----
  if (valid_slot) {
    const int out_base = bidx * params.os_bs_stride +
                         bidy * params.os_head_stride +
                         chunk_slot * CHUNK_SIZE + lane_in_chunk;
    const int effective_topk = is_dense ? dense_topk : sparse_topk;
    if (lane_in_chunk < effective_topk) {
      reinterpret_cast<float*>(params.out_scores_ptr)[out_base] = score_f;
      reinterpret_cast<int64_t*>(params.out_indices_ptr)[out_base] =
          static_cast<int64_t>(token_idx);
    } else {
      reinterpret_cast<float*>(params.out_scores_ptr)[out_base] = -1e30f;
      reinterpret_cast<int64_t*>(params.out_indices_ptr)[out_base] = 0;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
void partial_chunk_kivi_qk_dense_sparse_interface(
    at::Tensor chunk_ids,   // [bs, kv, nk] int64
    at::Tensor dense_mask,  // [bs, kv, nk] bool
    at::Tensor packed_K,    // [bs, kv, n_chunks, n_pack, group_size] int32
    at::Tensor chunk_min,   // [bs, kv, n_chunks_max, D] bf16
    at::Tensor chunk_max,   // [bs, kv, n_chunks_max, D] bf16
    at::Tensor raw_q,       // [bs, kv, D] bf16
    int dense_topk, int sparse_topk,
    int group_size,         // 8 or 16
    at::Tensor out_scores,  // [bs, kv, nk * group_size] fp32
    at::Tensor out_indices  // [bs, kv, nk * group_size] int64
) {
  TORCH_CHECK(chunk_ids.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(dense_mask.scalar_type() == at::ScalarType::Bool);
  TORCH_CHECK(packed_K.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(chunk_min.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(chunk_max.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(raw_q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      out_scores.scalar_type() == at::ScalarType::Float,
      "out_scores must be fp32 (the global top-K reads it as fp32 directly); "
      "got ",
      out_scores.scalar_type());
  TORCH_CHECK(out_indices.scalar_type() == at::ScalarType::Long);

  TORCH_CHECK(chunk_ids.dim() == 3 && dense_mask.dim() == 3);
  TORCH_CHECK(packed_K.dim() == 5,
              "packed_K must be chunk-major [bs, kv, n_chunks, n_pack, "
              "group_size]; got dim=",
              packed_K.dim());
  TORCH_CHECK(chunk_min.dim() == 4 && chunk_max.dim() == 4);
  TORCH_CHECK(raw_q.dim() == 3);
  TORCH_CHECK(out_scores.dim() == 3 && out_indices.dim() == 3);
  TORCH_CHECK(packed_K.stride(-1) == 1);
  TORCH_CHECK(chunk_min.stride(-1) == 1);
  TORCH_CHECK(chunk_max.stride(-1) == 1);
  TORCH_CHECK(raw_q.stride(-1) == 1);

  const int bs = packed_K.size(0);
  const int kv_heads = packed_K.size(1);
  const int n_chunks_pkd = packed_K.size(2);
  const int n_pack = packed_K.size(3);
  const int pkd_group = packed_K.size(4);

  const int n_chunks_max = chunk_min.size(2);
  const int head_dim = chunk_min.size(3);
  const int nk = chunk_ids.size(2);

  TORCH_CHECK(head_dim == 128 || head_dim == 256,
              "kivi cuda kernel: head_dim must be 128 or 256; got ", head_dim);
  TORCH_CHECK(n_pack * 8 == head_dim, "kivi cuda kernel: n_pack(", n_pack,
              ")*8 != head_dim(", head_dim, ")");
  TORCH_CHECK(raw_q.size(2) == head_dim, "raw_q head_dim mismatch: expected ",
              head_dim, " got ", raw_q.size(2));
  TORCH_CHECK(chunk_max.size(2) == n_chunks_max &&
              chunk_max.size(3) == head_dim);
  TORCH_CHECK(group_size == 8 || group_size == 16,
              "kivi cuda kernel: group_size must be 8 or 16; got ", group_size);
  TORCH_CHECK(pkd_group == group_size, "packed_K inner-dim (", pkd_group,
              ") must equal group_size (", group_size, ")");
  // Inner two axes of packed_K must be contiguous so that lane-stride 1
  // and dim-block-stride == group_size both hold without surprises.
  TORCH_CHECK(packed_K.stride(-2) == group_size,
              "packed_K dim-block stride must equal group_size (", group_size,
              "); got ", packed_K.stride(-2));
  TORCH_CHECK(dense_topk >= 1 && dense_topk <= group_size);
  TORCH_CHECK(sparse_topk >= 1 && sparse_topk <= group_size);
  TORCH_CHECK(out_scores.size(0) == bs && out_scores.size(1) == kv_heads);
  TORCH_CHECK(out_indices.size(0) == bs && out_indices.size(1) == kv_heads);
  TORCH_CHECK(out_scores.size(2) >= nk * group_size,
              "out_scores too small: need ", nk * group_size, " got ",
              out_scores.size(2));
  TORCH_CHECK(out_indices.size(2) >= nk * group_size,
              "out_indices too small: need ", nk * group_size, " got ",
              out_indices.size(2));

  constexpr int BLOCK_SIZE = 128;
  const int chunks_per_warp = WARP_SIZE / group_size;
  const int chunks_per_block = (BLOCK_SIZE / WARP_SIZE) * chunks_per_warp;
  const int grid_z = (nk + chunks_per_block - 1) / chunks_per_block;

  KiviChunkDenseSparseParams params;
  params.chunk_ids_ptr = chunk_ids.data_ptr();
  params.dense_mask_ptr = dense_mask.data_ptr();
  params.packed_K_ptr = packed_K.data_ptr();
  params.chunk_min_ptr = chunk_min.data_ptr();
  params.chunk_max_ptr = chunk_max.data_ptr();
  params.raw_q_ptr = raw_q.data_ptr();
  params.out_scores_ptr = out_scores.data_ptr();
  params.out_indices_ptr = out_indices.data_ptr();

  params.num_chunks = nk;
  params.n_chunks_packed = n_chunks_pkd;
  params.n_chunks_max = n_chunks_max;
  params.head_dim = head_dim;
  params.n_pack = n_pack;
  params.dense_topk = dense_topk;
  params.sparse_topk = sparse_topk;

  params.pK_bs_stride = static_cast<int>(packed_K.stride(0));
  params.pK_head_stride = static_cast<int>(packed_K.stride(1));
  params.pK_chunk_stride = static_cast<int>(packed_K.stride(2));
  params.pK_dimblock_stride = static_cast<int>(packed_K.stride(3));

  params.cm_bs_stride = static_cast<int>(chunk_min.stride(0));
  params.cm_head_stride = static_cast<int>(chunk_min.stride(1));
  params.cm_chunk_stride = static_cast<int>(chunk_min.stride(2));

  params.q_bs_stride = static_cast<int>(raw_q.stride(0));
  params.q_head_stride = static_cast<int>(raw_q.stride(1));

  params.ci_bs_stride = static_cast<int>(chunk_ids.stride(0));
  params.ci_head_stride = static_cast<int>(chunk_ids.stride(1));

  params.dm_bs_stride = static_cast<int>(dense_mask.stride(0));
  params.dm_head_stride = static_cast<int>(dense_mask.stride(1));

  params.os_bs_stride = static_cast<int>(out_scores.stride(0));
  params.os_head_stride = static_cast<int>(out_scores.stride(1));

  const c10::Device dev = packed_K.device();
  c10::cuda::CUDAGuard device_guard(dev);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{(unsigned)bs, (unsigned)kv_heads, (unsigned)grid_z};
  const auto block = dim3{BLOCK_SIZE};
  // Shared memory: chunks_per_block × (raw_q + min + scale) × D × 2 bytes
  const int smem_bytes =
      chunks_per_block * 3 * head_dim * sizeof(__nv_bfloat16);

  if (head_dim == 256) {
    if (group_size == 16) {
      partial_chunk_kivi_qk_dense_sparse_kernel<256, 16>
          <<<grid, block, smem_bytes, stream>>>(params);
    } else {
      partial_chunk_kivi_qk_dense_sparse_kernel<256, 8>
          <<<grid, block, smem_bytes, stream>>>(params);
    }
  } else if (group_size == 16) {
    partial_chunk_kivi_qk_dense_sparse_kernel<128, 16>
        <<<grid, block, smem_bytes, stream>>>(params);
  } else {
    partial_chunk_kivi_qk_dense_sparse_kernel<128, 8>
        <<<grid, block, smem_bytes, stream>>>(params);
  }

  auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "kivi qk dot kernel launch failed: ", cudaGetErrorString(result));
}

#ifndef ZOOMKV_UNIFIED_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial_chunk_kivi_qk_dense_sparse",
        &partial_chunk_kivi_qk_dense_sparse_interface);
}
#endif
