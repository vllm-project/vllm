#pragma once

#include "cutlass/bfloat16.h"

enum class ModelType { V32, MODEL1 };

struct __align__(4 * 8) DecodingSchedMeta {
  int begin_req_idx, end_req_idx;      // Both inclusive
  int begin_block_idx, end_block_idx;  // Inclusive, exclusive
  int begin_split_idx;
  int is_first_req_splitted, is_last_req_splitted;
  int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams {  // TODO Change name to DenseAttnDecodeParams
  using index_t = int64_t;

  int b;  // batch size
  int s_q;
  int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
  int d, d_v;         // K/V dimension
  int h_q, h_k;       // The number of Q/K heads
  int num_blocks;     // Number of blocks in total
  int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
  bool is_causal;
  float scale_softmax, scale_softmax_log2;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ o_ptr;
  float* __restrict__ softmax_lse_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t o_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t o_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t o_head_stride;

  int* __restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;
  int* __restrict__ seqlens_k_ptr;

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
  int num_sm_parts;
  int* __restrict__ num_splits_ptr;

  int total_num_splits;
  float* __restrict__ softmax_lseaccum_ptr;
  float* __restrict__ oaccum_ptr;

  cudaStream_t stream;
};

struct SparseAttnDecodeParams {
  int b, s_q;
  int h_q, h_kv;
  int d_qk, d_v;
  float sm_scale, sm_scale_div_log2;
  int num_blocks, page_block_size, topk;
  ModelType model_type;

  cutlass::bfloat16_t* __restrict__ q;   // [b, s_q, h_q, d_qk]
  cutlass::bfloat16_t* __restrict__ kv;  // [num_blocks, page_block_size, d_qk]
  int* __restrict__ indices;             // [b, s_q, topk]
  int* __restrict__ topk_length;         // [b], may be nullptr
  float* __restrict__ attn_sink;         // [h_q], may be nullptr

  float* __restrict__ lse;                // [b, s_q, h_q]
  cutlass::bfloat16_t* __restrict__ out;  // [b, s_q, h_q, d_v]

  int extra_num_blocks, extra_page_block_size, extra_topk;
  cutlass::bfloat16_t* __restrict__ extra_kv;  // [extra_num_blocks,
                                               // extra_page_block_size, d_qk]
  int* __restrict__ extra_indices;             // [b, s_q, extra_topk]
  int* __restrict__ extra_topk_length;         // [b], may be nullptr

  int stride_q_b, stride_q_s_q, stride_q_h_q;
  int stride_kv_block, stride_kv_row;
  int stride_indices_b, stride_indices_s_q;
  int stride_lse_b, stride_lse_s_q;
  int stride_o_b, stride_o_s_q, stride_o_h_q;
  int stride_extra_kv_block, stride_extra_kv_row;
  int stride_extra_indices_b, stride_extra_indices_s_q;

  cudaStream_t stream;

  // SplitKV-related parameters
  float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
  float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
  int stride_lse_accum_split, stride_lse_accum_s_q;
  int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;  // [num_sm_parts,
                                                                // ], contiguous
  int* __restrict__ num_splits_ptr;  // [batch_size+1, ], contiguous
  int num_sm_parts;
};

struct CombineParams {
  int b, s_q, h_q, d_v;

  float* __restrict__ lse;  // [b, s_q, h_q]
  void* __restrict__ out;   // [b, s_q, h_q, d_v]
  int stride_lse_b, stride_lse_s_q;
  int stride_o_b, stride_o_s_q, stride_o_h_q;

  float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
  float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
  int stride_lse_accum_split, stride_lse_accum_s_q;
  int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;  // [num_sm_parts,
                                                                // ], contiguous
  int* __restrict__ num_splits_ptr;  // [batch_size+1, ], contiguous
  int num_sm_parts;

  float* attn_sink;  // [h_q], may be nullptr

  cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
  int b;  // batch size
  int s_q;
  int block_size_n;
  int fixed_overhead_num_blocks;

  int topk, extra_topk;  // -1 if sparse attention (or extra topk) is disabled
  int* __restrict__ topk_length, * __restrict__ extra_topk_length;

  int* __restrict__ seqlens_k_ptr;  // Only necessary for dense attention

  DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
  int* __restrict__ num_splits_ptr;
  int num_sm_parts;

  cudaStream_t stream;
};

struct SparseAttnFwdParams {
  int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
  float sm_scale, sm_scale_div_log2;

  // Input tensors
  cutlass::bfloat16_t* __restrict__ q;   // [s_q, h_q, d_qk]
  cutlass::bfloat16_t* __restrict__ kv;  // [s_kv, h_kv, d_qk]
  int* __restrict__ indices;             // [s_q, h_kv, topk]
  float* __restrict__ attn_sink;         // [h_q], may be nullptr
  int* __restrict__ topk_length;         // [s_q], may be nullptr

  // Strides
  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  // Output tensors
  cutlass::bfloat16_t* __restrict__ out;  // [s_q, h_q, d_v]
  float* __restrict__ max_logits;         // [s_q, h_q]
  float* __restrict__ lse;                // [s_q, h_q]

  int num_sm;
  cudaStream_t stream;

  // Masked group-packing extension (LiteTopK): when membership != nullptr,
  // head-row r attends token j iff membership[s_q_idx*topk + j] has bit
  // (r / h_per_q) set. Membership bits are only set for valid entries, so
  // this REPLACES the per-token validity mask in the masked path.
  const uint16_t* membership = nullptr;  // [s_q, topk]
  int h_per_q = 0;

  // fp8 variant only: epilogue output multiplier (V dequant scale, i.e.
  // k_scale). The bf16 kernel ignores it.
  float out_scale = 1.0f;

  // Per-QUERY causal prefix bounds on the ascending union list [s_q, G]
  // (G = h_q / h_per_q). Row r may attend list entries < bound of query
  // r / h_per_q. Replaces membership for causal-superset semantics.
  const int* topk_length_per_q = nullptr;

  // KV-split: q_group_div consecutive s_q rows (list slices) share ONE Q
  // row; outputs stay per-row and are LSE-merged by the caller.
  int q_group_div = 1;

  // Query-major membership bits [s_q, G, topk/32]: bit r of row (s, q)
  // says union entry r belongs to query q's own top-k. Exact semantics
  // like `membership`, but consumed as two direct word loads per 64-token
  // half-block (no warp13 bit transpose). Mutually exclusive with
  // `membership`.
  const uint32_t* membership_qm = nullptr;
};

// We have some kernels that implement both prefill and decode modes in a single
// kernel (with different template instantiations). The following enum helps to
// distinguish the modes.
enum class SparseAttnFwdMode {
  Prefill,            // Normal prefill mode
  DecodeWithSplitKV,  // To trigger decoding mode for kernels that support both
                      // prefill and decode
};

template <SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v =
    std::bool_constant<FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV>::value;

template <SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT =
    std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams,
                       SparseAttnFwdParams>;
