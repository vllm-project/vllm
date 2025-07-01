// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#include <cuda.h>

#include <torch/all.h>

__device__ int64_t save_blocks(int* block_offset, int64_t range_start,
                               int64_t range_end, int64_t block_size,
                               int64_t input_block_count, int64_t kv_seqlen) {
  if (range_start >= kv_seqlen) {
    return input_block_count;
  }
  if (range_end > kv_seqlen) {
    range_end = kv_seqlen;
  }
  int64_t current_block_count = input_block_count;
  for (int idx = range_start; idx < range_end; idx += block_size) {
    block_offset[current_block_count++] = idx;
  }
  return current_block_count;
}

__global__ void convert_vertical_slash_indexes_kernel(
    const int* q_seqlens,         // [BATCH, ]
    const int* kv_seqlens,        // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int64_t N_HEADS, int64_t N_ROWS, int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
    int64_t NNZ_V, int64_t NNZ_S,
    bool causal  // True for intra, False for succ
) {
  const int batch_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int group_idx = blockIdx.z;

  int64_t q_seqlen = q_seqlens[batch_idx];
  int64_t kv_seqlen = kv_seqlens[batch_idx];
  int64_t block_idx_m = group_idx * blockDim.x + threadIdx.x;
  int64_t start_m = block_idx_m * BLOCK_SIZE_M;
  if (start_m >= q_seqlen) {
    return;
  }
  int64_t end_m = start_m + BLOCK_SIZE_M;
  vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
  slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
  int64_t row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
  block_count += row_offset;
  block_offset += row_offset * NNZ_S;
  column_count += row_offset;
  column_index += row_offset * NNZ_V;

  bool has_slash = true;
  int64_t tmp_col_cnt = 0, tmp_blk_cnt = 0;
  int64_t s = 0, v = 0;
  int64_t v_idx = vertical_indexes[v++];
  int64_t s_idx = slash_indexes[s++];
  if (causal) {
    while (s_idx >= end_m + (kv_seqlen - q_seqlen) && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + (kv_seqlen - q_seqlen)) has_slash = false;
    s_idx = max((kv_seqlen - q_seqlen) + end_m - s_idx, BLOCK_SIZE_M);
  } else {
    while (s_idx >= end_m + kv_seqlen && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + kv_seqlen) has_slash = false;
    s_idx = max(kv_seqlen + end_m - s_idx, BLOCK_SIZE_M);
  }

  int64_t range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
  if (!has_slash) {
    if (causal) {
      range_start = (kv_seqlen - q_seqlen) + end_m;
      range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
    } else {
      range_start = kv_seqlen;
      range_end = kv_seqlen + BLOCK_SIZE_N;
    }
  }

  bool slash_finished = false;
  while (1) {
    if (v_idx < range_end) {
      if (v_idx < range_start) {
        column_index[tmp_col_cnt++] = v_idx;
      }
      if (v < NNZ_V) {
        v_idx = vertical_indexes[v++];
      } else {
        if (causal)
          v_idx = end_m + BLOCK_SIZE_N + (kv_seqlen - q_seqlen);
        else
          v_idx = end_m + BLOCK_SIZE_N + kv_seqlen;
      }
    } else {
      if ((s < NNZ_S && causal) ||
          (s < NNZ_S && !causal && slash_indexes[s] >= start_m)) {
        if (causal)
          s_idx = max((kv_seqlen - q_seqlen) + end_m - slash_indexes[s++],
                      BLOCK_SIZE_M);
        else
          s_idx = max(kv_seqlen + end_m - slash_indexes[s++], BLOCK_SIZE_M);
      } else {
        if (v == NNZ_V || (v_idx > range_start && causal)) {
          // add the last vertical if no more slash
          if (v == NNZ_V && !causal && v_idx < kv_seqlen) {
            column_index[tmp_col_cnt++] = v_idx;
          }
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          break;
        } else {
          if (causal) {
            range_start = (kv_seqlen - q_seqlen) + end_m;
            range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
          } else {
            // if slash_finished but there are vertical left, save current
            // blocks
            tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                      BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
            range_start = kv_seqlen;
            range_end = kv_seqlen + BLOCK_SIZE_N;
          }
          slash_finished = true;
        }
      }
      if (!slash_finished) {
        if (s_idx > range_end + BLOCK_SIZE_M) {
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          range_start = s_idx - BLOCK_SIZE_M;
          range_end = s_idx;
        } else if (s_idx > range_end) {
          range_end += BLOCK_SIZE_M;
        }
      }
    }
  }

  block_count[0] = tmp_blk_cnt;
  column_count[0] = tmp_col_cnt;
}

void convert_vertical_slash_indexes_64x64(
    const int* q_seqlens,         // [BATCH, ]
    const int* kv_seqlens,        // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int64_t BATCH_SIZE, int64_t N_HEADS, int64_t N_ROWS, int64_t BLOCK_SIZE_M,
    int64_t BLOCK_SIZE_N, int64_t NNZ_V, int64_t NNZ_S, bool causal) {
  const int N_THREADS = 64;
  const dim3 dimBlock(N_THREADS);
  const dim3 dimGrid(N_HEADS, BATCH_SIZE, (N_ROWS + N_THREADS - 1) / N_THREADS);
  convert_vertical_slash_indexes_kernel<<<dimGrid, dimBlock>>>(
      q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, block_count,
      block_offset, column_count, column_index, N_HEADS, N_ROWS, BLOCK_SIZE_M,
      BLOCK_SIZE_N, NNZ_V, NNZ_S, causal);
}

/**
 * Implements the Algorithm 4 in paper https://arxiv.org/abs/2407.02490.
 *
 * This function builds the index of each row of blocks from vertical indices
 * and slash indices. The vertical indices are treated as points, while the
 * slash indices are converted as ranges. The output consists of the merged
 * ranges and separate column indices, where the ranges are represented by
 * block indices.
 *
 * The implementation is referenced from the original MInference repo:
 * https://github.com/microsoft/MInference/blob/main/csrc/vertical_slash_index.cu.
 */
void convert_vertical_slash_indexes(
    torch::Tensor& block_count,      // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,     // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,         // [BATCH, ]
    torch::Tensor kv_seqlens,        // [BATCH, ]
    torch::Tensor vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int64_t context_size, int64_t block_size_M, int64_t block_size_N,
    bool causal) {
  cudaSetDevice(q_seqlens.get_device());

  int batch_size = slash_indexes.size(0);
  int num_heads = slash_indexes.size(1);
  int nnz_slash = slash_indexes.size(2);
  int nnz_vertical = vertical_indexes.size(2);
  int num_rows = (context_size + block_size_M - 1) / block_size_M;

  convert_vertical_slash_indexes_64x64(
      q_seqlens.data_ptr<int>(), kv_seqlens.data_ptr<int>(),
      vertical_indexes.data_ptr<int>(), slash_indexes.data_ptr<int>(),
      block_count.data_ptr<int>(), block_offset.data_ptr<int>(),
      column_count.data_ptr<int>(), column_index.data_ptr<int>(), batch_size,
      num_heads, num_rows, block_size_M, block_size_N, nnz_vertical, nnz_slash,
      causal);
}

__global__ void convert_vertical_slash_indexes_kernel_mergehead(
    const int* q_seqlens,         // [BATCH, ]
    const int* kv_seqlens,        // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    const int* per_head_vertical_topkv, const int* per_head_slash_topkv,
    int* block_count,   // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int64_t N_HEADS, int64_t N_ROWS, int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
    int64_t NNZ_V, int64_t NNZ_S,
    bool causal  // True for intra, False for succ
) {
  const int batch_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int group_idx = blockIdx.z;

  int64_t q_seqlen = q_seqlens[batch_idx];
  int64_t kv_seqlen = kv_seqlens[batch_idx];
  int64_t block_idx_m = group_idx * blockDim.x + threadIdx.x;
  int64_t start_m = block_idx_m * BLOCK_SIZE_M;
  if (start_m >= q_seqlen) {
    return;
  }
  int64_t end_m = start_m + BLOCK_SIZE_M;
  vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
  slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
  int64_t row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
  block_count += row_offset;
  block_offset += row_offset * NNZ_S;
  column_count += row_offset;
  column_index += row_offset * NNZ_V;

  // MergeHead: each head has it's unique max topk NNZ_V，NNZ_S. (NNZ_V，NNZ_S
  // above is buffer size, use to compute offset)
  NNZ_S = per_head_slash_topkv[head_idx];
  NNZ_V = per_head_vertical_topkv[head_idx];

  bool has_slash = true;
  int64_t tmp_col_cnt = 0, tmp_blk_cnt = 0;
  int64_t s = 0, v = 0;
  int64_t v_idx = vertical_indexes[v++];
  int64_t s_idx = slash_indexes[s++];
  if (causal) {
    while (s_idx >= end_m + (kv_seqlen - q_seqlen) && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + (kv_seqlen - q_seqlen)) has_slash = false;
    s_idx = max((kv_seqlen - q_seqlen) + end_m - s_idx, BLOCK_SIZE_M);
  } else {
    while (s_idx >= end_m + kv_seqlen && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + kv_seqlen) has_slash = false;
    s_idx = max(kv_seqlen + end_m - s_idx, BLOCK_SIZE_M);
  }

  int64_t range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
  if (!has_slash) {
    if (causal) {
      range_start = (kv_seqlen - q_seqlen) + end_m;
      range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
    } else {
      range_start = kv_seqlen;
      range_end = kv_seqlen + BLOCK_SIZE_N;
    }
  }

  bool slash_finished = false;
  while (1) {
    if (v_idx < range_end) {
      if (v_idx < range_start) {
        column_index[tmp_col_cnt++] = v_idx;
      }
      if (v < NNZ_V) {
        v_idx = vertical_indexes[v++];
      } else {
        if (causal)
          v_idx = end_m + BLOCK_SIZE_N + (kv_seqlen - q_seqlen);
        else
          v_idx = end_m + BLOCK_SIZE_N + kv_seqlen;
      }
    } else {
      if ((s < NNZ_S && causal) ||
          (s < NNZ_S && !causal && slash_indexes[s] >= start_m)) {
        if (causal)
          s_idx = max((kv_seqlen - q_seqlen) + end_m - slash_indexes[s++],
                      BLOCK_SIZE_M);
        else
          s_idx = max(kv_seqlen + end_m - slash_indexes[s++], BLOCK_SIZE_M);
      } else {
        if (v == NNZ_V || (v_idx > range_start && causal)) {
          // add the last vertical if no more slash
          if (v == NNZ_V && !causal && v_idx < kv_seqlen) {
            column_index[tmp_col_cnt++] = v_idx;
          }
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          break;
        } else {
          if (causal) {
            range_start = (kv_seqlen - q_seqlen) + end_m;
            range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
          } else {
            // if slash_finished but there are vertical left, save current
            // blocks
            tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                      BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
            range_start = kv_seqlen;
            range_end = kv_seqlen + BLOCK_SIZE_N;
          }
          slash_finished = true;
        }
      }
      if (!slash_finished) {
        if (s_idx > range_end + BLOCK_SIZE_M) {
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          range_start = s_idx - BLOCK_SIZE_M;
          range_end = s_idx;
        } else if (s_idx > range_end) {
          range_end += BLOCK_SIZE_M;
        }
      }
    }
  }

  block_count[0] = tmp_blk_cnt;
  column_count[0] = tmp_col_cnt;
}

void convert_vertical_slash_indexes_64x64_mergehead(
    const int* q_seqlens,         // [BATCH, ]
    const int* kv_seqlens,        // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* per_head_vertical_topkv, int* per_head_slash_topkv,
    int* block_count,   // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,  // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int64_t BATCH_SIZE, int64_t N_HEADS, int64_t N_ROWS, int64_t BLOCK_SIZE_M,
    int64_t BLOCK_SIZE_N, int64_t NNZ_V, int64_t NNZ_S, bool causal) {
  const int N_THREADS = 64;
  const dim3 dimBlock(N_THREADS);
  const dim3 dimGrid(N_HEADS, BATCH_SIZE, (N_ROWS + N_THREADS - 1) / N_THREADS);
  convert_vertical_slash_indexes_kernel_mergehead<<<dimGrid, dimBlock>>>(
      q_seqlens, kv_seqlens, vertical_indexes, slash_indexes,
      per_head_vertical_topkv, per_head_slash_topkv, block_count, block_offset,
      column_count, column_index, N_HEADS, N_ROWS, BLOCK_SIZE_M, BLOCK_SIZE_N,
      NNZ_V, NNZ_S, causal);
}

/**
 * Implements the Algorithm 4 in paper https://arxiv.org/abs/2407.02490.
 *
 * Like the above convert_vertical_slash_indexes, but with
 * pre-computed vertical and slash counts.
 */
void convert_vertical_slash_indexes_mergehead(
    torch::Tensor& block_count,            // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,           // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,               // [BATCH, ]
    torch::Tensor kv_seqlens,              // [BATCH, ]
    torch::Tensor vertical_indexes,        // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,           // [BATCH, N_HEADS, NNZ_S]
    torch::Tensor vertical_indices_count,  // [N_HEADS, ]
    torch::Tensor slash_indices_count,     // [N_HEADS, ]
    int64_t context_size, int64_t block_size_M, int64_t block_size_N,
    bool causal) {
  cudaSetDevice(q_seqlens.get_device());

  int batch_size = slash_indexes.size(0);
  int num_heads = slash_indexes.size(1);
  int nnz_slash = slash_indexes.size(2);
  int nnz_vertical = vertical_indexes.size(2);
  int num_rows = (context_size + block_size_M - 1) / block_size_M;

  convert_vertical_slash_indexes_64x64_mergehead(
      q_seqlens.data_ptr<int>(), kv_seqlens.data_ptr<int>(),
      vertical_indexes.data_ptr<int>(), slash_indexes.data_ptr<int>(),
      vertical_indices_count.data_ptr<int>(),
      slash_indices_count.data_ptr<int>(), block_count.data_ptr<int>(),
      block_offset.data_ptr<int>(), column_count.data_ptr<int>(),
      column_index.data_ptr<int>(), batch_size, num_heads, num_rows,
      block_size_M, block_size_N, nnz_vertical, nnz_slash, causal);
}
