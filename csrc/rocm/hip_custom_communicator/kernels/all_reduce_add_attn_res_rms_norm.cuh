#pragma once

// All-reduce fused with AttnRes and its RMSNorm.

#include "all_gather.cuh"
#include "attn_res.cuh"
#include "common.cuh"
#include "reduce_scatter.cuh"

namespace hip_custom_communicator {

enum class AllReduceAddAttnResRmsNormAlgorithm {
  one_shot_pull,
  two_shot_pull_rows,
  two_shot_pull_columns,
  two_shot_push
};

template <int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, typename DTYPE,
          int ROWS, int K, int P>
DINLINE void add_attn_res_rms_norm_rows(
    int offs_m, const DTYPE (&sum)[ROWS][K][P], int input_size_m,
    int input_size_n, const DTYPE* prefix_ptr, int64_t prefix_stride_m,
    int64_t prefix_stride_n, const DTYPE* blocks_ptr, int64_t blocks_stride_m,
    int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    int num_blocks, float eps, float output_eps) {
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false>(
      sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
      blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
      norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
      output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
      prefix_output_stride_m, prefix_output_stride_n, output_ptr,
      output_stride_m, output_stride_n, block_output_ptr, block_output_stride_m,
      block_output_stride_n, input_size_m, input_size_n, offs_m, 0, num_blocks,
      eps, output_eps, inv_hidden);
}

template <typename DTYPE, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES, bool CHUNKED>
DINLINE void add_attn_res_rms_norm_rows(
    int offs_m, const DTYPE* input_ptr, int64_t input_stride_m,
    int64_t input_stride_n, int input_size_m, int input_size_n,
    const DTYPE* prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
    const DTYPE* blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const DTYPE* norm_weight_ptr,
    int64_t norm_weight_stride_n, const DTYPE* qk_weight_ptr,
    int64_t qk_weight_stride_n, const DTYPE* output_norm_weight_ptr,
    int64_t output_norm_weight_stride_n, DTYPE* prefix_output_ptr,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    DTYPE* output_ptr, int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* block_output_ptr, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int num_blocks, float eps,
    float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  const float inv_hidden = 1.0f / static_cast<float>(input_size_n);
  attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, Layout::tile, false, TILE_M,
           kPacks, kPackElems, CHUNKED>(
      input_ptr, input_stride_m, input_stride_n, prefix_ptr, prefix_stride_m,
      prefix_stride_n, blocks_ptr, blocks_stride_m, blocks_stride_r,
      blocks_stride_n, norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
      qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
      prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
      output_ptr, output_stride_m, output_stride_n, block_output_ptr,
      block_output_stride_m, block_output_stride_n, input_size_m, input_size_n,
      offs_m, 0, num_blocks, eps, output_eps, inv_hidden);
}

template <typename DTYPE, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* __restrict__ prefix_ptr, int64_t prefix_stride_m,
        int64_t prefix_stride_n, int prefix_size_m, int prefix_size_n,
        const DTYPE* __restrict__ blocks_ptr, int64_t blocks_stride_m,
        int64_t blocks_stride_r, int64_t blocks_stride_n, int blocks_size_m,
        int blocks_size_r, int blocks_size_n,
        const DTYPE* __restrict__ norm_weight_ptr, int64_t norm_weight_stride_n,
        int norm_weight_size_n, const DTYPE* __restrict__ qk_weight_ptr,
        int64_t qk_weight_stride_n, int qk_weight_size_n,
        const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* __restrict__ output_ptr,
        int64_t output_stride_m, int64_t output_stride_n, int output_size_m,
        int output_size_n, DTYPE* __restrict__ block_output_ptr,
        int64_t block_output_stride_m, int64_t block_output_stride_n,
        int block_output_size_m, int block_output_size_n, int num_blocks,
        float eps, float output_eps) {
  const int tiles = (input_size_m + TILE_M - 1) / TILE_M;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x)
    add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K, THREADS_PER_BLOCK,
                               WAVE_SIZE, PACK_BYTES, CHUNKED>(
        t * TILE_M, input_ptr, input_stride_m, input_stride_n, input_size_m,
        input_size_n, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, num_blocks, eps,
        output_eps);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_one_shot_pull(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;

  if constexpr (QUANTIZE_INT4)
    for (int t = blockIdx.x; t * TILE_M < m; t += gridDim.x)
      for (int c = 0; c < n; c += kSpan) {
        DTYPE mine[TILE_M][kPacks][kPackElems];
        load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            &input_ptr, 1, 0, 0, 0, input_stride_m, input_stride_n, 1, m, n, m,
            n, mine, t * TILE_M, c, 1);
        store<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, true, Layout::tile,
              TAIL>(mine, staging_ptr_array, staging_ptr_array_size, rank, 0, 0,
                    n, 1, 1, m, n, m, n, t * TILE_M, c, 1);
      }

  std::conditional_t<QUANTIZE_INT4, char*, const DTYPE*> entries[WORLD];
#pragma unroll
  for (int r = 0; r < WORLD; ++r) {
    if constexpr (QUANTIZE_INT4)
      entries[r] = load(staging_ptr_array, staging_ptr_array_size, r);
    else
      entries[r] = load(input_ptr_array, input_ptr_array_size, r);
  }
  constexpr Until kFirst = QUANTIZE_INT4 ? Until::launched : Until::started;
  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, kFirst, 0);

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE copies[WORLD][TILE_M][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::tile_pull, QUANTIZE_INT4, DTYPE, WORLD,
                 TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, TAIL>(
          entries, QUANTIZE_INT4 ? int64_t{n} : input_ptr_array_stride_m,
          QUANTIZE_INT4 ? int64_t{1} : input_ptr_array_stride_n, m, n, n, row,
          c, copies);
      DTYPE sum[TILE_M][kPacks][kPackElems];
#pragma unroll
      for (int i = 0; i < TILE_M; ++i)
#pragma unroll
        for (int k = 0; k < kPacks; ++k)
#pragma unroll
          for (int j = 0; j < kPackElems; ++j) {
            float acc = static_cast<float>(copies[0][i][k][j]);
#pragma unroll
            for (int r = 1; r < WORLD; ++r)
              acc += static_cast<float>(copies[r][i][k][j]);
            sum[i][k][j] = static_cast<DTYPE>(acc);
          }

      if constexpr (CHUNKED)
        store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
            m, n, m, n, row, c, 1);
      else
        add_attn_res_rms_norm_rows<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE>(
            row, sum, m, n, prefix_ptr, prefix_stride_m, prefix_stride_n,
            blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
            norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
            qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, num_blocks, eps,
            output_eps);
    }
    if constexpr (CHUNKED)
      add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K,
                                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                 CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
          prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
          output_ptr, output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, num_blocks, eps,
          output_eps);
  }

  barrier<WORLD>(signals_ptr_array, signals_ptr_array_size,
                 signals_stride_parity, signals_stride_block,
                 signals_stride_writer, signals_size_parity, signals_size_block,
                 signals_size_writer, status_ptr, status_stride_field,
                 status_size_field, rank, timeout_ticks, Until::read, 1);
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_two_shot_pull_rows(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;

  reduce_scatter<ReduceScatterAlgorithm::rows_pull, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true,
                 TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  const int slice = (m + WORLD - 1) / WORLD;

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[TILE_M][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::rows, QUANTIZE_INT4, DTYPE, WORLD, TILE_N,
                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m,
          partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m, n,
          row, c, slice, rank);

      if constexpr (CHUNKED)
        store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
            m, n, m, n, row, c, 1);
      else
        add_attn_res_rms_norm_rows<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE>(
            row, sum, m, n, prefix_ptr, prefix_stride_m, prefix_stride_n,
            blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
            norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
            qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, num_blocks, eps,
            output_eps);
    }
    if constexpr (CHUNKED)
      add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K,
                                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                 CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
          prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
          output_ptr, output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, num_blocks, eps,
          output_eps);
  }
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_two_shot_pull_columns(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;

  reduce_scatter<ReduceScatterAlgorithm::columns_pull, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true,
                 TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  const int slice = (n / kPackElems + WORLD - 1) / WORLD * kPackElems;

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[TILE_M][kPacks][kPackElems];
      all_gather<AllGatherAlgorithm::columns, QUANTIZE_INT4, DTYPE, WORLD,
                 TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true, TAIL>(
          partials_ptr_array, partials_ptr_array_size, partials_stride_m,
          partials_stride_n, partials_size_m, partials_size_n, sum, 0, 0, m, n,
          row, c, slice, rank);

      if constexpr (CHUNKED)
        store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
            m, n, m, n, row, c, 1);
      else
        add_attn_res_rms_norm_rows<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE>(
            row, sum, m, n, prefix_ptr, prefix_stride_m, prefix_stride_n,
            blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
            norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
            qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, num_blocks, eps,
            output_eps);
    }
    if constexpr (CHUNKED)
      add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K,
                                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                 CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
          prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
          output_ptr, output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, num_blocks, eps,
          output_eps);
  }
}

template <bool QUANTIZE_INT4, typename DTYPE, int WORLD, int TILE_M, int TILE_N,
          int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          bool CHUNKED, bool TAIL>
DINLINE void all_reduce_add_attn_res_rms_norm_two_shot_push(
    const DTYPE* input_ptr, int64_t input_stride_m, int64_t input_stride_n,
    int input_size_m, int input_size_n, const DTYPE* const* input_ptr_array,
    int input_ptr_array_size, int64_t input_ptr_array_stride_m,
    int64_t input_ptr_array_stride_n, const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    char* const* partials_ptr_array, int partials_ptr_array_size,
    int64_t partials_stride_m, int64_t partials_stride_n, int partials_size_m,
    int partials_size_n, char* const* staging_ptr_array,
    int staging_ptr_array_size, int64_t staging_stride_m,
    int64_t staging_stride_n, int staging_size_m, int staging_size_n,
    uint32_t* const* signals_ptr_array, int signals_ptr_array_size,
    int64_t signals_stride_parity, int64_t signals_stride_block,
    int64_t signals_stride_writer, int signals_size_parity,
    int signals_size_block, int signals_size_writer, uint32_t* status_ptr,
    int64_t status_stride_field, int status_size_field, int rank,
    uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  constexpr int kPackElems = traits<DTYPE, PACK_BYTES>::N;
  constexpr int kPacks = (TILE_N + kPackElems * THREADS_PER_BLOCK - 1) /
                         (kPackElems * THREADS_PER_BLOCK);
  constexpr int kSpan = kPacks * kPackElems * THREADS_PER_BLOCK;
  const int m = input_size_m, n = input_size_n;

  reduce_scatter<ReduceScatterAlgorithm::columns_push, QUANTIZE_INT4, WORLD,
                 TILE_M, TILE_N, THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES, true,
                 TAIL>(
      input_ptr, input_stride_m, input_stride_n, m, n, input_ptr_array,
      input_ptr_array_size, input_ptr_array_stride_m, input_ptr_array_stride_n,
      staging_ptr_array, staging_ptr_array_size, staging_stride_m,
      staging_stride_n, staging_size_m, staging_size_n, partials_ptr_array,
      partials_ptr_array_size, partials_stride_m, partials_stride_n,
      partials_size_m, partials_size_n, signals_ptr_array,
      signals_ptr_array_size, signals_stride_parity, signals_stride_block,
      signals_stride_writer, signals_size_parity, signals_size_block,
      signals_size_writer, status_ptr, status_stride_field, status_size_field,
      rank, timeout_ticks);

  for (int row = blockIdx.x * TILE_M; row < m; row += gridDim.x * TILE_M) {
    for (int c = 0; c < n; c += kSpan) {
      DTYPE sum[TILE_M][kPacks][kPackElems];
      load<WORLD, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, QUANTIZE_INT4,
           Layout::tile, TAIL>(partials_ptr_array, partials_ptr_array_size,
                               rank, 0, 0, partials_stride_m, partials_stride_n,
                               1, partials_size_m, partials_size_n, m, n, sum,
                               row, c, 1);

      if constexpr (CHUNKED)
        store<1, TILE_M, THREADS_PER_BLOCK, WAVE_SIZE, false, Layout::tile>(
            sum, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
            m, n, m, n, row, c, 1);
      else
        add_attn_res_rms_norm_rows<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE>(
            row, sum, m, n, prefix_ptr, prefix_stride_m, prefix_stride_n,
            blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
            norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
            qk_weight_stride_n, output_norm_weight_ptr,
            output_norm_weight_stride_n, prefix_output_ptr,
            prefix_output_stride_m, prefix_output_stride_n, output_ptr,
            output_stride_m, output_stride_n, block_output_ptr,
            block_output_stride_m, block_output_stride_n, num_blocks, eps,
            output_eps);
    }
    if constexpr (CHUNKED)
      add_attn_res_rms_norm_rows<DTYPE, TILE_M, TILE_N, TILE_K,
                                 THREADS_PER_BLOCK, WAVE_SIZE, PACK_BYTES,
                                 CHUNKED>(
          row, output_ptr, output_stride_m, output_stride_n, m, n, prefix_ptr,
          prefix_stride_m, prefix_stride_n, blocks_ptr, blocks_stride_m,
          blocks_stride_r, blocks_stride_n, norm_weight_ptr,
          norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
          output_norm_weight_ptr, output_norm_weight_stride_n,
          prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
          output_ptr, output_stride_m, output_stride_n, block_output_ptr,
          block_output_stride_m, block_output_stride_n, num_blocks, eps,
          output_eps);
  }
}

template <AllReduceAddAttnResRmsNormAlgorithm ALGORITHM, bool QUANTIZE_INT4,
          typename DTYPE, int WORLD, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_PER_BLOCK, int WAVE_SIZE, int PACK_BYTES,
          int WAVES_PER_EU, bool CHUNKED, bool TAIL>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, WAVES_PER_EU)
    all_reduce_add_attn_res_rms_norm(
        const DTYPE* __restrict__ input_ptr, int64_t input_stride_m,
        int64_t input_stride_n, int input_size_m, int input_size_n,
        const DTYPE* const* __restrict__ input_ptr_array,
        int input_ptr_array_size, int64_t input_ptr_array_stride_m,
        int64_t input_ptr_array_stride_n, const DTYPE* __restrict__ prefix_ptr,
        int64_t prefix_stride_m, int64_t prefix_stride_n, int prefix_size_m,
        int prefix_size_n, const DTYPE* __restrict__ blocks_ptr,
        int64_t blocks_stride_m, int64_t blocks_stride_r,
        int64_t blocks_stride_n, int blocks_size_m, int blocks_size_r,
        int blocks_size_n, const DTYPE* __restrict__ norm_weight_ptr,
        int64_t norm_weight_stride_n, int norm_weight_size_n,
        const DTYPE* __restrict__ qk_weight_ptr, int64_t qk_weight_stride_n,
        int qk_weight_size_n, const DTYPE* __restrict__ output_norm_weight_ptr,
        int64_t output_norm_weight_stride_n, int output_norm_weight_size_n,
        DTYPE* __restrict__ prefix_output_ptr, int64_t prefix_output_stride_m,
        int64_t prefix_output_stride_n, int prefix_output_size_m,
        int prefix_output_size_n, DTYPE* output_ptr, int64_t output_stride_m,
        int64_t output_stride_n, int output_size_m, int output_size_n,
        DTYPE* __restrict__ block_output_ptr, int64_t block_output_stride_m,
        int64_t block_output_stride_n, int block_output_size_m,
        int block_output_size_n, char* const* __restrict__ partials_ptr_array,
        int partials_ptr_array_size, int64_t partials_stride_m,
        int64_t partials_stride_n, int partials_size_m, int partials_size_n,
        char* const* __restrict__ staging_ptr_array, int staging_ptr_array_size,
        int64_t staging_stride_m, int64_t staging_stride_n, int staging_size_m,
        int staging_size_n, uint32_t* const* __restrict__ signals_ptr_array,
        int signals_ptr_array_size, int64_t signals_stride_parity,
        int64_t signals_stride_block, int64_t signals_stride_writer,
        int signals_size_parity, int signals_size_block,
        int signals_size_writer, uint32_t* __restrict__ status_ptr,
        int64_t status_stride_field, int status_size_field, int rank,
        uint64_t timeout_ticks, int num_blocks, float eps, float output_eps) {
  if constexpr (ALGORITHM == AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull)
    all_reduce_add_attn_res_rms_norm_one_shot_pull<
        QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, THREADS_PER_BLOCK,
        WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, num_blocks, eps, output_eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_rows)
    all_reduce_add_attn_res_rms_norm_two_shot_pull_rows<
        QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, THREADS_PER_BLOCK,
        WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, num_blocks, eps, output_eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns)
    all_reduce_add_attn_res_rms_norm_two_shot_pull_columns<
        QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, THREADS_PER_BLOCK,
        WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, num_blocks, eps, output_eps);
  else if constexpr (ALGORITHM ==
                     AllReduceAddAttnResRmsNormAlgorithm::two_shot_push)
    all_reduce_add_attn_res_rms_norm_two_shot_push<
        QUANTIZE_INT4, DTYPE, WORLD, TILE_M, TILE_N, TILE_K, THREADS_PER_BLOCK,
        WAVE_SIZE, PACK_BYTES, CHUNKED, TAIL>(
        input_ptr, input_stride_m, input_stride_n, input_size_m, input_size_n,
        input_ptr_array, input_ptr_array_size, input_ptr_array_stride_m,
        input_ptr_array_stride_n, prefix_ptr, prefix_stride_m, prefix_stride_n,
        blocks_ptr, blocks_stride_m, blocks_stride_r, blocks_stride_n,
        norm_weight_ptr, norm_weight_stride_n, qk_weight_ptr,
        qk_weight_stride_n, output_norm_weight_ptr, output_norm_weight_stride_n,
        prefix_output_ptr, prefix_output_stride_m, prefix_output_stride_n,
        output_ptr, output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, partials_ptr_array,
        partials_ptr_array_size, partials_stride_m, partials_stride_n,
        partials_size_m, partials_size_n, staging_ptr_array,
        staging_ptr_array_size, staging_stride_m, staging_stride_n,
        staging_size_m, staging_size_n, signals_ptr_array,
        signals_ptr_array_size, signals_stride_parity, signals_stride_block,
        signals_stride_writer, signals_size_parity, signals_size_block,
        signals_size_writer, status_ptr, status_stride_field, status_size_field,
        rank, timeout_ticks, num_blocks, eps, output_eps);
}

}  // namespace hip_custom_communicator
