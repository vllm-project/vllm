#pragma once

// AttnRes on reduced rows: update the prefix, softmax over the earlier blocks,
// then RMSNorm.

#include <type_traits>
#include "common.cuh"
#include "rms_normalize.cuh"

namespace hip_custom_communicator {

template <int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, Layout LAYOUT,
          bool TAIL, typename DTYPE, int ROWS, int K, int P>
DINLINE void attn_res(
    const DTYPE (&sum)[ROWS][K][P], const DTYPE* prefix_ptr,
    int64_t prefix_stride_m, int64_t prefix_stride_n, const DTYPE* blocks_ptr,
    int64_t blocks_stride_m, int64_t blocks_stride_r, int64_t blocks_stride_n,
    const DTYPE* norm_weight_ptr, int64_t norm_weight_stride_n,
    const DTYPE* qk_weight_ptr, int64_t qk_weight_stride_n,
    const DTYPE* output_norm_weight_ptr, int64_t output_norm_weight_stride_n,
    DTYPE* prefix_output_ptr, int64_t prefix_output_stride_m,
    int64_t prefix_output_stride_n, DTYPE* output_ptr, int64_t output_stride_m,
    int64_t output_stride_n, DTYPE* block_output_ptr,
    int64_t block_output_stride_m, int64_t block_output_stride_n,
    int input_size_m, int input_size_n, int offs_m, int offs_n, int num_blocks,
    float eps, float output_eps, float inv_hidden) {
  [[maybe_unused]] const auto pack_at = [&](int k) {
    const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
    return LAYOUT == Layout::tile
               ? offs_n / P + t + k * THREADS_PER_BLOCK
               : offs_n / P +
                     ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                         kInt4GroupPacks +
                     k % kInt4GroupPacks;
  };

  DTYPE new_prefix[ROWS][K][P];
  if (prefix_ptr != nullptr) {
    DTYPE old_prefix[ROWS][K][P];
    load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1,
        input_size_m, input_size_n, input_size_m, input_size_n, old_prefix,
        offs_m, offs_n, 1);
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          new_prefix[i][k][j] =
              static_cast<DTYPE>(static_cast<float>(old_prefix[i][k][j]) +
                                 static_cast<float>(sum[i][k][j]));
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) new_prefix[i][k][j] = sum[i][k][j];
  }
  store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
      new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m,
      prefix_output_stride_n, 1, input_size_m, input_size_n, input_size_m,
      input_size_n, offs_m, offs_n, 1);
  if (block_output_ptr != nullptr)
    store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m,
        block_output_stride_n, 1, input_size_m, input_size_n, input_size_m,
        input_size_n, offs_m, offs_n, 1);

  float acc[ROWS][K][P];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        acc[i][k][j] = static_cast<float>(new_prefix[i][k][j]);
  if (num_blocks != 0) {
    DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1,
        input_size_n, 1, input_size_n, norm_weight, 0, offs_n, 1);
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, input_size_n,
        1, input_size_n, qk_weight, 0, offs_n, 1);
    float w[1][K][P];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < P; ++j)
        w[0][k][j] = static_cast<float>(norm_weight[0][k][j]) *
                     static_cast<float>(qk_weight[0][k][j]);

#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) acc[i][k][j] = float(0);

    float softmax_max[ROWS], softmax_denominator[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      softmax_max[m] = -INFINITY;
      softmax_denominator[m] = 0.0f;
    }

    const int full = num_blocks / TILE_K * TILE_K;
    const auto step = [&](int src0, auto last) {
      DTYPE v[TILE_K][ROWS][K][P];
      float sums[1][ROWS * 2 * TILE_K];
#pragma unroll
      for (int s = 0; s < TILE_K; ++s) {
        const int src = src0 + s;

        const bool block = !decltype(last)::value || src < num_blocks;
        const DTYPE* const source_ptr = block ? blocks_ptr : prefix_output_ptr;
        load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0,
            block ? blocks_stride_m : prefix_output_stride_m,
            block ? blocks_stride_n : prefix_output_stride_n,
            block ? num_blocks : 1, input_size_m, input_size_n, input_size_m,
            input_size_n, v[s], offs_m, offs_n, 1);

        float ss[ROWS], dw[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          ss[m] = 0.0f;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float ss_k = 0.0f;
#pragma unroll
            for (int j = 0; j < P; ++j)
              ss_k = fmaf(static_cast<float>(v[s][m][k][j]),
                          static_cast<float>(v[s][m][k][j]), ss_k);
            const int n = pack_at(k);
            // Packs past the row are loaded clamped and may hold data another
            // thread has not stored yet. Select, never multiply, so it cannot
            // reach the sums.
            ss[m] += n < input_size_n / P ? ss_k : 0.0f;
          }
        }
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          dw[m] = 0.0f;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float dw_k = 0.0f;
#pragma unroll
            for (int j = 0; j < P; ++j)
              dw_k = fmaf(static_cast<float>(v[s][m][k][j]), w[0][k][j], dw_k);
            const int n = pack_at(k);
            dw[m] += n < input_size_n / P ? dw_k : 0.0f;
          }
        }
#pragma unroll
        for (int m = 0; m < ROWS; ++m) {
          sums[0][(m * TILE_K + s) * 2] = ss[m];
          sums[0][(m * TILE_K + s) * 2 + 1] = dw[m];
        }
      }
      reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(sums, sums);
      float old_scale[ROWS], scale[TILE_K][ROWS];
#pragma unroll
      for (int m = 0; m < ROWS; ++m) {
        float logit[TILE_K];
#pragma unroll
        for (int s = 0; s < TILE_K; ++s)
          logit[s] =
              !decltype(last)::value || src0 + s <= num_blocks
                  ? sums[0][(m * TILE_K + s) * 2 + 1] *
                        rsqrtf(sums[0][(m * TILE_K + s) * 2] * inv_hidden + eps)
                  : -INFINITY;
        float row_scale[TILE_K];
        float new_max = softmax_max[m];
#pragma unroll
        for (int t = 0; t < TILE_K; ++t) new_max = fmaxf(new_max, logit[t]);
        const float previous_scale = __expf(softmax_max[m] - new_max);
        softmax_denominator[m] *= previous_scale;
#pragma unroll
        for (int t = 0; t < TILE_K; ++t) {
          row_scale[t] = __expf(logit[t] - new_max);
          softmax_denominator[m] += row_scale[t];
        }
        softmax_max[m] = new_max;
        old_scale[m] = previous_scale;
#pragma unroll
        for (int s = 0; s < TILE_K; ++s) scale[s][m] = row_scale[s];
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j)
            acc[i][k][j] = fmaf(static_cast<float>(v[0][i][k][j]), scale[0][i],
                                acc[i][k][j] * old_scale[i]);
      if constexpr (TILE_K > 1) {
        float ones[ROWS];
#pragma unroll
        for (int m = 0; m < ROWS; ++m) ones[m] = 1.0f;
#pragma unroll
        for (int s = 1; s < TILE_K; ++s)
#pragma unroll
          for (int i = 0; i < ROWS; ++i)
#pragma unroll
            for (int k = 0; k < K; ++k)
#pragma unroll
              for (int j = 0; j < P; ++j)
                acc[i][k][j] = fmaf(static_cast<float>(v[s][i][k][j]),
                                    scale[s][i], acc[i][k][j] * ones[i]);
      }
    };
    for (int src0 = 0; src0 < full; src0 += TILE_K)
      step(src0, std::false_type{});
    step(full, std::true_type{});
    float inv_den[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) inv_den[m] = 1.0f / softmax_denominator[m];
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) acc[i][k][j] *= inv_den[i];
  }

  DTYPE result[ROWS][K][P];
  if (output_norm_weight_ptr != nullptr) {
    DTYPE output_norm_weight[1][K][P];
    load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n, 1,
        1, input_size_n, 1, input_size_n, output_norm_weight, 0, offs_n, 1);
    float scale[ROWS];
    rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
        acc, scale, output_norm_weight, result, output_eps, inv_hidden,
        input_size_n, offs_n, NormPart::row, false, LAYOUT);
  } else {
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          result[i][k][j] = static_cast<DTYPE>(acc[i][k][j]);
  }
  store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
      result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
      input_size_m, input_size_n, input_size_m, input_size_n, offs_m, offs_n,
      1);
}

template <int TILE_K, int THREADS_PER_BLOCK, int WAVE_SIZE, Layout LAYOUT,
          bool TAIL, int ROWS, int K, int P, bool CHUNKED, typename DTYPE>
DINLINE void attn_res(
    const DTYPE* sum_ptr, int64_t sum_stride_m, int64_t sum_stride_n,
    const DTYPE* prefix_ptr, int64_t prefix_stride_m, int64_t prefix_stride_n,
    const DTYPE* blocks_ptr, int64_t blocks_stride_m, int64_t blocks_stride_r,
    int64_t blocks_stride_n, const DTYPE* norm_weight_ptr,
    int64_t norm_weight_stride_n, const DTYPE* qk_weight_ptr,
    int64_t qk_weight_stride_n, const DTYPE* output_norm_weight_ptr,
    int64_t output_norm_weight_stride_n, DTYPE* prefix_output_ptr,
    int64_t prefix_output_stride_m, int64_t prefix_output_stride_n,
    DTYPE* output_ptr, int64_t output_stride_m, int64_t output_stride_n,
    DTYPE* block_output_ptr, int64_t block_output_stride_m,
    int64_t block_output_stride_n, int input_size_m, int input_size_n,
    int offs_m, int offs_n, int num_blocks, float eps, float output_eps,
    float inv_hidden) {
  constexpr int tile_n = K * P * THREADS_PER_BLOCK;
  if constexpr (CHUNKED) {
    const int n_all = input_size_n;
    [[maybe_unused]] const auto pack_at = [&](int k, int c) {
      const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
      return LAYOUT == Layout::tile
                 ? c / P + t + k * THREADS_PER_BLOCK
                 : c / P +
                       ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                           kInt4GroupPacks +
                       k % kInt4GroupPacks;
    };

    for (int c = 0; c < n_all; c += tile_n) {
      DTYPE d[ROWS][K][P];
      load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          &sum_ptr, 1, 0, 0, 0, sum_stride_m, sum_stride_n, 1, input_size_m,
          n_all, input_size_m, n_all, d, offs_m, c, 1);
      DTYPE new_prefix[ROWS][K][P];
      if (prefix_ptr != nullptr) {
        DTYPE old_prefix[ROWS][K][P];
        load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &prefix_ptr, 1, 0, 0, 0, prefix_stride_m, prefix_stride_n, 1,
            input_size_m, n_all, input_size_m, n_all, old_prefix, offs_m, c, 1);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              new_prefix[i][k][j] =
                  static_cast<DTYPE>(static_cast<float>(old_prefix[i][k][j]) +
                                     static_cast<float>(d[i][k][j]));
      } else {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j) new_prefix[i][k][j] = d[i][k][j];
      }
      store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          new_prefix, &prefix_output_ptr, 1, 0, 0, 0, prefix_output_stride_m,
          prefix_output_stride_n, 1, input_size_m, n_all, input_size_m, n_all,
          offs_m, c, 1);
      if (block_output_ptr != nullptr)
        store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            new_prefix, &block_output_ptr, 1, 0, 0, 0, block_output_stride_m,
            block_output_stride_n, 1, input_size_m, n_all, input_size_m, n_all,
            offs_m, c, 1);
    }

    const auto source = [&](int src, int c, DTYPE(&v)[ROWS][K][P]) {
      const bool block = src < num_blocks;
      const DTYPE* const source_ptr = block ? blocks_ptr : prefix_output_ptr;
      load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          &source_ptr, 1, 0, block ? src : 0, block ? blocks_stride_r : 0,
          block ? blocks_stride_m : prefix_output_stride_m,
          block ? blocks_stride_n : prefix_output_stride_n,
          block ? num_blocks : 1, input_size_m, n_all, input_size_m, n_all, v,
          offs_m, c, 1);
    };
    extern __shared__ float wide_logits[];
    const int sources = num_blocks + 1;
    float softmax_max[ROWS], softmax_denominator[ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      softmax_max[m] = -INFINITY;
      softmax_denominator[m] = 0.0f;
    }
    if (num_blocks != 0) {
      for (int src0 = 0; src0 < sources; src0 += TILE_K) {
        float sums[1][ROWS * 2 * TILE_K];
#pragma unroll
        for (int q = 0; q < ROWS * 2 * TILE_K; ++q) sums[0][q] = 0.0f;
        for (int c = 0; c < n_all; c += tile_n) {
          DTYPE norm_weight[1][K][P], qk_weight[1][K][P];
          load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
              &norm_weight_ptr, 1, 0, 0, 0, 0, norm_weight_stride_n, 1, 1,
              n_all, 1, n_all, norm_weight, 0, c, 1);
          load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
              &qk_weight_ptr, 1, 0, 0, 0, 0, qk_weight_stride_n, 1, 1, n_all, 1,
              n_all, qk_weight, 0, c, 1);
#pragma unroll
          for (int s = 0; s < TILE_K; ++s) {
            const int src = src0 + s < sources ? src0 + s : num_blocks;
            DTYPE v[ROWS][K][P];
            source(src, c, v);
#pragma unroll
            for (int m = 0; m < ROWS; ++m)
#pragma unroll
              for (int k = 0; k < K; ++k) {
                float ss_k = 0.0f, dw_k = 0.0f;
#pragma unroll
                for (int j = 0; j < P; ++j) {
                  const float x = static_cast<float>(v[m][k][j]);
                  ss_k = fmaf(x, x, ss_k);
                  dw_k = fmaf(x,
                              static_cast<float>(norm_weight[0][k][j]) *
                                  static_cast<float>(qk_weight[0][k][j]),
                              dw_k);
                }
                const bool in = pack_at(k, c) < n_all / P;
                sums[0][(m * TILE_K + s) * 2] += in ? ss_k : 0.0f;
                sums[0][(m * TILE_K + s) * 2 + 1] += in ? dw_k : 0.0f;
              }
          }
        }
        reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(sums, sums);
#pragma unroll
        for (int m = 0; m < ROWS; ++m)
#pragma unroll
          for (int s = 0; s < TILE_K; ++s) {
            if (src0 + s >= sources) continue;
            const float logit =
                sums[0][(m * TILE_K + s) * 2 + 1] *
                rsqrtf(sums[0][(m * TILE_K + s) * 2] * inv_hidden + eps);
            if (threadIdx.x == 0) wide_logits[m * sources + src0 + s] = logit;
            const float new_max = fmaxf(softmax_max[m], logit);
            softmax_denominator[m] =
                softmax_denominator[m] * __expf(softmax_max[m] - new_max) +
                __expf(logit - new_max);
            softmax_max[m] = new_max;
          }
      }
      __syncthreads();
    }

    const auto mix = [&](int c, float (&acc)[ROWS][K][P]) {
      if (num_blocks == 0) {
        DTYPE v[ROWS][K][P];
        source(num_blocks, c, v);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              acc[i][k][j] = static_cast<float>(v[i][k][j]);
        return;
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j) acc[i][k][j] = 0.0f;
      for (int src = 0; src < sources; ++src) {
        DTYPE v[ROWS][K][P];
        source(src, c, v);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
          const float p =
              __expf(wide_logits[i * sources + src] - softmax_max[i]) /
              softmax_denominator[i];
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              acc[i][k][j] =
                  fmaf(static_cast<float>(v[i][k][j]), p, acc[i][k][j]);
        }
      }
    };

    float scale[ROWS];
    if (output_norm_weight_ptr != nullptr) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m) scale[m] = 0.0f;
      for (int c = 0; c < n_all; c += tile_n) {
        float acc[ROWS][K][P];
        mix(c, acc);
        DTYPE wide_w[1][K][P], wide_out[ROWS][K][P];
        rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
            acc, scale, wide_w, wide_out, output_eps, inv_hidden, n_all, c,
            NormPart::wide, false, LAYOUT);
      }
    }
    for (int c = 0; c < n_all; c += tile_n) {
      float acc[ROWS][K][P];
      mix(c, acc);
      DTYPE result[ROWS][K][P];
      if (output_norm_weight_ptr != nullptr) {
        DTYPE output_norm_weight[1][K][P];
        load<1, 1, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
            &output_norm_weight_ptr, 1, 0, 0, 0, 0, output_norm_weight_stride_n,
            1, 1, n_all, 1, n_all, output_norm_weight, 0, c, 1);
        rms_normalize<THREADS_PER_BLOCK, WAVE_SIZE>(
            acc, scale, output_norm_weight, result, output_eps, inv_hidden,
            n_all, c, NormPart::apply, false, LAYOUT);
      } else {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int j = 0; j < P; ++j)
              result[i][k][j] = static_cast<DTYPE>(acc[i][k][j]);
      }
      store<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
          result, &output_ptr, 1, 0, 0, 0, output_stride_m, output_stride_n, 1,
          input_size_m, n_all, input_size_m, n_all, offs_m, c, 1);
    }
    __syncthreads();

  } else {
    DTYPE sum[ROWS][K][P];
    load<1, ROWS, THREADS_PER_BLOCK, WAVE_SIZE, false, LAYOUT, TAIL>(
        &sum_ptr, 1, 0, 0, 0, sum_stride_m, sum_stride_n, 1, input_size_m,
        input_size_n, input_size_m, input_size_n, sum, offs_m, offs_n, 1);
    attn_res<TILE_K, THREADS_PER_BLOCK, WAVE_SIZE, LAYOUT, TAIL>(
        sum, prefix_ptr, prefix_stride_m, prefix_stride_n, blocks_ptr,
        blocks_stride_m, blocks_stride_r, blocks_stride_n, norm_weight_ptr,
        norm_weight_stride_n, qk_weight_ptr, qk_weight_stride_n,
        output_norm_weight_ptr, output_norm_weight_stride_n, prefix_output_ptr,
        prefix_output_stride_m, prefix_output_stride_n, output_ptr,
        output_stride_m, output_stride_n, block_output_ptr,
        block_output_stride_m, block_output_stride_n, input_size_m,
        input_size_n, offs_m, offs_n, num_blocks, eps, output_eps, inv_hidden);
  }
}

}  // namespace hip_custom_communicator
