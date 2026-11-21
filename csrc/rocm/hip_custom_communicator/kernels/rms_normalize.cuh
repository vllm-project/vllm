#pragma once

// RMSNorm over a block's rows.

#include "common.cuh"

namespace hip_custom_communicator {

enum class NormPart : int { scale = 0, apply = 1, row = 2, wide = 3 };

template <int THREADS_PER_BLOCK, int WAVE_SIZE, typename W, typename T,
          int ROWS, int K, int P>
DINLINE void rms_normalize(const float (&input)[ROWS][K][P],
                           float (&scale)[ROWS], const W (&weight)[1][K][P],
                           T (&output)[ROWS][K][P], float eps, float inv_hidden,
                           int size_n, int offs_n, NormPart part,
                           bool round_scaled, Layout layout) {
  if (part != NormPart::apply) {
    float ss[1][ROWS];
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      ss[0][m] = part == NormPart::wide ? scale[m] : 0.0f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float ss_k = 0.0f;
#pragma unroll
        for (int j = 0; j < P; ++j)
          ss_k = fmaf(input[m][k][j], input[m][k][j], ss_k);
        const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
        const int n =
            layout == Layout::tile
                ? offs_n / P + t + k * THREADS_PER_BLOCK
                : offs_n / P +
                      ((k / kInt4GroupPacks) * THREADS_PER_BLOCK + t) *
                          kInt4GroupPacks +
                      k % kInt4GroupPacks;
        ss[0][m] += n < size_n / P ? ss_k : 0.0f;
      }
    }
    if (part == NormPart::wide && offs_n + K * P * THREADS_PER_BLOCK < size_n) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m) scale[m] = ss[0][m];
      return;
    }
    reduce<1, THREADS_PER_BLOCK, WAVE_SIZE, ReduceAxis::all>(ss, ss);
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
      scale[m] = rsqrtf(ss[0][m] * inv_hidden + eps);
  }
  if (part != NormPart::scale && part != NormPart::wide) {
    float normed[ROWS][K][P];
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j) normed[m][k][j] = input[m][k][j] * scale[m];
    if (round_scaled) {
#pragma unroll
      for (int m = 0; m < ROWS; ++m)
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
          for (int j = 0; j < P; ++j)
            normed[m][k][j] =
                static_cast<float>(static_cast<W>(normed[m][k][j]));
    }
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          normed[m][k][j] *= static_cast<float>(weight[0][k][j]);
#pragma unroll
    for (int m = 0; m < ROWS; ++m)
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < P; ++j)
          output[m][k][j] =
              round_scaled ? static_cast<T>(static_cast<W>(normed[m][k][j]))
                           : static_cast<T>(normed[m][k][j]);
  }
}

}  // namespace hip_custom_communicator
