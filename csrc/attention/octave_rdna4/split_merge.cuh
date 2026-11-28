// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace octave_rdna4 {

constexpr int kMergeWave = 32;
constexpr int kParallelMergeMinSplits = 16;

inline int merge_threads(int64_t rows, int splits, bool wmma_prefill = false) {
  // Wide prefill grids already occupy the GPU with a wave per output row.
  if (wmma_prefill && rows > 256 && splits <= 32) return kMergeWave;
  if (splits >= 32 && rows <= 256) return 256;
  if (splits >= kParallelMergeMinSplits && rows <= 512) return 128;
  if (splits >= 32 && rows <= 2048) return 64;
  return kMergeWave;
}

template <int Head>
inline int merge_shared_bytes(int threads, int splits) {
  return threads == kMergeWave
             ? 0
             : sizeof(float) * (splits + (threads / kMergeWave) * (Head + 3));
}

__device__ __forceinline__ float merge_wave_max(float value) {
#pragma unroll
  for (int delta = kMergeWave / 2; delta > 0; delta /= 2)
    value = fmaxf(value, __shfl_xor(value, delta, kMergeWave));
  return value;
}

__device__ __forceinline__ bool valid_partial(float marker, float maximum,
                                              float mass) {
  return __builtin_isfinite(marker) && __builtin_isfinite(mass) &&
         mass >= 0.f && (mass == 0.f || __builtin_isfinite(maximum));
}

// One wave is sufficient for a few splits or many independent output rows.
// Reuse the weight for every dimension owned by the lane.
template <int Head>
__device__ __forceinline__ void merge_wave(float* out, const float* src,
                                           int splits) {
  constexpr int Partial = Head + 2;
  constexpr int Values = (Head + kMergeWave - 1) / kMergeWave;
  const int lane = threadIdx.x;
  float maximum = -INFINITY;
  for (int split = lane; split < splits; split += kMergeWave)
    maximum = fmaxf(maximum, src[split * Partial + Head]);
  maximum = merge_wave_max(maximum);

  float mass = 0.f, accum[Values] = {};
  for (int split = 0; split < splits; ++split) {
    const float* p = src + split * Partial;
    const float l = p[Head + 1];
    if (!(l > 0.f)) continue;
    const float weight = __expf(p[Head] - maximum);
    mass = fmaf(weight, l, mass);
#pragma unroll
    for (int i = 0; i < Values; ++i) {
      const int d = lane + i * kMergeWave;
      if (d < Head) accum[i] = fmaf(weight, p[d], accum[i]);
    }
  }
  const float inv = mass > 0.f ? 1.f / mass : 0.f;
#pragma unroll
  for (int i = 0; i < Values; ++i) {
    const int d = lane + i * kMergeWave;
    if (d < Head) out[d] = mass > 0.f ? accum[i] * inv : 0.f;
  }
}

template <int Head, bool Check, bool PositiveMass = false>
__device__ __forceinline__ bool merge_parallel(float* out, const float* src,
                                               int splits, float* shared) {
  constexpr int Partial = Head + 2;
  const int tid = threadIdx.x, lane = tid % kMergeWave;
  const int wave = tid / kMergeWave, waves = blockDim.x / kMergeWave;
  float* weights = shared;
  float* vectors = weights + splits;
  float* maxima = vectors + waves * Head;
  float* masses = maxima + waves;
  float* valid_flags = masses + waves;

  float maximum = -INFINITY;
  bool valid = true;
  for (int split = tid; split < splits; split += blockDim.x) {
    const float* p = src + split * Partial;
    const float m = p[Head], l = p[Head + 1];
    maximum = fmaxf(maximum, m);
    if constexpr (Check) valid &= valid_partial(p[0], m, l);
  }
  maximum = merge_wave_max(maximum);
  if constexpr (Check) valid = __all(valid);
  if (lane == 0) {
    maxima[wave] = maximum;
    if constexpr (Check) valid_flags[wave] = valid ? 1.f : 0.f;
  }
  __syncthreads();

  maximum = -INFINITY;
  for (int w = 0; w < waves; ++w) {
    maximum = fmaxf(maximum, maxima[w]);
    if constexpr (Check) valid &= valid_flags[w] != 0.f;
  }
  if constexpr (Check)
    if (!valid) return false;

  float mass = 0.f;
  for (int split = tid; split < splits; split += blockDim.x) {
    const float* p = src + split * Partial;
    const float l = p[Head + 1];
    const bool nonempty = PositiveMass ? l > 0.f : l != 0.f;
    float weight = nonempty ? __expf(p[Head] - maximum) : -1.f;
    weights[split] = weight;
    if (nonempty) mass = fmaf(weight, l, mass);
  }
#pragma unroll
  for (int delta = kMergeWave / 2; delta > 0; delta /= 2)
    mass += __shfl_xor(mass, delta, kMergeWave);
  if (lane == 0) masses[wave] = mass;
  __syncthreads();

  float accum[(Head + kMergeWave - 1) / kMergeWave] = {};
  for (int split = wave; split < splits; split += waves) {
    const float weight = weights[split];
    if (weight < 0.f) continue;
    const float* p = src + split * Partial;
#pragma unroll
    for (int i = 0; i < (Head + kMergeWave - 1) / kMergeWave; ++i) {
      const int d = lane + i * kMergeWave;
      if (d < Head) accum[i] = fmaf(weight, p[d], accum[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < (Head + kMergeWave - 1) / kMergeWave; ++i) {
    const int d = lane + i * kMergeWave;
    if (d < Head) vectors[wave * Head + d] = accum[i];
  }
  __syncthreads();

  mass = 0.f;
  for (int w = 0; w < waves; ++w) mass += masses[w];
  const float inv = mass > 0.f ? 1.f / mass : 0.f;
  float largest = 0.f;
  for (int d = tid; d < Head; d += blockDim.x) {
    float value = 0.f;
    for (int w = 0; w < waves; ++w) value += vectors[w * Head + d];
    if constexpr (PositiveMass)
      value = mass > 0.f ? value * inv : 0.f;
    else
      value *= inv;
    out[d] = value;
    if constexpr (Check) {
      valid &= __builtin_isfinite(value);
      largest = fmaxf(largest, fabsf(value));
    }
  }
  if constexpr (Check) {
    largest = merge_wave_max(largest);
    valid = __all(valid);
    if (lane == 0) {
      maxima[wave] = largest;
      valid_flags[wave] = valid ? 1.f : 0.f;
    }
    __syncthreads();
    largest = 0.f;
    for (int w = 0; w < waves; ++w) {
      largest = fmaxf(largest, maxima[w]);
      valid &= valid_flags[w] != 0.f;
    }
    return valid && !(mass > 0.f && largest < 1.e-4f);
  }
  return true;
}

}  // namespace octave_rdna4
