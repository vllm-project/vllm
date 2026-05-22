// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "cpu_arch_macros.h"
#include "cpu_types.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr float kPadSentinel = -1e30f;      // logits below this are PAD
constexpr float kBoundaryTol = 1e-5f;       // tie tolerance for top-p cut
constexpr int kMeanStdSampleN = 8192;       // mean/std sample cap
constexpr int kBsearchMaxIters = 32;        // binary-search iteration cap
constexpr int kTernarySearchMaxIters = 18;  // matches Triton
constexpr float kKDuplicateTol = 1e-9f;     // tie tolerance for top-k (Triton)
}  // namespace

// Matches _NORMAL_CDF_TO_SIGMA_TABLE in topk_topp_triton.py (200 entries).
static const float NORMAL_CDF_TO_SIGMA_TABLE[200] = {
    3.656f,  3.650f,  3.650f, 3.650f, 3.626f, 3.626f, 3.626f, 3.514f, 3.514f,
    3.503f,  3.503f,  3.434f, 3.434f, 3.428f, 3.428f, 3.387f, 3.380f, 3.380f,
    3.376f,  3.373f,  3.373f, 3.356f, 3.354f, 3.354f, 3.291f, 3.249f, 3.234f,
    3.214f,  3.198f,  3.198f, 3.185f, 3.177f, 3.177f, 3.165f, 3.164f, 3.161f,
    3.138f,  3.120f,  3.115f, 3.113f, 3.093f, 3.066f, 3.054f, 3.043f, 3.037f,
    3.023f,  2.993f,  2.991f, 2.976f, 2.970f, 2.952f, 2.946f, 2.932f, 2.908f,
    2.902f,  2.895f,  2.886f, 2.874f, 2.861f, 2.844f, 2.836f, 2.810f, 2.801f,
    2.790f,  2.784f,  2.779f, 2.767f, 2.757f, 2.745f, 2.733f, 2.723f, 2.716f,
    2.693f,  2.678f,  2.671f, 2.656f, 2.649f, 2.629f, 2.611f, 2.595f, 2.592f,
    2.585f,  2.574f,  2.550f, 2.543f, 2.534f, 2.521f, 2.518f, 2.497f, 2.485f,
    2.468f,  2.450f,  2.441f, 2.430f, 2.412f, 2.402f, 2.389f, 2.383f, 2.377f,
    2.364f,  2.349f,  2.338f, 2.332f, 2.319f, 2.310f, 2.301f, 2.282f, 2.274f,
    2.266f,  2.250f,  2.242f, 2.236f, 2.226f, 2.215f, 2.207f, 2.196f, 2.179f,
    2.171f,  2.162f,  2.147f, 2.135f, 2.121f, 2.109f, 2.095f, 2.085f, 2.073f,
    2.063f,  2.045f,  2.030f, 2.016f, 2.003f, 1.992f, 1.983f, 1.972f, 1.960f,
    1.949f,  1.940f,  1.928f, 1.912f, 1.897f, 1.881f, 1.869f, 1.854f, 1.838f,
    1.824f,  1.807f,  1.792f, 1.779f, 1.764f, 1.751f, 1.739f, 1.726f, 1.711f,
    1.697f,  1.685f,  1.668f, 1.652f, 1.636f, 1.622f, 1.603f, 1.585f, 1.568f,
    1.551f,  1.534f,  1.513f, 1.499f, 1.480f, 1.464f, 1.441f, 1.422f, 1.394f,
    1.373f,  1.347f,  1.320f, 1.296f, 1.270f, 1.246f, 1.219f, 1.190f, 1.163f,
    1.135f,  1.104f,  1.073f, 1.041f, 1.006f, 0.969f, 0.931f, 0.894f, 0.851f,
    0.806f,  0.757f,  0.702f, 0.643f, 0.574f, 0.498f, 0.405f, 0.288f, 0.134f,
    -0.110f, -3.813f,
};

// Matches _PERCENTILE_TO_STD_TABLE in topk_topp_triton.py (200 entries).
static const float PERCENTILE_TO_STD_TABLE[200] = {
    2.576f,  2.319f,  2.178f,  2.064f,  1.968f,  1.892f,  1.819f,  1.757f,
    1.708f,  1.659f,  1.616f,  1.568f,  1.526f,  1.492f,  1.456f,  1.420f,
    1.382f,  1.342f,  1.309f,  1.280f,  1.249f,  1.221f,  1.193f,  1.169f,
    1.145f,  1.121f,  1.095f,  1.073f,  1.050f,  1.030f,  1.008f,  0.987f,
    0.966f,  0.945f,  0.926f,  0.910f,  0.891f,  0.871f,  0.854f,  0.837f,
    0.819f,  0.803f,  0.784f,  0.767f,  0.753f,  0.734f,  0.719f,  0.702f,
    0.690f,  0.675f,  0.658f,  0.640f,  0.625f,  0.609f,  0.595f,  0.578f,
    0.564f,  0.550f,  0.537f,  0.521f,  0.509f,  0.495f,  0.481f,  0.466f,
    0.453f,  0.439f,  0.424f,  0.410f,  0.397f,  0.383f,  0.370f,  0.356f,
    0.343f,  0.330f,  0.316f,  0.302f,  0.289f,  0.274f,  0.261f,  0.247f,
    0.235f,  0.223f,  0.209f,  0.196f,  0.184f,  0.172f,  0.159f,  0.149f,
    0.137f,  0.124f,  0.112f,  0.100f,  0.086f,  0.074f,  0.062f,  0.050f,
    0.035f,  0.023f,  0.009f,  -0.003f, -0.015f, -0.027f, -0.039f, -0.052f,
    -0.063f, -0.074f, -0.085f, -0.097f, -0.109f, -0.122f, -0.134f, -0.147f,
    -0.158f, -0.171f, -0.184f, -0.196f, -0.210f, -0.223f, -0.235f, -0.248f,
    -0.261f, -0.275f, -0.289f, -0.302f, -0.317f, -0.328f, -0.341f, -0.353f,
    -0.368f, -0.382f, -0.396f, -0.410f, -0.426f, -0.439f, -0.452f, -0.465f,
    -0.480f, -0.493f, -0.507f, -0.521f, -0.537f, -0.551f, -0.568f, -0.582f,
    -0.597f, -0.614f, -0.628f, -0.643f, -0.658f, -0.673f, -0.691f, -0.706f,
    -0.721f, -0.738f, -0.754f, -0.769f, -0.789f, -0.808f, -0.824f, -0.838f,
    -0.857f, -0.877f, -0.893f, -0.912f, -0.929f, -0.947f, -0.965f, -0.983f,
    -1.003f, -1.027f, -1.050f, -1.070f, -1.092f, -1.117f, -1.139f, -1.162f,
    -1.189f, -1.216f, -1.241f, -1.272f, -1.300f, -1.330f, -1.367f, -1.404f,
    -1.441f, -1.485f, -1.523f, -1.564f, -1.607f, -1.658f, -1.710f, -1.778f,
    -1.832f, -1.901f, -1.978f, -2.068f, -2.174f, -2.325f, -2.577f, -3.813f,
};

namespace {

// FP64 accumulator for softmax sum over FP32Vec16 chunks.
// FP32 accumulation over 128K elements loses ~3 ULP; FP64 avoids this.
#if defined(__AVX512F__)
struct Fp64Acc16 {
  __m512d acc0 = _mm512_setzero_pd();
  __m512d acc1 = _mm512_setzero_pd();
  inline void add(const vec_op::FP32Vec16& v) {
    __m256 lo8 = _mm512_castps512_ps256(v.reg);
    __m256 hi8 = _mm512_extractf32x8_ps(v.reg, 1);
    acc0 = _mm512_add_pd(acc0, _mm512_cvtps_pd(lo8));
    acc1 = _mm512_add_pd(acc1, _mm512_cvtps_pd(hi8));
  }
  inline double reduce() const {
    return _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
  }
};
#elif defined(__AVX2__)
struct Fp64Acc16 {
  __m256d acc0 = _mm256_setzero_pd();
  __m256d acc1 = _mm256_setzero_pd();
  __m256d acc2 = _mm256_setzero_pd();
  __m256d acc3 = _mm256_setzero_pd();
  inline void add(const vec_op::FP32Vec16& v) {
    __m128 e_ll = _mm256_extractf128_ps(v.reg_low, 0);
    __m128 e_lh = _mm256_extractf128_ps(v.reg_low, 1);
    __m128 e_hl = _mm256_extractf128_ps(v.reg_high, 0);
    __m128 e_hh = _mm256_extractf128_ps(v.reg_high, 1);
    acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(e_ll));
    acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(e_lh));
    acc2 = _mm256_add_pd(acc2, _mm256_cvtps_pd(e_hl));
    acc3 = _mm256_add_pd(acc3, _mm256_cvtps_pd(e_hh));
  }
  inline double reduce() const {
    __m256d sum_a =
        _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    __m128d hi128 = _mm256_extractf128_pd(sum_a, 1);
    __m128d lo128 = _mm256_castpd256_pd128(sum_a);
    __m128d s2 = _mm_add_pd(lo128, hi128);
    return _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
  }
};
#endif

__attribute__((always_inline)) static inline double sum_gt_to_double(
    const float* buf, int n, float threshold) {
  double s = 0.0;
  int j = 0;
#if defined(__AVX512F__)
  __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
  const __m512 thr16 = _mm512_set1_ps(threshold);
  for (; j + 16 <= n; j += 16) {
    __m512 b16 = _mm512_loadu_ps(buf + j);
    __mmask16 mask = _mm512_cmp_ps_mask(b16, thr16, _CMP_GT_OQ);
    __m256 lo8 = _mm512_castps512_ps256(b16);
    __m256 hi8 = _mm512_extractf32x8_ps(b16, 1);
    acc0 = _mm512_add_pd(acc0,
                         _mm512_maskz_cvtps_pd((__mmask8)(mask & 0xFF), lo8));
    acc1 =
        _mm512_add_pd(acc1, _mm512_maskz_cvtps_pd((__mmask8)(mask >> 8), hi8));
  }
  s = _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
#elif defined(__AVX2__)
  // and_ps zeros excluded lanes; blendv with -inf would corrupt the sum.
  const __m256 thr = _mm256_set1_ps(threshold);
  __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
  __m256d acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();
  for (; j + 16 <= n; j += 16) {
    __m256 b_lo = _mm256_loadu_ps(buf + j);
    __m256 b_hi = _mm256_loadu_ps(buf + j + 8);
    __m256 masked_lo =
        _mm256_and_ps(b_lo, _mm256_cmp_ps(b_lo, thr, _CMP_GT_OQ));
    __m256 masked_hi =
        _mm256_and_ps(b_hi, _mm256_cmp_ps(b_hi, thr, _CMP_GT_OQ));
    acc0 = _mm256_add_pd(acc0,
                         _mm256_cvtps_pd(_mm256_extractf128_ps(masked_lo, 0)));
    acc1 = _mm256_add_pd(acc1,
                         _mm256_cvtps_pd(_mm256_extractf128_ps(masked_lo, 1)));
    acc2 = _mm256_add_pd(acc2,
                         _mm256_cvtps_pd(_mm256_extractf128_ps(masked_hi, 0)));
    acc3 = _mm256_add_pd(acc3,
                         _mm256_cvtps_pd(_mm256_extractf128_ps(masked_hi, 1)));
  }
  __m256d sum_a =
      _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
  __m128d hi128 = _mm256_extractf128_pd(sum_a, 1);
  __m128d lo128 = _mm256_castpd256_pd128(sum_a);
  __m128d s2 = _mm_add_pd(lo128, hi128);
  s = _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
#endif
  for (; j < n; ++j)
    if (buf[j] > threshold) s += (double)buf[j];
  return s;
}

__attribute__((always_inline)) static inline int count_within_tol(
    const float* row, int V, float center, float tol) {
  int n = 0;
  int i = 0;
#if defined(__AVX512F__)
  const vec_op::FP32Vec16 c16(center);
  const vec_op::FP32Vec16 t16(tol);
  for (; i + 16 <= V; i += 16) {
    vec_op::FP32Vec16 absdiff = (vec_op::FP32Vec16(row + i) - c16).abs();
    __mmask16 mask = _mm512_cmp_ps_mask(absdiff.reg, t16.reg, _CMP_LT_OQ);
    n += __builtin_popcount((unsigned)mask);
  }
#elif defined(__AVX2__)
  const vec_op::FP32Vec16 c16(center);
  const vec_op::FP32Vec16 t16(tol);
  for (; i + 16 <= V; i += 16) {
    vec_op::FP32Vec16 absdiff = (vec_op::FP32Vec16(row + i) - c16).abs();
    __m256 clo = _mm256_cmp_ps(absdiff.reg_low, t16.reg_low, _CMP_LT_OQ);
    __m256 chi = _mm256_cmp_ps(absdiff.reg_high, t16.reg_high, _CMP_LT_OQ);
    n += __builtin_popcount((unsigned)_mm256_movemask_ps(clo));
    n += __builtin_popcount((unsigned)_mm256_movemask_ps(chi));
  }
#endif
  for (; i < V; ++i)
    if (fabsf(row[i] - center) < tol) ++n;
  return n;
}

__attribute__((always_inline)) static inline void mask_write_below(
    float* row, int V, float threshold, float fill) {
  int i = 0;
#if defined(__AVX512F__)
  const vec_op::FP32Vec16 thr16(threshold);
  const vec_op::FP32Vec16 f16(fill);
  for (; i + 16 <= V; i += 16) {
    vec_op::FP32Vec16 v16(row + i);
    __mmask16 keep = _mm512_cmp_ps_mask(v16.reg, thr16.reg, _CMP_GT_OQ);
    vec_op::FP32Vec16(_mm512_mask_blend_ps(keep, f16.reg, v16.reg))
        .save(row + i);
  }
#elif defined(__AVX2__)
  const vec_op::FP32Vec16 thr16(threshold);
  const vec_op::FP32Vec16 f16(fill);
  for (; i + 16 <= V; i += 16) {
    vec_op::FP32Vec16 v16(row + i);
    __m256 klo = _mm256_cmp_ps(v16.reg_low, thr16.reg_low, _CMP_GT_OQ);
    __m256 khi = _mm256_cmp_ps(v16.reg_high, thr16.reg_high, _CMP_GT_OQ);
    vec_op::FP32Vec16(_mm256_blendv_ps(f16.reg_low, v16.reg_low, klo),
                      _mm256_blendv_ps(f16.reg_high, v16.reg_high, khi))
        .save(row + i);
  }
#endif
  for (; i < V; ++i)
    if (!(row[i] > threshold)) row[i] = fill;
}

// Vectorised max/min/finite-count over row[0..V). PAD lanes (<= pad_sentinel)
// are blended to +inf so they don't corrupt reduce_min. Returns SIMD tail
// start in *tail_i_out; caller handles remaining elements and clamps min<=max.
__attribute__((always_inline)) static inline void vec_max_min_with_pad_blend(
    const float* row, int V, float pad_sentinel, float* max_out, float* min_out,
    int* n_finite_out, int* tail_i_out) {
  int i = 0;
  float max_l = -1e38f, min_l = 1e38f;
  int n_finite = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
  {
    const float pos_inf_val = std::numeric_limits<float>::infinity();
    vec_op::FP32Vec16 maxv(row[0]);
    vec_op::FP32Vec16 minv(pos_inf_val);
    const vec_op::FP32Vec16 sentinel16(pad_sentinel);
    const vec_op::FP32Vec16 pos_inf16(pos_inf_val);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      maxv = maxv.max(v16);
  #if defined(__AVX512F__)
      __mmask16 is_pad =
          _mm512_cmp_ps_mask(v16.reg, sentinel16.reg, _CMP_LE_OS);
      vec_op::FP32Vec16 safe(
          _mm512_mask_blend_ps(is_pad, v16.reg, pos_inf16.reg));
      __mmask16 fin = _mm512_cmp_ps_mask(v16.reg, sentinel16.reg, _CMP_GT_OQ);
      n_finite += __builtin_popcount((unsigned)fin);
  #else
      __m256 lt_lo = _mm256_cmp_ps(v16.reg_low, sentinel16.reg_low, _CMP_LE_OS);
      __m256 lt_hi =
          _mm256_cmp_ps(v16.reg_high, sentinel16.reg_high, _CMP_LE_OS);
      vec_op::FP32Vec16 safe(
          _mm256_blendv_ps(v16.reg_low, pos_inf16.reg_low, lt_lo),
          _mm256_blendv_ps(v16.reg_high, pos_inf16.reg_high, lt_hi));
      __m256 fin_lo =
          _mm256_cmp_ps(v16.reg_low, sentinel16.reg_low, _CMP_GT_OQ);
      __m256 fin_hi =
          _mm256_cmp_ps(v16.reg_high, sentinel16.reg_high, _CMP_GT_OQ);
      n_finite += __builtin_popcount((unsigned)_mm256_movemask_ps(fin_lo));
      n_finite += __builtin_popcount((unsigned)_mm256_movemask_ps(fin_hi));
  #endif
      minv = minv.min(safe);
    }
    max_l = maxv.reduce_max();
    min_l = minv.reduce_min();
  }
#endif
  *max_out = max_l;
  *min_out = min_l;
  *n_finite_out = n_finite;
  *tail_i_out = i;
}

// Zero exp lanes excluded by top-k without a separate row pass.
__attribute__((always_inline)) static inline vec_op::FP32Vec16
apply_topk_mask_zero(const vec_op::FP32Vec16& e16, const vec_op::FP32Vec16& v16,
                     float tk_pivot) {
#if defined(__AVX512F__)
  const __m512 vp = _mm512_set1_ps(tk_pivot);
  __mmask16 keep = _mm512_cmp_ps_mask(v16.reg, vp, _CMP_GT_OQ);
  return vec_op::FP32Vec16(_mm512_maskz_mov_ps(keep, e16.reg));
#elif defined(__AVX2__)
  const __m256 vp = _mm256_set1_ps(tk_pivot);
  const __m256 zero = _mm256_setzero_ps();
  __m256 keep_lo = _mm256_cmp_ps(v16.reg_low, vp, _CMP_GT_OQ);
  __m256 keep_hi = _mm256_cmp_ps(v16.reg_high, vp, _CMP_GT_OQ);
  return vec_op::FP32Vec16(_mm256_blendv_ps(zero, e16.reg_low, keep_lo),
                           _mm256_blendv_ps(zero, e16.reg_high, keep_hi));
#else
  (void)tk_pivot;
  return e16;
#endif
}

__attribute__((always_inline)) static inline int gather_outliers(
    const float* buf, int n, float threshold, float* out, double* sum_out) {
  int m = 0;
  double s = 0.0;
  for (int j = 0; j < n; ++j) {
    float v = buf[j];
    if (v > threshold) {
      out[m++] = v;
      if (sum_out) s += (double)v;
    }
  }
  if (sum_out) *sum_out = s;
  return m;
}

static inline void compute_mean_std(const float* row, int V, float* avg_out,
                                    float* std_out) {
  float sum_s = 0.f, sum_sq = 0.f;
  int nf = 0;
  for (int i = 0; i < V && nf < kMeanStdSampleN; ++i) {
    float v = row[i];
    if (v > kPadSentinel) {
      sum_s += v;
      sum_sq += v * v;
      ++nf;
    }
  }
  float avg = nf > 0 ? sum_s / nf : 0.f;
  // E[X²] - (E[X])² is numerically unstable for large logit magnitudes, but
  // this is only a heuristic pivot initializer; the ternary/binary search
  // converges correctly regardless of pivot quality.
  float var = nf > 0 ? sum_sq / nf - avg * avg : 1.f;
  *avg_out = avg;
  *std_out = sqrtf(var > 0.f ? var : 0.f);
}

// Apply boundary-tie resolution in row order. Keeps all v > pivot, plus up to
// n_keep_at_boundary values within tol of dup_value. Rare path: called only
// when binary/ternary search lands exactly on a duplicate boundary value.
static inline void apply_boundary_tie_loop(float* row, int V, float pivot,
                                           float dup_value, float tol,
                                           int n_keep_at_boundary) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  int kept_boundary = 0;
  for (int i = 0; i < V; ++i) {
    float v = row[i];
    bool keep = v > pivot;
    if (!keep && fabsf(v - dup_value) < tol &&
        kept_boundary < n_keep_at_boundary) {
      keep = true;
      ++kept_boundary;
    }
    if (!keep) row[i] = neg_inf;
  }
}

}  // namespace

static float binary_search_buffer(const float* buf, int n, float p_val,
                                  float lo, float hi, double* sum_above_out) {
  double s_hi = 0.0;
  for (int iter = 0; iter < kBsearchMaxIters; ++iter) {
    float mid = lo + (hi - lo) * 0.5f;
    if (mid == lo || mid == hi) break;
    double s = sum_gt_to_double(buf, n, mid);
    if (s >= (double)p_val) {
      lo = mid;
    } else {
      hi = mid;
      s_hi = s;
    }
  }
  *sum_above_out = s_hi;
  return hi;
}

static void scan_pivot_stats(const float* buf, int n, float pivot, float tol,
                             int* num_above, float* min_above,
                             int* num_at_min) {
  float ma = std::numeric_limits<float>::infinity();
  int na = 0;
  int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
  {
    const float pos_inf_val = std::numeric_limits<float>::infinity();
    const vec_op::FP32Vec16 vpivot16(pivot);
    const vec_op::FP32Vec16 pos_inf16(pos_inf_val);
    vec_op::FP32Vec16 vmin16(pos_inf_val);
    for (; i + 16 <= n; i += 16) {
      vec_op::FP32Vec16 b16(buf + i);
  #if defined(__AVX512F__)
      __mmask16 above = _mm512_cmp_ps_mask(b16.reg, vpivot16.reg, _CMP_GT_OQ);
      na += __builtin_popcount((unsigned)above);
      vmin16 = vmin16.min(vec_op::FP32Vec16(
          _mm512_mask_blend_ps(above, pos_inf16.reg, b16.reg)));
  #else
      __m256 above_lo =
          _mm256_cmp_ps(b16.reg_low, vpivot16.reg_low, _CMP_GT_OQ);
      __m256 above_hi =
          _mm256_cmp_ps(b16.reg_high, vpivot16.reg_high, _CMP_GT_OQ);
      na += __builtin_popcount((unsigned)_mm256_movemask_ps(above_lo));
      na += __builtin_popcount((unsigned)_mm256_movemask_ps(above_hi));
      vmin16 = vmin16.min(vec_op::FP32Vec16(
          _mm256_blendv_ps(pos_inf16.reg_low, b16.reg_low, above_lo),
          _mm256_blendv_ps(pos_inf16.reg_high, b16.reg_high, above_hi)));
  #endif
    }
    ma = vmin16.reduce_min();
  }
#endif
  for (; i < n; ++i) {
    if (buf[i] > pivot) {
      ++na;
      if (buf[i] < ma) ma = buf[i];
    }
  }
  *num_above = na;
  *min_above = ma;
  // count_within_tol runs over the full buf; elements below pivot are all
  // < min_above, so only ULP-equal values fall within kKDuplicateTol and
  // are legitimately counted as boundary duplicates.
  *num_at_min = count_within_tol(buf, n, ma, tol);
}

static float ternary_search_topk(const float* buf, int n, int k, float lo,
                                 float hi, float* dup_value_out,
                                 int* num_dup_out, int* num_above_out) {
  float final_pivot = (lo + hi) * 0.5f;
  float dup_value = std::numeric_limits<float>::infinity();
  int num_dup = 0, num_above = 0;
  for (int iter = 0; iter < kTernarySearchMaxIters; ++iter) {
    float p0 = lo + (hi - lo) * (1.0f / 3.0f);
    float p1 = lo + (hi - lo) * (2.0f / 3.0f);
    int na0, na1, nd0, nd1;
    float mn0, mn1;
    scan_pivot_stats(buf, n, p0, kKDuplicateTol, &na0, &mn0, &nd0);
    scan_pivot_stats(buf, n, p1, kKDuplicateTol, &na1, &mn1, &nd1);

    bool found0 = (na0 >= k) && (na0 - nd0 < k);
    bool found1 = (na1 >= k) && (na1 - nd1 < k);
    if (found1) {
      final_pivot = p1;
      dup_value = mn1;
      num_dup = nd1;
      num_above = na1;
      break;
    }
    if (found0) {
      final_pivot = p0;
      dup_value = mn0;
      num_dup = nd0;
      num_above = na0;
      break;
    }
    if (na1 > k)
      lo = p1;
    else if (na0 > k)
      lo = p0;
    if (na0 < k)
      hi = p0;
    else if (na1 < k)
      hi = p1;

    if (hi - lo < kKDuplicateTol) {
      final_pivot = (lo + hi) * 0.5f;
      scan_pivot_stats(buf, n, final_pivot, kKDuplicateTol, &num_above,
                       &dup_value, &num_dup);
      break;
    }
  }
  *dup_value_out = dup_value;
  *num_dup_out = num_dup;
  *num_above_out = num_above;
  return final_pivot;
}

static void top_k_row(float* __restrict__ row, int V, int k_val,
                      float* __restrict__ outlier_buf) {
  float avg, std_v;
  compute_mean_std(row, V, &avg, &std_v);

  int tidx = (int)((double)k_val / V * 200);
  if (tidx < 0) tidx = 0;
  if (tidx > 199) tidx = 199;
  float sigma = PERCENTILE_TO_STD_TABLE[tidx];
  sigma = sigma + fabsf(sigma) * -0.15f;
  float outlier_logit = avg + std_v * sigma;

  float max_l, min_l;
  int n_finite, tail_i;
  vec_max_min_with_pad_blend(row, V, kPadSentinel, &max_l, &min_l, &n_finite,
                             &tail_i);
  for (int i = tail_i; i < V; ++i) {
    float v = row[i];
    if (v > max_l) max_l = v;
    if (v > kPadSentinel) {
      ++n_finite;
      if (v < min_l) min_l = v;
    }
  }
  int n_outliers = gather_outliers(row, V, outlier_logit, outlier_buf, nullptr);
  if (min_l > max_l) min_l = max_l;

  if (n_finite <= k_val) return;

  const float neg_inf = -std::numeric_limits<float>::infinity();

  // Degenerate-flat: ternary search is unstable when all values are equal.
  if (max_l - min_l < kKDuplicateTol) {
    int kept = 0;
    for (int i = 0; i < V; ++i) {
      if (row[i] > kPadSentinel) {
        if (kept < k_val)
          ++kept;
        else
          row[i] = neg_inf;
      }
    }
    return;
  }

  float final_pivot, dup_value;
  int n_dup, n_above;
  if (n_outliers > k_val) {
    final_pivot =
        ternary_search_topk(outlier_buf, n_outliers, k_val, outlier_logit,
                            max_l, &dup_value, &n_dup, &n_above);
  } else {
    final_pivot = ternary_search_topk(row, V, k_val, min_l, max_l, &dup_value,
                                      &n_dup, &n_above);
  }

  int n_keep_at_boundary = n_dup - (n_above - k_val);
  if (n_keep_at_boundary < 0) n_keep_at_boundary = 0;
  if (n_keep_at_boundary > n_dup) n_keep_at_boundary = n_dup;

  if (n_keep_at_boundary == 0) {
    mask_write_below(row, V, final_pivot, neg_inf);
  } else {
    apply_boundary_tie_loop(row, V, dup_value, dup_value, kKDuplicateTol,
                            n_keep_at_boundary);
  }
}

static void top_p_row(float* __restrict__ row, int V, float p_val,
                      float* __restrict__ scratch,
                      float* __restrict__ outlier_buf) {
  if (p_val >= 1.0f) return;

  float avg, std_v;
  compute_mean_std(row, V, &avg, &std_v);

  int tidx = (int)(p_val * 200);
  if (tidx < 0) tidx = 0;
  if (tidx > 199) tidx = 199;
  float sigma = NORMAL_CDF_TO_SIGMA_TABLE[tidx];
  sigma = sigma + fabsf(sigma) * -0.25f;
  float outlier_logit = avg + std_v * sigma;

  float max_l, min_l;
  int n_finite, tail_i;
  vec_max_min_with_pad_blend(row, V, kPadSentinel, &max_l, &min_l, &n_finite,
                             &tail_i);
  for (int i = tail_i; i < V; ++i) {
    float v = row[i];
    if (v > max_l) max_l = v;
    if (v > kPadSentinel && v < min_l) min_l = v;
  }
  if (min_l > max_l) min_l = max_l;

  const float neg_inf = -std::numeric_limits<float>::infinity();
  if (p_val <= 0.0f) {
    mask_write_below(row, V, max_l - kBoundaryTol, neg_inf);
    return;
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  DEFINE_FAST_EXP
  vec_op::FP32Vec16 base16(max_l);
#endif
  double sum_exp_d = 0.0;
  {
    int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
    Fp64Acc16 acc;
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      vec_op::FP32Vec16 e16 = fast_exp(v16 - base16);
      acc.add(e16);
    }
    sum_exp_d += acc.reduce();
#endif
    for (; i < V; ++i) sum_exp_d += (double)expf(row[i] - max_l);
  }
  float sum_exp = (float)sum_exp_d;

  float outlier_prob = expf(outlier_logit - max_l) / sum_exp;
  float max_prob = 1.0f / sum_exp;
  float min_prob = expf(min_l - max_l) / sum_exp;

  double sum_out_d = 0.0;
  {
    const float inv_sum = 1.0f / sum_exp;
    int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
    vec_op::FP32Vec16 isum16(inv_sum);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      vec_op::FP32Vec16 p16 = fast_exp(v16 - base16) * isum16;
      p16.save(scratch + i);
    }
#endif
    for (; i < V; ++i) scratch[i] = expf(row[i] - max_l) * inv_sum;
  }
  int n_out =
      gather_outliers(scratch, V, outlier_prob, outlier_buf, &sum_out_d);

  float* sbuf;
  int sn;
  float buf_lo, buf_hi;
  if (sum_out_d >= (double)p_val) {
    sbuf = outlier_buf;
    sn = n_out;
    buf_hi = max_prob;
    buf_lo = outlier_prob;
  } else {
    sbuf = scratch;
    sn = V;
    buf_hi = max_prob;
    buf_lo = min_prob;
  }

  double sum_above;
  float boundary_prob =
      binary_search_buffer(sbuf, sn, p_val, buf_lo, buf_hi, &sum_above);

  float dup_logit = logf(boundary_prob * sum_exp) + max_l;

  // NaN or overflow above max_l means binary search degenerated; skip masking.
  if (!(dup_logit <= max_l)) {
    return;
  }

  int n_at_boundary = count_within_tol(row, V, dup_logit, kBoundaryTol);

  int n_keep_at_boundary = 0;
  if (n_at_boundary > 0) {
    double remaining = (double)p_val - sum_above;
    if (remaining > 0.0 && (double)boundary_prob > 0.0) {
      int64_t needed64 = (int64_t)ceil(remaining / (double)boundary_prob);
      if (needed64 < 0) needed64 = 0;
      if (needed64 > n_at_boundary) needed64 = n_at_boundary;
      n_keep_at_boundary = (int)needed64;
    }
  }

  if (n_keep_at_boundary == 0) {
    mask_write_below(row, V, dup_logit, neg_inf);
  } else {
    apply_boundary_tie_loop(row, V, dup_logit, dup_logit, kBoundaryTol,
                            n_keep_at_boundary);
  }
}

// Fused top-k + top-p: shared mean/std and max/min passes save ~2 row reads
// vs sequential top_k_row + top_p_row.
static void top_k_p_row(float* __restrict__ row, int V, int k_val, float p_val,
                        float* __restrict__ scratch,
                        float* __restrict__ outlier_buf) {
  float avg, std_v;
  compute_mean_std(row, V, &avg, &std_v);

  int k_tidx = (int)((double)k_val / V * 200);
  if (k_tidx < 0) k_tidx = 0;
  if (k_tidx > 199) k_tidx = 199;
  float k_sigma = PERCENTILE_TO_STD_TABLE[k_tidx];
  k_sigma = k_sigma + fabsf(k_sigma) * -0.15f;
  float k_outlier_logit = avg + std_v * k_sigma;

  int p_tidx = (int)(p_val * 200);
  if (p_tidx < 0) p_tidx = 0;
  if (p_tidx > 199) p_tidx = 199;
  float p_sigma = NORMAL_CDF_TO_SIGMA_TABLE[p_tidx];
  p_sigma = p_sigma + fabsf(p_sigma) * -0.25f;
  float p_outlier_logit = avg + std_v * p_sigma;

  float max_l, min_l;
  int n_finite, tail_i;
  vec_max_min_with_pad_blend(row, V, kPadSentinel, &max_l, &min_l, &n_finite,
                             &tail_i);
  for (int i = tail_i; i < V; ++i) {
    float v = row[i];
    if (v > max_l) max_l = v;
    if (v > kPadSentinel) {
      ++n_finite;
      if (v < min_l) min_l = v;
    }
  }
  int n_k_outliers =
      gather_outliers(row, V, k_outlier_logit, outlier_buf, nullptr);
  if (min_l > max_l) min_l = max_l;

  const float neg_inf = -std::numeric_limits<float>::infinity();

  if (k_val > 0 && k_val < V && n_finite > k_val) {
    if (max_l - min_l < kKDuplicateTol) {
      // Degenerate-flat: ternary search is unstable when all values are equal.
      int kept = 0;
      for (int i = 0; i < V; ++i) {
        if (row[i] > kPadSentinel) {
          if (kept < k_val)
            ++kept;
          else
            row[i] = neg_inf;
        }
      }
    } else {
      float dup_value;
      int n_dup, n_above;
      float final_pivot;
      if (n_k_outliers > k_val) {
        final_pivot = ternary_search_topk(outlier_buf, n_k_outliers, k_val,
                                          k_outlier_logit, max_l, &dup_value,
                                          &n_dup, &n_above);
      } else {
        final_pivot = ternary_search_topk(row, V, k_val, min_l, max_l,
                                          &dup_value, &n_dup, &n_above);
      }
      int tk_keep_at_boundary = n_dup - (n_above - k_val);
      if (tk_keep_at_boundary < 0) tk_keep_at_boundary = 0;
      if (tk_keep_at_boundary > n_dup) tk_keep_at_boundary = n_dup;

      if (tk_keep_at_boundary == 0) {
        mask_write_below(row, V, final_pivot, neg_inf);
      } else {
        apply_boundary_tie_loop(row, V, dup_value, dup_value, kKDuplicateTol,
                                tk_keep_at_boundary);
      }
    }
  }

  if (p_val >= 1.0f) return;
  if (p_val <= 0.0f) {
    mask_write_below(row, V, max_l - kBoundaryTol, neg_inf);
    return;
  }

  // Top-k filtered lanes hold neg_inf; exp(-inf)=0 so they don't affect sum.
#if defined(__AVX512F__) || defined(__AVX2__)
  DEFINE_FAST_EXP
  vec_op::FP32Vec16 base16(max_l);
#endif
  double sum_exp_d = 0.0;
  {
    int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
    Fp64Acc16 acc;
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      acc.add(fast_exp(v16 - base16));
    }
    sum_exp_d += acc.reduce();
#endif
    for (; i < V; ++i) sum_exp_d += (double)expf(row[i] - max_l);
  }
  float sum_exp = (float)sum_exp_d;

  float p_outlier_prob = expf(p_outlier_logit - max_l) / sum_exp;
  float max_prob = 1.0f / sum_exp;
  float min_prob = expf(min_l - max_l) / sum_exp;

  {
    const float inv_sum = 1.0f / sum_exp;
    int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
    vec_op::FP32Vec16 isum16(inv_sum);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      (fast_exp(v16 - base16) * isum16).save(scratch + i);
    }
#endif
    for (; i < V; ++i) scratch[i] = expf(row[i] - max_l) * inv_sum;
  }
  double sum_out_d = 0.0;
  int n_out =
      gather_outliers(scratch, V, p_outlier_prob, outlier_buf, &sum_out_d);

  float* sbuf;
  int sn;
  float buf_lo, buf_hi;
  if (sum_out_d >= (double)p_val) {
    sbuf = outlier_buf;
    sn = n_out;
    buf_hi = max_prob;
    buf_lo = p_outlier_prob;
  } else {
    sbuf = scratch;
    sn = V;
    buf_hi = max_prob;
    buf_lo = min_prob;
  }
  double sum_above;
  float boundary_prob =
      binary_search_buffer(sbuf, sn, p_val, buf_lo, buf_hi, &sum_above);

  float dup_logit = logf(boundary_prob * sum_exp) + max_l;

  // NaN or overflow above max_l means binary search degenerated; skip masking.
  if (!(dup_logit <= max_l)) {
    return;
  }

  int n_at_boundary = count_within_tol(row, V, dup_logit, kBoundaryTol);
  int n_keep_at_boundary = 0;
  if (n_at_boundary > 0) {
    double remaining = (double)p_val - sum_above;
    if (remaining > 0.0 && (double)boundary_prob > 0.0) {
      int64_t needed64 = (int64_t)ceil(remaining / (double)boundary_prob);
      if (needed64 < 0) needed64 = 0;
      if (needed64 > n_at_boundary) needed64 = n_at_boundary;
      n_keep_at_boundary = (int)needed64;
    }
  }

  if (n_keep_at_boundary == 0) {
    mask_write_below(row, V, dup_logit, neg_inf);
  } else {
    apply_boundary_tie_loop(row, V, dup_logit, dup_logit, kBoundaryTol,
                            n_keep_at_boundary);
  }
}

void cpu_topp_sampling(torch::Tensor& logits, const torch::Tensor& p) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.is_cpu(), "logits must be CPU");
  TORCH_CHECK(p.is_cpu(), "p must be CPU");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
  TORCH_CHECK(p.dim() == 1 && p.size(0) == logits.size(0),
              "p must be 1D with size == batch size");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const float* pp = p.data_ptr<float>();

  // thread_local avoids reallocation across calls; resize skips zero-fill.
#pragma omp parallel
  {
    thread_local std::vector<float> scratch_tls;
    thread_local std::vector<float> outlier_tls;
    if ((int)scratch_tls.size() < V) scratch_tls.resize(V);
    if ((int)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < B; ++b) {
      top_p_row(lp + b * V, V, pp[b], scratch_tls.data(), outlier_tls.data());
    }
  }
}

void cpu_topk_sampling(torch::Tensor& logits, const torch::Tensor& k) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.is_cpu(), "logits must be CPU");
  TORCH_CHECK(k.is_cpu(), "k must be CPU");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(k.dim() == 1 && k.size(0) == logits.size(0),
              "k must be 1D with size == batch size");
  TORCH_CHECK(k.dtype() == torch::kInt32, "k must be int32");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const int* kp = k.data_ptr<int>();

#pragma omp parallel
  {
    thread_local std::vector<float> outlier_tls;
    if ((int)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < B; ++b) {
      int kv = kp[b];
      if (kv > 0 && kv < V) top_k_row(lp + b * V, V, kv, outlier_tls.data());
    }
  }
}

void cpu_topk_topp_sampling(torch::Tensor& logits, const torch::Tensor& k,
                            const torch::Tensor& p) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.is_cpu(), "logits must be CPU");
  TORCH_CHECK(k.is_cpu(), "k must be CPU");
  TORCH_CHECK(p.is_cpu(), "p must be CPU");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(k.dim() == 1 && k.size(0) == logits.size(0),
              "k must be 1D with size == batch size");
  TORCH_CHECK(k.dtype() == torch::kInt32, "k must be int32");
  TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
  TORCH_CHECK(p.dim() == 1 && p.size(0) == logits.size(0),
              "p must be 1D with size == batch size");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const int* kp = k.data_ptr<int>();
  const float* pp = p.data_ptr<float>();

#pragma omp parallel
  {
    thread_local std::vector<float> scratch_tls;
    thread_local std::vector<float> outlier_tls;
    if ((int)scratch_tls.size() < V) scratch_tls.resize(V);
    if ((int)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < B; ++b) {
      float* row = lp + b * V;
      int kv = kp[b];
      float pv = pp[b];
      if (kv > 0 && kv < V) {
        top_k_p_row(row, V, kv, pv, scratch_tls.data(), outlier_tls.data());
      } else {
        top_p_row(row, V, pv, scratch_tls.data(), outlier_tls.data());
      }
    }
  }
}
