// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "cpu_arch_macros.h"
#include "cpu_types.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr float kPadSentinel = -1e30f;  // logits below this are PAD
constexpr float kBoundaryTol = 1e-5f;   // tie tolerance for top-p cut
constexpr int kMeanStdSampleN = 8192;   // Pass 0 sample size
constexpr int kBsearchMaxIters = 32;    // binary-search iteration cap
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

namespace {

// Stateful FP64 accumulator over FP32Vec16 chunks. Cross-lane reduction
// happens once at reduce() — keeps Pass 2's chained-acc pattern intact.
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

// Sum (in fp64) of buf[i] for i where buf[i] > threshold.
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
  // _mm256_and_ps with the cmp sign-mask zeroes excluded lanes; zero
  // contributes nothing to the FP64 sum (do NOT use blendv with -inf here).
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

// Count elements with |row[i] - center| < tol.
__attribute__((always_inline)) static inline int count_within_tol(
    const float* row, int V, float center, float tol) {
  int n = 0;
  int i = 0;
#if defined(__AVX512F__)
  const __m512 c = _mm512_set1_ps(center);
  const __m512 t = _mm512_set1_ps(tol);
  for (; i + 16 <= V; i += 16) {
    __m512 v = _mm512_loadu_ps(row + i);
    __m512 a = _mm512_abs_ps(_mm512_sub_ps(v, c));
    __mmask16 m = _mm512_cmp_ps_mask(a, t, _CMP_LT_OQ);
    n += _mm_popcnt_u32((unsigned)m);
  }
#elif defined(__AVX2__)
  const __m256 c = _mm256_set1_ps(center);
  const __m256 t = _mm256_set1_ps(tol);
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  for (; i + 16 <= V; i += 16) {
    __m256 vlo = _mm256_loadu_ps(row + i);
    __m256 vhi = _mm256_loadu_ps(row + i + 8);
    __m256 abslo = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(vlo, c));
    __m256 abshi = _mm256_andnot_ps(sign_mask, _mm256_sub_ps(vhi, c));
    __m256 clo = _mm256_cmp_ps(abslo, t, _CMP_LT_OQ);
    __m256 chi = _mm256_cmp_ps(abshi, t, _CMP_LT_OQ);
    n += __builtin_popcount((unsigned)_mm256_movemask_ps(clo));
    n += __builtin_popcount((unsigned)_mm256_movemask_ps(chi));
  }
#endif
  for (; i < V; ++i)
    if (fabsf(row[i] - center) < tol) ++n;
  return n;
}

// In-place: where !(row[i] > threshold), write fill.
__attribute__((always_inline)) static inline void mask_write_below(
    float* row, int V, float threshold, float fill) {
  int i = 0;
#if defined(__AVX512F__)
  const __m512 thr = _mm512_set1_ps(threshold);
  const __m512 f = _mm512_set1_ps(fill);
  for (; i + 16 <= V; i += 16) {
    __m512 v = _mm512_loadu_ps(row + i);
    __mmask16 keep = _mm512_cmp_ps_mask(v, thr, _CMP_GT_OQ);
    _mm512_storeu_ps(row + i, _mm512_mask_blend_ps(keep, f, v));
  }
#elif defined(__AVX2__)
  const __m256 thr = _mm256_set1_ps(threshold);
  const __m256 f = _mm256_set1_ps(fill);
  for (; i + 16 <= V; i += 16) {
    __m256 vlo = _mm256_loadu_ps(row + i);
    __m256 vhi = _mm256_loadu_ps(row + i + 8);
    __m256 klo = _mm256_cmp_ps(vlo, thr, _CMP_GT_OQ);
    __m256 khi = _mm256_cmp_ps(vhi, thr, _CMP_GT_OQ);
    _mm256_storeu_ps(row + i, _mm256_blendv_ps(f, vlo, klo));
    _mm256_storeu_ps(row + i + 8, _mm256_blendv_ps(f, vhi, khi));
  }
#endif
  for (; i < V; ++i)
    if (!(row[i] > threshold)) row[i] = fill;
}

}  // namespace

// Binary search on a pre-computed probability buffer.
static float binary_search_buffer(const float* buf, int n, float p_val,
                                  float lo, float hi, double* sum_above_out) {
  for (int iter = 0; iter < kBsearchMaxIters; ++iter) {
    float mid = lo + (hi - lo) * 0.5f;
    if (mid == lo || mid == hi) break;
    double s = sum_gt_to_double(buf, n, mid);
    if (s >= (double)p_val)
      lo = mid;
    else
      hi = mid;
  }
  double s = sum_gt_to_double(buf, n, hi);
  *sum_above_out = s;
  return hi;
}

static void top_p_row(float* __restrict__ row, int V, float p_val,
                      float* __restrict__ scratch,
                      float* __restrict__ outlier_buf) {
  if (p_val >= 1.0f) return;

  // Pass 0: scalar mean/std estimate over up to kMeanStdSampleN logits.
  int ns = V < kMeanStdSampleN ? V : kMeanStdSampleN;
  float sum_s = 0.f, sum_sq = 0.f;
  int nf = 0;
  for (int i = 0; i < ns; ++i) {
    float v = row[i];
    if (v > kPadSentinel) {
      sum_s += v;
      sum_sq += v * v;
      ++nf;
    }
  }
  float avg = nf > 0 ? sum_s / nf : 0.f;
  float var = nf > 0 ? sum_sq / nf - avg * avg : 1.f;
  float std_v = sqrtf(var > 0.f ? var : 0.f);

  int tidx = (int)(p_val * 200);
  if (tidx < 0) tidx = 0;
  if (tidx > 199) tidx = 199;
  float sigma = NORMAL_CDF_TO_SIGMA_TABLE[tidx];
  sigma = sigma + fabsf(sigma) * -0.25f;
  float outlier_logit = avg + std_v * sigma;

  // Pass 1: vectorised max/min scan. PAD lanes (<= kPadSentinel) are blended
  // to +inf so they don't corrupt reduce_min.
  float max_l, min_l;
  {
    int i = 0;
#if defined(__AVX512F__) || defined(__AVX2__)
    const float pos_inf_val = std::numeric_limits<float>::infinity();
    vec_op::FP32Vec16 maxv(row[0]);
    vec_op::FP32Vec16 minv(pos_inf_val);
  #if defined(__AVX512F__)
    const __m512 sentinel = _mm512_set1_ps(kPadSentinel);
    const __m512 pos_inf = _mm512_set1_ps(pos_inf_val);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      maxv = maxv.max(v16);
      __mmask16 lt = _mm512_cmp_ps_mask(v16.reg, sentinel, _CMP_LE_OS);
      __m512 safe = _mm512_mask_blend_ps(lt, v16.reg, pos_inf);
      minv.reg = _mm512_min_ps(minv.reg, safe);
    }
  #else  // AVX2
    const __m256 sentinel256 = _mm256_set1_ps(kPadSentinel);
    const __m256 pos_inf256 = _mm256_set1_ps(pos_inf_val);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      maxv = maxv.max(v16);
      __m256 lt_lo = _mm256_cmp_ps(v16.reg_low, sentinel256, _CMP_LE_OS);
      __m256 lt_hi = _mm256_cmp_ps(v16.reg_high, sentinel256, _CMP_LE_OS);
      minv.reg_low = _mm256_min_ps(
          minv.reg_low, _mm256_blendv_ps(v16.reg_low, pos_inf256, lt_lo));
      minv.reg_high = _mm256_min_ps(
          minv.reg_high, _mm256_blendv_ps(v16.reg_high, pos_inf256, lt_hi));
    }
  #endif
    max_l = maxv.reduce_max();
    min_l = minv.reduce_min();
    for (; i < V; ++i) {
      float v = row[i];
      if (v > max_l) max_l = v;
      if (v > kPadSentinel && v < min_l) min_l = v;
    }
#else
    max_l = -1e38f;
    min_l = 1e38f;
    for (; i < V; ++i) {
      float v = row[i];
      if (v > max_l) max_l = v;
      if (v > kPadSentinel && v < min_l) min_l = v;
    }
#endif
  }
  if (min_l > max_l) min_l = max_l;

  // fast_exp + base16 are reused by Pass 2 and Pass 3.
#if defined(__AVX512F__) || defined(__AVX2__)
  DEFINE_FAST_EXP
  vec_op::FP32Vec16 base16(max_l);
#endif

  // Pass 2: vectorised sum_exp; FP64 accumulation guards against rounding
  // over 128K elements.
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

  // Pass 3: vectorised prob buffer fill + outlier gather.
  int n_out = 0;
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

    // Outlier gather is scalar — branchy, not worth vectorising.
    for (int j = 0; j < V; ++j) {
      float prob = scratch[j];
      if (prob > outlier_prob) {
        outlier_buf[n_out++] = prob;
        sum_out_d += (double)prob;
      }
    }
  }

  // PATH A (peaked) or PATH B (flat).
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
  if (!(dup_logit < max_l)) return;

  int n_at_boundary = count_within_tol(row, V, dup_logit, kBoundaryTol);

  int n_keep_at_boundary = 0;
  if (n_at_boundary > 0) {
    double remaining = (double)p_val - sum_above;
    if (remaining > 0.0 && (double)boundary_prob > 0.0) {
      int needed = (int)ceil(remaining / (double)boundary_prob);
      n_keep_at_boundary = needed < n_at_boundary ? needed : n_at_boundary;
    }
  }

  const float neg_inf = -std::numeric_limits<float>::infinity();

  if (n_keep_at_boundary == 0) {
    // Fast path (>99% of calls): no boundary tokens to keep partially.
    mask_write_below(row, V, dup_logit, neg_inf);
  } else {
    // Rare PATH-A duplicate case: order-dependent boundary keep must stay
    // sequential.
    int kept_boundary = 0;
    for (int i = 0; i < V; ++i) {
      float v = row[i];
      bool keep = v > dup_logit;
      if (!keep && fabsf(v - dup_logit) < kBoundaryTol &&
          kept_boundary < n_keep_at_boundary) {
        keep = true;
        ++kept_boundary;
      }
      if (!keep) row[i] = neg_inf;
    }
  }
}

void cpu_topp_sampling(torch::Tensor& logits, const torch::Tensor& p) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(p.dim() == 1 && p.size(0) == logits.size(0),
              "p must be 1D with size == batch size");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

  const int B = static_cast<int>(logits.size(0));
  const int V = static_cast<int>(logits.size(1));
  float* lp = logits.data_ptr<float>();
  const float* pp = p.data_ptr<float>();

  // thread_local scratch survives across calls — subsequent invocations are
  // allocation-free. resize (not assign) avoids per-call zero-fill.
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
