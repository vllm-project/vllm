#include "common.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace prefill_plan_minority {
constexpr int ScoreHeads = dosa_geometry::Heads;
using HeadMask = dosa_geometry::UnsignedHeadMask;
__device__ int head_count(HeadMask value) {
  if constexpr (ScoreHeads == 64)
    return __popcll(value);
  else
    return __popc(value);
}
__host__ __device__ constexpr double gn(int n, double u) {
  return n * u / (1 - n * u);
}
constexpr double DOTG = gn(128, 0x1p-23);
constexpr double SCREENG = (1 + DOTG) * (1 + gn(32, 0x1p-24)) - 1;
// Original-head chain position: conversion, remaining FMAs, then two adds.
__device__ double native_chain_gain(int position) {
  double factor = 1.;
  for (int i = 0; i < ScoreHeads / 4 - position + 3; ++i)
    factor = __dmul_ru(factor, 1. + 0x1p-8);
  return factor;
}
__device__ float decode_e4(uint8_t b) {
  return float(*reinterpret_cast<cutlass::float_e4m3_t*>(&b));
}
__device__ uint8_t encode_e4(double x) {
  cutlass::float_e4m3_t y(float(fmin(448., fmax(-448., x))));
  return *reinterpret_cast<uint8_t*>(&y);
}
template <class T>
__device__ T wsum(T x) {
  for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
  return x;
}
__device__ double wmax(double x) {
  for (int d = 16; d; d >>= 1) x = fmax(x, __shfl_down_sync(0xffffffff, x, d));
  return x;
}

#if defined(DOSA_DIRECT_ABS) && DOSA_DIRECT_ABS
// Direct-ABS F64 outward fields; independent of the old encoded DD.
constexpr double DA_SCREEN = 0x1.5001690170d18p-16;
constexpr double DA_G64 = 0x1.0000000000041p-46;
constexpr double DA_G18 = 0x1.20001440016c9p-20;
constexpr double DA_INFLATE = 0x1.0000140071139p+0;
constexpr double DA_GT = 0x1.0000080000401p+0;
constexpr double DA_SQRT128 = 0x1.6a09e667f3bcep+3;
__device__ double da_sum_ru(double x) {
  for (int d = 16; d; d >>= 1)
    x = __dadd_ru(x, __shfl_down_sync(0xffffffff, x, d));
  return x;
}
__device__ float da_upper32(double x) {
  float y = __double2float_rn(x);
  return x >= 0. && isfinite(x) && isfinite(y)
             ? __uint_as_float(__float_as_uint(y) + 1u)
             : INFINITY;
}
__device__ float da_scale(double m, int* valid) {
  if (!isfinite(m)) {
    atomicExch(valid, 0);
    return 1.f;
  }
  auto b = __double_as_longlong(m);
  int e = int((b >> 52) & 2047) - 1023;
  unsigned long long mant = (b & 0xfffffffffffffull) | (1ull << 52);
  int se = m == 0. ? 0 : e - 8 + (mant > (7ull << 50));
  if (se < -64 || se > 16) {
    atomicExch(valid, 0);
    return 1.f;
  }
  return __uint_as_float(unsigned(se + 127) << 23);
}

#endif

// Outward interval for a_h = w_h (q_h - beta_h q_paid), where w_h > 0.
__device__ __forceinline__ void joint_residual_interval(float q, float anchor,
                                                        float weight,
                                                        float beta, double& lo,
                                                        double& hi) {
  lo =
      __dmul_rd(double(weight),
                __dadd_rd(double(q), -__dmul_ru(double(beta), double(anchor))));
  hi =
      __dmul_ru(double(weight),
                __dadd_ru(double(q), -__dmul_rd(double(beta), double(anchor))));
}
__device__ double joint_sum_lo(double x) {
  for (int d = 16; d; d >>= 1)
    x = __dadd_rd(x, __shfl_down_sync(0xffffffffu, x, d));
  return x;
}
__device__ double joint_sum_hi(double x) {
  for (int d = 16; d; d >>= 1)
    x = __dadd_ru(x, __shfl_down_sync(0xffffffffu, x, d));
  return x;
}

template <int Paid>
__global__ void quotient_plan(const uint8_t* q, const float* weights,
                              const HeadMask* capture, uint8_t* fine,
                              float* scales, float* meta, float* raw,
                              float* out, int* hint, const int* budget_tags) {
  static_assert(Paid == 14 || Paid == 16 || Paid == 20 || Paid == 28);
  constexpr int K = 4;
  constexpr double ScreenG =
      Paid <= 20 ? SCREENG : (1 + DOTG) * (1 + gn(40, 0x1p-24)) - 1;
  int b = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t / 32;
  // CTA-uniform query-group dispatch before any input decode, output write or
  // barrier. Inactive outputs remain untouched; only the matching screen may
  // consume them.
  if (budget_tags && budget_tags[b / 8] != Paid) return;
  constexpr int Width = Paid + 2, MetaStride = Paid <= 16 ? 64 : 96;
  // Compact input: 128 real sign masks followed by sticky status.
  __shared__ HeadMask negative_bits, paid_bits, omitted_bits, selected_mask,
      bestid[8];
  __shared__ int valid, paid[Paid], positive_count, fill_count, no_cone;
  __shared__ float strength[ScoreHeads], paid_weight[Paid],
      cone_beta[ScoreHeads];
  __shared__ int minority_votes[ScoreHeads];
  __shared__ int cone_slot[ScoreHeads], omitted_ids[ScoreHeads], omitted_count;
  __shared__ double joint_center[4], joint_gram[8][4], joint_tail[4];
  __shared__ double weight_round[Paid], cone_group[ScoreHeads][4];
  __shared__ double target_guard[ScoreHeads / 4];
  __shared__ float w[ScoreHeads], original_w[ScoreHeads], qf[ScoreHeads][128];
  __shared__ double ideal[2][128], imagnitude[2][128], eff[2][128],
      maximum[2][4], norms[2][4];
  __shared__ float hs[2];
  q += uint64_t(b) * ScoreHeads * 128;
  weights += b * ScoreHeads;
  capture += uint64_t(b) * 129;
  fine += uint64_t(b) * Width * 128;
  scales += b * Width;
  meta += b * MetaStride;
  if (t < ScoreHeads) {
    const float value = weights[t];
    const double gain = native_chain_gain(t / 4);
    original_w[t] = value;
    w[t] = __double2float_ru(__dadd_ru(
        double(value), __dmul_ru(__dadd_ru(gain, -1.), fabs(double(value)))));
    if (t % 4 == 0) {
      const double dot = __ddiv_ru(128. * 0x1p-23, 1. - 128. * 0x1p-23);
      target_guard[t / 4] = __dmul_ru(dot, gain);
    }
  }
  if (t < MetaStride) meta[t] = 0;
  if (t < Width) scales[t] = 1;
  __syncthreads();
  if (t == 0) {
    valid = capture[128] == 0;
    negative_bits = 0;
    paid_bits = 0;
    int np = 0, nn = 0;
    for (int h = 0; h < ScoreHeads; ++h) {
      if (!isfinite(w[h])) valid = 0;
      if (w[h] > 0) {
        paid_bits |= HeadMask(1) << h;
        ++np;
      } else if (w[h] < 0) {
        negative_bits |= HeadMask(1) << h;
        ++nn;
      } else
        valid = 0;
    }
    positive_count = np;
    fill_count = np <= Paid ? Paid - np : Paid;
    if (np > Paid) paid_bits = 0;
    if (np < 1 || nn != ScoreHeads - np) valid = 0;
  }
  __syncthreads();
  for (int i = t; i < ScoreHeads * 128; i += 256) {
    float x = decode_e4(q[i]);
    qf[i / 128][i % 128] = x;
    if (!isfinite(x)) atomicExch(&valid, 0);
  }
  __syncthreads();
  {
// Count128 sign masks once for negative-head priority and supporting vectors.
#pragma unroll
    for (int j = 0; j < ScoreHeads / 8; ++j) {
      int h = warp * (ScoreHeads / 8) + j, count = 0;
#pragma unroll
      for (int batch = 0; batch < 4; ++batch) {
        unsigned mask = __ballot_sync(
            0xffffffffu, ((capture[batch * 32 + lane] >> h) & 1u) != 0);
        count += __popc(mask);
      }
      if (lane == 0) minority_votes[h] = count;
    }
    __syncthreads();
  }
  // P<=Paid: all positives plus exact negatives. P>Paid: strongest Paid
  // positives; omitted positives use a one-sided cone.
  if (fill_count > 0) {
    for (int h = warp; h < ScoreHeads; h += 8) {
      float sum = 0;
      for (int d = lane; d < 128; d += 32)
        sum = __fmaf_rn(qf[h][d], qf[h][d], sum);
      sum = wsum(sum);
      if (lane == 0)
        strength[h] = fabsf(original_w[h]) * sqrtf(fmaxf(sum, 0.f));
    }
  }
  __syncthreads();
  if (t == 0) {
    for (int f = 0; f < fill_count; ++f) {
      int pick = -1;
      float best = -1, best_strength = -1;
      for (int h = 0; h < ScoreHeads; ++h)
        if ((positive_count > Paid ? w[h] > 0 : w[h] < 0) &&
            !((paid_bits >> h) & 1u)) {
          float priority = strength[h];
          {
            if (positive_count <= Paid) {
              int votes = minority_votes[h];
              priority = __fmul_rn(strength[h], float(min(votes, 128 - votes)));
            }
          }
          // Ascending h traversal implements the original head-ID tie break.
          // When all minority counts arezero, strength remains the next key.
          bool take = isfinite(priority) && (priority > best);
          {
            if (positive_count <= Paid)
              take = isfinite(priority) &&
                     (priority > best ||
                      (priority == best && strength[h] > best_strength));
          }
          if (take) {
            best = priority;
            best_strength = strength[h];
            pick = h;
          }
        }
      if (pick < 0) {
        valid = 0;
        break;
      }
      paid_bits |= HeadMask(1) << pick;
    }
    negative_bits &= ~paid_bits;
    int count = 0;
    for (int h = 0; h < ScoreHeads; ++h)
      if ((paid_bits >> h) & 1u) {
        if (count < Paid) paid[count] = h;
        ++count;
      }
    if (count != Paid ||
        head_count(negative_bits) != (positive_count > Paid
                                          ? ScoreHeads - positive_count
                                          : ScoreHeads - Paid))
      valid = 0;
    omitted_bits = 0;
    omitted_count = 0;
    for (int h = 0; h < ScoreHeads; ++h)
      if (w[h] > 0 && !((paid_bits >> h) & 1u)) {
        omitted_bits |= HeadMask(1) << h;
        if constexpr (ScoreHeads == 64 && Paid == 28)
          omitted_ids[omitted_count++] = h;
      }
    for (int j = count; j < Paid; ++j) paid[j] = 0;
    no_cone = (valid && omitted_bits == 0);
  }
  __syncthreads();
  // Fixed CTA-uniform fast branch. Do not re-read mutable valid later:
  // a later encoding failure must not expose uninitialized cone storage.
  if (no_cone) {
    if (t < Paid) {
      double exact = double(w[paid[t]]);
      paid_weight[t] = w[paid[t]];
      // Preserve the original nonzero FP64 formation guard exactly.
      weight_round[t] = gn(128, 0x1p-53) * fabs(exact);
    }
  } else {
    // Query-only heuristic: any finite nonnegative beta yields a legal cone.
    for (int h = warp; h < ScoreHeads; h += 8) {
      float best = -1.f, best_beta = 0.f;
      int pick = 0;
      if ((omitted_bits >> h) & 1u) {
        for (int j = 0; j < Paid; ++j) {
          float dot = 0.f, norm = 0.f;
          for (int d = lane; d < 128; d += 32) {
            float x = qf[paid[j]][d];
            dot = __fmaf_rn(x, qf[h][d], dot);
            norm = __fmaf_rn(x, x, norm);
          }
          dot = wsum(dot);
          norm = wsum(norm);
          if (lane == 0) {
            float score = fmaxf(dot, 0.f) * rsqrtf(fmaxf(norm, 0x1p-100f));
            if (isfinite(score) && score > best) {
              best = score;
              pick = j;
              best_beta = fmaxf(dot, 0.f) / fmaxf(norm, 0x1p-100f);
            }
          }
        }
      }
      if (lane == 0) {
        cone_slot[h] = pick;
        cone_beta[h] = isfinite(best_beta) ? best_beta : 0.f;
      }
    }
    __syncthreads();
    if (t < Paid) {
      double exact = double(w[paid[t]]), mag = fabs(exact);
      for (int h = 0; h < ScoreHeads; ++h)
        if (((omitted_bits >> h) & 1u) && cone_slot[h] == t) {
          double v = double(w[h]) * double(cone_beta[h]);
          exact += v;
          mag += fabs(v);
        }
      paid_weight[t] = float(exact);
      weight_round[t] =
          fabs(exact - double(paid_weight[t])) + gn(128, 0x1p-53) * mag;
      if (!isfinite(paid_weight[t])) atomicExch(&valid, 0);
    }
    for (int job = warp; job < ScoreHeads * 4; job += 8) {
      int h = job / 4, g = job % 4;
      double v = 0;
      if ((omitted_bits >> h) & 1u) {
        int d = g * 32 + lane;
        double x = double(qf[h][d]),
               y = double(cone_beta[h]) * double(qf[paid[cone_slot[h]]][d]);
        double r = fabs(x - y) + gn(128, 0x1p-53) * (fabs(x) + fabs(y));
        v = r * r;
      }
      v = wsum(v);
      if (lane == 0)
        cone_group[h][g] =
            double(w[h] > 0 ? w[h] : 0.f) * sqrt(fmax(v, 0.)) * (1 + 1e-10);
    }
  }
  __syncthreads();
  if constexpr (ScoreHeads == 64 && Paid == 28) {
    // Sum_h ReLU(a_h . k) <= .5 (||sum_h a_h|| + sqrt(sum_ij |a_i.a_j|)) ||k||.
    // Apply within each D32 group. Interval products/sums and RU square roots
    // certify every central/Gram term; existing target/screen error terms
    // remain.
    if (!no_cone) {
      if (warp < 4) {
        const int d = warp * 32 + lane;
        double lo = 0., hi = 0.;
        for (int i = 0; i < omitted_count; ++i) {
          int h = omitted_ids[i];
          double a, b;
          joint_residual_interval(qf[h][d], qf[paid[cone_slot[h]]][d], w[h],
                                  cone_beta[h], a, b);
          lo = __dadd_rd(lo, a);
          hi = __dadd_ru(hi, b);
        }
        double m = fmax(fabs(lo), fabs(hi));
        double n = joint_sum_hi(__dmul_ru(m, m));
        if (lane == 0) joint_center[warp] = __dsqrt_ru(n);
      }
      double sums[4] = {0., 0., 0., 0.};
      for (int job = warp; job < 4 * omitted_count * omitted_count; job += 8) {
        int g = job % 4, pair = job / 4, d = g * 32 + lane;
        int i = omitted_ids[pair / omitted_count],
            j = omitted_ids[pair % omitted_count];
        double a, b, c, e;
        joint_residual_interval(qf[i][d], qf[paid[cone_slot[i]]][d], w[i],
                                cone_beta[i], a, b);
        joint_residual_interval(qf[j][d], qf[paid[cone_slot[j]]][d], w[j],
                                cone_beta[j], c, e);
        double lo = fmin(fmin(__dmul_rd(a, c), __dmul_rd(a, e)),
                         fmin(__dmul_rd(b, c), __dmul_rd(b, e)));
        double hi = fmax(fmax(__dmul_ru(a, c), __dmul_ru(a, e)),
                         fmax(__dmul_ru(b, c), __dmul_ru(b, e)));
        lo = joint_sum_lo(lo);
        hi = joint_sum_hi(hi);
        if (lane == 0) sums[g] = __dadd_ru(sums[g], fmax(fabs(lo), fabs(hi)));
      }
      if (lane == 0)
        for (int g = 0; g < 4; ++g) joint_gram[warp][g] = sums[g];
    }
    __syncthreads();
    if (t < 4) {
      double old = 0.;
      if (!no_cone)
        for (int h = 0; h < ScoreHeads; ++h) old += cone_group[h][t];
      double joint = 0.;
      if (!no_cone) {
        double gram = 0.;
        for (int j = 0; j < 8; ++j) gram = __dadd_ru(gram, joint_gram[j][t]);
        joint = __dmul_ru(.5, __dadd_ru(joint_center[t], __dsqrt_ru(gram)));
      }
      joint_tail[t] = fmin(old, joint);
    }
    __syncthreads();
  }
  // Componentwise majority of the same128 paid masks, with exact tie rule.
  // All choices remain legal supporting vectors; no radix mode sort is needed.
  HeadMask partial = 0;
#pragma unroll
  for (int j = 0; j < ScoreHeads / 8; ++j) {
    int h = warp * (ScoreHeads / 8) + j, count = 0;
    count = minority_votes[h];
    if (count >= 64 && ((negative_bits >> h) & 1u)) partial |= HeadMask(1) << h;
  }
  if (lane == 0) bestid[warp] = partial;
  __syncthreads();
  if (t == 0) {
    HeadMask mask = 0;
    for (int j = 0; j < 8; ++j) mask |= bestid[j];
    selected_mask = mask;
  }
  __syncthreads();
#if defined(DOSA_DIRECT_ABS) && DOSA_DIRECT_ABS
  // Paid14/16 emit post-ABS fields; the high family keeps its original fields.
  if constexpr (Paid == 14 || Paid == 16) {
    if (t == 0) {
      if (!no_cone) valid = 0;
      for (int h = 0; h < ScoreHeads; ++h)
        if (!isfinite(w[h]) || fabsf(w[h]) > 65504.f) valid = 0;
    }
    if (t < 128) {
      double lo = 0., hi = 0.;
      for (int h = 0; h < ScoreHeads; ++h) {
        double wh = isfinite(w[h]) ? double(w[h]) : 0.;
        double z = isfinite(qf[h][t]) ? double(qf[h][t]) : 0.;
        double a = ((selected_mask >> h) & 1u) ? -wh : 0.;
        double b = ((paid_bits >> h) & 1u) ? __dmul_rn(wh, .5) : 0.;
        double term = __dmul_rn(__dadd_rn(a, -b), z);
        lo = __dadd_rd(lo, term);
        hi = __dadd_ru(hi, term);
      }
      ideal[0][t] = __dadd_rn(__dmul_rn(lo, .5), __dmul_rn(hi, .5));
      ideal[1][t] = lo;
      imagnitude[0][t] = hi;
      for (int j = 0; j < Paid; ++j)
        fine[(2 + j) * 128 + t] = q[paid[j] * 128 + t];
      double m = wmax(fabs(ideal[0][t]));
      if (lane == 0) maximum[0][warp] = m;
    }
    if (t < Paid) {
      meta[2 + t] = paid_weight[t];
    }
    __syncthreads();
    if (t == 0) {
      double m = 0.;
      for (int j = 0; j < 4; ++j) m = fmax(m, maximum[0][j]);
      hs[0] = da_scale(m, &valid);
      scales[0] = hs[0];
    }
    __syncthreads();
    if (t < 128) {
      uint8_t code = encode_e4(__ddiv_rn(ideal[0][t], double(hs[0])));
      fine[t] = code;
      eff[0][t] = __dmul_rn(double(decode_e4(code)), double(hs[0]));
      double m = wmax(fabs(__dadd_rn(ideal[0][t], -eff[0][t])));
      if (lane == 0) maximum[1][warp] = m;
    }
    __syncthreads();
    if (t == 0) {
      double m = 0.;
      for (int j = 0; j < 4; ++j) m = fmax(m, maximum[1][j]);
      hs[1] = da_scale(m, &valid);
      scales[1] = hs[1];
    }
    __syncthreads();
    if (t < 128) {
      uint8_t code = encode_e4(
          __ddiv_rn(__dadd_rn(ideal[0][t], -eff[0][t]), double(hs[1])));
      fine[128 + t] = code;
      eff[1][t] = __dmul_rn(double(decode_e4(code)), double(hs[1]));
      double pm = 0., tm = 0.;
      for (int h = 0; h < ScoreHeads; ++h) {
        double z = isfinite(qf[h][t]) ? double(qf[h][t]) : 0.;
        double wh = isfinite(w[h]) ? double(w[h]) : 0.;
        double v = __dmul_rn(fabs(wh), fabs(z));
        double ow = isfinite(original_w[h]) ? double(original_w[h]) : 0.;
        double original = __dmul_ru(fabs(ow), fabs(z));
        tm = __dadd_ru(tm, __dmul_ru(target_guard[h / 4], original));
        pm = __dadd_ru(pm, ((paid_bits >> h) & 1u) ? v : 0.);
      }
      double lower = __dadd_rd(eff[0][t], eff[1][t]);
      double upper = __dadd_ru(eff[0][t], eff[1][t]);
      double residual = fmax(fabs(__dadd_rd(ideal[1][t], -upper)),
                             fabs(__dadd_ru(imagnitude[0][t], -lower)));
      double row = __dadd_ru(fabs(eff[0][t]), fabs(eff[1][t]));
      double numeric = __dadd_ru(__dmul_ru(DA_SCREEN, __dadd_ru(pm, row)), tm);
      numeric = __dadd_ru(
          numeric, __dadd_ru(__dmul_ru(DA_G64, pm), __dmul_ru(DA_G18, row)));
      double joined = __dadd_ru(residual, numeric);
      double ns = da_sum_ru(__dmul_ru(joined, joined));
      if (lane == 0) norms[0][warp] = ns;
    }
    __syncthreads();
    if (t == 0) {
      double n01 = __dadd_ru(norms[0][0], norms[0][1]),
             n23 = __dadd_ru(norms[0][2], norms[0][3]);
      double n = __dsqrt_ru(__dadd_ru(n01, n23));
      float c = da_upper32(__dmul_ru(DA_INFLATE, __dmul_ru(DA_SQRT128, n)));
      double wm = 0., pm = 0.;
      for (int h = 0; h < ScoreHeads; ++h) wm += fabs(double(w[h]));
      for (int j = 0; j < Paid; ++j) pm += fabs(double(paid_weight[j]));
      // Same pre-ABS absolute floor followed by the published ABS floor
      // transform.
      float oldraw = nextafterf(float(fmax(4096., 2048 + 512 * wm + 512 * pm) *
                                      0x1p-126 * (1 + 1e-4)),
                                INFINITY);
      float oldout = 4 * 0x1p-126f;
      float rn =
          da_upper32(__dadd_ru(__dmul_ru(DA_GT, double(oldraw)), 0x1p-120));
      float on =
          da_upper32(__dadd_ru(__dmul_ru(DA_GT, double(oldout)), 0x1p-120));
      if (!(wm <= 0x1p100) || !isfinite(c) || !isfinite(rn) || !isfinite(on) ||
          c < 0.f || rn < oldraw || on < oldout)
        valid = 0;
      meta[Width] = c;
      for (int j = 1; j < 4; ++j) meta[Width + j] = 0.f;
      raw[b] = rn;
      out[b] = on;
      hint[b] = valid ? 2 : 1;
    }
    return;
  }
#endif
  if (t < 128) {
    double v = 0, mag = 0;
    for (int h = 0; h < ScoreHeads; ++h)
      if ((selected_mask >> h) & 1u) {
        double x = fabs(double(w[h])) * double(qf[h][t]);
        v += x;
        mag += fabs(x);
      }
    ideal[0][t] = v;
    imagnitude[0][t] = mag;
    for (int j = 0; j < Paid; ++j)
      fine[(2 + j) * 128 + t] = q[paid[j] * 128 + t];
  }
  if (t < Paid) {
    meta[2 + t] = paid_weight[t];
  }
  __syncthreads();
  // No mean, Gram, OMP or center: encode a0 directly as DD8. Meta beta is zero.
  if (t < 128) {
    double m = wmax(fabs(ideal[0][t]));
    if (lane == 0) maximum[0][warp] = m;
  }
  __syncthreads();
  if (t == 0) {
    double m = 0;
    for (int j = 0; j < 4; ++j) m = fmax(m, maximum[0][j]);
    float s = m > 0 ? float(exp2(ceil(log2(m / 448.)))) : 1.f;
    if (!isfinite(s) || s <= 0) {
      valid = 0;
      s = 1.f;
    }
    hs[0] = s;
    scales[0] = s;
  }
  __syncthreads();
  if (t < 128) {
    uint8_t code = encode_e4(ideal[0][t] / hs[0]);
    fine[t] = code;
    eff[0][t] = double(decode_e4(code)) * hs[0];
    double m = wmax(fabs(ideal[0][t] - eff[0][t]));
    if (lane == 0) maximum[1][warp] = m;
  }
  __syncthreads();
  if (t == 0) {
    double m = 0;
    for (int j = 0; j < 4; ++j) m = fmax(m, maximum[1][j]);
    float s = m > 0 ? float(exp2(ceil(log2(m / 448.)))) : 1.f;
    if (!isfinite(s) || s <= 0) {
      valid = 0;
      s = 1.f;
    }
    hs[1] = s;
    scales[1] = s;
  }
  __syncthreads();
  if (t < 128) {
    uint8_t code = encode_e4((ideal[0][t] - eff[0][t]) / hs[1]);
    fine[128 + t] = code;
    eff[1][t] = double(decode_e4(code)) * hs[1];
  }
  __syncthreads();
  if (t < 128) {
    double paid_mag = 0, round_mag = 0, target_error = 0,
           row_mag = fabs(eff[0][t]) + fabs(eff[1][t]);
    for (int h = 0; h < ScoreHeads; ++h) {
      const double magnitude =
          __dmul_ru(fabs(double(original_w[h])), fabs(double(qf[h][t])));
      target_error =
          __dadd_ru(target_error, __dmul_ru(target_guard[h / 4], magnitude));
    }
    for (int j = 0; j < Paid; ++j) {
      double x = fabs(double(qf[paid[j]][t]));
      paid_mag += fabs(double(paid_weight[j])) * x;
      round_mag += weight_round[j] * x;
    }
    constexpr double sg = gn(2 * K + 10, 0x1p-24), mg = DOTG * (1 + sg) + sg;
    double common = ScreenG * (paid_mag + row_mag) + target_error + round_mag;
    double err = gn(128, 0x1p-53) * imagnitude[0][t];
    double joined =
        fabs(ideal[0][t] - eff[0][t] - eff[1][t]) + err + common + sg * row_mag;
    double sum = wsum(joined * joined);
    if (lane == 0) norms[0][warp] = sum;
  }
  __syncthreads();
  if (t < 4) {
    double tail = 0;
    if (!no_cone)
      for (int h = 0; h < ScoreHeads; ++h) tail += cone_group[h][t];
    if constexpr (ScoreHeads == 64 && Paid == 28) tail = joint_tail[t];
    double value = sqrt(32.) * (sqrt(norms[0][t]) + tail) /
                   (1 - gn(Paid <= 20 ? 20 : 32, 0x1p-24)) * (1 + 1e-10);
    float c = nextafterf(float(value), INFINITY);
    meta[Width + t] = c;
    if (!isfinite(c)) atomicExch(&valid, 0);
  }
  if (t == 0) {
    double wm = 0, bm = 0;
    for (int h = 0; h < ScoreHeads; ++h) wm += fabs(double(w[h]));
    double pm = 0;
    for (int j = 0; j < Paid; ++j) pm += fabs(double(paid_weight[j]));
    if (!(wm <= 0x1p100))
      valid = 0;  // Reference H32 partial sums must not overflow.
    raw[b] =
        nextafterf(float(fmax(4096., 2048 + 512 * wm + 512 * pm + 512 * bm) *
                         0x1p-126 * (1 + 1e-4)),
                   INFINITY);
    out[b] = 4 * 0x1p-126f;
  }
  __syncthreads();
  // Paid rows remain in original-ID order; no active-center permutation exists.
  if (t == 0) {
    // The preceding CTA barrier makes all four emitted F32 coefficients
    // visible. Preserve their existing scalar-round inflation, then use Cauchy
    // over D128.
    {
      double sum = 0;
      for (int g = 0; g < 4; ++g) {
        double c = double(meta[Width + g]);
        sum += c * c;
      }
      float c = nextafterf(float(2 * sqrt(sum) * (1 + 0x1p-40)), INFINITY);
      meta[Width] = c;
      for (int g = 1; g < 4; ++g) meta[Width + g] = 0;
      if (!isfinite(c)) valid = 0;
    }
    hint[b] = valid ? 2 : 1;
  }
}

}  // namespace prefill_plan_minority

namespace minority_planner_host {
template <int Paid>
void launch_plan(Tensor q, Tensor w, Tensor capture,
                 const std::vector<Tensor>& p, Tensor budget) {
  // Six owned fields: FP8 fine[Q,Paid+2,128], F32 scales/meta/raw/out and
  // I32 validity hints. Paid-head IDs and support masks remain local to the
  // plan.
  prefill_plan_minority::quotient_plan<Paid>
      <<<q.size(0), 256, 0, get_current_cuda_stream()>>>(
          static_cast<const uint8_t*>(q.data_ptr()),
          w.mutable_data_ptr<float>(),
          reinterpret_cast<const dosa_geometry::UnsignedHeadMask*>(
              capture.mutable_data_ptr<dosa_geometry::HeadMask>()),
          p[0].mutable_data_ptr<uint8_t>(), p[1].mutable_data_ptr<float>(),
          p[2].mutable_data_ptr<float>(), p[3].mutable_data_ptr<float>(),
          p[4].mutable_data_ptr<float>(), p[5].mutable_data_ptr<int>(),
          budget.mutable_data_ptr<int>());
  DOSA_CUDA_CHECK(cudaGetLastError());
}
// Host selects an existing budget specialization; the per-query early return
// consumes only device tags, so no host query decision or reorder is
// introduced.
void plan128(int64_t paid, Tensor q, Tensor w, Tensor capture,
             std::vector<Tensor> outputs, Tensor budget) {
  DeviceGuard guard(q.get_device_index());
  // Q[Q,H,128], F32[Q,H] weights, masks[Q,129] and I32[Q/8] budgets
  // belong to the owner and remain alive through this current-stream launch.
  if (paid == 14)
    launch_plan<14>(q, w, capture, outputs, budget);
  else if (paid == 16)
    launch_plan<16>(q, w, capture, outputs, budget);
  else if (paid == 20)
    launch_plan<20>(q, w, capture, outputs, budget);
  else if (paid == 28) {
    if constexpr (dosa_geometry::Heads == 64)
      launch_plan<28>(q, w, capture, outputs, budget);
    else
      STD_TORCH_CHECK(false, "Paid28 requires H64");
  } else
    STD_TORCH_CHECK(false, "unsupported paid budget");
}

}  // namespace minority_planner_host

}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("plan128") "(int paid, Tensor q, Tensor weights, Tensor capture, Tensor(a!)[] outputs, Tensor budget) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("plan128"),
           TORCH_BOX(&DOSA_NAMESPACE::minority_planner_host::plan128));
}
