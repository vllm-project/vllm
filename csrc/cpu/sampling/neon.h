#pragma once

#include "common.h"

#include <arm_neon.h>
#include <torch/types.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cpu_sampling::neon {

FORCE_INLINE uint32x4_t murmur3_hash32x4(uint32x4_t hash,
                                         uint32x4_t token_ids) {
  // `hash` contains the seed/position prefix. Mix four token IDs and finalize
  // four independent hashes, lane-for-lane matching murmur3_hash32().
  token_ids = vmulq_n_u32(token_ids, 0xcc9e2d51);
  token_ids = vorrq_u32(vshlq_n_u32(token_ids, 15), vshrq_n_u32(token_ids, 17));
  token_ids = vmulq_n_u32(token_ids, 0x1b873593);
  hash = veorq_u32(hash, token_ids);
  hash = vorrq_u32(vshlq_n_u32(hash, 13), vshrq_n_u32(hash, 19));
  hash = vaddq_u32(vmulq_n_u32(hash, 5), vdupq_n_u32(0xe6546b64));
  hash = veorq_u32(hash, vdupq_n_u32(16));
  hash = veorq_u32(hash, vshrq_n_u32(hash, 16));
  hash = vmulq_n_u32(hash, 0x85ebca6b);
  hash = veorq_u32(hash, vshrq_n_u32(hash, 13));
  hash = vmulq_n_u32(hash, 0xc2b2ae35);
  return veorq_u32(hash, vshrq_n_u32(hash, 16));
}

FORCE_INLINE float32x4_t murmur3_uniform32x4(uint32x4_t hash,
                                             uint32x4_t token_ids) {
  const uint32x4_t random = murmur3_hash32x4(hash, token_ids);
  // Evaluate u = round_fp32((random + 0.5) / 2^32) without first rounding the
  // entire 32-bit integer to FP32.
  const uint32x4_t high = vshrq_n_u32(random, 16);
  const uint32x4_t low = vandq_u32(random, vdupq_n_u32(0xffff));
  return vmlaq_n_f32(vmulq_n_f32(vcvtq_f32_u32(high), 1.52587890625e-05f),
                     vaddq_f32(vcvtq_f32_u32(low), vdupq_n_f32(0.5f)),
                     2.3283064365386963e-10f);
}

// Adapted from data and inline_logf in Arm Optimized Routines:
// math/aarch64/advsimd/logf.c at commit
// 59afb6fed4275e2fc249df49fce43ee5e75af178.
// https://github.com/ARM-software/optimized-routines
struct PositiveNormalLogData {
  float32x4_t c2;
  float32x4_t c4;
  float32x4_t c6;
  float32x4_t ln2;
  uint32x4_t offset;
  uint32x4_t mantissa_mask;
  float32x4_t odd_coefficients;
};

constexpr PositiveNormalLogData kPositiveNormalLogData = {
    .c2 = {-0x1.4f9934p-3f, -0x1.4f9934p-3f, -0x1.4f9934p-3f, -0x1.4f9934p-3f},
    .c4 = {-0x1.00187cp-2f, -0x1.00187cp-2f, -0x1.00187cp-2f, -0x1.00187cp-2f},
    .c6 = {-0x1.ffffc8p-2f, -0x1.ffffc8p-2f, -0x1.ffffc8p-2f, -0x1.ffffc8p-2f},
    .ln2 = {0x1.62e43p-1f, 0x1.62e43p-1f, 0x1.62e43p-1f, 0x1.62e43p-1f},
    .offset = {0x3f2aaaab, 0x3f2aaaab, 0x3f2aaaab, 0x3f2aaaab},
    .mantissa_mask = {0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff},
    .odd_coefficients = {0x1.5a9aa2p-3f, 0x1.961348p-3f, 0x1.555d7cp-2f,
                         -0x1.3e737cp-3f},
};

// The midpoint conversion produces u = round_fp32((hash + 0.5) / 2^32), so
// 2^-33 <= u <= 1. The direct inner call is clamped to [2^-24, 1], and the
// value passed to the outer call is either the positive series result or
// -log(max(1-u, 2^-24)), approximately in [2^-33, 24*ln(2)]. Both inputs are
// therefore finite positive normal FP32 values (the smallest normal is
// 2^-126), so no special-value path is needed.
FORCE_INLINE float32x4_t positive_normal_log(float32x4_t value) {
  const PositiveNormalLogData& data = kPositiveNormalLogData;
  // Reduce x to x = 2^n * (1+r), evaluate the log1p polynomial in r, then
  // reconstruct log(x) = n*log(2) + log1p(r).
  const uint32x4_t offset_value =
      vsubq_u32(vreinterpretq_u32_f32(value), data.offset);
  const float32x4_t exponent =
      vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(offset_value), 23));
  const uint32x4_t mantissa =
      vaddq_u32(vandq_u32(offset_value, data.mantissa_mask), data.offset);
  const float32x4_t reduced =
      vsubq_f32(vreinterpretq_f32_u32(mantissa), vdupq_n_f32(1.0f));
  const float32x4_t reduced_squared = vmulq_f32(reduced, reduced);

  float32x4_t first =
      vfmaq_laneq_f32(data.c2, reduced, data.odd_coefficients, 0);
  float32x4_t second =
      vfmaq_laneq_f32(data.c4, reduced, data.odd_coefficients, 1);
  float32x4_t third =
      vfmaq_laneq_f32(data.c6, reduced, data.odd_coefficients, 2);
  first = vfmaq_laneq_f32(first, reduced_squared, data.odd_coefficients, 3);
  second = vfmaq_f32(second, first, reduced_squared);
  third = vfmaq_f32(third, second, reduced_squared);
  const float32x4_t leading = vfmaq_f32(reduced, data.ln2, exponent);
  return vfmaq_f32(leading, third, reduced_squared);
}

FORCE_INLINE float32x4_t log1p_neg_stable(float32x4_t value) {
  // Each lane evaluates log(1-x). Use the degree-8 series near zero and the
  // direct logarithm elsewhere; vbsl selects per lane without scalar branches.
  float32x4_t polynomial = vdupq_n_f32(1.0f / 8.0f);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 7.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 4.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f / 2.0f), value, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(1.0f), value, polynomial);
  const float32x4_t series = vnegq_f32(vmulq_f32(value, polynomial));
  const float32x4_t direct =
      positive_normal_log(vmaxq_f32(vsubq_f32(vdupq_n_f32(1.0f), value),
                                    vdupq_n_f32(5.960464477539063e-08f)));
  return vbslq_f32(vcltq_f32(value, vdupq_n_f32(0.25f)), series, direct);
}

FORCE_INLINE float32x4x2_t load_logits(const float* logits) {
  return {vld1q_f32(logits), vld1q_f32(logits + 4)};
}

FORCE_INLINE float32x4x2_t load_logits(const c10::Half* logits) {
  const auto values = vld1q_f16(reinterpret_cast<const float16_t*>(logits));
  return {vcvt_f32_f16(vget_low_f16(values)),
          vcvt_f32_f16(vget_high_f16(values))};
}

FORCE_INLINE float32x4x2_t load_logits(const c10::BFloat16* logits) {
#ifdef ARM_BF16_SUPPORT
  const auto values = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(logits));
  return {vcvt_f32_bf16(vget_low_bf16(values)),
          vcvt_f32_bf16(vget_high_bf16(values))};
#else
  const auto values = vld1q_u16(reinterpret_cast<const uint16_t*>(logits));
  return {
      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(values)), 16)),
      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(values)), 16)),
  };
#endif
}

template <typename scalar_t>
FORCE_INLINE void greedy_argmax(const scalar_t* logits, int64_t token_begin,
                                int64_t token_end, float& best_score,
                                int64_t& best_idx) {
  constexpr int64_t VEC_SIZE = 4;
  constexpr int64_t UNROLLED_SIZE = VEC_SIZE * 4;
  const uint32x4_t first_offsets = {0, 1, 2, 3};
  const uint32x4_t second_offsets =
      vaddq_u32(first_offsets, vdupq_n_u32(VEC_SIZE));
  const uint32x4_t third_offsets =
      vaddq_u32(second_offsets, vdupq_n_u32(VEC_SIZE));
  const uint32x4_t fourth_offsets =
      vaddq_u32(third_offsets, vdupq_n_u32(VEC_SIZE));

  float32x4_t maximum = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  uint32x4_t maximum_indices = vdupq_n_u32(static_cast<uint32_t>(token_begin));

  int64_t token_id = token_begin;
  const int64_t vector_end =
      token_end - (token_end - token_begin) % UNROLLED_SIZE;
  for (; token_id < vector_end; token_id += UNROLLED_SIZE) {
    // Reduce 16 logits into four lanes. Lane j carries the winner among token
    // offsets j, j+4, j+8, and j+12, together with its absolute token ID.
    const float32x4x2_t first = load_logits(logits + token_id);
    const float32x4x2_t second = load_logits(logits + token_id + 2 * VEC_SIZE);

    const float32x4_t first_half = vmaxnmq_f32(first.val[0], first.val[1]);
    const uint32x4_t first_half_offsets = vbslq_u32(
        vceqq_f32(first.val[0], first_half), first_offsets, second_offsets);
    const float32x4_t second_half = vmaxnmq_f32(second.val[0], second.val[1]);
    const uint32x4_t second_half_offsets = vbslq_u32(
        vceqq_f32(second.val[0], second_half), third_offsets, fourth_offsets);
    const float32x4_t block_maximum = vmaxnmq_f32(first_half, second_half);
    const uint32x4_t block_offsets =
        vbslq_u32(vceqq_f32(first_half, block_maximum), first_half_offsets,
                  second_half_offsets);
    const uint32x4_t block_indices =
        vaddq_u32(vdupq_n_u32(static_cast<uint32_t>(token_id)), block_offsets);
    const uint32x4_t update = vcgtq_f32(block_maximum, maximum);
    maximum = vbslq_f32(update, block_maximum, maximum);
    maximum_indices = vbslq_u32(update, block_indices, maximum_indices);
  }

  if (token_end - token_id >= 2 * VEC_SIZE) {
    const float32x4x2_t values = load_logits(logits + token_id);
    const float32x4_t block_maximum = vmaxnmq_f32(values.val[0], values.val[1]);
    const uint32x4_t block_offsets = vbslq_u32(
        vceqq_f32(values.val[0], block_maximum), first_offsets, second_offsets);
    const uint32x4_t block_indices =
        vaddq_u32(vdupq_n_u32(static_cast<uint32_t>(token_id)), block_offsets);
    const uint32x4_t update = vcgtq_f32(block_maximum, maximum);
    maximum = vbslq_f32(update, block_maximum, maximum);
    maximum_indices = vbslq_u32(update, block_indices, maximum_indices);
    token_id += 2 * VEC_SIZE;
  }

  best_score = vmaxvq_f32(maximum);
  // More than one lane can contain the global maximum. Select the smallest
  // associated token ID to preserve first-occurrence argmax semantics.
  const uint32x4_t matching_indices =
      vbslq_u32(vceqq_f32(maximum, vdupq_n_f32(best_score)), maximum_indices,
                vdupq_n_u32(std::numeric_limits<uint32_t>::max()));
  best_idx = vminvq_u32(matching_indices);

  for (; token_id < token_end; ++token_id) {
    const float score = static_cast<float>(logits[token_id]);
    if (score > best_score) {
      best_score = score;
      best_idx = token_id;
    }
  }
}

template <typename scalar_t, bool APPLY_TEMPERATURE>
FORCE_INLINE void score_gumbel_block(const scalar_t* logits, int64_t token_id,
                                     uint32_t token_offset, uint32x4_t hash,
                                     uint32x4_t offsets, float temperature,
                                     float32x4_t& winning_scores,
                                     uint32x4_t& winning_token_offsets) {
  constexpr int64_t VEC_SIZE = 8;
  // For each of eight tokens, compute
  //
  //   u = (Murmur3(seed, position, token) + 0.5) / 2^32
  //   g = -log(-log(1-u))
  //   scaled_logit = APPLY_TEMPERATURE ? logit / temperature : logit
  //   score = scaled_logit + g.
  //
  // The two four-lane score vectors are folded lane-wise. Output lane j is the
  // winner between token offsets j and j+4; token_offset makes that ID relative
  // to the caller's outer block (0, 8, 16, or 24 in the 32-token loop).
  const uint32x4_t first_tokens =
      vaddq_u32(vdupq_n_u32(static_cast<uint32_t>(token_id)), offsets);
  const uint32x4_t second_tokens =
      vaddq_u32(first_tokens, vdupq_n_u32(VEC_SIZE / 2));
  const float32x4x2_t uniforms = {murmur3_uniform32x4(hash, first_tokens),
                                  murmur3_uniform32x4(hash, second_tokens)};
  const float32x4x2_t inner_log = {
      vnegq_f32(log1p_neg_stable(uniforms.val[0])),
      vnegq_f32(log1p_neg_stable(uniforms.val[1]))};
  const float32x4x2_t noise = {
      vnegq_f32(positive_normal_log(inner_log.val[0])),
      vnegq_f32(positive_normal_log(inner_log.val[1]))};

  float32x4x2_t logit = load_logits(logits + token_id);
  if constexpr (APPLY_TEMPERATURE) {
    // Match Triton's temperature division. Its cached-logits path divides by
    // the same temperature on reload and relies on reproducing this value
    // bitwise.
    const float32x4_t divisor = vdupq_n_f32(temperature);
    logit.val[0] = vdivq_f32(logit.val[0], divisor);
    logit.val[1] = vdivq_f32(logit.val[1], divisor);
  }
  const float32x4_t first_scores = vaddq_f32(logit.val[0], noise.val[0]);
  const float32x4_t second_scores = vaddq_f32(logit.val[1], noise.val[1]);
  winning_scores = vmaxnmq_f32(first_scores, second_scores);
  const uint32x4_t first_token_offsets =
      vaddq_u32(offsets, vdupq_n_u32(token_offset));
  const uint32x4_t second_token_offsets =
      vaddq_u32(first_token_offsets, vdupq_n_u32(VEC_SIZE / 2));
  winning_token_offsets = vbslq_u32(vceqq_f32(first_scores, winning_scores),
                                    first_token_offsets, second_token_offsets);
}

template <typename scalar_t, bool APPLY_TEMPERATURE>
void gumbel_argmax(const scalar_t* logits, int64_t token_begin,
                   int64_t token_end, int64_t seed, int64_t position,
                   float temperature, float& best_score, int64_t& best_idx) {
  constexpr int64_t VEC_SIZE = 8;
  constexpr int64_t UNROLLED_SIZE = VEC_SIZE * 4;
  const int64_t vector_end =
      token_end - (token_end - token_begin) % UNROLLED_SIZE;
  const uint32x4_t hash = vdupq_n_u32(murmur3_hash32_prefix(seed, position, 0));
  const uint32x4_t offsets = {0, 1, 2, 3};

  best_score = -std::numeric_limits<float>::infinity();
  best_idx = token_begin;

  int64_t token_id = token_begin;
  for (; token_id < vector_end; token_id += UNROLLED_SIZE) {
    // Keep four independent eight-token pipelines in flight to hide the log
    // latency.
    float32x4_t first_maximum;
    uint32x4_t first_token_offsets;
    score_gumbel_block<scalar_t, APPLY_TEMPERATURE>(
        logits, token_id, 0, hash, offsets, temperature, first_maximum,
        first_token_offsets);
    float32x4_t second_maximum;
    uint32x4_t second_token_offsets;
    score_gumbel_block<scalar_t, APPLY_TEMPERATURE>(
        logits, token_id + VEC_SIZE, VEC_SIZE, hash, offsets, temperature,
        second_maximum, second_token_offsets);
    float32x4_t third_maximum;
    uint32x4_t third_token_offsets;
    score_gumbel_block<scalar_t, APPLY_TEMPERATURE>(
        logits, token_id + 2 * VEC_SIZE, 2 * VEC_SIZE, hash, offsets,
        temperature, third_maximum, third_token_offsets);
    float32x4_t fourth_maximum;
    uint32x4_t fourth_token_offsets;
    score_gumbel_block<scalar_t, APPLY_TEMPERATURE>(
        logits, token_id + 3 * VEC_SIZE, 3 * VEC_SIZE, hash, offsets,
        temperature, fourth_maximum, fourth_token_offsets);

    // reduce the four 8 token pipelines to find the block's argmax
    const float32x4_t first_half_maximum =
        vmaxnmq_f32(first_maximum, second_maximum);
    const uint32x4_t first_half_token_offsets =
        vbslq_u32(vceqq_f32(first_maximum, first_half_maximum),
                  first_token_offsets, second_token_offsets);
    const float32x4_t second_half_maximum =
        vmaxnmq_f32(third_maximum, fourth_maximum);
    const uint32x4_t second_half_token_offsets =
        vbslq_u32(vceqq_f32(third_maximum, second_half_maximum),
                  third_token_offsets, fourth_token_offsets);
    const float32x4_t block_maximum =
        vmaxnmq_f32(first_half_maximum, second_half_maximum);
    const uint32x4_t block_token_offsets =
        vbslq_u32(vceqq_f32(first_half_maximum, block_maximum),
                  first_half_token_offsets, second_half_token_offsets);

    const float score = vmaxnmvq_f32(block_maximum);
    if (score > best_score) {
      const uint32x4_t matching_offsets = vbslq_u32(
          vceqq_f32(block_maximum, vdupq_n_f32(score)), block_token_offsets,
          vdupq_n_u32(std::numeric_limits<uint32_t>::max()));
      best_score = score;
      best_idx = token_id + vminvq_u32(matching_offsets);
    }
  }

  const int64_t vector_tail_end = token_end - (token_end - token_id) % VEC_SIZE;
  for (; token_id < vector_tail_end; token_id += VEC_SIZE) {
    float32x4_t maximum;
    uint32x4_t token_offsets;
    score_gumbel_block<scalar_t, APPLY_TEMPERATURE>(logits, token_id, 0, hash,
                                                    offsets, temperature,
                                                    maximum, token_offsets);
    const float score = vmaxnmvq_f32(maximum);
    if (score > best_score) {
      const uint32x4_t matching_offsets =
          vbslq_u32(vceqq_f32(maximum, vdupq_n_f32(score)), token_offsets,
                    vdupq_n_u32(std::numeric_limits<uint32_t>::max()));
      best_score = score;
      best_idx = token_id + vminvq_u32(matching_offsets);
    }
  }

  for (; token_id < token_end; ++token_id) {
    float logit = static_cast<float>(logits[token_id]);
    if constexpr (APPLY_TEMPERATURE) logit /= temperature;
    const float score = logit + gumbel_noise32(seed, position, token_id);
    if (score > best_score) {
      best_score = score;
      best_idx = token_id;
    }
  }
}

}  // namespace cpu_sampling::neon

namespace cpu_sampling {

template <typename scalar_t>
class SamplingImpl<cpu_utils::ISA::NEON, scalar_t> {
 public:
  using scalar_type = scalar_t;
  static constexpr int64_t RangeAlignment = 32;

  void greedy_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, float& best_score,
                     int64_t& best_idx) const {
    neon::greedy_argmax(logits, token_begin, token_end, best_score, best_idx);
  }

  template <typename score_t, bool APPLY_TEMPERATURE>
  void gumbel_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, int64_t seed, int64_t position,
                     float temperature, score_t& best_score,
                     int64_t& best_idx) const {
    if constexpr (std::is_same_v<score_t, double>) {
      // TODO: do we need to vectorize this?
      scalar_gumbel_argmax<scalar_t, double, APPLY_TEMPERATURE>(
          logits, token_begin, token_end, seed, position, temperature,
          best_score, best_idx);
    } else {
      neon::gumbel_argmax<scalar_t, APPLY_TEMPERATURE>(
          logits, token_begin, token_end, seed, position, temperature,
          best_score, best_idx);
    }
  }
};

}  // namespace cpu_sampling
