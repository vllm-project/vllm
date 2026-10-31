#pragma once

#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cpu_sampling {

// Draft proposals and target verification must draw from disjoint noise
// streams even when they use the same request seed and token position.
inline constexpr int64_t DRAFT_NOISE_SALT = 1 << 30;

inline uint32_t murmur3_rotl32(uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

inline uint32_t murmur3_mix(uint32_t hash, uint32_t key) {
  key *= 0xcc9e2d51;
  key = murmur3_rotl32(key, 15);
  key *= 0x1b873593;
  hash ^= key;
  hash = murmur3_rotl32(hash, 13);
  return hash * 5 + 0xe6546b64;
}

inline uint32_t murmur3_fmix32(uint32_t hash) {
  hash ^= hash >> 16;
  hash *= 0x85ebca6b;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35;
  return hash ^ (hash >> 16);
}

inline uint32_t murmur3_hash32_prefix(int64_t seed, int64_t position,
                                      uint32_t domain) {
  // Seed and position are shared by every vocabulary entry in a row. Mix this
  // prefix once so the inner loop only needs to mix the token ID and finalize.
  const uint64_t seed_bits = static_cast<uint64_t>(seed);
  uint32_t hash = domain;
  hash = murmur3_mix(hash, static_cast<uint32_t>(seed_bits));
  hash = murmur3_mix(hash, static_cast<uint32_t>(seed_bits >> 32));
  hash = murmur3_mix(hash, static_cast<uint32_t>(position));
  return hash;
}

inline uint32_t murmur3_hash32(int64_t seed, int64_t position,
                               uint32_t token_id, uint32_t domain = 0) {
  uint32_t hash = murmur3_hash32_prefix(seed, position, domain);
  hash = murmur3_mix(hash, token_id);
  return murmur3_fmix32(hash ^ 16);
}

inline float murmur3_uniform32(int64_t seed, int64_t position,
                               uint32_t token_id) {
  const uint32_t random = murmur3_hash32(seed, position, token_id);
  // Splitting the hash makes both integer-to-float conversions exact:
  //   u = round_fp32((random + 0.5) / 2^32).
  // Converting random directly to FP32 could discard up to eight low bits.
  const int32_t high = static_cast<int32_t>(random >> 16);
  const int32_t low = static_cast<int32_t>(random & 0xffff);
  return static_cast<float>(high) * 1.52587890625e-05f +
         (static_cast<float>(low) + 0.5f) * 2.3283064365386963e-10f;
}

inline double murmur3_uniform64(int64_t seed, int64_t position,
                                uint32_t token_id) {
  const uint64_t low = murmur3_hash32(seed, position, token_id);
  const uint64_t high = murmur3_hash32(seed, position, token_id, 0x9e3779b9);
  // Combine two hashes and retain the 53 bits representable by an FP64
  // significand: u = (random53 + 0.5) / 2^53.
  const uint64_t random53 = ((high << 32) | low) >> 11;
  double uniform = (static_cast<double>(random53) + 0.5) * 0x1p-53;
  if (uniform == 1.0) uniform -= 0x1p-53;
  return uniform;
}

inline float log1p_neg_stable(float value) {
  // This is inherited from the triton impl:
  // For small x, evaluate the degree-8 expansion
  //   log(1-x) = -x * (1 + x/2 + x^2/3 + ... + x^7/8)
  // instead of forming 1-x, which loses the low bits of x.
  float polynomial = 1.0f / 8.0f;
  polynomial = 1.0f / 7.0f + value * polynomial;
  polynomial = 1.0f / 6.0f + value * polynomial;
  polynomial = 1.0f / 5.0f + value * polynomial;
  polynomial = 1.0f / 4.0f + value * polynomial;
  polynomial = 1.0f / 3.0f + value * polynomial;
  polynomial = 1.0f / 2.0f + value * polynomial;
  polynomial = 1.0f + value * polynomial;
  const float series = -value * polynomial;
  const float direct = std::log(std::max(1.0f - value, 5.960464477539063e-08f));
  return value < 0.25f ? series : direct;
}

inline float gumbel_noise32(int64_t seed, int64_t position, uint32_t token_id) {
  const float uniform = murmur3_uniform32(seed, position, token_id);
  return -std::log(-log1p_neg_stable(uniform));
}

inline double gumbel_noise64(int64_t seed, int64_t position,
                             uint32_t token_id) {
  const double uniform = murmur3_uniform64(seed, position, token_id);
  return -std::log(-std::log(uniform));
}

template <typename scalar_t>
void scalar_greedy_argmax(const scalar_t* logits, int64_t token_begin,
                          int64_t token_end, float& best_score,
                          int64_t& best_idx) {
  best_score = -std::numeric_limits<float>::infinity();
  best_idx = token_begin;
  for (int64_t token_id = token_begin; token_id < token_end; ++token_id) {
    const float score = static_cast<float>(logits[token_id]);
    if (score > best_score) {
      best_score = score;
      best_idx = token_id;
    }
  }
}

template <typename scalar_t, typename score_t, bool APPLY_TEMPERATURE>
void scalar_gumbel_argmax(const scalar_t* logits, int64_t token_begin,
                          int64_t token_end, int64_t seed, int64_t position,
                          float temperature, score_t& best_score,
                          int64_t& best_idx) {
  best_score = -std::numeric_limits<score_t>::infinity();
  best_idx = token_begin;

  for (int64_t token_id = token_begin; token_id < token_end; ++token_id) {
    float logit = static_cast<float>(logits[token_id]);
    if constexpr (APPLY_TEMPERATURE) logit /= temperature;

    score_t noise;
    if constexpr (std::is_same_v<score_t, double>) {
      noise = gumbel_noise64(seed, position, token_id);
    } else {
      noise = gumbel_noise32(seed, position, token_id);
    }
    const score_t score = static_cast<score_t>(logit) + noise;
    if (score > best_score) {
      best_score = score;
      best_idx = token_id;
    }
  }
}

template <cpu_utils::ISA isa, typename scalar_t>
class SamplingImpl;

template <typename sampling_impl_t>
class SamplingMainLoop {
 public:
  using scalar_t = typename sampling_impl_t::scalar_type;

  void operator()(int64_t* output, const scalar_t* logits,
                  int64_t logits_stride, const int64_t* expanded_idx_mapping,
                  const float* temperature, const int64_t* seeds,
                  const int64_t* positions, int64_t num_tokens,
                  int64_t vocab_size, bool apply_temperature, bool is_drafting,
                  bool use_fp64) const {
    if (num_tokens == 0) return;

    // Resolve runtime options before entering the vocabulary loop so its
    // temperature and score-type branches are compile-time specializations.
    if (apply_temperature) {
      if (use_fp64) {
        execute<double, true>(output, logits, logits_stride,
                              expanded_idx_mapping, temperature, seeds,
                              positions, num_tokens, vocab_size, is_drafting);
      } else {
        execute<float, true>(output, logits, logits_stride,
                             expanded_idx_mapping, temperature, seeds,
                             positions, num_tokens, vocab_size, is_drafting);
      }
    } else if (use_fp64) {
      execute<double, false>(output, logits, logits_stride,
                             expanded_idx_mapping, temperature, seeds,
                             positions, num_tokens, vocab_size, is_drafting);
    } else {
      execute<float, false>(output, logits, logits_stride, expanded_idx_mapping,
                            temperature, seeds, positions, num_tokens,
                            vocab_size, is_drafting);
    }
  }

 private:
  template <typename score_t, bool APPLY_TEMPERATURE>
  void execute(int64_t* output, const scalar_t* logits, int64_t logits_stride,
               const int64_t* expanded_idx_mapping, const float* temperature,
               const int64_t* seeds, const int64_t* positions,
               int64_t num_tokens, int64_t vocab_size, bool is_drafting) const {
    // Parallelize across logit rows first; num_tokens is not necessarily the
    // request count. When there are fewer rows than workers, split each row to
    // expose roughly one task per worker. Keep ranges large enough to amortize
    // queue overhead and align them to an ISA-chosen block multiple (32 on
    // NEON) so range boundaries do not create extra scalar tails.
    constexpr int64_t MIN_RANGE_SIZE = 2048;
    constexpr int64_t RANGE_ALIGNMENT = sampling_impl_t::RangeAlignment;
    static_assert(RANGE_ALIGNMENT > 0);

    const int max_workers = cpu_utils::get_max_threads();
    const int64_t target_ranges_per_row =
        std::max<int64_t>(1, (max_workers + num_tokens - 1) / num_tokens);
    const int64_t unaligned_range_size = std::max<int64_t>(
        MIN_RANGE_SIZE,
        (vocab_size + target_ranges_per_row - 1) / target_ranges_per_row);
    const int64_t range_size = (unaligned_range_size + RANGE_ALIGNMENT - 1) /
                               RANGE_ALIGNMENT * RANGE_ALIGNMENT;
    const int64_t num_ranges = (vocab_size + range_size - 1) / range_size;
    const int64_t num_tasks = num_tokens * num_ranges;

    std::vector<score_t> range_scores(num_tasks);
    std::vector<int64_t> range_indices(num_tasks);
    alignas(64) cpu_utils::Counter task_counter;
    const sampling_impl_t sampling_impl;

#pragma omp parallel
    {
      // Phase 1: find one winner per vocabulary range of each row.
      for (;;) {
        const int64_t task_id = task_counter.acquire_counter();
        if (task_id >= num_tasks) break;

        const int64_t token_idx = task_id / num_ranges;
        const int64_t range_idx = task_id % num_ranges;
        const int64_t token_begin = range_idx * range_size;
        const int64_t token_end =
            std::min(token_begin + range_size, vocab_size);
        const int64_t req_state_idx = expanded_idx_mapping[token_idx];
        const bool valid_request = req_state_idx >= 0;
        const float temp = valid_request ? temperature[req_state_idx] : 0.0f;
        const scalar_t* row = logits + token_idx * logits_stride;

        if (temp == 0.0f) {
          float best_score;
          sampling_impl.greedy_argmax(row, token_begin, token_end, best_score,
                                      range_indices[task_id]);
          range_scores[task_id] = best_score;
        } else {
          int64_t position = positions[token_idx];
          if (is_drafting) position += DRAFT_NOISE_SALT;
          sampling_impl.template gumbel_argmax<score_t, APPLY_TEMPERATURE>(
              row, token_begin, token_end, seeds[req_state_idx], position, temp,
              range_scores[task_id], range_indices[task_id]);
        }
      }

      // Every range winner must be visible before its row is reduced.
#pragma omp barrier
      // Phase 1 found the best token in each vocabulary range. Phase 2 assigns
      // each row to one worker, which compares its range winners to find the
      // best token for the complete vocabulary. Ranges are visited in order,
      // and the strict comparison retains the first token when scores tie.
#pragma omp for schedule(static)
      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        score_t best_score = -std::numeric_limits<score_t>::infinity();
        int64_t best_idx = 0;
        const int64_t first_range = token_idx * num_ranges;
        // Vectorizing this isn't worthwhile because each row has few ranges.
        for (int64_t range_idx = 0; range_idx < num_ranges; ++range_idx) {
          const int64_t task_id = first_range + range_idx;
          const score_t score = range_scores[task_id];
          if (score > best_score) {
            best_score = score;
            best_idx = range_indices[task_id];
          }
        }
        output[token_idx] = best_idx;
      }
    }
  }
};

}  // namespace cpu_sampling
