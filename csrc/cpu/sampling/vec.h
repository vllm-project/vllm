#pragma once

#include "common.h"

namespace cpu_sampling {

template <typename scalar_t>
class SamplingImpl<cpu_utils::ISA::VEC, scalar_t> {
 public:
  using scalar_type = scalar_t;
  static constexpr int64_t RangeAlignment = 1;

  void greedy_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, float& best_score,
                     int64_t& best_idx) const {
    scalar_greedy_argmax(logits, token_begin, token_end, best_score, best_idx);
  }

  template <typename score_t, bool APPLY_TEMPERATURE>
  void gumbel_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, int64_t seed, int64_t position,
                     float temperature, score_t& best_score,
                     int64_t& best_idx) const {
    scalar_gumbel_argmax<scalar_t, score_t, APPLY_TEMPERATURE>(
        logits, token_begin, token_end, seed, position, temperature, best_score,
        best_idx);
  }
};

}  // namespace cpu_sampling
