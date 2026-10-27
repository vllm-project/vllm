// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <arm_sve.h>

#if defined(ARM_BF16_SUPPORT)

template <int K, int BLOCK_N, bool has_bias, bool has_silu>
struct tinygemm_kernel<at::BFloat16, K, BLOCK_N, has_bias, has_silu> {
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  static constexpr int64_t PACK_N = 8;
  static constexpr int64_t MICRO_N = bVec::size();
  static constexpr int64_t WEIGHT_PAIR_STRIDE = 2 * PACK_N;
  static constexpr int64_t SECOND_ACC_WEIGHT_OFFSET =
      MICRO_N == PACK_N ? bVec::size() : K * PACK_N;

  static_assert(MICRO_N == PACK_N || MICRO_N == 2 * PACK_N);
  static_assert(2 * fVec::size() == MICRO_N);

  static inline bVec zip1(const bVec& a, const bVec& b) {
    return bVec(svzip1_bf16(static_cast<svbfloat16_t>(a),
                            static_cast<svbfloat16_t>(b)));
  }

  static inline bVec zip2(const bVec& a, const bVec& b) {
    return bVec(svzip2_bf16(static_cast<svbfloat16_t>(a),
                            static_cast<svbfloat16_t>(b)));
  }

  static inline fVec bfdot(const fVec& acc, const bVec& a, const bVec& b) {
    return fVec(svbfdot_f32(static_cast<svfloat32_t>(acc),
                            static_cast<svbfloat16_t>(a),
                            static_cast<svbfloat16_t>(b)));
  }

  // Input layout: A[m][channel]
  // Weight layout: B[N/PACK_N, K/2, PACK_N, 2]
  // The packed weights are reused across all M positions.
  // Supports both SVE-128 and SVE-256.
  static inline void tinygemm_sve(const at::BFloat16* __restrict__ A,
                                  const at::BFloat16* __restrict__ B,
                                  at::BFloat16* __restrict__ C,
                                  const at::BFloat16* __restrict__ bias,
                                  const at::BFloat16* __restrict__ conv_states,
                                  bool has_initial_state, int64_t M,
                                  int64_t lda, bool is_first_token) {
    // Load pairs of kernel weights into 4 registers
    const bVec vb0 = bVec::loadu(B);
    const bVec vb1 = bVec::loadu(B + SECOND_ACC_WEIGHT_OFFSET);
    const bVec vb2 = bVec::loadu(B + WEIGHT_PAIR_STRIDE);
    const bVec vb3 =
        bVec::loadu(B + SECOND_ACC_WEIGHT_OFFSET + WEIGHT_PAIR_STRIDE);

    // Load the input data. A single register holds the values for a
    // single token across all channels for this microtile.
    auto load_a = [&](int64_t m) {
      if (m < 0 && is_first_token) {
        if (!has_initial_state) {
          return bVec(0);
        }
        return bVec::loadu(conv_states + (m + K - 1) * lda);
      }
      return bVec::loadu(A + m * lda);
    };

    // Load the previous inputs from the cached state if available.
    bVec va1 = load_a(-3);
    bVec va2 = load_a(-2);
    bVec va3 = load_a(-1);
    const svbool_t store_pg = svptrue_b16();

    // Iterate over the sequence block, updating the previous conv inputs
    // and reading the next token's input data (8 channels for SVE-128,
    // 16 channels for SVE-256).
    for (int64_t m = 0; m < M; ++m) {
      const bVec va0 = va1;
      va1 = va2;
      va2 = va3;
      va3 = load_a(m);

      fVec vc0(0.0f);
      fVec vc1(0.0f);
      if constexpr (has_bias) {
        const bVec bias_vec = bVec::loadu(bias);
        std::tie(vc0, vc1) = at::vec::convert_to_float(bias_vec);
      }

      vc0 = bfdot(vc0, zip1(va0, va1), vb0);
      vc0 = bfdot(vc0, zip1(va2, va3), vb2);
      vc1 = bfdot(vc1, zip2(va0, va1), vb1);
      vc1 = bfdot(vc1, zip2(va2, va3), vb3);

      if constexpr (has_silu) {
        const fVec one(1.0f);
        vc0 = vc0 / (one + vc0.neg().fexp_u20());
        vc1 = vc1 / (one + vc1.neg().fexp_u20());
      }

      const bVec out = convert_from_float_ext<at::BFloat16>(vc0, vc1);
      svst1_bf16(store_pg, reinterpret_cast<bfloat16_t*>(C + m * lda),
                 static_cast<svbfloat16_t>(out));
    }
  }

  static inline void apply(const at::BFloat16* __restrict__ A,
                           const at::BFloat16* __restrict__ B,
                           at::BFloat16* __restrict__ C,
                           const at::BFloat16* __restrict__ bias,
                           const at::BFloat16* __restrict__ conv_states,
                           bool has_initial_state, int64_t M, int64_t lda,
                           bool is_first_token) {
    static_assert(K == 4);
    static_assert(BLOCK_N % MICRO_N == 0);

    // BLOCK_N is always 32 or 64. M is the block sequence length.
    // Input layout: A[m][channel]
    // Weight layout: B[N/PACK_N, K/2, PACK_N, 2]
    // Use one BF16 vector per microtile.
    for (int64_t n = 0; n < BLOCK_N; n += MICRO_N) {
      tinygemm_sve(A + n, B + n * K, C + n, has_bias ? bias + n : nullptr,
                   conv_states ? conv_states + n : nullptr, has_initial_state,
                   M, lda, is_first_token);
    }
  }
};

#endif
