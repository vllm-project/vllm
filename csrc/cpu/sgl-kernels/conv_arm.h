// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#if defined(ARM_BF16_SUPPORT)

inline bfloat16x8_t zip1q_bf16(bfloat16x8_t a, bfloat16x8_t b) {
  return vreinterpretq_bf16_u16(
      vzip1q_u16(vreinterpretq_u16_bf16(a), vreinterpretq_u16_bf16(b)));
}

inline bfloat16x8_t zip2q_bf16(bfloat16x8_t a, bfloat16x8_t b) {
  return vreinterpretq_bf16_u16(
      vzip2q_u16(vreinterpretq_u16_bf16(a), vreinterpretq_u16_bf16(b)));
}

template <int K, int BLOCK_N, bool has_bias, bool has_silu>
struct tinygemm_kernel<at::BFloat16, K, BLOCK_N, has_bias, has_silu> {
  static inline bfloat16x8_t load_bf16x8(const at::BFloat16* ptr) {
    return vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ptr));
  }

  // Input layout: A[m][channel]
  // Weight layout: B[N/MICRO_N, K/2, MICRO_N, 2]
  // The packed weights are reused across all M positions.
  static inline void tinygemm_neon_8(
      const at::BFloat16* __restrict__ A, const at::BFloat16* __restrict__ B,
      at::BFloat16* __restrict__ C, const at::BFloat16* __restrict__ bias,
      const at::BFloat16* __restrict__ conv_states, bool has_initial_state,
      int64_t M, int64_t lda, bool is_first_token) {
    bfloat16x8_t va[4];
    // Load 4 pairs of kernel weights into 4 registers
    const bfloat16x8_t vb0 = load_bf16x8(B);       // channels 0-3, taps 0-1
    const bfloat16x8_t vb1 = load_bf16x8(B + 8);   // channels 4-7, taps 0-1
    const bfloat16x8_t vb2 = load_bf16x8(B + 16);  // channels 0-3, taps 2-3
    const bfloat16x8_t vb3 = load_bf16x8(B + 24);  // channels 4-7, taps 2-3

    // Load the input data. A single register holds the values for a
    // single token across all 8 channels for this microtile.
    auto load_a = [&](int64_t m) {
      if (m < 0 && is_first_token) {
        if (!has_initial_state) {
          return vreinterpretq_bf16_u16(vdupq_n_u16(0));
        }
        return load_bf16x8(conv_states + (m + K - 1) * lda);
      }
      return load_bf16x8(A + m * lda);
    };

    // Load the previous inputs from the cached state if available.
    va[1] = load_a(-3);
    va[2] = load_a(-2);
    va[3] = load_a(-1);

    // Iterate over the sequence block, updating the previous conv inputs
    // and reading the next token's input data (8 channels).
    for (int64_t m = 0; m < M; ++m) {
      va[0] = va[1];
      va[1] = va[2];
      va[2] = va[3];
      va[3] = load_a(m);

      float32x4_t vc0;
      float32x4_t vc1;
      if constexpr (has_bias) {
        const bfloat16x8_t bias_vec = load_bf16x8(bias);
        vc0 = vcvtq_low_f32_bf16(bias_vec);
        vc1 = vcvtq_high_f32_bf16(bias_vec);
      } else {
        vc0 = vdupq_n_f32(0.0f);
        vc1 = vdupq_n_f32(0.0f);
      }

      // Use the zip operation to match the input layout to the weights layout
      bfloat16x8_t va_pair =
          zip1q_bf16(va[0], va[1]);  // channels 0-3, inputs for taps 0-1
      vc0 = vbfdotq_f32(vc0, va_pair,
                        vb0);  // 4 independent length-two dot products
      va_pair = zip2q_bf16(va[0], va[1]);  // channels 4-7, inputs for taps 0-1
      vc1 = vbfdotq_f32(vc1, va_pair, vb1);
      va_pair = zip1q_bf16(va[2], va[3]);  // channels 0-3, inputs for taps 2-3
      vc0 =
          vbfdotq_f32(vc0, va_pair,
                      vb2);  // Add to the previous result for the same channels
      va_pair = zip2q_bf16(va[2], va[3]);  // channels 4-7, inputs for taps 2-3
      vc1 = vbfdotq_f32(vc1, va_pair, vb3);

      using fVec = at::vec::Vectorized<float>;
      fVec out0(vc0);
      fVec out1(vc1);
      if constexpr (has_silu) {
        const fVec one(1.0f);
        out0 = out0 / (one + out0.neg().fexp_u20());
        out1 = out1 / (one + out1.neg().fexp_u20());
      }

      // Convert the 2 x 4 fp32 results to 1 x 8 bf16
      const bfloat16x8_t out =
          vcombine_bf16(vcvt_bf16_f32(static_cast<float32x4_t>(out0)),
                        vcvt_bf16_f32(static_cast<float32x4_t>(out1)));
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(C + m * lda), out);
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
    constexpr int64_t MICRO_N = 8;
    static_assert(BLOCK_N % MICRO_N == 0);

    // BLOCK_N is always 32 or 64. M is the block sequence length.
    // Input layout: A[m][channel]
    // Weight layout: B[N/MICRO_N, K/2, MICRO_N, 2]
    // Use an 8 channel microtile and iterate over the 8 channel blocks.
    for (int64_t n = 0; n < BLOCK_N; n += MICRO_N) {
      tinygemm_neon_8(A + n, B + n * K, C + n, has_bias ? bias + n : nullptr,
                      conv_states ? conv_states + n : nullptr,
                      has_initial_state, M, lda, is_first_token);
    }
  }
};

#endif
