// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include <c10/util/Unroll.h>

#include "common.h"
#include "gemm.h"
#include "moe.h"

namespace {

// Byte-wise copy for FP8 elements (no remainder assumption)
inline void copy_stub2(
    at::Float8_e4m3fn* __restrict__ out,
    const at::Float8_e4m3fn* __restrict__ input,
    int64_t size) {
  for (int64_t d = 0; d < size; d++) {
    out[d] = input[d];
  }
}

#define PER_TENSOR 1
#define PER_ROW 2
#define PER_GROUP 3

// Store float32 GEMM result to output buffer with dequantization and dtype
// conversion. Activations are PER_ROW quantized; weights are PER_GROUP quantized.
// Uses AVX512 on capable platforms.
#if defined(CPU_CAPABILITY_AVX512)

template <typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  __m512 va_scale, vb_scale;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
    vb_scale = _mm512_set1_ps(b_scale);
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    if constexpr (act_quant_mode != PER_GROUP) {
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto idx) {
      constexpr int j = idx * 16;
      __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
      __m512 bias_vec = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
      if constexpr (act_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, va_scale);
      }
      if constexpr (wei_quant_mode == PER_ROW) {
        vb_scale = _mm512_loadu_ps(scales_b + j);
      }
      if constexpr (wei_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, vb_scale);
      }
      y_vec = _mm512_add_ps(y_vec, bias_vec);
      if constexpr (std::is_same<out_dtype, float>::value) {
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      } else {
        TORCH_CHECK(false, "Unsupported output dtype");
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int j = tail_start; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  }  // for M
}

#else  // no AVX512

template <typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    for (int j = 0; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      float val = y_buf[i * N + j] * a_scale * b_scale;
      if (bias) val += bias[j];
      c_ptr[i * lda + j] = static_cast<out_dtype>(val);
    }
  }
}

#endif  // CPU_CAPABILITY_AVX512

}  // anonymous namespace

// FP8 W8A8 fused MoE kernel.
//
//   hidden_states: [M, K]   FP8 E4M3, pre-quantized activation
//   packed_w1:  [E, 2*Nc, Kc, BLOCK_K, BLOCK_N]  FP8 W8A8 prepack (float8_linear_prepack_cpu)
//   packed_w2:  [E, K, N_packed]  FP8 W8A16 VNNI pack (convert_weight_packed)
//   As:  [M]   float32 per-token activation scales
//   w1s: [E, num_groups, 2N]  float32 per-group weight scales for w1
//   w2s: [E, ...]             float32 per-group weight scales for w2
//
// Stage 1: hidden_states(FP8) @ w1(FP8) → ic0(BF16)  [M*topk, 2N]
//          Uses FP8×FP8 GEMM (tinygemm_kernel with FP8 inputs)
// Stage 1.5: ic1 = silu(ic0[:,:N]) * ic0[:,N:]      [M*topk, N]
// Stage 2: ic1(BF16) @ w2(FP8) → ic2(BF16)          [M*topk, K]
//          Uses BF16×FP8 GEMM (W8A16 path, tinygemm_kernel<scalar_t>)
// Stage 3: out = weighted_sum(ic2, topk_weights)     [M, K]

template <typename scalar_t>
void fused_experts_fp8_a8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    at::Float8_e4m3fn* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    float* __restrict__ Ukernel_tmp,
    const at::Float8_e4m3fn* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ As,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  //   1. intermediate_cache1 : [M * topk, N]
  //   2. intermediate_cache2 : [M * topk, K]
  //   3. A_tmp : [T, BLOCK_M * K]  FP8
  //   4. C_tmp : [T, 2 * BLOCK_M * BLOCK_N]  float32
  //   5. intermediate_cache0 : [M * topk, 2N]
  //   6. B_tmp : [T, MAX_CACHE_BLOCK_SIZE, BLOCK_N, max(K, N)]  scalar_t
  //   7. Ukernel_tmp: [T, 2 * BLOCK_M * BLOCK_N]  float32
  //   8. (non-FP8-brgemm only) dqA [T, BLOCK_M * BLOCK_K] + dqB [T, BLOCK_K * BLOCK_N] in BF16
  //      appended after Ukernel_tmp; see moe.cpp FP8_W8A8 buffer allocation

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N_VAL = block_size_n();  // = 32
  const int64_t KB = K / BLOCK_K;  // K blocks

  int64_t B_tmp_size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N_VAL * std::max(K, N);

  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N_VAL);

  int64_t scale_size_N = div_up(2 * N, block_size_N);
  int64_t scale_size_K = div_up(K, block_size_K);
  int64_t blocks_n_per_group = block_size_N / BLOCK_N_VAL;
  int64_t num_groups = div_up(K, block_size_K);   // G
  int64_t blocks_k_per_group = block_size_K / BLOCK_K;
  const int64_t stride_e = 2 * N * K;  // per-expert stride for w1 (total elements)
  bool use_brgemm = true;
  int64_t num_thread = at::get_num_threads();

  // For the BF16 fallback path (no native FP8 brgemm), tinygemm_kernel needs temporary
  // BF16 buffers to dequantize FP8 inputs before running AMX-BF16 brgemm.
  // These are placed after the Ukernel_tmp region (see moe.cpp FP8_W8A8 buffer allocation).
#ifndef CPUBLAS_BRGEMM_F8F8F32
  // dqA: [T, BLOCK_M, BLOCK_K] BF16
  // dqB: [T, BLOCK_K, BLOCK_N] BF16
  // Layout: immediately after Ukernel_tmp[num_thread * 2 * BLOCK_M * BLOCK_N_VAL]
  at::BFloat16* __restrict__ dq_base = reinterpret_cast<at::BFloat16*>(
      Ukernel_tmp + num_thread * 2 * BLOCK_M * BLOCK_N_VAL);
  const int64_t dq_per_thread = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N_VAL;
#endif

  at::parallel_for(0, MB * NB, 1, [&](int64_t begin, int64_t end) {
    int tid = get_thread_num();
    // A_tmp: FP8 activation buffer for this thread [BLOCK_M, K]
    at::Float8_e4m3fn* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N_VAL;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N_VAL;
    alignas(64) float As_[BLOCK_M];
    float* __restrict__ ukernel_buf_1 = Ukernel_tmp + tid * 2 * BLOCK_M * BLOCK_N_VAL;
#ifndef CPUBLAS_BRGEMM_F8F8F32
    at::BFloat16* __restrict__ dqA_buf = dq_base + tid * dq_per_thread;
    at::BFloat16* __restrict__ dqB_buf = dqA_buf + BLOCK_M * BLOCK_K;
#else
    at::BFloat16* __restrict__ dqA_buf = nullptr;
    at::BFloat16* __restrict__ dqB_buf = nullptr;
#endif
    float* __restrict__ ukernel_buf_2 = ukernel_buf_1 + BLOCK_M * BLOCK_N_VAL;
    auto ldsa = 1;  // activation per-row quantized

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;
      int64_t nb1 = nb + NB;  // gate is first N columns, up is second N columns

      int64_t n_size = std::min(N - nb * BLOCK_N_VAL, BLOCK_N_VAL);

      // B (weight) for expert expert_id: gate+up fused [2N, K] prepackaged as FP8
      int32_t expert_id = expert_ids[mb];
      const at::Float8_e4m3fn* __restrict__ B = packed_w1 + expert_id * stride_e;
      const float* __restrict__ Bs = w1s + expert_id * num_groups * (2 * N);

      // Load activation tile for this M-block
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub2(A + m * K, input + index * K, K);
        As_[m] = As[index];
      }

      const int64_t offset = offsets[mb];

      // Gate path (nb): w1[:N, :]
      zero_buffer(C0, BLOCK_M * BLOCK_N_VAL);
      for (int kci = 0; kci < KB; ++kci) {
        // FP8×FP8 tinygemm: act_quant=PER_ROW(2), wei_quant=PER_GROUP(3)
        tinygemm_kernel(
            /* C       */ C0,
            /* A       */ A + kci * BLOCK_K,
            /* scales_a*/ As_ + kci / blocks_k_per_group,
            /* B       */ B + (nb * KB + kci) * BLOCK_K * BLOCK_N_VAL,
            /* scales_b*/ Bs + nb * BLOCK_N_VAL * num_groups + kci / blocks_k_per_group * BLOCK_N_VAL,
            /* M       */ m_size,
            /* K       */ BLOCK_K,
            /* lda     */ K,
            /* ldc     */ BLOCK_N_VAL,
            /* ldsa    */ ldsa,
            /* ukernel */ ukernel_buf_1,
            /* dqA_buf */ dqA_buf,
            /* dqB_buf */ dqB_buf);
      }
      store_out<scalar_t, BLOCK_N_VAL, PER_ROW, PER_GROUP>(
          C0,
          ic0 + offset * 2 * N + nb * BLOCK_N_VAL,
          m_size,
          2 * N,  // lda
          As_,
          nullptr,
          nullptr);

      // Up path (nb1): w1[N:, :]
      zero_buffer(C1, BLOCK_M * BLOCK_N_VAL);
      for (int kci = 0; kci < KB; ++kci) {
        tinygemm_kernel(
            /* C       */ C1,
            /* A       */ A + kci * BLOCK_K,
            /* scales_a*/ As_ + kci / blocks_k_per_group,
            /* B       */ B + (nb1 * KB + kci) * BLOCK_K * BLOCK_N_VAL,
            /* scales_b*/ Bs + nb1 * BLOCK_N_VAL * num_groups + kci / blocks_k_per_group * BLOCK_N_VAL,
            /* M       */ m_size,
            /* K       */ BLOCK_K,
            /* lda     */ K,
            /* ldc     */ BLOCK_N_VAL,
            /* ldsa    */ ldsa,
            /* ukernel */ ukernel_buf_2,
            /* dqA_buf */ dqA_buf,
            /* dqB_buf */ dqB_buf);
      }
      store_out<scalar_t, BLOCK_N_VAL, PER_ROW, PER_GROUP>(
          C1,
          ic0 + offset * 2 * N + nb1 * BLOCK_N_VAL,
          m_size,
          2 * N,  // lda
          As_,
          nullptr,
          nullptr);
    }
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = silu(ic0[:, :N]) * ic0[:, N:]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2: [E, K, N]  FP8, VNNI packed; treated as W8A16 (BF16 act × FP8 weight)
  const int64_t OC = K;    // output channels (hidden dim)
  const int64_t IC = N;    // input channels (expert intermediate dim)
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N_VAL);
  scale_size_N = div_up(K, block_size_N);
  scale_size_K = div_up(N, block_size_K);
  const int64_t stride_e2 = OC * IC;   // per-expert stride for w2
  const int64_t stride_oc = IC;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  bool use_brgemm2 = can_use_brgemm<at::Float8_e4m3fn>(avg_M);

  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    alignas(64) scalar_t C[BLOCK_M * BLOCK_K];

    loop_2d<at::Float8_e4m3fn>(mb0, mb1, nb0, nb1, BLOCK_N_VAL * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N_VAL, BLOCK_N_VAL);

      // A from ic1: [M*topk, N] in sorted order (contiguous after stage 1.5)
      const scalar_t* __restrict__ A2 = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B from w2: [E, K, N] VNNI packed, offset to expert + output block
      int32_t expert_id = expert_ids[mb];
      const at::Float8_e4m3fn* __restrict__ B2 =
          packed_w2 + expert_id * stride_e2 + nb * BLOCK_N_VAL * stride_oc;
      const float* __restrict__ Bs2 =
          w2s + expert_id * scale_size_N * scale_size_K + (nb / blocks_n_per_group) * scale_size_K;

      // Only unpack B on first visit or new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      tinygemm_kernel<scalar_t>(
          /* A          */ A2,
          /* B          */ B2,
          /* C          */ C,
          /* Btmp       */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N_VAL * IC,
          /* Ctmp       */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N_VAL,
          /* Bbias      */ nullptr,
          /* scale      */ Bs2,
          /* M          */ m_size,
          /* N          */ n_size,
          /* K          */ IC,
          /* lda        */ IC,
          /* ldb        */ n_size,
          /* ldc        */ BLOCK_N_VAL,
          /* brg        */ use_brgemm2,
          /* block_sz_K */ block_size_K,
          /* do_unpack  */ do_unpack);

      // Scatter ic2 in original token order, weighting by topk_weights
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N_VAL, C + m * BLOCK_N_VAL, weight, n_size);
      }
    });

    if (use_brgemm2) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 3: output = sum(ic2, dim=topk)  [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#define INSTANTIATE_MOE_FP8_A8_TEMPLATE(TYPE)           \
  template void fused_experts_fp8_a8_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                        \
      TYPE* __restrict__ ic0,                           \
      TYPE* __restrict__ ic1,                           \
      TYPE* __restrict__ ic2,                           \
      at::Float8_e4m3fn* __restrict__ A_tmp,            \
      TYPE* __restrict__ B_tmp,                         \
      float* __restrict__ C_tmp,                        \
      float* __restrict__ Ukernel_tmp,                  \
      const at::Float8_e4m3fn* __restrict__ input,      \
      const at::Float8_e4m3fn* __restrict__ packed_w1,  \
      const at::Float8_e4m3fn* __restrict__ packed_w2,  \
      const float* __restrict__ As,                     \
      const float* __restrict__ w1s,                    \
      const float* __restrict__ w2s,                    \
      int64_t block_size_N,                             \
      int64_t block_size_K,                             \
      const float* __restrict__ topk_weights,           \
      const int32_t* __restrict__ sorted_ids,           \
      const int32_t* __restrict__ expert_ids,           \
      const int32_t* __restrict__ offsets,              \
      int64_t M,                                        \
      int64_t N,                                        \
      int64_t K,                                        \
      int64_t E,                                        \
      int64_t topk,                                     \
      int64_t num_tokens_post_pad)

INSTANTIATE_MOE_FP8_A8_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_FP8_A8_TEMPLATE(at::Half);
