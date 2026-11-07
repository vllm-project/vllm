// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include <atomic>
#include <cmath>
#include <limits>

#include "common.h"
#include "gemm.h"
#include "moe.h"
#include <ATen/cpu/vec/vec_convert.h>

namespace {

constexpr int64_t FP8_WEIGHT_BLOCK = 128;

std::pair<uint8_t, uint8_t> mxfp4_scale_range(
    const uint8_t* scales, int64_t panels, int64_t stride, int64_t count) {
#if defined(CPU_CAPABILITY_AVX512)
  __m256i hi = _mm256_setzero_si256();
  __m256i lo = _mm256_set1_epi8(-1);
  for (int64_t p = 0; p < panels; ++p) {
    for (int64_t k = 0; k < count; k += 32) {
      const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales + p * stride + k));
      hi = _mm256_max_epu8(hi, s);
      lo = _mm256_min_epu8(lo, s);
    }
  }
  const __m128i hi128 = _mm_max_epu8(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi, 1));
  const __m128i lo128 = _mm_min_epu8(_mm256_castsi256_si128(lo), _mm256_extracti128_si256(lo, 1));
  return {_mm512_reduce_max_epu32(_mm512_cvtepu8_epi32(hi128)),
          _mm512_reduce_min_epu32(_mm512_cvtepu8_epi32(lo128))};
#else
  uint8_t hi = 0, lo = 255;
  for (int64_t p = 0; p < panels; ++p) {
    for (int64_t k = 0; k < count; ++k) {
      const uint8_t s = scales[p * stride + k];
      hi = std::max(hi, s);
      lo = std::min(lo, s);
    }
  }
  return {hi, lo};
#endif
}

template <typename fp8_t, typename weight_t, typename scale_t, bool down, typename output_t = at::BFloat16>
void fp8_moe_project(
    output_t* __restrict__ output,
    const fp8_t* __restrict__ input, const float* __restrict__ activation_scales,
    const float* __restrict__ fp32_tiny_input,
    const weight_t* __restrict__ weights, const scale_t* __restrict__ scales, const float* __restrict__ bias,
    const float* __restrict__ block_metadata,
    const float* __restrict__ channel_metadata,
    fp8_t* __restrict__ A_tmp, fp8_t* __restrict__ B_tmp, float* __restrict__ C_tmp,
    float* __restrict__ S_tmp,
    const float* __restrict__ topk_weights, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ offsets,
    int64_t OC, int64_t IC, int64_t topk,
    int64_t MB, int64_t cache_width) {
  constexpr bool compressed = std::is_same_v<weight_t, uint8_t>;
  constexpr int64_t BM = block_size_m();
  constexpr int64_t BN = block_size_n();
  const int64_t groups = div_up(IC, FP8_WEIGHT_BLOCK);
  const int64_t cache_stride = MAX_CACHE_BLOCK_SIZE * BN * cache_width;
  std::atomic<bool> invalid_scale{false};

  parallel_2d(MB, OC / BN, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    const int tid = get_thread_num();
    fp8_t* __restrict__ gathered_input = A_tmp + tid * BM * cache_width;
    float* __restrict__ total = C_tmp + tid * 2 * BM * BN;
    float* __restrict__ partial = total + BM * BN;
    alignas(64) float row_scales[BM];
    alignas(64) at::BFloat16 tile[BM * BN];
    float* cache_scales = nullptr;
    if constexpr (compressed) {
      if (!block_metadata && S_tmp) cache_scales = S_tmp + tid * MAX_CACHE_BLOCK_SIZE * (cache_width / FP8_WEIGHT_BLOCK);
    }
    bool used_brgemm = false;

    loop_2d<weight_t>(mb0, mb1, nb0, nb1, BN * IC,
        [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      if constexpr (compressed) {
        if (!block_metadata && invalid_scale.load(std::memory_order_relaxed)) return;
      }
      const int64_t rows = offsets[mb + 1] - offsets[mb];
      if (rows == 0) return;
      const int32_t expert = expert_ids[mb];
      const int32_t* ids = sorted_ids + mb * BM;
      if constexpr (compressed) {
        // Keep tails of larger experts on FP8 to reuse decoded panels.
        if (rows <= 4 && (mb == 0 || expert != expert_ids[mb - 1])) {
          const auto* panel_scales = scales + expert * OC * (IC / 32) + nb * IC;
          if (!block_metadata && mxfp4_scale_range(panel_scales, 1, IC, IC).first == 255) {
            invalid_scale.store(true, std::memory_order_relaxed);
            return;
          }
          // The 32-row FP8 workspace also fits four BF16 rows.
          auto* tiny_input = reinterpret_cast<at::BFloat16*>(gathered_input);
          if (nb_offset == 0) {
            for (int64_t m = 0; m < rows; ++m) {
              const int64_t index = down ? offsets[mb] + m : ids[m] / topk;
              if (fp32_tiny_input) {
                copy_stub(tiny_input + m * IC, fp32_tiny_input + index * IC, IC);
              } else {
                using Vec = at::vec::Vectorized<float>;
                using QVec = at::vec::Vectorized<fp8_t>;
                const Vec scale(activation_scales[index]);
                for (int64_t k = 0; k < IC; k += Vec::size()) {
                  const Vec q = at::vec::convert<float>(QVec::loadu(input + index * IC + k, Vec::size()));
                  store_from_float_ext(tiny_input + m * IC + k, q * scale);
                }
              }
            }
          }
          const auto* packed = weights + expert * OC * (IC / 2) + nb * BN * (IC / 2);
          const float* b = bias ? bias + expert * OC + nb * BN : nullptr;
          using tiny_output_t = std::conditional_t<down, at::BFloat16, output_t>;
          tiny_output_t* C;
          if constexpr (down) {
            C = tile;
          } else {
            C = output + offsets[mb] * OC + nb * BN;
          }
          tinygemm_kernel<at::BFloat16>(
              tiny_input, packed, C, nullptr, nullptr, b, panel_scales,
              rows, BN, IC, IC, BN, down ? BN : OC, false, 32);
          if constexpr (down) {
            for (int64_t m = 0; m < rows; ++m) {
              copy_mul_stub(output + ids[m] * OC + nb * BN,
                  tile + m * BN, topk_weights[ids[m]], BN);
            }
          }
          return;
        }
      }
      const fp8_t* __restrict__ A = down ? input + offsets[mb] * IC : gathered_input;
      if (nb_offset == 0) {
        for (int64_t m = 0; m < rows; ++m) {
          const int64_t index = down ? offsets[mb] + m : ids[m] / topk;
          if constexpr (!down) std::memcpy(gathered_input + m * IC, input + index * IC, IC);
          row_scales[m] = activation_scales[index];
        }
      }

      const fp8_t* panel;
      const scale_t* panel_scales;
      const float* normalized_scales = nullptr;
      const float* channel_scales = nullptr;
      bool channel_normal = false;
      const int64_t scale_stride = block_metadata ? 3 : 1;
      if constexpr (compressed) {
        if (channel_metadata) {
          channel_scales = channel_metadata + (expert * (OC / BN) + nb) * 3 * BN;
          channel_normal = true;
          for (int64_t n = 0; n < BN; ++n) {
            channel_normal &= channel_scales[2 * BN + n] != 0;
          }
        }
        fp8_t* cached = B_tmp + tid * cache_stride + nb_offset * BN * IC;
        const bool unpack = mb == mb0 || expert != expert_ids[mb - 1];
        panel_scales = scales + expert * OC * (IC / 32) + nb * IC;
        const int64_t stripe = nb / 4;
        if (block_metadata) {
          normalized_scales = block_metadata + (expert * div_up(OC, FP8_WEIGHT_BLOCK) + stripe) * groups * 3;
        } else {
          normalized_scales = cache_scales + nb_offset * groups;
        }
        if (unpack) {
          const int64_t panels = std::min(int64_t(4), OC / BN - stripe * 4);
          const uint8_t* stripe_scales = scales + expert * OC * (IC / 32) + stripe * 128 * (IC / 32);
          const auto* packed = weights + expert * OC * (IC / 2) + nb * BN * (IC / 2);
          if (channel_normal) {
            unpack_mxfp4_e4m3(cached, packed, panel_scales, IC, 0, true, channel_scales + BN);
          } else for (int64_t g = 0; g < groups; ++g) {
            const int64_t start = g * FP8_WEIGHT_BLOCK;
            const int64_t count = std::min(FP8_WEIGHT_BLOCK, IC - start);
            int smax;
            bool normal;
            if (block_metadata) {
              smax = int(normalized_scales[g * 3 + 1]);
              normal = normalized_scales[g * 3 + 2] != 0;
            } else {
              // Read the full logical block, even if this cache starts inside its stripe.
              const auto [hi, lo] = mxfp4_scale_range(
                  stripe_scales + (start / 32) * BN, panels, IC, (count / 32) * BN);
              if (hi == 255) {
                invalid_scale.store(true, std::memory_order_relaxed);
                return;
              }
              smax = hi;
              normal = int(hi) - int(lo) <= 11;
              cache_scales[nb_offset * groups + g] = std::ldexp(1.0f, smax - 133);
            }
            unpack_mxfp4_e4m3(cached + start * BN, packed + (start / 2) * BN,
                panel_scales + (start / 32) * BN, count, smax, normal);
          }
        }
        panel = cached;
      } else {
        panel = weights + expert * OC * IC + nb * BN * IC;
        panel_scales = scales + (expert * (OC / BN) + nb) * groups * BN;
      }

      float final_scale = 1.0f;
      if (channel_normal) {
#ifdef CPUBLAS_BRGEMM_F8F8F32
        at::native::cpublas::brgemm(rows, BN, IC, IC, BN, BN, false, A, panel, total, true);
        used_brgemm = true;
#endif
      } else for (int64_t g = 0; g < groups;) {
        float weight_scale;
        int64_t next = g + 1;
        if constexpr (compressed) {
          weight_scale = normalized_scales[g * scale_stride];
          while (next < groups && normalized_scales[next * scale_stride] == weight_scale) ++next;
        } else {
          weight_scale = panel_scales[g * BN];
          while (next < groups && panel_scales[next * BN] == weight_scale) ++next;
        }
        const int64_t start = g * FP8_WEIGHT_BLOCK;
        const int64_t count = std::min(IC, next * FP8_WEIGHT_BLOCK) - start;
        const bool full_k = g == 0 && next == groups;
        if (!full_k && g == 0) zero_buffer(total, rows * BN);
#ifdef CPUBLAS_BRGEMM_F8F8F32
        at::native::cpublas::brgemm(rows, BN, count, IC, BN, BN,
            false, A + start, panel + start * BN, full_k ? total : partial, true);
        used_brgemm = true;
#endif
        if (full_k) {
          final_scale = weight_scale;
        } else {
          using Vec = at::vec::Vectorized<float>;
          const Vec scale(weight_scale);
          for (int64_t j = 0; j < rows * BN; j += Vec::size()) {
            (Vec::loadu(total + j) + Vec::loadu(partial + j) * scale).store(total + j);
          }
        }
        g = next;
      }
      using Vec = at::vec::Vectorized<float>;
      const float* b = bias ? bias + expert * OC + nb * BN : nullptr;
      const Vec weight_scale(final_scale);
      for (int64_t m = 0; m < rows; ++m) {
        const Vec activation_scale(row_scales[m]);
        for (int64_t n = 0; n < BN; n += Vec::size()) {
          Vec y = Vec::loadu(total + m * BN + n);
          if (channel_normal) y = y * Vec::loadu(channel_scales + n);
          if (final_scale != 1.0f) y = y * weight_scale;
          // Preserve fused bias rounding for already-scaled accumulations.
          if (b && final_scale == 1.0f) {
            y = at::vec::fmadd(y, activation_scale, Vec::loadu(b + n));
          } else {
            y = y * activation_scale;
            if (b) y = y + Vec::loadu(b + n);
          }
          y.store(total + m * BN + n);
        }
        if constexpr (down) {
          copy_stub(tile + m * BN, total + m * BN, BN);
          copy_mul_stub(output + ids[m] * OC + nb * BN,
              tile + m * BN, topk_weights[ids[m]], BN);
        } else if constexpr (std::is_same_v<output_t, float>) {
          std::memcpy(
              output + (offsets[mb] + m) * OC + nb * BN,
              total + m * BN,
              BN * sizeof(float));
        } else {
          copy_stub(output + (offsets[mb] + m) * OC + nb * BN, total + m * BN, BN);
        }
      }
    });
    if (used_brgemm) at::native::cpublas::brgemm_release();
  });
  TORCH_CHECK(!invalid_scale.load(std::memory_order_relaxed),
      "MXFP4 weight scales contain reserved E8M0 NaN encoding");
}

template <typename fp8_t, typename weight_t, typename scale_t>
void fused_experts_fp8_a8_kernel_impl(
    at::Tensor& output, const at::Tensor& input,
    const at::Tensor& packed_w1, const at::Tensor& packed_w2,
    const at::Tensor& As,
    const at::Tensor& w1s, const at::Tensor& w2s,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const float* topk_weights, const int32_t* sorted_ids,
    const int32_t* expert_ids, const int32_t* offsets,
    int64_t N, int64_t topk, int64_t num_tokens_post_pad,
    float alpha, float limit, CPUActMethod act_func,
    const std::optional<at::Tensor>& w1_block_metadata,
    const std::optional<at::Tensor>& w2_block_metadata,
    const std::optional<at::Tensor>& w1_channel_metadata,
    const std::optional<at::Tensor>& w2_channel_metadata) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N_VAL = block_size_n();
  constexpr bool compressed = std::is_same_v<weight_t, uint8_t>;
  const int64_t M = input.size(0), K = input.size(1);
  const int64_t num_thread = at::get_num_threads();
  const int64_t cache_width = div_up(std::max(K, N), FP8_WEIGHT_BLOCK) * FP8_WEIGHT_BLOCK;
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  bool has_brgemm = !compressed;
  if constexpr (compressed) {
    for (int64_t mb = 0; mb < MB; ++mb) {
      if (offsets[mb + 1] - offsets[mb] > 4) {
        has_brgemm = true;
        break;
      }
    }
  }
  const auto* fp8_input = input.data_ptr<fp8_t>();
  const auto* activation_scales = As.data_ptr<float>();
  const int64_t routed_rows = M * topk;
  const int64_t intermediate_bytes = compressed
      ? routed_rows * (3 * N * sizeof(float) + K * sizeof(at::BFloat16))
      : routed_rows * (3 * N + K) * sizeof(at::BFloat16);
  const int64_t a_bytes = num_thread * BLOCK_M * cache_width;
  const int64_t b_bytes = num_thread * (compressed && has_brgemm ? MAX_CACHE_BLOCK_SIZE : 0) * BLOCK_N_VAL * cache_width;
  const int64_t c_bytes = num_thread * 2 * BLOCK_M * BLOCK_N_VAL * sizeof(float);
  const int64_t s_bytes = num_thread * (compressed && has_brgemm && (!w1_block_metadata || !w2_block_metadata) ? MAX_CACHE_BLOCK_SIZE : 0) *
      (cache_width / FP8_WEIGHT_BLOCK) * sizeof(float);
  auto arena = at::empty({intermediate_bytes + a_bytes + b_bytes + c_bytes + s_bytes}, input.options().dtype(at::kByte));
  auto* arena_data = arena.data_ptr<uint8_t>();
  at::BFloat16* ic0_bf16 = nullptr;
  at::BFloat16* ic1_bf16 = nullptr;
  at::BFloat16* ic2 = nullptr;
  float* ic0_fp32 = nullptr;
  float* ic1_fp32 = nullptr;
  if constexpr (compressed) {
    ic0_fp32 = reinterpret_cast<float*>(arena_data);
    ic1_fp32 = ic0_fp32 + routed_rows * 2 * N;
    ic2 = reinterpret_cast<at::BFloat16*>(ic1_fp32 + routed_rows * N);
  } else {
    ic0_bf16 = reinterpret_cast<at::BFloat16*>(arena_data);
    ic1_bf16 = ic0_bf16 + routed_rows * 2 * N;
    ic2 = ic1_bf16 + routed_rows * N;
  }
  auto* A_tmp = reinterpret_cast<fp8_t*>(arena_data + intermediate_bytes);
  auto* B_tmp = A_tmp + a_bytes;
  auto* C_tmp = reinterpret_cast<float*>(B_tmp + b_bytes);
  auto* S_tmp = s_bytes ? C_tmp + c_bytes / sizeof(float) : nullptr;
  // stage 1: intermediate_cache0 = hidden_states @ w1
  if constexpr (compressed) {
    fp8_moe_project<fp8_t, weight_t, scale_t, false, float>(
        ic0_fp32, fp8_input, activation_scales,
        nullptr,
        packed_w1.data_ptr<weight_t>(), w1s.data_ptr<scale_t>(), w1_bias ? w1_bias->data_ptr<float>() : nullptr,
        w1_block_metadata ? w1_block_metadata->data_ptr<float>() : nullptr,
        w1_channel_metadata ? w1_channel_metadata->data_ptr<float>() : nullptr,
        A_tmp, B_tmp, C_tmp, S_tmp,
        topk_weights, sorted_ids, expert_ids, offsets, 2 * N, K, topk, MB, cache_width);
  } else {
    fp8_moe_project<fp8_t, weight_t, scale_t, false>(
        ic0_bf16, fp8_input, activation_scales,
        nullptr,
        packed_w1.data_ptr<weight_t>(), w1s.data_ptr<scale_t>(), w1_bias ? w1_bias->data_ptr<float>() : nullptr,
        w1_block_metadata ? w1_block_metadata->data_ptr<float>() : nullptr,
        w1_channel_metadata ? w1_channel_metadata->data_ptr<float>() : nullptr,
        A_tmp, B_tmp, C_tmp, S_tmp,
        topk_weights, sorted_ids, expert_ids, offsets, 2 * N, K, topk, MB, cache_width);
  }
  // stage 1.5: activation and gate/up multiplication
  at::parallel_for(0, routed_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      if constexpr (compressed) {
        activation_stub<float, float>(
            ic1_fp32 + m * N, ic0_fp32 + m * 2 * N, N,
            act_func, alpha, limit);
      } else {
        activation_stub<at::BFloat16, at::BFloat16>(
            ic1_bf16 + m * N, ic0_bf16 + m * 2 * N, N,
            act_func, alpha, limit);
      }
    }
  });
  at::Tensor quantized, scales;
  if (has_brgemm) {
    at::Tensor activated;
    if constexpr (compressed) {
      activated = at::from_blob(ic1_fp32, {routed_rows, N}, input.options().dtype(at::kFloat));
    } else {
      activated = at::from_blob(ic1_bf16, {routed_rows, N}, input.options().dtype(at::kBFloat16));
    }
    std::tie(quantized, scales) = quantize_fp8e4m3_rows(activated);
  }
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  if constexpr (compressed) {
    fp8_moe_project<fp8_t, weight_t, scale_t, true>(
        ic2, quantized.defined() ? quantized.data_ptr<fp8_t>() : nullptr,
        scales.defined() ? scales.data_ptr<float>() : nullptr,
        ic1_fp32,
        packed_w2.data_ptr<weight_t>(), w2s.data_ptr<scale_t>(), w2_bias ? w2_bias->data_ptr<float>() : nullptr,
        w2_block_metadata ? w2_block_metadata->data_ptr<float>() : nullptr,
        w2_channel_metadata ? w2_channel_metadata->data_ptr<float>() : nullptr,
        A_tmp, B_tmp, C_tmp, S_tmp,
        topk_weights, sorted_ids, expert_ids, offsets, K, N, topk, MB, cache_width);
  } else {
    fp8_moe_project<fp8_t, weight_t, scale_t, true>(
        ic2, quantized.defined() ? quantized.data_ptr<fp8_t>() : nullptr,
        scales.defined() ? scales.data_ptr<float>() : nullptr, nullptr,
        packed_w2.data_ptr<weight_t>(), w2s.data_ptr<scale_t>(), w2_bias ? w2_bias->data_ptr<float>() : nullptr,
        w2_block_metadata ? w2_block_metadata->data_ptr<float>() : nullptr,
        w2_channel_metadata ? w2_channel_metadata->data_ptr<float>() : nullptr,
        A_tmp, B_tmp, C_tmp, S_tmp,
        topk_weights, sorted_ids, expert_ids, offsets, K, N, topk, MB, cache_width);
  }
  // stage 3: output = sum(ic2, dim=topk)  [M, K]
  auto* output_data = output.data_ptr<at::BFloat16>();
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output_data + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

}  // namespace

at::Tensor mxfp4_moe_channel_metadata(const at::Tensor& scales) {
  TORCH_CHECK(scales.device().is_cpu() && scales.is_contiguous(), "MXFP4 scales must be contiguous CPU tensors");
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.dim() == 3, "MXFP4 scales must have uint8 shape [E, OC, IC/32]");
  constexpr int64_t BN = block_size_n();
  const int64_t E = scales.size(0), OC = scales.size(1), groups = scales.size(2);
  TORCH_CHECK(OC % BN == 0 && groups > 0, "MXFP4 scales require aligned outputs and nonempty input");
  auto metadata = at::empty({E, OC / BN, 3, BN}, scales.options().dtype(at::kFloat));
  const auto* source = scales.data_ptr<uint8_t>();
  auto* out = metadata.data_ptr<float>();
  std::atomic<bool> invalid_scale{false};
  at::parallel_for(0, E * OC / BN, 1, [&](int64_t begin, int64_t end) {
    for (int64_t p = begin; p < end; ++p) {
      for (int64_t n = 0; n < BN; ++n) {
        int hi = 0, lo = 255;
        for (int64_t g = 0; g < groups; ++g) {
          const int s = source[(p * groups + g) * BN + n];
          hi = std::max(hi, s);
          lo = std::min(lo, s);
        }
        if (hi == 255) invalid_scale.store(true, std::memory_order_relaxed);
        float* panel = out + p * 3 * BN;
        panel[n] = std::ldexp(1.0f, hi - 133);
        panel[BN + n] = hi;
        panel[2 * BN + n] = hi - lo <= 11;
      }
    }
  });
  TORCH_CHECK(!invalid_scale.load(std::memory_order_relaxed), "MXFP4 weight scales contain reserved E8M0 NaN encoding");
  return metadata;
}

at::Tensor mxfp4_moe_scale_metadata(const at::Tensor& scales) {
  TORCH_CHECK(scales.device().is_cpu() && scales.is_contiguous(), "MXFP4 scales must be contiguous CPU tensors");
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.dim() == 3, "MXFP4 scales must have uint8 shape [E, OC, IC/32]");
  constexpr int64_t BN = block_size_n();
  const int64_t E = scales.size(0), OC = scales.size(1), IC = scales.size(2) * 32;
  TORCH_CHECK(OC % BN == 0, "MXFP4 scales require 32-element output alignment");
  const int64_t stripes = div_up(OC, FP8_WEIGHT_BLOCK), groups = div_up(IC, FP8_WEIGHT_BLOCK);
  auto metadata = at::empty({E, stripes, groups, 3}, scales.options().dtype(at::kFloat));
  const auto* source = scales.data_ptr<uint8_t>();
  auto* out = metadata.data_ptr<float>();
  std::atomic<bool> invalid_scale{false};
  at::parallel_for(0, E * stripes, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t expert = i / stripes, stripe = i % stripes;
      const int64_t panels = std::min(int64_t(4), OC / BN - stripe * 4);
      const auto* s = source + expert * OC * (IC / 32) + stripe * FP8_WEIGHT_BLOCK * (IC / 32);
      for (int64_t g = 0; g < groups; ++g) {
        const int64_t start = g * FP8_WEIGHT_BLOCK;
        const int64_t count = std::min(FP8_WEIGHT_BLOCK, IC - start);
        const auto [hi, lo] = mxfp4_scale_range(s + (start / 32) * BN, panels, IC, (count / 32) * BN);
        if (hi == 255) {
          invalid_scale.store(true, std::memory_order_relaxed);
          continue;
        }
        float* block = out + (i * groups + g) * 3;
        block[0] = std::ldexp(1.0f, int(hi) - 133);
        block[1] = hi;
        block[2] = int(hi) - int(lo) <= 11;
      }
    }
  });
  TORCH_CHECK(!invalid_scale.load(std::memory_order_relaxed), "MXFP4 weight scales contain reserved E8M0 NaN encoding");
  return metadata;
}

void fused_experts_fp8_a8(
    at::Tensor& output, const at::Tensor& input,
    const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& activation_scales,
    const at::Tensor& w1s, const at::Tensor& w2s,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const float* topk_weights, const int32_t* sorted_ids,
    const int32_t* expert_ids, const int32_t* offsets,
    int64_t N, int64_t topk, int64_t num_tokens_post_pad,
    float alpha, float limit, CPUActMethod act_func,
    const std::optional<at::Tensor>& w1_block_metadata,
    const std::optional<at::Tensor>& w2_block_metadata,
    const std::optional<at::Tensor>& w1_channel_metadata,
    const std::optional<at::Tensor>& w2_channel_metadata) {
  if (w1.scalar_type() == at::kByte) {
    fused_experts_fp8_a8_kernel_impl<at::Float8_e4m3fn, uint8_t, uint8_t>(
        output, input, w1, w2, activation_scales, w1s, w2s, w1_bias, w2_bias,
        topk_weights, sorted_ids, expert_ids, offsets, N, topk, num_tokens_post_pad, alpha, limit, act_func, w1_block_metadata, w2_block_metadata, w1_channel_metadata, w2_channel_metadata);
  } else {
    fused_experts_fp8_a8_kernel_impl<at::Float8_e4m3fn, at::Float8_e4m3fn, float>(
        output, input, w1, w2, activation_scales, w1s, w2s, w1_bias, w2_bias,
        topk_weights, sorted_ids, expert_ids, offsets, N, topk, num_tokens_post_pad, alpha, limit, act_func, w1_block_metadata, w2_block_metadata, w1_channel_metadata, w2_channel_metadata);
  }
}
