// vLLM-native: DeepSeek-V4 sparse indexer Q-side RoPE + FP8 quant kernel.
// Ports the math of vLLM's own GPU/Triton reference
// (`_fused_indexer_q_rope_quant_kernel`, common/ops/fused_indexer_q.py) --
// not a port of SGLang's own indexer-Q kernel, since that applies RoPE plus
// a 128-point Hadamard rotation the real DeepSeek-V4-Flash checkpoint
// doesn't use (verified: it applies RoPE only, no Hadamard step).
//
// Per (token, head): GPT-J interleaved RoPE on the last `rot_dim` dims;
// leading NoPE dims pass through unrotated. RoPE-rotated values are bf16-
// round-tripped before the absmax reduction (matching the K-side compressor
// kernel), NoPE values are not. One UE8M0 scale per (token, head) covers the
// full head_dim; since the FP8 Q tensor has no companion scale tensor, the
// scale is folded into `index_weights_out` instead:
//   weights_out = weights * q_scale * softmax_scale * head_scale
// MXFP4 (`use_fp4=True`) has per-block scales that can't be folded this way
// and stays on triton-cpu -- out of scope here.
//
// AVX512 path: NoPE copy+absmax and the FP8 scale/clamp use plain 16-wide
// vector ops (final FP8 cast stays scalar, matching SGLang's own
// `act_quant_cpu_impl`). The RoPE rotation vectorizes the interleaved
// (x_even, x_odd) pairs via in-register permutes (`_mm512_permute_ps` /
// `_mm512_permutexvar_ps`), the same technique compressor.cpp's
// `apply_gptj_rope_inplace` and store_cache.cpp's
// `inverse_gptj_rope_o_proj_cpu` use, since gather/scatter is comparatively
// expensive on AVX512.

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"
#include "vec.h"

#if !defined(CPU_CAPABILITY_AVX512)
static_assert(
    false,
    "indexer.cpp is only ever compiled into the AVX512+AMX-flagged "
    "_C target (see cmake/cpu_extension.cmake's VLLM_EXT_SRC_SGL) and "
    "relies on that unconditionally -- CPU_CAPABILITY_AVX512 must be "
    "defined here.");
#endif

namespace {

constexpr float kFp8Max = 448.0f;

#if defined(CPU_CAPABILITY_AVX512)

// fp32 -> bf16 -> fp32 round trip via the AVX512-BF16 conversion
// instruction, numerically matching `static_cast<float>(at::BFloat16(x))`
// (round-to-nearest-even).
inline __m512 bf16_round_trip_avx512(__m512 x) {
  return CVT_BF16_TO_FP32((__m256i)(_mm512_cvtneps_pbh(x)));
}

inline __m512 abs_ps(__m512 x) {
  return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);
}

#endif  // CPU_CAPABILITY_AVX512

void fused_indexer_q_rope_quant_impl(
    const int64_t* __restrict__ positions,  // [num_tokens]
    const float* __restrict__ index_q,      // [num_tokens, num_heads, head_dim]
    int64_t index_q_stride0, int64_t index_q_stride1,
    const float* __restrict__ cos_sin_cache,  // [max_pos, rot_dim]
    int64_t cos_sin_stride, int64_t half_rot_dim,
    at::Float8_e4m3fn* __restrict__ index_q_fp8,  // [num_tokens, num_heads,
                                                  // head_dim]
    int64_t index_q_fp8_stride0, int64_t index_q_fp8_stride1, int64_t head_dim,
    const float* __restrict__ index_weights,  // [num_tokens, num_heads]
    int64_t index_weights_stride, float softmax_scale, float head_scale,
    float* __restrict__ index_weights_out,  // [num_tokens, num_heads]
    int64_t index_weights_out_stride, int64_t num_tokens, int64_t num_heads) {
  const int64_t rot_dim = 2 * half_rot_dim;
  const int64_t nope_dim = head_dim - rot_dim;

  at::parallel_for(
      0, num_tokens * num_heads, 0, [&](int64_t begin, int64_t end) {
        std::vector<float> buf(head_dim);
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t t = idx / num_heads;
          const int64_t h = idx % num_heads;

          const int64_t pos = positions[t];
          const float* const cos = cos_sin_cache + pos * cos_sin_stride;
          const float* const sin = cos + half_rot_dim;

          const float* const q =
              index_q + t * index_q_stride0 + h * index_q_stride1;

          float amax = 1e-4f;
#if defined(CPU_CAPABILITY_AVX512)
          {
            __m512 vamax = _mm512_set1_ps(amax);
            for (int64_t d = 0; d < nope_dim; d += 16) {
              __m512 v = _mm512_loadu_ps(q + d);
              _mm512_storeu_ps(&buf[d], v);
              vamax = _mm512_max_ps(vamax, abs_ps(v));
            }
            amax = _mm512_reduce_max_ps(vamax);
          }
#endif

          const float* const q_rot = q + nope_dim;
#if defined(CPU_CAPABILITY_AVX512)
          {
            // Permute-based (no gather/scatter), same technique as
            // compressor.cpp's apply_gptj_rope_inplace / store_cache.cpp's
            // inverse_gptj_rope_o_proj_cpu: `_mm512_permute_ps` swaps each
            // pair's (even, odd) lanes in one shuffle, and cos/sin -- loaded as
            // 8 contiguous values -- are duplicated pairwise via
            // `_mm512_permutexvar_ps` to align with that layout. Two batches of
            // 8 pairs per iteration (16 total, matching the old gather step).
            // Unlike the RoPE kernels above, the result here stays in its
            // original interleaved layout, so no un-permute/scatter is needed
            // before the bf16 round-trip and store.
            const __m512i dup_idx = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3,
                                                     3, 2, 2, 1, 1, 0, 0);
            // Forward GPT-J rotation negates the sin broadcast on the even
            // (real) lane of each pair; the odd (imag) lane keeps sin's sign.
            const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
                0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0,
                (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0,
                (int)0x80000000, 0, (int)0x80000000));
            __m512 vamax_rope = _mm512_setzero_ps();
            for (int64_t i = 0; i < half_rot_dim; i += 16) {
              const float* const base = q_rot + 2 * i;
              float* const dst = buf.data() + nope_dim + 2 * i;

              __m512 xv0 = _mm512_loadu_ps(base);
              __m512 cos_b0 = _mm512_permutexvar_ps(
                  dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cos + i)));
              __m512 sin_b0 = _mm512_permutexvar_ps(
                  dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sin + i)));
              __m512 out0 = _mm512_fmadd_ps(
                  xv0, cos_b0,
                  _mm512_mul_ps(_mm512_permute_ps(xv0, 0xB1),
                                _mm512_xor_ps(sin_b0, sign_mask)));
              // Match reference numerics: fp32 -> bf16 -> fp32 before the
              // absmax.
              out0 = bf16_round_trip_avx512(out0);
              _mm512_storeu_ps(dst, out0);
              vamax_rope = _mm512_max_ps(vamax_rope, abs_ps(out0));

              __m512 xv1 = _mm512_loadu_ps(base + 16);
              __m512 cos_b1 = _mm512_permutexvar_ps(
                  dup_idx,
                  _mm512_castps256_ps512(_mm256_loadu_ps(cos + i + 8)));
              __m512 sin_b1 = _mm512_permutexvar_ps(
                  dup_idx,
                  _mm512_castps256_ps512(_mm256_loadu_ps(sin + i + 8)));
              __m512 out1 = _mm512_fmadd_ps(
                  xv1, cos_b1,
                  _mm512_mul_ps(_mm512_permute_ps(xv1, 0xB1),
                                _mm512_xor_ps(sin_b1, sign_mask)));
              out1 = bf16_round_trip_avx512(out1);
              _mm512_storeu_ps(dst + 16, out1);
              vamax_rope = _mm512_max_ps(vamax_rope, abs_ps(out1));
            }
            amax = std::max(amax, _mm512_reduce_max_ps(vamax_rope));
          }
#endif

          const float exponent = std::ceil(std::log2(amax / kFp8Max));
          const float scale = std::exp2(exponent);
          const float inv_scale = 1.0f / scale;

          at::Float8_e4m3fn* const out =
              index_q_fp8 + t * index_q_fp8_stride0 + h * index_q_fp8_stride1;
#if defined(CPU_CAPABILITY_AVX512)
          {
            const __m512 vinv = _mm512_set1_ps(inv_scale);
            const __m512 vmax = _mm512_set1_ps(kFp8Max);
            const __m512 vmin = _mm512_set1_ps(-kFp8Max);
            alignas(64) float scaled_buf[16];
            for (int64_t d2 = 0; d2 < head_dim; d2 += 16) {
              __m512 v = _mm512_loadu_ps(&buf[d2]);
              v = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v, vinv), vmin),
                                vmax);
              _mm512_store_ps(scaled_buf, v);
              for (int64_t j = 0; j < 16; ++j)
                out[d2 + j] = at::Float8_e4m3fn(scaled_buf[j]);
            }
          }
#endif

          float w = index_weights[t * index_weights_stride + h];
          w *= scale;
          w *= softmax_scale;
          w *= head_scale;
          index_weights_out[t * index_weights_out_stride + h] = w;
        }
      });
}

}  // anonymous namespace

void fused_indexer_q_rope_quant_cpu(at::Tensor& positions, at::Tensor& index_q,
                                    at::Tensor& index_q_cos_sin_cache,
                                    at::Tensor& index_q_fp8,
                                    at::Tensor& index_weights,
                                    double index_weights_softmax_scale,
                                    double index_weights_head_scale,
                                    at::Tensor& index_weights_out) {
  TORCH_CHECK(
      positions.dim() == 1 && positions.scalar_type() == at::kLong,
      "fused_indexer_q_rope_quant_cpu: positions must be a 1D int64 tensor");
  TORCH_CHECK(positions.is_contiguous(),
              "fused_indexer_q_rope_quant_cpu: positions must be contiguous");
  TORCH_CHECK(index_q.dim() == 3,
              "fused_indexer_q_rope_quant_cpu: index_q must be 3D");
  TORCH_CHECK(index_q.scalar_type() == at::kFloat,
              "fused_indexer_q_rope_quant_cpu: index_q must be float32 "
              "(convert once in Python)");
  TORCH_CHECK(index_q.stride(2) == 1,
              "fused_indexer_q_rope_quant_cpu: index_q must be contiguous in "
              "the head_dim axis");
  TORCH_CHECK(
      index_q_cos_sin_cache.dim() == 2 && index_q_cos_sin_cache.stride(1) == 1,
      "fused_indexer_q_rope_quant_cpu: index_q_cos_sin_cache must be a "
      "contiguous 2D tensor");
  TORCH_CHECK(
      index_q_cos_sin_cache.scalar_type() == at::kFloat,
      "fused_indexer_q_rope_quant_cpu: index_q_cos_sin_cache must be float32 "
      "(convert once in Python)");
  TORCH_CHECK(index_q_fp8.sizes() == index_q.sizes() &&
                  index_q_fp8.scalar_type() == at::kFloat8_e4m3fn &&
                  index_q_fp8.stride(2) == 1,
              "fused_indexer_q_rope_quant_cpu: index_q_fp8 must match "
              "index_q's shape, be "
              "float8_e4m3fn, and contiguous in the head_dim axis");
  TORCH_CHECK(index_weights.dim() == 2 && index_weights.stride(1) == 1,
              "fused_indexer_q_rope_quant_cpu: index_weights must be a "
              "contiguous-in-head 2D tensor");
  TORCH_CHECK(index_weights.scalar_type() == at::kFloat,
              "fused_indexer_q_rope_quant_cpu: index_weights must be float32 "
              "(convert once in Python)");
  TORCH_CHECK(index_weights_out.sizes() == index_weights.sizes() &&
                  index_weights_out.scalar_type() == at::kFloat &&
                  index_weights_out.stride(1) == 1,
              "fused_indexer_q_rope_quant_cpu: index_weights_out must match "
              "index_weights' shape, "
              "be float32, and contiguous in the head axis");
  TORCH_CHECK(positions.device().is_cpu() && index_q.device().is_cpu() &&
                  index_q_cos_sin_cache.device().is_cpu() &&
                  index_q_fp8.device().is_cpu() &&
                  index_weights.device().is_cpu() &&
                  index_weights_out.device().is_cpu(),
              "fused_indexer_q_rope_quant_cpu: all inputs must be CPU tensors");

  const int64_t num_tokens = positions.size(0);
  const int64_t num_heads = index_q.size(1);
  const int64_t head_dim = index_q.size(2);
  const int64_t half_rot_dim = index_q_cos_sin_cache.size(-1) / 2;
  // Only ever called with head_dim=128, rot_dim=64 (DeepseekV4's sparse
  // indexer Q side); the AVX512 loops below process 16 elements/pairs per
  // iteration with no scalar remainder, so require even divisibility.
  TORCH_CHECK(head_dim % 16 == 0 && (head_dim - 2 * half_rot_dim) % 16 == 0 &&
                  half_rot_dim % 16 == 0,
              "fused_indexer_q_rope_quant_cpu: head_dim, head_dim-rot_dim, and "
              "rot_dim/2 "
              "must be multiples of 16");

  fused_indexer_q_rope_quant_impl(
      positions.data_ptr<int64_t>(), index_q.data_ptr<float>(),
      index_q.stride(0), index_q.stride(1),
      index_q_cos_sin_cache.data_ptr<float>(), index_q_cos_sin_cache.stride(0),
      half_rot_dim, index_q_fp8.data_ptr<at::Float8_e4m3fn>(),
      index_q_fp8.stride(0), index_q_fp8.stride(1), head_dim,
      index_weights.data_ptr<float>(), index_weights.stride(0),
      static_cast<float>(index_weights_softmax_scale),
      static_cast<float>(index_weights_head_scale),
      index_weights_out.data_ptr<float>(), index_weights_out.stride(0),
      num_tokens, num_heads);
}
