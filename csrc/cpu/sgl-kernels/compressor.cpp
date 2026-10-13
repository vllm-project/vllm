// Adapted from
// https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/cpu/compressor.cpp
//
// DeepSeek-V4 compressor: softmax-weighted pooling over a sliding window of
// raw KV/score state, RMSNorm, FP8/UE8M0 quant (NoPE) + GPT-J RoPE (RoPE),
// paged-insert into the fp8_ds_mla cache layout (see store_cache.cpp).
// Covers both the head_dim=512 attention-compressor path
// (`compress_norm_rope_store_cpu`) and the head_dim=128 indexer-compressor
// path (`compress_norm_rope_store_indexer_cpu`, distinct cache layout --
// see its own comment below).
//
// Differences from upstream:
// - Upstream splits decode (`compress_decode_cpu`) from prefill/extend
//   (eager `compress_extend_separate`); this port stays unified across
//   decode/prefill/extend, one program per token, matching vLLM's own
//   GPU/Triton reference (`_fused_kv_compress_norm_rope_insert_sparse_attn`,
//   common/ops/fused_compress_quant_cache.py).
// - Window-gather indices arrive pre-resolved to flat state-cache row ids
//   from the Python caller (mirroring flash_mla.cpp), not looked up via
//   block_table in-kernel.
// - The state cache is pooled with the main KV cache's physical pages
//   (CompressorStateCache in compressor.py), so its per-block byte stride
//   is page-aligned, not `block_size * row_width` -- row ids are addressed
//   as (block_idx, pos_in_block) against the tensor's real block stride,
//   never by flattening to a 2D view.
//
// clang-format off

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "common.h"
#include "vec.h"

#if !defined(CPU_CAPABILITY_AVX512)
static_assert(false,
              "compressor.cpp is only ever compiled into the AVX512+AMX-flagged "
              "_C target (see cmake/cpu_extension.cmake's VLLM_EXT_SRC_SGL) and "
              "relies on that unconditionally -- CPU_CAPABILITY_AVX512 must be "
              "defined here.");
#endif

namespace {

using fVec = at::vec::Vectorized<float>;

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHeadDim = kNopeDim + kRopeDim;
constexpr int64_t kQuantBlock = 64;
constexpr int64_t kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int64_t kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8Min = -448.0f;
constexpr float kQuantEps = 1e-4f;

// In-place interleaved (GPT-J-style) RoPE: x[2i]/x[2i+1] are one (even, odd)
// pair, cos[i]/sin[i] its angle. Adapted from SGLang's `rope.cpp`
// (`apply_rotary_emb_row`), which avoids gather/scatter (slow on AVX512) by
// broadcasting cos/sin across each pair via in-register permutes instead:
// `_mm512_permute_ps(xv, 0xB1)` swaps the (even, odd) lanes within every
// 2-float pair in one shuffle, and cos/sin -- loaded as 8 contiguous values
// -- are duplicated pairwise into the same layout via `_mm512_permutexvar_ps`.
// SGLang's version reads cos/sin pre-interleaved with x (`[c0,s0,c1,s1,...]`)
// since its freqs tensor is laid out that way; ours are two separate flat
// arrays (`cos[i]`, `sin[i]`, matching store_cache.cpp's convention), hence
// the extra broadcast-duplicate permute here instead of a plain load.
// Both call sites in this file pass half_dim=32 (kRopeDim/2, kIndexerRopeDim/2
// are both 64/2), an exact multiple of 8, so this file's build (always AVX512,
// see file header) never needs a scalar remainder.
inline void apply_gptj_rope_inplace(float* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin, int64_t half_dim) {
  // Duplicates each of 8 source lanes into 2 adjacent output lanes:
  // [c0,c1,...,c7] -> [c0,c0,c1,c1,...,c7,c7].
  const __m512i dup_idx = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  // Forward GPT-J rotation negates the sin broadcast on the even (real)
  // lane of each pair; the odd (imag) lane keeps sin's sign.
  const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
      0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000,
      0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000));
  for (int64_t i = 0; i + 8 <= half_dim; i += 8) {
    __m512 xv = _mm512_loadu_ps(x + 2 * i);
    __m512 xv_swapped = _mm512_permute_ps(xv, 0xB1);
    __m512 cos_b = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cos + i)));
    __m512 sin_b = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sin + i)));
    __m512 sin_signed = _mm512_xor_ps(sin_b, sign_mask);
    __m512 out = _mm512_fmadd_ps(xv, cos_b, _mm512_mul_ps(xv_swapped, sin_signed));
    _mm512_storeu_ps(x + 2 * i, out);
  }
}

// AVX512 tile-quantization, adapted verbatim from SGLang's `store_cache.cpp`
// (`quant_to_nope_fp8_rope_bf16_pack_cpu`'s
// `cvt_fp32x16_to_fp8x16`/`quantize_tile_avx512`) -- SGLang itself keeps
// `compress_decode_cpu` fp32-output-only and calls this quant step as a
// separate op; vLLM's compressor kernel stays fused (per this file's header
// comment), so the same AVX512 tile-quant math is inlined here instead.
#if defined(CPU_CAPABILITY_AVX512)

// Vectorized float32x16 -> fp8_e4m3fn x16 using AVX512 bit manipulation.
// Implements RNE (round-to-nearest-even). Flushes subnormal fp8 to zero.
// fp8_e4m3fn: sign(1), exp(4, bias=7), mant(3), no inf/nan special values.
// Normal: exp_field in [1..15], value = 2^(exp-7) * (1 + mant/8)
// Input assumed clamped to [-448, 448].
inline __m128i cvt_fp32x16_to_fp8x16(__m512 input) {
  __m512i bits = _mm512_castps_si512(input);

  __m512i signs = _mm512_srli_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x80000000)), 24);
  __m512i abs_bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFFFFFF));
  __m512i f32_exp = _mm512_srli_epi32(abs_bits, 23);
  __m512i f32_mant = _mm512_and_si512(abs_bits, _mm512_set1_epi32(0x7FFFFF));
  __m512i fp8_exp = _mm512_sub_epi32(f32_exp, _mm512_set1_epi32(120));
  __m512i mant3 = _mm512_srli_epi32(f32_mant, 20);

  __m512i round_bit = _mm512_and_si512(_mm512_srli_epi32(f32_mant, 19), _mm512_set1_epi32(1));
  __m512i sticky_bits = _mm512_and_si512(f32_mant, _mm512_set1_epi32(0x7FFFF));
  __mmask16 has_sticky = _mm512_cmpneq_epi32_mask(sticky_bits, _mm512_setzero_si512());

  __m512i lsb = _mm512_and_si512(mant3, _mm512_set1_epi32(1));
  __m512i sticky_or_lsb = _mm512_or_si512(_mm512_maskz_mov_epi32(has_sticky, _mm512_set1_epi32(1)), lsb);
  __m512i do_round = _mm512_and_si512(round_bit, sticky_or_lsb);
  mant3 = _mm512_add_epi32(mant3, do_round);

  __mmask16 mant_ovf = _mm512_cmpeq_epi32_mask(mant3, _mm512_set1_epi32(8));
  mant3 = _mm512_mask_mov_epi32(mant3, mant_ovf, _mm512_setzero_si512());
  fp8_exp = _mm512_mask_add_epi32(fp8_exp, mant_ovf, fp8_exp, _mm512_set1_epi32(1));

  fp8_exp = _mm512_min_epi32(fp8_exp, _mm512_set1_epi32(15));

  __m512i result = _mm512_or_si512(_mm512_slli_epi32(fp8_exp, 3), mant3);

  __mmask16 is_subnormal_or_zero = _mm512_cmplt_epi32_mask(f32_exp, _mm512_set1_epi32(121));
  result = _mm512_mask_mov_epi32(result, is_subnormal_or_zero, _mm512_setzero_si512());

  result = _mm512_or_si512(result, signs);

  return _mm512_cvtepi32_epi8(result);
}

// Process one 64-wide tile: find amax, compute a shared power-of-two scale,
// quantize to fp8. Returns the UE8M0 scale byte for the tile.
inline uint8_t quantize_tile_avx512(const at::BFloat16* __restrict__ src, uint8_t* __restrict__ dst) {
  __m256i bf16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
  __m256i bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
  __m256i bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
  __m256i bf16_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 48));

  __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_0), 16));
  __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_1), 16));
  __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_2), 16));
  __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_3), 16));

  const __m512i abs_mask_i = _mm512_set1_epi32(0x7FFFFFFF);
  __m512 abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask_i));
  __m512 abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask_i));
  __m512 abs2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask_i));
  __m512 abs3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask_i));

  __m512 max01 = _mm512_max_ps(abs0, abs1);
  __m512 max23 = _mm512_max_ps(abs2, abs3);
  __m512 max0123 = _mm512_max_ps(max01, max23);
  float amax = _mm512_reduce_max_ps(max0123);

  float scale = std::max(amax / kFp8Max, kQuantEps);
  float ceil_log2 = std::ceil(std::log2(scale));
  float scale_pow2 = std::exp2(ceil_log2);
  float scale_inv = 1.0f / scale_pow2;

  int exponent = static_cast<int>(ceil_log2);
  uint8_t scale_uint8 = static_cast<uint8_t>(exponent + 127);

  __m512 vinv = _mm512_set1_ps(scale_inv);
  __m512 s0 = _mm512_mul_ps(f0, vinv);
  __m512 s1 = _mm512_mul_ps(f1, vinv);
  __m512 s2 = _mm512_mul_ps(f2, vinv);
  __m512 s3 = _mm512_mul_ps(f3, vinv);

  __m512 vmax = _mm512_set1_ps(kFp8Max);
  __m512 vmin = _mm512_set1_ps(kFp8Min);
  s0 = _mm512_max_ps(_mm512_min_ps(s0, vmax), vmin);
  s1 = _mm512_max_ps(_mm512_min_ps(s1, vmax), vmin);
  s2 = _mm512_max_ps(_mm512_min_ps(s2, vmax), vmin);
  s3 = _mm512_max_ps(_mm512_min_ps(s3, vmax), vmin);

  __m128i fp8_0 = cvt_fp32x16_to_fp8x16(s0);
  __m128i fp8_1 = cvt_fp32x16_to_fp8x16(s1);
  __m128i fp8_2 = cvt_fp32x16_to_fp8x16(s2);
  __m128i fp8_3 = cvt_fp32x16_to_fp8x16(s3);

  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), fp8_0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), fp8_1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 32), fp8_2);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 48), fp8_3);

  // Patch subnormal-magnitude lanes: `cvt_fp32x16_to_fp8x16` flushes
  // anything below the smallest fp8_e4m3fn normal (2^-6) to zero, but
  // vLLM's dequant path relies on `at::Float8_e4m3fn`'s real subnormal
  // encoding -- redo just those rare lanes with the exact scalar
  // conversion to stay consistent with the read side.
  alignas(64) float scaled[kQuantBlock];
  _mm512_store_ps(scaled, s0);
  _mm512_store_ps(scaled + 16, s1);
  _mm512_store_ps(scaled + 32, s2);
  _mm512_store_ps(scaled + 48, s3);
  for (int64_t j = 0; j < kQuantBlock; ++j) {
    const float av = std::abs(scaled[j]);
    if (av != 0.f && av < 0x1p-6f) {
      dst[j] = at::Float8_e4m3fn(scaled[j]).x;
    }
  }

  return scale_uint8;
}

#endif  // CPU_CAPABILITY_AVX512

void save_partial_states_impl(
    float* __restrict__ state_cache,          // [num_blocks, block_stride]
    int64_t block_stride,                     // element stride between blocks
    int64_t block_size,                       // rows (slots) per block
    const float* __restrict__ kv,             // [num_tokens, state_width]
    int64_t kv_row_stride,                    // element stride between kv rows
    const float* __restrict__ score,          // [num_tokens, state_width]
    int64_t score_row_stride,                 // element stride between score rows
    const float* __restrict__ ape,            // [compress_ratio, state_width]
    const int64_t* __restrict__ positions,    // [num_tokens]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    int64_t state_width,
    int64_t compress_ratio,
    int64_t num_tokens) {
  const int64_t row_width = 2 * state_width;
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t slot_id = slot_mapping[t];
      if (slot_id < 0) continue;

      const int64_t block_idx = slot_id / block_size;
      const int64_t pos_in_block = slot_id % block_size;
      float* const row = state_cache + block_idx * block_stride + pos_in_block * row_width;
      const float* const kv_t = kv + t * kv_row_stride;
      // state_width is always a multiple of 16 for this model (512/1024 for
      // the main compressor, 128/256 for the indexer compressor).
      for (int64_t i = 0; i + 16 <= state_width; i += 16) {
        fVec::loadu(kv_t + i).store(row + i);
      }

      const int64_t position = positions[t];
      const float* const ape_row = ape + (position % compress_ratio) * state_width;
      const float* const score_t = score + t * score_row_stride;
      float* const score_out = row + state_width;
      for (int64_t i = 0; i + 16 <= state_width; i += 16) {
        (fVec::loadu(score_t + i) + fVec::loadu(ape_row + i)).store(score_out + i);
      }
    }
  });
}

void compress_norm_rope_store_impl(
    const float* __restrict__ state_cache,        // [num_blocks, state_block_stride]
    int64_t state_block_stride,                   // element stride between state blocks
    int64_t state_block_size,                     // rows (slots) per state block
    const int64_t* __restrict__ gather_slots,     // [num_tokens, window], -1 = invalid
    const int64_t* __restrict__ positions,        // [num_tokens]
    const int64_t* __restrict__ kv_slot_mapping,  // [num_tokens], -1 = skip
    const float* __restrict__ rms_norm_weight,    // [kHeadDim]
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache,      // [max_pos, kRopeDim]
    uint8_t* __restrict__ kv_cache,               // [num_blocks, cache_row_stride]
    int64_t cache_row_stride,
    int64_t kv_cache_block_size,
    int64_t compress_ratio,
    int64_t window,
    int64_t state_width,
    int64_t num_tokens) {
  const int64_t rope_half = kRopeDim / 2;
  const int64_t state_row_width = 2 * state_width;

  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    std::array<float, kHeadDim> max_v;
    std::array<float, kHeadDim> denom;
    std::array<float, kHeadDim> acc;
    std::array<float, kHeadDim> normed;
    std::array<at::BFloat16, kNopeDim> nope_bf16;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t position = positions[t];
      if ((position + 1) % compress_ratio != 0) continue;
      const int64_t kv_slot = kv_slot_mapping[t];
      if (kv_slot < 0) continue;

      std::fill(max_v.begin(), max_v.end(), -std::numeric_limits<float>::infinity());
      std::fill(denom.begin(), denom.end(), 0.f);
      std::fill(acc.begin(), acc.end(), 0.f);

      // Per-channel (not per-position) softmax over the window: each of the
      // kHeadDim columns is normalized independently across the window axis.
      for (int64_t w = 0; w < window; ++w) {
        const int64_t slot = gather_slots[t * window + w];
        if (slot < 0) continue;
        const int64_t block_idx = slot / state_block_size;
        const int64_t pos_in_block = slot % state_block_size;
        const int64_t head_offset = (w >= compress_ratio) ? kHeadDim : 0;
        const float* const score_row = state_cache + block_idx * state_block_stride +
            pos_in_block * state_row_width + head_offset + state_width;
        for (int64_t d = 0; d < kHeadDim; d += 16) {
          __m512 mx = _mm512_loadu_ps(max_v.data() + d);
          __m512 sv = _mm512_loadu_ps(score_row + d);
          _mm512_storeu_ps(max_v.data() + d, _mm512_max_ps(mx, sv));
        }
      }
      for (int64_t w = 0; w < window; ++w) {
        const int64_t slot = gather_slots[t * window + w];
        if (slot < 0) continue;
        const int64_t block_idx = slot / state_block_size;
        const int64_t pos_in_block = slot % state_block_size;
        const int64_t head_offset = (w >= compress_ratio) ? kHeadDim : 0;
        const float* const row = state_cache + block_idx * state_block_stride +
            pos_in_block * state_row_width + head_offset;
        const float* const score_row = row + state_width;
        for (int64_t d = 0; d < kHeadDim; d += 16) {
          fVec e_v = (fVec::loadu(score_row + d) - fVec::loadu(max_v.data() + d)).exp();
          (fVec::loadu(denom.data() + d) + e_v).store(denom.data() + d);
          (fVec::loadu(acc.data() + d) + e_v * fVec::loadu(row + d)).store(acc.data() + d);
        }
      }

      float sq_sum = 0.f;
      {
        fVec sq_sum_v(0.f);
        for (int64_t d = 0; d < kHeadDim; d += 16) {
          fVec v = fVec::loadu(acc.data() + d) / fVec::loadu(denom.data() + d);
          v.store(normed.data() + d);
          sq_sum_v = sq_sum_v + v * v;
        }
        sq_sum = vec_reduce_sum(sq_sum_v);
      }
      const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(kHeadDim) + rms_norm_eps);
      {
        fVec inv_rms_v(inv_rms);
        for (int64_t d = 0; d < kHeadDim; d += 16) {
          (fVec::loadu(normed.data() + d) * inv_rms_v * fVec::loadu(rms_norm_weight + d)).store(normed.data() + d);
        }
      }

      // bf16 round-trip feeds only the NoPE FP8 quantization (RoPE rotates
      // the un-rounded `normed` below, then casts once to bf16 at store
      // time for the separate bf16 rope region).
      for (int64_t d = 0; d < kNopeDim; d += 16)
        store_from_float_ext<at::BFloat16>(nope_bf16.data() + d, fVec::loadu(normed.data() + d));

      const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
      const float* const cos = cos_sin_cache + compressed_pos * kRopeDim;
      const float* const sin = cos + rope_half;
      apply_gptj_rope_inplace(normed.data() + kNopeDim, cos, sin, rope_half);

      const int64_t block_idx = kv_slot / kv_cache_block_size;
      const int64_t pos_in_block = kv_slot % kv_cache_block_size;
      uint8_t* const block_base = kv_cache + block_idx * cache_row_stride;
      uint8_t* const token_fp8_ptr = block_base + pos_in_block * kTokenDataBytes;
      uint8_t* const token_bf16_ptr = token_fp8_ptr + kNopeDim;
      uint8_t* const token_scale_ptr =
          block_base + kv_cache_block_size * kTokenDataBytes + pos_in_block * kScaleBytesPerToken;

      // AVX512 tile-quantization (SGLang's `quantize_tile_avx512`), one
      // 64-wide tile per iteration.
      for (int64_t qb = 0; qb < kNumQuantBlocks; ++qb) {
        const at::BFloat16* tile_src = nope_bf16.data() + qb * kQuantBlock;
        uint8_t* tile_dst = token_fp8_ptr + qb * kQuantBlock;
        token_scale_ptr[qb] = quantize_tile_avx512(tile_src, tile_dst);
      }
      token_scale_ptr[kNumQuantBlocks] = 0;

      auto* const out_rope = reinterpret_cast<at::BFloat16*>(token_bf16_ptr);
      for (int64_t ri = 0; ri < kRopeDim; ri += 16)
        store_from_float_ext<at::BFloat16>(out_rope + ri, fVec::loadu(normed.data() + kNopeDim + ri));
    }
  });
}

// Indexer compressor (head_dim=128): same softmax-pool + RMSNorm + GPT-J
// RoPE prologue as the head=512 path above, but a different cache layout:
// all 128 dims are FP8 in one combined quant block (no separate bf16 rope
// region, single raw fp32 scale per token) instead of 7 UE8M0 blocks over
// nope alone -- matching the paged indexer K-cache layout
// fp8_paged_mqa_logits_cpu/topk_transform_512_cpu (paged_mqa_logits.cpp/
// topk.cpp) read, and the GPU/Triton reference
// (`_fused_kv_compress_norm_rope_insert_indexer_attn`,
// common/ops/fused_compress_quant_cache.py) exactly.
constexpr int64_t kIndexerHeadDim = 128;
constexpr int64_t kIndexerRopeDim = 64;
constexpr int64_t kIndexerNopeDim = kIndexerHeadDim - kIndexerRopeDim;
constexpr int64_t kIndexerTokenBytes = kIndexerHeadDim;
constexpr int64_t kIndexerScaleBytes = 4;

void compress_norm_rope_store_indexer_impl(
    const float* __restrict__ state_cache,        // [num_blocks, state_block_stride]
    int64_t state_block_stride,
    int64_t state_block_size,
    const int64_t* __restrict__ gather_slots,     // [num_tokens, window], -1 = invalid
    const int64_t* __restrict__ positions,        // [num_tokens]
    const int64_t* __restrict__ kv_slot_mapping,  // [num_tokens], -1 = skip
    const float* __restrict__ rms_norm_weight,    // [kIndexerHeadDim]
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache,      // [max_pos, kIndexerRopeDim]
    uint8_t* __restrict__ kv_cache,               // [num_blocks, cache_row_stride]
    int64_t cache_row_stride,
    int64_t kv_cache_block_size,
    int64_t compress_ratio,
    int64_t window,
    int64_t state_width,
    int64_t num_tokens) {
  const int64_t rope_half = kIndexerRopeDim / 2;
  const int64_t state_row_width = 2 * state_width;

  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    std::array<float, kIndexerHeadDim> max_v;
    std::array<float, kIndexerHeadDim> denom;
    std::array<float, kIndexerHeadDim> acc;
    std::array<float, kIndexerHeadDim> normed;
    std::array<at::BFloat16, kIndexerHeadDim> quant_bf16;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t position = positions[t];
      if ((position + 1) % compress_ratio != 0) continue;
      const int64_t kv_slot = kv_slot_mapping[t];
      if (kv_slot < 0) continue;

      std::fill(max_v.begin(), max_v.end(), -std::numeric_limits<float>::infinity());
      std::fill(denom.begin(), denom.end(), 0.f);
      std::fill(acc.begin(), acc.end(), 0.f);

      for (int64_t w = 0; w < window; ++w) {
        const int64_t slot = gather_slots[t * window + w];
        if (slot < 0) continue;
        const int64_t block_idx = slot / state_block_size;
        const int64_t pos_in_block = slot % state_block_size;
        const int64_t head_offset = (w >= compress_ratio) ? kIndexerHeadDim : 0;
        const float* const score_row = state_cache + block_idx * state_block_stride +
            pos_in_block * state_row_width + head_offset + state_width;
        for (int64_t d = 0; d < kIndexerHeadDim; d += 16) {
          __m512 mx = _mm512_loadu_ps(max_v.data() + d);
          __m512 sv = _mm512_loadu_ps(score_row + d);
          _mm512_storeu_ps(max_v.data() + d, _mm512_max_ps(mx, sv));
        }
      }
      for (int64_t w = 0; w < window; ++w) {
        const int64_t slot = gather_slots[t * window + w];
        if (slot < 0) continue;
        const int64_t block_idx = slot / state_block_size;
        const int64_t pos_in_block = slot % state_block_size;
        const int64_t head_offset = (w >= compress_ratio) ? kIndexerHeadDim : 0;
        const float* const row = state_cache + block_idx * state_block_stride +
            pos_in_block * state_row_width + head_offset;
        const float* const score_row = row + state_width;
        for (int64_t d = 0; d < kIndexerHeadDim; d += 16) {
          fVec e_v = (fVec::loadu(score_row + d) - fVec::loadu(max_v.data() + d)).exp();
          (fVec::loadu(denom.data() + d) + e_v).store(denom.data() + d);
          (fVec::loadu(acc.data() + d) + e_v * fVec::loadu(row + d)).store(acc.data() + d);
        }
      }

      float sq_sum = 0.f;
      {
        fVec sq_sum_v(0.f);
        for (int64_t d = 0; d < kIndexerHeadDim; d += 16) {
          fVec v = fVec::loadu(acc.data() + d) / fVec::loadu(denom.data() + d);
          v.store(normed.data() + d);
          sq_sum_v = sq_sum_v + v * v;
        }
        sq_sum = vec_reduce_sum(sq_sum_v);
      }
      const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(kIndexerHeadDim) + rms_norm_eps);
      {
        fVec inv_rms_v(inv_rms);
        for (int64_t d = 0; d < kIndexerHeadDim; d += 16) {
          (fVec::loadu(normed.data() + d) * inv_rms_v * fVec::loadu(rms_norm_weight + d)).store(normed.data() + d);
        }
      }

      const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
      const float* const cos = cos_sin_cache + compressed_pos * kIndexerRopeDim;
      const float* const sin = cos + rope_half;
      apply_gptj_rope_inplace(normed.data() + kIndexerNopeDim, cos, sin, rope_half);

      // Indexer path rounds the FULL post-RoPE vector through bf16 before
      // quantizing (single combined block); the head=512 path above instead
      // rounds only the NoPE region before RoPE (separate bf16 rope store).
      for (int64_t d = 0; d < kIndexerHeadDim; d += 16)
        store_from_float_ext<at::BFloat16>(quant_bf16.data() + d, fVec::loadu(normed.data() + d));

      const int64_t block_idx = kv_slot / kv_cache_block_size;
      const int64_t pos_in_block = kv_slot % kv_cache_block_size;
      uint8_t* const block_base = kv_cache + block_idx * cache_row_stride;
      uint8_t* const token_fp8_ptr = block_base + pos_in_block * kIndexerTokenBytes;
      uint8_t* const token_scale_ptr =
          block_base + kv_cache_block_size * kIndexerTokenBytes + pos_in_block * kIndexerScaleBytes;

      float absmax = 1e-4f;
      {
        __m512 max_v_reg = _mm512_set1_ps(absmax);
        for (int64_t di = 0; di < kIndexerHeadDim; di += 16) {
          __m256i bf16_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quant_bf16.data() + di));
          __m512 f = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_v), 16));
          __m512 av = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), _mm512_set1_epi32(0x7FFFFFFF)));
          max_v_reg = _mm512_max_ps(max_v_reg, av);
        }
        absmax = _mm512_reduce_max_ps(max_v_reg);
      }
      const float exponent = std::ceil(std::log2(absmax / kFp8Max));
      const float scale = std::exp2(exponent);
      const float inv_scale = 1.0f / scale;

      // AVX512 quantize+store, reusing SGLang's `cvt_fp32x16_to_fp8x16` bit-
      // manipulation conversion (this single-shared-scale 128-wide block has
      // no direct SGLang equivalent -- see file header -- so only the fp8
      // conversion primitive is ported, the amax/scale formula is unchanged
      // from vLLM's own reference).
      uint8_t* const out_fp8 = token_fp8_ptr;
      {
        const __m512 vinv = _mm512_set1_ps(inv_scale);
        const __m512 vmax = _mm512_set1_ps(kFp8Max);
        const __m512 vmin = _mm512_set1_ps(kFp8Min);
        alignas(64) float scaled_buf[kIndexerHeadDim];
        for (int64_t d = 0; d < kIndexerHeadDim; d += 16) {
          __m256i bf16_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quant_bf16.data() + d));
          __m512 f = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_v), 16));
          __m512 s = _mm512_mul_ps(f, vinv);
          s = _mm512_max_ps(_mm512_min_ps(s, vmax), vmin);
          _mm512_store_ps(scaled_buf + d, s);
          __m128i fp8_v = cvt_fp32x16_to_fp8x16(s);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(out_fp8 + d), fp8_v);
        }
        // Patch subnormal-magnitude lanes (see `quantize_tile_avx512`'s
        // comment above for why the vectorized conversion's flush-to-zero
        // must not be trusted here).
        for (int64_t d = 0; d < kIndexerHeadDim; ++d) {
          const float av = std::abs(scaled_buf[d]);
          if (av != 0.f && av < 0x1p-6f) {
            out_fp8[d] = at::Float8_e4m3fn(scaled_buf[d]).x;
          }
        }
      }
      // Raw fp32 scale (not UE8M0-encoded) -- single quant block, matches
      // fp8_paged_mqa_logits_cpu's (paged_mqa_logits.cpp) read-side expectations.
      *reinterpret_cast<float*>(token_scale_ptr) = scale;
    }
  });
}

}  // anonymous namespace

void save_partial_states_cpu(
    at::Tensor& kv,
    at::Tensor& score,
    at::Tensor& ape,
    at::Tensor& positions,
    at::Tensor& state_cache,
    at::Tensor& slot_mapping) {
  TORCH_CHECK(
      kv.dim() == 2 && kv.scalar_type() == at::kFloat, "save_partial_states_cpu: kv must be a 2D fp32 tensor");
  TORCH_CHECK(score.sizes() == kv.sizes() && score.scalar_type() == at::kFloat,
              "save_partial_states_cpu: score must match kv's shape/dtype");
  TORCH_CHECK(ape.dim() == 2 && ape.size(1) == kv.size(1) && ape.scalar_type() == at::kFloat,
              "save_partial_states_cpu: ape must be [compress_ratio, state_width] fp32");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == kv.size(0),
              "save_partial_states_cpu: positions must be int64 with num_tokens entries");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong && slot_mapping.numel() == kv.size(0),
              "save_partial_states_cpu: slot_mapping must be int64 with num_tokens entries");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(2) == 2 * kv.size(1) &&
                  state_cache.scalar_type() == at::kFloat,
              "save_partial_states_cpu: state_cache must be [num_blocks, block_size, "
              "2*state_width] fp32");
  TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) == state_cache.size(2),
              "save_partial_states_cpu: state_cache rows must be contiguous within a block");
  TORCH_CHECK(
      kv.device().is_cpu() && score.device().is_cpu() && ape.device().is_cpu() && positions.device().is_cpu() &&
          state_cache.device().is_cpu() && slot_mapping.device().is_cpu(),
      "save_partial_states_cpu: all inputs must be CPU tensors");

  const int64_t num_tokens = kv.size(0);
  const int64_t state_width = kv.size(1);
  const int64_t compress_ratio = ape.size(0);
  const int64_t block_size = state_cache.size(1);

  TORCH_CHECK(
      kv.stride(1) == 1 && score.stride(1) == 1 && ape.is_contiguous() && positions.is_contiguous() &&
          slot_mapping.is_contiguous(),
      "save_partial_states_cpu: kv/score must be contiguous in their last dim "
      "(row stride may exceed state_width -- e.g. a column-split view of a "
      "fused kv_score projection); ape/positions/slot_mapping must be fully "
      "contiguous");

  save_partial_states_impl(
      state_cache.data_ptr<float>(),
      state_cache.stride(0),
      block_size,
      kv.data_ptr<float>(),
      kv.stride(0),
      score.data_ptr<float>(),
      score.stride(0),
      ape.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      slot_mapping.data_ptr<int64_t>(),
      state_width,
      compress_ratio,
      num_tokens);
}

void compress_norm_rope_store_cpu(
    at::Tensor& state_cache,
    at::Tensor& gather_slots,
    at::Tensor& positions,
    at::Tensor& kv_slot_mapping,
    at::Tensor& rms_norm_weight,
    double rms_norm_eps,
    at::Tensor& cos_sin_cache,
    at::Tensor& kv_cache_2d,
    int64_t kv_cache_block_size,
    int64_t compress_ratio) {
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.scalar_type() == at::kFloat,
              "compress_norm_rope_store_cpu: state_cache must be a 3D fp32 tensor");
  TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) == state_cache.size(2),
              "compress_norm_rope_store_cpu: state_cache rows must be contiguous within a block");
  TORCH_CHECK(gather_slots.dim() == 2 && gather_slots.scalar_type() == at::kLong,
              "compress_norm_rope_store_cpu: gather_slots must be a 2D int64 tensor");
  const int64_t num_tokens = gather_slots.size(0);
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == num_tokens,
              "compress_norm_rope_store_cpu: positions must be int64 with num_tokens entries");
  TORCH_CHECK(kv_slot_mapping.scalar_type() == at::kLong && kv_slot_mapping.numel() == num_tokens,
              "compress_norm_rope_store_cpu: kv_slot_mapping must be int64 with num_tokens entries");
  TORCH_CHECK(rms_norm_weight.numel() == kHeadDim,
              "compress_norm_rope_store_cpu: rms_norm_weight must have head_dim (512) entries");
  TORCH_CHECK(kv_cache_2d.dim() == 2 && kv_cache_2d.scalar_type() == at::kByte,
              "compress_norm_rope_store_cpu: kv_cache_2d must be a 2D uint8 tensor");
  TORCH_CHECK(state_cache.size(2) % 2 == 0, "compress_norm_rope_store_cpu: state_cache width must be even");
  TORCH_CHECK(
      rms_norm_weight.scalar_type() == at::kFloat && rms_norm_weight.is_contiguous() &&
          cos_sin_cache.scalar_type() == at::kFloat && cos_sin_cache.is_contiguous(),
      "compress_norm_rope_store_cpu: rms_norm_weight/cos_sin_cache must be "
      "float32 and contiguous (convert once in Python)");
  TORCH_CHECK(
      gather_slots.is_contiguous() && positions.is_contiguous() && kv_slot_mapping.is_contiguous(),
      "compress_norm_rope_store_cpu: gather_slots/positions/kv_slot_mapping must be contiguous");
  TORCH_CHECK(
      state_cache.device().is_cpu() && gather_slots.device().is_cpu() && positions.device().is_cpu() &&
          kv_slot_mapping.device().is_cpu() && rms_norm_weight.device().is_cpu() &&
          cos_sin_cache.device().is_cpu() && kv_cache_2d.device().is_cpu(),
      "compress_norm_rope_store_cpu: all inputs must be CPU tensors");

  const int64_t window = gather_slots.size(1);
  const int64_t state_width = state_cache.size(2) / 2;
  const int64_t state_block_size = state_cache.size(1);

  // Row strides are threaded through explicitly rather than assumed, since
  // both buffers may be pages of a larger pooled allocation (matches
  // store_cache.cpp/flash_mla.cpp's precedent).
  const int64_t state_block_stride = state_cache.stride(0);
  const int64_t cache_row_stride = kv_cache_2d.stride(0);

  compress_norm_rope_store_impl(
      state_cache.data_ptr<float>(),
      state_block_stride,
      state_block_size,
      gather_slots.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      kv_slot_mapping.data_ptr<int64_t>(),
      rms_norm_weight.data_ptr<float>(),
      static_cast<float>(rms_norm_eps),
      cos_sin_cache.data_ptr<float>(),
      kv_cache_2d.data_ptr<uint8_t>(),
      cache_row_stride,
      kv_cache_block_size,
      compress_ratio,
      window,
      state_width,
      num_tokens);
}

void compress_norm_rope_store_indexer_cpu(
    at::Tensor& state_cache,
    at::Tensor& gather_slots,
    at::Tensor& positions,
    at::Tensor& kv_slot_mapping,
    at::Tensor& rms_norm_weight,
    double rms_norm_eps,
    at::Tensor& cos_sin_cache,
    at::Tensor& kv_cache_2d,
    int64_t kv_cache_block_size,
    int64_t compress_ratio) {
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.scalar_type() == at::kFloat,
              "compress_norm_rope_store_indexer_cpu: state_cache must be a 3D fp32 tensor");
  TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) == state_cache.size(2),
              "compress_norm_rope_store_indexer_cpu: state_cache rows must be contiguous within a block");
  TORCH_CHECK(gather_slots.dim() == 2 && gather_slots.scalar_type() == at::kLong,
              "compress_norm_rope_store_indexer_cpu: gather_slots must be a 2D int64 tensor");
  const int64_t num_tokens = gather_slots.size(0);
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == num_tokens,
              "compress_norm_rope_store_indexer_cpu: positions must be int64 with num_tokens entries");
  TORCH_CHECK(kv_slot_mapping.scalar_type() == at::kLong && kv_slot_mapping.numel() == num_tokens,
              "compress_norm_rope_store_indexer_cpu: kv_slot_mapping must be int64 with num_tokens entries");
  TORCH_CHECK(rms_norm_weight.numel() == kIndexerHeadDim,
              "compress_norm_rope_store_indexer_cpu: rms_norm_weight must have head_dim (128) entries");
  TORCH_CHECK(kv_cache_2d.dim() == 2 && kv_cache_2d.scalar_type() == at::kByte,
              "compress_norm_rope_store_indexer_cpu: kv_cache_2d must be a 2D uint8 tensor");
  TORCH_CHECK(state_cache.size(2) % 2 == 0,
              "compress_norm_rope_store_indexer_cpu: state_cache width must be even");
  TORCH_CHECK(
      rms_norm_weight.scalar_type() == at::kFloat && rms_norm_weight.is_contiguous() &&
          cos_sin_cache.scalar_type() == at::kFloat && cos_sin_cache.is_contiguous(),
      "compress_norm_rope_store_indexer_cpu: rms_norm_weight/cos_sin_cache must "
      "be float32 and contiguous (convert once in Python)");
  TORCH_CHECK(
      gather_slots.is_contiguous() && positions.is_contiguous() && kv_slot_mapping.is_contiguous(),
      "compress_norm_rope_store_indexer_cpu: gather_slots/positions/kv_slot_mapping must be contiguous");
  TORCH_CHECK(
      state_cache.device().is_cpu() && gather_slots.device().is_cpu() && positions.device().is_cpu() &&
          kv_slot_mapping.device().is_cpu() && rms_norm_weight.device().is_cpu() &&
          cos_sin_cache.device().is_cpu() && kv_cache_2d.device().is_cpu(),
      "compress_norm_rope_store_indexer_cpu: all inputs must be CPU tensors");

  const int64_t window = gather_slots.size(1);
  const int64_t state_width = state_cache.size(2) / 2;
  const int64_t state_block_size = state_cache.size(1);

  const int64_t state_block_stride = state_cache.stride(0);
  const int64_t cache_row_stride = kv_cache_2d.stride(0);

  compress_norm_rope_store_indexer_impl(
      state_cache.data_ptr<float>(),
      state_block_stride,
      state_block_size,
      gather_slots.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      kv_slot_mapping.data_ptr<int64_t>(),
      rms_norm_weight.data_ptr<float>(),
      static_cast<float>(rms_norm_eps),
      cos_sin_cache.data_ptr<float>(),
      kv_cache_2d.data_ptr<uint8_t>(),
      cache_row_stride,
      kv_cache_block_size,
      compress_ratio,
      window,
      state_width,
      num_tokens);
}
