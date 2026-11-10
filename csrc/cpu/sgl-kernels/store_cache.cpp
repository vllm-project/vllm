// Adapted from
// https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/cpu/store_cache.cpp
//
// DeepSeek-V4 fused Q RMSNorm+RoPE / KV RoPE+UE8M0-quant+paged-cache-insert
// kernel for the fp8_ds_mla cache layout. Not a port of upstream's
// `set_k_and_s_cpu`/`quant_to_nope_fp8_rope_bf16_pack_cpu` -- their
// pool_kv/pool_score signature doesn't apply to vLLM's layout. This is the
// CPU counterpart to vLLM's own CUDA kernel
// (`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`); see
// `DeepseekV4Attention._fused_qnorm_rope_kv_insert_cpu` (attention.py) for
// the eager reference this replaces.
//
// clang-format off

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"
#include "vec.h"

#if !defined(CPU_CAPABILITY_AVX512)
static_assert(false,
              "store_cache.cpp is only ever compiled into the AVX512+AMX-flagged "
              "_C target (see cmake/cpu_extension.cmake's VLLM_EXT_SRC_SGL) and "
              "relies on that unconditionally -- CPU_CAPABILITY_AVX512 must be "
              "defined here.");
#endif

namespace {

constexpr int64_t kRopeDim = 64;
constexpr int64_t kQuantBlock = 64;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8Min = -448.0f;
constexpr float kQuantEps = 1e-4f;

inline void apply_gptj_rope_inplace(float* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin, int64_t half_dim) {
  for (int64_t i = 0; i < half_dim; ++i) {
    const float e = x[2 * i];
    const float o = x[2 * i + 1];
    x[2 * i] = e * cos[i] - o * sin[i];
    x[2 * i + 1] = e * sin[i] + o * cos[i];
  }
}

// AVX512 tile-quantization, adapted verbatim from SGLang's
// `store_cache.cpp` (`quant_to_nope_fp8_rope_bf16_pack_cpu`'s
// `cvt_fp32x16_to_fp8x16`/`quantize_tile_avx512`): per-64-element tile,
// computes the UE8M0 power-of-two scale and quantizes to fp8_e4m3fn with
// RNE rounding via bit manipulation instead of a per-element scalar
// `at::Float8_e4m3fn` cast. Subnormal-magnitude fp8 lanes are flushed to
// zero (matches SGLang's own convention, `cvt_fp32x16_to_fp8x16`) --
// vLLM's own AVX512 dequant of this cache (`flash_mla.cpp`'s
// `CVT_FP8_TO_BF16_EXT`) doesn't reconstruct exact subnormal magnitudes
// either way, so there is nothing to preserve by rounding them exactly on
// the write side.
#if defined(CPU_CAPABILITY_AVX512)

// Vectorized float32x16 -> fp8_e4m3fn x16 using AVX512 bit manipulation.
// Implements RNE (round-to-nearest-even). Flushes subnormal fp8 to zero.
// fp8_e4m3fn: sign(1), exp(4, bias=7), mant(3), no inf/nan special values.
// Normal: exp_field in [1..15], value = 2^(exp-7) * (1 + mant/8)
// Input assumed clamped to [-448, 448].
inline __m128i cvt_fp32x16_to_fp8x16(__m512 input) {
  __m512i bits = _mm512_castps_si512(input);

  // Extract sign -> bit 7 of fp8 byte
  __m512i signs = _mm512_srli_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x80000000)), 24);

  // Absolute value bits
  __m512i abs_bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFFFFFF));

  // Float32 biased exponent
  __m512i f32_exp = _mm512_srli_epi32(abs_bits, 23);

  // Float32 mantissa (23 bits)
  __m512i f32_mant = _mm512_and_si512(abs_bits, _mm512_set1_epi32(0x7FFFFF));

  // fp8_exp = f32_exp - 120 (rebias from 127 to 7)
  __m512i fp8_exp = _mm512_sub_epi32(f32_exp, _mm512_set1_epi32(120));

  // Top 3 mantissa bits
  __m512i mant3 = _mm512_srli_epi32(f32_mant, 20);

  // RNE rounding: round_bit = bit 19, sticky = bits[18:0]
  __m512i round_bit = _mm512_and_si512(_mm512_srli_epi32(f32_mant, 19), _mm512_set1_epi32(1));
  __m512i sticky_bits = _mm512_and_si512(f32_mant, _mm512_set1_epi32(0x7FFFF));
  __mmask16 has_sticky = _mm512_cmpneq_epi32_mask(sticky_bits, _mm512_setzero_si512());

  // Round up if round_bit AND (sticky OR lsb_of_mant3)
  __m512i lsb = _mm512_and_si512(mant3, _mm512_set1_epi32(1));
  __m512i sticky_or_lsb = _mm512_or_si512(_mm512_maskz_mov_epi32(has_sticky, _mm512_set1_epi32(1)), lsb);
  __m512i do_round = _mm512_and_si512(round_bit, sticky_or_lsb);
  mant3 = _mm512_add_epi32(mant3, do_round);

  // Mantissa overflow (mant3 == 8) -> increment exponent, zero mantissa
  __mmask16 mant_ovf = _mm512_cmpeq_epi32_mask(mant3, _mm512_set1_epi32(8));
  mant3 = _mm512_mask_mov_epi32(mant3, mant_ovf, _mm512_setzero_si512());
  fp8_exp = _mm512_mask_add_epi32(fp8_exp, mant_ovf, fp8_exp, _mm512_set1_epi32(1));

  // Clamp exponent: max 15
  fp8_exp = _mm512_min_epi32(fp8_exp, _mm512_set1_epi32(15));

  // Result: (fp8_exp << 3) | mant3
  __m512i result = _mm512_or_si512(_mm512_slli_epi32(fp8_exp, 3), mant3);

  // Flush to zero: if f32_exp < 121 (would be subnormal in fp8), output 0
  __mmask16 is_subnormal_or_zero = _mm512_cmplt_epi32_mask(f32_exp, _mm512_set1_epi32(121));
  result = _mm512_mask_mov_epi32(result, is_subnormal_or_zero, _mm512_setzero_si512());

  // Add sign
  result = _mm512_or_si512(result, signs);

  // Pack 16 x i32 -> 16 x i8
  return _mm512_cvtepi32_epi8(result);
}

// Process one tile (64 bf16 values): find amax, compute scale, quantize to fp8.
inline uint8_t quantize_tile_avx512(const at::BFloat16* __restrict__ src, uint8_t* __restrict__ dst) {
  // Load 64 bf16 -> 4 x 16 floats
  __m256i bf16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
  __m256i bf16_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 16));
  __m256i bf16_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
  __m256i bf16_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 48));

  __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_0), 16));
  __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_1), 16));
  __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_2), 16));
  __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_3), 16));

  // Absolute values
  const __m512i abs_mask_i = _mm512_set1_epi32(0x7FFFFFFF);
  __m512 abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask_i));
  __m512 abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask_i));
  __m512 abs2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask_i));
  __m512 abs3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask_i));

  // Horizontal max
  __m512 max01 = _mm512_max_ps(abs0, abs1);
  __m512 max23 = _mm512_max_ps(abs2, abs3);
  __m512 max0123 = _mm512_max_ps(max01, max23);
  float amax = _mm512_reduce_max_ps(max0123);

  // Scale computation
  float scale = std::max(amax / kFp8Max, kQuantEps);
  float ceil_log2 = std::ceil(std::log2(scale));
  float scale_pow2 = std::exp2(ceil_log2);
  float scale_inv = 1.0f / scale_pow2;

  int exponent = static_cast<int>(ceil_log2);
  uint8_t scale_uint8 = static_cast<uint8_t>(exponent + 127);

  // Scale all values
  __m512 vinv = _mm512_set1_ps(scale_inv);
  __m512 s0 = _mm512_mul_ps(f0, vinv);
  __m512 s1 = _mm512_mul_ps(f1, vinv);
  __m512 s2 = _mm512_mul_ps(f2, vinv);
  __m512 s3 = _mm512_mul_ps(f3, vinv);

  // Clamp
  __m512 vmax = _mm512_set1_ps(kFp8Max);
  __m512 vmin = _mm512_set1_ps(kFp8Min);
  s0 = _mm512_max_ps(_mm512_min_ps(s0, vmax), vmin);
  s1 = _mm512_max_ps(_mm512_min_ps(s1, vmax), vmin);
  s2 = _mm512_max_ps(_mm512_min_ps(s2, vmax), vmin);
  s3 = _mm512_max_ps(_mm512_min_ps(s3, vmax), vmin);

  // Vectorized float -> fp8 conversion (16 values at a time)
  __m128i fp8_0 = cvt_fp32x16_to_fp8x16(s0);
  __m128i fp8_1 = cvt_fp32x16_to_fp8x16(s1);
  __m128i fp8_2 = cvt_fp32x16_to_fp8x16(s2);
  __m128i fp8_3 = cvt_fp32x16_to_fp8x16(s3);

  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), fp8_0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), fp8_1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 32), fp8_2);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 48), fp8_3);

  return scale_uint8;
}

#endif  // CPU_CAPABILITY_AVX512

template <typename scalar_t>
static void qnorm_rope_kv_insert_impl(
    scalar_t* __restrict__ q_out,               // [num_tokens_full, q_head_padded, head_dim]
    uint8_t* __restrict__ cache,                // [num_blocks, cache_row_stride] bytes
    const scalar_t* __restrict__ q,             // [num_tokens_full, num_heads_q, head_dim]
    const scalar_t* __restrict__ kv,            // [>=num_tokens_insert, head_dim]
    const int64_t* __restrict__ positions,      // [num_tokens_full]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens_insert]
    const float* __restrict__ cos_sin_cache,    // [max_pos, rope_dim] (first half cos, second half sin)
    int64_t num_tokens_full,
    int64_t num_heads_q,
    int64_t q_head_padded,
    int64_t head_dim,
    int64_t nope_dim,
    int64_t num_quant_blocks,
    int64_t token_data_bytes,
    int64_t scale_bytes_per_token,
    int64_t num_tokens_insert,
    int64_t cache_block_size,
    int64_t cache_row_stride,
    float eps) {
  const int64_t rope_half = kRopeDim / 2;

  // Q: per-head weight-free RMSNorm + GPT-J RoPE on the last rope_dim dims.
  at::parallel_for(0, num_tokens_full * num_heads_q, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> buf(head_dim);
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / num_heads_q;
      const int64_t h = idx % num_heads_q;
      const scalar_t* q_th = q + (t * num_heads_q + h) * head_dim;

      float sq_sum = 0.f;
      for (int64_t i = 0; i < head_dim; ++i) {
        const float v = static_cast<float>(q_th[i]);
        buf[i] = v;
        sq_sum += v * v;
      }
      const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(head_dim) + eps);
      for (int64_t i = 0; i < head_dim; ++i)
        buf[i] *= inv_rms;

      const int64_t pos = positions[t];
      const float* cos = cos_sin_cache + pos * kRopeDim;
      const float* sin = cos + rope_half;
      apply_gptj_rope_inplace(buf.data() + nope_dim, cos, sin, rope_half);

      scalar_t* out_th = q_out + (t * q_head_padded + h) * head_dim;
      for (int64_t i = 0; i < head_dim; ++i)
        out_th[i] = static_cast<scalar_t>(buf[i]);
    }
  });

  // KV: GPT-J RoPE (no RMSNorm) + bf16 round-trip + UE8M0 FP8 quant + paged insert.
  at::parallel_for(0, num_tokens_insert, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> buf(head_dim);
    std::vector<at::BFloat16> nope_bf16(nope_dim);
    for (int64_t t = begin; t < end; ++t) {
      const int64_t slot_id = slot_mapping[t];
      if (slot_id < 0) continue;

      const scalar_t* kv_t = kv + t * head_dim;
      for (int64_t i = 0; i < head_dim; ++i)
        buf[i] = static_cast<float>(kv_t[i]);

      const int64_t pos = positions[t];
      const float* cos = cos_sin_cache + pos * kRopeDim;
      const float* sin = cos + rope_half;
      apply_gptj_rope_inplace(buf.data() + nope_dim, cos, sin, rope_half);

      // bf16 round-trip, matching the CUDA kernel's double convert.
      for (int64_t i = 0; i < head_dim; ++i)
        buf[i] = static_cast<float>(at::BFloat16(buf[i]));
      for (int64_t i = 0; i < nope_dim; ++i)
        nope_bf16[i] = at::BFloat16(buf[i]);

      const int64_t block_idx = slot_id / cache_block_size;
      const int64_t pos_in_block = slot_id % cache_block_size;
      uint8_t* const block_base = cache + block_idx * cache_row_stride;
      uint8_t* const token_fp8_ptr = block_base + pos_in_block * token_data_bytes;
      uint8_t* const token_bf16_ptr = token_fp8_ptr + nope_dim;
      uint8_t* const token_scale_ptr =
          block_base + cache_block_size * token_data_bytes + pos_in_block * scale_bytes_per_token;

      // AVX512 tile-quantization (SGLang's `quantize_tile_avx512`), one
      // 64-wide tile per iteration.
      for (int64_t qb = 0; qb < num_quant_blocks; ++qb) {
        const at::BFloat16* tile_src = nope_bf16.data() + qb * kQuantBlock;
        uint8_t* tile_dst = token_fp8_ptr + qb * kQuantBlock;
        token_scale_ptr[qb] = quantize_tile_avx512(tile_src, tile_dst);
      }
      token_scale_ptr[num_quant_blocks] = 0;

      auto* out_rope = reinterpret_cast<at::BFloat16*>(token_bf16_ptr);
      for (int64_t i = 0; i < kRopeDim; ++i)
        out_rope[i] = at::BFloat16(buf[nope_dim + i]);
    }
  });
}

}  // anonymous namespace

at::Tensor fused_qnorm_rope_kv_insert_cpu(
    at::Tensor& q,
    at::Tensor& kv,
    at::Tensor& positions,
    at::Tensor& swa_kv_cache_2d,
    at::Tensor& slot_mapping,
    at::Tensor& cos_sin_cache,
    int64_t q_head_padded,
    double eps,
    int64_t cache_block_size) {
  TORCH_CHECK(
      q.dim() == 3 && (q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf),
      "fused_qnorm_rope_kv_insert_cpu: q must be bf16/fp16 [num_tokens, num_heads_q, head_dim]");
  TORCH_CHECK(kv.dim() == 2 && kv.scalar_type() == q.scalar_type(), "fused_qnorm_rope_kv_insert_cpu: kv dtype must match q");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "fused_qnorm_rope_kv_insert_cpu: positions must be int64");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "fused_qnorm_rope_kv_insert_cpu: slot_mapping must be int64");
  TORCH_CHECK(
      swa_kv_cache_2d.dim() == 2 && swa_kv_cache_2d.scalar_type() == at::kByte,
      "fused_qnorm_rope_kv_insert_cpu: swa_kv_cache_2d must be a 2D uint8 tensor");
  TORCH_CHECK(
      q.device().is_cpu() && kv.device().is_cpu() && positions.device().is_cpu() && slot_mapping.device().is_cpu() &&
          swa_kv_cache_2d.device().is_cpu() && cos_sin_cache.device().is_cpu(),
      "fused_qnorm_rope_kv_insert_cpu: all inputs must be CPU tensors");

  const int64_t num_tokens_full = q.size(0);
  const int64_t num_heads_q = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t nope_dim = head_dim - kRopeDim;
  TORCH_CHECK(nope_dim > 0 && nope_dim % kQuantBlock == 0, "fused_qnorm_rope_kv_insert_cpu: unexpected head_dim");
  const int64_t num_quant_blocks = nope_dim / kQuantBlock;
  const int64_t scale_bytes_per_token = num_quant_blocks + 1;
  const int64_t token_data_bytes = nope_dim + kRopeDim * 2;

  TORCH_CHECK(q_head_padded >= num_heads_q, "fused_qnorm_rope_kv_insert_cpu: q_head_padded must be >= num_heads_q");
  TORCH_CHECK(kv.size(1) == head_dim, "fused_qnorm_rope_kv_insert_cpu: kv head_dim must match q");
  TORCH_CHECK(positions.numel() == num_tokens_full, "fused_qnorm_rope_kv_insert_cpu: positions must have num_tokens_full entries");

  const int64_t num_tokens_insert = slot_mapping.size(0);
  TORCH_CHECK(kv.size(0) >= num_tokens_insert, "fused_qnorm_rope_kv_insert_cpu: kv must have >= num_tokens_insert rows");
  TORCH_CHECK(cache_block_size > 0, "fused_qnorm_rope_kv_insert_cpu: cache_block_size must be positive");
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == at::kFloat && cos_sin_cache.is_contiguous(),
      "fused_qnorm_rope_kv_insert_cpu: cos_sin_cache must be float32 and "
      "contiguous (convert once in Python)");
  TORCH_CHECK(
      q.is_contiguous() && kv.is_contiguous() && positions.is_contiguous() && slot_mapping.is_contiguous(),
      "fused_qnorm_rope_kv_insert_cpu: q/kv/positions/slot_mapping must be contiguous");

  // swa_kv_cache_2d is mutated in place; index via its real row stride
  // rather than assuming it equals cache_block_size * (token_data_bytes +
  // scale_bytes_per_token) -- a paged/pooled buffer's row stride is not
  // guaranteed to match the logical per-block byte count.
  const int64_t cache_row_stride = swa_kv_cache_2d.stride(0);

  auto q_out = at::zeros({num_tokens_full, q_head_padded, head_dim}, q.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "fused_qnorm_rope_kv_insert_cpu", [&] {
    qnorm_rope_kv_insert_impl<scalar_t>(
        q_out.data_ptr<scalar_t>(),
        swa_kv_cache_2d.data_ptr<uint8_t>(),
        q.data_ptr<scalar_t>(),
        kv.data_ptr<scalar_t>(),
        positions.data_ptr<int64_t>(),
        slot_mapping.data_ptr<int64_t>(),
        cos_sin_cache.data_ptr<float>(),
        num_tokens_full,
        num_heads_q,
        q_head_padded,
        head_dim,
        nope_dim,
        num_quant_blocks,
        token_data_bytes,
        scale_bytes_per_token,
        num_tokens_insert,
        cache_block_size,
        cache_row_stride,
        static_cast<float>(eps));
  });

  return q_out;
}

// vLLM-native: inverse GPT-J RoPE for `_o_proj`'s output de-rotation (undoes
// the query's own rotation on the attention output before the low-rank
// wo_a/wo_b projection -- see the math writeup in cpu_sparse.py's `_o_proj`).
// Upstream needs this same step (`fused_rope_inplace(..., inverse=True)` in
// `deepseek_v4.py`) but only has a CUDA-JIT implementation, no CPU kernel --
// this fills that gap rather than porting one. Replaces the eager reference
// in `DeepseekV4CPUAttention._o_proj` (cpu_sparse.py). Uses the same
// half-half `[cos...|sin...]` cache layout as `apply_gptj_rope_inplace`
// above (not upstream's interleaved `[c0,s0,c1,s1,...]` layout).
//
// AVX512 path: NoPE dtype-convert copy uses `vec.h`'s `load_float_vec`
// (16-wide bf16/fp16 -> fp32). The interleaved (even,odd) rotation widens
// 32 raw elements to fp32 via `load_float_vec2`, round-trips them through
// an aligned scratch buffer, then gathers/scatters the even/odd lanes --
// same technique as `indexer.cpp`'s Q-side RoPE, since `cos`/`sin` here
// are two contiguous halves rather than per-pair-interleaved.
at::Tensor inverse_gptj_rope_o_proj_cpu(
    at::Tensor& o,          // [num_tokens, num_heads, head_dim]
    at::Tensor& positions,  // [num_tokens]
    at::Tensor& cos_sin_cache,  // [max_pos, rope_dim] (first half cos, second half sin)
    int64_t rope_dim) {
  TORCH_CHECK(o.dim() == 3, "inverse_gptj_rope_o_proj_cpu: o must be 3D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "inverse_gptj_rope_o_proj_cpu: positions must be int64");
  TORCH_CHECK(
      o.device().is_cpu() && positions.device().is_cpu() && cos_sin_cache.device().is_cpu(),
      "inverse_gptj_rope_o_proj_cpu: all inputs must be CPU tensors");

  const int64_t num_tokens = o.size(0);
  const int64_t num_heads = o.size(1);
  const int64_t head_dim = o.size(2);
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t half_dim = rope_dim / 2;
  TORCH_CHECK(nope_dim >= 0, "inverse_gptj_rope_o_proj_cpu: rope_dim must not exceed head_dim");
  // Only ever called with head_dim=512, rope_dim=64 (DeepseekV4CPUAttention's
  // _o_proj); the AVX512 loops below process 16 elements/pairs per iteration
  // with no scalar remainder, so require both to divide evenly.
  TORCH_CHECK(nope_dim % 16 == 0 && half_dim % 16 == 0,
              "inverse_gptj_rope_o_proj_cpu: head_dim-rope_dim and rope_dim/2 must be multiples of 16");
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == at::kFloat && cos_sin_cache.is_contiguous(),
      "inverse_gptj_rope_o_proj_cpu: cos_sin_cache must be float32 and "
      "contiguous (convert once in Python)");
  TORCH_CHECK(o.is_contiguous() && positions.is_contiguous(),
              "inverse_gptj_rope_o_proj_cpu: o/positions must be contiguous");

  auto out = at::empty({num_tokens, num_heads, head_dim}, o.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(o.scalar_type(), "inverse_gptj_rope_o_proj_cpu", [&] {
    const scalar_t* o_ptr = o.data_ptr<scalar_t>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t* pos_ptr = positions.data_ptr<int64_t>();
    const float* cos_sin_ptr = cos_sin_cache.data_ptr<float>();

    at::parallel_for(0, num_tokens * num_heads, 0, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t t = idx / num_heads;
        const int64_t h = idx % num_heads;
        const scalar_t* src = o_ptr + (t * num_heads + h) * head_dim;
        float* dst = out_ptr + (t * num_heads + h) * head_dim;

        // head_dim=512, rope_dim=64 in practice (checked above), so
        // nope_dim/half_dim are always multiples of 16 -- no scalar
        // remainder needed.
        for (int64_t i = 0; i < nope_dim; i += 16) {
          load_float_vec<scalar_t>(src + i).store(dst + i);
        }

        const int64_t pos = pos_ptr[t];
        const float* cos = cos_sin_ptr + pos * rope_dim;
        const float* sin = cos + half_dim;

        {
          // Permute-based (no gather/scatter), adapted from SGLang's
          // `rope.cpp` (`apply_rotary_emb_row`) the same way as
          // compressor.cpp's `apply_gptj_rope_inplace`: `_mm512_permute_ps`
          // swaps each pair's (even, odd) lanes in one shuffle, and cos/sin
          // -- loaded as 8 contiguous values -- are duplicated pairwise via
          // `_mm512_permutexvar_ps` to align with that layout, since our
          // cos/sin are two separate flat arrays rather than SGLang's
          // pre-interleaved freqs. Two batches of 8 pairs per iteration
          // (matching `load_float_vec2`'s 16+16-lane split).
          const __m512i dup_idx = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
          // Inverse GPT-J rotation negates the sin broadcast on the odd
          // (imag) lane of each pair; the even (real) lane keeps sin's sign.
          const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
              (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0,
              (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0));
          for (int64_t j = 0; j < half_dim; j += 16) {
            const scalar_t* rsrc = src + nope_dim + 2 * j;
            at::vec::Vectorized<float> x0, x1;
            std::tie(x0, x1) = load_float_vec2(rsrc);
            float* rdst = dst + nope_dim + 2 * j;

            __m512 xv0 = x0;
            __m512 cos_b0 = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cos + j)));
            __m512 sin_b0 = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sin + j)));
            __m512 out0 = _mm512_fmadd_ps(
                xv0, cos_b0, _mm512_mul_ps(_mm512_permute_ps(xv0, 0xB1), _mm512_xor_ps(sin_b0, sign_mask)));
            _mm512_storeu_ps(rdst, out0);

            __m512 xv1 = x1;
            __m512 cos_b1 = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cos + j + 8)));
            __m512 sin_b1 = _mm512_permutexvar_ps(dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sin + j + 8)));
            __m512 out1 = _mm512_fmadd_ps(
                xv1, cos_b1, _mm512_mul_ps(_mm512_permute_ps(xv1, 0xB1), _mm512_xor_ps(sin_b1, sign_mask)));
            _mm512_storeu_ps(rdst + 16, out1);
          }
        }
      }
    });
  });

  return out;
}
