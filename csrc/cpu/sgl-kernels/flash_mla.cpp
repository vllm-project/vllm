// Adapted from
// https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/cpu/flash_mla.cpp
//
// DeepSeek-V4 sparse MQA attention kernel for the fp8_ds_mla cache layout.
// Near-verbatim port of `sparse_mla_decode_kernel_impl` (VNNI K/V packing +
// AMX brgemm QK^T/PV + online softmax); the `V4_FP8Sparse` byte layout
// matches vLLM's fp8_ds_mla layout exactly (see store_cache.cpp).
//
// Differences from upstream:
// - Outer shape: upstream takes a padded batch-major query `[B, S_q, H, D]`
//   with per-batch topk sets; vLLM passes flat `[num_tokens, H, D]` with
//   per-token resolved indices. The kernel's own `at::parallel_for` already
//   flattens `(batch, s_q)` internally, so this is called with
//   `batches = num_tokens, s_q = 1` -- no padding, no kernel-body changes.
// - Two physically distinct paged caches (SWA "window" + compressor
//   "compressed") are addressed via block_table-resolved flat slot ids,
//   matching upstream's `k_main`/`k_extra` two-index-set convention.
// - `attn_sink` uses upstream's own post-hoc LSE correction
//   (`out *= 1 / (1 + exp(sink - lse))`).
// - bf16 query only (matching upstream's own restriction: its FP8->bf16
//   dequant intrinsics are bf16-specific); fp16 is not supported here.
//
// clang-format off

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common.h"
#include "gemm.h"
#include "vec.h"

#if !defined(CPU_CAPABILITY_AVX512)
static_assert(false,
              "flash_mla.cpp is only ever compiled into the AVX512+AMX-flagged "
              "_C target (see cmake/cpu_extension.cmake's VLLM_EXT_SRC_SGL) and "
              "relies on that unconditionally -- CPU_CAPABILITY_AVX512 must be "
              "defined here.");
#endif

namespace {

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHeadDim = kNopeDim + kRopeDim;
constexpr int64_t kQuantBlock = 64;
constexpr int64_t kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int64_t kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;

// ---------------------------------------------------------------------------
// V4_FP8Sparse layout (SGLang's `flash_mla.cpp`) -- byte-for-byte identical
// to vLLM's own fp8_ds_mla layout above (d=512, d_nope=448, d_rope=64,
// tile=64, num_tiles=7):
//   per block:
//     [block_size * (d_nope + 2*d_rope) bytes ; FP8 NoPE + bf16 RoPE per token]
//     [block_size * 8 bytes                   ; 7 e8m0 scales + 1 pad byte/token]
// ---------------------------------------------------------------------------

struct FP8LayoutMeta {
  int64_t d;
  int64_t d_nope;
  int64_t d_rope;
  int64_t tile_size;
  int64_t num_tiles;
};

constexpr FP8LayoutMeta kV4Meta = {kHeadDim, kNopeDim, kRopeDim, kQuantBlock, kNumQuantBlocks};

// Convert one fp8_e8m0 byte (= unsigned 8-bit exponent) to float.
// e8m0 only stores an exponent (bias 127); value = 2^(e - 127), with 0xFF
// reserved for NaN.
inline float fp8_e8m0_to_float(uint8_t v) {
  if (v == 0xFF) return std::numeric_limits<float>::quiet_NaN();
  if (v == 0) return 0.f;
  union {
    uint32_t u;
    float f;
  } u;
  u.u = static_cast<uint32_t>(v) << 23;
  return u.f;
}

template <typename index_t>
inline bool is_valid_sparse_index(index_t idx, int64_t pos, int64_t topk_limit, int64_t total_tokens) {
  const int64_t v = static_cast<int64_t>(idx);
  return pos < topk_limit && v >= 0 && v < total_tokens;
}

#if defined(CPU_CAPABILITY_AVX512)
inline __attribute__((always_inline)) __m512i cvt_fp8_32_to_scaled_bf16(__m256i fp8, float scale) {
  const __m512bh bf16_ext = CVT_FP8_TO_BF16_EXT(fp8);
  const __m512 scale_vec = _mm512_mul_ps(_mm512_set1_ps(scale), _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS)));
  const __m512 f0 = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_ext, 0)), scale_vec);
  const __m512 f1 = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_ext, 1)), scale_vec);
  return (__m512i)_mm512_cvtne2ps_pbh(f1, f0);
}

// Load 32 contiguous logical dims (NoPE fp8-dequant-to-bf16, or raw bf16
// RoPE) starting at `dim_offset` from one token row of the V4_FP8Sparse
// layout.
inline __attribute__((always_inline)) __m512i load_fp8_kvcache_32_from_row(
    const uint8_t* __restrict__ row_base, const uint8_t* __restrict__ scale_base, int dim_offset) {
  if (dim_offset < kV4Meta.d_nope) {
    const float scale = fp8_e8m0_to_float(scale_base[dim_offset / kV4Meta.tile_size]);
    return cvt_fp8_32_to_scaled_bf16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_base + dim_offset)), scale);
  }
  const auto* rope_ptr = reinterpret_cast<const at::BFloat16*>(row_base + kV4Meta.d_nope);
  return _mm512_loadu_si512(rope_ptr + dim_offset - kV4Meta.d_nope);
}

inline __attribute__((always_inline)) void init_fp8_kvcache_tile_rows(
    const uint8_t* __restrict__ fp8_storage,
    const int64_t block_size,
    const int64_t storage_block_stride_bytes,
    const int64_t token_idx,
    const uint8_t*& row_base,
    const uint8_t*& scale_base) {
  constexpr int64_t nope_rope_per_token = kV4Meta.d_nope + 2 * kV4Meta.d_rope;
  constexpr int64_t scale_stride = 8;
  const int64_t block_idx = token_idx / block_size;
  const int64_t block_off = token_idx - block_idx * block_size;
  const uint8_t* block_base = fp8_storage + block_idx * storage_block_stride_bytes;
  row_base = block_base + block_off * nope_rope_per_token;
  scale_base = block_base + block_size * nope_rope_per_token + block_off * scale_stride;
}

template <bool convert_v, typename scalar_t, typename index_t, typename load_vec_t>
inline __attribute__((always_inline)) void sparse_pack_vnni_Nx32(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const index_t* __restrict__ ind,
    const bool* __restrict__ valid_mask,
    int N,
    int dim_offset,
    int ld_dst0,
    int ld_dst1,
    const load_vec_t& load_vec) {
  __m512i vinputs[16];
  int n = 0;
  for (; n < N; ++n) {
    if (!valid_mask[n]) {
      vinputs[n] = _mm512_set1_epi32(0);
    } else {
      vinputs[n] = load_vec(n, static_cast<int64_t>(ind[n]), dim_offset);
    }
  }
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  if constexpr (convert_v) {
    for (int nn = 0; nn < 16; nn += 2) {
      __m512i d0, d1;
      std::tie(d0, d1) = transpose_2x32_16bit(vinputs[nn], vinputs[nn + 1]);
      _mm512_storeu_si512(dst1 + (nn >> 1) * ld_dst1 * 2, d0);
      _mm512_storeu_si512(dst1 + (nn >> 1) * ld_dst1 * 2 + 32, d1);
    }
  }

  transpose_16x16_32bit(vinputs);
  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst0 + k * ld_dst0 * 2, vmask, vinputs[k]);
  }
}
#endif  // CPU_CAPABILITY_AVX512

// Gather+VNNI-pack `N` fp8-cache rows (dequantized to bf16 while packing)
// into `dst0` (K, vnni-packed for brgemm's B operand) and `dst1` (V,
// vnni-packed).
template <typename scalar_t, typename index_t>
void sparse_pack_vnni_fp8(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const uint8_t* __restrict__ fp8_storage,
    const index_t* __restrict__ ind,
    const bool* __restrict__ valid_mask,
    int N,
    int K,
    int Kv,
    int64_t block_size,
    int64_t storage_block_stride_bytes,
    int ld_dst0,
    int ld_dst1) {
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int KBv = std::min(Kv / 32, KB);
  for (int nb = 0; nb < NB; ++nb) {
    const uint8_t* row_base[16];
    const uint8_t* scale_base[16];
    const int nb_size = std::min(N - nb * 16, 16);
    for (int n = 0; n < nb_size; ++n) {
      if (!valid_mask[nb * 16 + n]) {
        row_base[n] = nullptr;
        scale_base[n] = nullptr;
        continue;
      }
      init_fp8_kvcache_tile_rows(
          fp8_storage, block_size, storage_block_stride_bytes, static_cast<int64_t>(ind[nb * 16 + n]), row_base[n],
          scale_base[n]);
    }
    auto load_vec = [&row_base, &scale_base](int n, int64_t idx, int dim_offset) {
      UNUSED(idx);
      return load_fp8_kvcache_32_from_row(row_base[n], scale_base[n], dim_offset);
    };
    for (int kb = 0; kb < KBv; ++kb) {
      sparse_pack_vnni_Nx32<true, scalar_t, index_t>(
          dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2,
          dst1 + ((nb * 16) >> 1) * ld_dst1 * 2 + kb * 32 * 2,
          ind + nb * 16,
          valid_mask + nb * 16,
          nb_size,
          kb * 32,
          ld_dst0,
          ld_dst1,
          load_vec);
    }
    for (int kb = KBv; kb < KB; ++kb) {
      sparse_pack_vnni_Nx32<false, scalar_t, index_t>(
          dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2,
          dst1 + ((nb * 16) >> 1) * ld_dst1 * 2 + kb * 32 * 2,
          ind + nb * 16,
          valid_mask + nb * 16,
          nb_size,
          kb * 32,
          ld_dst0,
          ld_dst1,
          load_vec);
    }
  }
}

template <typename scalar_t>
inline void fmla_fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void fmla_copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    if constexpr (col % 2 == 0) {
      fVec a0 = fVec::loadu(input + col * 16);
      fVec a1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a0, a1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void fmla_finalize_out(scalar_t* __restrict__ out, const float* __restrict__ acc, float inv_s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(inv_s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a0 = fVec::loadu(acc + d) * s_fvec;
    fVec a1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a0, a1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * inv_s);
  }
}

// ---------------------------------------------------------------------------
// query    : [num_tokens, H_q, D_qk]      bf16 (SGLang's `[B, S_q, H_q, D]`
//            with S_q folded away -- vLLM's flat per-token convention)
// indices  : [num_tokens, topk_main]      int64 (flat, block-table-resolved
//            slot ids, -1 = invalid; SGLang's "main"/window index set)
// extra_*  : same, for the optional "extra"/compressed index set
// attn_sink: [H_q]                        float32
// output   : [num_tokens, H_q, D_v]       bf16
// ---------------------------------------------------------------------------

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void sparse_mla_decode_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ lse_out,
    const scalar_t* __restrict__ query,
    const uint8_t* __restrict__ k_main_fp8,
    const uint8_t* __restrict__ k_extra_fp8,
    const index_t* __restrict__ indices,
    const index_t* __restrict__ extra_indices,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ buffer,
    int64_t batches,
    int64_t s_q,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t topk_main,
    int64_t topk_extra,
    int64_t total_tokens_main,
    int64_t total_tokens_extra,
    int64_t q_strideB,
    int64_t q_strideH,
    int64_t k_main_block_size,
    int64_t k_extra_block_size,
    int64_t k_main_block_stride_bytes,
    int64_t k_extra_block_stride_bytes,
    int64_t idx_strideB,
    int64_t extra_idx_strideB,
    float scaling,
    int64_t buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;

  // partition heads
  constexpr int64_t kBLOCK_H_MAX = 16;
  const int64_t BLOCK_H = (batches * s_q) >= 16 ? kBLOCK_H_MAX : 8;
  const int64_t num_h_blocks = div_up(num_heads, BLOCK_H);

  // parallel on [B, S_q, head_block] -- with s_q == 1 (vLLM's flat call
  // convention) this collapses to a plain [num_tokens, head_block] loop,
  // one flat iteration per token with no padding to skip.
  at::parallel_for(0, batches * s_q * num_h_blocks, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, sq{0}, hb{0};
    data_index_init(begin, bs, batches, sq, s_q, hb, num_h_blocks);

    int tid = at::get_thread_num();
    scalar_t* __restrict__ Btmp0 = buffer + tid * buffer_size_per_thread;  // K  packed
    scalar_t* __restrict__ Btmp1 = Btmp0 + BLOCK_N * head_size;            // V  packed
    // f32 V accumulator follows the bf16 packing region (reinterpret cast).
    float* __restrict__ v_acc_local = reinterpret_cast<float*>(Btmp1 + BLOCK_N * head_size_v);
    fmla_fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);  // initialize V padding

    alignas(64) float s_i[kBLOCK_H_MAX * BLOCK_N];
    float* __restrict__ s_delta = s_i;
    alignas(64) scalar_t s_delta2[kBLOCK_H_MAX * BLOCK_N];

    alignas(64) float s_prime[kBLOCK_H_MAX];
    alignas(64) float m_prime[kBLOCK_H_MAX];

    for (int64_t i = begin; i < end; ++i) {
      const int64_t h_start = hb * BLOCK_H;
      const int64_t h_end = std::min(h_start + BLOCK_H, num_heads);
      const int64_t h_size = h_end - h_start;

      const scalar_t* __restrict__ q_ptr = query + bs * q_strideB + h_start * q_strideH;
      const index_t* __restrict__ idx_ptr = indices + bs * idx_strideB;
      const index_t* __restrict__ extra_idx_ptr = extra_indices == nullptr ? nullptr : extra_indices + bs * extra_idx_strideB;

      fmla_fill_stub(s_prime, 0.f, BLOCK_H);
      fmla_fill_stub(m_prime, -std::numeric_limits<float>::infinity(), BLOCK_H);
      for (int64_t h = 0; h < h_size; ++h) {
        fmla_fill_stub(v_acc_local + h * head_size_v, 0.f, head_size_v);
      }

      auto process_cache = [&](const uint8_t* __restrict__ fp8_ptr,
                               const index_t* __restrict__ cur_idx_ptr,
                               int64_t topk_count,
                               int64_t total_tokens,
                               int64_t fp8_block_size,
                               int64_t fp8_block_stride_bytes) {
        if (fp8_ptr == nullptr || cur_idx_ptr == nullptr || topk_count == 0) {
          return;
        }
        const int64_t topk_limit = topk_count;

        for (int64_t n = 0; n < topk_limit; n += BLOCK_N) {
          int64_t n_size = std::min<int64_t>(BLOCK_N, topk_limit - n);
          const int64_t padded_n_size = div_up(int(n_size), TILE_K) * TILE_K;
          bool valid_mask[BLOCK_N];
          bool has_valid = false;
          for (int64_t k = 0; k < n_size; ++k) {
            const bool valid = is_valid_sparse_index(cur_idx_ptr[n + k], n + k, topk_limit, total_tokens);
            valid_mask[k] = valid;
            has_valid |= valid;
          }
          if (!has_valid) {
            continue;
          }

          // Pack K (BLOCK_N rows via gather) into Btmp0 (key, vnni) and Btmp1
          // (value, vnni), dequantizing FP8 cache entries while packing.
          sparse_pack_vnni_fp8<scalar_t, index_t>(
              Btmp0, Btmp1, fp8_ptr, cur_idx_ptr + n, valid_mask, static_cast<int>(n_size), static_cast<int>(head_size),
              static_cast<int>(head_size_v), fp8_block_size, fp8_block_stride_bytes, static_cast<int>(BLOCK_N),
              static_cast<int>(head_size_v));

          // Q @ K
          at::native::cpublas::brgemm(
              /* M     */ h_size,
              /* N     */ n_size,
              /* K     */ head_size,
              /* lda   */ q_strideH,
              /* ldb   */ BLOCK_N,
              /* ldc   */ BLOCK_N,
              /* add_C */ false,
              /* A     */ q_ptr,
              /* B     */ Btmp0,
              /* C     */ s_i);

          const Vec scale_vec = Vec(scaling);
          for (int64_t h = 0; h < h_size; ++h) {
            float* row = s_i + h * BLOCK_N;
            at::vec::map<float>([scale_vec](Vec x) { return x * scale_vec; }, row, row, n_size);

            for (int64_t k = 0; k < n_size; ++k) {
              if (!valid_mask[k]) {
                row[k] = -std::numeric_limits<float>::infinity();
              }
            }

            // online softmax update
            float m_i = at::vec::reduce_all<float>([](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, row, n_size);
            m_i = std::max(m_i, m_prime[h]);

            if (!std::isfinite(m_i)) {
              fmla_fill_stub(s_delta + h * BLOCK_N, 0.f, padded_n_size);
              fmla_copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
              continue;
            }

            const float m_delta = std::exp(m_prime[h] - m_i);
            at::vec::map<float>([m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + h * BLOCK_N, row, n_size);

            s_prime[h] *= m_delta;
            s_prime[h] += at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + h * BLOCK_N, n_size);

            m_prime[h] = m_i;

            float scale_m = m_delta;
            at::vec::map<float>(
                [scale_m](Vec x) { return x * Vec(scale_m); }, v_acc_local + h * head_size_v, v_acc_local + h * head_size_v,
                head_size_v);

            fmla_fill_stub(s_delta + h * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
            fmla_copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
          }

          // V' <- s_delta @ V + V'   (accumulate into v_acc_local at f32)
          at::native::cpublas::brgemm(
              /* M     */ h_size,
              /* N     */ head_size_v,
              /* K     */ padded_n_size,
              /* lda   */ BLOCK_N,
              /* ldb   */ head_size_v,
              /* ldc   */ head_size_v,
              /* add_C */ true,
              /* A     */ s_delta2,
              /* B     */ Btmp1,
              /* C     */ v_acc_local);
        }
      };

      process_cache(k_main_fp8, idx_ptr, topk_main, total_tokens_main, k_main_block_size, k_main_block_stride_bytes);
      process_cache(
          k_extra_fp8, extra_idx_ptr, topk_extra, total_tokens_extra, k_extra_block_size, k_extra_block_stride_bytes);

      // Apply attention sink correction directly on the output and lse.
      //   out *= exp(lse_no_sink) / (exp(lse_no_sink) + exp(attn_sink))
      //        = 1 / (1 + exp(attn_sink - lse_no_sink))
      // (Algebraically equivalent to treating attn_sink as a virtual,
      // zero-value extra key inside the softmax.)
      for (int64_t h = 0; h < h_size; ++h) {
        const int64_t hh = h_start + h;
        const bool lonely = !std::isfinite(m_prime[h]) || s_prime[h] == 0.f;
        float lse_val = lonely ? std::numeric_limits<float>::infinity() : (m_prime[h] + std::log(s_prime[h]));
        float inv_s = lonely ? 0.f : (1.f / s_prime[h]);

        if (!lonely && attn_sink != nullptr) {
          const float sink = attn_sink[hh];
          const float corr = 1.f / (1.f + std::exp(sink - lse_val));
          inv_s *= corr;
        }

        scalar_t* out_row = output + bs * (s_q * num_heads * head_size_v) + sq * (num_heads * head_size_v) + hh * head_size_v;
        if (lonely) {
          fmla_fill_stub(out_row, 0.f, head_size_v);
        } else {
          fmla_finalize_out<scalar_t>(out_row, v_acc_local + h * head_size_v, inv_s, head_size_v);
        }

        lse_out[bs * num_heads * s_q + hh * s_q + sq] = lse_val;
      }

      data_index_step(bs, batches, sq, s_q, hb, num_h_blocks);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

void flash_mla_with_kvcache_cpu(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& window_cache_2d,
    at::Tensor& window_slots,
    int64_t window_block_size,
    at::Tensor& compressed_cache_2d,
    at::Tensor& compressed_slots,
    int64_t compressed_block_size,
    at::Tensor& attn_sink,
    double scale) {
  TORCH_CHECK(
      q.dim() == 3 && q.size(2) == kHeadDim,
      "flash_mla_with_kvcache_cpu: q must be [num_tokens, num_heads, 512]");
  TORCH_CHECK(
      out.dim() == 3 && out.size(0) == q.size(0) && out.size(1) == q.size(1) && out.size(2) == kHeadDim,
      "flash_mla_with_kvcache_cpu: out must be [num_tokens, num_heads, 512]");
  TORCH_CHECK(
      out.scalar_type() == at::kBFloat16 && out.is_contiguous(),
      "flash_mla_with_kvcache_cpu: out must be bf16 and contiguous "
      "(pre-allocate once in Python -- see DeepseekV4CPUAttention.forward_mqa)");
  TORCH_CHECK(out.device().is_cpu(), "flash_mla_with_kvcache_cpu: out must be a CPU tensor");
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16,
      "flash_mla_with_kvcache_cpu: q must be bf16 (the AMX/VNNI kernel's "
      "FP8->bf16 dequant intrinsics are bf16-specific)");
  TORCH_CHECK(
      window_slots.dim() == 2 && window_slots.size(0) == q.size(0),
      "flash_mla_with_kvcache_cpu: window_slots must be [num_tokens, num_window]");
  TORCH_CHECK(
      compressed_slots.dim() == 2 && compressed_slots.size(0) == q.size(0),
      "flash_mla_with_kvcache_cpu: compressed_slots must be [num_tokens, num_compressed]");
  TORCH_CHECK(
      window_slots.scalar_type() == at::kLong && compressed_slots.scalar_type() == at::kLong,
      "flash_mla_with_kvcache_cpu: slot tensors must be int64");
  TORCH_CHECK(
      attn_sink.numel() == q.size(1), "flash_mla_with_kvcache_cpu: attn_sink must have num_heads entries");
  TORCH_CHECK(
      attn_sink.scalar_type() == at::kFloat && attn_sink.is_contiguous(),
      "flash_mla_with_kvcache_cpu: attn_sink must be float32 and contiguous "
      "(convert once in Python -- see DeepseekV4CPUAttention.forward_mqa)");
  TORCH_CHECK(
      window_slots.is_contiguous() && compressed_slots.is_contiguous(),
      "flash_mla_with_kvcache_cpu: window_slots/compressed_slots must be contiguous");
  TORCH_CHECK(
      q.device().is_cpu() && window_cache_2d.device().is_cpu() && window_slots.device().is_cpu() &&
          compressed_cache_2d.device().is_cpu() && compressed_slots.device().is_cpu() &&
          attn_sink.device().is_cpu(),
      "flash_mla_with_kvcache_cpu: all inputs must be CPU tensors");

  const int64_t num_tokens = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t num_window = window_slots.size(1);
  const int64_t num_compressed = compressed_slots.size(1);
  TORCH_CHECK(num_window > 0, "flash_mla_with_kvcache_cpu: window_slots must have at least one column");

  // Row stride is threaded through explicitly rather than assumed to equal
  // block_size * (token_data_bytes + scale_bytes_per_token): a paged/pooled
  // cache buffer's physical row stride is not guaranteed to match the
  // logical per-block byte count (matches store_cache.cpp's precedent).
  const int64_t window_cache_row_stride = window_cache_2d.stride(0);
  const int64_t compressed_cache_row_stride = num_compressed > 0 ? compressed_cache_2d.stride(0) : 0;

  // AMX/VNNI path: flatten (batch, s_q) -> batches = num_tokens, s_q = 1.
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  constexpr int64_t BLOCK_N = 128;  // multiple of 32 for AMX brgemm
  constexpr int64_t kBLOCK_H_MAX = 16;
  static_assert(kHeadDim % 32 == 0, "head_dim must be a multiple of 32");

  const int num_threads = at::get_num_threads();
  // Layout per thread (in bf16 elements):
  //   [Btmp0 : BLOCK_N * D] [Btmp1 : BLOCK_N * D] [v_acc_local : kBLOCK_H_MAX * D floats]
  // f32 takes 2 bf16 elements -> multiply by 2.
  const int64_t buffer_size_per_thread = BLOCK_N * kHeadDim + BLOCK_N * kHeadDim + 2 * kBLOCK_H_MAX * kHeadDim;
  auto buffer = at::empty({num_threads, buffer_size_per_thread}, q.options());
  // lse is an internal scratch (attn_sink correction needs it per-token/head)
  // -- not surfaced to the Python call site, which only consumes `out`.
  auto lse = at::empty({num_tokens, num_heads}, q.options().dtype(at::kFloat));

  const int64_t q_strideB = q.stride(0);
  const int64_t q_strideH = q.stride(1);
  const int64_t idx_strideB = window_slots.stride(0);
  const int64_t extra_idx_strideB = num_compressed > 0 ? compressed_slots.stride(0) : 0;

  const int64_t total_tokens_main = window_cache_2d.size(0) * window_block_size;
  const int64_t total_tokens_extra = num_compressed > 0 ? compressed_cache_2d.size(0) * compressed_block_size : 0;

  sparse_mla_decode_kernel_impl<at::BFloat16, int64_t, BLOCK_N>(
      out.data_ptr<at::BFloat16>(),
      lse.data_ptr<float>(),
      q.data_ptr<at::BFloat16>(),
      window_cache_2d.data_ptr<uint8_t>(),
      num_compressed > 0 ? compressed_cache_2d.data_ptr<uint8_t>() : nullptr,
      window_slots.data_ptr<int64_t>(),
      num_compressed > 0 ? compressed_slots.data_ptr<int64_t>() : nullptr,
      attn_sink.data_ptr<float>(),
      buffer.data_ptr<at::BFloat16>(),
      /* batches */ num_tokens,
      /* s_q     */ 1,
      num_heads,
      /* head_size   */ kHeadDim,
      /* head_size_v */ kHeadDim,
      /* topk_main */ num_window,
      /* topk_extra */ num_compressed,
      total_tokens_main,
      total_tokens_extra,
      q_strideB,
      q_strideH,
      window_block_size,
      num_compressed > 0 ? compressed_block_size : 1,
      window_cache_row_stride,
      compressed_cache_row_stride,
      idx_strideB,
      extra_idx_strideB,
      static_cast<float>(scale),
      buffer_size_per_thread);
}

