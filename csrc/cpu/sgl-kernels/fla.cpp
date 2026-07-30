// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

namespace {

// [NOTE] GDN Optimizations on AMX CPU
//   * intra loop: fuse `kkt_solve` and `recompute_w_u` so as to avoid materialize `A`.
//   * inter loop: fuse `recompute_w_u` and `update_v` so as to avoid materialize `h` and `v_new`.
//   * intra loop parallel on H instead of Hv, remove duplicated key @ key.T
//   * fuse format pack with elemwise OP as much as possible.
//   * update state (FP32) with amx-bf16 where C(FP32) += A(BF16) * B(BF16)
//   * compile time mask out upper triangular part in decay mask and tril solve, reduce fma needed.

// * convert to vnni format， expect contiguous input and output
//     from [K/2, 2, N] FP32 to [K/2, N, 2] BF16
// * update src = src * exp(g_last)
template <typename scalar_t, int K, int N>
void pack_vnni2(scalar_t* __restrict__ dst, float* __restrict__ src, const float g_last, int ld_src, int ld_dst) {
  static_assert(K % 32 == 0);
  static_assert(N % 32 == 0);

  const float scale = std::exp(g_last);
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int KB = K / 2;
  constexpr int NB = N / 32;

  __m512i s0, s1, d0, d1;
  __m512 vd = _mm512_set1_ps(scale);

  const auto trans = [&](auto i) {
    constexpr int kb = i / NB;
    constexpr int nb = i % NB;

    // [K/2, 2, N/32, 32] -> [K/2, N/32, 32, 2]
    constexpr int k0 = kb * 2 + 0;
    constexpr int k1 = kb * 2 + 1;
    __m512 v00 = _mm512_loadu_ps(src + k0 * ld_src + nb * 32);
    __m512 v01 = _mm512_loadu_ps(src + k0 * ld_src + nb * 32 + 16);
    __m512 v10 = _mm512_loadu_ps(src + k1 * ld_src + nb * 32);
    __m512 v11 = _mm512_loadu_ps(src + k1 * ld_src + nb * 32 + 16);
    s0 = (__m512i)_mm512_cvtne2ps_pbh(v01, v00);
    s1 = (__m512i)_mm512_cvtne2ps_pbh(v11, v10);

    std::tie(d0, d1) = transpose_2x32_16bit(s0, s1);
    _mm512_storeu_si512(dst + kb * ld_dst * 2 + nb * 32 * 2, d0);
    _mm512_storeu_si512(dst + kb * ld_dst * 2 + nb * 32 * 2 + 32, d1);

    // update src = src * exp(g_last)
    _mm512_storeu_ps(src + k0 * ld_src + nb * 32, _mm512_mul_ps(v00, vd));
    _mm512_storeu_ps(src + k0 * ld_src + nb * 32 + 16, _mm512_mul_ps(v01, vd));
    _mm512_storeu_ps(src + k1 * ld_src + nb * 32, _mm512_mul_ps(v10, vd));
    _mm512_storeu_ps(src + k1 * ld_src + nb * 32 + 16, _mm512_mul_ps(v11, vd));
  };
  Unroll<KB * NB>{}(trans);
#else
  // [K/2, 2, N] -> [K/2, N, 2]
  for (int k = 0; k < K; k += 2) {
    for (int n = 0; n < N; ++n) {
      const float v0 = src[(k + 0) * ld_src + n];
      const float v1 = src[(k + 1) * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = static_cast<scalar_t>(v0);
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = static_cast<scalar_t>(v1);
      src[(k + 0) * ld_src + n] = v0 * scale;
      src[(k + 1) * ld_src + n] = v1 * scale;
    }
  }
#endif
}

template <typename scalar_t, int SIZE>
inline void fill_stub(scalar_t* __restrict__ out, float val) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  static_assert(SIZE % kVecSize == 0);
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
#pragma GCC unroll 8
  for (int d = 0; d < SIZE; d += kVecSize) {
    data_vec.store(out + d);
  }
}

// Portable fallback for non-AVX512 builds (ARM/NEON, old x86 without
// AVX512BF16), vectorized via at::vec::Vectorized<T> (portable across
// AVX2/NEON/generic) mirroring the idioms used elsewhere in this file (see
// l2norm_fwd_kernel_impl's predecessor and fused_gdn_gating_kernel_impl):
// bVec/fVec pairs with convert_to_float/convert_from_float for bf16<->float,
// plain scalar tails for remainders. Two kernels (cumsum_kernel,
// update_key_kernel) write a transposed layout relative to their vectorized
// read axis; those vectorize the load/compute and unpack lanes for the
// (unavoidably strided) store.
template <typename scalar_t, int D, bool has_scale>
struct l2norm_kernel {
  static inline void apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float eps) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int bVecSize = bVec::size();
    constexpr float scale = 1.f / std::sqrt(static_cast<float>(D));

    fVec sum_fvec0(0.f), sum_fvec1(0.f);
    int d = 0;
    for (; d <= D - bVecSize; d += bVecSize) {
      bVec in_bvec = bVec::loadu(input + d);
      fVec in0, in1;
      std::tie(in0, in1) = at::vec::convert_to_float(in_bvec);
      sum_fvec0 = sum_fvec0 + in0 * in0;
      sum_fvec1 = sum_fvec1 + in1 * in1;
    }
    float sqsum = vec_reduce_sum(sum_fvec0 + sum_fvec1);
    for (; d < D; ++d) {
      float v = static_cast<float>(input[d]);
      sqsum += v * v;
    }

    float rscale = 1.f / std::sqrt(sqsum + eps);
    fVec rscale_fvec(rscale);
    fVec scale_fvec(scale);
    d = 0;
    for (; d <= D - bVecSize; d += bVecSize) {
      bVec in_bvec = bVec::loadu(input + d);
      fVec in0, in1;
      std::tie(in0, in1) = at::vec::convert_to_float(in_bvec);
      in0 = in0 * rscale_fvec;
      in1 = in1 * rscale_fvec;
      if constexpr (has_scale) {
        in0 = in0 * scale_fvec;
        in1 = in1 * scale_fvec;
      }
      bVec out_bvec = at::vec::convert_from_float<scalar_t>(in0, in1);
      out_bvec.store(out + d);
    }
    for (; d < D; ++d) {
      float v = static_cast<float>(input[d]) * rscale;
      if constexpr (has_scale) {
        v *= scale;
      }
      out[d] = static_cast<scalar_t>(v);
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D, bool has_scale>
struct l2norm_kernel<at::BFloat16, D, has_scale> {
  static inline void apply(at::BFloat16* __restrict__ out, const at::BFloat16* __restrict__ input, float eps) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 32;

    __m512bh va[COLS];
    __m512 vrscale;

    constexpr float scale = 1.f / std::sqrt(D);
    __m512 vscale = _mm512_set1_ps(scale);

    // step 1: load input and do reduce with avx512-bf16
    __m512 vsum = _mm512_set1_ps(0.f);
    auto reduce = [&](auto col) {
      va[col] = (__m512bh)(_mm512_loadu_si512(input + col * 32));
      vsum = _mm512_dpbf16_ps(vsum, va[col], va[col]);
    };
    Unroll<COLS>{}(reduce);

    float sqsum = _mm512_reduce_add_ps(vsum);
    float rscale = 1.f / std::sqrt(sqsum + eps);
    vrscale = _mm512_set1_ps(rscale);

    // step 2: apply scale to output
    auto map = [&](auto col) {
      __m512i a16 = (__m512i)va[col];
      __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
      __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
      va0 = _mm512_mul_ps(va0, vrscale);
      va1 = _mm512_mul_ps(va1, vrscale);
      // keep the mul order same as torch code:
      //   query = l2norm(query) * scale
      if constexpr (has_scale) {
        va0 = _mm512_mul_ps(va0, vscale);
        va1 = _mm512_mul_ps(va1, vscale);
      }
      _mm512_storeu_si512(out + col * 32, (__m512i)(_mm512_cvtne2ps_pbh(va1, va0)));
    };
    Unroll<COLS>{}(map);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int BLOCK_H>
struct cumsum_kernel {
  static inline void apply(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      int mb_size,
      int hb_size,
      int ld_src,
      int ld_dst) {
    // out: [hb_size valid rows, CHUNK_SIZE] within a [BLOCK_H, ld_dst] buffer
    // input: [mb_size valid rows, hb_size] within a [CHUNK_SIZE(padded), ld_src] buffer
    // input is contiguous along j (vectorize the load/accumulate); out is
    // contiguous along i instead (transposed), so the store is per-lane.
    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int VecSize = Vec::size();
    alignas(64) scalar_t lane_buf[VecSize];
    int j = 0;
    for (; j <= hb_size - VecSize; j += VecSize) {
      Vec running(static_cast<scalar_t>(0));
      for (int i = 0; i < CHUNK_SIZE; ++i) {
        if (i < mb_size) {
          running = running + Vec::loadu(input + i * ld_src + j);
        }
        running.store(lane_buf);
        for (int lane = 0; lane < VecSize; ++lane) {
          out[(j + lane) * ld_dst + i] = lane_buf[lane];
        }
      }
    }
    for (; j < hb_size; ++j) {
      float running = 0.f;
      for (int i = 0; i < CHUNK_SIZE; ++i) {
        if (i < mb_size) {
          running += static_cast<float>(input[i * ld_src + j]);
        }
        out[j * ld_dst + i] = static_cast<scalar_t>(running);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int BLOCK_H>
struct cumsum_kernel<float, CHUNK_SIZE, BLOCK_H> {
  static inline void
  apply(float* __restrict__ out, const float* __restrict__ input, int mb_size, int hb_size, int ld_src, int ld_dst) {
    // vector length of fp32 for avx512
    static_assert(BLOCK_H == 16);
    TORCH_CHECK(hb_size > 0 && hb_size <= BLOCK_H);
    const __mmask16 vmask = static_cast<__mmask16>((1u << hb_size) - 1u);

    __m512i va[16];
    __m512 vsum = _mm512_set1_ps(0.f);

    for (int i = 0; i < CHUNK_SIZE; i += 16) {
      // load input data
      Unroll<16>{}([&](auto j) {
        __m512 v;
        if (i + j < mb_size) {
          v = _mm512_maskz_loadu_ps(vmask, input + (i + j) * ld_src);
        } else {
          v = _mm512_setzero_ps();
        }
        vsum = _mm512_add_ps(vsum, v);
        va[j] = _mm512_castps_si512(vsum);
      });
      // transpose
      transpose_16x16_32bit(va);
      // store output data
      Unroll<16>{}([&](auto j) {
        if (j < hb_size) {
          _mm512_storeu_si512(out + j * ld_dst + i, va[j]);
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE>
struct decay_mask_kernel {
  // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
  static inline void apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input) {
    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int VecSize = Vec::size();
    const Vec zero(static_cast<scalar_t>(0));
    for (int row = 0; row < CHUNK_SIZE; ++row) {
      Vec g_row(input[row]);
      Vec limit_vec(static_cast<scalar_t>(row));
      int col = 0;
      for (; col <= CHUNK_SIZE - VecSize; col += VecSize) {
        Vec g_col = Vec::loadu(input + col);
        Vec vc = (g_row - g_col).exp_u20();
        Vec idx = Vec::arange(static_cast<scalar_t>(col), static_cast<scalar_t>(1));
        Vec result = Vec::blendv(zero, vc, idx <= limit_vec);
        result.store(out + row * CHUNK_SIZE + col);
      }
      for (; col < CHUNK_SIZE; ++col) {
        out[row * CHUNK_SIZE + col] =
            col <= row
                ? static_cast<scalar_t>(std::exp(static_cast<float>(input[row]) - static_cast<float>(input[col])))
                : static_cast<scalar_t>(0);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE>
struct decay_mask_kernel<float, CHUNK_SIZE> {
  static inline void apply(float* __restrict__ out, const float* __restrict__ input) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int ROWS = CHUNK_SIZE;
    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 va;
    __m512 vb[COLS];

    // step 1: load g[j]
    auto loadb = [&](auto i) { vb[i] = _mm512_loadu_ps(input + i * 16); };
    Unroll<COLS>{}(loadb);

    // step2: exp(g[i] - g[j])
    auto compute = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_ps(input[row]);
      }

      // mask vb[col] (already loaded in step 1) for the lower-triangular region
      constexpr int len = std::max(0, std::min(row + 1 - col * 16, 16));

      __m512 vc;
      if constexpr (len == 16) {
        vc = _mm512_fexp_u20_ps(va - vb[col]);
      } else if constexpr (len == 0) {
        vc = _mm512_setzero_ps();
      } else {
        vc = _mm512_fexp_u20_ps(va - vb[col]);
        // do mask for vc
        constexpr __mmask16 vmask = (1 << len) - 1;
        vc = _mm512_mask_blend_ps(vmask, _mm512_setzero_ps(), vc);
      }
      _mm512_storeu_ps(out + row * CHUNK_SIZE + col * 16, vc);
    };
    Unroll<ROWS * COLS>{}(compute);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, bool has_beta>
struct apply_mask_kernel {
  // has_beta:  attn2 = -attn * beta * d  (strict lower, col < row)
  // !has_beta: attn2 = attn * d         (lower incl. diagonal, col <= row)
  static inline void apply(
      scalar_t* __restrict__ attn2,
      const float* __restrict__ attn,
      const scalar_t* __restrict__ beta,
      const float* __restrict__ d,
      int size,
      int b_stride = 0) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();
    const fVec zero(0.f);
    for (int row = 0; row < size; ++row) {
      int col_limit = has_beta ? row : row + 1;
      float beta_val = 1.f;
      if constexpr (has_beta) {
        beta_val = -static_cast<float>(beta[row * b_stride]);
      }
      fVec beta_fvec(beta_val);
      fVec limit_fvec(static_cast<float>(col_limit));
      int col = 0;
      for (; col <= CHUNK_SIZE - bVecSize; col += bVecSize) {
        fVec a0 = fVec::loadu(attn + row * CHUNK_SIZE + col);
        fVec a1 = fVec::loadu(attn + row * CHUNK_SIZE + col + fVecSize);
        fVec d0 = fVec::loadu(d + row * CHUNK_SIZE + col);
        fVec d1 = fVec::loadu(d + row * CHUNK_SIZE + col + fVecSize);
        fVec v0 = a0 * beta_fvec * d0;
        fVec v1 = a1 * beta_fvec * d1;
        fVec idx0 = fVec::arange(static_cast<float>(col), 1.f);
        fVec idx1 = fVec::arange(static_cast<float>(col + fVecSize), 1.f);
        v0 = fVec::blendv(zero, v0, idx0 < limit_fvec);
        v1 = fVec::blendv(zero, v1, idx1 < limit_fvec);
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(v0, v1);
        out_bvec.store(attn2 + row * CHUNK_SIZE + col);
      }
      for (; col < CHUNK_SIZE; ++col) {
        float v = 0.f;
        if (col < col_limit) {
          v = attn[row * CHUNK_SIZE + col] * beta_val * d[row * CHUNK_SIZE + col];
        }
        attn2[row * CHUNK_SIZE + col] = static_cast<scalar_t>(v);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, bool has_beta>
struct apply_mask_kernel<at::BFloat16, CHUNK_SIZE, has_beta> {
  static inline void apply(
      at::BFloat16* __restrict__ attn2,
      const float* __restrict__ attn,
      const at::BFloat16* __restrict__ beta,
      const float* __restrict__ d,
      int size,
      int b_stride = 0) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int ROWS = CHUNK_SIZE;
    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 vbeta;

    // has_beta: attn2 = -attn * beta * d  (strict lower)
    // !has_beta: attn2 = attn * d         (lower incl. diagonal)
    auto compute = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      constexpr int len =
          has_beta ? std::max(0, std::min(row - col * 16, 16)) : std::max(0, std::min(row + 1 - col * 16, 16));
      if (row < size) {
        if constexpr (has_beta) {
          if constexpr (col == 0) {
            vbeta = _mm512_set1_ps(-static_cast<float>(beta[row * b_stride]));
          }
        }

        __m512 vc;
        if constexpr (len == 0) {
          vc = _mm512_setzero_ps();
        } else {
          constexpr __mmask16 vmask = (1 << len) - 1;
          __m512 va = _mm512_maskz_loadu_ps(vmask, attn + row * CHUNK_SIZE + col * 16);
          __m512 vd = _mm512_maskz_loadu_ps(vmask, d + row * CHUNK_SIZE + col * 16);
          if constexpr (has_beta) {
            vc = _mm512_mul_ps(_mm512_mul_ps(va, vbeta), vd);
          } else {
            vc = _mm512_mul_ps(va, vd);
          }
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(attn2 + row * CHUNK_SIZE + col * 16), (__m256i)(_mm512_cvtneps_pbh(vc)));
      }
    };
    Unroll<ROWS * COLS>{}(compute);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE>
struct solve_tril_kernel {
  // (I + L)^{-1} via forward substitution, L = strict-lower part of attn2.
  static inline void apply(scalar_t* __restrict__ attn2, int size) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();

    // for len == 0 and row < size, we don't have to write back zero again
    // as in `apply_mask_kernel`, we already set zero for the upper-triangular region
    for (int i = 1; i < size; ++i) {
      scalar_t* __restrict__ row_ptr = attn2 + i * CHUNK_SIZE;
      float vsum[CHUNK_SIZE];
      int j = 0;
      for (; j <= i - bVecSize; j += bVecSize) {
        bVec row_bvec = bVec::loadu(row_ptr + j);
        fVec f0, f1;
        std::tie(f0, f1) = at::vec::convert_to_float(row_bvec);
        f0.store(vsum + j);
        f1.store(vsum + j + fVecSize);
      }
      for (; j < i; ++j) {
        vsum[j] = static_cast<float>(row_ptr[j]);
      }

      // row = attn[..., i, :i].clone()
      // sub = attn[..., :i, :i].clone()
      // vsum = row + (row.unsqueeze(-1) * sub).sum(-2)
      for (int k = 0; k < i; ++k) {
        // read BEFORE row_ptr is written back below (row k was finalized in
        // an earlier outer iteration; row i itself is untouched until the
        // final write-back after this loop)
        float va = static_cast<float>(row_ptr[k]);
        fVec va_vec(va);
        const scalar_t* __restrict__ row_k_ptr = attn2 + k * CHUNK_SIZE;
        int jj = 0;
        for (; jj <= k - bVecSize; jj += bVecSize) {
          bVec rk_bvec = bVec::loadu(row_k_ptr + jj);
          fVec rk0, rk1;
          std::tie(rk0, rk1) = at::vec::convert_to_float(rk_bvec);
          fVec vsum0 = fVec::loadu(vsum + jj);
          fVec vsum1 = fVec::loadu(vsum + jj + fVecSize);
          vsum0 = vsum0 + va_vec * rk0;
          vsum1 = vsum1 + va_vec * rk1;
          vsum0.store(vsum + jj);
          vsum1.store(vsum + jj + fVecSize);
        }
        for (; jj < k; ++jj) {
          vsum[jj] += va * static_cast<float>(row_k_ptr[jj]);
        }
      }

      j = 0;
      for (; j <= i - bVecSize; j += bVecSize) {
        fVec f0 = fVec::loadu(vsum + j);
        fVec f1 = fVec::loadu(vsum + j + fVecSize);
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(f0, f1);
        out_bvec.store(row_ptr + j);
      }
      for (; j < i; ++j) {
        row_ptr[j] = static_cast<scalar_t>(vsum[j]);
      }
    }

    // attn = attn + torch.eye(chunk_size)
    for (int i = 0; i < size; ++i) {
      attn2[i * CHUNK_SIZE + i] = static_cast<scalar_t>(static_cast<float>(attn2[i * CHUNK_SIZE + i]) + 1.f);
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE>
struct solve_tril_kernel<at::BFloat16, CHUNK_SIZE> {
  static inline void apply(at::BFloat16* __restrict__ attn2, int size) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 va;
    __m512 vb[COLS];
    __m512 vsum[COLS];

    // for len == 0 and row < size, we don't have to write back zero again
    // as in `apply_mask_kernel`, we already set zero for the upper-triangular region
    for (int i = 1; i < size; ++i) {
      // load row attn[..., i, :i]
      at::BFloat16* __restrict__ row_ptr = attn2 + i * CHUNK_SIZE;
      Unroll<COLS>{}([&](auto col) {
        int len = std::min(i - col * 16, 16);
        if (len > 0) {
          const __mmask16 vmask = (1 << len) - 1;
          vsum[col] = CVT_BF16_TO_FP32(_mm256_maskz_loadu_epi16(vmask, row_ptr + col * 16));
        }
      });

      // row = attn[..., i, :i].clone()
      // sub = attn[..., :i, :i].clone()
      // vsum = row + (row.unsqueeze(-1) * sub).sum(-2)
      for (int k = 0; k < i; ++k) {
        va = _mm512_set1_ps(static_cast<float>(row_ptr[k]));

        const at::BFloat16* __restrict__ row_k_ptr = attn2 + k * CHUNK_SIZE;
        Unroll<COLS>{}([&](auto col) {
          int len = std::min(k - col * 16, 16);
          if (len > 0) {
            const __mmask16 vmask = (1 << len) - 1;
            vb[col] = CVT_BF16_TO_FP32(_mm256_maskz_loadu_epi16(vmask, row_k_ptr + col * 16));
            vsum[col] = _mm512_fmadd_ps(va, vb[col], vsum[col]);
          }
        });
      }

      // attn[..., i, :i] = vsum
      Unroll<COLS>{}([&](auto col) {
        int len = std::min(i - col * 16, 16);
        if (len > 0) {
          const __mmask16 vmask = (1 << len) - 1;
          _mm256_mask_storeu_epi16(row_ptr + col * 16, vmask, (__m256i)(_mm512_cvtneps_pbh(vsum[col])));
        }
      });
    }

    // attn = attn + torch.eye(chunk_size)
    for (int i = 0; i < size; ++i) {
      attn2[i * CHUNK_SIZE + i] += 1.f;
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int D, bool has_beta, bool has_g>
struct apply_beta_kernel {
  static inline void apply(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ beta,
      const float* __restrict__ g,
      int size,
      int ld_src,
      int ld_dst,
      int b_stride) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();

    for (int i = 0; i < size; ++i) {
      float scale = 1.f;
      if constexpr (has_beta) {
        scale *= static_cast<float>(beta[i * b_stride]);
      }
      if constexpr (has_g) {
        scale *= std::exp(g[i]);
      }
      fVec scale_fvec(scale);
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        bVec in_bvec = bVec::loadu(input + i * ld_src + d);
        fVec in0, in1;
        std::tie(in0, in1) = at::vec::convert_to_float(in_bvec);
        in0 = in0 * scale_fvec;
        in1 = in1 * scale_fvec;
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(in0, in1);
        out_bvec.store(out + i * ld_dst + d);
      }
      for (; d < D; ++d) {
        out[i * ld_dst + d] = static_cast<scalar_t>(static_cast<float>(input[i * ld_src + d]) * scale);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int D, bool has_beta, bool has_g>
struct apply_beta_kernel<at::BFloat16, CHUNK_SIZE, D, has_beta, has_g> {
  static inline void apply(
      at::BFloat16* __restrict__ out,
      const at::BFloat16* __restrict__ input,
      const at::BFloat16* __restrict__ beta,
      const float* __restrict__ g,
      int size,
      int ld_src,
      int ld_dst,
      int b_stride) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    // get g.exp() and g is padded to CHUNK_SIZE
    alignas(64) float g_arr[CHUNK_SIZE];
    if constexpr (has_g) {
      Unroll<CHUNK_SIZE / 16>{}([&](auto col) {
        __m512 vg = _mm512_loadu_ps(g + col * 16);
        __m512 vg_exp = _mm512_fexp_u20_ps(vg);
        _mm512_storeu_ps(g_arr + col * 16, vg_exp);
      });
    }

    for (int i = 0; i < size; ++i) {
      __m512 vbeta;
      if constexpr (has_beta) {
        vbeta = _mm512_set1_ps(static_cast<float>(beta[i * b_stride]));
      }
      __m512 vg;
      if constexpr (has_g) {
        vg = _mm512_set1_ps(g_arr[i]);
      }

      Unroll<COLS>{}([&](auto col) {
        // load for 0, 2, 4, 6
        if constexpr (col % 2 == 0) {
          __m512i a16 = _mm512_loadu_si512(input + i * ld_src + col * 16);
          __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
          __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
          if constexpr (has_beta) {
            va0 = _mm512_mul_ps(va0, vbeta);
            va1 = _mm512_mul_ps(va1, vbeta);
          }
          if constexpr (has_g) {
            va0 = _mm512_mul_ps(va0, vg);
            va1 = _mm512_mul_ps(va1, vg);
          }
          _mm512_storeu_si512(out + i * ld_dst + col * 16, (__m512i)(_mm512_cvtne2ps_pbh(va1, va0)));
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int D>
struct update_kernel {
  static inline void
  apply(scalar_t* __restrict__ out, const float* __restrict__ input, int size, int ld_src, int ld_dst) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();

    for (int i = 0; i < size; ++i) {
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        fVec f0 = fVec::loadu(input + i * ld_src + d);
        fVec f1 = fVec::loadu(input + i * ld_src + d + fVecSize);
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(f0, f1);
        out_bvec.store(out + i * ld_dst + d);
      }
      for (; d < D; ++d) {
        out[i * ld_dst + d] = static_cast<scalar_t>(input[i * ld_src + d]);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D>
struct update_kernel<at::BFloat16, D> {
  static inline void
  apply(at::BFloat16* __restrict__ out, const float* __restrict__ input, int size, int ld_src, int ld_dst) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    for (int i = 0; i < size; ++i) {
      Unroll<COLS>{}([&](auto col) {
        if constexpr (col % 2 == 0) {
          __m512 va0 = _mm512_loadu_ps(input + i * ld_src + (col + 0) * 16);
          __m512 va1 = _mm512_loadu_ps(input + i * ld_src + (col + 1) * 16);
          __m512i a16 = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
          _mm512_storeu_si512(out + i * ld_dst + col * 16, a16);
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int D>
struct update_value_kernel {
  static inline void apply(
      scalar_t* __restrict__ v_prime2,
      const scalar_t* __restrict__ v,
      const float* __restrict__ v_prime,
      int size,
      int padded_size,
      int v_strideT) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();

    // v2' = v - v'
    for (int i = 0; i < size; ++i) {
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        bVec v_bvec = bVec::loadu(v + i * v_strideT + d);
        fVec v0, v1;
        std::tie(v0, v1) = at::vec::convert_to_float(v_bvec);
        fVec vp0 = fVec::loadu(v_prime + i * D + d);
        fVec vp1 = fVec::loadu(v_prime + i * D + d + fVecSize);
        v0 = v0 - vp0;
        v1 = v1 - vp1;
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(v0, v1);
        out_bvec.store(v_prime2 + i * D + d);
      }
      for (; d < D; ++d) {
        float val = static_cast<float>(v[i * v_strideT + d]) - v_prime[i * D + d];
        v_prime2[i * D + d] = static_cast<scalar_t>(val);
      }
    }
    // pad the last chunk
    const bVec zero_bvec(static_cast<scalar_t>(0));
    for (int i = size; i < padded_size; ++i) {
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        zero_bvec.store(v_prime2 + i * D + d);
      }
      for (; d < D; ++d) {
        v_prime2[i * D + d] = static_cast<scalar_t>(0);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D>
struct update_value_kernel<at::BFloat16, D> {
  static inline void apply(
      at::BFloat16* __restrict__ v_prime2,
      const at::BFloat16* __restrict__ v,
      const float* __restrict__ v_prime,
      int size,
      int padded_size,
      int v_strideT) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    // v2' = v - v'
    for (int i = 0; i < size; ++i) {
      Unroll<COLS>{}([&](auto col) {
        // load for 0, 2, 4, 6
        if constexpr (col % 2 == 0) {
          __m512i v16 = _mm512_loadu_si512(v + i * v_strideT + col * 16);
          __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(v16, 0));
          __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(v16, 1));

          __m512 v_prime0 = _mm512_loadu_ps(v_prime + i * D + col * 16);
          __m512 v_prime1 = _mm512_loadu_ps(v_prime + i * D + col * 16 + 16);
          va0 = _mm512_sub_ps(va0, v_prime0);
          va1 = _mm512_sub_ps(va1, v_prime1);
          __m512i o16 = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
          _mm512_storeu_si512(v_prime2 + i * D + col * 16, o16);
        }
      });
    }

    // pad the last chunk
    for (int i = size; i < padded_size; ++i) {
      Unroll<COLS>{}([&](auto col) {
        if constexpr (col % 2 == 0) {
          __m512i v16 = _mm512_setzero_si512();
          _mm512_storeu_si512(v_prime2 + i * D + col * 16, v16);
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int D>
struct update_key_kernel {
  static inline void apply(
      scalar_t* __restrict__ k_updated,
      const scalar_t* __restrict__ k,
      const float* __restrict__ g,
      int size,
      int k_strideT) {
    // k_updated is transposed: [D, CHUNK_SIZE], k_updated[d, t] = k[t, d] * exp(g_last - g[t]).
    // k's D dim is contiguous (vectorize load/compute); k_updated's D dim is
    // strided (CHUNK_SIZE apart), so the store is unpacked per lane.
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int fVecSize = fVec::size();
    constexpr int bVecSize = bVec::size();
    alignas(64) scalar_t lane_buf[bVecSize];

    const float g_last = g[size - 1];
    for (int t = 0; t < size; ++t) {
      float scale = std::exp(g_last - g[t]);
      fVec scale_fvec(scale);
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        bVec k_bvec = bVec::loadu(k + t * k_strideT + d);
        fVec k0, k1;
        std::tie(k0, k1) = at::vec::convert_to_float(k_bvec);
        k0 = k0 * scale_fvec;
        k1 = k1 * scale_fvec;
        bVec out_bvec = at::vec::convert_from_float<scalar_t>(k0, k1);
        out_bvec.store(lane_buf);
        for (int lane = 0; lane < bVecSize; ++lane) {
          k_updated[(d + lane) * CHUNK_SIZE + t] = lane_buf[lane];
        }
      }
      for (; d < D; ++d) {
        k_updated[d * CHUNK_SIZE + t] = static_cast<scalar_t>(static_cast<float>(k[t * k_strideT + d]) * scale);
      }
    }
    const bVec zero_bvec(static_cast<scalar_t>(0));
    for (int t = size; t < CHUNK_SIZE; ++t) {
      int d = 0;
      for (; d <= D - bVecSize; d += bVecSize) {
        zero_bvec.store(lane_buf);
        for (int lane = 0; lane < bVecSize; ++lane) {
          k_updated[(d + lane) * CHUNK_SIZE + t] = lane_buf[lane];
        }
      }
      for (; d < D; ++d) {
        k_updated[d * CHUNK_SIZE + t] = static_cast<scalar_t>(0);
      }
    }
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int D>
struct update_key_kernel<at::BFloat16, CHUNK_SIZE, D> {
  static inline void apply(
      at::BFloat16* __restrict__ k_updated,
      const at::BFloat16* __restrict__ k,
      const float* __restrict__ g,
      int size,
      int k_strideT) {
    static_assert(D % 32 == 0);
    const int MB = div_up(size, 16);
    const int KB = D / 16;

    const float g_last = g[size - 1];
    const __m512 vg_last = _mm512_set1_ps(g_last);

    float scale_arr[16];
    __m256i va[16];

    // from [C, D](MB, KB) to [D, C](KB, MB)
    // pad size to 16 in this kernel so that transpose can be done in one loop
    for (int mb = 0; mb < MB; ++mb) {
      const int mb_size = std::min(size - mb * 16, 16);
      // prepare exp(g_last - g)
      __m512 vg = _mm512_loadu_ps(g + mb * 16);
      _mm512_storeu_ps(scale_arr, _mm512_fexp_u20_ps(_mm512_sub_ps(vg_last, vg)));
      for (int kb = 0; kb < KB; ++kb) {
        const at::BFloat16* __restrict__ k_ptr = k + mb * 16 * k_strideT + kb * 16;
        at::BFloat16* __restrict__ k_updated_ptr = k_updated + kb * 16 * CHUNK_SIZE + mb * 16;
        // load 16 regs
        Unroll<16>{}([&](auto m) {
          if (m < mb_size) {
            __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_ptr + m * k_strideT));
            __m512 v32 = _mm512_mul_ps(CVT_BF16_TO_FP32(v16), _mm512_set1_ps(scale_arr[m]));
            va[m] = (__m256i)_mm512_cvtneps_pbh(v32);
          } else {
            va[m] = _mm256_setzero_si256();
          }
        });
        // transpose 16x16
        transpose_16x16_16bit(va);
        // store 16 regs
        Unroll<16>{}(
            [&](auto k) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(k_updated_ptr + k * CHUNK_SIZE), va[k]); });
      }
    }
  }
};
#endif


// template head_dim here to reduce extra read
//   * normal approach: read inputs 2 times:
//     - reduce: 1R
//     - scale: 1R + 1W
//   * keep input data in register:
//     - reduce: 1R
//     - scale: 1W
template <typename scalar_t, int D>
void l2norm_fwd_kernel_impl(
    scalar_t* __restrict__ query_norm,
    scalar_t* __restrict__ key_norm,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float eps,
    int64_t T,
    int64_t H,
    int64_t q_strideT,
    int64_t q_strideH,
    int64_t k_strideT,
    int64_t k_strideH) {
  // expected to be contuguous
  int64_t qn_strideH = D;
  int64_t kn_strideH = D;

  // parallel on [B, T, H]
  at::parallel_for(0, T * H, 0, [&](int64_t begin, int64_t end) {
    int64_t t{0}, h{0};
    data_index_init(begin, t, T, h, H);

    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* __restrict__ q_ptr = query + t * q_strideT + h * q_strideH;
      const scalar_t* __restrict__ k_ptr = key + t * k_strideT + h * k_strideH;
      scalar_t* __restrict__ qn_ptr = query_norm + i * qn_strideH;
      scalar_t* __restrict__ kn_ptr = key_norm + i * kn_strideH;

      l2norm_kernel<scalar_t, D, true>::apply(qn_ptr, q_ptr, eps);
      l2norm_kernel<scalar_t, D, false>::apply(kn_ptr, k_ptr, eps);

      // move to the next index
      data_index_step(t, T, h, H);
    }
  });
}

// g  : [B, T, Hv]
// g_ : [B, Hv, NT, C] -> [B, NT, HB, BLOCK_H, C]
// cu_seqlens : [num_seqs + 1]
// chunk_indices : [NT * 2]
template <typename scalar_t, int CHUNK_SIZE>
void chunk_local_cumsum_kernel_impl(
    scalar_t* __restrict__ g_,
    const scalar_t* __restrict__ g,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_indices,
    int64_t Hv,
    int64_t NT) {
  constexpr int BLOCK_H = 16;
  int64_t HB = div_up(Hv, int64_t(BLOCK_H));

  // parallel on [NT * HB] to increase parallelism
  at::parallel_for(0, NT * HB, 0, [&](int64_t begin, int64_t end) {
    int64_t nt{0}, hb{0};
    data_index_init(begin, nt, NT, hb, HB);

    for (int64_t i = begin; i < end; ++i) {
      int32_t bs = chunk_indices[nt * 2 + 0];
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t mb_start = chunk_indices[nt * 2 + 1] * CHUNK_SIZE;
      int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));
      int64_t hb_size = std::min(Hv - hb * BLOCK_H, int64_t(BLOCK_H));

      const scalar_t* __restrict__ g_ptr = g + (batch_offset + mb_start) * Hv + hb * BLOCK_H;
      scalar_t* __restrict__ gsum_ptr = g_ + nt * (Hv * CHUNK_SIZE) + hb * (BLOCK_H * CHUNK_SIZE);
      cumsum_kernel<scalar_t, CHUNK_SIZE, BLOCK_H>::apply(gsum_ptr, g_ptr, mb_size, hb_size, Hv, CHUNK_SIZE);

      // move to the next index
      data_index_step(nt, NT, hb, HB);
    }
  });
}

#define DECL_BUF(type, name, size_expr) alignas(64) type name[(size_expr)]
#define DECL_ZERO_BUF(type, name, size_expr) \
  DECL_BUF(type, name, size_expr);           \
  fill_stub<type, (size_expr)>(name, 0.f)

// w : [B, T, Hv, D]
// u : [B, T, Hv, Dv]
// d : [B, NT, Hv, C, C]
// k : [B, T, H, D]
// v : [B, T, Hv, Dv]
// g : [B, NT, Hv, C]
// beta : [B, T, Hv]
// cu_seqlens : [num_seqs + 1]
// chunk_indices : [NT * 2]
template <typename scalar_t, int D, int CHUNK_SIZE>
void chunk_gated_delta_rule_fwd_intra_kernel_impl(
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ u,
    float* __restrict__ d,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ g,
    const scalar_t* __restrict__ beta,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_indices,
    int64_t H,
    int64_t Hv,
    int64_t NT,
    int64_t k_strideT,
    int64_t k_strideH,
    int64_t v_strideT,
    int64_t v_strideH) {
  // head group, expect to be 1，2，4 for qwen3.5
  const int64_t HG = Hv / H;

  // strides
  const int64_t w_strideT = Hv * D;
  const int64_t w_strideH = D;
  const int64_t u_strideT = Hv * D;
  const int64_t u_strideH = D;

  // [NB]: parallel on [NT, H]
  //   * parallel on num_heads and go sequential on num_heads_v,
  //   * avoid instantialize k_beta (beta * k)
  //   * compute key @ key^T * beta instead of k_beta @ key^T, same as triton impl
  //   * compute key @ key^T once for each k head index and reuse for v head index
  at::parallel_for(0, NT * H, 0, [&](int64_t begin, int64_t end) {
    int64_t nt{0}, h{0};
    data_index_init(begin, nt, NT, h, H);

    // thread local temp buffer
    DECL_ZERO_BUF(scalar_t, tmp, CHUNK_SIZE * D);
    DECL_ZERO_BUF(scalar_t, tmp2, CHUNK_SIZE * D);
    DECL_ZERO_BUF(float, attn, CHUNK_SIZE* CHUNK_SIZE);
    DECL_ZERO_BUF(scalar_t, attn2, CHUNK_SIZE * CHUNK_SIZE);
    DECL_ZERO_BUF(float, tmp3, CHUNK_SIZE* D);

    // alias
    scalar_t* __restrict__ k_packed = tmp;
    scalar_t* __restrict__ k_beta = tmp;
    scalar_t* __restrict__ v_beta = tmp;
    scalar_t* __restrict__ k_beta_packed = tmp2;
    scalar_t* __restrict__ v_beta_packed = tmp2;
    float* __restrict__ k_updated = tmp3;
    float* __restrict__ v_updated = tmp3;

    for (int64_t i = begin; i < end; ++i) {
      int32_t bs = chunk_indices[nt * 2 + 0];
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t mb_start = chunk_indices[nt * 2 + 1] * CHUNK_SIZE;
      int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));

      // mb_size` is K in 5.c, 5.g, pad to TILE_K;
      const int64_t padded_mb_size = div_up((int)mb_size, TILE_K) * TILE_K;

      // step 1: decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
      for (int64_t hv = h * HG; hv < h * HG + HG; ++hv) {
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * CHUNK_SIZE;
        float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        decay_mask_kernel<float, CHUNK_SIZE>::apply(d_ptr, g_ptr);
      }

      // step 2: attn = key @ key^T
      const scalar_t* __restrict__ k_ptr = k + (batch_offset + mb_start) * k_strideT + h * k_strideH;
      if constexpr (brgemm_supported()) {
        pack_vnni<scalar_t>(
            /*    dst */ k_packed,
            /*    src */ k_ptr,
            /*     N  */ mb_size,
            /*     K  */ D,
            /* ld_src */ k_strideT,
            /* ld_dst */ CHUNK_SIZE);

        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ mb_size,
            /*     K */ D,
            /*   lda */ k_strideT,
            /*   ldb */ CHUNK_SIZE,
            /*   ldc */ CHUNK_SIZE,
            /* add_C */ false,
            /*     A */ k_ptr,
            /*     B */ k_packed,
            /*     C */ attn);
      } else {
        blas_gemm(
            at::native::TransposeType::Transpose,
            at::native::TransposeType::NoTranspose,
            mb_size,
            mb_size,
            D,
            1.0f,
            k_ptr,
            k_strideT,
            k_ptr,
            k_strideT,
            0.0f,
            attn,
            CHUNK_SIZE);
      }

      for (int64_t hv = h * HG; hv < h * HG + HG; ++hv) {
        // step 3: attn2 = -attn * beta * d
        const scalar_t* __restrict__ beta_ptr = beta + (batch_offset + mb_start) * Hv + hv;
        const float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        apply_mask_kernel<scalar_t, CHUNK_SIZE, true>::apply(attn2, attn, beta_ptr, d_ptr, mb_size, Hv);

        // step 4: solve_tril(attn2) -> (I + L)^{-1}, L = strict-lower from step 3
        //   for i in 1..C-1: attn2[i, :i] += (attn2[i, :i] * attn2[:i, :i]).sum(-1)
        //   attn2 += eye(C)
        solve_tril_kernel<scalar_t, CHUNK_SIZE>::apply(attn2, mb_size);

        // step 5: recompute_w_u
        //   w = attn2 @ (k_beta * g.exp().unsqueeze(-1))
        //   u = attn2 @ value * beta.unsqueeze(-1)
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * CHUNK_SIZE;
        const scalar_t* __restrict__ v_ptr = v + (batch_offset + mb_start) * v_strideT + hv * v_strideH;

        //  5.a key = key * beta * g.exp
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, true, true>::apply(
            k_beta, k_ptr, beta_ptr, g_ptr, mb_size, k_strideT, D, Hv);

        //  5.b pack key
        if constexpr (brgemm_supported()) {
          pack_vnni2<scalar_t>(
              /*    dst */ k_beta_packed,
              /*    src */ k_beta,
              /*     K  */ mb_size,
              /*     N  */ D,
              /* ld_src */ D,
              /* ld_dst */ D);

          // 5.c w = attn2 @ k_beta
          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ D,
              /*     K */ padded_mb_size,  // mb_size
              /*   lda */ CHUNK_SIZE,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ false,
              /*     A */ attn2,
              /*     B */ k_beta_packed,
              /*     C */ k_updated);
        } else {
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              mb_size,
              padded_mb_size,
              1.0f,
              k_beta,
              D,
              attn2,
              CHUNK_SIZE,
              0.0f,
              k_updated,
              D);
        }

        // 5.d k_updated -> w
        scalar_t* __restrict__ w_ptr = w + (batch_offset + mb_start) * w_strideT + hv * w_strideH;
        update_kernel<scalar_t, D>::apply(w_ptr, k_updated, mb_size, D, w_strideT);

        // 5.e value = value * beta
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, true, false>::apply(
            v_beta, v_ptr, beta_ptr, nullptr, mb_size, v_strideT, D, Hv);

        // 5.f pack value
        if constexpr (brgemm_supported()) {
          pack_vnni2<scalar_t>(
              /*    dst */ v_beta_packed,
              /*    src */ v_beta,
              /*     K  */ mb_size,
              /*     N  */ D,
              /* ld_src */ D,
              /* ld_dst */ D);

          // 5.g u = attn2 @ v_beta
          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ D,
              /*     K */ padded_mb_size,  // mb_size
              /*   lda */ CHUNK_SIZE,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ false,
              /*     A */ attn2,
              /*     B */ v_beta_packed,
              /*     C */ v_updated);
        } else {
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              mb_size,
              padded_mb_size,
              1.0f,
              v_beta,
              D,
              attn2,
              CHUNK_SIZE,
              0.0f,
              v_updated,
              D);
        }

        // 5.h v_updated -> u
        scalar_t* __restrict__ u_ptr = u + (batch_offset + mb_start) * u_strideT + hv * u_strideH;
        update_kernel<scalar_t, D>::apply(u_ptr, v_updated, mb_size, D, u_strideT);
      }

      // move to the next index
      data_index_step(nt, NT, h, H);
    }
    at::native::cpublas::brgemm_release();
  });
}

//
// out           : [B, T, Hv, Dv]
// state         : [num_seqs, Hv, Dv, D]
// q             : [B, T, H, D]
// k             : [B, T, H, D]
// w             : [B, T, Hv, D]
// u             : [B, T, Hv, Dv]
// g             : [B, NT, Hv, C]
// d             : [B, NT, Hv, C, C]
// cu_seqlens    : [num_seqs + 1]
// chunk_offsets : [num_seqs + 1]
template <typename scalar_t, int D, int CHUNK_SIZE>
void chunk_gated_delta_rule_fwd_inter_kernel_impl(
    scalar_t* __restrict__ out,
    float* __restrict__ state,
    const int32_t* __restrict__ indices,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ u,
    const float* __restrict__ g,
    const float* __restrict__ d,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    int64_t H,
    int64_t Hv,
    int64_t num_seqs,
    int64_t q_strideT,
    int64_t q_strideH,
    int64_t k_strideT,
    int64_t k_strideH,
    int64_t state_strideS) {
  // head group, expect to be 1，2，4 for qwen3.5
  const int64_t HG = Hv / H;

  // strides
  const int64_t w_strideT = Hv * D;
  const int64_t w_strideH = D;
  const int64_t u_strideT = Hv * D;
  const int64_t u_strideH = D;
  const int64_t o_strideT = Hv * D;
  const int64_t o_strideH = D;

  // [NB]: parallel on [num_seqs, Hv]
  //  * choose to parallel on Hv instead of H, though this means q @ kT has duplicated compute
  //  * H might be 16 which is not enough to use 32C when num_seqs is small
  at::parallel_for(0, num_seqs * Hv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, hv{0};
    data_index_init(begin, bs, num_seqs, hv, Hv);

    // thread local temp buffer
    DECL_ZERO_BUF(scalar_t, tmp, CHUNK_SIZE * D);
    DECL_ZERO_BUF(scalar_t, tmp2, D * D);
    DECL_ZERO_BUF(float, tmp3, CHUNK_SIZE* D);
    DECL_ZERO_BUF(scalar_t, tmp4, CHUNK_SIZE * D);
    DECL_ZERO_BUF(float, attn, CHUNK_SIZE* CHUNK_SIZE);
    DECL_ZERO_BUF(scalar_t, attn2, CHUNK_SIZE * CHUNK_SIZE);

    // alias
    scalar_t* __restrict__ k_packed = tmp;
    scalar_t* __restrict__ s_packed = tmp2;
    float* __restrict__ v_prime = tmp3;
    scalar_t* __restrict__ v_prime2 = tmp;
    float* __restrict__ attn_inter = tmp3;
    scalar_t* __restrict__ qg_exp = tmp4;
    scalar_t* __restrict__ v_packed = tmp4;
    scalar_t* __restrict__ k_updated = tmp;

    for (int64_t i = begin; i < end; ++i) {
      int64_t h = hv / HG;
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t nt = chunk_offsets[bs];

      for (int64_t mb_start = 0; mb_start < seqlen; mb_start += CHUNK_SIZE, ++nt) {
        int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));

        // mb_size` is K in 4.a, pad to TILE_K;
        const int64_t padded_mb_size = div_up((int)mb_size, TILE_K) * TILE_K;

        // step 1.a: attn = query @ key^T
        // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        const scalar_t* __restrict__ q_ptr = q + (batch_offset + mb_start) * q_strideT + h * q_strideH;
        const scalar_t* __restrict__ k_ptr = k + (batch_offset + mb_start) * k_strideT + h * k_strideH;
        if constexpr (brgemm_supported()) {
          pack_vnni<scalar_t>(
              /*    dst */ k_packed,
              /*    src */ k_ptr,
              /*     N  */ mb_size,
              /*     K  */ D,
              /* ld_src */ k_strideT,
              /* ld_dst */ CHUNK_SIZE);

          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ mb_size,
              /*     K */ D,
              /*   lda */ q_strideT,
              /*   ldb */ CHUNK_SIZE,
              /*   ldc */ CHUNK_SIZE,
              /* add_C */ false,
              /*     A */ q_ptr,
              /*     B */ k_packed,
              /*     C */ attn);
        } else {
          blas_gemm(
              at::native::TransposeType::Transpose,
              at::native::TransposeType::NoTranspose,
              mb_size,
              mb_size,
              D,
              1.0f,
              k_ptr,
              k_strideT,
              q_ptr,
              q_strideT,
              0.0f,
              attn,
              CHUNK_SIZE);
        }

        // step 1.b: attn = attn * decay_mask.masked_fill_(mask, 0)
        const float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        apply_mask_kernel<scalar_t, CHUNK_SIZE, false>::apply(attn2, attn, nullptr, d_ptr, mb_size);

        // step 2.a: v' = w @ state (fuse state *= exp(g_last) with packing)
        float* __restrict__ s_ptr = state + indices[bs] * state_strideS + hv * (D * D);
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * (CHUNK_SIZE);
        float g_last = g_ptr[mb_size - 1];
        const scalar_t* __restrict__ w_ptr = w + (batch_offset + mb_start) * w_strideT + hv * w_strideH;
        if constexpr (brgemm_supported()) {
          pack_vnni2<scalar_t, D, D>(
              /*    dst */ s_packed,
              /*    src */ s_ptr,
              /* g_last */ g_last,
              /* ld_src */ D,
              /* ld_dst */ D);

          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ D,
              /*     K */ D,
              /*   lda */ w_strideT,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ false,
              /*     A */ w_ptr,
              /*     B */ s_packed,
              /*     C */ v_prime);
        } else {
          // brgemm_supported()==false path: pack_vnni2 above packs the
          // *unscaled* state into its dst (for this GEMM's B operand) while
          // separately scaling src in place by exp(g_last) (consumed later,
          // at step 5.3's state accumulation). Replicate both halves in the
          // same order: snapshot s_ptr into s_packed BEFORE scaling s_ptr,
          // not after, since the GEMM below needs the pre-scale state.
          float g_last_scale = std::exp(g_last);
          for (int64_t d0 = 0; d0 < D * D; ++d0) {
            s_packed[d0] = static_cast<scalar_t>(s_ptr[d0]);
            s_ptr[d0] *= g_last_scale;
          }
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              mb_size,
              D,
              1.0f,
              s_packed,
              D,
              w_ptr,
              w_strideT,
              0.0f,
              v_prime,
              D);
        }

        // step 2.b: v2' = u - v'
        const scalar_t* __restrict__ u_ptr = u + (batch_offset + mb_start) * u_strideT + hv * u_strideH;
        update_value_kernel<scalar_t, D>::apply(v_prime2, u_ptr, v_prime, mb_size, padded_mb_size, u_strideT);

        // step 3.a: qg_exp = q * exp(g)
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, false, true>::apply(
            qg_exp, q_ptr, nullptr, g_ptr, mb_size, q_strideT, D, /*b_stride*/ 0);

        // step 3.b: attn_inter = qg_exp @ state
        if constexpr (brgemm_supported()) {
          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ D,
              /*     K */ D,
              /*   lda */ D,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ false,
              /*     A */ qg_exp,
              /*     B */ s_packed,
              /*     C */ attn_inter);
        } else {
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              mb_size,
              D,
              1.0f,
              s_packed,
              D,
              qg_exp,
              D,
              0.0f,
              attn_inter,
              D);
        }

        // step 4.a: attn_inter += attn2 @ v2'
        if constexpr (brgemm_supported()) {
          pack_vnni2<scalar_t>(
              /*    dst */ v_packed,
              /*    src */ v_prime2,
              /*     K  */ padded_mb_size,
              /*     N  */ D,
              /* ld_src */ D,
              /* ld_dst */ D);

          at::native::cpublas::brgemm(
              /*     M */ mb_size,
              /*     N */ D,
              /*     K */ padded_mb_size,
              /*   lda */ CHUNK_SIZE,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ true,
              /*     A */ attn2,
              /*     B */ v_packed,
              /*     C */ attn_inter);
        } else {
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              mb_size,
              padded_mb_size,
              1.0f,
              v_prime2,
              D,
              attn2,
              CHUNK_SIZE,
              1.0f,
              attn_inter,
              D);
        }

        // step 4.b: write attn_inter -> out
        scalar_t* __restrict__ o_ptr = out + (batch_offset + mb_start) * o_strideT + hv * o_strideH;
        update_kernel<scalar_t, D>::apply(o_ptr, attn_inter, mb_size, D, o_strideT);

        // brgemm_supported()==false path: step 5.2 below overwrites k_updated,
        // which aliases the same buffer as v_prime2 (both are `tmp`). Snapshot
        // v_prime2 into v_packed (otherwise brgemm-only, and free by now since
        // qg_exp was consumed at step 3.b) before that happens, so step 5.3
        // doesn't read k_updated's data under the v_prime2 name.
        if constexpr (!brgemm_supported()) {
          std::copy(v_prime2, v_prime2 + padded_mb_size * D, v_packed);
        }

        // step 5: update state
        //   state_new = state * exp(g_last) + (k * exp(g_last - g)).T @ v2'

        // step 5.1 state *= exp(g_last) fused with step 2.a

        // step 5.2 k' = k * exp(g_last - g).T; TODO: fuse this with 1.a
        update_key_kernel<scalar_t, CHUNK_SIZE, D>::apply(k_updated, k_ptr, g_ptr, mb_size, k_strideT);

        // step 5.3 state += k' @ v2'
        if constexpr (brgemm_supported()) {
          at::native::cpublas::brgemm(
              /*     M */ D,
              /*     N */ D,
              /*     K */ padded_mb_size,  // mb_size
              /*   lda */ CHUNK_SIZE,
              /*   ldb */ D,
              /*   ldc */ D,
              /* add_C */ true,
              /*     A */ k_updated,
              /*     B */ v_packed,
              /*     C */ s_ptr);
        } else {
          blas_gemm(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose,
              D,
              D,
              padded_mb_size,
              1.0f,
              v_packed,
              D,
              k_updated,
              CHUNK_SIZE,
              1.0f,
              s_ptr,
              D);
        }
      }

      // move to the next index
      data_index_step(bs, num_seqs, hv, Hv);
    }
    at::native::cpublas::brgemm_release();
  });
}

inline float softplus(float x, double threshold = 20.0) {
  if (x > threshold)
    return x;
  else if (x < -threshold)
    return std::exp(x);
  else
    return std::log1p(std::exp(x));
}

inline at::vec::Vectorized<float> softplus(const at::vec::Vectorized<float>& x, double threshold = 20.0) {
  using Vec = at::vec::Vectorized<float>;
  Vec mask_hi = x > Vec(threshold);
  Vec mask_lo = x < Vec(-threshold);

  Vec expx = x.exp_u20();
  Vec log1pex = (expx + Vec(1.0f)).log();

  return Vec::blendv(Vec::blendv(log1pex, expx, mask_lo), x, mask_hi);
}

template <typename scalar_t, typename param_t>
void fused_sigmoid_gating_delta_rule_update_kernel_impl(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const param_t* __restrict__ A_log_ptr,
    const scalar_t* __restrict__ a_ptr,
    const scalar_t* __restrict__ dt_bias_ptr,
    const scalar_t* __restrict__ b_ptr,
    const int32_t* __restrict__ indices_ptr,
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,
    float* __restrict__ qk_scale_buf,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t k_strideS,
    int64_t k_strideH,
    int64_t v_strideB,
    int64_t v_strideS,
    int64_t v_strideH,
    int64_t state_slot_stride,
    bool use_qk_l2norm_in_kernel,
    double softplus_threshold) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt(head_dim);
  fVec scale_vec = fVec(scale);
  if (use_qk_l2norm_in_kernel) {
    float eps = 1e-5;
    at::parallel_for(0, batch_size * seq_len * num_heads, 0, [&](int64_t begin, int64_t end) {
      int64_t bi{0}, si{0}, ni{0};
      data_index_init(begin, bi, batch_size, si, seq_len, ni, num_heads);
      for (int64_t i = begin; i < end; ++i) {
        float sum_q = float(0);
        float sum_k = float(0);
        fVec sum_q_fvec = fVec(float(0));
        fVec sum_k_fvec = fVec(float(0));
        int64_t q_offset = bi * q_strideB + si * q_strideS + ni * q_strideH;
        int64_t k_offset = bi * k_strideB + si * k_strideS + ni * k_strideH;
        int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + ni;
        int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
        int64_t d;
#pragma GCC unroll 4
        for (d = 0; d <= head_dim - VecSize; d += VecSize) {
          auto [q_fvec0, q_fvec1] = load_float_vec2(q_ptr + q_offset + d);
          sum_q_fvec += q_fvec0 * q_fvec0;
          sum_q_fvec += q_fvec1 * q_fvec1;
          auto [k_fvec0, k_fvec1] = load_float_vec2(k_ptr + k_offset + d);
          sum_k_fvec += k_fvec0 * k_fvec0;
          sum_k_fvec += k_fvec1 * k_fvec1;
        }
#pragma GCC unroll 4
        for (; d < head_dim; ++d) {
          float q_val = static_cast<float>(q_ptr[q_offset + d]);
          sum_q += q_val * q_val;
          float k_val = static_cast<float>(k_ptr[k_offset + d]);
          sum_k += k_val * k_val;
        }

        sum_q += vec_reduce_sum(sum_q_fvec);
        sum_k += vec_reduce_sum(sum_k_fvec);
        qk_scale_buf[q_scale_offset] = float(1) / std::sqrt(sum_q + eps);
        qk_scale_buf[k_scale_offset] = float(1) / std::sqrt(sum_k + eps);

        data_index_step(bi, batch_size, si, seq_len, ni, num_heads);
      }
    });
  }
  at::parallel_for(0, batch_size * seq_len * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, si{0}, ni{0};
    data_index_init(begin, bi, batch_size, si, seq_len, ni, v_num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t cache_index = indices_ptr[bi];
      int64_t state_offset = cache_index * state_slot_stride + ni * head_dim * v_head_dim;
      float g_val = -std::exp(float(A_log_ptr[ni])) *
                    softplus(float(a_ptr[bi * v_num_heads + ni]) + float(dt_bias_ptr[ni]), softplus_threshold);
      float g_val_exp = std::exp(g_val);
      fVec g_val_exp_vec = fVec(g_val_exp);
      int64_t q_offset = si * q_strideS + bi * q_strideB + (ni / group_size) * q_strideH;
      int64_t k_offset = si * k_strideS + bi * k_strideB + (ni / group_size) * k_strideH;
      int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + (ni / group_size);
      int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
      float q_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[q_scale_offset] : 1.0f;
      float k_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[k_scale_offset] : 1.0f;
      int64_t v_offset = si * v_strideS + bi * v_strideB + ni * v_strideH;
      int64_t o_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
      // See: https://github.com/sgl-project/sglang/pull/26634
      float beta_val = 1 / (1 + std::exp(-b_ptr[bi * v_num_heads + ni]));
      fVec beta_vec = fVec(beta_val);
      int64_t dvi = 0;
      for (; dvi <= v_head_dim - VecSize; dvi += VecSize) {
        fVec kv_mem_vec0 = fVec(float(0));
        fVec kv_mem_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec k_val_vec = fVec(k_ptr[k_offset + di] * k_scale);
          auto [state_vec0, state_vec1] = load_float_vec2(state_ptr + state_offset + di * v_head_dim + dvi);
          kv_mem_vec0 = kv_mem_vec0 + state_vec0 * g_val_exp_vec * k_val_vec;
          kv_mem_vec1 = kv_mem_vec1 + state_vec1 * g_val_exp_vec * k_val_vec;
        }
        auto [v_vec0, v_vec1] = load_float_vec2(v_ptr + v_offset + dvi);
        fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
        fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
        fVec o_vec0 = fVec(float(0));
        fVec o_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec q_vec = fVec(q_ptr[q_offset + di] * q_scale);
          fVec k_vec = fVec(k_ptr[k_offset + di] * k_scale);
          auto [state_vec0, state_vec1] = load_float_vec2(state_ptr + state_offset + di * v_head_dim + dvi);
          state_vec0 = state_vec0 * g_val_exp_vec + k_vec * dt_vec0;
          state_vec1 = state_vec1 * g_val_exp_vec + k_vec * dt_vec1;
          o_vec0 = o_vec0 + state_vec0 * q_vec * scale_vec;
          o_vec1 = o_vec1 + state_vec1 * q_vec * scale_vec;
          state_vec0.store(state_ptr + state_offset + di * v_head_dim + dvi);
          state_vec1.store(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
        }
        bVec o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
        o_vec.store(o_ptr + o_offset + dvi);
      }
      for (; dvi < v_head_dim; ++dvi) {
        float kv_mem_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] *= g_val_exp;
          kv_mem_val += state_ptr[state_offset + di * v_head_dim + dvi] * k_val;
        }
        float v_val = v_ptr[v_offset + dvi];
        float dt_val = (v_val - kv_mem_val) * beta_val;
        float o_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float q_val = q_ptr[q_offset + di] * q_scale;
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] += k_val * dt_val;
          o_val += state_ptr[state_offset + di * v_head_dim + dvi] * q_val * scale;
        }
        o_ptr[o_offset + dvi] = o_val;
      }
      data_index_step(bi, batch_size, si, seq_len, ni, v_num_heads);
    }
  });
}

// Speculative-decode variant: processes a varlen batch where each sequence has
// ``q_len`` draft tokens, runs the recurrence sequentially over those tokens
// (inside the kernel, so one dispatch handles the whole draft block), reads the
// initial state from cache slot ``num_accepted-1`` and stores the state *after*
// token ``t`` into cache slot ``t`` (multi-slot rollback, matching the GPU
// kernel). Parallelized over (sequence, v_head); the per-sequence token loop is
// sequential as required by the recurrence.
template <typename scalar_t, typename param_t>
void fused_sigmoid_gating_delta_rule_update_spec_kernel_impl(
    const scalar_t* __restrict__ q_ptr,  // [T, HK, EK]
    const scalar_t* __restrict__ k_ptr,  // [T, HK, EK]
    const scalar_t* __restrict__ v_ptr,  // [T, HV, EV]
    const param_t* __restrict__ A_log_ptr,
    const scalar_t* __restrict__ a_ptr,  // [T, HV]
    const scalar_t* __restrict__ dt_bias_ptr,
    const scalar_t* __restrict__ b_ptr,  // [T, HV]
    const int32_t* __restrict__ spec_indices_ptr,  // [N, S]
    const int32_t* __restrict__ num_accepted_ptr,  // [N]
    const int32_t* __restrict__ cu_seqlens_ptr,     // [N + 1]
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,  // [T, HV, EV]
    float* __restrict__ qk_scale_buf,  // [2, T, HK]
    int64_t total_tokens,
    int64_t batch_size,
    int64_t spec_stride,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideT,
    int64_t q_strideH,
    int64_t k_strideT,
    int64_t k_strideH,
    int64_t v_strideT,
    int64_t v_strideH,
    int64_t state_slot_stride,
    bool use_qk_l2norm_in_kernel,
    double softplus_threshold) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt((double)head_dim);
  fVec scale_vec = fVec((float)scale);

  if (use_qk_l2norm_in_kernel) {
    float eps = 1e-5f;
    at::parallel_for(0, total_tokens * num_heads, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        int64_t ti = i / num_heads;
        int64_t ni = i % num_heads;
        const scalar_t* qp = q_ptr + ti * q_strideT + ni * q_strideH;
        const scalar_t* kp = k_ptr + ti * k_strideT + ni * k_strideH;
        float sq = 0.f, sk = 0.f;
        for (int64_t d = 0; d < head_dim; ++d) {
          float qv = (float)qp[d];
          sq += qv * qv;
          float kv = (float)kp[d];
          sk += kv * kv;
        }
        qk_scale_buf[ti * num_heads + ni] = 1.f / std::sqrt(sq + eps);
        qk_scale_buf[total_tokens * num_heads + ti * num_heads + ni] = 1.f / std::sqrt(sk + eps);
      }
    });
  }

  at::parallel_for(0, batch_size * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t bi = idx / v_num_heads;
      int64_t ni = idx % v_num_heads;
      int64_t kh = ni / group_size;
      int64_t q_start = cu_seqlens_ptr[bi];
      int64_t q_len = cu_seqlens_ptr[bi + 1] - q_start;
      if (q_len <= 0) {
        continue;
      }
      int64_t acc = (int64_t)num_accepted_ptr[bi];
      // Clamp acc-1 to >=0: when num_accepted is 0 the unclamped index reads
      // out of bounds and yields an arbitrary prev_slot used to index the SSM
      // state. Mirrors the GPU guard tl.maximum(num_accepted - 1, 0).
      int64_t prev_slot =
          (int64_t)spec_indices_ptr[bi * spec_stride + (acc > 0 ? acc - 1 : 0)];
      for (int64_t t = 0; t < q_len; ++t) {
        int64_t cur_slot = (int64_t)spec_indices_ptr[bi * spec_stride + t];
        int64_t token = q_start + t;
        const float* src = state_ptr + prev_slot * state_slot_stride + ni * head_dim * v_head_dim;
        float* dst = state_ptr + cur_slot * state_slot_stride + ni * head_dim * v_head_dim;
        float g_val = -std::exp((float)A_log_ptr[ni]) *
            softplus((float)a_ptr[token * v_num_heads + ni] + (float)dt_bias_ptr[ni], softplus_threshold);
        float g_val_exp = std::exp(g_val);
        fVec g_val_exp_vec = fVec(g_val_exp);
        float beta_val = 1.f / (1.f + std::exp(-(float)b_ptr[token * v_num_heads + ni]));
        fVec beta_vec = fVec(beta_val);
        int64_t q_offset = token * q_strideT + kh * q_strideH;
        int64_t k_offset = token * k_strideT + kh * k_strideH;
        float q_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[token * num_heads + kh] : 1.f;
        float k_scale =
            use_qk_l2norm_in_kernel ? qk_scale_buf[total_tokens * num_heads + token * num_heads + kh] : 1.f;
        int64_t v_offset = token * v_strideT + ni * v_strideH;
        int64_t o_offset = (token * v_num_heads + ni) * v_head_dim;
        int64_t dvi = 0;
        for (; dvi <= v_head_dim - VecSize; dvi += VecSize) {
          fVec kv_mem_vec0 = fVec(0.f);
          fVec kv_mem_vec1 = fVec(0.f);
          for (int di = 0; di < head_dim; ++di) {
            fVec k_val_vec = fVec((float)k_ptr[k_offset + di] * k_scale);
            fVec sv0 = fVec::loadu(src + di * v_head_dim + dvi);
            fVec sv1 = fVec::loadu(src + di * v_head_dim + dvi + fVecSize);
            kv_mem_vec0 = kv_mem_vec0 + sv0 * g_val_exp_vec * k_val_vec;
            kv_mem_vec1 = kv_mem_vec1 + sv1 * g_val_exp_vec * k_val_vec;
          }
          bVec v_bvec = bVec::loadu(v_ptr + v_offset + dvi);
          fVec v_vec0, v_vec1;
          std::tie(v_vec0, v_vec1) = at::vec::convert_to_float(v_bvec);
          fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
          fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
          fVec o_vec0 = fVec(0.f);
          fVec o_vec1 = fVec(0.f);
          for (int di = 0; di < head_dim; ++di) {
            fVec q_vec = fVec((float)q_ptr[q_offset + di] * q_scale);
            fVec k_vec = fVec((float)k_ptr[k_offset + di] * k_scale);
            fVec sv0 = fVec::loadu(src + di * v_head_dim + dvi);
            fVec sv1 = fVec::loadu(src + di * v_head_dim + dvi + fVecSize);
            sv0 = sv0 * g_val_exp_vec + k_vec * dt_vec0;
            sv1 = sv1 * g_val_exp_vec + k_vec * dt_vec1;
            o_vec0 = o_vec0 + sv0 * q_vec * scale_vec;
            o_vec1 = o_vec1 + sv1 * q_vec * scale_vec;
            sv0.store(dst + di * v_head_dim + dvi);
            sv1.store(dst + di * v_head_dim + dvi + fVecSize);
          }
          bVec o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
          o_vec.store(o_ptr + o_offset + dvi);
        }
        for (; dvi < v_head_dim; ++dvi) {
          float kv_mem_val = 0.f;
          for (int di = 0; di < head_dim; ++di) {
            float k_val = (float)k_ptr[k_offset + di] * k_scale;
            kv_mem_val += src[di * v_head_dim + dvi] * g_val_exp * k_val;
          }
          float v_val = (float)v_ptr[v_offset + dvi];
          float dt_val = (v_val - kv_mem_val) * beta_val;
          float o_val = 0.f;
          for (int di = 0; di < head_dim; ++di) {
            float q_val = (float)q_ptr[q_offset + di] * q_scale;
            float k_val = (float)k_ptr[k_offset + di] * k_scale;
            float ns = src[di * v_head_dim + dvi] * g_val_exp + k_val * dt_val;
            dst[di * v_head_dim + dvi] = ns;
            o_val += ns * q_val * scale;
          }
          o_ptr[o_offset + dvi] = (scalar_t)o_val;
        }
        prev_slot = cur_slot;
      }
    }
  });
}

template <typename scalar_t>
void fused_gdn_gating_kernel_impl(
    float* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    float* __restrict__ out,
    scalar_t* __restrict__ beta,
    int64_t batch,
    int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  const fVec neg_one(-1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t j = 0;
      for (; j < num_heads - (num_heads % vec_size); j += vec_size) {
        auto [A_log_vec0, A_log_vec1] = load_float_vec2(A_log + j);
        auto [dt_bias_vec0, dt_bias_vec1] = load_float_vec2(dt_bias + j);
        auto [a0, a1] = load_float_vec2(a + i * num_heads + j);
        auto [b0, b1] = load_float_vec2(b + i * num_heads + j);

        fVec g0 = neg_one * A_log_vec0.exp_u20() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp_u20() * softplus(a1 + dt_bias_vec1);
        fVec beta0 = fast_sigmoid(b0);
        fVec beta1 = fast_sigmoid(b1);

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
        bVec beta_vec = at::vec::convert_from_float<scalar_t>(beta0, beta1);
        beta_vec.store(beta + i * num_heads + j);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(A_log[j]) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
        beta[i * num_heads + j] = 1 / (1 + std::exp(-b[i * num_heads + j]));
      }
    }
  });
}

template <typename scalar_t>
void fused_gdn_gating_kernel_impl(
    scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    float* __restrict__ out,
    scalar_t* __restrict__ beta,
    int64_t batch,
    int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  const fVec neg_one(-1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t j = 0;
      for (; j < num_heads - (num_heads % vec_size); j += vec_size) {
        auto [A_log_vec0, A_log_vec1] = load_float_vec2(A_log + j);
        auto [dt_bias_vec0, dt_bias_vec1] = load_float_vec2(dt_bias + j);
        auto [a0, a1] = load_float_vec2(a + i * num_heads + j);
        auto [b0, b1] = load_float_vec2(b + i * num_heads + j);

        fVec g0 = neg_one * A_log_vec0.exp_u20() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp_u20() * softplus(a1 + dt_bias_vec1);
        fVec beta0 = fast_sigmoid(b0);
        fVec beta1 = fast_sigmoid(b1);

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
        bVec beta_vec = at::vec::convert_from_float<scalar_t>(beta0, beta1);
        beta_vec.store(beta + i * num_heads + j);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(float(A_log[j])) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
        beta[i * num_heads + j] = 1 / (1 + std::exp(-b[i * num_heads + j]));
      }
    }
  });
}

}  // anonymous namespace

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor> prepare_chunk_indices(const at::Tensor& cu_seqlens) {
  int64_t num_seqs = cu_seqlens.size(0) - 1;
  at::Tensor chunk_offsets = at::empty({num_seqs + 1}, cu_seqlens.options());
  // get number of chunks and chunk offsets
  const int32_t* offsets_data = cu_seqlens.data_ptr<int32_t>();
  int32_t num_chunks = 0;
  chunk_offsets[0] = 0;
  for (int64_t row = 0; row < num_seqs; ++row) {
    num_chunks += div_up(offsets_data[row + 1] - offsets_data[row], CHUNK_SIZE);
    chunk_offsets[row + 1] = num_chunks;
  }
  // get chunk indices
  at::Tensor chunk_indices = at::empty({num_chunks, 2}, cu_seqlens.options());
  int32_t* indices_data = chunk_indices.data_ptr<int32_t>();

  int64_t idx = 0;
  for (int32_t row = 0; row < num_seqs; ++row) {
    int32_t num_chunks = div_up(offsets_data[row + 1] - offsets_data[row], CHUNK_SIZE);

    for (int32_t col = 0; col < num_chunks; ++col) {
      indices_data[idx * 2 + 0] = row;
      indices_data[idx * 2 + 1] = col;
      idx++;
    }
  }
  return std::make_tuple(chunk_indices, chunk_offsets);
}

#define DISPATCH_HEAD_DIM_CASE(launch_macro, hd) \
  case hd: {                                     \
    launch_macro(hd);                            \
    break;                                       \
  }

// [NB]: add new head_dim support here
#define DISPATCH_HEAD_DIM(dim, launch_macro)                 \
  switch (dim) {                                             \
    DISPATCH_HEAD_DIM_CASE(launch_macro, 64)                 \
    DISPATCH_HEAD_DIM_CASE(launch_macro, 128)                \
    default:                                                 \
      TORCH_CHECK(false, "Unexpected head dim size, ", dim); \
  }

#define LAUNCH_L2NORM_KERNEL(HD)        \
  l2norm_fwd_kernel_impl<scalar_t, HD>( \
      query_norm.data_ptr<scalar_t>(),  \
      key_norm.data_ptr<scalar_t>(),    \
      query.data_ptr<scalar_t>(),       \
      key.data_ptr<scalar_t>(),         \
      eps,                              \
      T,                                \
      H,                                \
      query.stride(1),                  \
      query.stride(2),                  \
      key.stride(1),                    \
      key.stride(2));

std::tuple<at::Tensor, at::Tensor> l2norm_fwd(const at::Tensor& query, const at::Tensor& key, double eps) {
  int64_t B = query.size(0);
  int64_t T = query.size(1);
  int64_t H = query.size(2);
  int64_t D = query.size(3);

  at::Tensor query_norm = at::empty_like(query);
  at::Tensor key_norm = at::empty_like(key);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(), "l2norm_fwd", [&] { DISPATCH_HEAD_DIM(D, LAUNCH_L2NORM_KERNEL); });

  return std::make_tuple(query_norm, key_norm);
}

// [NB]: instantiate decay_mask to avoid heavy recomputation in the kernel with exp
template <int CHUNK_SIZE>
at::Tensor chunk_local_cumsum(const at::Tensor& g, const at::Tensor& cu_seqlens, const at::Tensor& chunk_indices) {
  int64_t B = g.size(0);
  // int64_t T = g.size(1);
  int64_t Hv = g.size(2);
  int64_t NT = chunk_indices.size(0);

  at::Tensor g_ = at::empty({B, NT, Hv, CHUNK_SIZE}, g.options());
  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "chunk_local_cumsum", [&] {
    chunk_local_cumsum_kernel_impl<scalar_t, CHUNK_SIZE>(
        g_.data_ptr<scalar_t>(),
        g.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        chunk_indices.data_ptr<int32_t>(),
        Hv,
        NT);
  });
  return g_;
}

#define LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTRA_KERNEL(HD)                \
  chunk_gated_delta_rule_fwd_intra_kernel_impl<scalar_t, HD, CHUNK_SIZE>( \
      w.data_ptr<scalar_t>(),                                             \
      u.data_ptr<scalar_t>(),                                             \
      decay_mask.data_ptr<float>(),                                       \
      k.data_ptr<scalar_t>(),                                             \
      v.data_ptr<scalar_t>(),                                             \
      g.data_ptr<float>(),                                                \
      beta.data_ptr<scalar_t>(),                                          \
      cu_seqlens.data_ptr<int32_t>(),                                     \
      chunk_indices.data_ptr<int32_t>(),                                  \
      H,                                                                  \
      Hv,                                                                 \
      NT,                                                                 \
      k.stride(1),                                                        \
      k.stride(2),                                                        \
      v.stride(1),                                                        \
      v.stride(2));

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_intra(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t H = k.size(2);
  int64_t D = k.size(3);
  int64_t Hv = v.size(2);
  int64_t Dv = v.size(3);
  int64_t NT = chunk_indices.size(0);

  at::Tensor w = at::empty({B, T, Hv, D}, k.options());                                 // BFloat16
  at::Tensor u = at::empty({B, T, Hv, Dv}, k.options());                                // BFloat16
  at::Tensor decay_mask = at::empty({B, NT, Hv, CHUNK_SIZE, CHUNK_SIZE}, g.options());  // Float
  AT_DISPATCH_REDUCED_FLOATING_TYPES(k.scalar_type(), "chunk_gated_delta_rule_fwd_intra", [&] {
    DISPATCH_HEAD_DIM(D, LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTRA_KERNEL);
  });

  return std::make_tuple(w, u, decay_mask);
}

#define LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTER_KERNEL(HD)                \
  chunk_gated_delta_rule_fwd_inter_kernel_impl<scalar_t, HD, CHUNK_SIZE>( \
      o.data_ptr<scalar_t>(),                                             \
      initial_state.data_ptr<float>(),                                    \
      initial_state_indices.data_ptr<int32_t>(),                          \
      q.data_ptr<scalar_t>(),                                             \
      k.data_ptr<scalar_t>(),                                             \
      w.data_ptr<scalar_t>(),                                             \
      u.data_ptr<scalar_t>(),                                             \
      g.data_ptr<float>(),                                                \
      decay_mask.data_ptr<float>(),                                       \
      cu_seqlens.data_ptr<int32_t>(),                                     \
      chunk_offsets.data_ptr<int32_t>(),                                  \
      H,                                                                  \
      Hv,                                                                 \
      num_seqs,                                                           \
      q.stride(1),                                                        \
      q.stride(2),                                                        \
      k.stride(1),                                                        \
      k.stride(2),                                                        \
      initial_state.stride(0));

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_inter(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const at::Tensor& g,
    const at::Tensor& decay_mask,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_offsets,
    const at::Tensor& initial_state_indices) {
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t Hv = w.size(2);
  const int64_t Dv = u.size(3);
  const int64_t num_seqs = initial_state_indices.size(0);

  at::Tensor o = at::empty({B, T, Hv, Dv}, q.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "chunk_gated_delta_rule_fwd_inter", [&] {
    DISPATCH_HEAD_DIM(D, LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTER_KERNEL);
  });

  return std::make_tuple(o, initial_state);
}

// [NB]: Support only varlen inputs
//   B: packed batch dim of q/k/v (== 1)
//   num_seqs: number of variable-length sequences
//
//   query: [B, T, H, D]
//   key: [B, T, H, D]
//   value: [B, T, Hv, Dv]
//   g: [B, T, Hv] FP32
//   beta: [B, T, Hv]
//   initial_state: [num_seqs, Hv, Dv, D] FP32
//   cu_seqlens: [num_seqs + 1] INT32
//
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    bool head_first,
    bool use_qk_l2norm_in_kernel,
    const at::Tensor& initial_state_indices,
    double eps = 1e-6) {
  TORCH_CHECK(!head_first, "chunk_gated_delta_rule_cpu: does not support head first");

  int64_t B = query.size(0);
  int64_t T = query.size(1);
  int64_t H = query.size(2);
  int64_t D = query.size(3);
  int64_t Hv = value.size(2);
  int64_t Dv = value.size(3);
  int64_t num_seqs = initial_state_indices.size(0);

  TORCH_CHECK(B == 1, __func__, ": expect batch size to be 1");
  TORCH_CHECK(Hv % H == 0, __func__, ": expect num_heads_kv multiple of num_heads.");
  TORCH_CHECK(D % 32 == 0, __func__, ": expect head_dim to be multiples of 32.");
  TORCH_CHECK(Dv % 32 == 0, __func__, ": expect head_dim_v to be multiples of 32.");
  TORCH_CHECK(D == Dv, __func__, ": expect head_dim to be equal to head_dim_v.");
  CHECK_INPUT_SHAPE_DTYPE<true>(query, {B, T, H, D}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(key, {B, T, H, D}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(value, {B, T, Hv, Dv}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(g, {B, T, Hv}, at::kFloat);
  CHECK_INPUT_SHAPE_DTYPE<false>(beta, {B, T, Hv}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(cu_seqlens, {num_seqs + 1}, at::kInt);
  TORCH_CHECK(initial_state.sizes() == at::IntArrayRef({initial_state.size(0), Hv, Dv, D}),
              "chunk_gated_delta_rule_cpu: initial_state shape mismatch, got ", initial_state.sizes());
  TORCH_CHECK(initial_state.scalar_type() == at::kFloat, "chunk_gated_delta_rule_cpu: initial_state dtype mismatch");
  CHECK_CPU(initial_state);
  // initial_state may be a pooled/paged buffer with padding between slots
  // (e.g. mamba cache-align mode), so only the per-slot (Hv, Dv, D) layout
  // needs to be densely packed; dim 0's stride is read at runtime instead of
  // assumed, mirroring conv.cpp's conv_state_slot_stride handling.
  TORCH_CHECK(initial_state.stride(-1) == 1 && initial_state.stride(-2) == D &&
                  initial_state.stride(-3) == Dv * D,
              "chunk_gated_delta_rule_cpu: expect initial_state to be contiguous per pool slot.");
  CHECK_INPUT_SHAPE_DTYPE<false>(initial_state_indices, {num_seqs}, at::kInt);

  constexpr int CHUNK_SIZE = 64;

  // prepare chunk indices
  auto [chunk_indices, chunk_offsets] = prepare_chunk_indices<CHUNK_SIZE>(cu_seqlens);

  float scale = 1.0 / std::sqrt(D);
  auto [query_, key_] = use_qk_l2norm_in_kernel ? l2norm_fwd(query, key, eps) : std::make_tuple(query.mul(scale), key);

  auto g_ = chunk_local_cumsum<CHUNK_SIZE>(g, cu_seqlens, chunk_indices);

  // fused kkt + solve_tril + recompute_w_u
  auto [w, u, decay_mask] =
      chunk_gated_delta_rule_fwd_intra<CHUNK_SIZE>(key_, value, g_, beta, cu_seqlens, chunk_indices);

  // fused `chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o`
  auto [output, final_state] = chunk_gated_delta_rule_fwd_inter<CHUNK_SIZE>(
      query_,
      key_,
      w,
      u,
      g_,
      decay_mask,
      initial_state,
      output_final_state,
      cu_seqlens,
      chunk_offsets,
      initial_state_indices);

  return std::make_tuple(output, final_state);
}

// A_log: [v_num_heads]
// dt_bias: [v_num_heads]
// query: [seq_len, batch_size, num_heads, head_dim]
// key: [seq_len, batch_size, num_heads, head_dim]
// value: [seq_len, batch_size, v_num_heads, v_head_dim]
// a: [batch_size, v_num_heads]
// b: [batch_size, v_num_heads]
// initial_state_source:[num_tokens, v_num_heads, head_dim, v_head_dim]
// initial_state_indices: [batch_size]
// cu_seqlens: [batch_size + 1]
at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices,
    const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel,
    double softplus_beta = 1.0,
    double softplus_threshold = 20.0) {
  CHECK_DIM(4, q);
  CHECK_DIM(4, v);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  int64_t seq_len = q.size(0);
  int64_t batch_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t v_num_heads = v.size(2);
  int64_t v_head_dim = v.size(3);
  CHECK_INPUT_SHAPE_DTYPE<true>(k, {seq_len, batch_size, num_heads, head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(v, {seq_len, batch_size, v_num_heads, v_head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(a, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(dt_bias, {v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(b, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(initial_state_indices, {batch_size}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true>(cu_seqlens, {batch_size + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true>(
      initial_state_source, {initial_state_source.size(0), v_num_heads, head_dim, v_head_dim}, at::kFloat);
  CHECK(initial_state_source.size(0) >= batch_size);
  CHECK_EQ(v_num_heads % num_heads, 0);
  TORCH_CHECK(
      A_log.sizes() == at::IntArrayRef({v_num_heads}),
      "Input tensor shape mismatch: expected ",
      at::IntArrayRef({v_num_heads}),
      ", got ",
      A_log.sizes());

  int64_t q_strideB = q.stride(1);
  int64_t q_strideS = q.stride(0);
  int64_t q_strideH = q.stride(2);
  int64_t k_strideB = k.stride(1);
  int64_t k_strideS = k.stride(0);
  int64_t k_strideH = k.stride(2);
  int64_t v_strideB = v.stride(1);
  int64_t v_strideS = v.stride(0);
  int64_t v_strideH = v.stride(2);
  // IMPORTANT: To make the kernal compatible with vLLM KV cache layout 
  int64_t state_slot_stride = initial_state_source.stride(0);
  at::Tensor core_attn_out = at::empty({batch_size, seq_len, v_num_heads, v_head_dim}, q.options());
  at::Tensor qk_scale_buf = at::empty({2 * batch_size, seq_len, num_heads}, at::kFloat);

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(
      q.scalar_type(), A_log.scalar_type(), "fused_sigmoid_gating_delta_rule_update_kernel_impl", [&] {
        fused_sigmoid_gating_delta_rule_update_kernel_impl<scalar_t, param_t>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            A_log.data_ptr<param_t>(),
            a.data_ptr<scalar_t>(),
            dt_bias.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            initial_state_indices.data_ptr<int32_t>(),
            initial_state_source.data_ptr<float>(),
            core_attn_out.data_ptr<scalar_t>(),
            qk_scale_buf.data_ptr<float>(),
            seq_len,
            batch_size,
            num_heads,
            head_dim,
            v_num_heads,
            v_head_dim,
            q_strideB,
            q_strideS,
            q_strideH,
            k_strideB,
            k_strideS,
            k_strideH,
            v_strideB,
            v_strideS,
            v_strideH,
            state_slot_stride,
            use_qk_l2norm_in_kernel,
            softplus_threshold);
      });
  return core_attn_out;
}

// Speculative-decode update (multi-token, multi-slot rollback).
// q: [T, HK, EK]  k: [T, HK, EK]  v: [T, HV, EV]
// a: [T, HV]  b: [T, HV]
// initial_state_source: [N_slots, HV, EK, EV] FP32 (updated in place)
// spec_state_indices: [batch, S] INT32 (S = num_spec + 1)
// num_accepted_tokens: [batch] INT32
// cu_seqlens: [batch + 1] INT32
// Returns output: [T, HV, EV]
at::Tensor fused_sigmoid_gating_delta_rule_update_spec_cpu(
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& initial_state_source,
    const at::Tensor& spec_state_indices,
    const at::Tensor& num_accepted_tokens,
    const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel,
    double softplus_beta = 1.0,
    double softplus_threshold = 20.0) {
  CHECK_DIM(3, q);
  CHECK_DIM(3, v);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  int64_t total_tokens = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t v_num_heads = v.size(1);
  int64_t v_head_dim = v.size(2);
  int64_t batch_size = cu_seqlens.size(0) - 1;
  int64_t spec_stride = spec_state_indices.stride(0);
  CHECK_INPUT_SHAPE_DTYPE<true>(k, {total_tokens, num_heads, head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(v, {total_tokens, v_num_heads, v_head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(a, {total_tokens, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(b, {total_tokens, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(dt_bias, {v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(num_accepted_tokens, {batch_size}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true>(cu_seqlens, {batch_size + 1}, at::kInt);
  CHECK_EQ(v_num_heads % num_heads, 0);
  TORCH_CHECK(A_log.sizes() == at::IntArrayRef({v_num_heads}));
  CHECK_INPUT_SHAPE_DTYPE<true>(
      initial_state_source,
      {initial_state_source.size(0), v_num_heads, head_dim, v_head_dim},
      at::kFloat);
  TORCH_CHECK(initial_state_source.size(0) >= batch_size,
      "initial_state_source capacity too small: size(0)=",
      initial_state_source.size(0), ", batch_size=", batch_size);

  int64_t q_strideT = q.stride(0);
  int64_t q_strideH = q.stride(1);
  int64_t k_strideT = k.stride(0);
  int64_t k_strideH = k.stride(1);
  int64_t v_strideT = v.stride(0);
  int64_t v_strideH = v.stride(1);
  int64_t state_slot_stride = initial_state_source.stride(0);

  at::Tensor o = at::empty({total_tokens, v_num_heads, v_head_dim}, q.options());
  at::Tensor qk_scale_buf = at::empty({2, total_tokens, num_heads}, at::kFloat);

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(
      q.scalar_type(), A_log.scalar_type(), "fused_sigmoid_gating_delta_rule_update_spec_kernel_impl", [&] {
        fused_sigmoid_gating_delta_rule_update_spec_kernel_impl<scalar_t, param_t>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            A_log.data_ptr<param_t>(),
            a.data_ptr<scalar_t>(),
            dt_bias.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            spec_state_indices.data_ptr<int32_t>(),
            num_accepted_tokens.data_ptr<int32_t>(),
            cu_seqlens.data_ptr<int32_t>(),
            initial_state_source.data_ptr<float>(),
            o.data_ptr<scalar_t>(),
            qk_scale_buf.data_ptr<float>(),
            total_tokens,
            batch_size,
            spec_stride,
            num_heads,
            head_dim,
            v_num_heads,
            v_head_dim,
            q_strideT,
            q_strideH,
            k_strideT,
            k_strideH,
            v_strideT,
            v_strideH,
            state_slot_stride,
            use_qk_l2norm_in_kernel,
            softplus_threshold);
      });
  return o;
}

// A_log: [num_v_heads]
// a: [batch, num_v_heads]
// b: [batch, num_v_heads]
// dt_bias: [num_v_heads]
// -A_log.float().exp() * F.softplus(a.float() + dt_bias)
std::tuple<at::Tensor, at::Tensor>
fused_gdn_gating_cpu(const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b, const at::Tensor& dt_bias) {
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(1, dt_bias);
  CHECK_CONTIGUOUS(a);
  CHECK_EQ(A_log.size(0), a.size(1));
  CHECK_EQ(A_log.size(0), dt_bias.size(0));
  int batch = a.size(0);
  int num_heads = a.size(1);
  CHECK_EQ(b.size(0), batch);
  CHECK_EQ(b.size(1), num_heads);
  at::Tensor out = at::empty({1, batch, num_heads}, a.options().dtype(at::kFloat));
  at::Tensor beta = at::empty({1, batch, num_heads}, b.options());
  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(a.scalar_type(), A_log.scalar_type(), "fused_gdn_gating_kernel", [&] {
    fused_gdn_gating_kernel_impl<scalar_t>(
        A_log.data_ptr<param_t>(),
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        dt_bias.data_ptr<scalar_t>(),
        out.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        batch,
        num_heads);
  });
  return std::make_tuple(out, beta);
}
