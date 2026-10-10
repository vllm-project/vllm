// Adapted from
// https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/cpu/mhc.cpp
//
// DeepSeek-V4 mHC (multi-residual-stream hyper-connection) gating kernels.
//
// Difference from upstream: vLLM's `mhc_pre_torch` reference (kernels/
// mhc/torch.py) exposes separate `hc_pre_eps`/`hc_sinkhorn_eps`/
// `hc_post_mult_value`/`n_splits` knobs this kernel doesn't implement --
// only ever called with hc_pre_eps == hc_sinkhorn_eps, hc_post_mult_value
// == 2.0, n_splits == 1 (matches upstream), so the entry points take one
// merged `hc_eps` and hardcode the post-mix multiplier to 2.0. The Python
// wrapper (kernels/mhc/cpu.py) enforces the merged-eps assumption via
// TORCH_CHECK.
//
// clang-format off

#include <algorithm>
#include <cmath>
#include <limits>

#include "common.h"
#include "vec.h"

namespace {

// Loop tiling heuristics.
constexpr int64_t kMhcTileCacheBudgetBytes = 36 * 1024;
constexpr int64_t kMhcDefaultTokenBlock = 32;

inline int64_t round_to_nearest_multiple(int64_t value, int64_t multiple) {
  return ((value + multiple / 2) / multiple) * multiple;
}

inline int64_t choose_t_block(int64_t /*T*/) {
  return kMhcDefaultTokenBlock;
}

inline int64_t choose_k_block(int64_t weight_rows, int64_t vec_size, int64_t max_block) {
  const int64_t bytes_per_k = (weight_rows + 1) * static_cast<int64_t>(sizeof(float));
  const int64_t raw = std::max(vec_size, kMhcTileCacheBudgetBytes / bytes_per_k);
  const int64_t rounded = round_to_nearest_multiple(raw, vec_size);
  return std::max(vec_size, std::min(max_block, rounded));
}

template <typename func_t>
inline void parallel_mhc_phase1(int64_t n_t_blocks, int64_t n_k_blocks, const func_t& f) {
  const int64_t nth = static_cast<int64_t>(at::get_num_threads());
  if (n_t_blocks >= nth) {
    // Large prefill: enough token blocks exist to fill all workers.
    parallel_2d_tiled(static_cast<int>(n_t_blocks), static_cast<int>(n_k_blocks), static_cast<int>(nth), 1, f);
  } else {
    // Decode / small T: split both T and K to expose enough parallel tasks.
    parallel_2d(static_cast<int>(n_t_blocks), static_cast<int>(n_k_blocks), f);
  }
}

// In-place Sinkhorn on a row-major [hc, hc] matrix.
inline void
sinkhorn_inplace(float* __restrict__ comb_matrix, int hc, int sinkhorn_iters, float eps, float* __restrict__ scratch) {
  float* const row_sum = scratch;
  float* const col_sum = scratch + hc;

  // First iteration.
  for (int r = 0; r < hc; ++r) {
    const float* row = comb_matrix + r * hc;
    float row_acc = 0.f;
#pragma GCC unroll 4
    for (int c = 0; c < hc; ++c)
      row_acc += row[c];
    row_sum[r] = row_acc;
  }
  for (int r = 0; r < hc; ++r) {
    const float row_inv = 1.0f / row_sum[r];
    float* row = comb_matrix + r * hc;
#pragma GCC unroll 4
    for (int c = 0; c < hc; ++c)
      row[c] = row[c] * row_inv + eps;
  }
  for (int c = 0; c < hc; ++c) {
    float col_acc = 0.f;
#pragma GCC unroll 4
    for (int r = 0; r < hc; ++r)
      col_acc += comb_matrix[r * hc + c];
    col_sum[c] = col_acc;
  }
  for (int r = 0; r < hc; ++r)
#pragma GCC unroll 4
    for (int c = 0; c < hc; ++c)
      comb_matrix[r * hc + c] /= (col_sum[c] + eps);

  // Remaining iterations.
  for (int iter = 1; iter < sinkhorn_iters; ++iter) {
    for (int r = 0; r < hc; ++r) {
      const float* row = comb_matrix + r * hc;
      float row_acc = 0.f;
#pragma GCC unroll 4
      for (int c = 0; c < hc; ++c)
        row_acc += row[c];
      row_sum[r] = row_acc;
    }
    for (int r = 0; r < hc; ++r) {
      const float row_inv = 1.0f / (row_sum[r] + eps);
      float* row = comb_matrix + r * hc;
#pragma GCC unroll 4
      for (int c = 0; c < hc; ++c)
        row[c] *= row_inv;
    }
    for (int c = 0; c < hc; ++c) {
      float col_acc = 0.f;
#pragma GCC unroll 4
      for (int r = 0; r < hc; ++r)
        col_acc += comb_matrix[r * hc + c];
      col_sum[c] = col_acc;
    }
    for (int r = 0; r < hc; ++r)
#pragma GCC unroll 4
      for (int c = 0; c < hc; ++c)
        comb_matrix[r * hc + c] /= (col_sum[c] + eps);
  }
}

// Parse mixes -> pre/post/comb and run sinkhorn on comb.
inline void parse_mixes_and_sinkhorn(
    float* __restrict__ pre_gate,     // [hc] output
    float* __restrict__ post_gate,    // [hc] output
    float* __restrict__ comb_matrix,  // [hc*hc] output (row-major)
    const float* __restrict__ mixes,  // [mix_hc] input
    float s0,
    float s1,
    float s2,
    const float* __restrict__ hc_base,
    int sinkhorn_iters,
    float eps,
    int hc,
    float* __restrict__ scratch) {
#pragma GCC unroll 4
  for (int h = 0; h < hc; ++h) {
    const float v = mixes[h] * s0 + hc_base[h];
    pre_gate[h] = 1.f / (1.f + std::exp(-v)) + eps;
  }
#pragma GCC unroll 4
  for (int h = 0; h < hc; ++h) {
    const float v = mixes[hc + h] * s1 + hc_base[hc + h];
    post_gate[h] = 2.f / (1.f + std::exp(-v));
  }
  float* const row_max = scratch;
  for (int r = 0; r < hc; ++r)
    row_max[r] = -std::numeric_limits<float>::infinity();
  for (int r = 0; r < hc; ++r) {
    float* row = comb_matrix + r * hc;
    for (int c = 0; c < hc; ++c) {
      const int idx = r * hc + c;
      const float v = mixes[2 * hc + idx] * s2 + hc_base[2 * hc + idx];
      row[c] = v;
      if (v > row_max[r]) row_max[r] = v;
    }
  }
  for (int r = 0; r < hc; ++r) {
    float* row = comb_matrix + r * hc;
#pragma GCC unroll 4
    for (int c = 0; c < hc; ++c)
      row[c] = std::exp(row[c] - row_max[r]);
  }

  sinkhorn_inplace(comb_matrix, hc, sinkhorn_iters, eps, scratch + hc);
}

template <typename scalar_t>
static void hc_post_fuse_impl(
    scalar_t* __restrict__ out,             // [T, hc, d]  output
    const scalar_t* __restrict__ x,         // [T, d]      input
    const scalar_t* __restrict__ residual,  // [T, hc, d]  input
    const float* __restrict__ post,         // [T, hc]     float32
    const float* __restrict__ comb,         // [T, hc, hc] float32 row-major
    int64_t T,
    int64_t d,
    int hc) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();

  const int64_t nthreads = at::get_num_threads();
  const int64_t max_splits = std::max(int64_t(1), d / kVecSize);
  const int64_t token_head_tasks = T * hc;
  const int64_t k_splits = (token_head_tasks >= nthreads) ? int64_t(1) : std::min(nthreads, max_splits);
  const int64_t k_chunk = (d + k_splits - 1) / k_splits;

  const int64_t fvec_floats = static_cast<int64_t>(sizeof(fVec) / sizeof(float));
  auto comb_fvec_storage = at::empty({nthreads * hc * fvec_floats}, at::kFloat);
  fVec* const comb_fvec_base = reinterpret_cast<fVec*>(comb_fvec_storage.data_ptr<float>());

  at::parallel_for(0, T * hc * k_splits, 0, [&](int64_t begin, int64_t end) {
    const int tid = static_cast<int>(at::get_thread_num());
    fVec* const comb_fvec = comb_fvec_base + tid * hc;

    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t task = idx / k_splits;
      const int64_t split_id = idx % k_splits;
      const int64_t t = task / hc;
      const int h = static_cast<int>(task % hc);
      const int64_t k0 = split_id * k_chunk;
      const int64_t k1 = std::min(k0 + k_chunk, d);

      const float post_val = post[t * hc + h];
      const float* comb_t = comb + t * hc * hc;
      const scalar_t* x_t = x + t * d;
      const scalar_t* res_t = residual + t * hc * d;
      scalar_t* out_th = out + (t * hc + h) * d;

      const fVec post_fvec(post_val);
#pragma GCC unroll 4
      for (int i = 0; i < hc; ++i)
        comb_fvec[i] = fVec(comb_t[i * hc + h]);

      int64_t k;
      // bf16: convert and accumulate
      for (k = k0; k <= k1 - kVecSize; k += kVecSize) {
        fVec acc0, acc1;
        std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
        acc0 = post_fvec * acc0;
        acc1 = post_fvec * acc1;
#pragma GCC unroll 4
        for (int i = 0; i < hc; ++i) {
          fVec r0, r1;
          std::tie(r0, r1) = at::vec::convert_to_float(bVec::loadu(res_t + i * d + k));
          acc0 += comb_fvec[i] * r0;
          acc1 += comb_fvec[i] * r1;
        }
        at::vec::convert_from_float<scalar_t>(acc0, acc1).store(out_th + k);
      }
      if (k < k1) {
        const int64_t rem = k1 - k;
        fVec acc0, acc1;
        std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
        acc0 = post_fvec * acc0;
        acc1 = post_fvec * acc1;
#pragma GCC unroll 4
        for (int i = 0; i < hc; ++i) {
          fVec r0, r1;
          std::tie(r0, r1) = at::vec::convert_to_float(bVec::loadu(res_t + i * d + k, rem));
          acc0 += comb_fvec[i] * r0;
          acc1 += comb_fvec[i] * r1;
        }
        at::vec::convert_from_float<scalar_t>(acc0, acc1).store(out_th + k, rem);
      }
    }
  });
}

// Fused hc_pre path.
template <typename scalar_t>
static void hc_pre_fuse_impl(
    scalar_t* __restrict__ y,            // [T, d]      bf16 output
    float* __restrict__ post_out,        // [T, hc]     float32 output
    float* __restrict__ comb_out,        // [T, hc*hc]  float32 output
    const scalar_t* __restrict__ x,      // [T, hc, d]  bf16 input
    const float* __restrict__ hc_fn,     // [mix_hc, hc_d] float32 row-major
    const float* __restrict__ hc_scale,  // [3]
    const float* __restrict__ hc_base,   // [mix_hc]
    int64_t T,
    int64_t d,
    int hc,
    int sinkhorn_iters,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();   // 32 bf16
  constexpr int64_t kFVecSize = fVec::size();  // 16 fp32
  constexpr int64_t kSmallTKBlockCap = 128;
  constexpr int64_t kLargeTKBlockCap = 256;
  constexpr int64_t kDVecsPerBlock = 16;
  constexpr int64_t kCacheKBlockCap = kLargeTKBlockCap;
  constexpr int64_t MAX_K_VECS = kCacheKBlockCap / kVecSize;
  const int mix_hc = (2 + hc) * hc;
  const int64_t hc_d = hc * d;
  const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

  const int64_t T_BLOCK = choose_t_block(T);
  const int64_t num_threads = static_cast<int64_t>(at::get_num_threads());
  const int64_t K_BLOCK_CAP = (T >= num_threads) ? kLargeTKBlockCap : kSmallTKBlockCap;
  const int64_t K_BLOCK = choose_k_block(mix_hc, kVecSize, K_BLOCK_CAP);
  TORCH_CHECK(
      K_BLOCK <= K_BLOCK_CAP && K_BLOCK_CAP <= kCacheKBlockCap && K_BLOCK % kVecSize == 0,
      "hc_pre_fuse_impl: K_BLOCK must be within capacity and a multiple of vector size");
  const int64_t D_BLOCK = kDVecsPerBlock * kVecSize;

  const int64_t n_t_blocks = div_up(T, T_BLOCK);
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);

  // Partial buffer: [n_t_blocks, n_k_blocks, T_BLOCK, 1+mix_hc]
  const int64_t partial_stride = 1 + mix_hc;
  auto partials_tensor = at::empty({n_t_blocks * n_k_blocks * T_BLOCK * partial_stride}, at::kFloat);
  float* const partials = partials_tensor.data_ptr<float>();

  // Per-thread scratch: [mix_buffer[mix_hc] | pre_gate[hc] |
  //                      post_gate[hc] | comb_matrix[hc*hc] | sinkhorn_scratch[3*hc]]
  const int64_t per_thread_scratch = mix_hc + hc + hc + hc * hc + 3 * hc;
  auto scratch_tensor = at::empty({num_threads * per_thread_scratch}, at::kFloat);
  float* const scratch_base = scratch_tensor.data_ptr<float>();

  // Phase 1.
  parallel_mhc_phase1(n_t_blocks, n_k_blocks, [&](int64_t tb0, int64_t tb1, int64_t kb0, int64_t kb1) {
    for (int64_t kb = kb0; kb < kb1; ++kb) {
      const int64_t k0 = kb * K_BLOCK;
      const int64_t klen = std::min(K_BLOCK, hc_d - k0);

      for (int64_t tb = tb0; tb < tb1; ++tb) {
        const int64_t t0 = tb * T_BLOCK;
        const int64_t tlen = std::min(T_BLOCK, T - t0);
        float* partial_block = partials + (tb * n_k_blocks + kb) * T_BLOCK * partial_stride;

        const float* w_ptr[mix_hc];
        for (int h = 0; h < mix_hc; ++h)
          w_ptr[h] = hc_fn + h * hc_d + k0;

        // Stack cache is sized for the largest selected K block, but only the first
        // n_full entries are touched for the current K_BLOCK.
        fVec x_cache_lo[MAX_K_VECS], x_cache_hi[MAX_K_VECS];

        for (int64_t tl = 0; tl < tlen; ++tl) {
          const scalar_t* x_t = x + (t0 + tl) * hc_d + k0;
          float* partial_row = partial_block + tl * partial_stride;

          const int64_t n_full = klen / kVecSize;
          const int64_t k_tail = n_full * kVecSize;
          const int64_t rem = klen - k_tail;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;

          // Prefetch next token.
          if (tl + 1 < tlen) {
            const scalar_t* xn = x + (t0 + tl + 1) * hc_d + k0;
            __builtin_prefetch(xn, 0, 0);
            if (klen > kVecSize * 4) __builtin_prefetch(xn + kVecSize * 4, 0, 0);
          }

          // h=0..3: convert/cache x and accumulate sq + four dots.
          if (tl == 0) {
            for (int pw = 1; pw < mix_hc; ++pw)
              __builtin_prefetch(w_ptr[pw], 0, 3);
          }
          {
            const float* w0 = w_ptr[0];
            const float* w1 = w_ptr[1];
            const float* w2 = w_ptr[2];
            const float* w3 = w_ptr[3];
            fVec sq_lo(0.f), sq_hi(0.f);
            fVec d0_lo(0.f), d0_hi(0.f);
            fVec d1_lo(0.f), d1_hi(0.f);
            fVec d2_lo(0.f), d2_hi(0.f);
            fVec d3_lo(0.f), d3_hi(0.f);
            for (int64_t ki = 0; ki < n_full; ++ki) {
              const int64_t k = ki * kVecSize;
              std::tie(x_cache_lo[ki], x_cache_hi[ki]) = at::vec::convert_to_float(bVec::loadu(x_t + k));
              sq_lo += x_cache_lo[ki] * x_cache_lo[ki];
              sq_hi += x_cache_hi[ki] * x_cache_hi[ki];
              d0_lo += x_cache_lo[ki] * fVec::loadu(w0 + k);
              d0_hi += x_cache_hi[ki] * fVec::loadu(w0 + k + kFVecSize);
              d1_lo += x_cache_lo[ki] * fVec::loadu(w1 + k);
              d1_hi += x_cache_hi[ki] * fVec::loadu(w1 + k + kFVecSize);
              d2_lo += x_cache_lo[ki] * fVec::loadu(w2 + k);
              d2_hi += x_cache_hi[ki] * fVec::loadu(w2 + k + kFVecSize);
              d3_lo += x_cache_lo[ki] * fVec::loadu(w3 + k);
              d3_hi += x_cache_hi[ki] * fVec::loadu(w3 + k + kFVecSize);
            }

            fVec x_tail_lo(0.f), x_tail_hi(0.f);
            if (rem > 0) {
              std::tie(x_tail_lo, x_tail_hi) = at::vec::convert_to_float(bVec::loadu(x_t + k_tail, rem));
              sq_lo += x_tail_lo * x_tail_lo;
              if (rem1 > 0) sq_hi += x_tail_hi * x_tail_hi;
              d0_lo += x_tail_lo * fVec::loadu(w0 + k_tail, rem0);
              d1_lo += x_tail_lo * fVec::loadu(w1 + k_tail, rem0);
              d2_lo += x_tail_lo * fVec::loadu(w2 + k_tail, rem0);
              d3_lo += x_tail_lo * fVec::loadu(w3 + k_tail, rem0);
              if (rem1 > 0) {
                d0_hi += x_tail_hi * fVec::loadu(w0 + k_tail + kFVecSize, rem1);
                d1_hi += x_tail_hi * fVec::loadu(w1 + k_tail + kFVecSize, rem1);
                d2_hi += x_tail_hi * fVec::loadu(w2 + k_tail + kFVecSize, rem1);
                d3_hi += x_tail_hi * fVec::loadu(w3 + k_tail + kFVecSize, rem1);
              }
            }

            partial_row[0] = vec_reduce_sum(sq_lo + sq_hi);
            partial_row[1 + 0] = vec_reduce_sum(d0_lo + d0_hi);
            partial_row[1 + 1] = vec_reduce_sum(d1_lo + d1_hi);
            partial_row[1 + 2] = vec_reduce_sum(d2_lo + d2_hi);
            partial_row[1 + 3] = vec_reduce_sum(d3_lo + d3_hi);

            // h=4..mix_hc-1: reuse cached x, four dots per pass.
            int h = 4;
            for (; h + 3 < mix_hc; h += 4) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              const float* wc = w_ptr[h + 2];
              const float* wd = w_ptr[h + 3];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              fVec c0(0.f), c1(0.f), d0(0.f), d1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                c0 += x_cache_lo[ki] * fVec::loadu(wc + k);
                d0 += x_cache_lo[ki] * fVec::loadu(wd + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
                c1 += x_cache_hi[ki] * fVec::loadu(wc + k + kFVecSize);
                d1 += x_cache_hi[ki] * fVec::loadu(wd + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                c0 += x_tail_lo * fVec::loadu(wc + k_tail, rem0);
                d0 += x_tail_lo * fVec::loadu(wd + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                  c1 += x_tail_hi * fVec::loadu(wc + k_tail + kFVecSize, rem1);
                  d1 += x_tail_hi * fVec::loadu(wd + k_tail + kFVecSize, rem1);
                }
              }
              partial_row[1 + h] = vec_reduce_sum(a0 + a1);
              partial_row[1 + h + 1] = vec_reduce_sum(b0 + b1);
              partial_row[1 + h + 2] = vec_reduce_sum(c0 + c1);
              partial_row[1 + h + 3] = vec_reduce_sum(d0 + d1);
            }
            for (; h + 1 < mix_hc; h += 2) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                }
              }
              partial_row[1 + h] = vec_reduce_sum(a0 + a1);
              partial_row[1 + h + 1] = vec_reduce_sum(b0 + b1);
            }
            // odd leftover
            if (h < mix_hc) {
              const float* w = w_ptr[h];
              fVec hd0(0.f), hd1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                hd0 += x_cache_lo[ki] * fVec::loadu(w + k);
                hd1 += x_cache_hi[ki] * fVec::loadu(w + k + kFVecSize);
              }
              if (rem > 0) {
                hd0 += x_tail_lo * fVec::loadu(w + k_tail, rem0);
                if (rem1 > 0) hd1 += x_tail_hi * fVec::loadu(w + k_tail + kFVecSize, rem1);
              }
              partial_row[1 + h] = vec_reduce_sum(hd0 + hd1);
            }
          }
        }  // tl

        // Prefetch next t-block head.
        if (tb + 1 < tb1) {
          const scalar_t* x_next_tb = x + ((tb + 1) * T_BLOCK) * hc_d + k0;
          __builtin_prefetch(x_next_tb, 0, 0);
          if (klen > kVecSize * 4) __builtin_prefetch(x_next_tb + kVecSize * 4, 0, 0);
        }
      }  // tb
    }  // kb
  });  // Phase 1

  // Phase 2+3: reduce partials, sinkhorn, and combine.
  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    fVec v0, v1, pre_fvec;
    float* const thread_scratch = scratch_base + tid * per_thread_scratch;
    float* const mix_buffer = thread_scratch;
    float* const pre_gate = thread_scratch + mix_hc;
    float* const post_gate = pre_gate + hc;
    float* const comb_matrix = post_gate + hc;
    float* const sinkhorn_scratch = comb_matrix + hc * hc;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t tb = t / T_BLOCK;
      const int64_t tl = t % T_BLOCK;

      std::fill(mix_buffer, mix_buffer + mix_hc, 0.f);
      double sq_total = 0.0;
      for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
        const float* partial_row = partials + (tb * n_k_blocks + kb) * T_BLOCK * partial_stride + tl * partial_stride;
        sq_total += static_cast<double>(partial_row[0]);
#pragma GCC unroll 4
        for (int i = 0; i < mix_hc; ++i)
          mix_buffer[i] += partial_row[1 + i];
      }
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));
#pragma GCC unroll 4
      for (int i = 0; i < mix_hc; ++i)
        mix_buffer[i] *= inv_rms;

      parse_mixes_and_sinkhorn(
          pre_gate,
          post_gate,
          comb_matrix,
          mix_buffer,
          s0,
          s1,
          s2,
          hc_base,
          sinkhorn_iters,
          hc_eps,
          hc,
          sinkhorn_scratch);

      std::copy_n(post_gate, hc, post_out + t * hc);
      std::copy_n(comb_matrix, hc * hc, comb_out + t * hc * hc);

      // Phase 3: d-tiled combine.
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;
      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        int64_t kk = 0;
        for (; kk <= jlen - kVecSize; kk += kVecSize) {
          pre_fvec = fVec(pre_gate[0]);
          std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_t + j0 + kk));
          fVec acc0 = pre_fvec * v0;
          fVec acc1 = pre_fvec * v1;
#pragma GCC unroll 4
          for (int h = 1; h < hc; ++h) {
            pre_fvec = fVec(pre_gate[h]);
            const scalar_t* x_th = x_t + (int64_t)h * d + j0 + kk;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th));
            acc0 += pre_fvec * v0;
            acc1 += pre_fvec * v1;
          }
          at::vec::convert_from_float<scalar_t>(acc0, acc1).store(y_t + j0 + kk);
        }
        if (kk < jlen) {
          const int64_t rem = jlen - kk, rem1 = rem - std::min(rem, kFVecSize);
          pre_fvec = fVec(pre_gate[0]);
          std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_t + j0 + kk, rem));
          fVec acc0 = pre_fvec * v0;
          fVec acc1 = pre_fvec * v1;
#pragma GCC unroll 4
          for (int h = 1; h < hc; ++h) {
            pre_fvec = fVec(pre_gate[h]);
            const scalar_t* x_th = x_t + (int64_t)h * d + j0 + kk;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th, rem));
            acc0 += pre_fvec * v0;
            acc1 += pre_fvec * v1;
          }
          at::vec::convert_from_float<scalar_t>(acc0, rem1 > 0 ? acc1 : fVec(0.f)).store(y_t + j0 + kk, rem);
        }
      }  // d-tile
    }  // t
  });  // Phase 2+3
}

// Fused hc_head path.
template <typename scalar_t>
static void hc_head_fuse_impl(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ hc_fn,  // [hc, hc*d] row-major
    float hc_scale_val,
    const float* __restrict__ hc_base,
    int64_t T,
    int64_t d,
    int hc,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();   // 32 bf16
  constexpr int64_t kFVecSize = fVec::size();  // 16 fp32
  constexpr int64_t kDVecsPerBlock = 16;
  const int64_t hc_d = hc * d;

  constexpr int64_t K_BLOCK_CAP = 1024;
  constexpr int64_t MAX_K_VECS = K_BLOCK_CAP / kVecSize;

  const int64_t T_BLOCK = choose_t_block(T);
  const int64_t num_threads = static_cast<int64_t>(at::get_num_threads());
  const int64_t K_BLOCK = choose_k_block(hc, kVecSize, K_BLOCK_CAP);
  TORCH_CHECK(
      K_BLOCK <= K_BLOCK_CAP && K_BLOCK % kVecSize == 0,
      "hc_head_fuse_impl: K_BLOCK must be <= K_BLOCK_CAP and a multiple of vector size");
  const int64_t D_BLOCK = kDVecsPerBlock * kVecSize;

  const int64_t n_t_blocks = div_up(T, T_BLOCK);
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);

  // Partial buffer: [n_t_blocks, n_k_blocks, T_BLOCK, 1+hc]
  //   p[0] = partial sq,  p[1..hc] = partial dots
  const int64_t partial_stride = 1 + hc;
  auto partials_tensor = at::empty({n_t_blocks * n_k_blocks * T_BLOCK * partial_stride}, at::kFloat);
  float* const partials = partials_tensor.data_ptr<float>();

  // Per-thread scratch: [pre_gate[hc]]
  const int64_t per_thread_head = hc;
  auto scratch_tensor = at::empty({num_threads * per_thread_head}, at::kFloat);
  float* const scratch_base = scratch_tensor.data_ptr<float>();

  // Phase 1.
  parallel_mhc_phase1(n_t_blocks, n_k_blocks, [&](int64_t tb0, int64_t tb1, int64_t kb0, int64_t kb1) {
    for (int64_t kb = kb0; kb < kb1; ++kb) {
      const int64_t k0 = kb * K_BLOCK;
      const int64_t klen = std::min(K_BLOCK, hc_d - k0);

      for (int64_t tb = tb0; tb < tb1; ++tb) {
        const int64_t t0 = tb * T_BLOCK;
        const int64_t tlen = std::min(T_BLOCK, T - t0);
        float* partial_block = partials + (tb * n_k_blocks + kb) * T_BLOCK * partial_stride;

        const float* w_ptr[hc];
        for (int h = 0; h < hc; ++h)
          w_ptr[h] = hc_fn + h * hc_d + k0;

        // Stack cache is sized for the largest selected K block, but only the first
        // n_full entries are touched for the current K_BLOCK.
        fVec x_cache_lo[MAX_K_VECS], x_cache_hi[MAX_K_VECS];

        for (int64_t tl = 0; tl < tlen; ++tl) {
          const scalar_t* x_t = x + (t0 + tl) * hc_d + k0;
          float* partial_row = partial_block + tl * partial_stride;

          const int64_t n_full = klen / kVecSize;
          const int64_t k_tail = n_full * kVecSize;
          const int64_t rem = klen - k_tail;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;

          // Prefetch next token.
          if (tl + 1 < tlen) {
            const scalar_t* xn = x + (t0 + tl + 1) * hc_d + k0;
            __builtin_prefetch(xn, 0, 0);
            if (klen > kVecSize * 4) __builtin_prefetch(xn + kVecSize * 4, 0, 0);
          }

          // Convert/cache x and accumulate sq + multiple dots.
          if (tl == 0) {
            for (int pw = 1; pw < hc; ++pw)
              __builtin_prefetch(w_ptr[pw], 0, 3);
          }
          {
            fVec x_tail_lo(0.f), x_tail_hi(0.f);
            int h = 0;

            if (hc >= 4) {
              const float* w0 = w_ptr[0];
              const float* w1 = w_ptr[1];
              const float* w2 = w_ptr[2];
              const float* w3 = w_ptr[3];
              fVec sq_lo(0.f), sq_hi(0.f);
              fVec d0_lo(0.f), d0_hi(0.f);
              fVec d1_lo(0.f), d1_hi(0.f);
              fVec d2_lo(0.f), d2_hi(0.f);
              fVec d3_lo(0.f), d3_hi(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                std::tie(x_cache_lo[ki], x_cache_hi[ki]) = at::vec::convert_to_float(bVec::loadu(x_t + k));
                sq_lo += x_cache_lo[ki] * x_cache_lo[ki];
                sq_hi += x_cache_hi[ki] * x_cache_hi[ki];
                d0_lo += x_cache_lo[ki] * fVec::loadu(w0 + k);
                d0_hi += x_cache_hi[ki] * fVec::loadu(w0 + k + kFVecSize);
                d1_lo += x_cache_lo[ki] * fVec::loadu(w1 + k);
                d1_hi += x_cache_hi[ki] * fVec::loadu(w1 + k + kFVecSize);
                d2_lo += x_cache_lo[ki] * fVec::loadu(w2 + k);
                d2_hi += x_cache_hi[ki] * fVec::loadu(w2 + k + kFVecSize);
                d3_lo += x_cache_lo[ki] * fVec::loadu(w3 + k);
                d3_hi += x_cache_hi[ki] * fVec::loadu(w3 + k + kFVecSize);
              }
              if (rem > 0) {
                std::tie(x_tail_lo, x_tail_hi) = at::vec::convert_to_float(bVec::loadu(x_t + k_tail, rem));
                sq_lo += x_tail_lo * x_tail_lo;
                if (rem1 > 0) sq_hi += x_tail_hi * x_tail_hi;
                d0_lo += x_tail_lo * fVec::loadu(w0 + k_tail, rem0);
                d1_lo += x_tail_lo * fVec::loadu(w1 + k_tail, rem0);
                d2_lo += x_tail_lo * fVec::loadu(w2 + k_tail, rem0);
                d3_lo += x_tail_lo * fVec::loadu(w3 + k_tail, rem0);
                if (rem1 > 0) {
                  d0_hi += x_tail_hi * fVec::loadu(w0 + k_tail + kFVecSize, rem1);
                  d1_hi += x_tail_hi * fVec::loadu(w1 + k_tail + kFVecSize, rem1);
                  d2_hi += x_tail_hi * fVec::loadu(w2 + k_tail + kFVecSize, rem1);
                  d3_hi += x_tail_hi * fVec::loadu(w3 + k_tail + kFVecSize, rem1);
                }
              }
              partial_row[0] = vec_reduce_sum(sq_lo + sq_hi);
              partial_row[1 + 0] = vec_reduce_sum(d0_lo + d0_hi);
              partial_row[1 + 1] = vec_reduce_sum(d1_lo + d1_hi);
              partial_row[1 + 2] = vec_reduce_sum(d2_lo + d2_hi);
              partial_row[1 + 3] = vec_reduce_sum(d3_lo + d3_hi);
              h = 4;
            } else {
              const float* w0 = w_ptr[0];
              fVec sq_lo(0.f), sq_hi(0.f);
              fVec d0(0.f), d1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                std::tie(x_cache_lo[ki], x_cache_hi[ki]) = at::vec::convert_to_float(bVec::loadu(x_t + k));
                sq_lo += x_cache_lo[ki] * x_cache_lo[ki];
                sq_hi += x_cache_hi[ki] * x_cache_hi[ki];
                d0 += x_cache_lo[ki] * fVec::loadu(w0 + k);
                d1 += x_cache_hi[ki] * fVec::loadu(w0 + k + kFVecSize);
              }
              if (rem > 0) {
                std::tie(x_tail_lo, x_tail_hi) = at::vec::convert_to_float(bVec::loadu(x_t + k_tail, rem));
                sq_lo += x_tail_lo * x_tail_lo;
                if (rem1 > 0) sq_hi += x_tail_hi * x_tail_hi;
                d0 += x_tail_lo * fVec::loadu(w0 + k_tail, rem0);
                if (rem1 > 0) d1 += x_tail_hi * fVec::loadu(w0 + k_tail + kFVecSize, rem1);
              }
              partial_row[0] = vec_reduce_sum(sq_lo + sq_hi);
              partial_row[1 + 0] = vec_reduce_sum(d0 + d1);
              h = 1;
            }

            // h: reuse cached x, four dots per pass.
            for (; h + 3 < hc; h += 4) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              const float* wc = w_ptr[h + 2];
              const float* wd = w_ptr[h + 3];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              fVec c0(0.f), c1(0.f), d0(0.f), d1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                c0 += x_cache_lo[ki] * fVec::loadu(wc + k);
                d0 += x_cache_lo[ki] * fVec::loadu(wd + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
                c1 += x_cache_hi[ki] * fVec::loadu(wc + k + kFVecSize);
                d1 += x_cache_hi[ki] * fVec::loadu(wd + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                c0 += x_tail_lo * fVec::loadu(wc + k_tail, rem0);
                d0 += x_tail_lo * fVec::loadu(wd + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                  c1 += x_tail_hi * fVec::loadu(wc + k_tail + kFVecSize, rem1);
                  d1 += x_tail_hi * fVec::loadu(wd + k_tail + kFVecSize, rem1);
                }
              }
              partial_row[1 + h] = vec_reduce_sum(a0 + a1);
              partial_row[1 + h + 1] = vec_reduce_sum(b0 + b1);
              partial_row[1 + h + 2] = vec_reduce_sum(c0 + c1);
              partial_row[1 + h + 3] = vec_reduce_sum(d0 + d1);
            }
            for (; h + 1 < hc; h += 2) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                }
              }
              partial_row[1 + h] = vec_reduce_sum(a0 + a1);
              partial_row[1 + h + 1] = vec_reduce_sum(b0 + b1);
            }
            // odd leftover
            if (h < hc) {
              const float* w = w_ptr[h];
              fVec hd0(0.f), hd1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                hd0 += x_cache_lo[ki] * fVec::loadu(w + k);
                hd1 += x_cache_hi[ki] * fVec::loadu(w + k + kFVecSize);
              }
              if (rem > 0) {
                hd0 += x_tail_lo * fVec::loadu(w + k_tail, rem0);
                if (rem1 > 0) hd1 += x_tail_hi * fVec::loadu(w + k_tail + kFVecSize, rem1);
              }
              partial_row[1 + h] = vec_reduce_sum(hd0 + hd1);
            }
          }
        }  // tl

        // Prefetch next t-block head.
        if (tb + 1 < tb1) {
          const scalar_t* x_next_tb = x + ((tb + 1) * T_BLOCK) * hc_d + k0;
          __builtin_prefetch(x_next_tb, 0, 0);
          if (klen > kVecSize * 2) __builtin_prefetch(x_next_tb + kVecSize * 2, 0, 0);
        }
      }  // tb
    }  // kb
  });  // Phase 1

  // Phase 2+3.
  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const pre_gate = scratch_base + tid * per_thread_head;
    fVec v0, v1, pre_fvec;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t tb = t / T_BLOCK;
      const int64_t tl = t % T_BLOCK;

      // Phase 2: reduce partials + sigmoid gate.
      double sq_total = 0.0;
      std::fill(pre_gate, pre_gate + hc, 0.f);
      for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
        const float* partial_row = partials + (tb * n_k_blocks + kb) * T_BLOCK * partial_stride + tl * partial_stride;
        sq_total += static_cast<double>(partial_row[0]);
#pragma GCC unroll 4
        for (int h = 0; h < hc; ++h)
          pre_gate[h] += partial_row[1 + h];
      }
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));
#pragma GCC unroll 4
      for (int h = 0; h < hc; ++h) {
        const float gate = pre_gate[h] * inv_rms * hc_scale_val + hc_base[h];
        pre_gate[h] = 1.f / (1.f + std::exp(-gate)) + hc_eps;
      }

      // Phase 3: d-tiled combine.
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;
      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        int64_t kk = 0;
        for (; kk <= jlen - kVecSize; kk += kVecSize) {
          pre_fvec = fVec(pre_gate[0]);
          std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_t + j0 + kk));
          fVec acc0 = pre_fvec * v0;
          fVec acc1 = pre_fvec * v1;
#pragma GCC unroll 4
          for (int h = 1; h < hc; ++h) {
            pre_fvec = fVec(pre_gate[h]);
            const scalar_t* x_th = x_t + (int64_t)h * d + j0 + kk;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th));
            acc0 += pre_fvec * v0;
            acc1 += pre_fvec * v1;
          }
          at::vec::convert_from_float<scalar_t>(acc0, acc1).store(y_t + j0 + kk);
        }
        if (kk < jlen) {
          const int64_t rem = jlen - kk, rem1 = rem - std::min(rem, kFVecSize);
          pre_fvec = fVec(pre_gate[0]);
          std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_t + j0 + kk, rem));
          fVec acc0 = pre_fvec * v0;
          fVec acc1 = pre_fvec * v1;
#pragma GCC unroll 4
          for (int h = 1; h < hc; ++h) {
            pre_fvec = fVec(pre_gate[h]);
            const scalar_t* x_th = x_t + (int64_t)h * d + j0 + kk;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th, rem));
            acc0 += pre_fvec * v0;
            acc1 += pre_fvec * v1;
          }
          at::vec::convert_from_float<scalar_t>(acc0, rem1 > 0 ? acc1 : fVec(0.f)).store(y_t + j0 + kk, rem);
        }
      }  // d-tile
    }  // t
  });  // Phase 2+3
}

}  // anonymous namespace

// Fused hc_pre entry.
std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_fused_cpu(
    at::Tensor& x,
    at::Tensor& hc_fn,
    at::Tensor& hc_scale,
    at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double rms_eps,
    double hc_eps) {
  TORCH_CHECK(hc_mult >= 2, "hc_pre_fused_cpu: hc_mult must be >= 2");
  TORCH_CHECK(
      x.dim() == 3 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
      "hc_pre_fused_cpu: x must be bf16/fp16 [T, hc, d]");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_fn must be float32");
  TORCH_CHECK(
      hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3, "hc_pre_fused_cpu: hc_scale must be float32 [3]");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_base must be float32");
  TORCH_CHECK(
      x.device().is_cpu() && hc_fn.device().is_cpu() && hc_scale.device().is_cpu() && hc_base.device().is_cpu(),
      "hc_pre_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  const int64_t hc_d = hc_mult * d;
  const int64_t mix_hc = (2 + hc_mult) * hc_mult;

  TORCH_CHECK(T > 0 && d > 0, "hc_pre_fused_cpu: T and d must be positive");
  TORCH_CHECK(x.size(1) == hc_mult, "hc_pre_fused_cpu: x shape must be [T, hc_mult, d]");
  TORCH_CHECK(
      hc_fn.dim() == 2 && hc_fn.size(0) == mix_hc && hc_fn.size(1) == hc_d,
      "hc_pre_fused_cpu: hc_fn must be [mix_hc, hc_mult*d]");
  TORCH_CHECK(hc_base.numel() == mix_hc, "hc_pre_fused_cpu: hc_base.numel() must equal mix_hc");

  // Static per-layer weights (hc_fn/hc_scale/hc_base) are allocated fp32 and
  // never reassigned after weight loading; x/residual come straight from a
  // preceding op. All are expected contiguous already -- fail loudly instead
  // of silently absorbing a copy if that contract is ever violated.
  TORCH_CHECK(
      x.is_contiguous() && hc_fn.is_contiguous() && hc_scale.is_contiguous() && hc_base.is_contiguous(),
      "hc_pre_fused_cpu: x/hc_fn/hc_scale/hc_base must be contiguous");

  auto f32_opts = at::TensorOptions().dtype(at::kFloat).device(x.device());
  auto post = at::empty({T, hc_mult}, f32_opts);
  auto comb = at::empty({T, hc_mult, hc_mult}, f32_opts);
  auto y = at::empty({T, d}, x.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "hc_pre_fused_cpu", [&] {
    hc_pre_fuse_impl<scalar_t>(
        y.data_ptr<scalar_t>(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        x.data_ptr<scalar_t>(),
        hc_fn.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<int>(hc_mult),
        static_cast<int>(sinkhorn_iters),
        static_cast<float>(hc_eps),
        static_cast<float>(rms_eps));
  });
  return {y, post, comb};
}

// Fused hc_post entry.
at::Tensor hc_post_fused_cpu(at::Tensor& x, at::Tensor& residual, at::Tensor& post, at::Tensor& comb) {
  TORCH_CHECK(
      x.dim() == 2 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
      "hc_post_fused_cpu: x must be bf16/fp16 [T, d]");
  TORCH_CHECK(
      residual.dim() == 3 && residual.scalar_type() == x.scalar_type(),
      "hc_post_fused_cpu: residual must be same dtype as x [T, hc, d]");
  TORCH_CHECK(post.scalar_type() == at::kFloat, "hc_post_fused_cpu: post must be float32");
  TORCH_CHECK(comb.scalar_type() == at::kFloat, "hc_post_fused_cpu: comb must be float32");
  TORCH_CHECK(
      x.device().is_cpu() && residual.device().is_cpu() && post.device().is_cpu() && comb.device().is_cpu(),
      "hc_post_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(1);
  const int64_t hc = residual.size(1);
  TORCH_CHECK(hc >= 2, "hc_post_fused_cpu: hc_mult must be >= 2");
  TORCH_CHECK(T > 0 && d > 0, "hc_post_fused_cpu: T and d must be positive");
  TORCH_CHECK(residual.size(0) == T && residual.size(2) == d, "hc_post_fused_cpu: residual shape must be [T, hc, d]");
  TORCH_CHECK(
      post.dim() == 2 && post.size(0) == T && post.size(1) == hc, "hc_post_fused_cpu: post shape must be [T, hc]");
  TORCH_CHECK(
      comb.dim() == 3 && comb.size(0) == T && comb.size(1) == hc && comb.size(2) == hc,
      "hc_post_fused_cpu: comb shape must be [T, hc, hc]");

  TORCH_CHECK(
      x.is_contiguous() && residual.is_contiguous() && post.is_contiguous() && comb.is_contiguous(),
      "hc_post_fused_cpu: x/residual/post/comb must be contiguous");

  auto out = at::empty({T, hc, d}, x.options());  // same dtype as x

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "hc_post_fused_cpu", [&] {
    hc_post_fuse_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        T,
        d,
        static_cast<int>(hc));
  });

  return out;
}

// Fused hc_head entry.
at::Tensor hc_head_fused_cpu(
    at::Tensor& x, at::Tensor& hc_fn, at::Tensor& hc_scale, at::Tensor& hc_base, double hc_eps, double norm_eps) {
  TORCH_CHECK(
      x.dim() == 3 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
      "hc_head_fused_cpu: x must be bf16/fp16 [T, hc, d]");
  const int64_t hc_mult = x.size(1);
  TORCH_CHECK(hc_mult >= 2, "hc_head_fused_cpu: hc_mult must be >= 2");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_head_fused_cpu: hc_fn must be float32");
  TORCH_CHECK(
      hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 1,
      "hc_head_fused_cpu: hc_scale must be a scalar float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_head_fused_cpu: hc_base must be float32");
  TORCH_CHECK(hc_base.numel() == hc_mult, "hc_head_fused_cpu: hc_base.numel() must equal hc_mult");
  TORCH_CHECK(
      x.device().is_cpu() && hc_fn.device().is_cpu() && hc_scale.device().is_cpu() && hc_base.device().is_cpu(),
      "hc_head_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  TORCH_CHECK(T > 0 && d > 0, "hc_head_fused_cpu: T and d must be positive");
  TORCH_CHECK(
      hc_fn.dim() == 2 && hc_fn.size(0) == hc_mult && hc_fn.size(1) == hc_mult * d,
      "hc_head_fused_cpu: hc_fn must be [hc_mult, hc_mult*d]");

  TORCH_CHECK(
      x.is_contiguous() && hc_fn.is_contiguous() && hc_scale.is_contiguous() && hc_base.is_contiguous(),
      "hc_head_fused_cpu: x/hc_fn/hc_scale/hc_base must be contiguous");
  const float hc_scale_val = hc_scale.data_ptr<float>()[0];

  auto y = at::empty({T, d}, x.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "hc_head_fused_cpu", [&] {
    hc_head_fuse_impl<scalar_t>(
        y.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        hc_fn.data_ptr<float>(),
        hc_scale_val,
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<int>(hc_mult),
        static_cast<float>(hc_eps),
        static_cast<float>(norm_eps));
  });

  return y;
}
