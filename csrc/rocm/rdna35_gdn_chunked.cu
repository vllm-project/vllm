// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Chunked delta rule (WY representation / UT transform) for the scalar-gate
// gated delta net, on RDNA3.5.
//
// The sequence is cut into chunks of GDN_CHUNK tokens and the rank-1 state
// updates inside a chunk are folded into (I + A)^-1, so the recurrence's inner
// loops become dense matmuls over LDS.  One block walks the chunks of one
// (head, sequence), which keeps the per-chunk intermediates in LDS and the
// state in registers.
//
// clang-format off
// Per head and chunk, rows are tokens and indices are (row, col):
//   g_cs        = inclusive cumsum of the log decay within the chunk,
//                 g_last = g_cs[GDN_CHUNK-1]
//   A[i][j]     = beta[i] * dot(K_i, K_j) * exp(g_cs[i] - g_cs[j])   j <  i, else 0
//   Tinv        = (I + A)^-1, unit lower triangular
//   qk[i][j]    = dot(Q_i, K_j) * exp(g_cs[i] - g_cs[j])             j <= i, else 0
//   Y[r][c]     = V[r][c]*beta[r] - sum_a K[r][a]*beta[r]*exp(g_cs[r]) * S[c][a]
//   v_new[t][c] = sum_{r<=t} Tinv[t][r] * Y[r][c]
//   out[i][c]   = sum_a S[c][a]*Q[i][a]*exp(g_cs[i]) + sum_t v_new[t][c]*qk[i][t]
//   S[c][a]     = S[c][a]*exp(g_last) + sum_t K[t][a]*exp(g_last-g_cs[t]) * v_new[t][c]
// clang-format on
//
// Everything after Tinv is independent per state column, which is what lets the
// state live in registers spread across the block.
//
// Two constraints keep a whole chunk inside the 64 KB LDS budget:
//   - GDN_CHUNK is 32.  At 64 the chunk's U, W and qk alone need 80 KB.
//   - U and W are never materialised.  v_new = U - W S, with U = Tinv (V beta)
//     and W = Tinv (K beta exp(g_cs)), is rewritten as the identical
//     v_new = Tinv (V beta - (K beta exp(g_cs)) S), so one
//     [GDN_CHUNK][GDN_HEAD_DIM] buffer holds V beta, becomes Y, then becomes
//     v_new in place.
//
// q, k, v and the output are bf16 or fp16 (all four alike, chosen by the
// IS_FP16 template parameter); g, beta and the state are f32. Sequences are
// packed varlen behind cu_seqlens. A value head reads key head head / (H / Hg).
// g is the raw per-token log decay: the cumsum is taken here, over this
// kernel's own chunk.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>

#include <bit>
#include <cstdint>
#include <string>
#include <type_traits>

namespace {

constexpr int GDN_CHUNK = 32;
constexpr int GDN_HEAD_DIM = 128;  // the only head size this path handles
constexpr int GDN_BLOCK_SIZE = 1024;
constexpr int GDN_WAVE = 32;

// +1 breaks the power-of-two stride so that a column walk spreads over the LDS
// banks
constexpr int GDN_HEAD_PITCH = GDN_HEAD_DIM + 1;
constexpr int GDN_CHUNK_PITCH = GDN_CHUNK + 1;

// Lanes cooperating on one adjacent pair of state columns.  Sixteen of them fit
// inside a wave32, so their reductions stay cross-lane with no LDS traffic and
// no barrier, and holding a pair rather than a single column lets every staged
// K/Q element feed two FMAs.
constexpr int GDN_COL_LANES = 16;
constexpr int GDN_ROWS_PER_LANE = GDN_HEAD_DIM / GDN_COL_LANES;

// The whole RDNA3.5 family: wave32, four SIMDs and 128 KB of LDS per WGP,
// which is what the block layout below assumes.  The four are spelled out
// because there is no gfx11.5 family macro; the compiler defines a per-target
// __gfxNNNN__ and __GFX11__, and the latter also covers gfx1100 through
// gfx1103, which are RDNA3 and do not share that layout.
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
    defined(__gfx1153__)
  #define GDN_SUPPORTED_ARCH 1
#endif

#ifdef GDN_SUPPORTED_ARCH
template <int N>
__device__ __forceinline__ float gdn_dpp_shl_add(float x) {
  // row_shl:N makes lane i read lane i+N within its row of 16, and the move
  // folds into the add
  const int y = __builtin_amdgcn_update_dpp(0, std::bit_cast<int>(x), 0x100 | N,
                                            0xf, 0xf, true);
  return x + std::bit_cast<float>(y);
}
#endif

// Sum x across the GDN_COL_LANES lanes of a column pair; only lane 0 of the
// group ends up with the result, and the callers write from that lane.  On RDNA
// the DPP form keeps the reduction on the VALU, where the cross-lane move is a
// free modifier on the add, while the portable shuffle lowers to
// ds_bpermute_b32 and would contend with the staged tiles for the LDS pipe.
__device__ __forceinline__ float gdn_sum_to_lane0(float x) {
#ifdef GDN_SUPPORTED_ARCH
  static_assert(GDN_COL_LANES == 16,
                "the row_shl chain below covers exactly 16 lanes");
  x = gdn_dpp_shl_add<1>(x);
  x = gdn_dpp_shl_add<2>(x);
  x = gdn_dpp_shl_add<4>(x);
  x = gdn_dpp_shl_add<8>(x);
  return x;
#else
  #pragma unroll
  for (int mask = 1; mask < GDN_COL_LANES; mask <<= 1) {
    x += __shfl_xor(x, mask, GDN_WAVE);
  }
  return x;
#endif
}

// The decay cumsum is converted to base 2 once per chunk, so every decay
// afterwards is a single v_exp_f32 with no scaling multiply.  The raw
// instruction also skips the denormal rescue that OCML wraps around expf, which
// costs nothing here: an exponent that far negative means a fully decayed
// state, and flushing it to zero is the intended result.
constexpr float GDN_LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float gdn_exp2(float x) {
  return __builtin_amdgcn_exp2f(x);
}

__device__ __forceinline__ float gdn_bf16_to_f32(uint16_t x) {
  return std::bit_cast<float>(static_cast<uint32_t>(x) << 16);
}

// Round to nearest even.
__device__ __forceinline__ uint16_t gdn_f32_to_bf16(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>(0x7fc0);  // quiet NaN
  }
  const uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

// fp16 goes through the hardware conversions rather than the hand-rolled shift
// and round above: bf16 is a truncation of f32 so it is a couple of integer
// ops, while fp16 needs a different exponent bias and subnormal handling that
// v_cvt_f32_f16 / v_cvt_f16_f32 already do in one instruction each.
__device__ __forceinline__ float gdn_fp16_to_f32(uint16_t x) {
  return static_cast<float>(std::bit_cast<_Float16>(x));
}

__device__ __forceinline__ uint16_t gdn_f32_to_fp16(float f) {
  return std::bit_cast<uint16_t>(static_cast<_Float16>(f));
}

// The element type is a compile-time parameter so the conversion resolves at
// compile time: this sits in the innermost load loop, and a runtime branch
// there would cost more than the duplicated code costs.
template <bool IS_FP16>
__device__ __forceinline__ float gdn_load(uint16_t x) {
  if constexpr (IS_FP16) {
    return gdn_fp16_to_f32(x);
  } else {
    return gdn_bf16_to_f32(x);
  }
}

template <bool IS_FP16>
__device__ __forceinline__ uint16_t gdn_store(float f) {
  if constexpr (IS_FP16) {
    return gdn_f32_to_fp16(f);
  } else {
    return gdn_f32_to_bf16(f);
  }
}

// A block of GDN_BLOCK_SIZE threads is 32 waves, so two blocks per WGP come to
// 16 waves per SIMD.  Requesting that caps the register allocation at
// file_size/16, and the family is not uniform in file size: gfx1151 has 1536
// registers per SIMD, so the cap lands at 96 and both blocks stay resident,
// while the other three have 1024, where the same request caps at 64 and
// spills 32 registers per thread.  Asking those for 8 gives 100 registers and
// no spill, at one block per WGP.
#if defined(__gfx1151__)
  #define GDN_MIN_WAVES_PER_SIMD 16
#elif defined(GDN_SUPPORTED_ARCH)
  #define GDN_MIN_WAVES_PER_SIMD 8
#else
  #define GDN_MIN_WAVES_PER_SIMD 1
#endif

// CU mode confines a block to one CU, which halves the LDS contention domain.
// It costs no residency: a WGP holds floor(128 KB / 57.25 KB) = 2 of these
// blocks in WGP mode and 2*floor(64 KB / 57.25 KB) = 2 in CU mode.  LDS is
// what binds on gfx1151; on the 1024-register parts the register file binds
// first and holds one block either way.  It only pays once every CU has a
// block to run, hence the launch-time choice below.
#ifdef GDN_SUPPORTED_ARCH
  #define GDN_CU_MODE __attribute__((target("cumode")))
#else
  #define GDN_CU_MODE
#endif

// The work-group processor mode is a function attribute, so it cannot be a
// template parameter: the two entry points below share this body and differ
// only in that attribute.
#define GDN_CHUNKED_PARAMS                                                \
  const uint16_t *__restrict__ q, const uint16_t *__restrict__ k,         \
      const uint16_t *__restrict__ v, const float *__restrict__ g,        \
      const float *__restrict__ beta, const float *__restrict__ state_in, \
      uint16_t *__restrict__ dst, float *__restrict__ state_out,          \
      const int32_t *__restrict__ cu_seqlens, const int H, const int Hg,  \
      const int v_per_k, const float scale

#define GDN_CHUNKED_ARGS \
  q, k, v, g, beta, state_in, dst, state_out, cu_seqlens, H, Hg, v_per_k, scale

template <bool IS_FP16>
__device__ __forceinline__ void gdn_chunked_body(GDN_CHUNKED_PARAMS) {
  constexpr int CHUNK = GDN_CHUNK;
  constexpr int HEAD_DIM = GDN_HEAD_DIM;
  constexpr int COL_LANES = GDN_COL_LANES;
  constexpr int ROWS = GDN_ROWS_PER_LANE;

  const int head = blockIdx.x;
  const int sequence = blockIdx.y;
  const int tid = threadIdx.x;

  const int bos = cu_seqlens[sequence];
  const int eos = cu_seqlens[sequence + 1];
  const int n_tokens = eos - bos;

  const int head_qk = head / v_per_k;

  const int col_pair = tid / COL_LANES;
  const int col_lane = tid - col_pair * COL_LANES;
  const int col0 = col_pair * 2;
  const int col1 = col0 + 1;

  __shared__ float s_k[CHUNK][GDN_HEAD_PITCH];
  __shared__ float s_q[CHUNK][GDN_HEAD_PITCH];
  __shared__ float s_y[CHUNK][GDN_HEAD_PITCH];  // V*beta, then Y, then v_new
  __shared__ float s_tinv[CHUNK][GDN_CHUNK_PITCH];  // A, then Tinv
  __shared__ float s_qk[CHUNK][GDN_CHUNK_PITCH];
  __shared__ float s_g_cs[CHUNK];
  __shared__ float s_beta[CHUNK];
  __shared__ float s_beta_decay[CHUNK];  // beta[r] * exp(g_cs[r])
  __shared__ float s_decay[CHUNK];       // exp(g_cs[i])
  __shared__ float s_decay_end[CHUNK];   // exp(g_last - g_cs[t])

  // state rows owned by this thread: row = u*COL_LANES + col_lane, which keeps
  // the lanes of a reduction on consecutive LDS banks
  float state0[ROWS];
  float state1[ROWS];

  const int64_t state_off =
      (static_cast<int64_t>(sequence) * H + head) * HEAD_DIM * HEAD_DIM;
#pragma unroll
  for (int u = 0; u < ROWS; u++) {
    const int64_t idx = state_off + static_cast<int64_t>(col0) * HEAD_DIM +
                        u * COL_LANES + col_lane;
    state0[u] = state_in ? state_in[idx] : 0.0f;
    state1[u] = state_in ? state_in[idx + HEAD_DIM] : 0.0f;
  }

  // Only the lower triangles are written below: the packed loop covers j <= i
  // and the inversion stores zeros above the diagonal, so one clear here holds
  // for every chunk.
  for (int idx = tid; idx < CHUNK * GDN_CHUNK_PITCH; idx += GDN_BLOCK_SIZE) {
    s_tinv[idx / GDN_CHUNK_PITCH][idx % GDN_CHUNK_PITCH] = 0.0f;
    s_qk[idx / GDN_CHUNK_PITCH][idx % GDN_CHUNK_PITCH] = 0.0f;
  }

  const int n_chunks = (n_tokens + CHUNK - 1) / CHUNK;

  // One chunk of the sequence.  FULL says n_valid == CHUNK is known at compile
  // time, which is true of every chunk but the last of a sequence; the caller
  // below dispatches on it so the common case keeps constant trip counts.
  //
  // Rows past the end of the sequence are staged as zero and every phase they
  // feed is linear in them, so the partial path stops the row loops at n_valid:
  // Y, v_new and the state carry are unchanged and no output is stored.  A
  // sequence that ends just past a chunk boundary then pays for its tokens
  // rather than for a whole chunk.
  auto chunk_step = [&](auto full_tag, const int tok0, const int n_valid) {
    constexpr bool FULL = decltype(full_tag)::value;
    const int n_rows = FULL ? CHUNK : n_valid;

    __syncthreads();

    if (tid < CHUNK) {
      const int64_t gb_off = static_cast<int64_t>(bos + tok0 + tid) * H + head;
      const bool valid = FULL || tid < n_valid;
      s_g_cs[tid] = valid ? g[gb_off] : 0.0f;
      s_beta[tid] = valid ? beta[gb_off] : 0.0f;
    }
    __syncthreads();

    // CHUNK == GDN_WAVE, so the cumsum is one shuffle scan in wave 0,
    // overlapped with the K/Q/V staging done by the rest of the block.  The
    // scan result is still in registers here, so the per-token decay tables
    // cost no extra barrier.
    if (tid < GDN_WAVE) {
      float g_cs = s_g_cs[tid] * GDN_LOG2E;
#pragma unroll
      for (int off = 1; off < CHUNK; off <<= 1) {
        const float prev = __shfl_up(g_cs, off, GDN_WAVE);
        if (tid >= off) {
          g_cs += prev;
        }
      }
      const float g_last = __shfl(g_cs, CHUNK - 1, GDN_WAVE);

      s_g_cs[tid] = g_cs;
      s_beta_decay[tid] = s_beta[tid] * gdn_exp2(g_cs);
      s_decay[tid] = gdn_exp2(g_cs);
      s_decay_end[tid] = gdn_exp2(g_last - g_cs);
    }

    for (int idx = tid; idx < CHUNK * HEAD_DIM; idx += GDN_BLOCK_SIZE) {
      const int t = idx / HEAD_DIM;
      const int s = idx - t * HEAD_DIM;
      const bool valid = FULL || t < n_valid;
      const int64_t tok = bos + tok0 + t;

      const int64_t qk_idx =
          tok * Hg * HEAD_DIM + static_cast<int64_t>(head_qk) * HEAD_DIM + s;
      const int64_t v_idx =
          tok * H * HEAD_DIM + static_cast<int64_t>(head) * HEAD_DIM + s;

      s_k[t][s] = valid ? gdn_load<IS_FP16>(k[qk_idx]) : 0.0f;
      s_q[t][s] = valid ? gdn_load<IS_FP16>(q[qk_idx]) * scale : 0.0f;
      s_y[t][s] = valid ? gdn_load<IS_FP16>(v[v_idx]) : 0.0f;
    }
    __syncthreads();

    const float g_last = s_g_cs[CHUNK - 1];

    // A and qk, both triangular.  Mapping (i,j) to (tid/CHUNK, tid%CHUNK) would
    // leave half the machine idle: a wave covers one row i and still issues the
    // whole dot product with only its j < i lanes unmasked.  The two triangles
    // hold 496 + 496 + 32 entries, exactly the block size, so a
    // triangle-to-rectangle fold gives every lane one entry to compute.
    {
      constexpr int n_strict = CHUNK * (CHUNK - 1) / 2;

      int row_i, col_j;
      bool is_qk;
      if (tid < 2 * n_strict) {
        const int tri = tid < n_strict ? tid : tid - n_strict;

        is_qk = tid >= n_strict;

        const int blk = tri / (CHUNK / 2);
        const int off = tri - blk * (CHUNK / 2);
        if (off <= blk) {
          row_i = blk + 1;
          col_j = off;
        } else {
          row_i = CHUNK - 1 - blk;
          col_j = CHUNK - 1 - off;
        }
      } else {
        row_i = col_j = tid - 2 * n_strict;
        is_qk = true;
      }

      const float* __restrict__ lhs = is_qk ? &s_q[row_i][0] : &s_k[row_i][0];

      float acc0 = 0.0f;
      float acc1 = 0.0f;
      for (int s = 0; s < HEAD_DIM; s += 2) {
        acc0 += lhs[s] * s_k[col_j][s];
        acc1 += lhs[s + 1] * s_k[col_j][s + 1];
      }
      const float dot_decayed =
          (acc0 + acc1) * gdn_exp2(s_g_cs[row_i] - s_g_cs[col_j]);

      if (is_qk) {
        s_qk[row_i][col_j] = dot_decayed;
      } else {
        s_tinv[row_i][col_j] = s_beta[row_i] * dot_decayed;
      }
    }
    __syncthreads();

    // Tinv = (I+A)^-1 by right-looking forward substitution: once x[i] is
    // final, subtract A[r][i]*x[i] from the rows below it.  Lane r owns row r,
    // so the update is a plain FMA with no cross-lane reduction.  One column
    // per wave, and A is read-only until the store.
    {
      const int lane = tid & (GDN_WAVE - 1);
      const int col = tid / GDN_WAVE;

      float x = lane == col ? 1.0f : 0.0f;
      for (int i = col; i < CHUNK; i++) {
        const float x_i = __shfl(x, i, GDN_WAVE);
        if (lane > i) {
          x -= s_tinv[lane][i] * x_i;
        }
      }
      __syncthreads();
      s_tinv[lane][col] = x;
    }
    __syncthreads();

    // Y[r][c] = V[r][c]*beta[r] - sum_a K[r][a]*beta[r]*exp(g_cs[r]) * S[c][a]
    for (int r = 0; r < n_rows; r++) {
      float ks0 = 0.0f;
      float ks1 = 0.0f;
#pragma unroll
      for (int u = 0; u < ROWS; u++) {
        const float k_ra = s_k[r][u * COL_LANES + col_lane];
        ks0 += k_ra * state0[u];
        ks1 += k_ra * state1[u];
      }
      ks0 = gdn_sum_to_lane0(ks0);
      ks1 = gdn_sum_to_lane0(ks1);
      if (col_lane == 0) {
        s_y[r][col0] = s_y[r][col0] * s_beta[r] - s_beta_decay[r] * ks0;
        s_y[r][col1] = s_y[r][col1] * s_beta[r] - s_beta_decay[r] * ks1;
      }
    }
    __syncthreads();

    // v_new = Tinv * Y, in place.  Tinv is lower triangular, so row t only
    // reads Y[0..t]: accumulating the upper half into registers before storing
    // it leaves the lower half untouched for the second pass, and no second
    // [CHUNK][HEAD_DIM] buffer is needed.
    {
      const int row_hi = CHUNK / 2 + col_lane;
      float hi0 = 0.0f;
      float hi1 = 0.0f;
      // A row past n_valid reads only zeroed Y rows, so stopping at n_valid-1
      // drops exactly the wasted work; what it then stores into s_y is dead,
      // since the readers below either stop at n_valid or multiply it by a zero
      // qk entry.  Truncating the trip count rather than skipping the row is
      // what saves the work: a lane's row index is col_lane, so the rows of one
      // wave straddle n_valid and a guard would keep the wave running anyway.
      const int end_hi = FULL || row_hi < n_valid ? row_hi : n_valid - 1;
      for (int r = 0; r <= end_hi; r++) {
        const float tinv = s_tinv[row_hi][r];
        hi0 += tinv * s_y[r][col0];
        hi1 += tinv * s_y[r][col1];
      }
      __syncthreads();
      s_y[row_hi][col0] = hi0;
      s_y[row_hi][col1] = hi1;

      const int row_lo = col_lane;
      float lo0 = 0.0f;
      float lo1 = 0.0f;
      const int end_lo = FULL || row_lo < n_valid ? row_lo : n_valid - 1;
      for (int r = 0; r <= end_lo; r++) {
        const float tinv = s_tinv[row_lo][r];
        lo0 += tinv * s_y[r][col0];
        lo1 += tinv * s_y[r][col1];
      }
      __syncthreads();
      s_y[row_lo][col0] = lo0;
      s_y[row_lo][col1] = lo1;
    }
    __syncthreads();

    // out[i][c] = sum_a S[c][a]*Q[i][a]*exp(g_cs[i]) + sum_t
    // v_new[t][c]*qk[i][t]
    for (int i = 0; i < n_rows; i++) {
      float qs0 = 0.0f;
      float qs1 = 0.0f;
#pragma unroll
      for (int u = 0; u < ROWS; u++) {
        const float q_ia = s_q[i][u * COL_LANES + col_lane];
        qs0 += q_ia * state0[u];
        qs1 += q_ia * state1[u];
      }

      // This sweeps all CHUNK rows of v_new even when the chunk is partial.
      // That is deliberate and load bearing: qk[i][t] is zero for t > i, and
      // i < n_valid <= t on every extra row, so the rows past n_valid
      // contribute nothing no matter what they hold.  Keep it that way if the
      // triangular solve above ever stops leaving them well defined.
      float vk0 = 0.0f;
      float vk1 = 0.0f;
#pragma unroll
      for (int w = 0; w < CHUNK / COL_LANES; w++) {
        const int t = col_lane * (CHUNK / COL_LANES) + w;
        const float qk_it = s_qk[i][t];
        vk0 += s_y[t][col0] * qk_it;
        vk1 += s_y[t][col1] * qk_it;
      }

      // s_decay[i] is lane-uniform and the reduction is linear, so applying it
      // first leaves one value per column to reduce instead of two.
      float out0 = qs0 * s_decay[i] + vk0;
      float out1 = qs1 * s_decay[i] + vk1;
      out0 = gdn_sum_to_lane0(out0);
      out1 = gdn_sum_to_lane0(out1);

      if (col_lane == 0) {
        const int64_t dst_off =
            static_cast<int64_t>(bos + tok0 + i) * H * HEAD_DIM +
            static_cast<int64_t>(head) * HEAD_DIM + col0;
        // col0 is even, so the 16-bit pair is 4-byte aligned
        const uint32_t packed =
            static_cast<uint32_t>(gdn_store<IS_FP16>(out0)) |
            (static_cast<uint32_t>(gdn_store<IS_FP16>(out1)) << 16);
        *reinterpret_cast<uint32_t*>(&dst[dst_off]) = packed;
      }
    }

    // S[c][a] = S[c][a]*exp(g_last) + sum_t
    // K[t][a]*exp(g_last-g_cs[t])*v_new[t][c]
    const float chunk_decay = gdn_exp2(g_last);
#pragma unroll
    for (int u = 0; u < ROWS; u++) {
      state0[u] *= chunk_decay;
      state1[u] *= chunk_decay;
    }

    for (int t = 0; t < n_rows; t++) {
      const float decayed0 = s_decay_end[t] * s_y[t][col0];
      const float decayed1 = s_decay_end[t] * s_y[t][col1];
#pragma unroll
      for (int u = 0; u < ROWS; u++) {
        const float k_ta = s_k[t][u * COL_LANES + col_lane];
        state0[u] += k_ta * decayed0;
        state1[u] += k_ta * decayed1;
      }
    }
  };

  for (int chunk = 0; chunk < n_chunks; chunk++) {
    const int tok0 = chunk * CHUNK;
    const int n_valid = n_tokens - tok0 < CHUNK ? n_tokens - tok0 : CHUNK;
    if (n_valid == CHUNK) {
      chunk_step(std::true_type{}, tok0, CHUNK);
    } else {
      chunk_step(std::false_type{}, tok0, n_valid);
    }
  }

#pragma unroll
  for (int u = 0; u < ROWS; u++) {
    const int64_t idx = state_off + static_cast<int64_t>(col0) * HEAD_DIM +
                        u * COL_LANES + col_lane;
    state_out[idx] = state0[u];
    state_out[idx + HEAD_DIM] = state1[u];
  }
}

template <bool IS_FP16>
__global__ void __launch_bounds__(GDN_BLOCK_SIZE, GDN_MIN_WAVES_PER_SIMD)
    gdn_chunked_kernel_wgp(GDN_CHUNKED_PARAMS) {
  gdn_chunked_body<IS_FP16>(GDN_CHUNKED_ARGS);
}

template <bool IS_FP16>
__global__ void GDN_CU_MODE __launch_bounds__(GDN_BLOCK_SIZE,
                                              GDN_MIN_WAVES_PER_SIMD)
    gdn_chunked_kernel_cu(GDN_CHUNKED_PARAMS) {
  gdn_chunked_body<IS_FP16>(GDN_CHUNKED_ARGS);
}

}  // namespace

void gdn_chunked(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                 torch::Tensor& g, torch::Tensor& beta,
                 std::optional<torch::Tensor> initial_state,
                 torch::Tensor& cu_seqlens, torch::Tensor& out,
                 torch::Tensor& final_state, double scale) {
  // bf16 and fp16 are both 16 bits wide and the kernel works in fp32
  // throughout, so the element type only picks the conversion at the load and
  // the store.  All four tensors must agree: the kernel reads one type.
  const auto dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
              "q must be bf16 or fp16, got ", dtype);
  TORCH_CHECK(k.scalar_type() == dtype, "k must match q's dtype");
  TORCH_CHECK(v.scalar_type() == dtype, "v must match q's dtype");
  TORCH_CHECK(out.scalar_type() == dtype, "out must match q's dtype");
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be fp32");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be fp32");
  TORCH_CHECK(final_state.scalar_type() == at::kFloat, "state must be fp32");
  TORCH_CHECK(q.size(-1) == GDN_HEAD_DIM && v.size(-1) == GDN_HEAD_DIM,
              "this path handles head_dim 128 only");

  const int Hg = q.size(-2);
  const int H = v.size(-2);
  TORCH_CHECK(H % Hg == 0, "value heads must be a multiple of key heads");
  const int n_seqs = cu_seqlens.size(0) - 1;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* props = at::cuda::getCurrentDeviceProperties();

  // The block layout assumes wave32 throughout: GDN_CHUNK is the wave width, a
  // column pair maps to 16 lanes, and the inversion gives one column to each
  // wave. The device-side fallbacks let the file compile for any architecture,
  // so without this check a wave64 device would launch and return wrong
  // numbers.
  const std::string arch(props->gcnArchName);  // e.g. "gfx1151:xnack-"
  TORCH_CHECK(arch.rfind("gfx1150", 0) == 0 || arch.rfind("gfx1151", 0) == 0 ||
                  arch.rfind("gfx1152", 0) == 0 ||
                  arch.rfind("gfx1153", 0) == 0,
              "gdn_chunked is RDNA3.5-only (wave32 + DPP); device is ", arch);
  TORCH_CHECK(props->warpSize == GDN_WAVE,
              "gdn_chunked requires wave32; device reports warpSize ",
              props->warpSize);

  const dim3 grid(H, n_seqs, 1);
  const dim3 block(GDN_BLOCK_SIZE, 1, 1);

  // multiProcessorCount counts WGPs on RDNA, and a block occupies one CU in CU
  // mode but spreads over the whole WGP in WGP mode.  Below one block per WGP,
  // CU mode leaves the second CU of each WGP idle and costs about a quarter of
  // the op; above it every CU is busy either way and CU mode is worth ~10%.
  const int nsm = props->multiProcessorCount;
  const bool cu_mode = H * n_seqs > nsm;
  const bool is_fp16 = dtype == at::kHalf;
  const auto kernel =
      is_fp16 ? (cu_mode ? gdn_chunked_kernel_cu</*IS_FP16=*/true>
                         : gdn_chunked_kernel_wgp</*IS_FP16=*/true>)
              : (cu_mode ? gdn_chunked_kernel_cu</*IS_FP16=*/false>
                         : gdn_chunked_kernel_wgp</*IS_FP16=*/false>);

  kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      reinterpret_cast<const uint16_t*>(k.data_ptr()),
      reinterpret_cast<const uint16_t*>(v.data_ptr()), g.data_ptr<float>(),
      beta.data_ptr<float>(),
      initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr,
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      final_state.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(), H, Hg,
      H / Hg, static_cast<float>(scale));
}
