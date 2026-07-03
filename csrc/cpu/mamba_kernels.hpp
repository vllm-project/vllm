// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused CPU vector kernels for Mamba decode-step hotspots:
//   - causal_conv1d_update  (depthwise 1-D conv state roll + compute)
//   - selective_state_update (SSM recurrence, single-step)

#pragma once

#include "cpu_types.hpp"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace mamba_cpu {

// ---------------------------------------------------------------------------
// causal_conv1d_update — templated for native BF16/FP32
//
// state_ptr may point to a NON-CONTIGUOUS paged KV cache tensor.
// Explicit strides are passed so the kernel writes directly into the
// correct memory locations without making a contiguous copy of the full
// paged tensor (which was the source of the 34-41% direct_copy_kernel).
//
//   stride_s_slot  = state.stride(0)  — between cache slots
//   stride_s_dim   = state.stride(1)  — between conv_dim channels
//   stride_s_state = state.stride(2)  — between state elements
//
// When stride_s_state == 1 (contiguous), the memmove fast path is used.
// ---------------------------------------------------------------------------
template<typename scalar_t>
inline void causal_conv1d_update_kernel(
    const scalar_t* __restrict__ x_ptr,
    scalar_t*       __restrict__ state_ptr,
    int64_t stride_s_slot, int64_t stride_s_dim, int64_t stride_s_state,
    const scalar_t* __restrict__ weight_ptr, const float* __restrict__ bias_ptr,
    scalar_t* __restrict__ out_ptr, const int32_t* __restrict__ cache_idxs,
    int32_t pad_slot_id, int64_t batch, int64_t dim, int64_t seqlen,
    int64_t width, int64_t state_len, bool do_silu) {

#pragma omp parallel for
  for (int64_t b = 0; b < batch; ++b) {
    int64_t cache_idx = (cache_idxs != nullptr) ? cache_idxs[b] : b;
    if (cache_idx == pad_slot_id) continue;

    for (int64_t t = 0; t < seqlen; ++t) {
      const scalar_t* x_b = x_ptr + (b * dim * seqlen + t);
      scalar_t* out_b = out_ptr + (b * dim * seqlen + t);
      // Base of this slot in the (possibly non-contiguous) paged state
      scalar_t* s_base = state_ptr + cache_idx * stride_s_slot;

      for (int64_t d = 0; d < dim; ++d) {
        float x_val = static_cast<float>(x_b[d * seqlen]);
        scalar_t* sd  = s_base + d * stride_s_dim;  // start of this dim's state
        const scalar_t* w = weight_ptr + d * width;

        // Accumulate in float32 for precision
        float acc = (bias_ptr != nullptr) ? bias_ptr[d] : 0.0f;
        for (int64_t k = 0; k < state_len; ++k) {
          acc += static_cast<float>(w[k]) *
                 static_cast<float>(sd[k * stride_s_state]);
        }
        acc += static_cast<float>(w[state_len]) * x_val;

        // Shift state left and append new input.
        // Use memmove when contiguous (stride==1); element loop otherwise.
        if (stride_s_state == 1) {
          if (state_len > 1)
            std::memmove(sd, sd + 1, (state_len - 1) * sizeof(scalar_t));
          if (state_len > 0)
            sd[state_len - 1] = static_cast<scalar_t>(x_val);
        } else {
          for (int64_t k = 0; k < state_len - 1; ++k)
            sd[k * stride_s_state] = sd[(k + 1) * stride_s_state];
          if (state_len > 0)
            sd[(state_len - 1) * stride_s_state] = static_cast<scalar_t>(x_val);
        }

        if (do_silu) {
          float sigmoid = (acc >= 0) ?
              1.0f / (1.0f + std::exp(-acc)) :
              std::exp(acc) / (1.0f + std::exp(acc));
          acc *= sigmoid;
        }
        out_b[d * seqlen] = static_cast<scalar_t>(acc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// selective_state_update
//
// Template parameters:
//   state_t  - dtype of ssm_state cache (typically BFloat16)
//   input_t  - dtype of x, B, C (typically BFloat16)
//   out_t    - dtype of output tensor (typically BFloat16)
//              Write directly — no float32 intermediate buffer needed.
//
// A, D, dt_bias are accepted as const float* (they are always float32
// model parameters in Mamba2).  This eliminates the per-call float32→BF16
// conversion and the .contiguous() materialisation of the broadcast-expand.
//
// dt is accepted as a (N, nheads) scalar-per-head tensor, not as the
// (N, nheads, head_dim) expansion, so no .contiguous() copy is needed.
// ---------------------------------------------------------------------------
template <typename state_t, typename input_t, typename out_t = float>
inline void selective_state_update_kernel(
    state_t* __restrict__ state_ptr,
    int64_t stride_state_n, int64_t stride_state_h, int64_t stride_state_d,
    const input_t* __restrict__ x_ptr,
    int64_t stride_x_n, int64_t stride_x_h,
    // dt: (N, nheads) — scalar per head, NOT expanded to head_dim
    const float* __restrict__ dt_ptr,
    int64_t stride_dt_n,
    // A: (nheads,) float32 — scalar per head
    const float* __restrict__ A_ptr,
    const input_t* __restrict__ B_ptr, const input_t* __restrict__ C_ptr,
    int64_t stride_BC_n, int64_t stride_BC_g,
    // D: (nheads,) float32 — scalar per head (nullptr if not used)
    const float* __restrict__ D_ptr,
    // z: same shape as x (optional)
    const input_t* __restrict__ z_ptr,
    // dt_bias: (nheads,) float32 — scalar per head (nullptr if not used)
    const float* __restrict__ dt_bias_ptr,
    out_t* __restrict__ out_ptr,
    int64_t stride_out_n, int64_t stride_out_h,
    const int32_t* __restrict__ state_batch_indices,
    const int32_t* __restrict__ dst_state_batch_indices, int32_t null_block_id,
    const int32_t* __restrict__ num_accepted_tokens,
    const int32_t* __restrict__ cu_seqlens,
    int64_t N, int64_t nheads, int64_t ngroups, int64_t dim, int64_t dstate,
    bool dt_softplus) {

  using state_vec_t = vec_op::vec_t<state_t>;
  using input_vec_t = vec_op::vec_t<input_t>;
  constexpr int VEC_ELEM_NUM = 8;

  int64_t nheads_per_group = nheads / ngroups;

  for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
    int64_t bos, seq_len;
    if (cu_seqlens != nullptr) {
      bos = cu_seqlens[seq_idx];
      seq_len = cu_seqlens[seq_idx + 1] - bos;
    } else {
      bos = seq_idx;
      seq_len = 1;
    }

    int64_t state_read_idx = (state_batch_indices != nullptr) ?
        state_batch_indices[seq_idx] : seq_idx;
    if (state_read_idx == null_block_id) continue;

    int64_t state_write_idx = (num_accepted_tokens == nullptr) ?
        ((dst_state_batch_indices != nullptr) ?
             dst_state_batch_indices[seq_idx] : state_read_idx) : -1;

    state_t* s = state_ptr + state_read_idx * stride_state_n;

    for (int64_t t = 0; t < seq_len; ++t) {
      int64_t token_idx = bos + t;
      const input_t* x_tok = x_ptr + token_idx * stride_x_n;
      // dt: (N, nheads) — one float per head per token
      const float* dt_tok = dt_ptr + token_idx * stride_dt_n;
      const input_t* B_tok = B_ptr + token_idx * stride_BC_n;
      const input_t* C_tok = C_ptr + token_idx * stride_BC_n;
      out_t* out_tok = out_ptr + token_idx * stride_out_n;

#pragma omp parallel for
      for (int64_t h = 0; h < nheads; ++h) {
        int64_t g = h / nheads_per_group;
        const input_t* x_h = x_tok + h * stride_x_h;
        const input_t* B_g = B_tok + g * stride_BC_g;
        const input_t* C_g = C_tok + g * stride_BC_g;
        out_t* out_h = out_tok + h * stride_out_h;
        state_t* s_h = s + h * stride_state_h;

        // Read scalars-per-head (A, dt, dt_bias, D) — no per-dim indexing
        float dt_val = dt_tok[h];
        if (dt_bias_ptr != nullptr) dt_val += dt_bias_ptr[h];
        if (dt_softplus) {
          dt_val = (dt_val <= 20.0f) ? std::log1p(std::exp(dt_val)) : dt_val;
        }
        const float A_val = A_ptr[h];   // scalar: same for all dim, dstate
        const float D_val = (D_ptr != nullptr) ? D_ptr[h] : 0.0f;

        const input_t* z_h = (z_ptr != nullptr) ?
            z_ptr + token_idx * stride_x_n + h * stride_x_h : nullptr;

        vec_op::FP32Vec8 dt_vec(dt_val);
        // dA = exp(A * dt): A and dt are SCALARS per head, so compute once
        // and broadcast.  This saves 7 redundant std::exp() calls that
        // FP32Vec8::exp() would otherwise make on the broadcast vector.
        const float dA_scalar = std::exp(A_val * dt_val);
        vec_op::FP32Vec8 dA(dA_scalar);  // broadcast

        for (int64_t d = 0; d < dim; ++d) {
          float x_val = static_cast<float>(x_h[d]);

          vec_op::FP32Vec8 out_vec(0.0f);
          state_t* s_hd = s_h + d * stride_state_d;
          const input_t* B_g_base = B_g;
          const input_t* C_g_base = C_g;

          vec_op::FP32Vec8 x_vec(x_val);
          // dBx = B * x * dt — same dA for all dstate (A is scalar)
          // s_new = s * dA + B * x * dt

          int64_t n = 0;
          for (; n <= dstate - VEC_ELEM_NUM; n += VEC_ELEM_NUM) {
            vec_op::FP32Vec8 B_v((input_vec_t(B_g_base + n)));
            vec_op::FP32Vec8 C_v((input_vec_t(C_g_base + n)));
            vec_op::FP32Vec8 s_v((state_vec_t(s_hd + n)));

            vec_op::FP32Vec8 dBx = B_v * x_vec * dt_vec;
            vec_op::FP32Vec8 s_new = s_v * dA + dBx;

            state_vec_t(s_new).save(s_hd + n);
            out_vec = out_vec + s_new * C_v;
          }

          float out_val = out_vec.reduce_sum();
          for (; n < dstate; ++n) {
            // Reuse dA_scalar computed once per head — no exp() re-call
            float dBx = static_cast<float>(B_g[n]) * x_val * dt_val;
            float s_new = static_cast<float>(s_hd[n]) * dA_scalar + dBx;
            s_hd[n] = static_cast<state_t>(s_new);
            out_val += s_new * static_cast<float>(C_g[n]);
          }

          if (D_ptr != nullptr) out_val += x_val * D_val;
          if (z_h != nullptr) {
            float z_val = static_cast<float>(z_h[d]);
            float sigmoid = (z_val >= 0) ?
                1.0f / (1.0f + std::exp(-z_val)) :
                std::exp(z_val) / (1.0f + std::exp(z_val));
            out_val *= z_val * sigmoid;
          }
          out_h[d] = static_cast<out_t>(out_val);
        }
      }

      if (num_accepted_tokens != nullptr && dst_state_batch_indices != nullptr) {
        int64_t token_dst_idx = dst_state_batch_indices[seq_idx * seq_len + t];
        if (token_dst_idx != null_block_id && token_dst_idx != state_read_idx) {
          state_t* dst_s = state_ptr + token_dst_idx * stride_state_n;
          std::memmove(dst_s, s, nheads * stride_state_h * sizeof(state_t));
        }
      }
    }

    if (num_accepted_tokens == nullptr && state_write_idx != null_block_id &&
        state_write_idx != state_read_idx) {
      state_t* dst_s = state_ptr + state_write_idx * stride_state_n;
      std::memmove(dst_s, s, nheads * stride_state_h * sizeof(state_t));
    }
  }
}

// ---------------------------------------------------------------------------
// mamba_chunk_scan_fwd
//
// Prefill SSM recurrence for Mamba2 / SSD models.
//
// Key difference from selective_state_update_kernel (decode path):
//   - #pragma omp parallel for collapse(2) is OUTSIDE the time loop.
//     Each thread owns a (batch, head) slice and runs the entire token
//     sequence without any per-token OpenMP synchronisation overhead.
//     For seqlen=256, this eliminates 256 thread-barrier launches per batch.
//
// `dt` arrives already processed (float32, after bias + softplus + clamp)
// to keep this kernel simple. Preprocessing is done in the Python wrapper.
//
// `states_ptr` points to the [batch, nheads, headdim, dstate] float32 output
// tensor, pre-initialised by the caller (zero or from initial_states).
// Each (b, h) slice is private to exactly one thread via collapse(2), so
// there are no write conflicts.
//
// D is treated as a scalar per head ([nheads] float32).
// ---------------------------------------------------------------------------
template <typename input_t>
inline void mamba_chunk_scan_fwd_kernel(
    float*         __restrict__ states_ptr,  // [batch, nheads, headdim, dstate] f32
    const input_t* __restrict__ x_ptr,       // [seqlen, nheads, headdim]
    const float*   __restrict__ dt_ptr,      // [seqlen, nheads] f32 (preprocessed)
    const float*   __restrict__ A_ptr,       // [nheads] f32
    const input_t* __restrict__ B_ptr,       // [seqlen, ngroups, dstate]
    const input_t* __restrict__ C_ptr,       // [seqlen, ngroups, dstate]
    const float*   __restrict__ D_ptr,       // [nheads] f32 (nullable)
    const input_t* __restrict__ z_ptr,       // [seqlen, nheads, headdim] (nullable)
    input_t*       __restrict__ out_ptr,     // [seqlen, nheads, headdim]
    const int32_t* __restrict__ cu_seqlens,  // [batch+1] int32
    int64_t batch, int64_t nheads, int64_t ngroups,
    int64_t headdim, int64_t dstate) {

  using input_vec_t = vec_op::vec_t<input_t>;
  constexpr int VEC_ELEM_NUM = 8;

  const int64_t nheads_per_group = nheads / ngroups;
  // states layout: [batch, nheads, headdim, dstate] contiguous (caller guarantee)
  const int64_t stride_s_b = nheads * headdim * dstate;
  const int64_t stride_s_h = headdim * dstate;
  // stride_s_d = dstate, stride_s_n = 1

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < nheads; ++h) {
      const int64_t seq_start = cu_seqlens[b];
      const int64_t seq_end   = cu_seqlens[b + 1];
      const int64_t g         = h / nheads_per_group;

      const float A_val = A_ptr[h];
      const float D_val = (D_ptr != nullptr) ? D_ptr[h] : 0.0f;

      // Working state slice: states[b, h, :, :] — float32, headdim * dstate.
      // Fits in L1/L2 for typical dims (e.g. 64*128*4 = 32 KB).
      float* s_bh = states_ptr + b * stride_s_b + h * stride_s_h;

      for (int64_t t = seq_start; t < seq_end; ++t) {
        const input_t* x_h  = x_ptr  + t * nheads * headdim + h * headdim;
        const float*   dt_h = dt_ptr + t * nheads + h;
        const input_t* B_g  = B_ptr  + t * ngroups * dstate  + g * dstate;
        const input_t* C_g  = C_ptr  + t * ngroups * dstate  + g * dstate;
        const input_t* z_h  = (z_ptr != nullptr) ?
            z_ptr + t * nheads * headdim + h * headdim : nullptr;
        input_t* out_h = out_ptr + t * nheads * headdim + h * headdim;

        const float dt_val = *dt_h;
        const float dA_val = std::exp(A_val * dt_val);
        const vec_op::FP32Vec8 dA_vec(dA_val);  // broadcast scalar
        const vec_op::FP32Vec8 dt_vec(dt_val);

        for (int64_t d = 0; d < headdim; ++d) {
          const float x_val = static_cast<float>(x_h[d]);
          float* s_bhd = s_bh + d * dstate;  // [dstate] contiguous float32

          // Vectorised SSM update + readout over dstate:
          //   s_new = s * dA + x * dt * B
          //   y    += s_new * C
          int64_t n = 0;
          vec_op::FP32Vec8 y_vec(0.0f);
          const vec_op::FP32Vec8 x_vec(x_val);

          for (; n <= dstate - VEC_ELEM_NUM; n += VEC_ELEM_NUM) {
            const vec_op::FP32Vec8 B_v((input_vec_t(B_g + n)));
            const vec_op::FP32Vec8 C_v((input_vec_t(C_g + n)));
            const vec_op::FP32Vec8 s_v(s_bhd + n);

            const vec_op::FP32Vec8 s_new = s_v * dA_vec + x_vec * dt_vec * B_v;
            s_new.save(s_bhd + n);
            y_vec = y_vec + s_new * C_v;
          }

          float y_val = y_vec.reduce_sum();

          // Scalar tail for remaining dstate elements
          for (; n < dstate; ++n) {
            const float B_n   = static_cast<float>(B_g[n]);
            const float C_n   = static_cast<float>(C_g[n]);
            const float s_new = s_bhd[n] * dA_val + x_val * dt_val * B_n;
            s_bhd[n] = s_new;
            y_val += s_new * C_n;
          }

          // D skip connection (scalar per head)
          if (D_ptr != nullptr) y_val += x_val * D_val;

          // z gating: out = y * z * sigmoid(z)  (SiLU)
          if (z_h != nullptr) {
            const float z_val = static_cast<float>(z_h[d]);
            const float sigmoid = (z_val >= 0.0f) ?
                1.0f / (1.0f + std::exp(-z_val)) :
                std::exp(z_val) / (1.0f + std::exp(z_val));
            y_val *= z_val * sigmoid;
          }

          out_h[d] = static_cast<input_t>(y_val);
        }
      }
    }
  }
}

} // namespace mamba_cpu

