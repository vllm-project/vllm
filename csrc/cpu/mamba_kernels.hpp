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
#include <iostream>
#include <algorithm>

namespace mamba_cpu {

// ---------------------------------------------------------------------------
// causal_conv1d_update
// ---------------------------------------------------------------------------
inline void causal_conv1d_update_kernel(
    const float* __restrict__ x_ptr, float* __restrict__ state_ptr,
    const float* __restrict__ weight_ptr, const float* __restrict__ bias_ptr,
    float* __restrict__ out_ptr, const int32_t* __restrict__ cache_idxs,
    int32_t pad_slot_id, int64_t batch, int64_t dim, int64_t seqlen,
    int64_t width, int64_t state_len, bool do_silu) {
    
#pragma omp parallel for
  for (int64_t b = 0; b < batch; ++b) {
    int64_t cache_idx = (cache_idxs != nullptr) ? cache_idxs[b] : b;
    if (cache_idx == pad_slot_id) continue;

    for (int64_t t = 0; t < seqlen; ++t) {
      const float* x_b = x_ptr + (b * dim * seqlen + t);
      float* out_b = out_ptr + (b * dim * seqlen + t);
      float* s = state_ptr + cache_idx * dim * state_len;

      for (int64_t d = 0; d < dim; ++d) {
        float x_val = x_b[d * seqlen];
        float* sd = s + d * state_len;

        const float* w = weight_ptr + d * width;
        float acc = (bias_ptr != nullptr) ? bias_ptr[d] : 0.0f;
        
        for (int64_t k = 0; k < state_len; ++k) {
          acc += w[k] * sd[k];
        }
        acc += w[state_len] * x_val;

        if (state_len > 1) {
          std::memmove(sd, sd + 1, (state_len - 1) * sizeof(float));
        }
        if (state_len > 0) {
          sd[state_len - 1] = x_val;
        }

        if (do_silu) {
            float sigmoid = (acc >= 0) ? 
                1.0f / (1.0f + std::exp(-acc)) : 
                std::exp(acc) / (1.0f + std::exp(acc));
            acc *= sigmoid;
        }
        out_b[d * seqlen] = acc;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// selective_state_update
// ---------------------------------------------------------------------------
template <typename state_t, typename input_t>
inline void selective_state_update_kernel(
    state_t* __restrict__ state_ptr,
    int64_t stride_state_n, int64_t stride_state_h, int64_t stride_state_d,
    const input_t* __restrict__ x_ptr, const input_t* __restrict__ dt_ptr,
    int64_t stride_xdt_n, int64_t stride_xdt_h,
    const input_t* __restrict__ A_ptr, int64_t stride_A_h,
    const input_t* __restrict__ B_ptr, const input_t* __restrict__ C_ptr,
    int64_t stride_BC_n, int64_t stride_BC_g,
    const input_t* __restrict__ D_ptr, int64_t stride_D_h,
    const input_t* __restrict__ z_ptr,
    const input_t* __restrict__ dt_bias_ptr, int64_t stride_dtbias_h,
    float* __restrict__ out_ptr, int64_t stride_out_n, int64_t stride_out_h,
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
        ((dst_state_batch_indices != nullptr) ? dst_state_batch_indices[seq_idx] : state_read_idx) : -1;

    state_t* s = state_ptr + state_read_idx * stride_state_n;

    for (int64_t t = 0; t < seq_len; ++t) {
      int64_t token_idx = bos + t;
      const input_t* x_tok = x_ptr + token_idx * stride_xdt_n;
      const input_t* dt_tok = dt_ptr + token_idx * stride_xdt_n;
      const input_t* B_tok = B_ptr + token_idx * stride_BC_n;
      const input_t* C_tok = C_ptr + token_idx * stride_BC_n;
      float* out_tok = out_ptr + token_idx * stride_out_n;

#pragma omp parallel for
      for (int64_t h = 0; h < nheads; ++h) {
        int64_t g = h / nheads_per_group;
        const input_t* x_h = x_tok + h * stride_xdt_h;
        const input_t* dt_h = dt_tok + h * stride_xdt_h;
        const input_t* B_g = B_tok + g * stride_BC_g;
        const input_t* C_g = C_tok + g * stride_BC_g;
        const input_t* A_h = A_ptr + h * stride_A_h;
        const input_t* dt_bias_h = (dt_bias_ptr != nullptr) ? dt_bias_ptr + h * stride_dtbias_h : nullptr;
        const input_t* D_h = (D_ptr != nullptr) ? D_ptr + h * stride_D_h : nullptr;
        const input_t* z_h = (z_ptr != nullptr) ? z_ptr + token_idx * stride_xdt_n + h * stride_xdt_h : nullptr;
        float* out_h = out_tok + h * stride_out_h;
        state_t* s_h = s + h * stride_state_h;

        for (int64_t d = 0; d < dim; ++d) {
          float x_val = static_cast<float>(x_h[d]);
          float dt_val = static_cast<float>(dt_h[d]);
          if (dt_bias_h != nullptr) dt_val += static_cast<float>(dt_bias_h[d]);
          if (dt_softplus) {
            dt_val = (dt_val <= 20.0f) ? std::log1p(std::exp(dt_val)) : dt_val;
          }

          vec_op::FP32Vec8 out_vec(0.0f);
          state_t* s_hd = s_h + d * stride_state_d;
          const input_t* A_hd = A_h + d * dstate;
          
          vec_op::FP32Vec8 x_vec(x_val);
          vec_op::FP32Vec8 dt_vec(dt_val);

          int64_t n = 0;
          for (; n <= dstate - VEC_ELEM_NUM; n += VEC_ELEM_NUM) {
            vec_op::FP32Vec8 A_v((input_vec_t(A_hd + n)));
            vec_op::FP32Vec8 B_v((input_vec_t(B_g + n)));
            vec_op::FP32Vec8 C_v((input_vec_t(C_g + n)));
            
            vec_op::FP32Vec8 s_v((state_vec_t(s_hd + n)));
            
            vec_op::FP32Vec8 dA = (A_v * dt_vec).exp();
            vec_op::FP32Vec8 dBx = B_v * x_vec * dt_vec;
            vec_op::FP32Vec8 s_new = s_v * dA + dBx;
            
            state_vec_t(s_new).save(s_hd + n);
            out_vec = out_vec + s_new * C_v;
          }
          
          float out_val = out_vec.reduce_sum();
          for (; n < dstate; ++n) {
            float dA = std::exp(static_cast<float>(A_hd[n]) * dt_val);
            float dBx = static_cast<float>(B_g[n]) * x_val * dt_val;
            float s_new = static_cast<float>(s_hd[n]) * dA + dBx;
            s_hd[n] = static_cast<state_t>(s_new);
            out_val += s_new * static_cast<float>(C_g[n]);
          }

          if (D_h != nullptr) out_val += x_val * static_cast<float>(D_h[d]);
          if (z_h != nullptr) {
            float z_val = static_cast<float>(z_h[d]);
            float sigmoid = (z_val >= 0) ? 1.0f / (1.0f + std::exp(-z_val)) : std::exp(z_val) / (1.0f + std::exp(z_val));
            out_val *= z_val * sigmoid;
          }
          out_h[d] = out_val;
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

    if (num_accepted_tokens == nullptr && state_write_idx != null_block_id && state_write_idx != state_read_idx) {
      state_t* dst_s = state_ptr + state_write_idx * stride_state_n;
      std::memmove(dst_s, s, nheads * stride_state_h * sizeof(state_t));
    }
  }
}

} // namespace mamba_cpu
