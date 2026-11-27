/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

// Shared between the fused GDN MTP decode kernels:
// fused_gdn_decode_kernel.cu (any state dtype) and
// fused_gdn_decode_16bit_kernel.cu (BF16/FP16 state, no V split).

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Most tokens per request (MTP draft length + 1); checked by the host entry.
constexpr int kMaxMtpTokens = 16;

constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

// StateT is __nv_bfloat16 or __half. Launches one 256-thread CTA per
// (request, value head): grid = (num_requests, HV).
template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
cudaError_t launch_gdn_decode_post_conv_mtp_16bit(
    dim3 grid, cudaStream_t stream, const __nv_bfloat16* mixed_qkv,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const float* a_log,
    const void* dt_bias, const int* state_indices, const int* cu_seqlens,
    const int* num_accepted_tokens, StateT* state,
    const __nv_bfloat16* output_gate, const void* norm_weight,
    __nv_bfloat16* out, int H, int HV, int state_indices_width,
    int dt_bias_type, bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides);
