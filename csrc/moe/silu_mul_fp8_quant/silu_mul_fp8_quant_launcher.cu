// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Launcher implementation for silu_mul_fp8_quant_deep_gemm kernel.
// This file is compiled separately from torch for fast kernel iteration.
// NO torch headers should be included here.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "silu_mul_fp8_quant_launcher.h"
#include "quantization_activation_kernels.cuh"
#include "silu_mul_fp8_quant_flat_kernel.cuh"
#include "silu_mul_fp8_quant_tma_kernel.cuh"
#include "silu_mul_fp8_quant_tma_ws_kernel.cuh"
#include "silu_mul_nvfp4_quant_tma_ws_kernel.cuh"

namespace vllm {

static constexpr int GROUP_SIZE = 128;
static constexpr int SILU_V1_BLOCK_COUNT = 132 * 32;
static constexpr int SILU_V2_BLOCK_COUNT = 132 * 32;

template <typename scale_t, bool CEIL_UE8M0, typename fp8_type,
          bool USE_TANH_SILU = false>
void launch_kernel_impl(void* input, void* y_q, void* y_s,
                        int32_t* tokens_per_expert, int64_t E, int64_t T,
                        int64_t H, int64_t stride_i_e, int64_t stride_i_t,
                        int64_t stride_i_h, int64_t stride_yq_e,
                        int64_t stride_yq_t, int64_t stride_yq_h,
                        int64_t stride_ys_e, int64_t stride_ys_t,
                        int64_t stride_ys_g, int64_t stride_ys_p,
                        int64_t stride_counts_e, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  int const NUM_GROUPS = H / GROUP_SIZE;
  int sms = SILU_V1_BLOCK_COUNT;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * 2;

    dim3 grid(sms), block(THREAD_COUNT);
    silu_mul_fp8_quant_deep_gemm_kernel<
        SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, scale_t,
        THREAD_COUNT, Idx_t, CEIL_UE8M0, __nv_bfloat16, float, GROUP_SIZE,
        NUM_STAGES, USE_TANH_SILU>
        <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<scale_t*>(y_s),
            tokens_per_expert, (const float*)nullptr, E, T, H, (Idx_t)0,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
            stride_counts_e);
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * 2;

    dim3 grid(sms), block(THREAD_COUNT);
    silu_mul_fp8_quant_deep_gemm_kernel<
        SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, scale_t,
        THREAD_COUNT, Idx_t, CEIL_UE8M0, __nv_bfloat16, float, GROUP_SIZE,
        NUM_STAGES, USE_TANH_SILU>
        <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<scale_t*>(y_s),
            tokens_per_expert, (const float*)nullptr, E, T, H, (Idx_t)0,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
            stride_counts_e);
  }
}

template <typename scale_t, bool CEIL_UE8M0, typename fp8_type,
          bool USE_TANH_SILU = false>
void launch_kernel_v2_impl(void* input, void* y_q, void* y_s,
                           int32_t* tokens_per_expert, int64_t E, int64_t T,
                           int64_t H, int64_t stride_i_e, int64_t stride_i_t,
                           int64_t stride_i_h, int64_t stride_yq_e,
                           int64_t stride_yq_t, int64_t stride_yq_h,
                           int64_t stride_ys_e, int64_t stride_ys_t,
                           int64_t stride_ys_g, int64_t stride_ys_p,
                           int64_t stride_counts_e, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  int const NUM_GROUPS = H / GROUP_SIZE;
  int sms = SILU_V2_BLOCK_COUNT;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * 2;

    dim3 grid(sms), block(THREAD_COUNT);
    silu_mul_fp8_quant_deep_gemm_kernel_v2<
        SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, scale_t,
        THREAD_COUNT, Idx_t, CEIL_UE8M0, __nv_bfloat16, float, GROUP_SIZE,
        NUM_STAGES, USE_TANH_SILU>
        <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<scale_t*>(y_s),
            tokens_per_expert, (const float*)nullptr, E, T, H, (Idx_t)0,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
            stride_counts_e);
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * 2;

    dim3 grid(sms), block(THREAD_COUNT);
    silu_mul_fp8_quant_deep_gemm_kernel_v2<
        SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, scale_t,
        THREAD_COUNT, Idx_t, CEIL_UE8M0, __nv_bfloat16, float, GROUP_SIZE,
        NUM_STAGES, USE_TANH_SILU>
        <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<scale_t*>(y_s),
            tokens_per_expert, (const float*)nullptr, E, T, H, (Idx_t)0,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
            stride_counts_e);
  }
}

template <bool USE_TANH_SILU = false>
void launch_kernel_fp8in_impl(void* input, void* input_scales, void* y_q,
                              void* y_s, int32_t* tokens_per_expert, int64_t E,
                              int64_t T, int64_t H, int64_t stride_i_e,
                              int64_t stride_i_t, int64_t stride_i_h,
                              int64_t stride_yq_e, int64_t stride_yq_t,
                              int64_t stride_yq_h, int64_t stride_ys_e,
                              int64_t stride_ys_t, int64_t stride_ys_g,
                              int64_t stride_counts_e,
                              int64_t total_padded_tokens, bool ceil_ue8m0,
                              void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_fp8_e4m3;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;
  int sms = SILU_V1_BLOCK_COUNT;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(sms), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<
          SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<
          SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(sms), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<
          SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<
          SILU_V1_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    }
  }
}

template <bool USE_TANH_SILU = false>
void launch_kernel_v2_fp8in_impl(void* input, void* input_scales, void* y_q,
                                 void* y_s, int32_t* tokens_per_expert,
                                 int64_t E, int64_t T, int64_t H,
                                 int64_t stride_i_e, int64_t stride_i_t,
                                 int64_t stride_i_h, int64_t stride_yq_e,
                                 int64_t stride_yq_t, int64_t stride_yq_h,
                                 int64_t stride_ys_e, int64_t stride_ys_t,
                                 int64_t stride_ys_g, int64_t stride_counts_e,
                                 int64_t total_padded_tokens, bool ceil_ue8m0,
                                 void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_fp8_e4m3;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;
  int sms = SILU_V2_BLOCK_COUNT;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(sms), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(sms), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          SILU_V2_BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float,
          THREAD_COUNT, Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES,
          USE_TANH_SILU>
          <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              tokens_per_expert, reinterpret_cast<float*>(input_scales), E, T,
              H, total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e);
    }
  }
}

// --- Public BF16 launchers ---

void launch_silu_mul_fp8_quant_deep_gemm_f32(
    void* input, void* y_q, void* y_s, int32_t* tokens_per_expert, int64_t E,
    int64_t T, int64_t H, int64_t stride_i_e, int64_t stride_i_t,
    int64_t stride_i_h, int64_t stride_yq_e, int64_t stride_yq_t,
    int64_t stride_yq_h, int64_t stride_ys_e, int64_t stride_ys_t,
    int64_t stride_ys_g, int64_t stride_counts_e, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  auto call = [&]<bool TANH>() {
    if (ceil_ue8m0)
      launch_kernel_impl<float, true, __nv_fp8_e4m3, TANH>(
          input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
          stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
          stride_ys_t, stride_ys_g, 0, stride_counts_e, stream);
    else
      launch_kernel_impl<float, false, __nv_fp8_e4m3, TANH>(
          input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
          stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
          stride_ys_t, stride_ys_g, 0, stride_counts_e, stream);
  };
  if (use_tanh_silu)
    call.template operator()<true>();
  else
    call.template operator()<false>();
}

void launch_silu_mul_fp8_quant_deep_gemm_ue8m0(
    void* input, void* y_q, void* y_s, int32_t* tokens_per_expert, int64_t E,
    int64_t T, int64_t H, int64_t stride_i_e, int64_t stride_i_t,
    int64_t stride_i_h, int64_t stride_yq_e, int64_t stride_yq_t,
    int64_t stride_yq_h, int64_t stride_ys_e, int64_t stride_ys_t,
    int64_t stride_ys_g, int64_t stride_ys_p, int64_t stride_counts_e,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_impl<uint8_t, true, __nv_fp8_e4m3, true>(
        input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
        stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
        stride_ys_t, stride_ys_g, stride_ys_p, stride_counts_e, stream);
  else
    launch_kernel_impl<uint8_t, true, __nv_fp8_e4m3, false>(
        input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
        stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
        stride_ys_t, stride_ys_g, stride_ys_p, stride_counts_e, stream);
}

void launch_silu_mul_fp8_quant_deep_gemm_v2_f32(
    void* input, void* y_q, void* y_s, int32_t* tokens_per_expert, int64_t E,
    int64_t T, int64_t H, int64_t stride_i_e, int64_t stride_i_t,
    int64_t stride_i_h, int64_t stride_yq_e, int64_t stride_yq_t,
    int64_t stride_yq_h, int64_t stride_ys_e, int64_t stride_ys_t,
    int64_t stride_ys_g, int64_t stride_counts_e, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  auto call = [&]<bool TANH>() {
    if (ceil_ue8m0)
      launch_kernel_v2_impl<float, true, __nv_fp8_e4m3, TANH>(
          input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
          stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
          stride_ys_t, stride_ys_g, 0, stride_counts_e, stream);
    else
      launch_kernel_v2_impl<float, false, __nv_fp8_e4m3, TANH>(
          input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
          stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
          stride_ys_t, stride_ys_g, 0, stride_counts_e, stream);
  };
  if (use_tanh_silu)
    call.template operator()<true>();
  else
    call.template operator()<false>();
}

void launch_silu_mul_fp8_quant_deep_gemm_v2_ue8m0(
    void* input, void* y_q, void* y_s, int32_t* tokens_per_expert, int64_t E,
    int64_t T, int64_t H, int64_t stride_i_e, int64_t stride_i_t,
    int64_t stride_i_h, int64_t stride_yq_e, int64_t stride_yq_t,
    int64_t stride_yq_h, int64_t stride_ys_e, int64_t stride_ys_t,
    int64_t stride_ys_g, int64_t stride_ys_p, int64_t stride_counts_e,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_v2_impl<uint8_t, true, __nv_fp8_e4m3, true>(
        input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
        stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
        stride_ys_t, stride_ys_g, stride_ys_p, stride_counts_e, stream);
  else
    launch_kernel_v2_impl<uint8_t, true, __nv_fp8_e4m3, false>(
        input, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e, stride_i_t,
        stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e,
        stride_ys_t, stride_ys_g, stride_ys_p, stride_counts_e, stream);
}

// --- Public FP8-in launchers ---

void launch_silu_mul_fp8_quant_deep_gemm_fp8in(
    void* input, void* input_scales, void* y_q, void* y_s,
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_fp8in_impl<true>(
        input, input_scales, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e,
        stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
        total_padded_tokens, ceil_ue8m0, stream);
  else
    launch_kernel_fp8in_impl<false>(
        input, input_scales, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e,
        stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
        total_padded_tokens, ceil_ue8m0, stream);
}

void launch_silu_mul_fp8_quant_deep_gemm_v2_fp8in(
    void* input, void* input_scales, void* y_q, void* y_s,
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_v2_fp8in_impl<true>(
        input, input_scales, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e,
        stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
        total_padded_tokens, ceil_ue8m0, stream);
  else
    launch_kernel_v2_fp8in_impl<false>(
        input, input_scales, y_q, y_s, tokens_per_expert, E, T, H, stride_i_e,
        stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
        total_padded_tokens, ceil_ue8m0, stream);
}

// --- BF16 flat layout internal templates ---

template <int BLOCK_COUNT, bool USE_TANH_SILU = false>
void launch_kernel_bf16_flat_impl(void* input, void* y_q, void* y_s,
                                  int32_t n_tokens, int64_t N, int64_t H,
                                  bool ceil_ue8m0, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_bfloat16;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;

  Idx_t stride_i_e = 0, stride_i_t = 2 * H, stride_i_h = 1;
  Idx_t stride_yq_e = 0, stride_yq_t = H, stride_yq_h = 1;
  Idx_t stride_ys_e = 0, stride_ys_t = 1, stride_ys_g = N;
  Idx_t stride_counts_e = 1;
  Idx_t E = 1, T = N;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          true, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, (const float*)nullptr, E, T, H, (Idx_t)0, stride_i_e,
              stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
              stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0, stride_counts_e,
              n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          false, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, (const float*)nullptr, E, T, H, (Idx_t)0, stride_i_e,
              stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
              stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0, stride_counts_e,
              n_tokens);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          true, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, (const float*)nullptr, E, T, H, (Idx_t)0, stride_i_e,
              stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
              stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0, stride_counts_e,
              n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          false, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, (const float*)nullptr, E, T, H, (Idx_t)0, stride_i_e,
              stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, stride_yq_h,
              stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0, stride_counts_e,
              n_tokens);
    }
  }
}

template <int BLOCK_COUNT, bool USE_TANH_SILU = false>
void launch_kernel_v2_bf16_flat_impl(void* input, void* y_q, void* y_s,
                                     int32_t n_tokens, int64_t N, int64_t H,
                                     bool ceil_ue8m0, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_bfloat16;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;

  Idx_t stride_i_e = 0, stride_i_t = 2 * H, stride_i_h = 1;
  Idx_t stride_yq_e = 0, stride_yq_t = H, stride_yq_h = 1;
  Idx_t stride_ys_e = 0, stride_ys_t = 1, stride_ys_g = N;
  Idx_t stride_counts_e = 1;
  Idx_t E = 1, T = N;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr, (const float*)nullptr, E, T,
          H, (Idx_t)0, stride_i_e, stride_i_t, stride_i_h, stride_yq_e,
          stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,
          (Idx_t)0, stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr, (const float*)nullptr, E, T,
          H, (Idx_t)0, stride_i_e, stride_i_t, stride_i_h, stride_yq_e,
          stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,
          (Idx_t)0, stride_counts_e, n_tokens);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr, (const float*)nullptr, E, T,
          H, (Idx_t)0, stride_i_e, stride_i_t, stride_i_h, stride_yq_e,
          stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,
          (Idx_t)0, stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr, (const float*)nullptr, E, T,
          H, (Idx_t)0, stride_i_e, stride_i_t, stride_i_h, stride_yq_e,
          stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,
          (Idx_t)0, stride_counts_e, n_tokens);
    }
  }
}

// --- Public BF16 flat layout launchers ---

void launch_silu_mul_fp8_quant_deep_gemm_bf16_flat(
    void* input, void* y_q, void* y_s, int32_t n_tokens, int64_t N, int64_t H,
    bool ceil_ue8m0, bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_bf16_flat_impl<SILU_V1_BLOCK_COUNT, true>(
        input, y_q, y_s, n_tokens, N, H, ceil_ue8m0, stream);
  else
    launch_kernel_bf16_flat_impl<SILU_V1_BLOCK_COUNT, false>(
        input, y_q, y_s, n_tokens, N, H, ceil_ue8m0, stream);
}

void launch_silu_mul_fp8_quant_deep_gemm_v2_bf16_flat(
    void* input, void* y_q, void* y_s, int32_t n_tokens, int64_t N, int64_t H,
    bool ceil_ue8m0, bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_v2_bf16_flat_impl<SILU_V2_BLOCK_COUNT, true>(
        input, y_q, y_s, n_tokens, N, H, ceil_ue8m0, stream);
  else
    launch_kernel_v2_bf16_flat_impl<SILU_V2_BLOCK_COUNT, false>(
        input, y_q, y_s, n_tokens, N, H, ceil_ue8m0, stream);
}

// --- Flat layout internal templates ---

template <int BLOCK_COUNT, bool USE_TANH_SILU = false>
void launch_kernel_flat_impl(void* input, void* input_scales, void* y_q,
                             void* y_s, int32_t n_tokens, int64_t N, int64_t H,
                             int64_t total_padded_tokens, bool ceil_ue8m0,
                             void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_fp8_e4m3;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;

  Idx_t stride_i_e = 0, stride_i_t = 2 * H, stride_i_h = 1;
  Idx_t stride_yq_e = 0, stride_yq_t = H, stride_yq_h = 1;
  Idx_t stride_ys_e = 0, stride_ys_t = 1, stride_ys_g = total_padded_tokens;
  Idx_t stride_counts_e = 1;
  Idx_t E = 1, T = N;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          true, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, reinterpret_cast<float*>(input_scales), E, T, H,
              total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          false, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, reinterpret_cast<float*>(input_scales), E, T, H,
              total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e, n_tokens);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          true, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, reinterpret_cast<float*>(input_scales), E, T, H,
              total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel<BLOCK_COUNT, max_shared_mem_bytes,
                                          fp8_type, float, THREAD_COUNT, Idx_t,
                                          false, InputType, float, GROUP_SIZE,
                                          NUM_STAGES, USE_TANH_SILU, true>
          <<<grid, block, max_shared_mem_bytes + 32, stream>>>(
              reinterpret_cast<InputType*>(input),
              reinterpret_cast<fp8_type*>(y_q), reinterpret_cast<float*>(y_s),
              nullptr, reinterpret_cast<float*>(input_scales), E, T, H,
              total_padded_tokens, stride_i_e, stride_i_t, stride_i_h,
              stride_yq_e, stride_yq_t, stride_yq_h, stride_ys_e, stride_ys_t,
              stride_ys_g, (Idx_t)0, stride_counts_e, n_tokens);
    }
  }
}

template <int BLOCK_COUNT, bool USE_TANH_SILU = false>
void launch_kernel_v2_flat_impl(void* input, void* input_scales, void* y_q,
                                void* y_s, int32_t n_tokens, int64_t N,
                                int64_t H, int64_t total_padded_tokens,
                                bool ceil_ue8m0, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using Idx_t = int64_t;
  using InputType = __nv_fp8_e4m3;
  using fp8_type = __nv_fp8_e4m3;
  int const NUM_GROUPS = H / GROUP_SIZE;

  Idx_t stride_i_e = 0, stride_i_t = 2 * H, stride_i_h = 1;
  Idx_t stride_yq_e = 0, stride_yq_t = H, stride_yq_h = 1;
  Idx_t stride_ys_e = 0, stride_ys_t = 1, stride_ys_g = total_padded_tokens;
  Idx_t stride_counts_e = 1;
  Idx_t E = 1, T = N;

  if (H >= 4096 && (NUM_GROUPS % 8) == 0) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr,
          reinterpret_cast<float*>(input_scales), E, T, H, total_padded_tokens,
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0,
          stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr,
          reinterpret_cast<float*>(input_scales), E, T, H, total_padded_tokens,
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0,
          stride_counts_e, n_tokens);
    }
  } else {
    static constexpr int NUM_STAGES = 2;
    static constexpr int THREAD_COUNT = 32;
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;
    static constexpr int max_shared_mem_bytes =
        GROUP_SIZE * 2 * NUM_STAGES * NUM_WARPS * sizeof(InputType);

    dim3 grid(BLOCK_COUNT), block(THREAD_COUNT);
    if (ceil_ue8m0) {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, true, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr,
          reinterpret_cast<float*>(input_scales), E, T, H, total_padded_tokens,
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0,
          stride_counts_e, n_tokens);
    } else {
      silu_mul_fp8_quant_deep_gemm_kernel_v2<
          BLOCK_COUNT, max_shared_mem_bytes, fp8_type, float, THREAD_COUNT,
          Idx_t, false, InputType, float, GROUP_SIZE, NUM_STAGES, USE_TANH_SILU,
          true><<<grid, block, max_shared_mem_bytes + 32, stream>>>(
          reinterpret_cast<InputType*>(input), reinterpret_cast<fp8_type*>(y_q),
          reinterpret_cast<float*>(y_s), nullptr,
          reinterpret_cast<float*>(input_scales), E, T, H, total_padded_tokens,
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, (Idx_t)0,
          stride_counts_e, n_tokens);
    }
  }
}

// --- Public flat layout launchers ---

void launch_silu_mul_fp8_quant_deep_gemm_flat(
    void* input, void* input_scales, void* y_q, void* y_s, int32_t n_tokens,
    int64_t N, int64_t H, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_flat_impl<SILU_V1_BLOCK_COUNT, true>(
        input, input_scales, y_q, y_s, n_tokens, N, H, total_padded_tokens,
        ceil_ue8m0, stream);
  else
    launch_kernel_flat_impl<SILU_V1_BLOCK_COUNT, false>(
        input, input_scales, y_q, y_s, n_tokens, N, H, total_padded_tokens,
        ceil_ue8m0, stream);
}

void launch_silu_mul_fp8_quant_deep_gemm_v2_flat(
    void* input, void* input_scales, void* y_q, void* y_s, int32_t n_tokens,
    int64_t N, int64_t H, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream) {
  if (use_tanh_silu)
    launch_kernel_v2_flat_impl<SILU_V2_BLOCK_COUNT, true>(
        input, input_scales, y_q, y_s, n_tokens, N, H, total_padded_tokens,
        ceil_ue8m0, stream);
  else
    launch_kernel_v2_flat_impl<SILU_V2_BLOCK_COUNT, false>(
        input, input_scales, y_q, y_s, n_tokens, N, H, total_padded_tokens,
        ceil_ue8m0, stream);
}

// --- V3 flat kernel launchers ---

void launch_silu_mul_fp8_quant_flat_v3(void* input, void* input_scales,
                                       void* output, void* output_scales,
                                       int32_t n_tokens, int64_t H,
                                       bool use_tanh_silu, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  static constexpr int WARPS = 4;
  static constexpr int STAGES = 4;
  int const G = H / GROUP_SIZE;
  int const gridX = (G + WARPS - 1) / WARPS;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  static constexpr int STAGE_BYTES = flat_v3::StageLayout<__nv_fp8_e4m3>::BYTES;
  static constexpr int smem = WARPS * STAGES * STAGE_BYTES;

  dim3 grid(gridX, numSms);
  dim3 block(WARPS * 32);

  if (use_tanh_silu) {
    flat_v3::silu_mul_fp8_quant_flat_v3_kernel<__nv_fp8_e4m3, WARPS, STAGES,
                                               true>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(input),
            reinterpret_cast<float*>(input_scales),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  } else {
    flat_v3::silu_mul_fp8_quant_flat_v3_kernel<__nv_fp8_e4m3, WARPS, STAGES,
                                               false>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(input),
            reinterpret_cast<float*>(input_scales),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  }
}

void launch_silu_mul_fp8_quant_flat_v3_bf16(void* input, void* output,
                                            void* output_scales,
                                            int32_t n_tokens, int64_t H,
                                            bool use_tanh_silu,
                                            void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  static constexpr int WARPS = 4;
  static constexpr int STAGES = 4;
  int const G = H / GROUP_SIZE;
  int const gridX = (G + WARPS - 1) / WARPS;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  static constexpr int STAGE_BYTES = 2 * GROUP_SIZE * sizeof(__nv_bfloat16);
  static constexpr int smem = WARPS * STAGES * STAGE_BYTES;

  dim3 grid(gridX, numSms);
  dim3 block(WARPS * 32);

  if (use_tanh_silu) {
    flat_v3::silu_mul_fp8_quant_flat_v3_kernel<__nv_bfloat16, WARPS, STAGES,
                                               true>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            static_cast<float*>(nullptr),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  } else {
    flat_v3::silu_mul_fp8_quant_flat_v3_kernel<__nv_bfloat16, WARPS, STAGES,
                                               false>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            static_cast<float*>(nullptr),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  }
}

// --- V4 TMA kernel launchers ---

void launch_silu_mul_fp8_quant_tma(void* input, void* input_scales,
                                   void* output, void* output_scales,
                                   int32_t n_tokens, int64_t H,
                                   bool use_tanh_silu, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  static constexpr int WARPS = 4;
  static constexpr int STAGES = 4;
  int const G = H / GROUP_SIZE;
  int const gridX = (G + WARPS - 1) / WARPS;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  static constexpr int MBAR_BYTES = WARPS * STAGES * sizeof(uint64_t);
  static constexpr int STAGE_BYTES = tma_v4::StageLayout<__nv_fp8_e4m3>::BYTES;
  static constexpr int smem = MBAR_BYTES + WARPS * STAGES * STAGE_BYTES;

  dim3 grid(gridX, numSms);
  dim3 block(WARPS * 32);

  if (use_tanh_silu) {
    tma_v4::silu_mul_fp8_quant_tma_kernel<__nv_fp8_e4m3, WARPS, STAGES, true>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(input),
            reinterpret_cast<float*>(input_scales),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  } else {
    tma_v4::silu_mul_fp8_quant_tma_kernel<__nv_fp8_e4m3, WARPS, STAGES, false>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_fp8_e4m3*>(input),
            reinterpret_cast<float*>(input_scales),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  }
}

void launch_silu_mul_fp8_quant_tma_bf16(void* input, void* output,
                                        void* output_scales, int32_t n_tokens,
                                        int64_t H, bool use_tanh_silu,
                                        void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  static constexpr int WARPS = 4;
  static constexpr int STAGES = 4;
  int const G = H / GROUP_SIZE;
  int const gridX = (G + WARPS - 1) / WARPS;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  static constexpr int MBAR_BYTES = WARPS * STAGES * sizeof(uint64_t);
  static constexpr int STAGE_BYTES = tma_v4::StageLayout<__nv_bfloat16>::BYTES;
  static constexpr int smem = MBAR_BYTES + WARPS * STAGES * STAGE_BYTES;

  dim3 grid(gridX, numSms);
  dim3 block(WARPS * 32);

  if (use_tanh_silu) {
    tma_v4::silu_mul_fp8_quant_tma_kernel<__nv_bfloat16, WARPS, STAGES, true>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            static_cast<float*>(nullptr),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  } else {
    tma_v4::silu_mul_fp8_quant_tma_kernel<__nv_bfloat16, WARPS, STAGES, false>
        <<<grid, block, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input),
            static_cast<float*>(nullptr),
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<float*>(output_scales), n_tokens, H);
  }
}

// ============================================================
// V5 TMA launchers: full-row 1D TMA pipeline (FP8)
//                    cooperative LDG (BF16)
// ============================================================

static constexpr int V5_STAGES = 4;
static constexpr int NVFP4_STAGES = 2;

template <int N_COMPUTE, int BATCH_SIZE>
void launch_tma_ws_fp8_dispatch(void* input, void* input_scales, void* output,
                                void* output_scales, int32_t n_tokens,
                                int64_t H, int64_t scale_stride,
                                bool use_tanh_silu, cudaStream_t stream) {
  int const G = H / GROUP_SIZE;
  int const gridX = (G + N_COMPUTE - 1) / N_COMPUTE;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  constexpr int MBAR_REGION = ((2 * V5_STAGES * 8) + 127) & ~127;
  constexpr int NC_SLICE = N_COMPUTE * GROUP_SIZE;
  constexpr int ROW_BYTES = 2 * NC_SLICE;
  int smem = MBAR_REGION + V5_STAGES * BATCH_SIZE * ROW_BYTES;

  dim3 grid(gridX, numSms);
  dim3 block((N_COMPUTE + 1) * 32);

  auto launch_kernel = [&](auto kernel_fn) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
    kernel_fn<<<grid, block, smem, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(input),
        reinterpret_cast<float*>(input_scales),
        reinterpret_cast<__nv_fp8_e4m3*>(output),
        reinterpret_cast<float*>(output_scales), n_tokens, H, scale_stride);
  };

  if (use_tanh_silu) {
    launch_kernel(tma_v5::silu_mul_fp8_quant_tma_ws_kernel<N_COMPUTE, V5_STAGES,
                                                           BATCH_SIZE, true>);
  } else {
    launch_kernel(tma_v5::silu_mul_fp8_quant_tma_ws_kernel<N_COMPUTE, V5_STAGES,
                                                           BATCH_SIZE, false>);
  }
}

template <int N_COMPUTE>
void launch_tma_ws_bf16_dispatch(void* input, void* output, void* output_scales,
                                 int32_t n_tokens, int64_t H,
                                 bool use_tanh_silu, cudaStream_t stream) {
  int const G = H / GROUP_SIZE;
  int const gridX = (G + N_COMPUTE - 1) / N_COMPUTE;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  dim3 grid(gridX, numSms);
  dim3 block(N_COMPUTE * 32);

  if (use_tanh_silu) {
    tma_v5::silu_mul_fp8_quant_tma_ws_kernel_bf16<N_COMPUTE, 0, 0, true>
        <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(input),
                                     reinterpret_cast<__nv_fp8_e4m3*>(output),
                                     reinterpret_cast<float*>(output_scales),
                                     n_tokens, H);
  } else {
    tma_v5::silu_mul_fp8_quant_tma_ws_kernel_bf16<N_COMPUTE, 0, 0, false>
        <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(input),
                                     reinterpret_cast<__nv_fp8_e4m3*>(output),
                                     reinterpret_cast<float*>(output_scales),
                                     n_tokens, H);
  }
}

template <int N_COMPUTE>
void launch_tma_ws_fp8_bs_dispatch(void* input, void* input_scales,
                                   void* output, void* output_scales,
                                   int32_t n_tokens, int64_t H,
                                   int64_t scale_stride, int64_t batch_size,
                                   bool use_tanh_silu, cudaStream_t stream) {
  switch (batch_size) {
    case 1:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 1>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 2>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 4>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 8>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
    case 16:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 16>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
    default:
      launch_tma_ws_fp8_dispatch<N_COMPUTE, 2>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          use_tanh_silu, stream);
      break;
  }
}

void launch_silu_mul_fp8_quant_tma_ws(void* input, void* input_scales,
                                      void* output, void* output_scales,
                                      int32_t n_tokens, int64_t H,
                                      int64_t scale_stride, int64_t n_compute,
                                      int64_t batch_size, bool use_tanh_silu,
                                      void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  switch (n_compute) {
    case 1:
      launch_tma_ws_fp8_bs_dispatch<1>(input, input_scales, output,
                                       output_scales, n_tokens, H, scale_stride,
                                       batch_size, use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_fp8_bs_dispatch<2>(input, input_scales, output,
                                       output_scales, n_tokens, H, scale_stride,
                                       batch_size, use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_fp8_bs_dispatch<4>(input, input_scales, output,
                                       output_scales, n_tokens, H, scale_stride,
                                       batch_size, use_tanh_silu, stream);
      break;
    case 7:
      launch_tma_ws_fp8_bs_dispatch<7>(input, input_scales, output,
                                       output_scales, n_tokens, H, scale_stride,
                                       batch_size, use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_fp8_bs_dispatch<8>(input, input_scales, output,
                                       output_scales, n_tokens, H, scale_stride,
                                       batch_size, use_tanh_silu, stream);
      break;
    case 14:
      launch_tma_ws_fp8_bs_dispatch<14>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, stream);
      break;
    case 28:
      launch_tma_ws_fp8_bs_dispatch<28>(
          input, input_scales, output, output_scales, n_tokens, H, scale_stride,
          batch_size, use_tanh_silu, stream);
      break;
    default:
      break;
  }
}

void launch_silu_mul_fp8_quant_tma_ws_bf16(
    void* input, void* output, void* output_scales, int32_t n_tokens, int64_t H,
    int64_t n_compute, bool use_tanh_silu, void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  switch (n_compute) {
    case 1:
      launch_tma_ws_bf16_dispatch<1>(input, output, output_scales, n_tokens, H,
                                     use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_bf16_dispatch<2>(input, output, output_scales, n_tokens, H,
                                     use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_bf16_dispatch<4>(input, output, output_scales, n_tokens, H,
                                     use_tanh_silu, stream);
      break;
    case 7:
      launch_tma_ws_bf16_dispatch<7>(input, output, output_scales, n_tokens, H,
                                     use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_bf16_dispatch<8>(input, output, output_scales, n_tokens, H,
                                     use_tanh_silu, stream);
      break;
    case 14:
      launch_tma_ws_bf16_dispatch<14>(input, output, output_scales, n_tokens, H,
                                      use_tanh_silu, stream);
      break;
    case 28:
      launch_tma_ws_bf16_dispatch<28>(input, output, output_scales, n_tokens, H,
                                      use_tanh_silu, stream);
      break;
    default:
      break;
  }
}

// ============================================================
// V5 NVFP4 TMA warp-specialized launcher (BF16 → FP4 e2m1)
// ============================================================

template <int N_COMPUTE, int BATCH_SIZE>
void launch_tma_ws_nvfp4_bf16_dispatch(void* input, void* output,
                                       void* output_sf, void* global_scale,
                                       int32_t n_tokens, int64_t H, int64_t N,
                                       bool use_tanh_silu,
                                       cudaStream_t stream) {
  constexpr int WARP_ELTS = 512;
  int const numGroups = H / WARP_ELTS;
  int const gridX = (numGroups + N_COMPUTE - 1) / N_COMPUTE;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  constexpr int MBAR_REGION = ((2 * NVFP4_STAGES * 8) + 1023) & ~1023;
  constexpr int NC_SLICE_BYTES = N_COMPUTE * WARP_ELTS * 2;
  constexpr int ROW_BYTES = 2 * NC_SLICE_BYTES;
  int smem = MBAR_REGION + NVFP4_STAGES * BATCH_SIZE * ROW_BYTES;

  int smemPerSM = 0;
  cudaDeviceGetAttribute(&smemPerSM,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  constexpr int blockThreads = (N_COMPUTE + 1) * 32;
  int maxBySmem = smem > 0 ? smemPerSM / smem : 1;
  int maxByThreads = 2048 / blockThreads;
  int maxCTAsPerSM = maxBySmem < maxByThreads ? maxBySmem : maxByThreads;
  if (maxCTAsPerSM < 1) maxCTAsPerSM = 1;
  int totalDesiredCTAs = maxCTAsPerSM * numSms;
  int gridY = (totalDesiredCTAs + gridX - 1) / gridX;
  if (gridY < numSms) gridY = numSms;

  CUtensorMap tensorMap;
  cuuint64_t globalDim[3] = {64, static_cast<cuuint64_t>(2 * H / 64),
                             static_cast<cuuint64_t>(N)};
  cuuint64_t globalStrides[2] = {128, static_cast<cuuint64_t>(2 * H * 2)};
  cuuint32_t boxDim[3] = {64, 8, 1};
  cuuint32_t elementStrides[3] = {1, 1, 1};
  cuTensorMapEncodeTiled(
      &tensorMap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, input, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid(gridX, gridY);
  dim3 block(blockThreads);

  auto launch_kernel = [&](auto kernel_fn) {
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
    kernel_fn<<<grid, block, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(input),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_sf),
        reinterpret_cast<float*>(global_scale), n_tokens, H, tensorMap);
  };

  if (use_tanh_silu) {
    launch_kernel(
        tma_v5::silu_mul_nvfp4_quant_tma_ws_kernel_bf16<N_COMPUTE, NVFP4_STAGES,
                                                        BATCH_SIZE, true>);
  } else {
    launch_kernel(
        tma_v5::silu_mul_nvfp4_quant_tma_ws_kernel_bf16<N_COMPUTE, NVFP4_STAGES,
                                                        BATCH_SIZE, false>);
  }
}

template <int N_COMPUTE>
void launch_tma_ws_nvfp4_bf16_bs_dispatch(void* input, void* output,
                                          void* output_sf, void* global_scale,
                                          int32_t n_tokens, int64_t H,
                                          int64_t N, int64_t batch_size,
                                          bool use_tanh_silu,
                                          cudaStream_t stream) {
  switch (batch_size) {
    case 1:
      launch_tma_ws_nvfp4_bf16_dispatch<N_COMPUTE, 1>(input, output, output_sf,
                                                      global_scale, n_tokens, H,
                                                      N, use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_nvfp4_bf16_dispatch<N_COMPUTE, 2>(input, output, output_sf,
                                                      global_scale, n_tokens, H,
                                                      N, use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_nvfp4_bf16_dispatch<N_COMPUTE, 4>(input, output, output_sf,
                                                      global_scale, n_tokens, H,
                                                      N, use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_nvfp4_bf16_dispatch<N_COMPUTE, 8>(input, output, output_sf,
                                                      global_scale, n_tokens, H,
                                                      N, use_tanh_silu, stream);
      break;
    default:
      launch_tma_ws_nvfp4_bf16_dispatch<N_COMPUTE, 2>(input, output, output_sf,
                                                      global_scale, n_tokens, H,
                                                      N, use_tanh_silu, stream);
      break;
  }
}

void launch_silu_mul_nvfp4_quant_tma_ws_bf16(
    void* input, void* output, void* output_sf, void* global_scale,
    int32_t n_tokens, int64_t H, int64_t N, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  switch (n_compute) {
    case 1:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<1>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 2:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<2>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 4:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<4>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 7:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<7>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 8:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<8>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 14:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<14>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    case 28:
      launch_tma_ws_nvfp4_bf16_bs_dispatch<28>(
          input, output, output_sf, global_scale, n_tokens, H, N, batch_size,
          use_tanh_silu, stream);
      break;
    default:
      break;
  }
}

}  // namespace vllm
