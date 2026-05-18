#pragma once

#include <cstdint>

// Launcher functions for silu_mul_fp8_quant_deep_gemm kernel
// These are compiled separately from torch for fast iteration

namespace vllm {

// Launch the silu_mul_fp8_quant_deep_gemm kernel with float scales
void launch_silu_mul_fp8_quant_deep_gemm_f32(
    void* input,  // __nv_bfloat16*
    void* y_q,    // __nv_fp8_e4m3*
    void* y_s,    // float*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, bool ceil_ue8m0, bool use_tanh_silu, void* stream);

// Launch the silu_mul_fp8_quant_deep_gemm kernel with packed ue8m0 scales
void launch_silu_mul_fp8_quant_deep_gemm_ue8m0(
    void* input,  // __nv_bfloat16*
    void* y_q,    // __nv_fp8_e4m3*
    void* y_s,    // uint8_t*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_ys_p, int64_t stride_counts_e, bool use_tanh_silu,
    void* stream);

// Launch the silu_mul_fp8_quant_deep_gemm_v2 kernel with float scales
void launch_silu_mul_fp8_quant_deep_gemm_v2_f32(
    void* input,  // __nv_bfloat16*
    void* y_q,    // __nv_fp8_e4m3*
    void* y_s,    // float*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, bool ceil_ue8m0, bool use_tanh_silu, void* stream);

// Launch the silu_mul_fp8_quant_deep_gemm_v2 kernel with packed ue8m0 scales
void launch_silu_mul_fp8_quant_deep_gemm_v2_ue8m0(
    void* input,  // __nv_bfloat16*
    void* y_q,    // __nv_fp8_e4m3*
    void* y_s,    // uint8_t*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_ys_p, int64_t stride_counts_e, bool use_tanh_silu,
    void* stream);

// FP8-in variants: FP8 input with dequant scales, float32 output scales
void launch_silu_mul_fp8_quant_deep_gemm_fp8in(
    void* input,         // __nv_fp8_e4m3*
    void* input_scales,  // float*
    void* y_q,           // __nv_fp8_e4m3*
    void* y_s,           // float*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream);

void launch_silu_mul_fp8_quant_deep_gemm_v2_fp8in(
    void* input,         // __nv_fp8_e4m3*
    void* input_scales,  // float*
    void* y_q,           // __nv_fp8_e4m3*
    void* y_s,           // float*
    int32_t* tokens_per_expert, int64_t E, int64_t T, int64_t H,
    int64_t stride_i_e, int64_t stride_i_t, int64_t stride_i_h,
    int64_t stride_yq_e, int64_t stride_yq_t, int64_t stride_yq_h,
    int64_t stride_ys_e, int64_t stride_ys_t, int64_t stride_ys_g,
    int64_t stride_counts_e, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream);

// BF16 flat layout variants: BF16 input, flat (N, 2*H) layout, no expert
// indexing
void launch_silu_mul_fp8_quant_deep_gemm_bf16_flat(
    void* input, void* y_q, void* y_s, int32_t n_tokens, int64_t N, int64_t H,
    bool ceil_ue8m0, bool use_tanh_silu, void* stream);
void launch_silu_mul_fp8_quant_deep_gemm_v2_bf16_flat(
    void* input, void* y_q, void* y_s, int32_t n_tokens, int64_t N, int64_t H,
    bool ceil_ue8m0, bool use_tanh_silu, void* stream);

// Flat layout variants: FP8 input, flat (N, 2*H) layout matching flashinfer
void launch_silu_mul_fp8_quant_deep_gemm_flat(
    void* input, void* input_scales, void* y_q, void* y_s, int32_t n_tokens,
    int64_t N, int64_t H, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream);
void launch_silu_mul_fp8_quant_deep_gemm_v2_flat(
    void* input, void* input_scales, void* y_q, void* y_s, int32_t n_tokens,
    int64_t N, int64_t H, int64_t total_padded_tokens, bool ceil_ue8m0,
    bool use_tanh_silu, void* stream);

// V3 flat kernel: fi_v5 structure + cp_async pipelining
void launch_silu_mul_fp8_quant_flat_v3(void* input, void* input_scales,
                                       void* output, void* output_scales,
                                       int32_t n_tokens, int64_t H,
                                       bool use_tanh_silu, void* stream);
void launch_silu_mul_fp8_quant_flat_v3_bf16(void* input, void* output,
                                            void* output_scales,
                                            int32_t n_tokens, int64_t H,
                                            bool use_tanh_silu, void* stream);

// V4 TMA kernel: cp.async.bulk with mbarrier pipelining
void launch_silu_mul_fp8_quant_tma(void* input, void* input_scales,
                                   void* output, void* output_scales,
                                   int32_t n_tokens, int64_t H,
                                   bool use_tanh_silu, void* stream);
void launch_silu_mul_fp8_quant_tma_bf16(void* input, void* output,
                                        void* output_scales, int32_t n_tokens,
                                        int64_t H, bool use_tanh_silu,
                                        void* stream);

// V5 TMA warp-specialized kernel: producer/consumer pipeline
// n_compute: number of consumer warps per CTA (supported: 1,2,4,7,8,14)
// batch_size: tokens per pipeline stage (supported: 8,16,32)
void launch_silu_mul_fp8_quant_tma_ws(void* input, void* input_scales,
                                      void* output, void* output_scales,
                                      int32_t n_tokens, int64_t H,
                                      int64_t scale_stride, int64_t n_compute,
                                      int64_t batch_size, bool use_tanh_silu,
                                      void* stream);
void launch_silu_mul_fp8_quant_tma_ws_bf16(void* input, void* output,
                                           void* output_scales,
                                           int32_t n_tokens, int64_t H,
                                           int64_t n_compute,
                                           bool use_tanh_silu, void* stream);

// V5 TMA warp-specialized NVFP4 kernel: BF16 input → FP4 e2m1 output
void launch_silu_mul_nvfp4_quant_tma_ws_bf16(
    void* input, void* output, void* output_sf, void* global_scale,
    int32_t n_tokens, int64_t H, int64_t N, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, void* stream);

// V5 persistent TMA warp-specialized FP8 kernel
void launch_silu_mul_fp8_quant_tma_ws_persistent(
    void* input, void* input_scales, void* output, void* output_scales,
    int32_t n_tokens, int64_t H, int64_t scale_stride, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, int64_t N, void* stream);

// V5 persistent TMA warp-specialized NVFP4 kernel
void launch_silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
    void* input, void* output, void* output_sf, void* global_scale,
    int32_t n_tokens, int64_t H, int64_t N, int64_t n_compute,
    int64_t batch_size, bool use_tanh_silu, void* stream);

}  // namespace vllm
